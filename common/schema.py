"""Wire formats shared by the simulator and ingestion: Modbus register map and Pydantic models."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

SENSORS = ("temperature_c", "vibration_mm_s", "current_a", "pressure_bar", "rpm")

# Register layout of one machine block (holding registers, uint16, PLC-style scaled integers).
REGISTERS_PER_MACHINE = 8
INVALID = 0xFFFF  # sensor-fault sentinel, as many PLCs publish it
# name -> (offset, scale): raw = round(value * scale)
REGISTER_MAP = {
    "temperature_c": (2, 10.0),    # 0.1 degC
    "vibration_mm_s": (3, 1000.0),  # 0.001 mm/s RMS
    "current_a": (4, 100.0),       # 0.01 A
    "pressure_bar": (5, 100.0),    # 0.01 bar
    "rpm": (6, 1.0),
}
TS_HI, TS_LO, STATUS = 0, 1, 7  # PLC clock (uint32 epoch seconds across two registers), status word

# Plausible physical limits; anything outside is a sensor/communication glitch.
LIMITS = {
    "temperature_c": (-20.0, 200.0),
    "vibration_mm_s": (0.0, 50.0),
    "current_a": (0.0, 200.0),
    "pressure_bar": (0.0, 25.0),
    "rpm": (0.0, 4000.0),
}


def encode_registers(ts_epoch: int, values: dict, running: bool = True) -> list[int]:
    """Encode one machine reading into its register block."""
    regs = [0] * REGISTERS_PER_MACHINE
    regs[TS_HI], regs[TS_LO] = (ts_epoch >> 16) & 0xFFFF, ts_epoch & 0xFFFF
    for name, (off, scale) in REGISTER_MAP.items():
        v = values.get(name)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            regs[off] = INVALID
        else:
            raw = round(v * scale)
            regs[off] = min(max(raw, 0), INVALID - 1)  # clamp into uint16, keep sentinel free
    regs[STATUS] = 1 if running else 0
    return regs


def decode_registers(regs: list[int]) -> tuple[int, dict, bool]:
    """Inverse of encode_registers: (epoch seconds, scaled values with None for sentinel, running)."""
    ts = (regs[TS_HI] << 16) | regs[TS_LO]
    values = {
        name: (None if regs[off] == INVALID else regs[off] / scale)
        for name, (off, scale) in REGISTER_MAP.items()
    }
    return ts, values, bool(regs[STATUS] & 1)


class Reading(BaseModel):
    """One validated telemetry sample. Sensor fields are optional (missing is a valid state)."""

    ts: datetime
    line: str
    machine_id: str
    temperature_c: Optional[float] = Field(default=None, ge=LIMITS["temperature_c"][0], le=LIMITS["temperature_c"][1])
    vibration_mm_s: Optional[float] = Field(default=None, ge=LIMITS["vibration_mm_s"][0], le=LIMITS["vibration_mm_s"][1])
    current_a: Optional[float] = Field(default=None, ge=LIMITS["current_a"][0], le=LIMITS["current_a"][1])
    pressure_bar: Optional[float] = Field(default=None, ge=LIMITS["pressure_bar"][0], le=LIMITS["pressure_bar"][1])
    rpm: Optional[float] = Field(default=None, ge=LIMITS["rpm"][0], le=LIMITS["rpm"][1])

    @field_validator("ts")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @field_validator(*SENSORS)
    @classmethod
    def _finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("non-finite value")
        return v


def clean_record(raw: dict) -> tuple[Optional[Reading], list[str]]:
    """Validate a raw record. Out-of-range or malformed sensor fields are nulled and reported;
    a record whose identity/timestamp is invalid is rejected (returns None).

    Returns (reading or None, list of field names that were dropped).
    """
    dropped: list[str] = []
    raw = dict(raw)
    try:
        return Reading(**raw), dropped
    except ValidationError as exc:
        bad = {e["loc"][0] for e in exc.errors() if e["loc"]}
        if bad - set(SENSORS):
            return None, sorted(bad)
        for name in bad:
            raw[name] = None
            dropped.append(name)
    try:
        return Reading(**raw), dropped
    except ValidationError:
        return None, dropped
