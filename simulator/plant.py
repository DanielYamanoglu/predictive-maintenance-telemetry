"""Seeded simulator of a small plant: pumps and motors that wear out and occasionally misbehave.

Each machine is run to failure: a wear variable grows from 0 to 1 over its (random) lifetime and
drives temperature, vibration, current, pressure and rpm. On top of that, transient faults
(overheat, vibration burst, current spike) and communication glitches (dropouts, stuck sensor,
out-of-range spikes) are injected. Everything is precomputed with numpy from the seed, so a
plant is fully reproducible and the same trajectory can be served over Modbus, MQTT or in-process.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common.schema import SENSORS

START = pd.Timestamp("2026-01-01T00:00:00Z")
DT_S = 10  # raw sample period in seconds
LINES = ("line1", "line2")


@dataclass
class Machine:
    machine_id: str
    line: str
    kind: str  # "pump" | "motor"
    failure_ts: pd.Timestamp
    data: pd.DataFrame  # index: ts; columns: SENSORS (NaN = dropout) + fault_active (ground truth)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.data)


def _fault_windows(rng: np.random.Generator, n: int, life_h: float):
    """Pick a few transient fault windows (start index, length, kind) in the first 90% of life."""
    wins = []
    for _ in range(rng.integers(2, 5)):
        kind = rng.choice(["overheat", "vib_burst", "current_spike"])
        length = int(rng.integers(15, 45) * 60 / DT_S)  # 15-45 minutes
        start = int(rng.uniform(0.05, 0.9) * n)
        wins.append((start, min(length, n - start), str(kind)))
    return wins


def _simulate_machine(rng: np.random.Generator, line: str, kind: str, life_h: float) -> Machine:
    n = int(life_h * 3600 / DT_S)
    t = np.arange(n) * DT_S / 3600.0  # hours
    wear = (t / life_h) ** 2.0  # slow early, accelerating late
    load = 0.9 + 0.1 * np.sin(2 * np.pi * t / 1.0 + rng.uniform(0, 6.28))  # ~1 h duty cycle
    base_temp = rng.uniform(52, 60)
    base_vib = rng.uniform(1.2, 1.8)
    base_cur = rng.uniform(14, 18) if kind == "motor" else rng.uniform(9, 12)
    base_rpm = 1480 if kind == "motor" else 2900
    base_prs = rng.uniform(5.5, 6.5)
    noise = lambda s: rng.normal(0, s, n)  # noqa: E731

    # slow, autocorrelated ambient drift
    ambient = np.cumsum(rng.normal(0, 0.02, n))
    ambient = ambient - np.convolve(ambient, np.ones(360) / 360, mode="same")

    mod = np.sin(2 * np.pi * t * 60 / 8.0)  # 8-minute periodic component, amplitude grows with bearing wear
    temp = base_temp + 6 * load + 30 * wear + ambient + noise(0.4)
    vib = base_vib * (1 + 3.5 * wear) * (0.9 + 0.1 * load) + (0.15 + 1.2 * wear) * mod * base_vib * 0.5 + noise(0.05)
    cur = base_cur * (0.85 + 0.15 * load) * (1 + 0.25 * wear) + noise(0.15)
    rpm = base_rpm * (1 - 0.03 * wear) + noise(4)
    prs = (base_prs * (1 - 0.30 * wear) + noise(0.04)) if kind == "pump" else np.full(n, np.nan)

    fault = np.zeros(n, dtype=bool)
    for s, ln, fk in _fault_windows(rng, n, life_h):
        sl = slice(s, s + ln)
        ramp = np.sin(np.linspace(0, np.pi, ln))
        if fk == "overheat":
            temp[sl] += 18 * ramp
        elif fk == "vib_burst":
            vib[sl] += 3.0 * ramp * (1 + rng.random(ln))
        else:
            cur[sl] += 7 * ramp
        fault[sl] = True

    sensors = {"temperature_c": temp, "vibration_mm_s": np.abs(vib), "current_a": cur, "pressure_bar": prs, "rpm": rpm}

    # communication / sensor glitches (not part of the fault ground truth)
    for name in SENSORS:
        x = sensors[name]
        if np.isnan(x).all():
            continue
        for _ in range(rng.integers(3, 8)):  # dropouts: NaN runs
            s = int(rng.integers(0, n - 20)); x[s:s + int(rng.integers(1, 20))] = np.nan
        if rng.random() < 0.5:  # stuck sensor: value frozen for a while
            s = int(rng.integers(0, n - 200)); x[s:s + int(rng.integers(30, 150))] = x[s]
        for s in rng.integers(0, n, size=rng.integers(1, 4)):  # absurd out-of-range spikes
            x[s] = {"temperature_c": 850.0, "vibration_mm_s": 400.0, "current_a": 655.0,
                    "pressure_bar": 90.0, "rpm": 9000.0}[name]

    idx = pd.date_range(START + pd.Timedelta(hours=float(rng.uniform(0, 6))), periods=n, freq=f"{DT_S}s")
    df = pd.DataFrame(sensors, index=idx)
    df["fault_active"] = fault
    df.index.name = "ts"
    return Machine(machine_id="", line=line, kind=kind, failure_ts=idx[-1] + pd.Timedelta(seconds=DT_S), data=df,
                   meta={"life_h": round(life_h, 2)})


def build_plant(seed: int = 42, n_machines: int = 16, min_life_h: float = 20.0, max_life_h: float = 40.0) -> list[Machine]:
    """Create the full plant. Same arguments always produce identical data."""
    rng = np.random.default_rng(seed)
    machines = []
    for i in range(n_machines):
        kind = "pump" if i % 2 == 0 else "motor"
        line = LINES[(i // 2) % len(LINES)]
        life = float(rng.uniform(min_life_h, max_life_h))
        m = _simulate_machine(rng, line, kind, life)
        m.machine_id = f"{kind}-{i // 2 + 1:02d}"
        machines.append(m)
    return machines


def layout(n_machines: int = 16) -> list[tuple[str, str, str]]:
    """(line, machine_id, kind) for each machine index; matches build_plant ordering."""
    out = []
    for i in range(n_machines):
        kind = "pump" if i % 2 == 0 else "motor"
        out.append((LINES[(i // 2) % len(LINES)], f"{kind}-{i // 2 + 1:02d}", kind))
    return out


def truth_frame(machines: list[Machine]) -> pd.DataFrame:
    """Long ground-truth table (including fault flags) for evaluation only; never seen by ingestion."""
    frames = []
    for m in machines:
        d = m.data[["fault_active"]].copy()
        d["machine_id"] = m.machine_id
        d["failure_ts"] = m.failure_ts
        frames.append(d.reset_index())
    return pd.concat(frames, ignore_index=True)
