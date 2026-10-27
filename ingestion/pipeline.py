"""Decode -> validate -> buffer -> Parquet. Transport-agnostic: Modbus, MQTT and offline mode all feed this."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from common.schema import SENSORS, Reading, clean_record, decode_registers


class Ingestor:
    """Collects validated readings and tracks data-quality counters."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.stats: Counter = Counter()

    def _accept(self, raw: dict) -> Optional[Reading]:
        self.stats["received"] += 1
        reading, dropped = clean_record(raw)
        if reading is None:
            self.stats["rejected"] += 1
            return None
        for name in dropped:
            self.stats[f"out_of_range_{name}"] += 1
        for name in SENSORS:
            if getattr(reading, name) is None:
                self.stats[f"missing_{name}"] += 1
        self.rows.append(reading.model_dump())
        return reading

    def from_registers(self, line: str, machine_id: str, regs: list[int]) -> Optional[Reading]:
        """One machine's register block as read from the PLC."""
        ts, values, _running = decode_registers(regs)
        return self._accept({"ts": datetime.fromtimestamp(ts, tz=timezone.utc), "line": line,
                             "machine_id": machine_id, **values})

    def from_mqtt(self, topic: str, payload: bytes | str) -> Optional[Reading]:
        """topic: plant/<line>/<machine>/telemetry, payload: JSON."""
        parts = topic.split("/")
        if len(parts) != 4 or parts[0] != "plant" or parts[3] != "telemetry":
            self.stats["received"] += 1
            self.stats["rejected"] += 1
            return None
        try:
            body = json.loads(payload)
            if not isinstance(body, dict):
                raise ValueError("payload is not an object")
        except ValueError:
            self.stats["received"] += 1
            self.stats["rejected"] += 1
            return None
        return self._accept({**body, "line": parts[1], "machine_id": parts[2]})

    @staticmethod
    def _to_frame(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["ts", "line", "machine_id", *SENSORS])

    def frame(self) -> pd.DataFrame:
        return self._to_frame(self.rows)

    def flush(self, out_dir: Path, name: str) -> Optional[Path]:
        """Write buffered rows to <out_dir>/<name>.parquet and clear the buffer."""
        rows, self.rows = self.rows, []  # swap so MQTT callback threads keep appending safely
        if not rows:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.parquet"
        self._to_frame(rows).to_parquet(path, index=False)
        return path
