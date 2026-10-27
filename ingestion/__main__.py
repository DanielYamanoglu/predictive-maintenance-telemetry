"""CLI: `python -m ingestion offline|live`."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import pandas as pd

from ingestion.pipeline import Ingestor
from ingestion.resample import resample


def finish(ing: Ingestor, out: Path, name: str = "telemetry") -> None:
    out.mkdir(parents=True, exist_ok=True)
    ing.flush(out / "raw", name)
    raw = pd.read_parquet(out / "raw")
    rs = resample(raw)
    (out / "processed").mkdir(exist_ok=True)
    rs.to_parquet(out / "processed" / "resampled.parquet", index=False)
    (out / "quality.json").write_text(json.dumps(dict(sorted(ing.stats.items())), indent=2))
    print(f"raw rows: {len(raw)}, resampled rows: {len(rs)}")
    print("quality:", json.dumps(dict(sorted(ing.stats.items()))))


async def live(a, ing: Ingestor) -> None:
    from ingestion.modbus_poller import poll_modbus
    from ingestion.mqtt_subscriber import subscribe_mqtt

    tasks = []
    if a.modbus_host:
        tasks.append(poll_modbus(ing, a.modbus_host, a.modbus_port, a.machines, a.poll_interval, a.duration))
    if a.mqtt_host:
        tasks.append(subscribe_mqtt(ing, a.mqtt_host, a.mqtt_port, a.duration))

    async def flusher() -> None:
        part = 0
        while True:
            await asyncio.sleep(a.flush_every)
            ing.flush(Path(a.out) / "live", f"part-{part:05d}")
            part += 1

    await asyncio.gather(*tasks, flusher() if a.duration is None else asyncio.sleep(0))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(prog="ingestion")
    ap.add_argument("--out", default="data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("offline", help="simulator -> ingestion in-process (no brokers)")
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--machines", type=int, default=16)
    o.add_argument("--max-steps", type=int, default=None, help="limit raw steps per machine (default: full run)")
    lv = sub.add_parser("live", help="poll Modbus and/or subscribe to MQTT")
    lv.add_argument("--modbus-host", default=os.getenv("MODBUS_HOST"))
    lv.add_argument("--modbus-port", type=int, default=int(os.getenv("MODBUS_PORT", "5020")))
    lv.add_argument("--mqtt-host", default=os.getenv("MQTT_HOST"))
    lv.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    lv.add_argument("--machines", type=int, default=16)
    lv.add_argument("--poll-interval", type=float, default=1.0)
    lv.add_argument("--flush-every", type=float, default=30.0)
    lv.add_argument("--duration", type=float, default=None, help="seconds; default run forever")
    a = ap.parse_args()

    if a.cmd == "offline":
        from ingestion.offline import run_offline
        from simulator.plant import build_plant

        ing = run_offline(build_plant(a.seed, a.machines), a.max_steps)
        finish(ing, Path(a.out))
    else:
        ing = Ingestor()
        try:
            asyncio.run(live(a, ing))
        except KeyboardInterrupt:
            pass
        ing.flush(Path(a.out) / "live", "part-final")


if __name__ == "__main__":
    main()
