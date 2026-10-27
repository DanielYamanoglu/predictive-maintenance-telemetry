"""CLI: `python -m simulator generate` writes ground truth; `python -m simulator serve` runs the live servers."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from simulator.plant import build_plant, truth_frame


def main() -> None:
    ap = argparse.ArgumentParser(prog="simulator")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--machines", type=int, default=16)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write plant metadata and ground truth (no network)")
    g.add_argument("--out", default="data/sim")
    s = sub.add_parser("serve", help="serve the plant over Modbus TCP and/or MQTT in (accelerated) real time")
    s.add_argument("--modbus-port", type=int, default=5020)
    s.add_argument("--mqtt-host", default=None, help="enable MQTT publishing to this broker")
    s.add_argument("--mqtt-port", type=int, default=1883)
    s.add_argument("--speed", type=float, default=60.0, help="simulated seconds per wall second")
    a = ap.parse_args()

    machines = build_plant(a.seed, a.machines)
    if a.cmd == "generate":
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        truth_frame(machines).to_parquet(out / "truth.parquet")
        meta = {m.machine_id: {"line": m.line, "kind": m.kind, "failure_ts": m.failure_ts.isoformat(), **m.meta}
                for m in machines}
        (out / "machines.json").write_text(json.dumps(meta, indent=2))
        print(f"wrote ground truth for {len(machines)} machines ({sum(len(m) for m in machines)} samples) to {out}")
        return

    from simulator.modbus_server import serve_modbus
    from simulator.mqtt_publisher import publish_mqtt

    async def run() -> None:
        tasks = [serve_modbus(machines, port=a.modbus_port, speed=a.speed)]
        if a.mqtt_host:
            tasks.append(publish_mqtt(machines, a.mqtt_host, a.mqtt_port, a.speed))
        await asyncio.gather(*tasks)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
