"""In-process mode: simulator -> ingestion with no brokers or sockets.

Machines on line1 take the Modbus path (encode to scaled uint16 registers, decode back), machines on
line2 take the MQTT path (JSON payload + topic). Both end in the same validation and storage code.
"""
from __future__ import annotations

from simulator.modbus_server import machine_reading
from simulator.mqtt_publisher import payload_for, topic_for
from simulator.plant import Machine
from common.schema import encode_registers
from ingestion.pipeline import Ingestor


def run_offline(machines: list[Machine], max_steps: int | None = None) -> Ingestor:
    ing = Ingestor()
    n = max(len(m) for m in machines)
    for step in range(n if max_steps is None else min(n, max_steps)):
        for m in machines:
            if step >= len(m):
                continue
            if m.line == "line1":
                ts, vals = machine_reading(m, step)
                ing.from_registers(m.line, m.machine_id, encode_registers(ts, vals))
            else:
                ing.from_mqtt(topic_for(m), payload_for(m, step))
    return ing
