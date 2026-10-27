"""Publish the simulated plant to MQTT: one JSON message per machine per step on plant/<line>/<machine>/telemetry."""
from __future__ import annotations

import asyncio
import json
import math
import time

import paho.mqtt.client as mqtt

from simulator.plant import DT_S, Machine


def connect_with_retry(client: mqtt.Client, host: str, port: int, attempts: int = 30) -> None:
    """The broker may still be starting (docker compose); retry for about a minute."""
    for i in range(attempts):
        try:
            client.connect(host, port)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(2)


def topic_for(m: Machine) -> str:
    return f"plant/{m.line}/{m.machine_id}/telemetry"


def payload_for(m: Machine, i: int) -> str:
    """JSON payload for step i. NaN (sensor dropout) becomes null."""
    j = i % len(m)
    row = m.data.iloc[j]
    body = {"ts": m.data.index[j].isoformat()}
    for k in m.data.columns:
        if k != "fault_active":
            body[k] = None if math.isnan(row[k]) else float(row[k])
    return json.dumps(body)


async def publish_mqtt(machines: list[Machine], host: str = "localhost", port: int = 1883, speed: float = 60.0,
                       stop: asyncio.Event | None = None) -> None:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="plant-simulator")
    connect_with_retry(client, host, port)
    client.loop_start()
    stop = stop or asyncio.Event()
    step = 0
    try:
        while not stop.is_set():
            for m in machines:
                client.publish(topic_for(m), payload_for(m, step), qos=0)
            step += 1
            await asyncio.sleep(DT_S / speed)
    finally:
        client.loop_stop()
        client.disconnect()
