"""Subscribe to plant/+/+/telemetry and feed the ingestor."""
from __future__ import annotations

import asyncio

import paho.mqtt.client as mqtt

from ingestion.pipeline import Ingestor
from simulator.mqtt_publisher import connect_with_retry


async def subscribe_mqtt(ing: Ingestor, host: str, port: int, duration_s: float | None = None) -> None:
    def on_connect(client, userdata, flags, reason_code, properties):
        client.subscribe("plant/+/+/telemetry")

    def on_message(client, userdata, msg):
        ing.from_mqtt(msg.topic, msg.payload)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ingestion")
    client.on_connect, client.on_message = on_connect, on_message
    connect_with_retry(client, host, port)
    client.loop_start()  # paho runs its network thread; callbacks append to the ingestor
    try:
        if duration_s is None:
            await asyncio.Event().wait()
        else:
            await asyncio.sleep(duration_s)
    finally:
        client.loop_stop()
        client.disconnect()
