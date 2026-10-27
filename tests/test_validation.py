import json

from common.schema import clean_record
from ingestion.pipeline import Ingestor

BASE = {"ts": "2026-01-01T00:00:00Z", "line": "line1", "machine_id": "pump-01",
        "temperature_c": 60.0, "vibration_mm_s": 1.5, "current_a": 10.0, "pressure_bar": 6.0, "rpm": 2900}


def test_valid_record_passes_and_is_utc():
    r, dropped = clean_record(BASE)
    assert r is not None and not dropped and r.ts.tzinfo is not None


def test_out_of_range_field_is_nulled_not_the_record():
    r, dropped = clean_record({**BASE, "temperature_c": 850.0})
    assert r is not None and r.temperature_c is None and dropped == ["temperature_c"]
    assert r.vibration_mm_s == 1.5


def test_bad_timestamp_rejects_record():
    r, _ = clean_record({**BASE, "ts": "not-a-time"})
    assert r is None


def test_nan_and_garbage_types_are_handled():
    r, dropped = clean_record({**BASE, "rpm": float("nan"), "current_a": "abc"})
    assert r is not None and r.rpm is None and r.current_a is None and set(dropped) == {"rpm", "current_a"}


def test_mqtt_path_counts_quality():
    ing = Ingestor()
    topic = "plant/line2/motor-03/telemetry"
    assert ing.from_mqtt(topic, json.dumps({**BASE, "pressure_bar": None, "rpm": 9000})) is not None
    assert ing.from_mqtt(topic, b"{not json") is None
    assert ing.from_mqtt("plant/line2/telemetry", json.dumps(BASE)) is None
    assert ing.from_mqtt(topic, "[1, 2]") is None
    assert ing.stats["received"] == 4 and ing.stats["rejected"] == 3
    assert ing.stats["missing_pressure_bar"] == 1 and ing.stats["out_of_range_rpm"] == 1
    r = ing.rows[0]
    assert r["line"] == "line2" and r["machine_id"] == "motor-03"  # identity comes from the topic
