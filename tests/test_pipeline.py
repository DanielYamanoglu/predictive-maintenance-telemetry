import pandas as pd

from features.windows import build_features
from ingestion.offline import run_offline
from ingestion.resample import resample
from simulator.plant import DT_S, build_plant


def tiny_plant():
    return build_plant(seed=7, n_machines=4, min_life_h=1.5, max_life_h=1.8)


def test_plant_is_deterministic():
    a, b = tiny_plant(), tiny_plant()
    pd.testing.assert_frame_equal(a[0].data, b[0].data)


def test_offline_end_to_end(tmp_path):
    machines = tiny_plant()
    ing = run_offline(machines)
    assert ing.stats["received"] == sum(len(m) for m in machines)
    assert ing.stats["rejected"] == 0
    assert any(k.startswith("out_of_range_") for k in ing.stats)  # injected glitches were caught
    assert any(k.startswith("missing_") for k in ing.stats)

    path = ing.flush(tmp_path / "raw", "telemetry")
    raw = pd.read_parquet(path)
    assert set(raw.machine_id) == {m.machine_id for m in machines}
    assert raw[["temperature_c", "vibration_mm_s"]].max().le([200.0, 50.0]).all()  # nothing out of range stored

    rs = resample(raw)
    for _, g in rs.groupby("machine_id"):
        assert (g.ts.diff().dropna() == pd.Timedelta("1min")).all()  # fixed rate
    assert rs.imputed.any()
    assert (rs.n_raw <= 60 // DT_S).all()

    feats = build_features(rs)
    assert len(feats) > 0 and not feats.drop(columns=["ts", "line", "machine_id"]).isna().any().any()


def test_modbus_and_mqtt_paths_agree_on_values():
    """line1 goes through registers (quantised), line2 through JSON; both reach the same schema."""
    ing = run_offline(tiny_plant(), max_steps=50)
    df = ing.frame()
    assert set(df.line) == {"line1", "line2"}
    assert df.groupby("machine_id").temperature_c.median().between(40, 80).all()
