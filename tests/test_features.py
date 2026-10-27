import numpy as np
import pandas as pd

from features.windows import BANDS, WINDOW, _slope, fft_band_energy, machine_features


def _frame(n=200, period=8.0):
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    t = np.arange(n)
    return pd.DataFrame({
        "temperature_c": 60 + 0.1 * t, "vibration_mm_s": 2 + np.sin(2 * np.pi * t / period),
        "current_a": 10.0 + 0 * t, "pressure_bar": np.nan, "rpm": 1480.0 + 0 * t,
    }, index=idx)


def test_feature_shapes_and_missing_sensor_dropped():
    f = machine_features(_frame())
    assert len(f) == 200
    assert "pressure_bar_mean" not in f.columns
    assert f.vib_band_mid.iloc[: WINDOW - 1].isna().all() and f.vib_band_mid.iloc[WINDOW - 1:].notna().all()
    assert f.temperature_c_mean.iloc[WINDOW - 1:].notna().all()
    assert {f"vib_band_{b}" for b in BANDS} <= set(f.columns)


def test_slope_of_ramp():
    x = pd.Series(0.5 * np.arange(100.0))
    assert np.allclose(_slope(x, 16).dropna(), 0.5)


def test_fft_band_picks_the_right_band():
    t = np.arange(128)
    e = fft_band_energy(2 + np.sin(2 * np.pi * t / 8.0))  # period 8 samples = bin 4 of 32 -> "mid"
    last = e[-1]
    assert np.argmax(last) == list(BANDS).index("mid") and last.sum() <= 1.0 + 1e-9


def test_fft_nan_window_gives_nan():
    x = np.sin(np.arange(100) / 3.0); x[50] = np.nan
    e = fft_band_energy(x)
    assert np.isnan(e[60]).all() and not np.isnan(e[99]).any()
