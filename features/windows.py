"""Rolling-window features on the fixed-rate (1 min) resampled signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from common.schema import SENSORS

WINDOW = 32  # samples (= minutes at 1 min resampling); power of two for the FFT
# Bands of the vibration-RMS spectrum, in FFT bins of a WINDOW-sample window (bin k = k/WINDOW cycles/min).
# With 1 min sampling, bin 4 (period 8 min) is where the simulated load/bearing modulation sits.
BANDS = {"low": (1, 3), "mid": (4, 8), "high": (9, WINDOW // 2)}


def _slope(x: pd.Series, w: int) -> pd.Series:
    """Rolling least-squares slope per sample, closed form from rolling means."""
    t = pd.Series(np.arange(len(x), dtype=float), index=x.index)
    mt, mx = t.rolling(w).mean(), x.rolling(w).mean()
    cov = (t * x).rolling(w).mean() - mt * mx
    return cov / t.rolling(w).var(ddof=0)


def fft_band_energy(x: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Per-sample band energy (fraction of non-DC spectral energy) over trailing windows.

    Returns array (len(x), len(BANDS)); first window-1 rows are NaN, as are windows containing NaN.
    """
    out = np.full((len(x), len(BANDS)), np.nan)
    if len(x) < window:
        return out
    win = sliding_window_view(x, window)
    ok = ~np.isnan(win).any(axis=1)
    w = np.where(ok[:, None], win, 0.0)
    w = (w - w.mean(axis=1, keepdims=True)) * np.hanning(window)
    spec = np.abs(np.fft.rfft(w, axis=1)) ** 2
    total = spec[:, 1:].sum(axis=1) + 1e-12
    for j, (lo, hi) in enumerate(BANDS.values()):
        e = spec[:, lo:hi + 1].sum(axis=1) / total
        out[window - 1:, j] = np.where(ok, e, np.nan)
    return out


def machine_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Features for one machine's resampled frame (indexed by ts). One row per input row."""
    f = {}
    for s in SENSORS:
        x = df[s].astype(float)
        if x.isna().all():
            continue
        r = x.rolling(window, min_periods=int(window * 0.8))
        f[f"{s}_mean"] = r.mean()
        f[f"{s}_std"] = r.std()
        f[f"{s}_rms"] = np.sqrt((x ** 2).rolling(window, min_periods=int(window * 0.8)).mean())
        f[f"{s}_kurt"] = r.kurt()
        f[f"{s}_slope"] = _slope(x, window)
    bands = fft_band_energy(df["vibration_mm_s"].to_numpy(dtype=float), window)
    for j, name in enumerate(BANDS):
        f[f"vib_band_{name}"] = bands[:, j]
    return pd.DataFrame(f, index=df.index)


def build_features(resampled: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Features for every machine; keeps line/machine_id/ts columns. Rows without a full window are dropped.
    Sensors a machine does not have (pressure on motors) are filled with 0 so one model covers both kinds."""
    parts = []
    for (line, mid), g in resampled.groupby(["line", "machine_id"], sort=True):
        g = g.sort_values("ts").set_index("ts")
        feats = machine_features(g, window)
        feats.insert(0, "machine_id", mid)
        feats.insert(0, "line", line)
        parts.append(feats.reset_index())
    out = pd.concat(parts, ignore_index=True)
    fcols = [c for c in out.columns if c not in ("ts", "line", "machine_id")]
    out["has_pressure"] = out.get("pressure_bar_mean", pd.Series(np.nan, index=out.index)).notna().astype(int)
    out[fcols] = out[fcols].fillna({c: 0.0 for c in fcols if c.startswith("pressure_bar")})
    return out.dropna(subset=[c for c in fcols if not c.startswith("pressure_bar")]).reset_index(drop=True)
