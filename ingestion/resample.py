"""Resample irregular raw telemetry to a fixed rate with bounded gap filling."""
from __future__ import annotations

import pandas as pd

from common.schema import SENSORS


def resample(raw: pd.DataFrame, rule: str = "1min", max_gap: int = 3) -> pd.DataFrame:
    """Per machine: sort, drop duplicate timestamps, bin-mean to `rule`, linearly fill gaps of at most
    `max_gap` bins (longer gaps stay NaN). `imputed` marks bins with any filled value; `n_raw` is the
    raw sample count in the bin."""
    out = []
    for (line, mid), g in raw.groupby(["line", "machine_id"], sort=True):
        g = g.drop_duplicates("ts").sort_values("ts").set_index("ts")
        bins = g[list(SENSORS)].resample(rule)
        r = bins.mean()
        n_raw = bins.size().rename("n_raw")
        # all-NaN sensors (e.g. no pressure transducer on a motor) are left untouched
        filled = r.interpolate(limit=max_gap, limit_area="inside")
        imputed = (r.isna() & filled.notna()).any(axis=1).rename("imputed")
        d = pd.concat([filled, n_raw, imputed], axis=1)
        d.insert(0, "machine_id", mid)
        d.insert(0, "line", line)
        out.append(d.reset_index())
    return pd.concat(out, ignore_index=True) if out else raw.iloc[0:0]
