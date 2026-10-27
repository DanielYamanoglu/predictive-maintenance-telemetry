"""Train and evaluate the anomaly detector and the failure-within-N-hours classifier.

Evaluation is on machines never seen in training (split by machine, not by row).
Usage: python -m models.train [--data data] [--horizon-h 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GroupKFold

from features.windows import build_features

SEED = 42
HEALTHY_LIFE_FRACTION = 0.30  # the first 30% of a run is treated as the healthy commissioning period
ALARM_QUANTILE = 0.99  # anomaly threshold: 99th percentile of train healthy-phase scores
SUSTAIN = 5  # consecutive alarm minutes before an alert counts


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("ts", "line", "machine_id", "failure_ts", "ttf_h", "life_frac",
                                               "fault_active", "label")]


def load_dataset(data: Path, horizon_h: float) -> pd.DataFrame:
    rs = pd.read_parquet(data / "processed" / "resampled.parquet")
    feats = build_features(rs)
    meta = json.loads((data / "sim" / "machines.json").read_text())
    truth = pd.read_parquet(data / "sim" / "truth.parquet")
    truth["ts"] = truth["ts"].dt.floor("1min")
    fault = truth.groupby(["machine_id", "ts"])["fault_active"].max().reset_index()
    df = feats.merge(fault, on=["machine_id", "ts"], how="left")
    df["fault_active"] = df["fault_active"].fillna(False).astype(bool)
    df["failure_ts"] = df["machine_id"].map(lambda m: pd.Timestamp(meta[m]["failure_ts"]))
    df["ttf_h"] = (df["failure_ts"] - df["ts"]).dt.total_seconds() / 3600
    start = df.groupby("machine_id")["ts"].transform("min")
    df["life_frac"] = (df["ts"] - start) / (df["failure_ts"] - start)
    df["label"] = (df["ttf_h"] <= horizon_h).astype(int)
    df.attrs["kinds"] = {m: v["kind"] for m, v in meta.items()}
    return df


def split_machines(kinds: dict[str, str], n_test_per_kind: int = 2) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(SEED)
    test = []
    for kind in sorted(set(kinds.values())):
        ids = sorted(m for m, k in kinds.items() if k == kind)
        test += [str(m) for m in rng.choice(ids, size=n_test_per_kind, replace=False)]
    return sorted(m for m in kinds if m not in test), sorted(test)


def episodes(flag: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    d = np.diff(np.r_[0, flag.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def sustained(alarm: np.ndarray, k: int) -> np.ndarray:
    """True where the last k samples were all alarms."""
    return pd.Series(alarm.astype(int)).rolling(k).min().fillna(0).to_numpy().astype(bool)


def evaluate_anomaly(df: pd.DataFrame, cols: list[str], train_ids: list[str], test_ids: list[str]):
    tr = df[df.machine_id.isin(train_ids) & (df.life_frac < HEALTHY_LIFE_FRACTION) & ~df.fault_active]
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=SEED, n_jobs=-1).fit(tr[cols])
    thr = float(np.quantile(-iso.score_samples(tr[cols]), ALARM_QUANTILE))
    te = df[df.machine_id.isin(test_ids)].copy()
    te["anomaly_score"] = -iso.score_samples(te[cols])
    te["alarm"] = te["anomaly_score"] > thr
    early = te[te.life_frac < 0.6]  # before wear dominates: isolates the injected transient faults
    hit, n_ep = 0, 0
    for _, g in te.sort_values("ts").groupby("machine_id"):
        g = g[g.life_frac < 0.6].reset_index(drop=True)
        for s, e in episodes(g.fault_active.to_numpy()):
            n_ep += 1
            hit += int(g.alarm.iloc[s:e].any())
    quiet = early[~early.fault_active & (early.life_frac < HEALTHY_LIFE_FRACTION)]
    res = {
        "train_rows": int(len(tr)),
        "alarm_threshold": round(thr, 4),
        "auc_vs_injected_faults_life<0.6": round(float(roc_auc_score(early.fault_active, early.anomaly_score)), 4),
        "fault_episodes_test": n_ep,
        "fault_episodes_detected": hit,
        "false_alarm_rate_healthy_phase": round(float(quiet.alarm.mean()), 4),
        "alarm_rate_last_10pct_of_life": round(float(te[te.life_frac > 0.9].alarm.mean()), 4),
    }
    return iso, thr, te, res


def first_alert_lead_h(g: pd.DataFrame, prob_col: str, thr: float) -> float | None:
    g = g.sort_values("ts")
    s = sustained(g[prob_col].to_numpy() > thr, SUSTAIN)
    return float(g.ttf_h.to_numpy()[np.argmax(s)]) if s.any() else None


def new_clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, l2_regularization=1.0,
                                          random_state=SEED)


def evaluate_classifier(df: pd.DataFrame, cols: list[str], train_ids: list[str], test_ids: list[str]):
    tr = df[df.machine_id.isin(train_ids)].reset_index(drop=True)
    te = df[df.machine_id.isin(test_ids)].copy()
    # out-of-fold predictions on training machines (grouped by machine) pick the alert threshold
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(n_splits=4).split(tr, tr.label, tr.machine_id):
        oof[b] = new_clf().fit(tr.loc[a, cols], tr.label[a]).predict_proba(tr.loc[b, cols])[:, 1]
    p, r, t = precision_recall_curve(tr.label, oof)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    thr = float(t[np.argmax(f1)])
    clf = new_clf().fit(tr[cols], tr.label)
    te["prob"] = clf.predict_proba(te[cols])[:, 1]
    pred = (te.prob > thr).astype(int)
    pr, rc, f, _ = precision_recall_fscore_support(te.label, pred, average="binary", zero_division=0)
    leads = {m: first_alert_lead_h(g, "prob", thr) for m, g in te.groupby("machine_id")}
    res = {
        "train_machines": len(train_ids), "test_machines": test_ids,
        "test_rows": int(len(te)), "test_positive_rate": round(float(te.label.mean()), 4),
        "threshold_from_train_oof": round(thr, 4),
        "train_oof_roc_auc": round(float(roc_auc_score(tr.label, oof)), 4),
        "test_roc_auc": round(float(roc_auc_score(te.label, te.prob)), 4),
        "test_pr_auc": round(float(average_precision_score(te.label, te.prob)), 4),
        "test_precision": round(float(pr), 4), "test_recall": round(float(rc), 4), "test_f1": round(float(f), 4),
        "test_confusion_matrix_[[tn,fp],[fn,tp]]": confusion_matrix(te.label, pred).tolist(),
        "first_sustained_alert_hours_before_failure": {m: (None if v is None else round(v, 2)) for m, v in leads.items()},
    }
    return clf, thr, te, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--horizon-h", type=float, default=4.0)
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    data, reports = Path(a.data), Path(a.reports)

    df = load_dataset(data, a.horizon_h)
    kinds = df.attrs["kinds"]
    cols = feature_columns(df)
    train_ids, test_ids = split_machines(kinds)
    print(f"features: {len(cols)}  rows: {len(df)}  train machines: {len(train_ids)}  held-out: {test_ids}")

    iso, athr, te_a, anomaly_res = evaluate_anomaly(df, cols, train_ids, test_ids)
    clf, cthr, te_c, clf_res = evaluate_classifier(df, cols, train_ids, test_ids)
    metrics = {"horizon_hours": a.horizon_h, "anomaly_detector": anomaly_res, "failure_classifier": clf_res}
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (data / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump({"iso": iso, "clf": clf, "columns": cols, "alarm_threshold": athr, "clf_threshold": cthr},
                data / "models" / "models.joblib")
    print(json.dumps(metrics, indent=2))
    if not a.no_figures:
        from models.figures import make_figures
        make_figures(df, te_a, te_c, cthr, reports / "figures")


if __name__ == "__main__":
    main()
