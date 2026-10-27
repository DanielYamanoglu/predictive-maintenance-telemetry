"""Figures written to reports/figures/ by models.train."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve  # noqa: E402

C = {"blue": "#2a6fb0", "red": "#c8412e", "grey": "#8a8f98", "amber": "#d9902f", "green": "#3a8f6a"}
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _shade(ax, t, flag, color, alpha=0.18):
    d = np.diff(np.r_[0, flag.astype(int), 0])
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        ax.axvspan(t.iloc[s], t.iloc[e - 1], color=color, alpha=alpha, lw=0)


def sensor_traces(te_a: pd.DataFrame, out: Path) -> None:
    mid = sorted(m for m in te_a.machine_id.unique() if m.startswith("pump"))[0]
    g = te_a[te_a.machine_id == mid].sort_values("ts").reset_index(drop=True)
    sig = [("temperature_c_mean", "Temperature (degC, 32 min mean)"), ("vibration_mm_s_mean", "Vibration RMS (mm/s)"),
           ("current_a_mean", "Current (A)"), ("pressure_bar_mean", "Pressure (bar)")]
    fig, axes = plt.subplots(len(sig) + 1, 1, figsize=(10, 9), sharex=True)
    for ax, (c, lab) in zip(axes, sig):
        ax.plot(g.ts, g[c], color=C["blue"], lw=0.9)
        _shade(ax, g.ts, g.fault_active.to_numpy(), C["amber"])
        a = g[g.alarm]
        ax.scatter(a.ts, a[c], s=5, color=C["red"], zorder=3)
        ax.set_ylabel(lab, fontsize=8)
    axes[-1].plot(g.ts, g.anomaly_score, color=C["grey"], lw=0.8)
    axes[-1].set_ylabel("Anomaly score", fontsize=8)
    axes[-1].scatter(g[g.alarm].ts, g[g.alarm].anomaly_score, s=5, color=C["red"], zorder=3)
    axes[0].set_title(f"{mid} (held-out): detected anomalies (red) vs injected fault windows (shaded)", loc="left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "sensor_traces_anomalies.png", dpi=130)
    plt.close(fig)


def degradation_curves(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    kinds = df.groupby("machine_id").has_pressure.max()
    for ax, (c, lab) in zip(axes, [("vibration_mm_s_mean", "Vibration RMS (mm/s)"),
                                   ("temperature_c_mean", "Temperature (degC)"), ("current_a_mean", "Current (A)")]):
        for mid, g in df.groupby("machine_id"):
            g = g.sort_values("life_frac")
            y = g[c].rolling(60, min_periods=10).median()
            ax.plot(g.life_frac, y, lw=0.8, alpha=0.7, color=C["blue"] if kinds[mid] else C["green"])
        ax.set_xlabel("Fraction of life elapsed")
        ax.set_title(lab, loc="left", fontsize=10)
    axes[0].plot([], [], color=C["blue"], label="pump"); axes[0].plot([], [], color=C["green"], label="motor")
    axes[0].legend(frameon=False)
    fig.suptitle("Simulated degradation, all 16 machines (hourly-median smoothed)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "degradation_curves.png", dpi=130)
    plt.close(fig)


def classifier_eval(te_c: pd.DataFrame, thr: float, horizon_h: float, out: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    pred = (te_c.prob > thr).astype(int)
    ConfusionMatrixDisplay.from_predictions(te_c.label, pred, ax=a1, cmap="Blues", colorbar=False,
                                            display_labels=["healthy", f"fail <{horizon_h:g} h"], values_format="d")
    a1.grid(False)
    a1.set_title("Confusion matrix (held-out machines)", loc="left", fontsize=10)
    p, r, _ = precision_recall_curve(te_c.label, te_c.prob)
    a2.plot(r, p, color=C["blue"], label=f"GBM (AP = {average_precision_score(te_c.label, te_c.prob):.2f})")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_ylim(0, 1.02)
    a2.axhline(te_c.label.mean(), color=C["grey"], ls="--", lw=1, label="prevalence")
    a2.legend(frameon=False)
    a2.set_title("Precision-recall curve", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "classifier_eval.png", dpi=130)
    plt.close(fig)


def risk_timeline(te_c: pd.DataFrame, thr: float, horizon_h: float, out: Path) -> None:
    ids = sorted(te_c.machine_id.unique())
    fig, axes = plt.subplots(1, len(ids), figsize=(3.2 * len(ids), 3.2), sharey=True)
    for ax, mid in zip(np.atleast_1d(axes), ids):
        g = te_c[te_c.machine_id == mid].sort_values("ts")
        ax.plot(g.ttf_h, g.prob, color=C["blue"], lw=0.8)
        ax.axhline(thr, color=C["red"], ls="--", lw=1)
        ax.axvspan(0, horizon_h, color=C["amber"], alpha=0.18, lw=0)
        ax.invert_xaxis()
        ax.set_title(mid, loc="left", fontsize=9)
        ax.set_xlabel("Hours to failure")
    np.atleast_1d(axes)[0].set_ylabel("Predicted failure probability")
    fig.suptitle(f"Failure risk over time (shaded: true {horizon_h:g} h pre-failure window, dashed: alert threshold)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "failure_risk_timeline.png", dpi=130)
    plt.close(fig)


def make_figures(df, te_a, te_c, thr, out: Path, horizon_h: float = 4.0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sensor_traces(te_a, out)
    degradation_curves(df, out)
    classifier_eval(te_c, thr, horizon_h, out)
    risk_timeline(te_c, thr, horizon_h, out)
