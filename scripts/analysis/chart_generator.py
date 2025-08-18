import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ltv_distribution(pred_fp: Path, out_dir=OUT_DIR):
    df = pd.read_csv(pred_fp)
    fig, ax = plt.subplots(figsize=(8,5))
    sns.kdeplot(df["y_true_ltv90"], ax=ax, label="True LTV 90d", fill=True, alpha=0.3)
    sns.kdeplot(df["y_pred_ltv90"], ax=ax, label="Predicted LTV 90d", fill=True, alpha=0.3)
    ax.set_title("LTV 90d — True vs Predicted (Density)")
    ax.set_xlabel("LTV (90 days)")
    ax.legend()
    _savefig(out_dir / "ltv90_true_vs_pred_density.png")


def plot_pred_vs_actual(pred_fp: Path, out_dir=OUT_DIR):
    df = pd.read_csv(pred_fp)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df["y_true_ltv90"], df["y_pred_ltv90"], s=8, alpha=0.5)
    m = max(df["y_true_ltv90"].max(), df["y_pred_ltv90"].max())
    ax.plot([0, m], [0, m], linestyle="--")
    ax.set_xlabel("Actual LTV 90d")
    ax.set_ylabel("Predicted LTV 90d")
    ax.set_title("Predicted vs Actual — LTV 90d")
    _savefig(out_dir / "pred_vs_actual_ltv90.png")


def plot_feature_importance(fi_fp: Path, out_dir=OUT_DIR, topn=15):
    fi = pd.read_csv(fi_fp).sort_values("importance", ascending=True).tail(topn)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(fi["feature"], fi["importance"])
    ax.set_title(f"Top {topn} Feature Importances")
    ax.set_xlabel("Importance (Permutation / Model)")
    _savefig(out_dir / "feature_importance_top15.png")


def plot_decile_lift(lift_fp: Path, out_dir=OUT_DIR):
    df = pd.read_csv(lift_fp).sort_values("decile", ascending=False)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(df["decile"], df["cum_rev_share"], marker="o")
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Decile (10 = Highest predicted LTV)")
    ax.set_ylabel("Cumulative Revenue Share")
    ax.set_title("Decile Lift — Revenue Capture")
    for x, y in zip(df["decile"], df["cum_rev_share"]):
        ax.text(x, y, f"{y:.2f}", fontsize=8, ha="center", va="bottom")
    _savefig(out_dir / "decile_lift.png")


def plot_rfm_segment_dist(rfm_fp: Path, out_dir=OUT_DIR):
    rfm = pd.read_csv(rfm_fp)
    seg = rfm["segment"].value_counts(normalize=False).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(seg.index, seg.values)
    ax.set_title("RFM Segment Distribution")
    ax.set_ylabel("Customers")
    ax.set_xticklabels(seg.index, rotation=20, ha="right")
    _savefig(out_dir / "rfm_segment_distribution.png")


def plot_cohort_retention_heatmap(cohort_fp: Path, out_dir=OUT_DIR):
    heat = pd.read_csv(cohort_fp)
    # Convert cohort_month to string for y-axis readability
    if "cohort_month" in heat.columns:
        heat["cohort_month"] = pd.to_datetime(heat["cohort_month"]).dt.strftime("%Y-%m")
        heat = heat.set_index("cohort_month").sort_index()
    # melt back to matrix
    mat = heat.copy()
    # ensure numeric columns only (offset 0..N)
    mat = mat[[c for c in mat.columns if c != "cohort_month"]]
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(mat, cmap="Blues", annot=False, cbar=True, ax=ax)
    ax.set_title("Cohort Retention Heatmap (ratio)")
    ax.set_xlabel("Months Since Cohort Start")
    ax.set_ylabel("Cohort (YYYY-MM)")
    _savefig(out_dir / "cohort_retention_heatmap.png")


def main():
    pred_fp = ROOT / "outputs" / "predictions" / "predictions.csv"
    fi_fp = DATA_DIR / "feature_importances.csv"
    lift_fp = DATA_DIR / "decile_lift.csv"
    rfm_fp = DATA_DIR / "rfm.csv"
    cohort_fp = DATA_DIR / "cohorts_retention.csv"

    if not pred_fp.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_fp}")
    plot_ltv_distribution(pred_fp)
    plot_pred_vs_actual(pred_fp)

    if fi_fp.exists():
        plot_feature_importance(fi_fp)
    if lift_fp.exists():
        plot_decile_lift(lift_fp)
    if rfm_fp.exists():
        plot_rfm_segment_dist(rfm_fp)
    if cohort_fp.exists():
        plot_cohort_retention_heatmap(cohort_fp)

    print("✅ Charts generated in:", OUT_DIR)


if __name__ == "__main__":
    main()