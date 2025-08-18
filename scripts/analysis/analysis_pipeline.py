from __future__ import annotations
import os, json, logging, math, warnings
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional XGBoost
XGB_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# ---------- Paths & logging ----------
ROOT = Path(__file__).resolve().parents[2]  # <repo_root>
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT / "outputs"
CHART_DIR = OUT_DIR / "charts"
PRED_DIR = OUT_DIR / "predictions"
LOG_DIR = ROOT / "logs" / "application"
MODEL_DIR = ROOT / "models" / "validation"

for p in [PROC_DIR, CHART_DIR, PRED_DIR, LOG_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "analysis_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("analysis_pipeline")


# ---------- Helpers ----------
def _read_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    tx_fp = RAW_DIR / "transactions.csv"
    cust_fp = RAW_DIR / "customers.csv"
    if not tx_fp.exists() or not cust_fp.exists():
        logger.error("Không tìm thấy file raw: %s | %s", tx_fp, cust_fp)
        raise FileNotFoundError("Missing raw CSVs in data/raw/")
    tx = pd.read_csv(tx_fp, parse_dates=["transaction_date"])
    cust = pd.read_csv(cust_fp, parse_dates=["signup_date"], infer_datetime_format=True)
    return tx, cust


def _clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    # loại bỏ records lỗi / âm / null
    tx = tx.copy()
    tx = tx.drop_duplicates(subset=["customer_id", "transaction_date", "amount", "channel"])
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    tx = tx[tx["amount"] > 0]
    # chuẩn hóa channel
    if "channel" in tx.columns:
        tx["channel"] = tx["channel"].fillna("Unknown").str.title()
    else:
        tx["channel"] = "Unknown"
    return tx


def _build_snapshot_label(tx: pd.DataFrame, horizon_days: int = 90) -> tuple[pd.DataFrame, pd.Timestamp]:
    max_dt = tx["transaction_date"].max()
    snapshot = max_dt - pd.Timedelta(days=horizon_days)
    # features dùng lịch sử <= snapshot; label là sum amount trong (snapshot, snapshot + H]
    hist = tx[tx["transaction_date"] <= snapshot].copy()
    future = tx[(tx["transaction_date"] > snapshot) & (tx["transaction_date"] <= snapshot + pd.Timedelta(days=horizon_days))].copy()

    y = future.groupby("customer_id")["amount"].sum().rename("ltv_90d").reset_index()
    return (hist, y, snapshot)


def _rfm_features(hist: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    Robust RFM + channel-share + interpurchase feature builder.
    Tránh dùng reset_index trên Series có MultiIndex để không gây duplicate column errors.
    """
    hist = hist.copy()
    # Basic aggregates per customer
    agg = hist.groupby("customer_id").agg(
        last_tx=("transaction_date", "max"),
        first_tx=("transaction_date", "min"),
        tx_count=("transaction_date", "count"),
        monetary=("amount", "sum"),
        avg_amount=("amount", "mean")
    ).reset_index()

    # Recency / Tenure
    agg["recency_days"] = (ref_date - pd.to_datetime(agg["last_tx"])).dt.days.clip(lower=0)
    agg["tenure_days"] = (ref_date - pd.to_datetime(agg["first_tx"])).dt.days.clip(lower=0)
    agg["freq_per_90d"] = agg["tx_count"] / (agg["tenure_days"].replace(0, 1) / 90.0)

    # Interpurchase stats (mean/std of days between purchases)
    diffs = (
        hist.sort_values(["customer_id", "transaction_date"])
            .groupby("customer_id")["transaction_date"]
            .diff()
            .dt.days
    )
    tmp = pd.DataFrame({"customer_id": hist["customer_id"], "ip_days": diffs})
    ip_stats = tmp.groupby("customer_id")["ip_days"].agg(["mean", "std"]).rename(columns={"mean": "ip_mean", "std": "ip_std"}).reset_index()
    agg = agg.merge(ip_stats, on="customer_id", how="left")
    # Fill NaN ip stats with medians (robust fallback)
    if "ip_mean" in agg.columns:
        agg["ip_mean"] = agg["ip_mean"].fillna(agg["ip_mean"].median())
    else:
        agg["ip_mean"] = 0.0
    if "ip_std" in agg.columns:
        agg["ip_std"] = agg["ip_std"].fillna(agg["ip_std"].median())
    else:
        agg["ip_std"] = 0.0

    # Channel share (robust approach)
    if "channel" in hist.columns and not hist["channel"].isna().all():
        # ensure channel as string
        hist["channel"] = hist["channel"].fillna("Unknown").astype(str)

        # 1) sum amount by customer x channel -> DataFrame
        ch = hist.groupby(["customer_id", "channel"], as_index=False)["amount"].sum()
        # 2) compute share per customer via transform
        ch["channel_share"] = ch.groupby("customer_id")["amount"].transform(lambda s: s / s.sum())
        # 3) pivot to wide
        ch_pivot = ch.pivot(index="customer_id", columns="channel", values="channel_share").fillna(0.0)
        # sanitize column names
        ch_pivot.columns = [f"ch_{str(c).strip().lower().replace(' ', '_')}" for c in ch_pivot.columns]
        ch_pivot = ch_pivot.reset_index()

        # Merge; merge on customer_id is safe because both frames include it
        agg = agg.merge(ch_pivot, on="customer_id", how="left")
        # Fill any newly created channel cols' NaN with 0.0
        ch_cols = [c for c in agg.columns if str(c).startswith("ch_")]
        for c in ch_cols:
            agg[c] = agg[c].fillna(0.0)
    else:
        pass

    return agg


def _rfm_scoring(rfm: pd.DataFrame) -> pd.DataFrame:
    # Quintile scores 1..5
    r_bins = pd.qcut(rfm["recency_days"], 5, labels=[5,4,3,2,1])  # recency thấp = tốt
    f_bins = pd.qcut(rfm["tx_count"].rank(method="first"), 5, labels=[1,2,3,4,5])
    m_bins = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["R"] = r_bins.astype(int)
    rfm["F"] = f_bins.astype(int)
    rfm["M"] = m_bins.astype(int)
    rfm["RFM_Score"] = rfm[["R","F","M"]].sum(axis=1)
    # segment đơn giản
    def seg(row):
        if row["RFM_Score"] >= 12: return "Champions"
        if row["RFM_Score"] >= 10: return "Loyal"
        if row["RFM_Score"] >= 8:  return "Potential Loyalist"
        if row["RFM_Score"] >= 6:  return "At Risk"
        return "Hibernating"
    rfm["segment"] = rfm.apply(seg, axis=1)
    return rfm


def _monthly_cohort_retention(tx: pd.DataFrame) -> pd.DataFrame:
    # cohort = tháng đầu mua, retention = % active (có >=1 giao dịch) theo offset tháng
    tx = tx.copy()
    tx["order_month"] = tx["transaction_date"].values.astype("datetime64[M]")
    first_month = tx.groupby("customer_id")["order_month"].min().rename("cohort_month")
    tx = tx.merge(first_month, on="customer_id", how="left")
    tx["month_offset"] = ((tx["order_month"].dt.year - tx["cohort_month"].dt.year) * 12 +
                          (tx["order_month"].dt.month - tx["cohort_month"].dt.month))
    cohort_size = first_month.value_counts().rename("cohort_size").reset_index().rename(columns={"index":"cohort_month"})
    active = tx.groupby(["cohort_month", "month_offset"])["customer_id"].nunique().reset_index(name="active_users")
    ret = active.merge(cohort_size, on="cohort_month", how="left")
    ret["retention"] = (ret["active_users"] / ret["cohort_size"]).round(4)
    # pivot heatmap
    heat = ret.pivot(index="cohort_month", columns="month_offset", values="retention").fillna(0.0)
    heat = heat.sort_index()
    return heat.reset_index()


def _train_model(df: pd.DataFrame, target_col="ltv_90d"):
    feature_cols = [c for c in df.columns if c not in {"customer_id", target_col, "segment", "first_tx", "last_tx"} and df[c].dtype != "O"]
    X = df[feature_cols].fillna(0.0)
    y = df[target_col].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if XGB_AVAILABLE:
        model = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    r2 = float(r2_score(y_test, pred))

    # decile lift
    dec = pd.DataFrame({"y_true": y_test.values, "y_pred": pred})
    dec["decile"] = pd.qcut(dec["y_pred"].rank(method="first"), 10, labels=list(range(10,0,-1)))
    dec_grp = dec.groupby("decile").agg(rev=("y_true","sum"), n=("y_true","size")).reset_index()
    dec_grp = dec_grp.sort_values("decile", ascending=False)
    dec_grp["cum_rev_share"] = (dec_grp["rev"].cumsum() / dec_grp["rev"].sum()).round(4)

    # permutation feature importance (model-agnostic)
    try:
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        fi = pd.DataFrame({"feature": feature_cols, "importance": perm.importances_mean})
        fi = fi.sort_values("importance", ascending=False)
    except Exception:
        # fallback using model native importance (if any)
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            fi = pd.DataFrame({"feature": feature_cols, "importance": imp}).sort_values("importance", ascending=False)
        else:
            fi = pd.DataFrame({"feature": feature_cols, "importance": 0.0})

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return model, metrics, fi, dec_grp, (X_test.index, y_test, pred, feature_cols)


def main():
    logger.info("🔧 Loading raw data ...")
    tx, cust = _read_raw()
    tx = _clean_transactions(tx)

    logger.info("📌 Building snapshot + labels (LTV 90d) ...")
    hist, y, snapshot = _build_snapshot_label(tx, horizon_days=90)
    logger.info("Snapshot date: %s", snapshot.date())

    logger.info("🧮 Feature engineering (RFM, channel share, interpurchase) ...")
    rfm = _rfm_features(hist, ref_date=snapshot)
    rfm = rfm.merge(y, on="customer_id", how="left").fillna({"ltv_90d": 0.0})
    rfm = _rfm_scoring(rfm)

    # Save RFM & features for modeling
    rfm_out = PROC_DIR / "rfm.csv"
    rfm.to_csv(rfm_out, index=False)
    logger.info("✅ Saved RFM: %s", rfm_out)

    features_out = PROC_DIR / "features_training.csv"
    rfm.to_csv(features_out, index=False)
    logger.info("✅ Saved modeling dataset: %s", features_out)

    logger.info("📊 Cohort retention heatmap data ...")
    cohorts = _monthly_cohort_retention(tx)
    cohorts_out = PROC_DIR / "cohorts_retention.csv"
    cohorts.to_csv(cohorts_out, index=False)
    logger.info("✅ Saved cohorts: %s", cohorts_out)

    logger.info("🤖 Training model ...")
    model, metrics, fi, dec_grp, packed = _train_model(rfm, target_col="ltv_90d")
    (idx_test, y_test, pred_test, feat_cols) = packed

    # Predictions export
    # Map back customer_id using index if available; ensure alignment
    test_ids = rfm.iloc[idx_test]["customer_id"].values if isinstance(idx_test, pd.Index) else rfm.loc[idx_test]["customer_id"].values
    preds_df = pd.DataFrame({
        "customer_id": test_ids,
        "y_true_ltv90": y_test.values,
        "y_pred_ltv90": pred_test
    })
    preds_df["decile"] = pd.qcut(preds_df["y_pred_ltv90"].rank(method="first"), 10, labels=list(range(10,0,-1)))
    preds_out = PRED_DIR / "predictions.csv"
    preds_df.to_csv(preds_out, index=False)
    logger.info("✅ Saved predictions: %s", preds_out)

    # Feature importance
    fi_out = PROC_DIR / "feature_importances.csv"
    fi.to_csv(fi_out, index=False)
    logger.info("✅ Saved feature importances: %s", fi_out)

    # Decile lift table
    lift_out = PROC_DIR / "decile_lift.csv"
    dec_grp.to_csv(lift_out, index=False)
    logger.info("✅ Saved decile lift: %s", lift_out)

    # Metrics
    metrics_out = MODEL_DIR / "metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("✅ Metrics: %s | %s", metrics, metrics_out)

    logger.info("🎉 Analysis pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        raise