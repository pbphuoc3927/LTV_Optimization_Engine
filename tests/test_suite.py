import unittest
from pathlib import Path
import pandas as pd
import json
import os

class TestLTVAnalysisOutputs(unittest.TestCase):
    def setUp(self):
        # repo_root (parent of tests/)
        self.ROOT = Path(__file__).resolve().parents[1]

        # Các thư mục chuẩn trong project
        self.RAW_DIR = self.ROOT / "data" / "raw"
        self.PROC_DIR = self.ROOT / "data" / "processed"
        self.OUT_CHARTS = self.ROOT / "outputs" / "charts"
        self.OUT_PRED = self.ROOT / "outputs" / "predictions"
        self.MODEL_DIR = self.ROOT / "models" / "validation"
        self.LOG_DIR = self.ROOT / "logs" / "application"

        # Các file quan trọng đầu ra pipeline (tên đúng theo analysis_pipeline.py)
        self.tx_file = self.RAW_DIR / "transactions.csv"
        self.cust_file = self.RAW_DIR / "customers.csv"

        self.rfm_file = self.PROC_DIR / "rfm.csv"
        self.features_file = self.PROC_DIR / "features_training.csv"
        self.cohort_file = self.PROC_DIR / "cohorts_retention.csv"
        self.fi_file = self.PROC_DIR / "feature_importances.csv"
        self.lift_file = self.PROC_DIR / "decile_lift.csv"

        self.pred_file = self.OUT_PRED / "predictions.csv"
        self.metrics_file = self.MODEL_DIR / "metrics.json"

    # -------- Existence tests --------
    def test_raw_files_exist(self):
        self.assertTrue(self.tx_file.exists(), f"Missing raw transactions file: {self.tx_file}")
        self.assertTrue(self.cust_file.exists(), f"Missing raw customers file: {self.cust_file}")

    def test_processed_files_exist(self):
        missing = [p for p in [self.rfm_file, self.features_file, self.cohort_file, self.fi_file, self.lift_file] if not p.exists()]
        self.assertFalse(missing, f"Missing processed files: {missing}")

    def test_predictions_exist(self):
        self.assertTrue(self.pred_file.exists(), f"Missing predictions file: {self.pred_file}")

    def test_metrics_exist(self):
        self.assertTrue(self.metrics_file.exists(), f"Missing metrics file: {self.metrics_file}")

    def test_charts_exist(self):
        # At least one PNG in outputs/charts
        if not self.OUT_CHARTS.exists():
            self.fail(f"Charts folder not found: {self.OUT_CHARTS}")
        pngs = list(self.OUT_CHARTS.glob("*.png"))
        self.assertTrue(len(pngs) >= 1, f"No PNG charts found in {self.OUT_CHARTS}")

    # -------- Content / schema tests --------
    def test_features_structure(self):
        df = pd.read_csv(self.features_file)
        required = {"customer_id", "ltv_90d", "recency_days", "tx_count", "monetary"}
        missing = required - set(df.columns)
        self.assertFalse(missing, f"features_training.csv is missing columns: {missing}")
        self.assertFalse(df.empty, "features_training.csv is empty")
        # basic value checks
        self.assertTrue((df["ltv_90d"] >= 0).all(), "ltv_90d contains negative values")

    def test_rfm_structure(self):
        df = pd.read_csv(self.rfm_file)
        required = {"customer_id", "R", "F", "M", "RFM_Score", "segment"}
        missing = required - set(df.columns)
        self.assertFalse(missing, f"rfm.csv is missing columns: {missing}")

    def test_predictions_structure_and_values(self):
        df = pd.read_csv(self.pred_file)
        required = {"customer_id", "y_true_ltv90", "y_pred_ltv90", "decile"}
        missing = required - set(df.columns)
        self.assertFalse(missing, f"predictions.csv is missing columns: {missing}")
        self.assertFalse(df.empty, "predictions.csv is empty")
        # numeric checks
        self.assertTrue(pd.api.types.is_numeric_dtype(df["y_true_ltv90"]), "y_true_ltv90 must be numeric")
        self.assertTrue(pd.api.types.is_numeric_dtype(df["y_pred_ltv90"]), "y_pred_ltv90 must be numeric")

    def test_metrics_content(self):
        with open(self.metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in ("mae", "rmse", "r2"):
            self.assertIn(k, data, f"metrics.json missing key: {k}")
            self.assertIsInstance(data[k], (int, float), f"metrics.json {k} must be numeric")
        # sanity ranges
        self.assertGreaterEqual(data["mae"], 0, "MAE should be >= 0")
        self.assertGreaterEqual(data["rmse"], 0, "RMSE should be >= 0")

    # -------- Optional smoke checks --------
    def test_cohort_retention_pivot(self):
        df = pd.read_csv(self.cohort_file)
        # cohort_retention.csv expected to have cohort_month and month offset columns (0..N)
        self.assertTrue("cohort_month" in df.columns, "cohorts_retention.csv missing 'cohort_month' column")
        self.assertTrue(df.shape[1] >= 2, "cohorts_retention.csv seems to have too few columns")

    # -------- Business benchmark tests (added) --------
    def test_mean_ltv_positive(self):
        df = pd.read_csv(self.features_file)
        # Guard: if column not present, fail early with clear message
        self.assertIn("ltv_90d", df.columns, "features_training.csv missing 'ltv_90d'")
        mean_ltv = float(df["ltv_90d"].mean())
        self.assertGreater(mean_ltv, 0.0, f"Average ltv_90d should be > 0 (got {mean_ltv:.4f})")

    def test_min_customers_count(self):
        df = pd.read_csv(self.features_file)
        n_customers = int(df["customer_id"].nunique())
        self.assertGreaterEqual(n_customers, 50, f"Number of unique customers should be >= 50 (got {n_customers})")


if __name__ == "__main__":
    unittest.main(verbosity=2)