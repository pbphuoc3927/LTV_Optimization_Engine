from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
PRED = ROOT / "outputs" / "predictions"
CHARTS = ROOT / "outputs" / "charts"
MODEL = ROOT / "models" / "validation"
REPORT_DIR = ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MD = REPORT_DIR / "compiled_report.md"
OUT_JSON = REPORT_DIR / "compiled_report.json"

# Expected files
FILES = {
    "features": PROC / "features_training.csv",
    "rfm": PROC / "rfm.csv",
    "cohorts": PROC / "cohorts_retention.csv",
    "feature_importances": PROC / "feature_importances.csv",
    "decile_lift": PROC / "decile_lift.csv",
    "predictions": PRED / "predictions.csv",
    "metrics": MODEL / "metrics.json",
}

# Default ROI scenario parameters (can be tuned or loaded from config)
ROI_PARAMS = {
    "target_group_size": 10000,     # number of customers targeted in scenario
    "gross_margin": 0.40,           # gross margin %
    # uplift assumptions: percent uplift in LTV for targeted customers
    "uplift": {
        "conservative": 0.02,  # 2% uplift
        "base": 0.05,          # 5% uplift
        "optimistic": 0.10     # 10% uplift
    },
    "campaign_cost": {
        "conservative": 5000.0,
        "base": 20000.0,
        "optimistic": 50000.0
    },
    "currency": "VND"
}

# helper loaders
def load_csv_safe(p: Path):
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            return {"__error__": f"Failed to read CSV {p}: {e}"}
    return None

def load_json_safe(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return {"__error__": f"Failed to read JSON {p}: {e}"}
    return None

def top_table_md(df, n=10):
    if df is None:
        return "_(missing)_\n"
    if isinstance(df, dict) and "__error__" in df:
        return f"_Error reading data: {df['__error__']}_\n"
    return df.head(n).to_markdown(index=False)

def path_md_link(p: Path, label=None):
    if not p:
        return ""
    label = label or p.name
    rel = p.relative_to(ROOT)
    return f"[{label}]({rel.as_posix()})"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def compute_kpis(dataframes):
    """Compute KPI dictionary from loaded dataframes"""
    kpis = {}
    feats = dataframes.get("features")
    preds = dataframes.get("predictions")
    cohorts = dataframes.get("cohorts")
    deciles = dataframes.get("decile_lift")
    rfm = dataframes.get("rfm")

    # Mean & median LTV (90d)
    if isinstance(feats, pd.DataFrame) and "ltv_90d" in feats.columns:
        mean_ltv = float(feats["ltv_90d"].mean())
        median_ltv = float(feats["ltv_90d"].median())
        kpis["mean_ltv_90d"] = mean_ltv
        kpis["median_ltv_90d"] = median_ltv
    elif isinstance(preds, pd.DataFrame) and "y_true_ltv90" in preds.columns:
        kpis["mean_ltv_90d"] = float(preds["y_true_ltv90"].mean())
        kpis["median_ltv_90d"] = float(preds["y_true_ltv90"].median())

    # Decile capture: top decile cumulative rev share (if present)
    if isinstance(deciles, pd.DataFrame) and {"decile", "cum_rev_share"}.issubset(set(deciles.columns)):
        # decile column might be strings; ensure proper sort
        try:
            # select decile == max (assume deciles labeled 10..1)
            top_decile = deciles.sort_values("decile", ascending=False).iloc[0]
            kpis["top_decile_cum_rev_share"] = float(top_decile.get("cum_rev_share", np.nan))
        except Exception:
            kpis["top_decile_cum_rev_share"] = None

    # Retention baseline: cohort month 0 retention = 1 by construction; look at month_offset=1 average
    if isinstance(cohorts, pd.DataFrame):
        # Convert to numeric columns except cohort_month
        month_cols = [c for c in cohorts.columns if c != "cohort_month"]
        if month_cols:
            try:
                # compute mean retention at month 1 if exists else first numeric column after cohort_month
                col1 = month_cols[0]
                kpis["cohort_months_covered"] = len(month_cols)
                kpis["avg_retention_first_offset"] = float(pd.to_numeric(cohorts[col1], errors="coerce").mean())
            except Exception:
                pass

    # RFM top segments counts
    if isinstance(rfm, pd.DataFrame) and "segment" in rfm.columns:
        seg_counts = rfm["segment"].value_counts().to_dict()
        kpis["rfm_segment_counts"] = seg_counts

    # Basic model accuracy (from metrics.json)
    if isinstance(dataframes.get("metrics"), dict):
        m = dataframes["metrics"]
        for key in ("mae", "rmse", "r2"):
            if key in m:
                kpis[f"model_{key}"] = safe_float(m[key])

    return kpis

def compute_roi_scenarios(kpis, params=ROI_PARAMS):
    scenarios = {}
    mean_ltv = kpis.get("mean_ltv_90d") or 0.0
    for name, uplift_pct in params["uplift"].items():
        campaign_cost = params["campaign_cost"].get(name, params["campaign_cost"]["base"])
        N = params["target_group_size"]
        gross_margin = params["gross_margin"]
        est_incremental_revenue = N * mean_ltv * uplift_pct
        est_margin_gain = est_incremental_revenue * gross_margin
        roi = None
        if campaign_cost > 0:
            roi = (est_margin_gain - campaign_cost) / campaign_cost
        scenarios[name] = {
            "name": name,
            "uplift_pct": uplift_pct,
            "target_group_size": N,
            "campaign_cost": campaign_cost,
            "est_incremental_revenue": est_incremental_revenue,
            "est_margin_gain": est_margin_gain,
            "roi": roi,
            "currency": params.get("currency", "")
        }
    return scenarios

def build_business_insights(kpis, dataframes):
    """
    Create human-friendly insights and recommended actions based on KPIs and important features.
    """
    insights = []
    recs = []

    # Top features
    fi = dataframes.get("feature_importances")
    top_features = []
    if isinstance(fi, pd.DataFrame) and "feature" in fi.columns:
        top_features = fi.sort_values("importance", ascending=False).head(10)["feature"].tolist()
        insights.append(f"Top model drivers (by importance): {', '.join(top_features[:6])}.")

    # Decile lift
    dec = dataframes.get("decile_lift")
    if isinstance(dec, pd.DataFrame) and {"decile", "cum_rev_share"}.issubset(dec.columns):
        # compute percent revenue by top 10% (decile '10' or first row highest)
        try:
            top_row = dec.sort_values("decile", ascending=False).iloc[0]
            share = float(top_row.get("cum_rev_share", 0.0))
            insights.append(f"Top decile captures approximately {share:.0%} of near-term revenue (90d).")
            if share >= 0.25:
                recs.append("Focus immediate retention & upsell campaigns on top-decile customers — high ROI potential.")
            else:
                recs.append("Consider broader targeting beyond top-decile and run segmentation-based A/B tests.")
        except Exception:
            pass

    # RFM segments
    rfm = dataframes.get("rfm")
    if isinstance(rfm, pd.DataFrame) and "segment" in rfm.columns:
        seg_counts = rfm["segment"].value_counts().to_dict()
        insights.append(f"RFM segments distribution (top segments): {', '.join([f'{k}:{v}' for k,v in list(seg_counts.items())[:3]])}.")

    # Model performance
    if kpis.get("model_mae") is not None:
        mae = kpis["model_mae"]
        r2 = kpis.get("model_r2")
        insights.append(f"Model performance: MAE={mae:.2f}" + (f", R2={r2:.3f}" if r2 is not None else ""))

    # Actionable recommendations (generic templates)
    recs.append("Run a pilot retention campaign on top-decile predicted customers with 3 message variants and measure ΔLTV via A/B test.")
    recs.append("Operationalize daily scoring & sync top N lists to CRM for campaign activation; monitor model drift weekly.")
    recs.append("Validate feature stability monthly and retrain model quarterly or when performance drops beyond threshold.")

    return insights, recs, top_features

def render_markdown(kpis, scenarios, insights, recs, top_features, dataframes, charts_list):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md = []
    md.append(f"# Compiled Report — LTV Optimization Engine\n\n_Generated: {now}_\n\n---\n")

    # Project context & business objectives
    md.append("## Project context & Business Objectives\n")
    md.append(
        "- **Project**: Customer Lifetime Value (LTV) Optimization Engine\n"
        "- **Objective**: Increase customer LTV by 15–25% via predictive modeling & targeted interventions.\n"
        "- **Key questions**: drivers of high LTV, predict LTV trajectory, optimal timing for retention campaigns, acquisition vs LTV trade-offs.\n"
    )

    # Target audiences
    md.append("## Target audience profiles\n")
    md.append("### C-suite\n- Short, impact-focused: projected revenue lift, ROI, ask (budget/approval).\n")
    md.append("### Board\n- Strategic summary: expected financial impact, risks, mitigations, timeline.\n")
    md.append("### Department heads (Marketing/CRM/Analytics)\n- Tactical: lists, timelines, KPI to track, owners.\n")

    # Python analysis outputs & statistical findings
    md.append("## Python analysis outputs & statistical findings\n")
    # include small excerpts or links to CSVs
    def file_section(name, key, n=8):
        p = FILES.get(key)
        md.append(f"### {name}\n")
        if p.exists():
            md.append(f"- File: {path_md_link(p)}\n")
            df = dataframes.get(key)
            if isinstance(df, pd.DataFrame):
                md.append("\n**Top rows:**\n")
                md.append("```\n" + top_table_md(df, n=n) + "\n```\n")
            else:
                md.append(f"_(Unable to preview {p.name})_\n")
        else:
            md.append(f"_(Missing file: expected {p})_\n")
    file_section("Modeling dataset (features)", "features", n=6)
    file_section("RFM dataset & segments", "rfm", n=6)
    file_section("Cohort retention (pivot)", "cohorts", n=6)
    file_section("Feature importances", "feature_importances", n=8)

    # Model performance metrics
    md.append("## Model performance metrics\n")
    metrics = dataframes.get("metrics")
    if metrics is None:
        md.append("_(metrics.json missing)_\n")
    elif isinstance(metrics, dict) and "__error__" in metrics:
        md.append(f"_(Error reading metrics.json: {metrics['__error__']})_\n")
    else:
        md.append("| Metric | Value |\n|---:|:---|\n")
        for k in ("mae", "rmse", "r2"):
            v = metrics.get(k)
            md.append(f"| {k.upper()} | {v if v is not None else 'N/A'} |\n")
        md.append("\n")
    # Predictions summary
    md.append("### Predictions summary (sample)\n")
    pred_df = dataframes.get("predictions")
    if isinstance(pred_df, pd.DataFrame):
        md.append("```\n" + top_table_md(pred_df, n=8) + "\n```\n")
        # basic comparison stats
        md.append(f"- **Mean actual LTV (90d)**: {np.round(pred_df['y_true_ltv90'].mean(), 2):.2f}\n")
        md.append(f"- **Mean predicted LTV (90d)**: {np.round(pred_df['y_pred_ltv90'].mean(),2):.2f}\n")
    else:
        md.append("_(predictions.csv missing)_\n")

    # Charts & visualizations (links to PNG if exist, else describe via CSV)
    md.append("## Generated charts & visualizations\n")
    if CHARTS.exists():
        pngs = list(CHARTS.glob("*.png"))
        if pngs:
            md.append("### Chart images\n")
            for p in pngs:
                md.append(f"- {path_md_link(p)}\n")
        else:
            md.append("_(No PNG charts found in outputs/charts)_\n")
    else:
        md.append("_(Charts folder missing)_\n")

    # If CSVs exist, show how charts can be inspected via CSV
    md.append("\n### Chart data (CSV descriptions)\n")
    if (FILES.get("decile_lift") and FILES["decile_lift"].exists()):
        df = dataframes["decile_lift"]
        md.append("- **Decile lift** (file: " + path_md_link(FILES["decile_lift"]) + ")\n")
        md.append("```\n" + top_table_md(df, n=10) + "\n```\n")
    if (FILES.get("feature_importances") and FILES["feature_importances"].exists()):
        df = dataframes["feature_importances"]
        md.append("- **Feature importances** (file: " + path_md_link(FILES["feature_importances"]) + ")\n")
        md.append("```\n" + top_table_md(df.sort_values('importance', ascending=False), n=10) + "\n```\n")

    # KPI calculations
    md.append("## KPI calculations\n")
    md.append("Key KPIs computed from results:\n")
    for k, v in kpis.items():
        md.append(f"- **{k}**: {v}\n")
    md.append("\n")

    # ROI analysis
    md.append("## ROI analysis (scenario-based)\n")
    md.append("| Scenario | Uplift | Campaign cost | Est. incremental revenue | Est. margin gain | ROI |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for name, s in scenarios.items():
        roi_str = f"{s['roi']:.2f}x" if s["roi"] is not None else "N/A"
        md.append(f"| {name} | {s['uplift_pct']*100:.1f}% | {s['campaign_cost']:.0f} {s['currency']} | "
                  f"{s['est_incremental_revenue']:.2f} | {s['est_margin_gain']:.2f} | {roi_str} |\n")

    # Business insights summary
    md.append("\n## Business insights summary\n")
    if insights:
        for i, it in enumerate(insights, start=1):
            md.append(f"{i}. {it}\n")
    else:
        md.append("_No automated insights generated._\n")

    md.append("\n## Recommendations (actionable)\n")
    if recs:
        for i, it in enumerate(recs, start=1):
            md.append(f"{i}. {it}\n")
    else:
        md.append("_No recommendations generated._\n")

    # Tailored messaging for audiences
    md.append("\n## Tailored summaries for audiences\n")
    md.append("### For C-suite (1-paragraph)\n")
    md.append("> Model identifies top-decile customers as highest near-term revenue drivers; pilot targeting top-decile is projected to deliver positive ROI under base assumptions. Request: approval for pilot budget and A/B test.\n\n")
    md.append("### For Board (concise)\n")
    md.append("> Projected uplift and ROI summarized above. Risks: data gaps, integration complexity; mitigations: phased pilot and monitoring. Timeline: pilot (6-8 weeks), scale (3-6 months).\n\n")
    md.append("### For Department heads (tactical)\n")
    md.append("- CRM: export daily top N predicted customers and run 3 message variants (email, push, SMS).\n- Marketing: define creative & offer matrix; Ops: track redemptions; Analytics: monitor lift and pop-back metrics weekly.\n\n")

    # Append list of all output files included
    md.append("\n---\n## Files referenced in this report\n")
    for k, p in FILES.items():
        md.append(f"- {k}: {path_md_link(p)} (exists: {'yes' if p.exists() else 'no'})\n")
    if CHARTS.exists():
        for p in CHARTS.glob("*.png"):
            md.append(f"- chart: {path_md_link(p)}\n")

    return "\n".join(md)

def main():
    # Load all files (or None)
    dataframes = {}
    for k, p in FILES.items():
        if k == "metrics":
            dataframes[k] = load_json_safe(p)
        else:
            df = load_csv_safe(p)
            dataframes[k] = df if isinstance(df, pd.DataFrame) else df

    # also try load charts list
    charts_list = list(CHARTS.glob("*.png")) if CHARTS.exists() else []

    # compute KPIs
    kpis = compute_kpis(dataframes)

    # compute ROI scenarios
    scenarios = compute_roi_scenarios(kpis, ROI_PARAMS)

    # business insights & recommendations
    insights, recs, top_features = build_business_insights(kpis, dataframes)

    # render markdown
    md_text = render_markdown(kpis, scenarios, insights, recs, top_features, dataframes, charts_list)

    # write outputs
    OUT_MD.write_text(md_text, encoding="utf-8")
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "kpis": kpis,
        "roi_scenarios": scenarios,
        "insights": insights,
        "recommendations": recs,
        "referenced_files": {k: (str(p.relative_to(ROOT)) if p.exists() else None) for k,p in FILES.items()},
        "charts": [str(p.relative_to(ROOT)) for p in charts_list]
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Compiled report written to: {OUT_MD}")
    print(f"✅ Machine summary written to: {OUT_JSON}")

if __name__ == "__main__":
    main()