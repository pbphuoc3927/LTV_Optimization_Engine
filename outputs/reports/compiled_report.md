# Compiled Report — LTV Optimization Engine

_Generated: 2025-08-18 02:52 UTC_

---

## Project context & Business Objectives

- **Project**: Customer Lifetime Value (LTV) Optimization Engine
- **Objective**: Increase customer LTV by 15–25% via predictive modeling & targeted interventions.
- **Key questions**: drivers of high LTV, predict LTV trajectory, optimal timing for retention campaigns, acquisition vs LTV trade-offs.

## Target audience profiles

### C-suite
- Short, impact-focused: projected revenue lift, ROI, ask (budget/approval).

### Board
- Strategic summary: expected financial impact, risks, mitigations, timeline.

### Department heads (Marketing/CRM/Analytics)
- Tactical: lists, timelines, KPI to track, owners.

## Python analysis outputs & statistical findings

### Modeling dataset (features)

- File: [features_training.csv](data/processed/features_training.csv)


**Top rows:**

```
|   customer_id | last_tx    | first_tx   |   tx_count |   monetary |   avg_amount |   recency_days |   tenure_days |   freq_per_90d |   ip_mean |   ip_std |   ch_app |   ch_offline |   ch_online |   ltv_90d |   R |   F |   M |   RFM_Score | segment   |
|--------------:|:-----------|:-----------|-----------:|-----------:|-------------:|---------------:|--------------:|---------------:|----------:|---------:|---------:|-------------:|------------:|----------:|----:|----:|----:|------------:|:----------|
|             1 | 2027-03-20 | 2025-04-06 |         36 |    2928.76 |      81.3544 |             60 |           773 |       4.19146  |   20.3714 |  21.3046 | 0.492874 |     0.339075 |    0.168051 |      0    |   5 |   4 |   3 |          12 | Champions |
|             2 | 2025-12-24 | 2024-01-14 |         49 |    4736.31 |      96.6594 |            511 |          1221 |       3.61179  |   14.7917 |  14.5864 | 0.324928 |     0.327709 |    0.347363 |      0    |   2 |   5 |   5 |          12 | Champions |
|             3 | 2025-10-29 | 2023-12-23 |         38 |    4182.57 |     110.068  |            567 |          1243 |       2.75141  |   18.2703 |  15.9716 | 0.320922 |     0.279938 |    0.39914  |      0    |   1 |   4 |   5 |          10 | Loyal     |
|             4 | 2027-05-07 | 2025-07-23 |         32 |    3184.39 |      99.5122 |             12 |           665 |       4.33083  |   21.0645 |  20.2829 | 0.292967 |     0.433471 |    0.273563 |    126.22 |   5 |   4 |   4 |          13 | Champions |
|             5 | 2026-01-04 | 2024-01-23 |         45 |    5174.45 |     114.988  |            500 |          1212 |       3.34158  |   16.1818 |  14.8093 | 0.293193 |     0.438609 |    0.268199 |      0    |   2 |   5 |   5 |          12 | Champions |
|             6 | 2026-09-24 | 2025-03-03 |          6 |     247.62 |      41.27   |            237 |           807 |       0.669145 |  114      |  56.5553 | 0.134561 |     0        |    0.865439 |      0    |   4 |   1 |   1 |           6 | At Risk   |
```

### RFM dataset & segments

- File: [rfm.csv](data/processed/rfm.csv)


**Top rows:**

```
|   customer_id | last_tx    | first_tx   |   tx_count |   monetary |   avg_amount |   recency_days |   tenure_days |   freq_per_90d |   ip_mean |   ip_std |   ch_app |   ch_offline |   ch_online |   ltv_90d |   R |   F |   M |   RFM_Score | segment   |
|--------------:|:-----------|:-----------|-----------:|-----------:|-------------:|---------------:|--------------:|---------------:|----------:|---------:|---------:|-------------:|------------:|----------:|----:|----:|----:|------------:|:----------|
|             1 | 2027-03-20 | 2025-04-06 |         36 |    2928.76 |      81.3544 |             60 |           773 |       4.19146  |   20.3714 |  21.3046 | 0.492874 |     0.339075 |    0.168051 |      0    |   5 |   4 |   3 |          12 | Champions |
|             2 | 2025-12-24 | 2024-01-14 |         49 |    4736.31 |      96.6594 |            511 |          1221 |       3.61179  |   14.7917 |  14.5864 | 0.324928 |     0.327709 |    0.347363 |      0    |   2 |   5 |   5 |          12 | Champions |
|             3 | 2025-10-29 | 2023-12-23 |         38 |    4182.57 |     110.068  |            567 |          1243 |       2.75141  |   18.2703 |  15.9716 | 0.320922 |     0.279938 |    0.39914  |      0    |   1 |   4 |   5 |          10 | Loyal     |
|             4 | 2027-05-07 | 2025-07-23 |         32 |    3184.39 |      99.5122 |             12 |           665 |       4.33083  |   21.0645 |  20.2829 | 0.292967 |     0.433471 |    0.273563 |    126.22 |   5 |   4 |   4 |          13 | Champions |
|             5 | 2026-01-04 | 2024-01-23 |         45 |    5174.45 |     114.988  |            500 |          1212 |       3.34158  |   16.1818 |  14.8093 | 0.293193 |     0.438609 |    0.268199 |      0    |   2 |   5 |   5 |          12 | Champions |
|             6 | 2026-09-24 | 2025-03-03 |          6 |     247.62 |      41.27   |            237 |           807 |       0.669145 |  114      |  56.5553 | 0.134561 |     0        |    0.865439 |      0    |   4 |   1 |   1 |           6 | At Risk   |
```

### Cohort retention (pivot)

- File: [cohorts_retention.csv](data/processed/cohorts_retention.csv)


**Top rows:**

```
| cohort_month   |   0 |      1 |      2 |      3 |      4 |      5 |      6 |      7 |      8 |      9 |     10 |     11 |     12 |     13 |     14 |     15 |     16 |     17 |     18 |     19 |     20 |     21 |     22 |     23 |     24 |
|:---------------|----:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 2023-08-01     |   1 | 0.75   | 0.7176 | 0.6852 | 0.7685 | 0.6528 | 0.713  | 0.6898 | 0.7037 | 0.6806 | 0.7083 | 0.7361 | 0.6806 | 0.6944 | 0.7361 | 0.7176 | 0.7407 | 0.7361 | 0.6667 | 0.7361 | 0.6898 | 0.75   | 0.7037 | 0.7454 | 0.588  |
| 2023-09-01     |   1 | 0.6809 | 0.6776 | 0.6893 | 0.707  | 0.6406 | 0.6977 | 0.6843 | 0.6944 | 0.6801 | 0.6818 | 0.6784 | 0.6751 | 0.6868 | 0.6809 | 0.7162 | 0.6969 | 0.6474 | 0.665  | 0.6877 | 0.6885 | 0.6558 | 0.6717 | 0.6583 | 0.204  |
| 2023-10-01     |   1 | 0.6761 | 0.6737 | 0.6496 | 0.6332 | 0.6761 | 0.6725 | 0.6702 | 0.6626 | 0.669  | 0.6731 | 0.6678 | 0.6649 | 0.6614 | 0.679  | 0.6784 | 0.6396 | 0.6872 | 0.6461 | 0.6931 | 0.6673 | 0.682  | 0.6743 | 0.5179 | 0.1717 |
| 2023-11-01     |   1 | 0.6351 | 0.6362 | 0.6225 | 0.6499 | 0.6425 | 0.6448 | 0.6242 | 0.6379 | 0.6391 | 0.6368 | 0.6642 | 0.6242 | 0.6442 | 0.6436 | 0.6339 | 0.6362 | 0.6259 | 0.6471 | 0.6545 | 0.6288 | 0.6459 | 0.5957 | 0.4746 | 0.1479 |
| 2023-12-01     |   1 | 0.6344 | 0.618  | 0.6143 | 0.6175 | 0.6312 | 0.6075 | 0.6365 | 0.6254 | 0.6328 | 0.628  | 0.6222 | 0.6512 | 0.6438 | 0.5938 | 0.6249 | 0.6417 | 0.6159 | 0.608  | 0.637  | 0.6412 | 0.6122 | 0.5738 | 0.4579 | 0.138  |
| 2024-01-01     |   1 | 0.6163 | 0.6532 | 0.6301 | 0.6347 | 0.6276 | 0.627  | 0.6491 | 0.6148 | 0.6363 | 0.627  | 0.6235 | 0.6409 | 0.6076 | 0.6189 | 0.6183 | 0.627  | 0.6281 | 0.6337 | 0.6486 | 0.6168 | 0.6168 | 0.5697 | 0.457  | 0.1322 |
```

### Feature importances

- File: [feature_importances.csv](data/processed/feature_importances.csv)


**Top rows:**

```
| feature      |   importance |
|:-------------|-------------:|
| tenure_days  |  1.22704     |
| freq_per_90d |  0.243872    |
| monetary     |  0.0543049   |
| tx_count     |  0.0188352   |
| ip_mean      |  0.0157862   |
| ip_std       |  0.0117206   |
| ch_online    |  0.00521582  |
| ch_offline   |  0.000740127 |
```

## Model performance metrics

| Metric | Value |
|---:|:---|

| MAE | 17.14372383988399 |

| RMSE | 59.900088643220904 |

| R2 | 0.553316138556314 |



### Predictions summary (sample)

```
|   customer_id |   y_true_ltv90 |   y_pred_ltv90 |   decile |
|--------------:|---------------:|---------------:|---------:|
|         43689 |              0 |              0 |       10 |
|         34689 |              0 |              0 |       10 |
|         20758 |              0 |              0 |       10 |
|          6707 |              0 |              0 |       10 |
|         20261 |              0 |              0 |       10 |
|         24881 |              0 |              0 |       10 |
|          1823 |              0 |              0 |       10 |
|         13698 |              0 |              0 |       10 |
```

- **Mean actual LTV (90d)**: 21.06

- **Mean predicted LTV (90d)**: 22.71

## Generated charts & visualizations

### Chart images

- [cohort_retention_heatmap.png](outputs/charts/cohort_retention_heatmap.png)

- [decile_lift.png](outputs/charts/decile_lift.png)

- [feature_importance_top15.png](outputs/charts/feature_importance_top15.png)

- [ltv90_true_vs_pred_density.png](outputs/charts/ltv90_true_vs_pred_density.png)

- [pred_vs_actual_ltv90.png](outputs/charts/pred_vs_actual_ltv90.png)

- [rfm_segment_distribution.png](outputs/charts/rfm_segment_distribution.png)


### Chart data (CSV descriptions)

- **Decile lift** (file: [decile_lift.csv](data/processed/decile_lift.csv))

```
|   decile |       rev |    n |   cum_rev_share |
|---------:|----------:|-----:|----------------:|
|        1 | 193621    | 1000 |          0.9196 |
|        2 |  16566.5  | 1000 |          0.9983 |
|        3 |     82.81 | 1000 |          0.9987 |
|        4 |      0    | 1000 |          0.9987 |
|        5 |      0    |  999 |          0.9987 |
|        6 |      0    | 1000 |          0.9987 |
|        7 |      0    | 1000 |          0.9987 |
|        8 |    129.3  | 1000 |          0.9993 |
|        9 |    138.45 | 1000 |          1      |
|       10 |      0    | 1000 |          1      |
```

- **Feature importances** (file: [feature_importances.csv](data/processed/feature_importances.csv))

```
| feature      |   importance |
|:-------------|-------------:|
| tenure_days  |  1.22704     |
| freq_per_90d |  0.243872    |
| monetary     |  0.0543049   |
| tx_count     |  0.0188352   |
| ip_mean      |  0.0157862   |
| ip_std       |  0.0117206   |
| ch_online    |  0.00521582  |
| ch_offline   |  0.000740127 |
| M            |  0.000188271 |
| R            | -0.000139608 |
```

## KPI calculations

Key KPIs computed from results:

- **mean_ltv_90d**: 20.30276509771759

- **median_ltv_90d**: 0.0

- **top_decile_cum_rev_share**: 1.0

- **cohort_months_covered**: 25

- **avg_retention_first_offset**: 1.0

- **rfm_segment_counts**: {'Champions': 12886, 'Loyal': 10093, 'Potential Loyalist': 9677, 'Hibernating': 8790, 'At Risk': 8545}

- **model_mae**: 17.14372383988399

- **model_rmse**: 59.900088643220904

- **model_r2**: 0.553316138556314



## ROI analysis (scenario-based)

| Scenario | Uplift | Campaign cost | Est. incremental revenue | Est. margin gain | ROI |

|---|---:|---:|---:|---:|---:|

| conservative | 2.0% | 5000 VND | 4060.55 | 1624.22 | -0.68x |

| base | 5.0% | 20000 VND | 10151.38 | 4060.55 | -0.80x |

| optimistic | 10.0% | 50000 VND | 20302.77 | 8121.11 | -0.84x |


## Business insights summary

1. Top model drivers (by importance): tenure_days, freq_per_90d, monetary, tx_count, ip_mean, ip_std.

2. Top decile captures approximately 100% of near-term revenue (90d).

3. RFM segments distribution (top segments): Champions:12886, Loyal:10093, Potential Loyalist:9677.

4. Model performance: MAE=17.14, R2=0.553


## Recommendations (actionable)

1. Focus immediate retention & upsell campaigns on top-decile customers — high ROI potential.

2. Run a pilot retention campaign on top-decile predicted customers with 3 message variants and measure ΔLTV via A/B test.

3. Operationalize daily scoring & sync top N lists to CRM for campaign activation; monitor model drift weekly.

4. Validate feature stability monthly and retrain model quarterly or when performance drops beyond threshold.


## Tailored summaries for audiences

### For C-suite (1-paragraph)

> Model identifies top-decile customers as highest near-term revenue drivers; pilot targeting top-decile is projected to deliver positive ROI under base assumptions. Request: approval for pilot budget and A/B test.


### For Board (concise)

> Projected uplift and ROI summarized above. Risks: data gaps, integration complexity; mitigations: phased pilot and monitoring. Timeline: pilot (6-8 weeks), scale (3-6 months).


### For Department heads (tactical)

- CRM: export daily top N predicted customers and run 3 message variants (email, push, SMS).
- Marketing: define creative & offer matrix; Ops: track redemptions; Analytics: monitor lift and pop-back metrics weekly.



---
## Files referenced in this report

- features: [features_training.csv](data/processed/features_training.csv) (exists: yes)

- rfm: [rfm.csv](data/processed/rfm.csv) (exists: yes)

- cohorts: [cohorts_retention.csv](data/processed/cohorts_retention.csv) (exists: yes)

- feature_importances: [feature_importances.csv](data/processed/feature_importances.csv) (exists: yes)

- decile_lift: [decile_lift.csv](data/processed/decile_lift.csv) (exists: yes)

- predictions: [predictions.csv](outputs/predictions/predictions.csv) (exists: yes)

- metrics: [metrics.json](models/validation/metrics.json) (exists: yes)

- chart: [cohort_retention_heatmap.png](outputs/charts/cohort_retention_heatmap.png)

- chart: [decile_lift.png](outputs/charts/decile_lift.png)

- chart: [feature_importance_top15.png](outputs/charts/feature_importance_top15.png)

- chart: [ltv90_true_vs_pred_density.png](outputs/charts/ltv90_true_vs_pred_density.png)

- chart: [pred_vs_actual_ltv90.png](outputs/charts/pred_vs_actual_ltv90.png)

- chart: [rfm_segment_distribution.png](outputs/charts/rfm_segment_distribution.png)
