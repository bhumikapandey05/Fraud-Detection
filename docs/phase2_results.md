# Phase 2 Results — Frozen Release `phase2-rf-v1`

This is the permanent record of the Phase 2 model. The model, its threshold, and every
number below are frozen: they are not retrained, re-tuned or overwritten by later work.

| | |
|---|---|
| Model | Random Forest (300 trees, `class_weight="balanced_subsample"`, seed 42) |
| Decision threshold | **0.0499** (flag if fraud score ≥ 0.0499), tuned on **validation** by lowest cost |
| Cost assumptions | $120 per missed fraud, $5 per false alarm (illustrative) |
| Selected by | highest validation PR-AUC among Logistic Regression, Random Forest, XGBoost |
| Frozen snapshot | `artifacts/frozen/phase2-rf-v1/` (read-only, SHA-256 checksums in `MANIFEST.json`) |
| MLflow | run `phase2-rf-v1 (frozen)`; registered as **`fraud-detector` version 1**, alias **`champion`** |
| Data version | `bd686364a12d` (hash of the exact train / validation / test files) |

## Data

| Split | Period | Transactions | Fraud | Fraud rate |
|---|---|---:|---:|---:|
| Train | oldest 70% | 198,607 | 366 | 0.184% |
| Validation | next 15% | 42,558 | 55 | 0.129% |
| Test | newest 15% | 42,561 | 52 | 0.122% |

## Validation results (used for selection and threshold tuning)

| Model | PR-AUC (95% CI) | Threshold | Precision | Recall | F1 | Caught / missed | False alarms |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.809 (0.711–0.893) | 0.996 | 0.457 | 0.873 | 0.600 | 48 / 7 | 57 |
| **Random Forest** ✓ | **0.874** (0.782–0.950) | 0.050 | 0.538 | 0.891 | 0.671 | 49 / 6 | 42 |
| XGBoost | 0.840 (0.744–0.924) | 0.193 | 0.378 | 0.873 | 0.527 | 48 / 7 | 79 |

## Held-out test result (evaluated once, never used for any decision)

| PR-AUC (95% CI) | Precision | Recall | F1 | Caught | Missed | False alarms | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.778 (0.668–0.876) | 0.429 | 0.750 | 0.545 | 39 of 52 | 13 | 52 of 42,509 | $1,820 (vs $6,240 with no model, −71%) |

## The validation → test drop

| Metric | Validation | Test | Change |
|---|---:|---:|---:|
| PR-AUC | 0.874 | 0.778 | −0.096 |
| ROC-AUC | 0.979 | 0.944 | −0.034 |
| Precision | 0.538 | 0.429 | −0.110 |
| Recall | 0.891 (49/55) | 0.750 (39/52) | −0.141 |
| F1 | 0.671 | 0.545 | −0.126 |
| False alarms | 42 | 52 | +10 |
| Cost reduction vs no model | 85.9% | 70.8% | −15.1 pts |

**Is it real or chance?** Measured with the frozen model and threshold only:

| Check | Result |
|---|---|
| PR-AUC difference, bootstrap 95% CI (2,000 resamples) | −0.231 to +0.039 (includes 0) |
| Recall difference, bootstrap 95% CI | −0.283 to +0.003 (includes 0) |
| Resamples in which test PR-AUC < validation | 92% |
| Recall, Fisher exact test | p = 0.077 |
| Precision, Fisher exact test | p = 0.182 |
| False-alarm rate, Fisher exact test | p = 0.353 |

**Conclusion.** Every metric is lower on test and the direction is consistent (92% of
resamples), so the drop is probably partly real. But **none of the differences is
statistically significant at the 5% level**: with only ~50 frauds per split, each fraud case
is worth about 2 points of recall, and a handful of hard cases can explain the gap.

**Likely causes**

1. **Selection optimism.** The model and the threshold were both chosen for being best
   on validation, so validation flatters them a little ("winner's curse").
2. **Time drift.** Test is the latest period; fraud patterns shift over time.
3. **Small samples.** 52–55 frauds per split give wide uncertainty.

**What we report.** The **test** numbers are the honest performance estimate: about **75% of
fraud caught**, about **43% of alerts genuine**, **0.12% of legitimate customers flagged**, and
**71% lower cost** than no model. The validation numbers are kept only to document how
the model and threshold were selected. The threshold was **not** changed after seeing test.

**What to watch in production (Phase 4):** recall on confirmed fraud, alert volume, and the
score distribution. A drop in recall is the most expensive failure under our cost assumptions.

## How to check this is still true

```bash
fraud-detection verify-frozen --name phase2-rf-v1   # checksums, data identity, test metrics
fraud-detection evaluate --model-dir artifacts/frozen/phase2-rf-v1/model
cat artifacts/frozen/phase2-rf-v1/reports/validation_vs_test.md
```

Re-running `fraud-detection train` with the frozen config reproduces the same model choice,
threshold and metrics (checked on 2026-11-24; logged in MLflow as a `source=train` run).

**Technical note.** The Random Forest averages its trees across several CPU threads, so
repeated scoring can differ in the last floating-point bit (≈2×10⁻¹⁶). No fraud/no-fraud
decision changed in repeated checks, and metrics are compared with a 10⁻⁹ tolerance.
