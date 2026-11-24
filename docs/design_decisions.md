# Design Decisions

Each section states the decision, why it was made, and what was rejected.

# Phase 1 — Data preparation

---

## 1. A Python package, not a notebook

**Decision:** All logic lives in `src/fraud_detection/` as small, importable modules,
run via a CLI (`fraud-detection ...`). There are no notebooks.

**Why:** Code in functions can be tested, reused by later phases (the FastAPI service
will import the same preprocessing code), and reviewed in pull requests. Notebooks hide
state, run out of order, and are hard to test.

The `src/` layout means tests run against the *installed* package, which catches
packaging mistakes early.

## 2. One YAML config, validated on load

**Decision:** Settings live in `configs/config.yaml` and are parsed into frozen
dataclasses that check their own values.

**Why:** Changing the split ratio or scaler should not need a code change. Validating at
load time turns "a weird error 3 minutes into the run" into "split sizes must sum to 1.0"
before anything starts. `pydantic` / `hydra` would also work but are extra dependencies
for little gain at this size.

## 3. Never invent data

**Decision:** If `data/raw/creditcard.csv` is missing, the loader raises
`DatasetNotFoundError` with download instructions. No synthetic fallback exists in the
pipelines.

**Why:** A pipeline that silently runs on fake data produces convincing but meaningless
results. The synthetic table in `tests/conftest.py` is used only to unit test code
behaviour and is clearly labelled as such.

## 4. Validation: errors vs warnings

**Decision:** `ValidationReport` collects *errors* (pipeline stops) and *warnings*
(logged and saved in the report).

| Check | Severity | Reason |
|---|---|---|
| Required columns present, numeric | error | everything downstream depends on them |
| No missing / infinite values | error | the scalers and models cannot handle them |
| Labels ⊆ {0, 1} and both present | error | otherwise this is not our binary problem |
| `Amount ≥ 0`, `Time ≥ 0` | error | impossible values signal a corrupt file |
| Row count ≠ 284,807 | warning | could be a newer file version |
| `Time` beyond 2 days | warning | unexpected but not invalid |
| Duplicate rows | warning | expected in this dataset; handled by cleaning |

A custom validator (≈150 lines) was chosen over Great Expectations / pandera: the rules
are simple, the code is easy to read, and it adds no dependencies. Evidently will cover
drift monitoring in Phase 4.

## 5. Remove exact duplicates *before* splitting

**Decision:** The 1,081 exact duplicate rows (19 fraud) are dropped before the split.

**Why:** If one copy lands in train and another in test, the model is tested on a row it
has already seen, inflating test scores — a classic leakage bug. Duplicate removal is a
deterministic, row-level rule (it learns nothing from the data), so it is safe to apply
to the whole dataset before splitting. The split validator double-checks that no
identical row exists in two splits.

## 6. Time-based split (70 / 15 / 15)

**Decision:** Sort by `Time` and cut at two timestamps: train is the oldest 70%,
validation the next 15%, test the newest 15%.

**Why:**
* **It mirrors production.** A deployed model is trained on past transactions and
  scores future ones. A random split lets the model "see the future", which usually
  makes offline scores look better than real-world performance.
* **Fraud patterns drift.** A time split shows whether the model still works on a later
  period. Notice that the fraud rate is 0.184% in train but ~0.125% in validation/test —
  exactly the kind of shift a random split would hide.
* **Cut on timestamps, not row numbers.** Transactions sharing a timestamp are never split
  across two sets, so `max(train.Time) < min(validation.Time) < min(test.Time)` holds
  strictly (and is tested).

**Imbalance consideration:** With only ~473 frauds after de-duplication, the minimum
fraud count per split matters more than the exact percentage. The pipeline fails if any
split has fewer than `min_positives_per_split` (30) frauds. Real data: 366 / 55 / 52.

**Why a separate validation set?** Phase 2 will tune hyperparameters and choose a
decision threshold. Doing that on the test set would make the final score optimistic;
the test set is touched once, at the end.

**Alternative kept:** `strategy: stratified` gives a random split with an identical fraud
rate in each part. It is useful as a baseline but is not the default.

## 7. Preprocessing fitted on training data only

**Decision:** `build_preprocessor()` returns an *unfitted* scikit-learn
`ColumnTransformer`. The prepare pipeline calls `.fit()` on the training split only and
`.transform()` on all three.

**Why:** If scaling statistics (median, IQR) are computed on all data, information from
validation/test leaks into training. A test (`test_statistics_are_learned_from_train_only`)
verifies the fitted means equal the training means and not the full-data means.

Packaging preprocessing as a scikit-learn object means in Phase 2 it can be placed in a
single `Pipeline(preprocessor, model)` — cross-validation then refits it inside each fold
automatically, and the exact same object is used for serving (no training/serving skew).

The splits are saved **untransformed**; the fitted `preprocessor.joblib` and its JSON
report are saved separately. This keeps raw splits reusable if feature engineering changes.

### Column treatment

| Columns | Treatment | Reason |
|---|---|---|
| `V1`–`V28` | `RobustScaler` | Already PCA outputs, but spreads differ and there are extreme outliers; median/IQR scaling is less affected by them than mean/std. Matters for linear models; harmless for trees. |
| `Amount` | `log1p` → `RobustScaler` | Very right-skewed (median ≈ 22, max ≈ 25,691). `log1p` compresses the tail and handles zero amounts (1,825 of them). |
| `Time` | → `hour_sin`, `hour_cos`; raw value dropped | Raw `Time` just counts up; test values are always bigger than anything in training, so a model would extrapolate. Hour-of-day repeats daily and EDA shows fraud is relatively more common at night. Sine/cosine keeps 23:00 next to 00:00. |
| `Class` | dropped by `remainder="drop"` | The label can never reach the features. |

**Caveat:** The dataset does not say what clock time the first transaction happened at,
so "hour" is relative to the start of recording, not true local time.

## 8. Class imbalance: measured now, handled in Phase 2

**Decision:** No resampling (SMOTE, undersampling) in Phase 1.

**Why:** Resampling must only touch training data — and inside cross-validation, only the
training folds — otherwise synthetic fraud cases leak into evaluation. That belongs in the
modelling pipeline. Phase 1 records the train negative:positive ratio (541.6 : 1) in
`data_preparation_report.json` for class weighting (e.g. XGBoost `scale_pos_weight`).
Evaluation will use precision-recall metrics (PR-AUC, recall at fixed precision), since
accuracy is meaningless here: predicting "never fraud" scores 99.83%.

## 9. EDA on the full file vs the training split

**Decision:** `fraud-detection eda` analyses the full raw file; `eda --on train` analyses
only the training split.

**Why:** Data-quality questions (missing values, duplicates) must look at the whole file.
But decisions that shape the model should ideally rely only on training data. Providing
both makes this trade-off explicit. The Phase 1 choices made from EDA (log-transform the
amount, hour-of-day features) are generic and also hold on the training split alone.

## 10. Storage and dependencies

* **Parquet** for intermediate data: about half the size of the CSV (72 MB vs 151 MB —
  random PCA floats compress poorly), much faster to read, and it keeps column types.
  Needs `pyarrow`.
* **joblib** (installed with scikit-learn) to save the fitted preprocessor.
* **matplotlib** only for charts — `seaborn` is not needed.
* Exact versions are pinned in `requirements.txt` for reproducibility; `pyproject.toml`
  lists looser minimums so the package can be installed elsewhere.
* MLflow, FastAPI, Streamlit, Evidently and Docker are deliberately **not** installed
  yet; each will be added in the phase that needs it. XGBoost was added in Phase 2 (on
  macOS it needs the OpenMP runtime: `brew install libomp`).

## 11. Testing strategy

* **Unit tests** on a small synthetic table with the real schema: fast (~4 s), no
  downloads, and suitable for CI.
* **Leakage tests** that assert the properties that matter most: chronological order,
  no shared timestamps across splits, no duplicate rows across splits, preprocessing
  statistics come from train only, target never in features.
* **Failure tests**: the pipeline must *stop* on bad data (e.g. label `7`) and must not
  write any split files when it does.
* **Integration tests** against the real CSV, auto-skipped if it is absent.

---

# Phase 2 — Model training and evaluation

## 12. Verify the data again before training

**Decision:** `fraud-detection train` starts by re-reading the Parquet splits and running
61 checks: schema/values per split, no duplicates, no shared rows, strict time order,
minimum fraud per split, and row/fraud counts identical to `data_preparation_report.json`.
It also refits the preprocessing inside the model pipeline and confirms it produces exactly
the same output as the saved Phase 1 `preprocessor.joblib`.

**Why:** Phase 2 reads files from disk that anyone could have edited or regenerated with
different settings. Training on stale or altered data is a silent failure; checking costs
two seconds.

## 13. One pipeline object per model (preprocessing + classifier)

**Decision:** Each model is `Pipeline([("preprocessor", build_preprocessor()), ("model", clf)])`.
The Phase 1 preprocessing *code* is reused, and it is refitted inside the pipeline on the
training split. The saved Phase 1 artifact is used only for the consistency check above.

**Why:** One object to fit, save and serve means serving uses exactly the same transformations
as training. It also makes cross-validation leakage-safe later on, because the preprocessing
is refitted inside every fold automatically.

## 14. Class imbalance: class weights, not resampling

| Model | Setting | Effect |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | each fraud counts ~540× in the loss |
| Random Forest | `class_weight="balanced_subsample"` | weights recomputed on each tree's bootstrap sample |
| XGBoost | `scale_pos_weight = neg/pos` (from **train** labels) | same idea for gradient boosting |

**Why not SMOTE/undersampling?** Class weights need no extra dependency, invent no
synthetic transactions, and cannot leak into evaluation because they only exist inside
`fit`. Validation and test keep their natural fraud rate (~0.13%), which is what
production looks like; resampling them would make precision look far better than reality.
A test confirms that training never changes the split files.

## 15. Metrics

* **PR-AUC (average precision)** is the headline metric. It measures how well fraud is
  ranked above legitimate transactions *with a focus on the fraud class*. A random model
  scores about the fraud rate (0.0013), so 0.8 is very strong.
* **ROC-AUC** is reported but not used for selection: with 99.87% negatives it is
  dominated by easy negatives and all three models look similar (0.98–0.99).
* **Precision, recall, F1 and the confusion matrix** are reported at the *tuned*
  threshold, because that is how the model will actually be used.
* **95% bootstrap confidence intervals** (1,000 resamples, fixed seed). With ~50 frauds per
  split, one or two cases more or less moves recall by several points; the intervals keep
  us honest about this.
* **Accuracy is not reported**: "never fraud" is 99.87% accurate.

## 16. Threshold tuning on validation, by expected cost

**Decision:** For each model, choose the threshold that minimises
`cost_false_negative × FN + cost_false_positive × FP` on the validation split
(defaults: $120 per missed fraud, roughly the average fraud amount; $5 per false alarm for
review and customer friction). Ties go to the higher threshold (fewer alerts).
`method: f_beta` (F2 by default) is available when costs are unknown.

**Why:** 0.5 is arbitrary. Class weighting pushes Logistic Regression scores up (at 0.5 it
raises 2,721 false alarms on validation), while the Random Forest is too cautious at 0.5
(it misses 29% of fraud). A cost-based rule makes the business trade-off explicit and easy
to change. The threshold is a learned parameter, so it is fitted on validation, never test.

Search is exact and vectorised: sort scores once, use cumulative sums to get TP/FP at every
distinct score (O(n log n)); a test compares it with brute force.

## 17. Model selection by validation PR-AUC; test used once

**Decision:** The model with the highest validation PR-AUC wins. Only the winner, with its
validation-tuned threshold, is scored on the test split, once. Nothing is changed after that.

**Why:** If the test set influenced any choice (model, threshold, features), its score would
no longer be an unbiased estimate of future performance. A unit test spies on the metric
function and asserts the test labels are evaluated exactly once.

**What happened:** Random Forest (0.874) beat XGBoost (0.840) and Logistic Regression
(0.809) on validation. The confidence intervals overlap a lot, so this ranking is not
certain. On test, the Random Forest reached PR-AUC 0.778. A drop is expected: picking the
highest of three noisy validation scores favours a lucky one (the "winner's curse"), and the
test period is later in time.

**Rejected:** refitting the winner on train + validation before testing. It would use ~20%
more data, but the tuned threshold belongs to the train-only model's score scale, and a
refit model's scores shift, especially for a Random Forest. Kept simple and consistent.

## 18. Saving the inference artifact

**Decision:** Save a `FraudDetector` (fitted pipeline + threshold + expected input columns)
with joblib, plus a human-readable `model_metadata.json`.

**Why:** Keeping the threshold inside the artifact prevents the common production bug of
serving with the default 0.5. The metadata records model, hyperparameters, threshold and
method, all candidates' validation scores, test metrics with intervals, SHA-256 hashes of the
data files, and library versions: enough to audit or reproduce the model. MLflow (Phase 3)
will store the same information in a model registry.

`fraud-detection evaluate` reloads the saved artifact, rescores the test split, and fails
(exit code 1) if any metric differs from what was recorded, which catches broken
environments or swapped files.

## 19. Reproducibility

One seed (`training.random_state = 42`) is passed to every estimator and to the bootstrap.
Random Forest and XGBoost (`tree_method="hist"`) are deterministic for a fixed seed even
with multiple threads. Two runs from a clean state produced identical metrics and
thresholds, and a unit test asserts identical scores for each model type.

## 20. Known limitations / next steps

* Hyperparameters are reasonable defaults, not tuned. Proper tuning would use
  time-ordered cross-validation on the training split (`TimeSeriesSplit`) or the validation
  split, never the test split. This fits naturally with MLflow tracking in Phase 3.
* The validation and test splits are small in fraud cases (55 / 52). Conclusions about
  which model is "best" are tentative; the intervals overlap.
* The Random Forest's tuned threshold (0.05) sits where its scores are sparse, so small
  changes in data move it. Monitoring precision and alert volume after deployment (Phase 4)
  is essential.
* Costs are assumptions. In a real deployment they come from the fraud-operations team.

---

# Phase 3 — Experiment tracking and model versioning

## 21. Freeze released models before adding tracking

**Decision:** Before touching MLflow, the Phase 2 model was frozen into
`artifacts/frozen/phase2-rf-v1/` with `fraud-detection freeze`. The snapshot holds the
detector (pipeline + threshold), metadata, reports, the validation-to-test analysis, the
data-preparation report and the config, plus a `MANIFEST.json` with SHA-256 hashes. Files
are made read-only, the folder appears atomically, and an existing name is never overwritten.

**Why:** `artifacts/model/` is a working folder that every `train` overwrites and
`make clean` deletes. A released model must not depend on nobody re-running a command.
`verify-frozen` proves three things: the files are unchanged (checksums), the split data is
byte-identical to what the model was evaluated on, and the test metrics still reproduce.
Freezing refuses a model that cannot reproduce its own recorded metrics.

## 22. Document the validation → test gap with the frozen model only

**Decision:** `analysis/generalization.py` compares validation and test at the frozen
threshold: bootstrap 95% CIs of the differences (each split resampled independently) and
Fisher exact tests on the 2×2 count tables. It is read-only and is saved inside the snapshot.

**Why:** "Test is lower than validation" is expected after selection. The useful question
is whether the gap is larger than chance. Answer: consistent in direction, not significant at
5% (details in `docs/phase2_results.md`). The threshold was **not** re-tuned on test: that would
turn the test set into a second validation set and leave no unbiased estimate.

## 23. MLflow, fully local: SQLite + a folder

**Decision:** Tracking URI `sqlite:///<project>/mlflow/mlflow.db`, artifacts in
`mlflow/artifacts/`. No server is needed to log; `mlflow ui` is started only to browse.

**Why:** Free, offline, one folder to back up. A database-backed store is MLflow's
recommended setup and fully supports the **Model Registry** and aliases. Paths are absolute
so commands work from any directory. `make clean` does not delete MLflow history.

## 24. Log from saved artifacts, not from training internals

**Decision:** `tracking/mlflow_tracking.log_training_run()` reads what a training run
already saved (`model_metadata.json`, reports, config) and logs it. `train` calls it after
saving, and `track-frozen` calls it on the frozen snapshot.

**Why:** One code path logs both new runs and the historical Phase 2 model, **without
retraining it**. Training stays independent of MLflow (`--no-mlflow` or
`log_training_runs: false` turns tracking off), and tests can run training without a tracking
store.

**What is logged**

| MLflow field | Content |
|---|---|
| Parameters | seed, selection metric, split config, preprocessing config, cleaning, threshold method + costs, imbalance strategy, selected model's hyperparameters, `data_version` |
| Metrics | `threshold`; `val_*` and `test_*` PR-AUC, ROC-AUC, precision, recall, F1, TP/FP/FN/TN, alert rate, cost, cost reduction; 95% CI bounds |
| Datasets | train / validation / test as MLflow datasets (name includes the data version, plus MLflow's own digest) |
| Tags | `source` (`train` / `frozen_snapshot`), `data_version`, SHA-256 of each split file, raw-data SHA-256, snapshot name and manifest hash |
| Artifacts | `reports/` (evaluation report, tables, figures, validation-vs-test), `model_metadata/`, `data/`, `config/`, `frozen/MANIFEST.json` |
| Child runs | one per candidate model with its parameters and validation metrics, for side-by-side comparison |
| Model | `fraud_detector` pyfunc (see 25) |

**Training data version** = first 12 hex characters of a SHA-256 over the three split-file
hashes (`bd686364a12d` for Phase 2). Logging is refused if the files on disk don't match the
hashes recorded at training time, so a run can never claim the wrong data version.

## 25. Register a pyfunc that carries the threshold

**Decision:** The registered model is an MLflow `pyfunc` wrapping the saved
`FraudDetector`. `predict()` returns `fraud_score`, `is_fraud` and `threshold`. The package
source is bundled (`code_paths`) and library versions are pinned from the training metadata.

**Why:** Registering only the scikit-learn pipeline would lose the tuned threshold, and
serving would silently fall back to 0.5. A test loads `models:/fraud-detector@champion`
and checks that it makes the same decisions as the frozen joblib file.

## 26. Aliases instead of stages; champion only moves on purpose

**Decision:** The frozen Phase 2 model is version 1 with alias **`champion`**.
`fraud-detection train` logs a new run but does **not** register by default.
`train --register` creates a new version but **never** moves `champion`. `track-frozen` is
idempotent: running it again reuses the existing run and version.

**Why:** MLflow 3 replaces the old "Staging/Production" stages with aliases. Moving the
champion is a deployment decision. It should be explicit (e.g. in the MLflow UI, or
`MlflowClient().set_registered_model_alias(...)`), happen after review, and never be a side
effect of an experiment.

## 27. Reproducibility evidence

* Re-running `fraud-detection train` with the Phase 2 config on the same data produced the
  same selected model, an identical threshold, and identical validation/test metrics for all
  candidates. It is logged as a separate `source=train` run next to the frozen run.
* Known limitation: Random Forest averages trees across threads, so repeated scoring can
  differ by one floating-point rounding step (~2×10⁻¹⁶) for some rows. No decision changed
  in repeated checks; metric comparisons use a 10⁻⁹ tolerance and tests compare scores with
  1e-12 tolerance and decisions exactly. Setting the forest's `n_jobs=1` at inference would
  make scores bit-identical, at the cost of speed.

---

# Phase 4 — Serving: prediction API and dashboard

## 28. Serve only the approved release, and fail closed

**Decision:** The API loads `fraud-detector@champion` and refuses to serve unless the version
is tagged with the approved snapshot (`phase2-rf-v1`), its `fraud_detector.joblib` matches the
snapshot's SHA-256, and its threshold equals the frozen threshold and the registry tag. On
failure the process stays up, `/health/ready` returns 503 with the reason, and predictions get
503.

**Why:** An alias can be moved by mistake, and a file can be replaced. Checksums make "the
served model *is* the evaluated model" something the service proves at startup instead of
assuming. Staying alive while not ready lets a supervisor (or a person) see *why*, which a crash
loop would hide. Loading reads no data split, so serving never touches test data.

**Alternative kept:** `serving.model_source: frozen` loads the snapshot folder directly with the
same checks, for when MLflow is not available.

## 29. Score with the exact Phase 2 object

**Decision:** The API calls `FraudDetector.predict_proba` on the unwrapped registered model:
the same pipeline object (Phase 1 preprocessing + Random Forest) that produced the Phase 2
metrics, with the threshold stored alongside it. The decision rule is `score >= threshold`,
identical to evaluation.

**Why:** No re-implementation of preprocessing in the API means no training/serving skew. Tests
assert API scores equal the frozen detector's to 1e-12.

## 30. Strict input validation; warnings for unusual-but-valid inputs

**Decision:** Pydantic models with `extra="forbid"`, `strict=True` and `allow_inf_nan=False`.
All 30 features are required. `Time` and `Amount` must be ≥ 0, and there are generous sanity
limits (|V| ≤ 1,000, Amount ≤ 1,000,000). Values outside the *training* range are scored but
return a warning.

**Why:**
- Strict mode stops silent coercions such as `"12.5"` → 12.5 or `true` → 1.0, which hide
  client bugs.
- Non-finite numbers would crash or corrupt the model.
- Hard limits catch unit errors (cents vs dollars).
- Hard-rejecting every out-of-training-range value would block legitimate rare transactions,
  and rare transactions are exactly where fraud lives. A warning tells the reviewer the score
  is an extrapolation.

## 31. Safe error handling

**Decision:** Errors are JSON `{error, detail, request_id}`:
- 422 for validation, listing only field and message (submitted values are never echoed);
- 413 for oversized batches;
- 503 when not ready;
- 500 with a generic message for anything unexpected, logged server-side with the request id.

Request logs contain method, path, status and latency, never transaction contents.

**Why:** Transaction data is sensitive and stack traces leak internals. A request id is enough
to connect a user-reported problem to the server log.

## 32. Separate liveness and readiness

`/health` says "the process answers"; `/health/ready` says "it can make correct predictions".
Load balancers and orchestrators treat these differently (restart vs. stop routing traffic).
The split costs nothing now and is what Docker/Kubernetes will need later.

## 33. Dashboard talks only to the API

**Decision:** The Streamlit app never loads the model. It calls the API for scores, model info,
recorded performance and figures. `dashboard/client.py` holds all HTTP logic.

**Why:** What the dashboard shows is exactly what the service returns, so there is one source of
truth. The client is unit-tested against the real FastAPI app, and the page is executed in tests
with `streamlit.testing.AppTest`. Example transactions come from the **training** split only.

**Performance tab** shows the frozen release's recorded results: test by default, validation
on request, labelled "used for selection". Nothing is recomputed, so the dashboard can't drift
from the documented evaluation. Business cost is shown as an explicit formula
(missed × $120 + false alarms × $5) because the costs are assumptions a stakeholder should see.

## 34. Not done in this phase (by design)

No monitoring, drift detection, automatic retraining or containerisation. These are Phase 5.
There is also no authentication: the service binds to `127.0.0.1` (local only). It would need an
API key or OAuth before being exposed on a network.

---

# Phase 4b — Local explanations (SHAP)

## 35. Explain the displayed number, in its own units

**Decision:** TreeSHAP on the frozen forest with `model_output="raw"`, using the class-1 column.
For a scikit-learn Random Forest the raw output *is* `predict_proba`: the average of the trees'
leaf class fractions. So the explanation is in probability space, the same units as the
score in the API and dashboard. No log-odds or other link function is involved. (For XGBoost
the raw output is log-odds, and this choice would be wrong.)

Every explanation is checked: `|baseline + Σ contributions − predict_proba| ≤ 1e-6` (observed
≈ 4×10⁻⁸ from SHAP's float32 internals). Otherwise the request fails instead of showing
numbers that don't add up.

## 36. Interventional TreeSHAP with a real-traffic background

**Decision:** `feature_perturbation="interventional"` with a seeded sample of 500 **training**
transactions (natural ~0.2% fraud rate), passed through `shap.maskers.Independent(...,
max_samples=500)`.

**Why not the default (`tree_path_dependent`)?** It takes the baseline from the sample
weights stored in the trees. The forest was trained with `class_weight="balanced_subsample"`,
so the weighted root is 50% fraud and the baseline is **0.5**. A chart starting at "50% fraud"
for a population where 0.2% is fraud would mislead every reader, even though it is
mathematically consistent. With interventional SHAP the baseline is the model's average score
on real training transactions (≈ 0.0022), and the contributions explain "why this transaction
differs from a typical one".

**Pitfall found and handled:** given a plain array with more than 100 rows, SHAP silently
subsamples the background to 100 rows. The baseline then no longer matches the chosen sample
(0.00053 instead of 0.00119 in a test). Passing an `Independent` masker with an explicit
`max_samples` uses exactly the chosen rows, and the explainer checks at startup that the
baseline equals the model's mean score on the background.

**Trade-offs:** cost grows with background size (500 rows ≈ 0.1–0.3 s per explanation). The
baseline depends on which rare fraud cases land in the sample, so it is fixed by seed and
reported in every response.

## 37. Contributions per transaction field, not per model feature

The model sees 31 engineered features; users submit 30 fields. Because SHAP values are
additive, a field's contribution is the exact sum over the features derived from it
(`Time` → `hour_sin` + `hour_cos`). The mapping is read from the fitted `ColumnTransformer`
rather than hard-coded, and transformers that mix several inputs are rejected, not guessed.

## 38. Honest labels

`V1`–`V28` are labelled "anonymized PCA component" with the note that their meaning was not
disclosed by the data provider. No business meaning is attached to them. `Time` is labelled
"used as hour of day (relative to the start of the recording)", which is what the model
actually uses. The UI states that explanations describe model behaviour, not causes.

## 39. Prediction behaviour is untouched

Explanations live in a separate `POST /explain` endpoint. `/predict` and `/predict/batch` are
unchanged: same code path, and a test pins their response fields. `/explain` computes its
score and decision with the same function as `/predict`. The explainer is built after the
model loads. If it fails (e.g. the training split's checksum doesn't match the frozen
snapshot), only `/explain` returns 503. A regression check against the previously running API
on 300 real transactions gave identical decisions (score difference ≤ 2×10⁻¹⁶, the known
multi-threaded rounding noise).

---

# Phase 5 — Monitoring and controlled retraining

## 40. Evidently in an isolated environment

**Decision:** Evidently 0.7.23 runs in `.venv-monitoring` (pandas 2.x) as a subprocess
(`monitoring/evidently_runner.py`). Inputs are passed as Parquet files and results come back as a
JSON summary plus HTML reports. The temporary input files are deleted afterwards.

**Why:** Evidently (including its development branch) imports a private pandas function that
pandas 3 removed, and its package metadata doesn't prevent installing it next to pandas 3, so the
failure only appears at import time. Downgrading pandas in the main environment would change the
stack the frozen champion, its preprocessing and every recorded metric were verified on. An
isolated tool environment keeps that stack byte-for-byte identical. The runner imports nothing
from this package, so it doesn't depend on the main environment. Cost: one extra setup step
(`make monitoring-env`) and about 1–2 s of subprocess start-up per run.

## 41. What is compared with what

* **Features vs training data** (20,000 seeded rows): "has the population moved away from what
  the model learned from?"
* **Scores vs validation scores:** the forest has memorised its training rows, so training scores
  are over-confident and would make every production batch look drifted. Validation rows are
  out-of-sample, as production is.
* **Expected review rate** = share of validation scores ≥ threshold. It is computed for whichever
  version is champion, and needs no test data.
* **Hour of day instead of raw Time:** raw Time only increases (seconds since the first
  transaction), so it always "drifts". The model only uses hour of day, so that's what is monitored.
* **Performance** uses the scores that were actually served, at the served threshold, and only
  when there are at least 20 labelled frauds. With fewer, recall moves 5 points per fraud.
* Data quality is checked in pandas, so it works even without Evidently. Rows the API would reject
  can only arrive through batch ingestion; they are counted, not scored, and raise a critical alert.

## 42. Alert thresholds

They live in config, so they can be tuned without code changes. Two lessons from the demo:

1. **A share-of-features rule misses single important features.** A tripled Amount drifts 1 of
   31 features (3%), far below a 20% share rule. A *watched features* list (Amount, hour of day,
   and the components with the strongest fraud signal in EDA) raises a warning on its own.
2. **The recall floor must sit below normal variation but above real degradation.** 0.65 is
   about the lower end of the champion's test recall interval (0.63–0.86). A first value of 0.60
   let a drop from 1.00 to 0.61 pass silently.

## 43. Prediction capture without changing predictions

The API writes each scored transaction to a local SQLite store (WAL mode, one connection per
write, a lock for thread safety). Any storage error is logged and swallowed, so the prediction is
still returned. Responses are unchanged, and when a client sends no `transaction_id`, an internal
one (`<request_id>-<i>`) is stored. The store holds transaction features and is treated like the
dataset: local only and git-ignored.

## 44. Permitted training data, enforced in code

* **Training:** the training split (verified against the champion's recorded hash) plus
  labelled production rows, except the newest 30%.
* **Recent holdout:** those newest 30% of labelled production rows, used to compare against the
  champion on data that reflects current traffic.
* **Validation split:** never trained on. It stays the shared yardstick, as in Phase 2.
* **Test split:** never read. `read_permitted` refuses it by file name and by checksum (taken
  from the frozen manifest, so the check itself doesn't read test data). A test spies on
  `pd.read_parquet` during retraining and asserts `test.parquet` is never opened.

Labelled production rows are de-duplicated by transaction id (latest label wins) and kept in
arrival order. The training frame's content fingerprint is recorded so a candidate can be
reproduced and audited.

## 45. Same recipe, new data

Candidates reuse the champion's own frozen configuration: preprocessing, Random Forest
hyperparameters, seed and cost-based threshold rule, read from the snapshot's `config.yaml`.
This isolates the effect of the new data. Changing the recipe (other models or hyperparameters)
is model development, not routine retraining, and should go through the Phase 2 process.

The candidate's threshold is tuned on validation with the same $120 / $5 costs. A candidate can
therefore differ from the champion in both scores and threshold (demo: 0.0600 vs 0.0499), and
the criteria compare each model at its own tuned threshold.

## 46. Promotion criteria: non-inferior on the yardstick, better on recent data

A candidate must be **no worse** on validation (PR-AUC −0.01, recall −0.05, cost +10%) and **at
least as good** on the recent holdout. Requiring it to *beat* the champion on validation would be
unreasonable: the champion was selected on validation, so its score there is optimistic. Requiring
only recent-data improvement would let a model that overfits recent fraud patterns forget old
ones. The holdout is required by default; without one, the candidate isn't eligible.

## 47. Explicit, recorded human approval; aliases as the deployment switch

* **Registration:** only eligible candidates become registry versions, as `pending` with no alias.
* **Approval:** a named approver, a written reason, and typing the version number to confirm.
  `--yes` exists for scripted use, but the approver and reason are still mandatory and recorded.
* **Audit trail:** each action is written as version tags (`approved_by`, `approved_at`,
  `approval_reason`, `superseded_by`, `rolled_back_*`), a run in the governance experiment, and
  the store's `events` table.
* **Rollback:** only to approved versions. The default target is `previous-champion`.
* **Serving:** `serving.frozen_snapshot: auto` serves the champion only if it is approved and its
  snapshot verifies, so assigning the alias by hand to an unapproved version doesn't put it into
  service.
* **Legacy release:** the original Phase 2 model is treated as approved because it went through
  the Phase 3 freeze-and-register release. No approval record was created on anyone's behalf.
* **No auto-promotion:** nothing moves an alias automatically. A dashboard button was left out
  on purpose; promotion is a CLI action with a typed confirmation.

## 48. Backward compatibility of frozen configs

Frozen snapshots carry the config they were made with. The Phase 2 snapshot predates
`paths.monitoring_dir`, and retraining reads that config, so new settings must have defaults or
older releases become unusable. A test loads a config without any of the Phase 4–5 sections.

## 49. Sandbox demo

The demo copies the frozen release into `artifacts/demo/` with its own registry, store and
snapshots, and registers it there with the same tags `track-frozen` uses (without re-scoring the
test split). It runs monitoring, retraining, promotion and rollback with a clearly labelled
*simulated* approver. Synthetic batches are resampled from the training split with noise clipped
to the training range, so "stable" really is stable, and each scenario changes one known thing.
Because the rows come from training data, performance numbers are optimistic. The demo
demonstrates mechanics, not accuracy.

## 50. Limitations and next steps

* Labels are assumed to arrive eventually and correctly. Real chargebacks arrive weeks late and
  are incomplete; monitoring reports label coverage, but delay-aware evaluation is not modelled.
* Drift statistics on a few hundred rows are noisy (an info alert flags batches below
  `min_rows`), and with 31 features some false alarms are expected.
* The API picks up a new champion on restart. A hot-reload endpoint would need authentication
  first.
* Monitoring runs on demand (CLI). Scheduling (cron, an orchestrator) and alert delivery
  (email/Slack) belong with deployment (Phase 6).
