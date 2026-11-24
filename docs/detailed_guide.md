# Detailed Guide — Credit Card Fraud Detection with MLOps

> The phase-by-phase reference for every command and component. For the overview, see the
> [README](../README.md).

A production-style machine learning project that detects fraudulent credit card
transactions. It is built in phases, the way a real ML system grows:

| Phase | Scope | Status |
|---|---|---|
| **1. Project setup & data preparation** | structure, config, logging, data loading, EDA, leakage-safe splits, preprocessing, validation, tests | ✅ done |
| **2. Model training & evaluation** | Logistic Regression / Random Forest / XGBoost, class weights, threshold tuning, model selection, one-time test evaluation, saved inference pipeline | ✅ done |
| **3. Experiment tracking & model versioning** | frozen Phase 2 release, validation→test analysis, local MLflow tracking, model registry with `champion` alias | ✅ done |
| **4. Serving** | FastAPI prediction service (single + batch), Streamlit dashboard, health checks, input validation, local SHAP explanations | ✅ done |
| **5. Monitoring & controlled retraining** | prediction capture, Evidently drift, data quality, labelled performance, alerts, monitoring dashboard, candidate retraining, approval-gated promotion and rollback | ✅ done |
| 6. Packaging & deployment | Docker, cloud | planned |

**Dataset:** [Kaggle — Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
284,807 European card transactions over two days, of which only 492 (0.17%) are fraud.
Features `V1`–`V28` are anonymised PCA components; `Time` is seconds since the first
transaction; `Amount` is the transaction value; `Class` is the label (1 = fraud).

---

## Quick start

Requires **Python 3.11** (macOS / Linux). On macOS, XGBoost also needs the OpenMP
runtime: `brew install libomp`.

```bash
# 1. Create the environment and install the project
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .

# 2. Get the data (pick ONE option, see "Getting the dataset" below)
fraud-detection download --source mirror

# 3. Phase 1 - data preparation
fraud-detection validate      # data quality checks on the raw CSV
fraud-detection eda           # EDA report + charts  -> reports/eda/
fraud-detection prepare       # clean, split, validate, fit preprocessor
fraud-detection eda --on train  # optional: EDA restricted to the training split

# 4. Phase 2 - training and evaluation
fraud-detection verify        # re-check splits: counts, schema, leakage
fraud-detection train         # train 3 models, tune thresholds, select, test once, save
fraud-detection evaluate      # reload the saved model and reproduce the test metrics

# 5. Phase 3 - freeze the release, track it in MLflow, browse experiments
fraud-detection freeze --name phase2-rf-v1        # immutable, checksummed snapshot
fraud-detection verify-frozen --name phase2-rf-v1 # files + data + metrics unchanged?
fraud-detection track-frozen --name phase2-rf-v1  # log to MLflow + register as champion
fraud-detection registry                          # list model versions and aliases
make mlflow-ui                                    # open http://127.0.0.1:5000

# 6. Phase 4 - serve the approved model (two terminals)
make api                      # API on http://127.0.0.1:8000  (interactive docs: /docs)
make dashboard                # dashboard on http://127.0.0.1:8501

# 7. Run the tests
pytest
```

Every command also has a `make` shortcut (`make setup`, `make prepare`, `make train`,
`make evaluate`, `make all`, `make freeze`, `make track-frozen`, `make mlflow-ui`,
`make api`, `make dashboard`, `make test`, `make lint`, …). Run `make help` to list them.

## Getting the dataset

The dataset is **not** included in this repository and the code never generates
fake data in its place. If `data/raw/creditcard.csv` is missing, every command
stops and prints these instructions:

* **Option A – Kaggle website:** sign in, open the dataset page, click *Download*,
  unzip, and move `creditcard.csv` to `data/raw/creditcard.csv`.
* **Option B – Kaggle API:** `pip install kaggle`, create an API token under
  *Kaggle → Settings → API*, save it to `~/.kaggle/kaggle.json` (`chmod 600`),
  then run `fraud-detection download --source kaggle`.
* **Option C – public mirror (no account):** `fraud-detection download --source mirror`
  downloads the byte-identical copy that TensorFlow hosts for its imbalanced-data
  tutorial.

After downloading, `fraud-detection validate` confirms the file has the published
shape (284,807 rows × 31 columns).

## Project structure

```
.
├── configs/config.yaml          # every tunable setting in one place
├── .streamlit/config.toml       # dashboard theme + privacy settings
├── src/fraud_detection/
│   ├── config.py                # YAML -> typed, validated dataclasses
│   ├── logging_utils.py         # console + rotating file logging
│   ├── cli.py                   # `fraud-detection` command
│   ├── data/
│   │   ├── schema.py            # expected columns and dtypes
│   │   ├── download.py          # Kaggle / mirror download + instructions
│   │   ├── loader.py            # read CSV, read/write Parquet
│   │   ├── validation.py        # raw-data and split validation checks
│   │   ├── cleaning.py          # duplicate removal, chronological sort
│   │   └── split.py             # time-based and stratified splitting
│   ├── features/
│   │   ├── transformers.py      # HourOfDayEncoder (custom sklearn transformer)
│   │   └── preprocessing.py     # ColumnTransformer factory
│   ├── data/verification.py     # pre-training checks on the saved splits
│   ├── models/
│   │   ├── factory.py           # preprocessing + classifier pipelines, class weights
│   │   ├── metrics.py           # PR-AUC, ROC-AUC, precision/recall/F1, cost, bootstrap CIs
│   │   ├── thresholds.py        # cost-based / F-beta threshold tuning
│   │   └── detector.py          # FraudDetector (pipeline + threshold), save/load
│   ├── analysis/
│   │   ├── style.py             # shared chart styling
│   │   ├── eda.py               # EDA statistics, charts, Markdown report
│   │   ├── model_report.py      # comparison tables, PR curves, confusion matrices
│   │   └── generalization.py    # validation -> test gap analysis (read-only)
│   ├── serving/
│   │   ├── model_loader.py      # load + verify the approved model (fails closed)
│   │   ├── schemas.py           # strict request/response schemas, feature validation
│   │   ├── reports.py           # read-only access to the frozen evaluation reports
│   │   └── api.py               # FastAPI app: health, model info, predictions
│   ├── explainability/
│   │   ├── shap_explainer.py    # TreeSHAP on the frozen forest, per-field contributions
│   │   └── service.py           # builds the explainer from the verified training split
│   ├── dashboard/
│   │   ├── client.py            # HTTP client for the API (used by the dashboard)
│   │   ├── app.py               # Streamlit dashboard (scoring, performance, SHAP)
│   │   └── monitoring_app.py    # Streamlit monitoring dashboard (Phase 5)
│   ├── monitoring/
│   │   ├── store.py             # SQLite store: predictions, labels, runs, alerts, events
│   │   ├── evidently_runner.py  # runs INSIDE .venv-monitoring (Evidently, pandas 2)
│   │   ├── drift.py             # bridge to the isolated Evidently environment
│   │   ├── quality.py / performance.py / alerts.py / reference.py
│   │   └── simulate.py          # synthetic batches (from the training split)
│   ├── retraining/
│   │   ├── data.py              # permitted training data; test split refused
│   │   ├── candidate.py         # reproducible candidate + read-only snapshot
│   │   ├── compare.py           # promotion criteria
│   │   ├── approval.py          # approval states (shared with serving)
│   │   └── governance.py        # register, promote, rollback, audit trail
│   ├── tracking/
│   │   ├── mlflow_tracking.py   # log runs, datasets, artifacts; register versions
│   │   └── pyfunc_model.py      # MLflow model wrapper that keeps the threshold
│   └── pipelines/
│       ├── eda.py               # load -> analyse -> report
│       ├── prepare.py           # load -> validate -> clean -> split -> fit
│       ├── train.py             # verify -> train -> tune -> select -> test once -> save
│       ├── evaluate.py          # reload saved model -> reproduce test metrics
│       ├── freeze.py            # immutable snapshots + verification
│       ├── track.py             # MLflow: track frozen snapshot / new runs, registry
│       ├── monitor.py           # ingest batches, labels, monitoring runs
│       └── demo.py              # end-to-end lifecycle in an isolated sandbox
├── tests/                       # pytest suite (unit + integration)
├── docs/design_decisions.md     # why things are built the way they are
├── docs/phase2_results.md       # permanent record of the frozen Phase 2 model
├── data/{raw,interim,processed} # git-ignored
├── artifacts/                   # preprocessor, prep report, model/ (git-ignored)
│   └── frozen/phase2-rf-v1/     # read-only released model + reports + MANIFEST.json
├── mlflow/                      # local MLflow database + artifacts (git-ignored)
├── monitoring/                  # monitoring store + drift reports (git-ignored)
├── .venv-monitoring/            # isolated Evidently environment (git-ignored)
└── reports/{eda,training}/      # generated reports (git-ignored)
```

## What each component does (in plain English)

* **Config (`configs/config.yaml`, `config.py`)** — All settings (paths, split sizes,
  scaler choice…) live in one YAML file. On load they are checked, so a typo such as
  split sizes that add up to 1.1 fails immediately with a clear message.
* **Logging (`logging_utils.py`)** — Every step writes timestamped messages to the
  terminal and to `logs/fraud_detection.log`, so you can see what happened after the fact.
* **Download & loading (`download.py`, `loader.py`)** — Fetches the CSV, reads it with
  fixed column types, and refuses to continue (with instructions) if it is missing.
* **Validation (`validation.py`)** — Automated data-quality checks. *Errors* stop the
  pipeline (missing columns, missing values, labels other than 0/1, negative amounts).
  *Warnings* are reported but not fatal (duplicates, unexpected row count). A second set
  of checks verifies the splits: no shared rows, correct time order, enough fraud in each.
* **Cleaning (`cleaning.py`)** — Removes exact duplicate rows and sorts by time.
* **Splitting (`split.py`)** — Oldest 70% of transactions → train, next 15% →
  validation, newest 15% → test. This copies real life: you train on the past and
  predict the future.
* **Preprocessing (`preprocessing.py`, `transformers.py`)** — A reusable scikit-learn
  object that scales features, log-transforms `Amount`, and converts `Time` into
  hour-of-day features. It learns its statistics from the training data only.
* **EDA (`analysis/eda.py`)** — Produces a JSON summary, five charts, and a Markdown
  report describing the data.
* **Verification (`verification.py`)** — Before training, re-reads the saved splits and
  checks that nothing changed since Phase 1 (same row and fraud counts), that there is no
  leakage, and that the preprocessing code still matches the saved Phase 1 preprocessor.
* **Model factory (`factory.py`)** — Builds each model as one pipeline:
  *Phase 1 preprocessing → classifier*. Fitting it on the training split fits both
  parts on training data only. Rare fraud cases get more weight during training
  (class weights), so no rows are invented or thrown away.
* **Metrics (`metrics.py`)** — Fraud-appropriate scores, plus 95% confidence intervals
  (bootstrap), because ~50 fraud cases per split make single numbers shaky.
* **Threshold tuning (`thresholds.py`)** — A model outputs a fraud score from 0 to 1. The
  threshold decides above which score we raise an alert. It is chosen on validation data
  to minimise the total cost of missed fraud and false alarms.
* **Detector (`detector.py`)** — The thing you deploy: the fitted pipeline *and* its tuned
  threshold, saved together so they can never get out of sync.
* **Pipelines & CLI** — Wire the pieces above into single commands.

## Outputs

| Command | Creates |
|---|---|
| `fraud-detection eda` | `reports/eda/eda_report.md`, `eda_summary.json`, `figures/*.png` |
| `fraud-detection prepare` | `data/interim/creditcard_clean.parquet`, `data/processed/{train,validation,test}.parquet`, `artifacts/preprocessor.joblib`, `artifacts/data_preparation_report.json` |
| `fraud-detection train` | `artifacts/model/fraud_detector.joblib`, `artifacts/model/model_metadata.json`, `reports/training/evaluation_report.md`, `model_comparison.csv`, `threshold_tradeoff_validation.csv`, `figures/*.png`; plus an MLflow run |
| `fraud-detection gap-report` | `reports/training/validation_vs_test.{md,json}` |
| `fraud-detection freeze --name X` | `artifacts/frozen/X/` (model, reports, config, data report, `MANIFEST.json`) |
| `fraud-detection track-frozen --name X` | MLflow run `X (frozen)` + registered model version (alias `champion`) |

## How Phase 2 works

```
train split ──► fit 3 pipelines (preprocessing + model, class-weighted)
                         │
validation split ──► score ──► tune each model's threshold (min. expected cost)
                         │
                         └──► pick best model by validation PR-AUC
                                         │
test split ──────────────────────► score ONCE with the chosen model + threshold
                                         │
                                         └──► save detector + metadata + reports
```

* **Imbalance:** `class_weight="balanced"` (Logistic Regression),
  `"balanced_subsample"` (Random Forest), `scale_pos_weight = negatives/positives`
  computed from training labels (XGBoost). Validation and test data are never resampled
  or reweighted.
* **Threshold:** minimises `120 × missed frauds + 5 × false alarms` on validation. The
  dollar values are illustrative assumptions in `configs/config.yaml` (`threshold:`); set
  `method: f_beta` to maximise F2 instead.
* **Selection:** validation PR-AUC (it does not depend on the threshold and focuses on the
  fraud class). The test split is used once, for the winning model only. A test checks this.
* **Reproducibility:** one seed (`training.random_state`) drives every model. Two runs from a
  clean state give identical results, and `fraud-detection evaluate` checks that the saved
  artifact reproduces the recorded test metrics exactly.
* **Metadata** (`model_metadata.json`): model and hyperparameters, threshold and how it was
  chosen, every candidate's validation scores, test metrics with confidence intervals,
  SHA-256 hashes of the split files, and library versions.

Using the saved model:

```python
import pandas as pd
from fraud_detection.models.detector import load_detector

detector = load_detector("artifacts/model")
transactions = pd.read_parquet("data/processed/test.parquet")   # raw columns: Time, V1..V28, Amount
scores = detector.predict_proba(transactions)    # fraud probability per row
alerts = detector.predict(transactions)          # 1 = flag, using the tuned threshold
```

## Phase 1 results on the real dataset

| | Value |
|---|---|
| Raw rows / fraud | 284,807 / 492 (0.173%) — 1 fraud per ~578 transactions |
| Missing values | 0 |
| Exact duplicates removed | 1,081 (19 of them fraud) → 283,726 rows |
| Train (oldest 70%) | 198,607 rows, 366 fraud (0.184%) |
| Validation (next 15%) | 42,558 rows, 55 fraud (0.129%) |
| Test (newest 15%) | 42,561 rows, 52 fraud (0.122%) |
| Preprocessor output | 31 features (V1–V28, Amount, hour_sin, hour_cos) |
| Train negative:positive ratio | 541.6 : 1 (input for class weighting in Phase 2) |

Strongest single-feature signals (correlation with fraud): V17, V14, V12, V10, V16.
Fraud is relatively more frequent during the low-volume night hours.

## Phase 2 results on the real dataset

**Validation split** (used to tune thresholds and choose the model):

| Model | PR-AUC (95% CI) | ROC-AUC | Threshold | Precision | Recall | Caught / missed | False alarms |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.809 (0.711–0.893) | 0.981 | 0.996 | 0.457 | 0.873 | 48 / 7 | 57 |
| **Random Forest** ✓ | **0.874** (0.782–0.950) | 0.979 | 0.050 | 0.538 | 0.891 | 49 / 6 | 42 |
| XGBoost | 0.840 (0.744–0.924) | 0.988 | 0.193 | 0.378 | 0.873 | 48 / 7 | 79 |

**Held-out test split** (Random Forest, threshold 0.050, evaluated once):

| PR-AUC | ROC-AUC | Precision | Recall | Fraud caught | Missed | False alarms |
|---|---|---|---|---:|---:|---:|
| 0.778 (0.668–0.876) | 0.944 | 0.429 | 0.750 | 39 of 52 | 13 | 52 of 42,509 |

With the assumed costs, the model reduces the cost of fraud on the test period by
**71%** compared with having no model ($1,820 vs $6,240). The full write-up, including
the charts and caveats, is in `reports/training/evaluation_report.md`.

**Validation → test drop.** Every metric is lower on test (PR-AUC −0.096, recall 89% → 75%),
but none of the differences is statistically significant at 5% with ~50 frauds per split
(PR-AUC difference 95% CI −0.231 to +0.039; recall Fisher p = 0.077). The test numbers are
the honest estimate; the threshold was not re-tuned. Full analysis:
[`docs/phase2_results.md`](phase2_results.md).

## Phase 3: experiment tracking and model versioning (MLflow)

Everything runs locally and for free: MLflow stores runs and the model registry in a
SQLite file (`mlflow/mlflow.db`) and files in `mlflow/artifacts/`. No server or account
is needed. Disable tracking with `train --no-mlflow` or `mlflow.log_training_runs: false`.

### What is tracked

| In MLflow | Content |
|---|---|
| **Parameters** | seed, split + preprocessing + cleaning config, threshold method and costs, imbalance strategy, model hyperparameters, `data_version` |
| **Metrics** | `threshold`, `val_*` and `test_*` (PR-AUC, ROC-AUC, precision, recall, F1, TP/FP/FN/TN, cost, cost reduction) and 95% CI bounds |
| **Datasets** | train / validation / test with MLflow digests; `data_version` = hash of the three split files |
| **Child runs** | one per candidate model (Logistic Regression, Random Forest, XGBoost) with its validation metrics |
| **Artifacts** | evaluation report, comparison tables, figures, validation-vs-test analysis, model metadata, data-prep report, config, frozen manifest |
| **Model** | `fraud_detector` (pyfunc): pipeline + tuned threshold; `predict()` returns `fraud_score`, `is_fraud`, `threshold` |

### Model registry

| Version | Source | Alias | Meaning |
|---|---|---|---|
| `fraud-detector` **v1** | frozen snapshot `phase2-rf-v1` | **`champion`** | the Phase 2 Random Forest, threshold 0.0499 |

* `fraud-detection train` logs a new run but does **not** register it.
  `train --register` adds a new version but **never** moves `champion`.
* `track-frozen` is idempotent: running it again reuses the existing run.
* The Phase 2 results are protected three ways: the read-only frozen snapshot, its
  checksums (`verify-frozen`), and the registry alias. Re-running training with the same
  config reproduced identical metrics (logged as a separate `source=train` run).

Load the champion anywhere:

```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:////ABSOLUTE/PATH/TO/project/mlflow/mlflow.db")
model = mlflow.pyfunc.load_model("models:/fraud-detector@champion")
model.predict(pd.read_parquet("data/processed/test.parquet").drop(columns=["Class"]))
```

### Viewing and comparing experiments in the MLflow UI

```bash
source .venv/bin/activate
make mlflow-ui                 # or, from the project root:
# mlflow ui --backend-store-uri "sqlite:///$PWD/mlflow/mlflow.db" --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> and stop the UI with `Ctrl+C`. (If port 5000 is busy, e.g. macOS
AirPlay Receiver, use `make mlflow-ui MLFLOW_PORT=5001`.)

1. **Experiments → `credit-card-fraud-detection`.** Each training run is a row: the
   frozen `phase2-rf-v1 (frozen)` and any `train-YYYYMMDD-HHMMSS` runs. Click the **+** next
   to a run to expand its child runs (one per model).
2. **Choose columns** to show `val_pr_auc`, `test_pr_auc`, `threshold`, `val_recall`,
   `test_recall` and the `data_version` parameter. Click a header to sort.
3. **Compare models:** tick the three child runs → **Compare**. You get parameters
   side by side (differences highlighted), metrics in a table, and parallel-coordinates /
   scatter plots (e.g. `val_precision` vs `val_recall`).
4. **Compare training runs:** tick two parent runs → **Compare** to see whether a config
   change moved any metric. Filter with the search box, e.g.
   `tags.source = 'train'` or `params.data_version = 'bd686364a12d'`.
5. **Inside a run:** *Overview* (parameters, metrics, tags, description, datasets used:
   train/validation/test with digests), *Artifacts* (open `reports/evaluation_report.md`,
   `reports/validation_vs_test.md`, figures, `config/config.yaml`, the logged model).
6. **Models → `fraud-detector`:** versions, the `champion` alias, tags (`threshold`,
   `data_version`, `test_pr_auc`), and a link back to the source run. Moving `champion` to
   another version is a deliberate action here (or via `MlflowClient`), never automatic.

## Phase 4: prediction API and dashboard

### Launch

You need two terminals, both started in the project folder with the environment active
(`source .venv/bin/activate`):

```bash
# Terminal 1 - prediction API
make api                # = fraud-detection serve  ->  http://127.0.0.1:8000
                        #   interactive API docs   ->  http://127.0.0.1:8000/docs

# Terminal 2 - dashboard (talks to the API)
make dashboard          # = fraud-detection dashboard  ->  http://127.0.0.1:8501
```

Stop either with `Ctrl+C`. Other ports: `fraud-detection serve --port 8001` and
`fraud-detection dashboard --port 8502 --api-url http://127.0.0.1:8001`.

### Which model is served

At startup the API loads **`models:/fraud-detector@champion`** from the local MLflow registry
and verifies, before accepting traffic, that:

1. the champion version is tagged as the approved frozen release (`phase2-rf-v1`);
2. the SHA-256 of its `fraud_detector.joblib` (preprocessing + Random Forest + threshold)
   equals the checksum recorded in the frozen snapshot's `MANIFEST.json`;
3. its threshold equals the frozen threshold (0.0499) and the registry's `threshold` tag.

If any check fails, the API still starts but **fails closed**: `/health/ready` returns 503 with
the reason and predictions are refused. Set `serving.model_source: frozen` to load the frozen
snapshot directly (same checks, no MLflow needed). The service never trains, re-tunes or
reads the test split; the only data it reads is the training split, to warn about unusual values.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness: the process is up (always 200) |
| GET | `/health/ready` | readiness: approved model loaded and verified (200) or not (503 + reason) |
| GET | `/model` | model name, registry version, alias, threshold, run id, data version, checksum |
| GET | `/model/performance` | recorded evaluation results of the frozen release |
| GET | `/model/performance/figures/{name}` | evaluation charts (PNG) |
| POST | `/predict` | score one transaction |
| POST | `/predict/batch` | score up to 1,000 transactions (`serving.max_batch_size`) |
| POST | `/explain?top_k=5` | score one transaction **and** explain it with SHAP (see below) |

Example:

(The values below resemble a fraud pattern; they are illustrative, not a real transaction.)

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "transaction_id": "demo-1", "Time": 406, "Amount": 0.0,
  "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 4.0, "V5": -0.52, "V6": -1.43, "V7": -2.54,
  "V8": 1.39, "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.90, "V13": -0.60,
  "V14": -4.29, "V15": 0.39, "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42,
  "V20": 0.13, "V21": 0.52, "V22": -0.04, "V23": -0.47, "V24": 0.32, "V25": 0.04,
  "V26": 0.18, "V27": 0.26, "V28": -0.14}'
```

```json
{"transaction_id": "demo-1", "fraud_score": 0.9464, "is_fraud": true, "decision": "REVIEW",
 "warnings": [], "model": {"name": "random_forest", "version": "1", "source": "registry",
 "threshold": 0.0499}}
```

Scores and thresholds are shown rounded here; the API returns full precision.
`decision` is **REVIEW** when `fraud_score >= threshold`, otherwise **APPROVE**.

### Input validation and error handling

| Situation | Response |
|---|---|
| Missing feature, unknown field, string / boolean / null value, `NaN` / `Infinity` | 422 `validation_error` with one `{field, message}` per problem |
| `Amount` < 0, `Time` < 0, absurd values (e.g. \|V\| > 1,000, Amount > 1,000,000) | 422 `validation_error` |
| Invalid `transaction_id` (max 64 chars: letters, digits, `_ . : -`) | 422 `validation_error` |
| Malformed JSON or empty batch | 422 `validation_error` |
| Batch larger than 1,000 | 413 `batch_too_large` |
| Model not loaded / failed verification | 503 `model_unavailable` |
| Unexpected server error | 500 `internal_error`, generic message only (details in server log) |
| Value valid but outside the range seen in training | 200, scored, with a `warnings` entry |

Error bodies never echo submitted values or stack traces. Every response has an
`X-Request-ID` header (send your own to correlate), and the server logs method, path, status
and latency, never transaction contents.

### Dashboard

| Tab | What it shows |
|---|---|
| **Score a transaction** | form for Amount, Time and V1–V28 (load a legitimate or fraud example from the *training* data), fraud score, REVIEW/APPROVE decision, score bar with the threshold, model version |
| **Score a transaction → Why this score?** | SHAP explanation: baseline, top features pushing toward / away from fraud, a diverging bar chart, and an arithmetic check that baseline + contributions = score |
| **Batch scoring** | upload a CSV (`Time, V1..V28, Amount`, optional `transaction_id`; `Class` ignored; up to 50,000 rows), sends it to the API in batches, summary, sortable results, CSV download, and "explain a transaction from this file" |
| **Model performance** | recorded test (or validation) results: PR-AUC, precision, recall, F1, ROC-AUC with 95% CIs, confusion matrix, business cost with vs without the model, validation→test comparison, model comparison, evaluation charts |
| **Model details** | registry version, alias, threshold, run id, data version, detector checksum |

The sidebar always shows the API status, model, registry version and threshold. All numbers
on the performance tab come from the frozen release's existing reports; nothing is recomputed.

### Explanations (SHAP)

For any single transaction, the dashboard and `POST /explain` show which fields pushed the
fraud score **up** (toward fraud) and **down** (away from fraud):

```
fraud score = baseline + Σ contributions of Amount, Time, V1 … V28
```

* **What is explained:** the exact number `/predict` returns, the Random Forest's probability of
  fraud. A forest's probability is the average of its trees' outputs, so SHAP works directly in
  probability space (no log-odds conversion). Every response reports
  `additivity_error = |baseline + Σ contributions − fraud_score|`, about 1e-8 in practice;
  anything above 1e-6 is refused.
* **Baseline:** the model's average score over **500 randomly sampled training transactions**
  (seed 42), about **0.0022**, i.e. "a typical transaction". SHAP's default mode would report 0.5
  here, because the model was trained with balanced class weights. That number describes the
  50/50 world the trees were trained in, not real traffic, so we use *interventional* TreeSHAP
  with this background instead. The training split is checked against the frozen snapshot's
  SHA-256 before use. Validation and test data are never used.
* **Per field:** the model sees 31 preprocessed features. Each transaction field's contribution
  is the sum of the features derived from it (`Time` → `hour_sin` + `hour_cos`;
  `Amount` → scaled `log1p(Amount)`; `Vn` → scaled `Vn`). SHAP values are additive, so this is exact.
* **Labels:** `V1`–`V28` are shown as *"anonymized PCA component"*. The data provider did not
  disclose their meaning, so no business meaning is invented. `Time` is labelled as "used as
  hour of day" because that is all the model sees.
* **Reading it:** contributions describe how the model used these values for this
  transaction. They are not proof of fraud and not cause-and-effect.

Example (`/explain?top_k=3` for a fraud example from the training data):

```
fraud_score 0.9497   baseline 0.0022   additivity_error 4.2e-08
toward fraud : V12 (anonymized PCA component) +0.181, V14 (…) +0.178, V17 (…) +0.104
away         : V20 (anonymized PCA component) −0.0054, V26 (…) −0.0045, V28 (…) −0.0022
other 24 features combined: +0.497          0.0022 + Σ = 0.9497 ✓
```

An explanation takes about 0.1–0.3 s. `/predict` and `/predict/batch` are unchanged. If
explanations can't be built (e.g. the training split was altered), `/explain` returns 503
`explanations_unavailable`, predictions keep working, and `/model` reports why.

## Phase 5: monitoring and controlled retraining

### Setup (once)

Evidently 0.7 does not work with pandas 3, which the model and API were built and verified on.
Rather than downgrade the model's environment, Evidently runs in a **separate virtual
environment** and is called as a subprocess. The main environment never imports it.

```bash
make monitoring-env     # creates .venv-monitoring from requirements-monitoring.txt
```

### The workflow

```
API / batch file ──► monitoring store (features, served score, decision, model version)
                          │                          ▲
                          │                    add-labels (confirmed outcomes, later)
                          ▼
  fraud-detection monitor ──► data quality · feature drift · score drift · review rate
                          │     · performance on labelled rows ──► alerts
                          │     HTML reports + summary + MLflow run (monitoring experiment)
                          ▼
  fraud-detection retrain-candidate   (permitted data only; never the test split)
                          │  same recipe as the champion, threshold tuned on validation
                          ▼
  compare with champion ──► eligible? ──no──► logged as rejected, not registered
                          │ yes
                          ▼
  registered as a new version, approval_status = pending (no alias)
                          │
  fraud-detection promote --version N --approved-by "Name" --reason "..."
                          │  shows the comparison, asks you to type N to confirm
                          ▼
  champion → vN, previous-champion → old version   (restart the API to serve it)
                          │
  fraud-detection rollback --approved-by "Name" --reason "..."  (if needed)
```

**1. Capturing traffic.** With `monitoring.capture_predictions: true`, `/predict` and
`/predict/batch` store each scored transaction (30 features, served score, decision, model
version, request id) in `monitoring/monitoring.db`. A storage failure never fails a prediction.
Pass your own `transaction_id` if you want to attach outcomes later. Batch files can be scored
and stored with `fraud-detection ingest-batch --file f.csv`: rows are validated exactly like the
API, and rejected rows are counted. Confirmed outcomes are attached with
`fraud-detection add-labels --file labels.csv` (`transaction_id,label`).

**2. What `fraud-detection monitor` checks** (per batch; default: latest):

| Check | Reference | How | Alert |
|---|---|---|---|
| Feature drift | 20,000 seeded **training** transactions | Evidently `DataDriftPreset` on hour of day, Amount, V1–V28 | warning ≥ 20% of features drifted, critical ≥ 40%; warning if a key feature (Amount, hour, V14, V17, V12, V10) drifts |
| Score drift | champion's scores on the **validation** split | Evidently `ValueDrift` on `fraud_score` | warning if detected |
| Review rate | share of validation scores ≥ threshold | ratio | warning if > 3× or < 1/3 of expected |
| Data quality | training min/max | pandas | critical if rows had to be rejected; warning for > 1% out of range or > 1% duplicates |
| Performance | — | metrics at the served threshold, using the served scores | critical if recall < 0.65; warning if precision < 0.25 or PR-AUC < 0.60 (only with ≥ 20 labelled frauds) |

Raw `Time` isn't monitored: it only ever increases, so it would always "drift". The model uses
the hour of day, so that's what is monitored. Scores on training rows would be over-confident, so
the score reference is the validation split. Thresholds are in `configs/config.yaml` →
`monitoring.alerts`. Each run writes `monitoring/reports/<run_id>/` (`data_drift.html`,
`score_drift.html`, `summary.json`), stores the summary and alerts, and logs an MLflow run in
**`credit-card-fraud-monitoring`**.

**3. Retraining (`fraud-detection retrain-candidate`).**
* **Training data:** the original training split (checked byte-for-byte against the champion's)
  plus labelled production transactions, **except the newest 30%**, which are held back as a
  *recent holdout*.
* **Never trained on:** the validation split (the fixed yardstick) and the recent holdout.
  **The test split is never read**: every file passes through a guard that refuses it by name
  and by checksum, and a test checks that `test.parquet` is never opened.
* **Recipe:** the champion's own frozen config (preprocessing, Random Forest hyperparameters,
  cost-based threshold rule). Only the data changes. Seeded, with a fingerprint of the exact
  training rows: re-running gives an identical model.
* **Output:** a read-only candidate snapshot `artifacts/frozen/candidate-<id>/` with
  `reports/comparison.md`.

**4. Promotion criteria** (`retraining.criteria`, all must pass):

| Criterion | Default |
|---|---|
| Validation PR-AUC | ≥ champion − 0.01 |
| Validation recall (each at its own tuned threshold) | ≥ champion − 0.05 |
| Validation expected cost | ≤ champion + 10% |
| Recent-holdout PR-AUC | ≥ champion (holdout with ≥ 20 frauds required) |

**5. Approval, promotion and rollback.**
* **Registration:** only eligible candidates are registered, as `approval_status = pending`
  with no alias. Rejected ones are logged as MLflow runs only.
* **`promote`:** requires `--approved-by` and `--reason`, shows the comparison, and asks you to
  **type the version number** (`--yes` skips only the prompt). It then moves `champion` to the
  new version and `previous-champion` to the old one.
* **`rollback`:** only to an approved version, with the same requirements.
* **Audit trail:** every action is recorded as model-version tags, a run in the MLflow
  experiment **`credit-card-fraud-governance`**, and the store's `events` table.
* **Serving:** the API uses `serving.frozen_snapshot: auto`. It serves the champion only if it
  is approved and its snapshot's checksums match, so a pending version given the alias by hand
  is refused. The original Phase 2 model (v1) counts as an *approved legacy release*: it was
  released through the Phase 3 freeze-and-register process, and no approval record was invented
  for it. **Restart the API** after a promotion or rollback.

### Monitoring dashboard

```bash
make monitoring-dashboard   # http://127.0.0.1:8502  (your real monitoring store)
make demo-dashboard         # the sandbox demo's data
```

| Tab | Shows |
|---|---|
| Over time | drift share, review rate vs expected, recall / precision / PR-AUC, data quality, with alert limits |
| Alerts | every alert, filterable by severity |
| Drift report | per-feature drift statistic vs threshold, quality and performance details, embedded Evidently reports |
| Models & approvals | registered versions with approval state and aliases, pending candidates with their comparison report, governance log |

The dashboard is read-only. Approvals need the CLI, so a click can't promote a model.

### Demo with synthetic batches (sandbox)

```bash
make monitoring-demo        # ~80 s; everything under artifacts/demo
make demo-dashboard         # look at the result
```

The demo copies the frozen Phase 2 release into `artifacts/demo/` with **its own** MLflow
registry and monitoring store. Your real champion and MLflow history are not touched. Batches
are generated from the **training split** (resampled, small noise clipped to the training range,
moved to later days):

| Day | Scenario | Champion | Status | Recall | Alerts |
|---|---|---|---|---|---|
| 1 | stable | v1 | ok | 1.00 | — |
| 2 | amounts × 2.5 | v1 | warning | 1.00 | key-feature drift (Amount) |
| 3 | 8 components shifted 1 SD | v1 | warning | 0.95 | 27% of features drifted, out of range |
| 4 | missing / out-of-range / duplicate rows | v1 | critical | 0.97 | 408 rows rejected, out of range, duplicates |
| 5 | concept drift (fraud looks legitimate) | v1 | critical | 0.61 | recall |
| 6 | concept drift | v1 | critical | 0.65 | recall |
| — | retrain → candidate eligible (holdout PR-AUC 0.874 vs 0.792; validation PR-AUC 0.874 vs 0.874; cost $890 vs $930) → **v2 pending** → promoted with a *simulated* sandbox approval | | | | |
| 7 | concept drift | **v2** | ok | 0.75 | — |
| — | rollback drill (simulated approval) → champion back to **v1** | | | | |

Because the batches are built from training rows, absolute recall and precision are optimistic.
The demo shows that each kind of change triggers the right alert and that the governance steps
work, not real-world accuracy. On day 3, 45% of rows fall outside the training range because V1
has a hard upper limit in the data (max 2.46, standard deviation 1.88), so a one-standard-
deviation shift pushes many rows past it.

### Using it on real traffic

```bash
make api                                                      # captures predictions
fraud-detection add-labels --file confirmed_outcomes.csv      # when outcomes are known
fraud-detection monitor                                       # latest batch (api-YYYY-MM-DD)
fraud-detection retrain-candidate                             # when monitoring says so
fraud-detection governance                                    # versions, approvals, events
fraud-detection promote --version 2 --approved-by "Your Name" --reason "..."
fraud-detection rollback --approved-by "Your Name" --reason "..."
```

## Testing

```bash
pytest                          # all 238 tests (~80 s)
pytest -m "not integration"     # skip tests that need the real CSV
pytest --cov=fraud_detection    # with coverage
```

* **Unit tests** use a small synthetic table with the same schema, so they run anywhere
  (e.g. in CI) without the 150 MB dataset. It is used *only* inside tests.
* **Integration tests** (`tests/test_integration_real_data.py`) run against the real file
  and check published facts (284,807 rows, 492 frauds, 1,081 duplicates). They are skipped
  automatically if the dataset has not been downloaded.
* **Phase 2 tests** cover: class-weight settings, same seed → identical scores, threshold
  tuning against brute-force search, metrics against hand-computed values, detection of
  tampered or leaking splits, save/load of the detector, the test split being scored
  exactly once, training never modifying the split files, and `evaluate` catching a
  metrics mismatch.
* **Phase 3 tests** use a throwaway MLflow store per test project and cover: frozen
  snapshots are complete, read-only, never overwritten, and survive retraining/cleaning;
  tampered files or changed data are detected; the validation→test analysis; the MLflow run
  contains the expected params, metrics, datasets, child runs and artifacts; the registered
  champion makes the same decisions as the frozen model; tracking is idempotent; new
  training runs never move the champion; logging is refused if the data changed.
* **Phase 4 tests** run the API in-process against a throwaway registered model and cover:
  health/readiness, scores identical to the frozen model, batch order and limits, 18 kinds of
  invalid input, non-echoing error bodies, request IDs, out-of-range warnings, performance
  numbers equal to the frozen reports, figure whitelisting, fail-closed startup (unapproved
  version, checksum mismatch, missing registry), hidden 500 errors, the `frozen` source, and
  that serving never alters the snapshot.
* **Explanation tests** check: baseline + contributions = displayed score; the baseline equals
  the mean score of the background (and is *not* the class-weighted 0.5); the class-1
  probability is explained; Time = hour_sin + hour_cos and one-to-one fields map exactly; every
  field appears once; results are deterministic; explaining never changes predictions;
  inconsistent SHAP values are refused; non-tree models are rejected; the background must be
  the frozen training split; `/explain` matches `/predict` exactly; `/predict`'s response is
  unchanged; anonymized labels; and the dashboard renders the explanation.
* **Phase 5 tests** (each in its own sandbox registry):
  - *Store and alerts:* store round-trips, corrected labels and 40,000-label inserts; every alert
    rule, plus a quiet stable batch.
  - *Data:* the simulator's scenarios are deterministic; raw Time is never monitored.
  - *End to end:* ingest → labels → Evidently → alerts → reports → MLflow; a clear error when the
    Evidently environment is missing.
  - *Capture:* API prediction capture, and that a capture failure never breaks a prediction.
  - *Retraining:* the test split is refused by name and by checksum, and `test.parquet` is never
    opened during retraining; candidates are reproducible; snapshots are read-only; the criteria
    logic; rejected candidates are not registered.
  - *Approval:* promotion needs an approver, a reason and confirmation; an unapproved champion is
    refused by serving; promote → serve v2 → rollback → serve v1; rollback refuses unapproved
    targets and tampered snapshots.
  - *Compatibility:* old frozen configs still load.
  - *Dashboard:* the monitoring dashboard renders with and without data. The dashboard's client is tested against the real
  API app, and the Streamlit page is executed with Streamlit's `AppTest` (all tabs render,
  scoring works, and the page degrades gracefully when the API is down).

## Design decisions

See [`docs/design_decisions.md`](design_decisions.md) for the reasoning behind the
split strategy, duplicate handling, feature treatment, imbalance strategy and more.
