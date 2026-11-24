# Convenience commands. Run `make help` to list them.
PYTHON ?= python3.11
VENV   := .venv
BIN    := $(VENV)/bin

.PHONY: help setup bootstrap docker-build docker-up docker-down download download-mirror validate eda eda-train prepare verify train evaluate all gap-report freeze verify-frozen track-frozen registry mlflow-ui api dashboard monitoring-env monitoring-demo monitoring-dashboard demo-dashboard monitor retrain-candidate governance test test-unit lint format clean

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

setup:  ## Create the virtual environment and install the project
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

bootstrap:  ## Fresh clone -> download data, prepare, train, freeze + register champion
	$(BIN)/fraud-detection bootstrap

download:  ## Download the dataset with the Kaggle API (needs ~/.kaggle/kaggle.json)
	$(BIN)/fraud-detection download --source kaggle

download-mirror:  ## Download the identical public copy of the dataset (no account)
	$(BIN)/fraud-detection download --source mirror

validate:  ## Validate the raw dataset
	$(BIN)/fraud-detection validate

eda:  ## Generate EDA report + figures in reports/eda
	$(BIN)/fraud-detection eda

eda-train:  ## EDA on the training split only (run `make prepare` first)
	$(BIN)/fraud-detection eda --on train

prepare:  ## Clean, split, validate and fit the preprocessor
	$(BIN)/fraud-detection prepare

verify:  ## Re-check prepared splits before training (counts, schema, leakage)
	$(BIN)/fraud-detection verify

train:  ## Train + tune + select models, test the winner once, save it
	$(BIN)/fraud-detection train

evaluate:  ## Re-score the saved model on the test split (reproducibility check)
	$(BIN)/fraud-detection evaluate

all: validate eda prepare train evaluate  ## Run Phase 1 + Phase 2 end to end

# --- Phase 3: frozen releases + MLflow ------------------------------------------------
SNAPSHOT ?= phase2-rf-v1
MLFLOW_PORT ?= 5000

gap-report:  ## Validation vs test analysis for the working model (read-only)
	$(BIN)/fraud-detection gap-report

freeze:  ## Freeze the working model as an immutable snapshot (SNAPSHOT=name)
	$(BIN)/fraud-detection freeze --name $(SNAPSHOT)

verify-frozen:  ## Check a snapshot's checksums, data and test metrics (SNAPSHOT=name)
	$(BIN)/fraud-detection verify-frozen --name $(SNAPSHOT)

track-frozen:  ## Log + register a snapshot in MLflow as the champion (SNAPSHOT=name)
	$(BIN)/fraud-detection track-frozen --name $(SNAPSHOT)

registry:  ## List registered model versions and aliases
	$(BIN)/fraud-detection registry

mlflow-ui:  ## Open the local MLflow UI at http://127.0.0.1:5000
	$(BIN)/mlflow ui --backend-store-uri "sqlite:///$(CURDIR)/mlflow/mlflow.db" \
		--host 127.0.0.1 --port $(MLFLOW_PORT)

# --- Phase 4: serving ---------------------------------------------------------------------
api:  ## Start the prediction API at http://127.0.0.1:8000 (docs at /docs)
	$(BIN)/fraud-detection serve

dashboard:  ## Start the Streamlit dashboard at http://127.0.0.1:8501 (needs `make api`)
	$(BIN)/fraud-detection dashboard

# --- Phase 5: monitoring and controlled retraining ----------------------------------------
monitoring-env:  ## Create the isolated Evidently environment (.venv-monitoring)
	$(PYTHON) -m venv .venv-monitoring
	.venv-monitoring/bin/pip install --upgrade pip
	.venv-monitoring/bin/pip install -r requirements-monitoring.txt

monitoring-demo:  ## Full monitor -> retrain -> approve -> rollback lifecycle in a sandbox
	$(BIN)/fraud-detection monitoring-demo --workdir artifacts/demo

monitoring-dashboard:  ## Monitoring dashboard at http://127.0.0.1:8502 (real store)
	$(BIN)/fraud-detection monitoring-dashboard

demo-dashboard:  ## Monitoring dashboard for the sandbox demo
	$(BIN)/fraud-detection --config artifacts/demo/config.yaml monitoring-dashboard

monitor:  ## Monitor the latest stored batch (drift, quality, performance, alerts)
	$(BIN)/fraud-detection monitor

retrain-candidate:  ## Train a candidate on permitted data and compare with the champion
	$(BIN)/fraud-detection retrain-candidate

governance:  ## Registry versions, approval states, aliases and governance events
	$(BIN)/fraud-detection governance

# --- Docker (local) ----------------------------------------------------------------------
docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Bootstrap (first run) and start API, dashboards and MLflow UI in Docker
	docker compose up --build -d

docker-down:  ## Stop the Docker stack (data volumes are kept)
	docker compose down

test:  ## Run all tests with coverage
	$(BIN)/pytest --cov=fraud_detection --cov-report=term-missing

test-unit:  ## Run tests that do not need the real dataset
	$(BIN)/pytest -m "not integration"

lint:  ## Check code style
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:  ## Auto-format code
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

clean:  ## Remove generated outputs (keeps raw data, frozen snapshots and MLflow history)
	rm -rf data/interim/*.parquet data/processed/*.parquet artifacts/*.joblib artifacts/*.json \
		artifacts/model reports/eda reports/training
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
