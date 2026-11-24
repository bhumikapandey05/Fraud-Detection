# Fraud Detection with MLOps - one image for bootstrap, API, dashboards and MLflow UI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# libgomp1: OpenMP runtime needed by XGBoost on Linux. curl: container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached layer). Evidently needs pandas < 3, so it gets its own
# virtual environment at the path configs/config.yaml expects (.venv-monitoring).
COPY requirements.txt requirements-monitoring.txt ./
RUN pip install -r requirements.txt \
    && python -m venv /app/.venv-monitoring \
    && /app/.venv-monitoring/bin/pip install -r requirements-monitoring.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY .streamlit ./.streamlit
RUN pip install --no-deps -e . \
    && mkdir -p data/raw data/interim data/processed artifacts mlflow monitoring reports logs \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app
# MLflow tries to record the Git commit of each run; there is no git in the image.
ENV GIT_PYTHON_REFRESH=quiet

EXPOSE 8000 8501 8502 5000
CMD ["fraud-detection", "--help"]
