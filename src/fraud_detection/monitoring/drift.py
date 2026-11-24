"""Bridge to the isolated Evidently environment (see ``evidently_runner.py``)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from fraud_detection.config import Config
from fraud_detection.data.schema import PCA_FEATURES, TIME_COLUMN

RUNNER = Path(__file__).with_name("evidently_runner.py")
# Raw Time only ever increases (seconds since the first transaction), so it would
# always look "drifted". The model uses the hour of day, so that is what we monitor.
MONITORED_COLUMNS = ["hour_of_day", "Amount", *PCA_FEATURES]


class MonitoringError(RuntimeError):
    """Monitoring could not run (e.g. the Evidently environment is missing)."""


def monitoring_view(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["hour_of_day"] = ((out[TIME_COLUMN] // 3600) % 24).astype(float)
    return out[MONITORED_COLUMNS]


def evidently_available(cfg: Config) -> bool:
    return cfg.monitoring.evidently_python_path().is_file()


def run_drift(
    cfg: Config,
    reference_features: pd.DataFrame,
    current_features: pd.DataFrame,
    reference_scores: pd.Series,
    current_scores: pd.Series,
    out_dir: Path,
    timeout: int = 900,
) -> dict:
    python = cfg.monitoring.evidently_python_path()
    if not python.is_file():
        raise MonitoringError(
            f"Evidently environment not found at {python}. Create it with `make monitoring-env`."
        )
    out_dir = Path(out_dir)
    work = out_dir / "_inputs"
    work.mkdir(parents=True, exist_ok=True)
    monitoring_view(reference_features).to_parquet(work / "reference.parquet", index=False)
    monitoring_view(current_features).to_parquet(work / "current.parquet", index=False)
    pd.DataFrame({"fraud_score": reference_scores.to_numpy()}).to_parquet(
        work / "score_reference.parquet", index=False
    )
    pd.DataFrame({"fraud_score": pd.Series(current_scores).to_numpy()}).to_parquet(
        work / "score_current.parquet", index=False
    )
    job = {
        "columns": MONITORED_COLUMNS,
        "reference": str(work / "reference.parquet"),
        "current": str(work / "current.parquet"),
        "score_column": "fraud_score",
        "score_reference": str(work / "score_reference.parquet"),
        "score_current": str(work / "score_current.parquet"),
        "data_drift_html": str(out_dir / "data_drift.html"),
        "score_drift_html": str(out_dir / "score_drift.html"),
        "summary": str(out_dir / "drift_summary.json"),
    }
    (work / "job.json").write_text(json.dumps(job))
    try:
        result = subprocess.run(
            [str(python), str(RUNNER), str(work / "job.json")],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise MonitoringError(f"Evidently runner failed: {result.stderr.strip()[-800:]}")
        return json.loads((out_dir / "drift_summary.json").read_text())
    finally:
        # Do not keep copies of transaction data next to the reports.
        for f in work.glob("*"):
            f.unlink()
        work.rmdir()
