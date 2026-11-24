"""Run Evidently drift reports - executed INSIDE the isolated monitoring environment.

Evidently 0.7 does not support pandas 3, which the model and API use. To keep the
verified model environment untouched, this script runs in ``.venv-monitoring``
(pandas 2 + Evidently) as a subprocess. It deliberately imports nothing from the
``fraud_detection`` package: inputs arrive as Parquet files described by a JSON
job file, outputs are HTML reports plus a JSON summary.

Usage:  python evidently_runner.py job.json
"""

from __future__ import annotations

import json
import sys


def _detected(method: str, value: float, threshold: float) -> bool:
    # p-value tests flag drift when p < threshold; distance metrics when value >= threshold.
    return value < threshold if "p_value" in method else value >= threshold


def _value_drifts(snapshot_dict: dict) -> dict[str, dict]:
    out = {}
    for metric in snapshot_dict["metrics"]:
        cfg = metric.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            value = float(metric["value"])
            method, threshold = cfg["method"], float(cfg["threshold"])
            out[cfg["column"]] = {
                "method": method,
                "threshold": threshold,
                "value": value,
                "detected": _detected(method, value, threshold),
            }
    return out


def main(job_path: str) -> None:
    import evidently
    import pandas as pd
    from evidently import Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    with open(job_path) as fh:
        job = json.load(fh)
    columns = job["columns"]
    reference = pd.read_parquet(job["reference"])[columns]
    current = pd.read_parquet(job["current"])[columns]

    data_snapshot = Report(
        [DataDriftPreset(columns=columns), DataSummaryPreset(columns=columns)]
    ).run(reference_data=reference, current_data=current)
    data_snapshot.save_html(job["data_drift_html"])
    feature_drift = _value_drifts(data_snapshot.dict())

    score_col = job["score_column"]
    score_reference = pd.read_parquet(job["score_reference"])[[score_col]]
    score_current = pd.read_parquet(job["score_current"])[[score_col]]
    score_snapshot = Report([ValueDrift(column=score_col)]).run(
        reference_data=score_reference, current_data=score_current
    )
    score_snapshot.save_html(job["score_drift_html"])
    score_drift = _value_drifts(score_snapshot.dict())[score_col]

    drifted = sorted(
        (c for c, d in feature_drift.items() if d["detected"]),
        key=lambda c: feature_drift[c]["value"],
        reverse=True,
    )
    summary = {
        "evidently_version": evidently.__version__,
        "n_reference": len(reference),
        "n_current": len(current),
        "columns": feature_drift,
        "drifted_columns": drifted,
        "n_drifted": len(drifted),
        "drift_share": len(drifted) / len(columns),
        "score_drift": score_drift,
    }
    with open(job["summary"], "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
