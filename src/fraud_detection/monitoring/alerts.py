"""Turn a monitoring summary into alerts using the thresholds in ``monitoring.alerts``."""

from __future__ import annotations

from fraud_detection.config import AlertThresholds

SEVERITY_ORDER = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


def _alert(severity, category, metric, value, threshold, message) -> dict:
    return {
        "severity": severity,
        "category": category,
        "metric": metric,
        "value": None if value is None else float(value),
        "threshold": None if threshold is None else float(threshold),
        "message": message,
    }


def evaluate_alerts(summary: dict, t: AlertThresholds, min_rows: int) -> list[dict]:
    alerts: list[dict] = []
    n = summary["n_rows"]
    if n < min_rows:
        alerts.append(
            _alert(
                "info",
                "volume",
                "n_rows",
                n,
                min_rows,
                f"Only {n} rows; drift statistics are unreliable below {min_rows}",
            )
        )

    drift = summary.get("drift")
    if drift:
        share = drift["drift_share"]
        top = ", ".join(drift["drifted_columns"][:5]) or "none"
        if share >= t.drift_share_critical:
            alerts.append(
                _alert(
                    "critical",
                    "data_drift",
                    "drift_share",
                    share,
                    t.drift_share_critical,
                    f"{share:.0%} of monitored features drifted (top: {top})",
                )
            )
        elif share >= t.drift_share_warning:
            alerts.append(
                _alert(
                    "warning",
                    "data_drift",
                    "drift_share",
                    share,
                    t.drift_share_warning,
                    f"{share:.0%} of monitored features drifted (top: {top})",
                )
            )
        watched = [c for c in drift["drifted_columns"] if c in t.watched_features]
        if watched and share < t.drift_share_warning:
            alerts.append(
                _alert(
                    "warning",
                    "data_drift",
                    "watched_feature_drift",
                    len(watched),
                    0,
                    f"Drift in key feature(s): {', '.join(watched)}",
                )
            )
        elif drift["n_drifted"] and share < t.drift_share_warning:
            alerts.append(
                _alert(
                    "info",
                    "data_drift",
                    "drift_share",
                    share,
                    t.drift_share_warning,
                    f"Drift in {top} (below alert level)",
                )
            )
        sd = drift["score_drift"]
        if sd["detected"]:
            alerts.append(
                _alert(
                    "warning",
                    "prediction_drift",
                    "score_drift",
                    sd["value"],
                    sd["threshold"],
                    f"Fraud-score distribution drifted vs validation reference "
                    f"({sd['method']} = {sd['value']:.3f})",
                )
            )

    scores = summary["scores"]
    ratio = scores.get("alert_rate_ratio")
    if ratio is not None and n >= min_rows:
        if ratio > t.alert_rate_ratio_max:
            alerts.append(
                _alert(
                    "warning",
                    "prediction_drift",
                    "alert_rate_ratio",
                    ratio,
                    t.alert_rate_ratio_max,
                    f"Review rate {scores['alert_rate']:.2%} is {ratio:.1f}x the "
                    f"expected {scores['expected_alert_rate']:.2%}",
                )
            )
        elif ratio < 1 / t.alert_rate_ratio_max:
            alerts.append(
                _alert(
                    "warning",
                    "prediction_drift",
                    "alert_rate_ratio",
                    ratio,
                    1 / t.alert_rate_ratio_max,
                    f"Review rate {scores['alert_rate']:.2%} is unusually low "
                    f"({ratio:.2f}x the expected {scores['expected_alert_rate']:.2%})",
                )
            )

    q = summary["quality"]
    if q["rejected_share"] > t.missing_share_max:
        alerts.append(
            _alert(
                "critical",
                "data_quality",
                "rejected_share",
                q["rejected_share"],
                t.missing_share_max,
                f"{q['n_rejected']} rows could not be scored (missing/invalid "
                f"values: {q['rejection_reasons']})",
            )
        )
    if q["out_of_range_share"] > t.out_of_range_share_max:
        cols = ", ".join(list(q["out_of_range_by_column"])[:5])
        alerts.append(
            _alert(
                "warning",
                "data_quality",
                "out_of_range_share",
                q["out_of_range_share"],
                t.out_of_range_share_max,
                f"{q['out_of_range_share']:.1%} of rows outside the training range ({cols})",
            )
        )
    if q["duplicate_share"] > t.duplicate_share_max:
        alerts.append(
            _alert(
                "warning",
                "data_quality",
                "duplicate_share",
                q["duplicate_share"],
                t.duplicate_share_max,
                f"{q['duplicate_rows']} duplicate transactions ({q['duplicate_share']:.1%})",
            )
        )

    perf = summary["performance"]
    if perf["status"] == "evaluated":
        for metric, limit, severity in (
            ("recall", t.min_recall, "critical"),
            ("precision", t.min_precision, "warning"),
            ("pr_auc", t.min_pr_auc, "warning"),
        ):
            if perf[metric] < limit:
                alerts.append(
                    _alert(
                        severity,
                        "performance",
                        metric,
                        perf[metric],
                        limit,
                        f"{metric} {perf[metric]:.3f} below {limit:.2f} on "
                        f"{perf['n_labeled']} labelled transactions "
                        f"({perf['n_fraud_labels']} frauds)",
                    )
                )
    return alerts


def overall_status(alerts: list[dict]) -> str:
    level = max((SEVERITY_ORDER[a["severity"]] for a in alerts), default=0)
    return {0: "ok", 1: "ok", 2: "warning", 3: "critical"}[level]
