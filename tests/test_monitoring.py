"""Monitoring: store, quality, alerts, performance, simulation, reference, drift, capture."""

import numpy as np
import pandas as pd
import pytest

from fraud_detection.config import AlertThresholds
from fraud_detection.data.schema import FEATURE_COLUMNS
from fraud_detection.monitoring.alerts import evaluate_alerts, overall_status
from fraud_detection.monitoring.drift import MONITORED_COLUMNS, evidently_available, monitoring_view
from fraud_detection.monitoring.performance import labeled_performance
from fraud_detection.monitoring.quality import data_quality
from fraud_detection.monitoring.simulate import SCENARIOS, generate_batch
from fraud_detection.monitoring.store import MonitoringStore

from .conftest import make_sandbox


def _features(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 30)), columns=FEATURE_COLUMNS).assign(
        Time=lambda d: d["Time"].abs() * 1000, Amount=lambda d: d["Amount"].abs() * 50
    )


# --- store ------------------------------------------------------------------------------
def test_store_roundtrip_and_labels(tmp_path):
    store = MonitoringStore(tmp_path)
    f = _features(200)
    store.log_predictions(
        f, [f"t{i}" for i in range(200)], np.linspace(0, 1, 200), 0.5, "3", "b1", "test"
    )
    result = store.add_labels(
        pd.DataFrame({"transaction_id": ["t0", "t199", "ghost"], "label": [0, 1, 1]})
    )
    assert result == {"received": 3, "stored": 2, "unmatched": 1}
    store.add_labels(pd.DataFrame({"transaction_id": ["t0"], "label": [1]}))  # correction
    window = store.predictions("b1")
    assert len(window) == 200 and window.loc[0, "label"] == 1
    assert window["is_fraud"].sum() == (np.linspace(0, 1, 200) >= 0.5).sum()
    b = store.batches().iloc[0]
    assert (b["n_rows"], b["n_labeled"], b["n_fraud_labels"]) == (200, 2, 2)
    with pytest.raises(ValueError):
        store.add_labels(pd.DataFrame({"transaction_id": ["t1"], "label": [2]}))
    store.log_event("promote", "m", "1", "2", "alice", "why", {"x": 1})
    assert store.events().iloc[0]["actor"] == "alice"


def test_store_handles_many_labels(tmp_path):
    store = MonitoringStore(tmp_path)
    n = 40_000  # more than SQLite's bound-parameter limit
    store.log_predictions(_features(n), [str(i) for i in range(n)], np.zeros(n), 0.5, "1", "b", "t")
    assert (
        store.add_labels(pd.DataFrame({"transaction_id": [str(i) for i in range(n)], "label": 0}))[
            "stored"
        ]
        == n
    )


# --- quality / performance / alerts ---------------------------------------------------------
def test_data_quality():
    f = _features(100)
    low = {c: -5.0 for c in FEATURE_COLUMNS}
    high = {c: 5.0 for c in FEATURE_COLUMNS} | {"Amount": 1e6, "Time": 1e9}
    f.loc[0, "V3"] = 50.0
    f = pd.concat([f, f.iloc[[1, 2]]], ignore_index=True)
    q = data_quality(f, low, high, rejected={"n_rejected": 3, "reasons": {"missing": 3}})
    assert q["out_of_range_by_column"]["V3"] >= 1
    assert q["duplicate_rows"] == 2
    assert q["rejected_share"] == pytest.approx(3 / 105)


def test_labeled_performance_requires_enough_frauds():
    w = pd.DataFrame({"fraud_score": [0.9, 0.1, 0.8, 0.2], "label": [1, 0, 1, None]})
    assert labeled_performance(w, 0.5, 5, 120, 5)["status"] == "insufficient_labels"
    p = labeled_performance(w, 0.5, 5, 120, 2)
    assert p["status"] == "evaluated" and p["recall"] == 1.0 and p["n_labeled"] == 3


def _summary(**over):
    s = {
        "n_rows": 1000,
        "drift": {
            "drift_share": 0.0,
            "n_drifted": 0,
            "drifted_columns": [],
            "score_drift": {"detected": False, "value": 0.01, "threshold": 0.1, "method": "w"},
        },
        "scores": {"alert_rate": 0.002, "expected_alert_rate": 0.002, "alert_rate_ratio": 1.0},
        "quality": {
            "rejected_share": 0.0,
            "n_rejected": 0,
            "rejection_reasons": {},
            "out_of_range_share": 0.0,
            "out_of_range_by_column": {},
            "duplicate_share": 0.0,
            "duplicate_rows": 0,
        },
        "performance": {"status": "insufficient_labels"},
    }
    for k, v in over.items():
        s[k] = {**s[k], **v} if isinstance(v, dict) else v
    return s


@pytest.mark.parametrize(
    "override, metric, severity",
    [
        (
            {"drift": {"drift_share": 0.5, "n_drifted": 15, "drifted_columns": ["V1"]}},
            "drift_share",
            "critical",
        ),
        (
            {"drift": {"drift_share": 0.25, "n_drifted": 8, "drifted_columns": ["V1"]}},
            "drift_share",
            "warning",
        ),
        (
            {"drift": {"drift_share": 0.03, "n_drifted": 1, "drifted_columns": ["Amount"]}},
            "watched_feature_drift",
            "warning",
        ),
        ({"scores": {"alert_rate": 0.01, "alert_rate_ratio": 5.0}}, "alert_rate_ratio", "warning"),
        ({"quality": {"rejected_share": 0.02, "n_rejected": 20}}, "rejected_share", "critical"),
        (
            {"quality": {"duplicate_share": 0.05, "duplicate_rows": 50}},
            "duplicate_share",
            "warning",
        ),
        (
            {
                "performance": {
                    "status": "evaluated",
                    "recall": 0.4,
                    "precision": 0.9,
                    "pr_auc": 0.9,
                    "n_labeled": 100,
                    "n_fraud_labels": 30,
                }
            },
            "recall",
            "critical",
        ),
        ({"n_rows": 10}, "n_rows", "info"),
    ],
)
def test_alert_rules(override, metric, severity):
    alerts = evaluate_alerts(_summary(**override), AlertThresholds(), min_rows=200)
    assert any(a["metric"] == metric and a["severity"] == severity for a in alerts)


def test_clean_summary_has_no_alerts():
    alerts = evaluate_alerts(_summary(), AlertThresholds(), min_rows=200)
    assert alerts == [] and overall_status(alerts) == "ok"
    assert overall_status([{"severity": "info"}]) == "ok"
    assert overall_status([{"severity": "warning"}, {"severity": "critical"}]) == "critical"


# --- simulation / drift view ----------------------------------------------------------------
def test_simulated_batches(synthetic_df):
    a, la = generate_batch(synthetic_df, "stable", 500, seed=1, day=2)
    b, lb = generate_batch(synthetic_df, "stable", 500, seed=1, day=2)
    pd.testing.assert_frame_equal(a, b)  # deterministic
    assert list(a["transaction_id"]) == list(la["transaction_id"])
    assert (a["Time"] >= 4 * 86400).all()  # later days
    amt, _ = generate_batch(synthetic_df, "amount_shift", 2000, seed=1)
    base, _ = generate_batch(synthetic_df, "stable", 2000, seed=1)
    assert amt["Amount"].median() / base["Amount"].median() == pytest.approx(2.5, rel=0.05)
    q, lq = generate_batch(synthetic_df, "quality_issues", 1000, seed=1)
    original = q.iloc[:1000]  # duplicates are appended after the original rows
    assert original["V3"].isna().sum() == 20 and (original["V7"] == 300).sum() == 20
    assert len(q) == len(lq) == 1030
    cd, lcd = generate_batch(synthetic_df, "concept_drift", 2000, seed=1)
    assert lcd["label"].mean() > la["label"].mean()
    with pytest.raises(ValueError):
        generate_batch(synthetic_df, "nope", 10, 1)
    assert set(SCENARIOS) == {
        "stable",
        "amount_shift",
        "feature_shift",
        "quality_issues",
        "concept_drift",
    }


def test_monitoring_view_uses_hour_of_day_not_raw_time():
    view = monitoring_view(_features(10).assign(Time=[3600 * 25] * 10))
    assert list(view.columns) == MONITORED_COLUMNS and "Time" not in view
    assert (view["hour_of_day"] == 1).all()


# --- end-to-end monitoring in a sandbox ------------------------------------------------------
needs_evidently = pytest.mark.skipif(
    not evidently_available(__import__("fraud_detection.config", fromlist=["x"]).load_config()),
    reason="Evidently environment (.venv-monitoring) not installed",
)


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory, serving_cfg):
    return make_sandbox(tmp_path_factory, serving_cfg, "monitoring")


@needs_evidently
def test_ingest_label_and_monitor(sandbox, synthetic_df, tmp_path):
    from fraud_detection.pipelines.monitor import add_labels, ingest_batch, run_monitoring

    _, cfg = sandbox
    x, y = generate_batch(synthetic_df, "quality_issues", 800, seed=3, batch_id="q1")
    x.to_csv(tmp_path / "q1.csv", index=False)
    y.to_csv(tmp_path / "q1_labels.csv", index=False)
    ing = ingest_batch(cfg, tmp_path / "q1.csv")
    assert ing["rejected"] == 16 and ing["scored"] == len(x) - 16
    assert add_labels(cfg, tmp_path / "q1_labels.csv")["stored"] == ing["scored"]

    s = run_monitoring(cfg, "q1")
    assert s["status"] == "critical"
    assert {"rejected_share", "duplicate_share"} <= {a["metric"] for a in s["alerts"]}
    assert s["performance"]["status"] == "evaluated"
    assert set(s["drift"]["columns"]) == set(MONITORED_COLUMNS)
    for name in ("data_drift.html", "score_drift.html", "summary.json"):
        assert (cfg.paths.monitoring_dir / "reports" / s["run_id"] / name).is_file()
    assert not (cfg.paths.monitoring_dir / "reports" / s["run_id"] / "_inputs").exists()
    assert s["mlflow_run_id"]
    from fraud_detection.monitoring.store import MonitoringStore

    st = MonitoringStore(cfg.paths.monitoring_dir)
    assert st.runs().iloc[-1]["run_id"] == s["run_id"] and len(st.alerts()) >= 2


@needs_evidently
def test_stable_batch_is_quiet(sandbox, tmp_path):
    from fraud_detection.pipelines.monitor import ingest_batch, run_monitoring

    _, cfg = sandbox
    train = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")  # as the CLI does
    x, _ = generate_batch(train, "stable", 600, seed=4, batch_id="s1")
    x.to_csv(tmp_path / "s1.csv", index=False)
    ingest_batch(cfg, tmp_path / "s1.csv")
    s = run_monitoring(cfg, "s1", log_to_mlflow=False)
    assert s["drift"]["drift_share"] < cfg.monitoring.alerts.drift_share_warning
    assert not [a for a in s["alerts"] if a["category"] == "data_quality"]


def test_monitoring_without_evidently_env_fails_clearly(sandbox, monkeypatch):
    from dataclasses import replace

    from fraud_detection.monitoring.drift import MonitoringError, run_drift

    _, cfg = sandbox
    broken = replace(cfg, monitoring=replace(cfg.monitoring, evidently_python="/nope/python"))
    with pytest.raises(MonitoringError, match="make monitoring-env"):
        run_drift(broken, _features(), _features(), pd.Series([0.1]), pd.Series([0.1]), "/tmp")


# --- API capture -----------------------------------------------------------------------------
def test_api_captures_predictions(api_client, serving_cfg):
    _, cfg = serving_cfg
    val = pd.read_parquet(cfg.paths.processed_dir / "validation.parquet")
    rows = val.drop(columns="Class").head(3).to_dict(orient="records")
    store = MonitoringStore(cfg.paths.monitoring_dir)
    before = len(store.predictions())
    api_client.post("/predict", json={**rows[0], "transaction_id": "cap-1"})
    api_client.post("/predict/batch", json={"transactions": rows[1:]})
    after = store.predictions()
    assert len(after) == before + 3
    assert "cap-1" in set(after["transaction_id"])
    assert after.iloc[-1]["source"] == "api" and after.iloc[-1]["model_version"] == "1"


def test_capture_failure_never_breaks_predictions(api_client, serving_cfg, monkeypatch):
    _, cfg = serving_cfg
    row = (
        pd.read_parquet(cfg.paths.processed_dir / "validation.parquet")
        .drop(columns="Class")
        .iloc[0]
        .to_dict()
    )

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(api_client.app.state.store, "log_predictions", boom)
    assert api_client.post("/predict", json=row).status_code == 200


# --- monitoring dashboard ---------------------------------------------------------------------
MONITORING_APP = str(
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "src"
    / "fraud_detection"
    / "dashboard"
    / "monitoring_app.py"
)


def _run_dashboard(config_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("FRAUD_CONFIG", str(config_path))
    st.cache_resource.clear()
    return AppTest.from_file(MONITORING_APP, default_timeout=120).run()


def test_monitoring_dashboard_without_data(tmp_path_factory, serving_cfg, monkeypatch):
    path, _ = make_sandbox(tmp_path_factory, serving_cfg, "empty-dashboard")
    at = _run_dashboard(path, monkeypatch)
    assert not at.exception
    assert any("No monitoring data yet" in i.value for i in at.info)


@needs_evidently
def test_monitoring_dashboard_with_runs(sandbox, monkeypatch, tmp_path):
    from fraud_detection.pipelines.monitor import ingest_batch, run_monitoring

    path, cfg = sandbox
    train = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    x, _ = generate_batch(train, "amount_shift", 400, seed=9, batch_id="dash1")
    x.to_csv(tmp_path / "dash1.csv", index=False)
    ingest_batch(cfg, tmp_path / "dash1.csv")
    run_monitoring(cfg, "dash1", log_to_mlflow=False)
    at = _run_dashboard(path, monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Over time",
        "Alerts",
        "Drift report",
        "Models & approvals",
    ]
    assert "Latest status" in {m.label for m in at.metric}
    assert any("Champion" in m.value for m in at.sidebar.markdown)
