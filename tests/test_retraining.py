"""Retraining: permitted data only, reproducible candidates, criteria, approval, rollback."""

import json
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from fraud_detection.config import PromotionCriteria, load_config
from fraud_detection.monitoring.simulate import generate_batch
from fraud_detection.retraining.approval import LEGACY, approval_state, is_approved
from fraud_detection.retraining.compare import evaluate_criteria
from fraud_detection.retraining.data import ForbiddenDataError, read_permitted
from fraud_detection.retraining.governance import (
    GovernanceError,
    log_candidate,
    promote,
    registry_status,
    rollback,
)

from .conftest import make_sandbox

YES = lambda _summary: True  # noqa: E731
NO = lambda _summary: False  # noqa: E731


def _feed(cfg, df, scenario, rows, seed, batch_id, tmp):
    from fraud_detection.pipelines.monitor import add_labels, ingest_batch

    x, y = generate_batch(df, scenario, rows, seed=seed, batch_id=batch_id)
    x.to_csv(tmp / f"{batch_id}.csv", index=False)
    y.to_csv(tmp / f"{batch_id}_l.csv", index=False)
    ingest_batch(cfg, tmp / f"{batch_id}.csv")
    add_labels(cfg, tmp / f"{batch_id}_l.csv")


@pytest.fixture(scope="module")
def gov(tmp_path_factory, serving_cfg, synthetic_df_session):
    """Sandbox with labelled production data and one ELIGIBLE (lenient criteria) candidate."""
    from fraud_detection.retraining.candidate import train_candidate

    _, cfg = make_sandbox(tmp_path_factory, serving_cfg, "governance", lenient=True)
    tmp = tmp_path_factory.mktemp("batches")
    _feed(cfg, synthetic_df_session, "stable", 1500, 1, "b1", tmp)
    _feed(cfg, synthetic_df_session, "concept_drift", 1500, 2, "b2", tmp)
    candidate = train_candidate(cfg)
    logged = log_candidate(cfg, Path(candidate["snapshot_dir"]))
    return cfg, candidate, logged


@pytest.fixture(scope="module")
def synthetic_df_session():
    from .conftest import make_synthetic_transactions

    return make_synthetic_transactions()


# --- data rules --------------------------------------------------------------------------------
def test_test_split_is_refused(serving_cfg, tmp_path):
    _, cfg = serving_cfg
    with pytest.raises(ForbiddenDataError, match="test split"):
        read_permitted(cfg.paths.processed_dir / "test.parquet", set(), "training")
    copy = tmp_path / "renamed.parquet"
    copy.write_bytes((cfg.paths.processed_dir / "test.parquet").read_bytes())
    from fraud_detection.pipelines.train import _file_sha256

    with pytest.raises(ForbiddenDataError, match="checksum"):
        read_permitted(copy, {_file_sha256(copy)}, "training")


def test_candidate_never_reads_test_and_uses_permitted_sources(gov, monkeypatch):
    cfg, candidate, _ = gov
    meta = candidate["metadata"]
    data = meta["training_data"]
    assert data["sources"]["production_labels"]["rows"] > 0
    assert data["holdout"]["rows"] > 0  # newest rows held out
    assert set(data["holdout"]["batches"]) == {"b2"}  # the newest batch
    assert "never read" in data["test_split"]
    assert meta["threshold"]["tuned_on"] == "validation"

    from fraud_detection.retraining import candidate as cand_mod

    opened = []
    real = pd.read_parquet

    def spy(path, *a, **k):
        opened.append(Path(path).name)
        return real(path, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", spy)
    cand_mod.train_candidate(cfg)
    assert "test.parquet" not in opened and "train.parquet" in opened


def test_candidate_training_is_reproducible(gov):
    cfg, first, _ = gov
    from fraud_detection.models.detector import load_detector
    from fraud_detection.retraining.candidate import train_candidate

    second = train_candidate(cfg)
    m1, m2 = first["metadata"], second["metadata"]
    assert (
        m1["training_data"]["training_fingerprint"] == m2["training_data"]["training_fingerprint"]
    )
    assert m1["threshold"]["value"] == m2["threshold"]["value"]
    assert m1["validation_metrics"]["pr_auc"] == m2["validation_metrics"]["pr_auc"]
    val = pd.read_parquet(cfg.paths.processed_dir / "validation.parquet")
    a = load_detector(Path(first["snapshot_dir"]) / "model").predict(val)
    b = load_detector(Path(second["snapshot_dir"]) / "model").predict(val)
    assert (a == b).all()


def test_candidate_snapshot_is_read_only_with_manifest(gov):
    _, candidate, _ = gov
    root = Path(candidate["snapshot_dir"])
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert manifest["kind"] == "candidate" and "test" not in manifest["split_files_sha256"]
    assert not (root / "model" / "fraud_detector.joblib").stat().st_mode & 0o222
    assert (root / "reports" / "comparison.md").read_text().startswith("# Candidate")


# --- criteria -------------------------------------------------------------------------------------
def _m(pr, rec, cost):
    return {"pr_auc": pr, "recall": rec, "expected_cost": cost}


def test_criteria():
    c = PromotionCriteria()
    ok = evaluate_criteria(
        _m(0.80, 0.85, 900), _m(0.80, 0.85, 900), _m(0.9, 0.9, 1), _m(0.8, 0.8, 1), c
    )
    assert ok["eligible"]
    worse = evaluate_criteria(
        _m(0.70, 0.85, 900), _m(0.80, 0.85, 900), _m(0.9, 0.9, 1), _m(0.8, 0.8, 1), c
    )
    assert not worse["eligible"]
    assert [ch["name"] for ch in worse["checks"] if not ch["passed"]] == ["validation_pr_auc"]
    costly = evaluate_criteria(
        _m(0.8, 0.85, 1200), _m(0.8, 0.85, 900), _m(0.9, 0.9, 1), _m(0.8, 0.8, 1), c
    )
    assert not costly["eligible"]
    no_holdout = evaluate_criteria(_m(0.8, 0.85, 900), _m(0.8, 0.85, 900), None, None, c)
    assert not no_holdout["eligible"]
    assert evaluate_criteria(
        _m(0.8, 0.85, 900), _m(0.8, 0.85, 900), None, None, PromotionCriteria(require_holdout=False)
    )["eligible"]


def test_approval_states():
    assert approval_state({"source": "frozen_snapshot"}) == LEGACY
    assert is_approved({"source": "frozen_snapshot"})
    assert approval_state({"approval_status": "pending"}) == "pending"
    assert not is_approved({"source": "candidate", "approval_status": "pending"})
    assert not is_approved({})


# --- governance: promotion and rollback --------------------------------------------------------
def _aliases(cfg):
    return registry_status(cfg)["aliases"]


def test_eligible_candidate_is_registered_pending(gov):
    cfg, _, logged = gov
    assert logged["eligible"] and logged["version"] == "2"
    v2 = next(v for v in registry_status(cfg)["versions"] if v["version"] == "2")
    assert v2["approval"] == "pending" and v2["aliases"] == []
    assert _aliases(cfg) == {"champion": "1"}


def test_promotion_requires_explicit_approval(gov):
    cfg, _, logged = gov
    v = logged["version"]
    with pytest.raises(GovernanceError, match="approved-by"):
        promote(cfg, v, " ", "reason", YES)
    with pytest.raises(GovernanceError, match="reason"):
        promote(cfg, v, "alice", "", YES)
    with pytest.raises(GovernanceError, match="not confirmed"):
        promote(cfg, v, "alice", "looks good", NO)
    with pytest.raises(GovernanceError, match="not an eligible candidate"):
        promote(cfg, "1", "alice", "x", YES)
    assert _aliases(cfg) == {"champion": "1"}  # nothing changed


def test_unapproved_champion_is_refused_by_serving(gov):
    cfg, _, logged = gov
    from fraud_detection.serving.model_loader import ModelLoadError, load_served_model

    client = MlflowClient()
    client.set_registered_model_alias(
        cfg.mlflow.registered_model_name, "champion", logged["version"]
    )
    try:
        with pytest.raises(ModelLoadError, match="not approved"):
            load_served_model(cfg)
    finally:
        client.set_registered_model_alias(cfg.mlflow.registered_model_name, "champion", "1")


def test_promote_serve_and_rollback(gov):
    cfg, candidate, logged = gov
    from fraud_detection.monitoring.store import MonitoringStore
    from fraud_detection.serving.model_loader import load_served_model

    result = promote(cfg, logged["version"], "alice", "recovers recall", YES)
    assert result == {**result, "champion": "2", "previous_champion": "1"}
    assert _aliases(cfg) == {"champion": "2", "previous-champion": "1"}
    served = load_served_model(cfg)
    assert served.version == "2" and served.snapshot == candidate["candidate_id"]
    v2 = next(v for v in registry_status(cfg)["versions"] if v["version"] == "2")
    assert v2["approval"] == "approved" and v2["approved_by"] == "alice"
    with pytest.raises(GovernanceError, match="not pending"):
        promote(cfg, "2", "alice", "again", YES)

    with pytest.raises(GovernanceError, match="not confirmed"):
        rollback(cfg, "bob", "drill", NO)
    back = rollback(cfg, "bob", "incident drill", YES)
    assert back["champion"] == "1" and _aliases(cfg) == {"champion": "1", "previous-champion": "2"}
    assert load_served_model(cfg).version == "1"
    actions = MonitoringStore(cfg.paths.monitoring_dir).events()["action"].tolist()
    assert actions[-2:] == ["promote", "rollback"] and "candidate_registered" in actions
    exp = MlflowClient().get_experiment_by_name(cfg.monitoring.governance_experiment_name)
    assert exp is not None


def test_rollback_refuses_unapproved_target_and_tampered_snapshot(
    gov, tmp_path_factory, serving_cfg
):
    cfg, _, _ = gov
    from fraud_detection.retraining.candidate import train_candidate

    # A second candidate that is registered but never approved cannot be a rollback target.
    extra = log_candidate(cfg, Path(train_candidate(cfg)["snapshot_dir"]))
    with pytest.raises(GovernanceError, match="never approved"):
        rollback(cfg, "bob", "x", YES, to_version=extra["version"])
    # Tampering with a candidate snapshot blocks its promotion.
    snap = cfg.paths.frozen_dir / next(
        v["snapshot"] for v in registry_status(cfg)["versions"] if v["version"] == extra["version"]
    )
    report = snap / "reports" / "comparison.md"
    report.chmod(0o644)
    report.write_text("edited")
    with pytest.raises(RuntimeError, match="modified"):
        promote(cfg, extra["version"], "alice", "x", YES)


def test_rejected_candidate_is_not_registered(tmp_path_factory, serving_cfg, synthetic_df_session):
    import yaml

    from fraud_detection.retraining.candidate import train_candidate

    path, cfg = make_sandbox(tmp_path_factory, serving_cfg, "strict")
    raw = yaml.safe_load(path.read_text())
    raw["retraining"]["criteria"]["require_holdout"] = True
    raw["retraining"]["use_production_labels"] = False  # -> no holdout -> not eligible
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    logged = log_candidate(cfg, Path(train_candidate(cfg)["snapshot_dir"]))
    assert not logged["eligible"] and logged["version"] is None
    assert [v["version"] for v in registry_status(cfg)["versions"]] == ["1"]


def test_frozen_configs_without_new_sections_still_load(serving_cfg, tmp_path):
    import yaml

    config_path, _ = serving_cfg
    raw = yaml.safe_load(Path(config_path).read_text())
    raw["paths"].pop("monitoring_dir")
    for key in ("monitoring", "retraining", "explainability", "serving"):
        raw.pop(key, None)
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump(raw))
    cfg = load_config(old)
    assert cfg.monitoring.min_fraud_labels == 20 and cfg.paths.monitoring_dir.name == "monitoring"


def test_promoted_candidate_reports_and_dashboard(gov, monkeypatch):
    """A retrained champion has no test results by design; reports + dashboard must cope."""
    import streamlit as st
    from fastapi.testclient import TestClient
    from streamlit.testing.v1 import AppTest

    from fraud_detection.dashboard import client as client_module
    from fraud_detection.serving.api import create_app

    cfg, candidate, logged = gov
    rollback(
        cfg,
        "carol",
        "serve the approved candidate for this test",
        YES,
        to_version=logged["version"],
    )
    try:
        with TestClient(create_app(cfg)) as client:
            perf = client.get("/model/performance").json()
            assert perf["snapshot"] == candidate["candidate_id"]
            assert perf["kind"] == "candidate" and perf["test"] is None
            assert perf["validation"]["metrics"]["pr_auc"] > 0
            assert perf["model_comparison"] == [] and perf["figures"] == []

            app_path = str(
                Path(__file__).resolve().parents[1]
                / "src"
                / "fraud_detection"
                / "dashboard"
                / "app.py"
            )
            monkeypatch.setenv(
                "FRAUD_CONFIG", str(Path(cfg.paths.frozen_dir).parent / "config.yaml")
            )
            monkeypatch.setattr(client_module, "make_http_client", lambda *a, **k: client)
            st.cache_data.clear()
            st.cache_resource.clear()
            at = AppTest.from_file(app_path, default_timeout=60).run()
            assert not at.exception
            assert any("never evaluated on the frozen test set" in i.value for i in at.info)
            assert "PR-AUC" in {m.label for m in at.metric}
    finally:
        rollback(cfg, "carol", "restore", YES, to_version="1")
