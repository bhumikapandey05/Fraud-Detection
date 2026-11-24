"""Prediction API tests against the session's frozen + registered model."""

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_detection.data.schema import FEATURE_COLUMNS
from fraud_detection.models.detector import load_detector
from fraud_detection.serving import model_loader
from fraud_detection.serving.api import create_app
from fraud_detection.serving.model_loader import (
    ModelLoadError,
    load_from_frozen,
    load_from_registry,
)


@pytest.fixture(scope="module")
def rows(serving_cfg):
    _, cfg = serving_cfg
    val = pd.read_parquet(cfg.paths.processed_dir / "validation.parquet")
    return val.drop(columns="Class").head(20).to_dict(orient="records")


def frozen_detector(cfg):
    return load_detector(cfg.paths.frozen_dir / "v1" / "model")


# --- health / model -----------------------------------------------------------------
def test_health_and_readiness(api_client, serving_cfg):
    _, cfg = serving_cfg
    assert api_client.get("/health").json()["status"] == "ok"
    ready = api_client.get("/health/ready")
    assert ready.status_code == 200
    model = ready.json()["model"]
    manifest = json.loads((cfg.paths.frozen_dir / "v1" / "MANIFEST.json").read_text())
    assert model["version"] == "1" and model["alias"] == "champion"
    assert model["source"] == "registry"
    assert model["detector_sha256"] == manifest["files"]["model/fraud_detector.joblib"]
    assert model["threshold"] == pytest.approx(manifest["threshold"], abs=1e-12)


def test_model_endpoint(api_client):
    info = api_client.get("/model").json()
    assert info["registered_model"] == "fraud-detector"
    assert info["max_batch_size"] == 50


# --- predictions ----------------------------------------------------------------------
def test_single_prediction_matches_frozen_model(api_client, serving_cfg, rows):
    _, cfg = serving_cfg
    detector = frozen_detector(cfg)
    for row in rows[:5]:
        r = api_client.post("/predict", json={**row, "transaction_id": "abc-1"})
        assert r.status_code == 200
        body = r.json()
        expected = detector.predict_proba(pd.DataFrame([row]))[0]
        assert body["fraud_score"] == pytest.approx(expected, abs=1e-12)
        assert body["is_fraud"] == (body["fraud_score"] >= detector.threshold)
        assert body["decision"] == ("REVIEW" if body["is_fraud"] else "APPROVE")
        assert body["transaction_id"] == "abc-1"
        assert body["model"]["threshold"] == detector.threshold
        assert body["model"]["version"] == "1"


def test_batch_preserves_order_and_matches_single(api_client, rows):
    payload = [{**row, "transaction_id": f"t{i}"} for i, row in enumerate(rows)]
    r = api_client.post("/predict/batch", json={"transactions": payload})
    assert r.status_code == 200
    body = r.json()
    assert [p["transaction_id"] for p in body["predictions"]] == [f"t{i}" for i in range(20)]
    assert body["n_transactions"] == 20
    assert body["n_flagged"] == sum(p["is_fraud"] for p in body["predictions"])
    single = api_client.post("/predict", json=payload[3]).json()
    assert body["predictions"][3]["fraud_score"] == pytest.approx(single["fraud_score"], abs=1e-12)


def test_batch_limit(api_client, rows):
    r = api_client.post("/predict/batch", json={"transactions": (rows * 3)[:51]})
    assert r.status_code == 413
    assert r.json()["error"] == "batch_too_large"


def test_request_id_header(api_client, rows):
    r = api_client.post("/predict", json=rows[0], headers={"X-Request-ID": "my-req-42"})
    assert r.headers["X-Request-ID"] == "my-req-42"
    assert api_client.get("/health").headers["X-Request-ID"]


def test_out_of_training_range_warning(api_client, rows):
    normal = api_client.post("/predict", json=rows[0]).json()
    assert normal["warnings"] == []
    odd = api_client.post("/predict", json={**rows[0], "V4": 900.0, "Amount": 900_000.0}).json()
    assert any(w.startswith("V4=") for w in odd["warnings"])
    assert any(w.startswith("Amount=") for w in odd["warnings"])


# --- validation -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda r: r.pop("V7"), "V7"),
        (lambda r: r.update(extra_field=1), "extra_field"),
        (lambda r: r.update(Amount="12.5"), "Amount"),
        (lambda r: r.update(V1=True), "V1"),
        (lambda r: r.update(V2=None), "V2"),
        (lambda r: r.update(Amount=-1.0), "Amount"),
        (lambda r: r.update(Time=-5.0), "Time"),
        (lambda r: r.update(V9=1e6), "V9"),
        (lambda r: r.update(transaction_id="bad id with spaces"), "transaction_id"),
    ],
)
def test_invalid_transactions_rejected(api_client, rows, mutate, field):
    row = dict(rows[0])
    mutate(row)
    r = api_client.post("/predict", json=row)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    assert any(d["field"] == field for d in body["detail"])
    # Only field name + message are returned: submitted values are never echoed back.
    assert all(set(d) == {"field", "message"} for d in body["detail"])


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_rejected(api_client, rows, token):
    text = json.dumps(rows[0]).replace(f'"Amount": {rows[0]["Amount"]}', f'"Amount": {token}')
    r = api_client.post("/predict", content=text, headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert any(d["field"] == "Amount" for d in r.json()["detail"])


def test_malformed_and_empty_requests(api_client):
    assert (
        api_client.post(
            "/predict", content=b"{oops", headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )
    assert api_client.post("/predict/batch", json={"transactions": []}).status_code == 422
    assert api_client.post("/predict/batch", json=[]).status_code == 422
    assert api_client.get("/predict").status_code == 405


# --- performance reports --------------------------------------------------------------
def test_performance_comes_from_frozen_reports(api_client, serving_cfg):
    _, cfg = serving_cfg
    meta = json.loads((cfg.paths.frozen_dir / "v1" / "model" / "model_metadata.json").read_text())
    perf = api_client.get("/model/performance").json()
    test = meta["test_metrics"]
    assert perf["test"]["metrics"]["pr_auc"] == test["pr_auc"]
    assert perf["test"]["confusion_matrix"] == {k: test[k] for k in ("tn", "fp", "fn", "tp")}
    business = perf["test"]["business"]
    assert business["total_cost"] == test["expected_cost"]
    assert business["cost_missed_fraud"] + business["cost_false_alarms"] == pytest.approx(
        test["expected_cost"]
    )
    assert perf["validation_vs_test"] is not None
    assert len(perf["model_comparison"]) == len(meta["selection"]["candidates"])


def test_figures(api_client):
    perf = api_client.get("/model/performance").json()
    assert perf["figures"]
    r = api_client.get(f"/model/performance/figures/{perf['figures'][0]}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert api_client.get("/model/performance/figures/secret.png").status_code == 404
    assert api_client.get("/model/performance/figures/..%2F..%2FMANIFEST.json").status_code == 404


# --- failure modes ---------------------------------------------------------------------
def test_fails_closed_when_model_cannot_load(serving_cfg, rows):
    _, cfg = serving_cfg

    def broken_loader(_):
        raise ModelLoadError("checksum mismatch")

    with TestClient(create_app(cfg, loader=broken_loader)) as client:
        assert client.get("/health").status_code == 200  # process is alive
        ready = client.get("/health/ready")
        assert ready.status_code == 503 and "checksum mismatch" in ready.json()["reason"]
        r = client.post("/predict", json=rows[0])
        assert r.status_code == 503 and r.json()["error"] == "model_unavailable"
        assert client.get("/model").status_code == 503


def test_unexpected_errors_are_hidden(serving_cfg, rows):
    _, cfg = serving_cfg
    with TestClient(create_app(cfg), raise_server_exceptions=False) as client:

        def explode(_):
            raise RuntimeError("secret internal detail")

        client.app.state.model.score = explode
        r = client.post("/predict", json=rows[0])
        assert r.status_code == 500
        assert r.json()["error"] == "internal_error"
        assert "secret" not in r.text and r.json()["request_id"]


def test_registry_loader_rejects_unapproved_or_altered_models(serving_cfg, monkeypatch):
    _, cfg = serving_cfg
    with pytest.raises(ModelLoadError, match="not found"):
        load_from_registry(cfg, "does-not-exist")
    monkeypatch.setattr(model_loader, "_file_sha256", lambda _: "0" * 64)
    with pytest.raises(ModelLoadError, match="SHA-256"):
        load_from_registry(cfg, "v1")


def test_registry_loader_requires_snapshot_tag(serving_cfg):
    _, cfg = serving_cfg
    import shutil

    other = cfg.paths.frozen_dir / "other"
    shutil.copytree(cfg.paths.frozen_dir / "v1", other)
    try:
        with pytest.raises(ModelLoadError, match="not the approved snapshot"):
            load_from_registry(cfg, "other")
    finally:
        for p in other.rglob("*"):
            if p.is_file():
                p.chmod(0o644)
        shutil.rmtree(other)


def test_registry_loader_requires_mlflow_db(serving_cfg, tmp_path):
    _, cfg = serving_cfg
    missing = replace(cfg, mlflow=replace(cfg.mlflow, tracking_db=tmp_path / "none.db"))
    with pytest.raises(ModelLoadError, match="database not found"):
        load_from_registry(missing, "v1")


def test_frozen_source_serves_identical_scores(serving_cfg, rows):
    _, cfg = serving_cfg
    frozen_cfg = replace(cfg, serving=replace(cfg.serving, model_source="frozen"))
    reg = load_from_registry(cfg, "v1")
    frz = load_from_frozen(frozen_cfg, "v1")
    assert frz.detector_sha256 == reg.detector_sha256
    assert math.isclose(frz.threshold, reg.threshold, abs_tol=1e-15)
    frame = pd.DataFrame(rows)[FEATURE_COLUMNS]
    np.testing.assert_allclose(frz.score(frame), reg.score(frame), atol=1e-12)
    with TestClient(create_app(frozen_cfg)) as client:
        assert client.get("/health/ready").json()["model"]["source"] == "frozen"
        assert client.post("/predict", json=rows[0]).status_code == 200


def test_serving_never_writes_to_frozen_snapshot(api_client, serving_cfg):
    from fraud_detection.pipelines.freeze import verify_frozen

    _, cfg = serving_cfg
    api_client.post("/predict", json={k: 0.0 for k in FEATURE_COLUMNS})
    assert verify_frozen(cfg, "v1")["ok"]


def test_performance_follows_served_snapshot_in_auto_mode(serving_cfg):
    """Regression: with frozen_snapshot='auto' the reports were read from frozen/auto/."""
    _, cfg = serving_cfg
    auto = replace(cfg, serving=replace(cfg.serving, frozen_snapshot="auto"))
    with TestClient(create_app(auto)) as client:
        assert client.get("/health/ready").json()["model"]["frozen_snapshot"] == "v1"
        perf = client.get("/model/performance")
        assert perf.status_code == 200
        body = perf.json()
        assert body["snapshot"] == "v1" and body["test"] is not None
        assert client.get(f"/model/performance/figures/{body['figures'][0]}").status_code == 200


def test_reports_require_a_concrete_snapshot(serving_cfg):
    from fraud_detection.serving.reports import load_performance

    _, cfg = serving_cfg
    for bad in ("auto", "", "../x"):
        with pytest.raises(ValueError):
            load_performance(cfg, bad)
