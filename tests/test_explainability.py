"""SHAP explanation tests: correct output space, baseline, additivity, labels, API, dashboard."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import shap
from fastapi.testclient import TestClient

from fraud_detection.config import PreprocessingConfig
from fraud_detection.data.schema import FEATURE_COLUMNS, PCA_FEATURES
from fraud_detection.data.split import time_based_split
from fraud_detection.explainability.service import build_explainer
from fraud_detection.explainability.shap_explainer import (
    ExplanationError,
    FraudExplainer,
    feature_description,
    feature_label,
    output_to_input_map,
    sample_background,
)
from fraud_detection.features.preprocessing import build_preprocessor, split_features_target
from fraud_detection.models.detector import FraudDetector
from fraud_detection.models.factory import build_training_pipeline
from fraud_detection.serving.api import create_app

RF_PARAMS = {
    "n_estimators": 25,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "n_jobs": 1,
}


@pytest.fixture(scope="module")
def rf(synthetic_df_module):
    splits = time_based_split(synthetic_df_module, 0.7, 0.15)
    X, y = split_features_target(splits.train)
    pipe = build_training_pipeline("random_forest", RF_PARAMS, 0, PreprocessingConfig()).fit(X, y)
    detector = FraudDetector(pipe, 0.3, "random_forest", FEATURE_COLUMNS)
    background = sample_background(splits.train, 150, random_state=1)
    explainer = FraudExplainer(detector, background)
    rows = splits.validation[FEATURE_COLUMNS].head(60)
    return detector, explainer, background, rows


@pytest.fixture(scope="module")
def synthetic_df_module():
    from tests.conftest import make_synthetic_transactions

    return make_synthetic_transactions()


# --- correctness of the numbers -------------------------------------------------------------
def test_contributions_add_up_to_the_displayed_score(rf):
    detector, explainer, _, rows = rf
    scores = detector.predict_proba(rows)  # what /predict shows
    for exp, score in zip(explainer.explain(rows), scores, strict=True):
        total = exp.base_value + sum(c.contribution for c in exp.contributions)
        assert total == pytest.approx(score, abs=1e-6)
        assert exp.fraud_score == score
        assert exp.additivity_error < 1e-6


def test_baseline_is_mean_score_of_background_not_class_weighted_half(rf):
    detector, explainer, background, _ = rf
    assert explainer.base_value == pytest.approx(
        detector.predict_proba(background).mean(), abs=1e-9
    )
    # The default tree_path_dependent baseline reflects balanced class weights (~0.5);
    # ours must reflect real transactions instead.
    default = shap.TreeExplainer(detector.pipeline[-1]).expected_value
    assert abs(np.atleast_1d(default)[1] - 0.5) < 0.05
    assert explainer.base_value < 0.3


def test_explains_the_fraud_class_probability(rf):
    detector, explainer, _, rows = rf
    model = detector.pipeline[-1]
    assert list(model.classes_) == [0, 1] and explainer.class_index == 1
    (exp,) = explainer.explain(rows.head(1))
    assert (
        exp.fraud_score == model.predict_proba(detector.pipeline[:-1].transform(rows.head(1)))[0, 1]
    )


def test_each_transaction_field_appears_once(rf):
    _, explainer, _, rows = rf
    (exp,) = explainer.explain(rows.head(1))
    assert sorted(c.feature for c in exp.contributions) == sorted(FEATURE_COLUMNS)
    magnitudes = [abs(c.contribution) for c in exp.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all(c.contribution > 0 for c in exp.toward_fraud(30))
    assert all(c.contribution < 0 for c in exp.away_from_fraud(30))


def test_time_contribution_is_sum_of_hour_features(rf):
    detector, explainer, _, rows = rf
    transformed = detector.pipeline[:-1].transform(rows.head(3))
    raw = np.asarray(explainer._explainer.shap_values(transformed))[..., 1]
    names = explainer.model_feature_names
    hour = raw[:, names.index("hour_sin")] + raw[:, names.index("hour_cos")]
    for i, exp in enumerate(explainer.explain(rows.head(3))):
        assert exp.contribution_of("Time") == pytest.approx(hour[i], abs=1e-12)
        assert exp.contribution_of("V3") == pytest.approx(raw[i, names.index("V3")], abs=1e-12)
        assert exp.contribution_of("Amount") == pytest.approx(
            raw[i, names.index("Amount")], abs=1e-12
        )


def test_explanations_are_deterministic(rf):
    detector, _, background, rows = rf
    a = FraudExplainer(detector, background).explain(rows.head(5))
    b = FraudExplainer(detector, background).explain(rows.head(5))
    for x, y in zip(a, b, strict=True):
        assert [(c.feature, c.contribution) for c in x.contributions] == [
            (c.feature, c.contribution) for c in y.contributions
        ]


def test_explaining_does_not_change_predictions(rf):
    detector, explainer, _, rows = rf
    before = detector.predict(rows)
    explainer.explain(rows)
    np.testing.assert_array_equal(before, detector.predict(rows))


def test_inconsistent_shap_values_are_refused(rf, monkeypatch):
    _, explainer, _, rows = rf
    monkeypatch.setattr(
        explainer._explainer,
        "shap_values",
        lambda X, **_: np.zeros((len(X), len(explainer.model_feature_names), 2)),
    )
    with pytest.raises(ExplanationError, match="do not add up"):
        explainer.explain(rows.head(2))


# --- labels: anonymized features are not given invented meanings ------------------------------
def test_labels():
    for name in PCA_FEATURES:
        assert feature_label(name) == f"{name} (anonymized PCA component)"
        assert "not disclosed" in feature_description(name)
    assert feature_label("Amount").startswith("Amount")
    assert "hour of day" in feature_label("Time")


# --- mapping and guards ---------------------------------------------------------------------
def test_output_to_input_map_variants(synthetic_df_module):
    X, _ = split_features_target(synthetic_df_module)
    pre = build_preprocessor(PreprocessingConfig(drop_raw_time=False)).fit(X)
    mapping = output_to_input_map(pre)
    assert {k for k, v in mapping.items() if v == "Time"} == {"hour_sin", "hour_cos", "Time"}
    assert mapping["Amount"] == "Amount" and mapping["V28"] == "V28"


def test_non_tree_models_are_rejected(synthetic_df_module):
    X, y = split_features_target(synthetic_df_module)
    pipe = build_training_pipeline(
        "logistic_regression", {"max_iter": 500}, 0, PreprocessingConfig()
    ).fit(X, y)
    with pytest.raises(ExplanationError, match="tree ensembles"):
        FraudExplainer(FraudDetector(pipe, 0.5, "lr", FEATURE_COLUMNS), X.head(50))


def test_background_sampling(synthetic_df_module):
    a = sample_background(synthetic_df_module, 100, 3)
    assert a.equals(sample_background(synthetic_df_module, 100, 3))
    assert list(a.columns) == FEATURE_COLUMNS
    with pytest.raises(ExplanationError):
        sample_background(synthetic_df_module.head(10), 100, 3)


# --- service: background must be the frozen model's own training split ------------------------
def test_build_explainer_requires_the_frozen_training_split(serving_cfg, tmp_path):
    _, cfg = serving_cfg
    from fraud_detection.models.detector import load_detector

    detector = load_detector(cfg.paths.frozen_dir / "v1" / "model")
    assert build_explainer(cfg, detector, "v1").base_value >= 0

    altered = tmp_path / "processed"
    altered.mkdir()
    train = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
    train.iloc[:-1].to_parquet(altered / "train.parquet", index=False)
    other = replace(cfg, paths=replace(cfg.paths, processed_dir=altered))
    with pytest.raises(ExplanationError, match="differs"):
        build_explainer(other, detector, "v1")


# --- API ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tx(serving_cfg):
    _, cfg = serving_cfg
    val = pd.read_parquet(cfg.paths.processed_dir / "validation.parquet")
    return val.drop(columns="Class").head(8).to_dict(orient="records")


def test_served_model_is_the_random_forest(api_client):
    info = api_client.get("/model").json()
    assert info["model_name"] == "random_forest", "explanation tests expect a Random Forest"
    assert info["explanations_available"] is True


def test_explain_endpoint_is_consistent_with_predict(api_client, tx):
    for row in tx:
        p = api_client.post("/predict", json=row).json()
        r = api_client.post("/explain", json=row, params={"top_k": 4})
        assert r.status_code == 200
        e = r.json()
        assert e["fraud_score"] == p["fraud_score"]  # identical, not just close
        assert e["decision"] == p["decision"] and e["model"] == p["model"]
        assert len(e["contributions"]) == 30
        total = e["base_value"] + sum(c["contribution"] for c in e["contributions"])
        assert total == pytest.approx(e["fraud_score"], abs=1e-6)
        listed = e["top_toward_fraud"] + e["top_away_from_fraud"]
        assert len(e["top_toward_fraud"]) <= 4 and len(e["top_away_from_fraud"]) <= 4
        assert all(c["contribution"] > 0 for c in e["top_toward_fraud"])
        assert all(c["contribution"] < 0 for c in e["top_away_from_fraud"])
        other = (
            e["base_value"]
            + sum(c["contribution"] for c in listed)
            + e["other_features_contribution"]
        )
        assert other == pytest.approx(e["fraud_score"], abs=1e-6)
        assert e["additivity_error"] < 1e-6
        for c in e["contributions"]:
            if c["feature"].startswith("V"):
                assert c["label"].endswith("(anonymized PCA component)")


def test_explain_method_description(api_client, tx):
    method = api_client.post("/explain", json=tx[0]).json()["explanation_method"]
    assert "interventional" in method["method"]
    assert "probability of fraud" in method["output"]
    assert "training transactions" in method["baseline_meaning"]
    assert any("anonymized" in n for n in method["notes"])


def test_explain_validation(api_client, tx):
    assert api_client.post("/explain", json=tx[0], params={"top_k": 0}).status_code == 422
    assert api_client.post("/explain", json=tx[0], params={"top_k": 31}).status_code == 422
    bad = {**tx[0], "Amount": -3.0}
    r = api_client.post("/explain", json=bad)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_predict_response_is_unchanged(api_client, tx):
    """Regression guard: explanations must not alter the /predict contract."""
    body = api_client.post("/predict", json=tx[0]).json()
    assert set(body) == {
        "transaction_id",
        "fraud_score",
        "is_fraud",
        "decision",
        "warnings",
        "model",
    }
    batch = api_client.post("/predict/batch", json={"transactions": tx}).json()
    assert set(batch) == {"model", "n_transactions", "n_flagged", "predictions"}


def test_explanations_unavailable_does_not_affect_predictions(serving_cfg, tx):
    _, cfg = serving_cfg

    def broken(*_):
        raise ExplanationError("training split differs")

    with TestClient(create_app(cfg, explainer_factory=broken)) as client:
        assert client.post("/predict", json=tx[0]).status_code == 200
        r = client.post("/explain", json=tx[0])
        assert r.status_code == 503 and r.json()["error"] == "explanations_unavailable"
        info = client.get("/model").json()
        assert info["explanations_available"] is False
        assert "differs" in info["explanations_unavailable_reason"]
    with TestClient(create_app(cfg, explainer_factory=None)) as client:
        assert client.post("/explain", json=tx[0]).status_code == 503


# --- dashboard ------------------------------------------------------------------------------
def test_dashboard_client_explain(api_client, tx):
    from fraud_detection.dashboard.client import ApiClient

    e = ApiClient(api_client).explain(tx[0], top_k=2)
    assert len(e["top_toward_fraud"]) <= 2 and "base_value" in e
