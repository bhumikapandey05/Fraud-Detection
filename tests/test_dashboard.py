"""Dashboard tests: the HTTP client (against the real API app) and the Streamlit page."""

import httpx
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from fraud_detection.config import PROJECT_ROOT
from fraud_detection.dashboard import client as client_module
from fraud_detection.dashboard.client import ApiClient, ApiError, prepare_batch, read_csv_upload
from fraud_detection.data.schema import FEATURE_COLUMNS

APP = str(PROJECT_ROOT / "src" / "fraud_detection" / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def frame(serving_cfg):
    _, cfg = serving_cfg
    return pd.read_parquet(cfg.paths.processed_dir / "validation.parquet").head(7)


# --- client ---------------------------------------------------------------------------
def test_client_calls(api_client, frame):
    api = ApiClient(api_client)
    assert api.health()["status"] == "ok"
    assert api.ready()["status"] == "ready"
    assert api.model_info()["version"] == "1"
    assert api.performance()["test"]["metrics"]["pr_auc"] > 0
    row = frame[FEATURE_COLUMNS].iloc[0].to_dict()
    assert api.predict(row)["decision"] in {"REVIEW", "APPROVE"}


def test_predict_frame_chunks_and_keeps_order(api_client, frame):
    api = ApiClient(api_client)
    chunked = api.predict_frame(frame, batch_size=3)  # 7 rows -> 3 requests
    whole = api.predict_frame(frame, batch_size=50)
    assert list(chunked["transaction_id"]) == [f"row-{i}" for i in range(1, 8)]
    pd.testing.assert_series_equal(chunked["fraud_score"], whole["fraud_score"], atol=1e-12)


def test_client_surfaces_api_errors(api_client, frame):
    api = ApiClient(api_client)
    row = frame[FEATURE_COLUMNS].iloc[0].to_dict()
    with pytest.raises(ApiError) as exc:
        api.predict({**row, "Amount": -1.0})
    assert exc.value.status == 422 and "Amount" in exc.value.message


def test_client_handles_unreachable_api():
    api = ApiClient(httpx.Client(base_url="http://127.0.0.1:9", timeout=0.5))
    with pytest.raises(ApiError) as exc:
        api.health()
    assert exc.value.status is None and "make api" in exc.value.message
    assert api.ready()["status"] == "not_ready"


def test_prepare_batch(frame):
    out = prepare_batch(frame)  # includes the Class column -> ignored
    assert list(out.columns) == ["transaction_id", *FEATURE_COLUMNS]
    ids = frame.assign(transaction_id=["a b", "c/d", *[f"x{i}" for i in range(5)]])
    assert list(prepare_batch(ids)["transaction_id"][:2]) == ["a_b", "c_d"]
    with pytest.raises(ValueError, match="Missing required columns: V3"):
        prepare_batch(frame.drop(columns="V3"))
    bad = frame.astype({"V1": object})
    bad.loc[bad.index[2], "V1"] = "oops"
    with pytest.raises(ValueError, match=r"first rows: \[4\]"):
        prepare_batch(bad)
    with pytest.raises(ValueError, match="no rows"):
        prepare_batch(frame.iloc[0:0])


def test_read_csv_upload_limit(frame):
    data = frame.to_csv(index=False).encode()
    assert len(read_csv_upload(data, max_rows=10)) == 7
    with pytest.raises(ValueError, match="limit"):
        read_csv_upload(data, max_rows=5)


# --- Streamlit page -------------------------------------------------------------------
@pytest.fixture
def dashboard(serving_cfg, api_client, monkeypatch):
    config_path, _ = serving_cfg
    monkeypatch.setenv("FRAUD_CONFIG", str(config_path))
    monkeypatch.setattr(client_module, "make_http_client", lambda *a, **k: api_client)
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=60)


def test_dashboard_renders_all_tabs(dashboard):
    at = dashboard.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Score a transaction",
        "Batch scoring",
        "Model performance",
        "Model details",
    ]
    assert any("API ready" in s.value for s in at.sidebar.success)
    labels = {m.label for m in at.metric}
    assert {"PR-AUC", "Precision", "Recall", "F1", "Cost with model", "Savings"} <= labels


def test_dashboard_scores_an_example(dashboard):
    at = dashboard.run()
    next(b for b in at.button if b.label == "Fraud example").click().run()
    next(b for b in at.button if "Score" in b.label).click().run()
    assert not at.exception
    decisions = [e.value for e in at.error] + [s.value for s in at.success]
    assert any(v.startswith("**REVIEW**") or v.startswith("**APPROVE**") for v in decisions)
    assert "Fraud score" in {m.label for m in at.metric}
    # The SHAP explanation is rendered under the result.
    assert any("Why this score?" in m.value for m in at.markdown)
    assert any("typical transaction scores" in m.value for m in at.markdown)
    assert any("the model score shown above" in c.value for c in at.caption)


def test_dashboard_degrades_gracefully_without_api(serving_cfg, monkeypatch):
    config_path, _ = serving_cfg
    monkeypatch.setenv("FRAUD_CONFIG", str(config_path))
    monkeypatch.setattr(
        client_module,
        "make_http_client",
        lambda *a, **k: httpx.Client(base_url="http://127.0.0.1:9", timeout=0.5),
    )
    st.cache_data.clear()
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("not available" in w.value for w in at.warning)


# --- SHAP explanation integration (regression tests) --------------------------------------------
def _score_example(at, label):
    next(b for b in at.button if b.label == label).click().run()
    next(b for b in at.button if "Score" in b.label).click().run()
    return at


def _texts(at):
    return (
        [m.value for m in at.markdown]
        + [i.value for i in at.info]
        + [w.value for w in at.warning]
        + [c.value for c in at.caption]
    )


def _expected(api_client, at):
    """What /predict returns for the transaction currently in the form."""
    return api_client.post("/predict", json=dict(at.session_state["tx"])).json()


def _shows_prediction(at, expected):
    decisions = [e.value for e in at.error] + [s.value for s in at.success]
    assert any(v.startswith(f"**{expected['decision']}**") for v in decisions)
    score = next(m.value for m in at.metric if m.label == "Fraud score")
    assert score == f"{expected['fraud_score']:.4f}"


@pytest.mark.parametrize("label", ["Legitimate example", "Fraud example"])
def test_explanation_appears_under_prediction(dashboard, api_client, label):
    at = _score_example(dashboard.run(), label)
    assert not at.exception
    expected = _expected(api_client, at)
    _shows_prediction(at, expected)
    texts = _texts(at)
    assert any("Why this score?" in t for t in texts)
    assert any("the model score shown above" in t for t in texts)
    # The explanation is for the same score the prediction shows.
    assert any(f"giving the fraud score **{expected['fraud_score']:.4f}**" in t for t in texts)


def test_client_without_explain_method_does_not_crash(dashboard, api_client, monkeypatch):
    """Regression: AttributeError: 'ApiClient' object has no attribute 'explain'."""
    monkeypatch.delattr(client_module.ApiClient, "explain")
    at = _score_example(dashboard.run(), "Fraud example")
    assert not at.exception
    _shows_prediction(at, _expected(api_client, at))
    assert any("Restart the dashboard" in i.value for i in at.info)
    assert not any("Why this score?" in m.value for m in at.markdown)


@pytest.mark.parametrize(
    "error, message",
    [
        (ApiError(404, "Not Found"), "Restart it with `make api`"),
        (ApiError(503, "training split differs"), "currently unavailable"),
        (ApiError(500, "Unexpected server error"), "Explanation not available"),
        (RuntimeError("boom"), "failed unexpectedly"),
        (httpx.ReadTimeout("slow"), "failed unexpectedly"),
    ],
)
def test_explanation_failures_are_reported_not_raised(
    dashboard, api_client, monkeypatch, error, message
):
    def failing(self, transaction, top_k=5):
        raise error

    monkeypatch.setattr(client_module.ApiClient, "explain", failing)
    at = _score_example(dashboard.run(), "Legitimate example")
    assert not at.exception
    _shows_prediction(at, _expected(api_client, at))
    assert any(message in i.value for i in at.info)


def test_malformed_explanation_response_is_reported(dashboard, api_client, monkeypatch):
    monkeypatch.setattr(
        client_module.ApiClient,
        "explain",
        lambda self, transaction, top_k=5: {"unexpected": "shape"},
    )
    at = _score_example(dashboard.run(), "Fraud example")
    assert not at.exception
    _shows_prediction(at, _expected(api_client, at))
    assert any("could not be displayed" in w.value for w in at.warning)


def test_real_api_with_explanations_disabled(serving_cfg, monkeypatch):
    from fastapi.testclient import TestClient

    from fraud_detection.serving.api import create_app

    config_path, cfg = serving_cfg
    with TestClient(create_app(cfg, explainer_factory=None)) as client:
        monkeypatch.setenv("FRAUD_CONFIG", str(config_path))
        monkeypatch.setattr(client_module, "make_http_client", lambda *a, **k: client)
        st.cache_data.clear()
        st.cache_resource.clear()
        at = _score_example(AppTest.from_file(APP, default_timeout=60).run(), "Fraud example")
        assert not at.exception
        _shows_prediction(at, client.post("/predict", json=dict(at.session_state["tx"])).json())
        assert any("currently unavailable" in i.value for i in at.info)


def test_failed_scoring_never_shows_previous_explanation(dashboard, monkeypatch):
    at = _score_example(dashboard.run(), "Fraud example")
    assert any("Why this score?" in m.value for m in at.markdown)

    def broken_predict(self, transaction):
        raise ApiError(503, "model unavailable")

    monkeypatch.setattr(client_module.ApiClient, "predict", broken_predict)
    next(b for b in at.button if "Score" in b.label).click().run()
    assert not at.exception
    assert any("Could not score" in e.value for e in at.error)
    assert not any("Why this score?" in m.value for m in at.markdown)
