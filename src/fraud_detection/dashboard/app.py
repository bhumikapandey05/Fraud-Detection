"""Fraud detection dashboard (Streamlit).

Run with:  streamlit run src/fraud_detection/dashboard/app.py
or:        make dashboard

Tabs
----
* Score a transaction - enter / load a transaction, get score + decision from the API
* Batch scoring       - upload a CSV, score it through the API, download results
* Model performance   - recorded test results of the frozen release (no recomputation)
* Model details       - registry version, threshold, checksums, how it was selected
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

from fraud_detection.config import load_config
from fraud_detection.dashboard.client import (
    ApiClient,
    ApiError,
    make_http_client,
    prepare_batch,
    read_csv_upload,
)
from fraud_detection.data.schema import AMOUNT_COLUMN, FEATURE_COLUMNS, PCA_FEATURES, TIME_COLUMN

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
MAX_UPLOAD_ROWS = 50_000
EXPLAIN_TOP_K = 8  # features listed toward / away from fraud in each explanation

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")


# ---------------------------------------------------------------------------------- setup
@st.cache_resource
def get_config():
    return load_config(os.environ.get("FRAUD_CONFIG"))


def get_api() -> ApiClient:
    cfg = get_config()
    url = os.environ.get("FRAUD_API_URL", cfg.serving.dashboard_api_url)
    return ApiClient(make_http_client(url))


@st.cache_data(show_spinner=False)
def training_examples() -> dict[str, dict] | None:
    """One legitimate and one fraud example from the TRAINING split (never test data)."""
    path = get_config().paths.processed_dir / "train.parquet"
    if not path.is_file():
        return None
    train = pd.read_parquet(path)
    legit = train[train["Class"] == 0].sample(1, random_state=7).iloc[0]
    fraud = train[train["Class"] == 1].sample(1, random_state=7).iloc[0]
    return {
        "legitimate": legit[FEATURE_COLUMNS].to_dict(),
        "fraud": fraud[FEATURE_COLUMNS].to_dict(),
    }


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


api = get_api()
ready = api.ready()
model = ready.get("model") if ready.get("status") == "ready" else None
max_batch = api.model_info().get("max_batch_size", 1000) if model else 1000

# -------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛡️ Fraud Detection")
    if model:
        st.success("API ready")
        st.markdown(
            f"**Model:** {model['display_name']}  \n"
            f"**Registry:** `{model.get('registered_model') or '—'}` "
            f"v{model.get('version') or '—'} @{model.get('alias') or '—'}  \n"
            f"**Threshold:** `{model['threshold']:.4f}`  \n"
            f"**Release:** `{model['frozen_snapshot']}`  \n"
            f"**Data version:** `{model.get('data_version') or '—'}`"
        )
    else:
        st.error("API not ready")
        st.caption(ready.get("reason", "Unknown reason"))
        st.code("make api", language="bash")
    st.divider()
    st.caption(
        "Scores come from the prediction API. Transactions with a fraud score at or "
        "above the threshold are sent for **REVIEW**; the rest are **APPROVED**."
    )

st.title("Credit Card Fraud Detection")
if not model:
    st.warning(
        "The prediction API is not available, so scoring and performance views are "
        "disabled. Start it in another terminal with `make api`, then refresh."
    )
    st.stop()

tab_single, tab_batch, tab_perf, tab_model = st.tabs(
    ["Score a transaction", "Batch scoring", "Model performance", "Model details"]
)


# ------------------------------------------------------------------------- single transaction
def score_bar(score: float, threshold: float) -> alt.LayerChart:
    """Horizontal score bar with the review threshold marked and labelled."""
    data = pd.DataFrame({"label": ["Fraud score"], "score": [score]})
    color = ORANGE if score >= threshold else BLUE
    x_scale = alt.Scale(domain=[0, 1])
    bar = (
        alt.Chart(data)
        .mark_bar(color=color, cornerRadiusEnd=4, height=22)
        .encode(
            x=alt.X(
                "score:Q",
                scale=x_scale,
                title="Fraud score (0 = safe, 1 = fraud)",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("label:N", title=None, axis=None),
            tooltip=[alt.Tooltip("score:Q", format=".4f", title="Fraud score")],
        )
    )
    marker = pd.DataFrame({"t": [threshold]})
    rule = (
        alt.Chart(marker)
        .mark_rule(color="#0b0b0b", strokeDash=[4, 3], size=2)
        .encode(
            x=alt.X("t:Q", scale=x_scale),
            tooltip=[alt.Tooltip("t:Q", format=".4f", title="Review threshold")],
        )
    )
    return (bar + rule).properties(height=90)


def short_label(c: dict) -> str:
    if c["feature"].startswith("V"):
        return f"{c['feature']} · anonymized"
    if c["feature"] == "Time":
        return f"Time · {c['display_value'].split('(')[-1].rstrip(')')}"
    return f"Amount · {c['display_value']}"


def _axis_format(values: pd.Series) -> str:
    """Enough decimals that tick labels stay distinct, even for tiny contributions."""
    largest = float(values.abs().max() or 0.0)
    decimals = 3 if largest >= 0.05 else 4 if largest >= 0.005 else 5
    return f"+.{decimals}f"


def contribution_chart(exp: dict) -> alt.Chart:
    """Diverging bars: toward fraud (orange, right) vs away from fraud (blue, left)."""
    rows = [
        {
            "feature": short_label(c),
            "contribution": c["contribution"],
            "direction": "Toward fraud" if c["contribution"] > 0 else "Away from fraud",
            "value": c["display_value"],
            "label": c["label"],
        }
        for c in exp["top_toward_fraud"] + exp["top_away_from_fraud"]
    ]
    n_other = len(exp["contributions"]) - len(rows)
    rows.append(
        {
            "feature": f"Other {n_other} features (combined)",
            "contribution": exp["other_features_contribution"],
            "direction": "Other (combined)",
            "value": "—",
            "label": f"Sum of the {n_other} features not listed above",
        }
    )
    data = pd.DataFrame(rows)
    order = data["feature"].tolist()
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("feature:N", sort=order, title=None),
            x=alt.X(
                "contribution:Q",
                title="Contribution to the fraud score",
                axis=alt.Axis(format=_axis_format(data["contribution"])),
            ),
            color=alt.Color(
                "direction:N",
                legend=alt.Legend(orient="bottom", title=None),
                scale=alt.Scale(
                    domain=["Toward fraud", "Away from fraud", "Other (combined)"],
                    range=[ORANGE, BLUE, GREY],
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip("contribution:Q", format="+.5f", title="Contribution"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#0b0b0b").encode(x="x:Q")
    return (bars + zero).properties(height=alt.Step(26))


def render_explanation(exp: dict) -> None:
    contributions = exp["contributions"]
    up = sum(c["contribution"] for c in contributions if c["contribution"] > 0)
    down = sum(c["contribution"] for c in contributions if c["contribution"] < 0)
    st.markdown("#### Why this score?")
    st.markdown(
        f"A typical transaction scores **{exp['base_value']:.4f}** (the baseline). For this "
        f"transaction, features pushing **toward fraud** add **{up:+.4f}** and features pushing "
        f"**away from fraud** add **{down:+.4f}**, giving the fraud score "
        f"**{exp['fraud_score']:.4f}**."
    )
    st.altair_chart(contribution_chart(exp), width="stretch")
    st.caption(
        f"Check: baseline {exp['base_value']:.4f} {up:+.4f} {down:+.4f} = "
        f"{exp['fraud_score']:.4f}, the model score shown above "
        f"(baseline + contributions differs from it by only {exp['additivity_error']:.1e})."
    )
    toward, away = st.columns(2)
    for col, title, items in (
        (toward, "Toward fraud", exp["top_toward_fraud"]),
        (away, "Away from fraud", exp["top_away_from_fraud"]),
    ):
        col.markdown(f"**{title}**")
        if items:
            col.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Feature": c["label"],
                            "Value": c["display_value"],
                            "Contribution": c["contribution"],
                        }
                        for c in items
                    ]
                ).style.format({"Contribution": "{:+.4f}"}),
                hide_index=True,
                width="stretch",
            )
        else:
            col.caption("No feature pushes in this direction.")
    n_background = exp["explanation_method"]["background_size"]
    with st.expander("How to read this explanation"):
        st.markdown(
            "- Contributions are **SHAP values** computed on the frozen Random Forest's fraud "
            "probability. They always add up: baseline + all contributions = fraud score.\n"
            f"- The **baseline** is the model's average score over {n_background}"
            " randomly sampled *training* transactions, i.e. a typical transaction.\n"
            "- **V1–V28 are anonymized PCA components.** The data provider did not disclose "
            "what they represent, so they are shown by name only and not interpreted.\n"
            "- **Time** affects the model only through the hour of day derived from it "
            "(relative to the start of the recording). **Amount** enters on a log scale.\n"
            "- An explanation shows how the *model* used these values for this transaction. "
            "It is not proof of fraud and not a cause-and-effect statement."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": c["label"],
                        "Value": c["display_value"],
                        "Contribution": c["contribution"],
                    }
                    for c in contributions
                ]
            ).style.format({"Contribution": "{:+.5f}"}),
            hide_index=True,
            width="stretch",
        )


def fetch_explanation(api: ApiClient, payload: dict) -> tuple[dict | None, str | None]:
    """Ask the API for a SHAP explanation. Never raises: returns (explanation, problem).

    Explanations are optional extras on top of a prediction that has already been
    shown, so every failure becomes a message instead of crashing the page. This
    lives in app.py on purpose: Streamlit re-reads this file on every rerun, so the
    guard keeps working even when a long-running process holds an older ApiClient.
    """
    explain = getattr(api, "explain", None)
    if not callable(explain):
        return None, (
            "This dashboard process is running an older API client without explanation "
            "support. Restart the dashboard (`make dashboard`) to load the current code."
        )
    try:
        return explain(payload, top_k=EXPLAIN_TOP_K), None
    except ApiError as exc:
        if exc.status == 404:
            return None, (
                "The prediction API has no `/explain` endpoint (it was started before "
                "explanations were added). Restart it with `make api`."
            )
        if exc.status == 503:
            return None, f"Explanations are currently unavailable: {exc.message}"
        return None, f"Explanation not available: {exc.message}"
    except Exception as exc:  # anything else must not take the page down
        return (
            None,
            f"Explanation failed unexpectedly ({type(exc).__name__}); the prediction "
            "above is unaffected.",
        )


def show_explanation(explanation: dict | None, problem: str | None) -> None:
    """Render an explanation, or the reason there is none. Never raises."""
    if problem:
        st.info(problem)
        return
    if not explanation:
        return
    try:
        render_explanation(explanation)
    except Exception as exc:  # e.g. a response from an incompatible API version
        st.warning(
            f"The explanation could not be displayed ({type(exc).__name__}: {exc}). "
            "The prediction above is unaffected."
        )


with tab_single:
    examples = training_examples()
    if "tx" not in st.session_state:
        st.session_state.tx = (examples or {}).get("legitimate") or {
            c: 0.0 for c in FEATURE_COLUMNS
        }

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Transaction")
        if examples:
            b1, b2, _ = st.columns([1, 1, 1])
            if b1.button("Legitimate example", width="stretch"):
                st.session_state.tx = examples["legitimate"]
            if b2.button("Fraud example", width="stretch"):
                st.session_state.tx = examples["fraud"]
            st.caption("Examples come from the training data the model learned from.")
        tx = dict(st.session_state.tx)
        with st.form("single"):
            c1, c2 = st.columns(2)
            tx[AMOUNT_COLUMN] = c1.number_input(
                "Amount", min_value=0.0, value=float(tx[AMOUNT_COLUMN]), step=1.0, format="%.2f"
            )
            tx[TIME_COLUMN] = c2.number_input(
                "Time (seconds since first transaction)",
                min_value=0.0,
                value=float(tx[TIME_COLUMN]),
                step=60.0,
                format="%.0f",
            )
            with st.expander("Anonymised features V1–V28", expanded=False):
                cols = st.columns(4)
                for i, name in enumerate(PCA_FEATURES):
                    tx[name] = cols[i % 4].number_input(name, value=float(tx[name]), format="%.6f")
            tx_id = st.text_input("Transaction ID (optional)", value="", max_chars=64)
            submitted = st.form_submit_button("Score transaction", type="primary")

    with right:
        st.subheader("Result")
        if submitted:
            payload = {**tx, **({"transaction_id": tx_id} if tx_id else {})}
            st.session_state.single_explanation = None  # never show a previous transaction's
            try:
                result = api.predict(payload)
            except ApiError as exc:
                st.error(f"Could not score this transaction: {exc.message}")
            else:
                flagged = result["decision"] == "REVIEW"
                (st.error if flagged else st.success)(
                    f"**{result['decision']}** — "
                    + (
                        "send to the fraud team for review"
                        if flagged
                        else "let the payment through"
                    )
                )
                m1, m2 = st.columns(2)
                m1.metric("Fraud score", f"{result['fraud_score']:.4f}")
                m2.metric("Review threshold", f"{result['model']['threshold']:.4f}")
                st.altair_chart(
                    score_bar(result["fraud_score"], result["model"]["threshold"]),
                    width="stretch",
                )
                st.caption("Dashed line = review threshold. Scores at or above it are reviewed.")
                st.caption(
                    f"Model `{result['model']['name']}` · version "
                    f"{result['model']['version'] or '—'} · source {result['model']['source']}"
                )
                for warning in result["warnings"]:
                    st.warning(warning)
                st.session_state.single_explanation = fetch_explanation(api, payload)
        else:
            st.info("Enter or load a transaction and press **Score transaction**.")
    if submitted and st.session_state.get("single_explanation"):
        show_explanation(*st.session_state.single_explanation)


# ------------------------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Score a CSV file")
    st.markdown(
        "Upload a CSV with the columns `Time`, `V1`…`V28`, `Amount` "
        "(an optional `transaction_id` column is echoed back; a `Class` column is ignored). "
        f"Up to {MAX_UPLOAD_ROWS:,} rows; they are sent to the API in batches of "
        f"{max_batch:,}."
    )
    upload = st.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        cached = st.session_state.get("batch")
        try:
            if cached is None or cached["file_id"] != upload.file_id:
                frame = read_csv_upload(upload.getvalue(), MAX_UPLOAD_ROWS)
                with st.spinner(f"Scoring {len(frame):,} transactions..."):
                    results = api.predict_frame(frame, batch_size=max_batch)
                # Keep the validated features so any row can be explained without re-scoring.
                st.session_state.batch = cached = {
                    "file_id": upload.file_id,
                    "features": prepare_batch(frame),
                    "results": results,
                }
        except ValueError as exc:
            st.error(f"File rejected: {exc}")
        except ApiError as exc:
            st.error(f"Scoring failed: {exc.message}")
        else:
            results, features = cached["results"], cached["features"]
            flagged = int(results["is_fraud"].sum())
            k1, k2, k3 = st.columns(3)
            k1.metric("Transactions scored", f"{len(results):,}")
            k2.metric("Sent for review", f"{flagged:,}")
            k3.metric("Review rate", pct(flagged / len(results), 2))
            view = results.sort_values("fraud_score", ascending=False)
            st.dataframe(
                view,
                width="stretch",
                hide_index=True,
                column_config={
                    "fraud_score": st.column_config.ProgressColumn(
                        "Fraud score", min_value=0.0, max_value=1.0, format="%.4f"
                    ),
                    "is_fraud": st.column_config.CheckboxColumn("Flagged"),
                },
            )
            st.download_button(
                "Download results (CSV)",
                view.to_csv(index=False).encode(),
                file_name="fraud_scores.csv",
                mime="text/csv",
            )

            st.markdown("#### Explain a transaction from this file")
            position = {tid: i for i, tid in enumerate(features["transaction_id"])}
            choice = st.selectbox(
                "Transaction (highest scores first)",
                view["transaction_id"].tolist(),
                format_func=lambda tid: (
                    f"{tid} — score {results.loc[position[tid], 'fraud_score']:.4f} "
                    f"({results.loc[position[tid], 'decision']})"
                ),
            )
            if st.button("Explain selected transaction"):
                row = features.iloc[position[choice]].to_dict()
                show_explanation(*fetch_explanation(api, row))


# ------------------------------------------------------------------------- performance
def confusion_chart(cm: dict) -> alt.Chart:
    cells = pd.DataFrame(
        [
            {
                "actual": "Legitimate",
                "predicted": "Approve",
                "count": cm["tn"],
                "what": "Correctly approved",
            },
            {
                "actual": "Legitimate",
                "predicted": "Review",
                "count": cm["fp"],
                "what": "False alarm",
            },
            {"actual": "Fraud", "predicted": "Approve", "count": cm["fn"], "what": "Missed fraud"},
            {"actual": "Fraud", "predicted": "Review", "count": cm["tp"], "what": "Fraud caught"},
        ]
    )
    cells["row_share"] = cells["count"] / cells.groupby("actual")["count"].transform("sum")
    cells["label"] = cells.apply(lambda r: f"{r['count']:,}\n{r['what']}", axis=1)
    base = alt.Chart(cells).encode(
        x=alt.X(
            "predicted:N",
            title="Model decision",
            sort=["Approve", "Review"],
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("actual:N", title="Actual", sort=["Legitimate", "Fraud"]),
    )
    heat = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "row_share:Q", scale=alt.Scale(range=["#eef4fc", "#104281"], domain=[0, 1]), legend=None
        ),
        tooltip=[
            alt.Tooltip("what:N", title="Cell"),
            alt.Tooltip("count:Q", format=","),
            alt.Tooltip("row_share:Q", format=".1%", title="Share of actual class"),
        ],
    )
    text = base.mark_text(fontSize=14, lineBreak="\n").encode(
        text="label:N",
        color=alt.condition("datum.row_share > 0.55", alt.value("white"), alt.value("#0b0b0b")),
    )
    return (heat + text).properties(height=260)


def cost_chart(business: dict) -> alt.Chart:
    """Stacked horizontal bars: total cost with no model vs with the model."""
    data = pd.DataFrame(
        [
            {
                "scenario": "No model",
                "part": "Missed fraud",
                "cost": business["cost_without_model"],
                "rank": 0,
            },
            {
                "scenario": "With model",
                "part": "Missed fraud",
                "cost": business["cost_missed_fraud"],
                "rank": 0,
            },
            {
                "scenario": "With model",
                "part": "False alarms",
                "cost": business["cost_false_alarms"],
                "rank": 1,
            },
        ]
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=2)
        .encode(
            y=alt.Y(
                "scenario:N",
                title=None,
                sort=["No model", "With model"],
                scale=alt.Scale(paddingInner=0.35),
            ),
            x=alt.X("sum(cost):Q", title="Cost ($)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color(
                "part:N",
                scale=alt.Scale(domain=["Missed fraud", "False alarms"], range=[ORANGE, BLUE]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            order=alt.Order("rank:Q"),
            tooltip=["scenario", "part", alt.Tooltip("cost:Q", format="$,.0f")],
        )
        # A fixed pixel height + bottom legend collapses the bands in Streamlit;
        # sizing per category (Step) renders reliably.
        .properties(height=alt.Step(40))
    )


with tab_perf:
    try:
        perf = api.performance()
    except ApiError as exc:
        st.error(f"Could not load evaluation results: {exc.message}")
        st.stop()

    th = perf["threshold"]
    views = {}
    if perf.get("test"):
        views["Held-out test (final estimate)"] = perf["test"]
    views["Validation (used for selection)"] = perf["validation"]
    if perf.get("holdout"):
        views["Recent labelled holdout (retraining comparison)"] = perf["holdout"]
    if not perf.get("test"):
        st.info(
            "This champion is a retrained candidate. By design it was never evaluated on the "
            "frozen test set (the test split is used only once, in Phase 2); its validation "
            "and recent-holdout results are shown instead."
        )
    split = st.radio("Results for", list(views), horizontal=True)
    res = views[split]
    m, ci = res["metrics"], res["ci95"]

    def ci_help(key: str) -> str | None:
        return (
            f"95% confidence interval: {ci[key][0]:.3f} – {ci[key][1]:.3f}" if key in ci else None
        )

    st.caption(
        f"Recorded results of release `{perf['snapshot']}` ({perf['model']}, threshold "
        f"{th['value']:.4f} tuned on validation). Nothing is recomputed here. "
        f"{res['n_transactions']:,} transactions, {res['n_fraud']} frauds."
    )
    k = st.columns(5)
    k[0].metric("PR-AUC", f"{m['pr_auc']:.3f}", help=ci_help("pr_auc"))
    k[1].metric("Precision", pct(m["precision"]), help=ci_help("precision"))
    k[2].metric("Recall", pct(m["recall"]), help=ci_help("recall"))
    k[3].metric("F1", f"{m['f1']:.3f}")
    k[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}", help=ci_help("roc_auc"))

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        st.altair_chart(confusion_chart(res["confusion_matrix"]), width="stretch")
        cm = res["confusion_matrix"]
        st.caption(
            f"Caught {cm['tp']} of {cm['tp'] + cm['fn']} frauds; "
            f"{cm['fp']} false alarms among {cm['tn'] + cm['fp']:,} legitimate transactions."
        )
    with right:
        b = res["business"]
        st.markdown(
            f"**Business cost** (assumed ${th['cost_false_negative']:,.0f} per missed "
            f"fraud, ${th['cost_false_positive']:,.0f} per false alarm)"
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost with model", f"${b['total_cost']:,.0f}")
        c2.metric("Cost without model", f"${b['cost_without_model']:,.0f}")
        c3.metric(
            "Savings",
            f"${b['savings']:,.0f}",
            f"-{pct(b['cost_reduction'], 0)} cost",
            delta_color="inverse",
        )
        st.altair_chart(cost_chart(b), width="stretch")
        st.caption(
            f"{b['missed_fraud']} missed frauds × ${th['cost_false_negative']:,.0f} + "
            f"{b['false_alarms']} false alarms × ${th['cost_false_positive']:,.0f} = "
            f"${b['total_cost']:,.0f}"
        )

    gap = perf.get("validation_vs_test")
    if gap and perf.get("test"):
        st.markdown("**Validation → test**")
        rows = [
            {
                "Metric": name,
                "Validation": perf["validation"]["metrics"][key],
                "Test": perf["test"]["metrics"][key],
                "Change": gap["difference"][key],
            }
            for name, key in (
                ("PR-AUC", "pr_auc"),
                ("Precision", "precision"),
                ("Recall", "recall"),
                ("F1", "f1"),
            )
        ]
        st.dataframe(
            pd.DataFrame(rows).style.format(
                {"Validation": "{:.3f}", "Test": "{:.3f}", "Change": "{:+.3f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        p = gap["fisher_exact_p_values"]
        lo, hi = gap["bootstrap_ci95_difference"]["pr_auc"]
        st.info(
            (
                "At least one difference is statistically significant. "
                if gap["any_significant"]
                else "Test is lower on every metric, but **no difference is statistically "
                "significant** at the 5% level "
            )
            + f"(PR-AUC change 95% CI {lo:+.3f} to {hi:+.3f}; recall p = {p['recall']:.3f}). "
            "The test numbers are the honest estimate."
        )

    if perf["model_comparison"]:
        with st.expander("Model comparison on validation (how the model was chosen)"):
            comp = pd.DataFrame(perf["model_comparison"])[
                [
                    "model",
                    "selected",
                    "val_pr_auc",
                    "val_roc_auc",
                    "threshold",
                    "precision",
                    "recall",
                    "f1",
                    "tp",
                    "fp",
                    "fn",
                    "expected_cost",
                ]
            ]
            st.dataframe(
                comp.style.format(
                    {
                        c: "{:.3f}"
                        for c in (
                            "val_pr_auc",
                            "val_roc_auc",
                            "threshold",
                            "precision",
                            "recall",
                            "f1",
                        )
                    }
                    | {"expected_cost": "${:,.0f}"}
                ),
                hide_index=True,
                width="stretch",
            )
    else:
        st.caption(
            "No multi-model comparison for this release: a retrained candidate reuses "
            "the champion's recipe and only changes the training data."
        )

    with st.expander("Evaluation charts") if perf["figures"] else st.container():
        cols = st.columns(2)
        for i, name in enumerate(perf["figures"]):
            try:
                cols[i % 2].image(
                    api.figure(name), caption=name.replace("_", " ")[:-4], width="stretch"
                )
            except ApiError as exc:
                cols[i % 2].warning(f"{name}: {exc.message}")


# ----------------------------------------------------------------------------- model details
with tab_model:
    info = api.model_info()
    st.subheader(f"{info['display_name']} — approved release `{info['frozen_snapshot']}`")
    st.markdown(
        "- The model, its preprocessing pipeline and the decision threshold are **frozen**: the "
        "API verifies the registry version against the release's SHA-256 checksum before "
        "serving and refuses to start scoring if anything differs.\n"
        "- The threshold was chosen on **validation** data by lowest expected cost; the test set "
        "was used once, only to report final performance.\n"
        "- The service never retrains or re-tunes the model."
    )
    st.json(
        {
            k: info[k]
            for k in (
                "registered_model",
                "version",
                "alias",
                "threshold",
                "source",
                "run_id",
                "data_version",
                "detector_sha256",
                "loaded_at",
                "max_batch_size",
            )
        }
    )
