"""Monitoring dashboard (Streamlit) - read-only view of monitoring and governance.

Run with:  make monitoring-dashboard        (or fraud-detection monitoring-dashboard)
Sandbox:   fraud-detection --config artifacts/demo/config.yaml monitoring-dashboard

Reads the local monitoring store and the MLflow registry. It never changes
anything: promotions and rollbacks require the CLI with an explicit approval.
"""

from __future__ import annotations

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fraud_detection.config import PROJECT_ROOT, load_config
from fraud_detection.monitoring.store import STORE_FILE, MonitoringStore

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SEVERITY_COLORS = {"critical": "#c0392b", "warning": "#b7791f", "info": "#52514e"}
STATUS_ICON = {"ok": "🟢 ok", "warning": "🟠 warning", "critical": "🔴 critical"}

st.set_page_config(page_title="Fraud Monitoring", page_icon="📈", layout="wide")


def display_path(path: Path) -> str:
    """Show paths relative to the project, so local folder names are not displayed."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return Path(path).name


@st.cache_resource
def get_config():
    return load_config(os.environ.get("FRAUD_CONFIG"))


cfg = get_config()
store_path = Path(cfg.paths.monitoring_dir) / STORE_FILE

st.title("Fraud Model Monitoring")
if not store_path.is_file():
    st.info(
        "No monitoring data yet. Score some traffic through the API, or ingest a batch:\n\n"
        "```bash\nfraud-detection simulate-batch --scenario stable\n"
        "fraud-detection ingest-batch --file monitoring/incoming/stable-day1.csv\n"
        "fraud-detection monitor\n```\n\nOr run the sandbox demo: `make monitoring-demo`."
    )
    st.stop()

store = MonitoringStore(cfg.paths.monitoring_dir)
runs = store.runs()
alerts = store.alerts()


def registry():
    try:
        from fraud_detection.retraining.governance import registry_status

        return registry_status(cfg), None
    except Exception as exc:  # registry missing or unreadable: show monitoring anyway
        return None, f"{type(exc).__name__}: {exc}"


reg, reg_error = registry()

# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📈 Monitoring")
    st.caption(f"Store: `{display_path(store_path)}`")
    if reg:
        champ = reg["aliases"].get(cfg.mlflow.champion_alias)
        info = next((v for v in reg["versions"] if v["version"] == champ), {})
        st.markdown(
            f"**Champion:** `{reg['name']}` v{champ}  \n"
            f"**Approval:** {info.get('approval', '—')}  \n"
            f"**Release:** `{info.get('snapshot', '—')}`  \n"
            f"**Threshold:** `{float(info['threshold']):.4f}`"
            if info.get("threshold")
            else ""
        )
        prev = reg["aliases"].get("previous-champion")
        if prev:
            st.caption(f"Previous champion (rollback target): v{prev}")
    else:
        st.warning(f"Registry unavailable: {reg_error}")
    st.divider()
    st.caption(
        "Read-only. Promotions and rollbacks require `fraud-detection promote` / `rollback` "
        "with a named approver, a reason and an explicit confirmation."
    )

if runs.empty:
    st.info("Predictions are stored but no monitoring run exists yet: `fraud-detection monitor`.")
    st.stop()

# Flatten run summaries for charts.
rows = []
for order, r in enumerate(runs.itertuples()):
    s = r.summary
    perf = s["performance"]
    rows.append(
        {
            "order": order,
            "run": f"{order + 1}. {s['batch_id']}",
            "run_no": order + 1,
            "batch_id": s["batch_id"],
            "created_at": s["created_at"],
            "model_version": str(s["model_version"]),
            "status": s["status"],
            "n_rows": s["n_rows"],
            "drift_share": s["drift"]["drift_share"],
            "score_drift": s["drift"]["score_drift"]["value"],
            "alert_rate": s["scores"]["alert_rate"],
            "expected_alert_rate": s["scores"]["expected_alert_rate"],
            "rejected_share": s["quality"]["rejected_share"],
            "out_of_range_share": s["quality"]["out_of_range_share"],
            "duplicate_share": s["quality"]["duplicate_share"],
            "recall": perf.get("recall"),
            "precision": perf.get("precision"),
            "pr_auc": perf.get("pr_auc"),
            "n_alerts": len(s["alerts"]),
        }
    )
history = pd.DataFrame(rows)
order = history["run"].tolist()
thr = cfg.monitoring.alerts

latest = history.iloc[-1]
k = st.columns(5)
k[0].metric("Latest status", STATUS_ICON[latest["status"]])
k[1].metric("Latest run", f"#{int(latest['run_no'])}", help=latest["batch_id"])
k[2].metric("Alerts (latest run)", int(latest["n_alerts"]))
k[3].metric("Monitoring runs", len(history))
k[4].metric("Model version (latest run)", f"v{latest['model_version']}")
st.caption(f"Latest batch: `{latest['batch_id']}` · monitored {latest['created_at']}")

tab_over, tab_alerts, tab_drift, tab_models = st.tabs(
    ["Over time", "Alerts", "Drift report", "Models & approvals"]
)


def line_chart(
    data: pd.DataFrame,
    series: dict[str, str],
    title: str,
    fmt: str,
    rules: list[tuple[float, str]] = (),
) -> alt.LayerChart:
    long = data.melt(
        id_vars=["run_no", "batch_id", "model_version"],
        value_vars=list(series),
        var_name="metric",
        value_name="value",
    ).dropna()
    long["metric"] = long["metric"].map(series)
    names = list(series.values())
    base = alt.Chart(long).encode(
        x=alt.X("run_no:O", title="Monitoring run #", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format=fmt)),
        color=alt.Color(
            "metric:N",
            scale=alt.Scale(domain=names, range=[BLUE, ORANGE, AQUA][: len(names)]),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=[
            alt.Tooltip("run_no:O", title="run"),
            alt.Tooltip("batch_id:N", title="batch"),
            "metric",
            alt.Tooltip("value:Q", format=fmt),
            alt.Tooltip("model_version:N", title="model version"),
        ],
    )
    layers = [base.mark_line(strokeWidth=2), base.mark_point(size=60, filled=True)]
    for value, label in rules:
        rule = pd.DataFrame({"y": [value], "label": [label]})
        layers.append(
            alt.Chart(rule)
            .mark_rule(color=GREY, strokeDash=[4, 3])
            .encode(y="y:Q", tooltip=["label"])
        )
    return alt.layer(*layers).properties(title=title, height=300)


with tab_over:
    st.caption(
        "Each point is one monitoring run (one batch). Dashed lines are alert limits "
        "from `monitoring.alerts` in the config."
    )
    c1, c2 = st.columns(2)
    c1.altair_chart(
        line_chart(
            history,
            {"drift_share": "Share of features drifted"},
            "Data drift (Evidently)",
            ".0%",
            [(thr.drift_share_warning, "warning"), (thr.drift_share_critical, "critical")],
        ),
        width="stretch",
    )
    c2.altair_chart(
        line_chart(
            history,
            {"alert_rate": "Review rate", "expected_alert_rate": "Expected (validation)"},
            "Share of transactions sent for review",
            ".2%",
        ),
        width="stretch",
    )
    c3, c4 = st.columns(2)
    perf = history.dropna(subset=["recall"])
    if perf.empty:
        c3.info("No batch has enough labelled frauds to evaluate performance yet.")
    else:
        c3.altair_chart(
            line_chart(
                history,
                {"recall": "Recall", "precision": "Precision", "pr_auc": "PR-AUC"},
                "Performance on labelled transactions",
                ".2f",
                [(thr.min_recall, "minimum recall")],
            ),
            width="stretch",
        )
    c4.altair_chart(
        line_chart(
            history,
            {
                "rejected_share": "Rejected rows",
                "out_of_range_share": "Outside training range",
                "duplicate_share": "Duplicates",
            },
            "Data quality",
            ".1%",
        ),
        width="stretch",
    )
    st.dataframe(
        history[
            [
                "run",
                "created_at",
                "model_version",
                "status",
                "n_rows",
                "drift_share",
                "alert_rate",
                "recall",
                "precision",
                "n_alerts",
            ]
        ].style.format(
            {
                "drift_share": "{:.0%}",
                "alert_rate": "{:.2%}",
                "recall": "{:.3f}",
                "precision": "{:.3f}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )

with tab_alerts:
    if alerts.empty:
        st.success("No alerts raised so far.")
    else:
        severities = st.multiselect(
            "Severity", ["critical", "warning", "info"], default=["critical", "warning"]
        )
        view = alerts[alerts["severity"].isin(severities)].sort_values("id", ascending=False)
        counts = alerts.groupby("severity").size()
        a1, a2, a3 = st.columns(3)
        a1.metric("Critical", int(counts.get("critical", 0)))
        a2.metric("Warning", int(counts.get("warning", 0)))
        a3.metric("Info", int(counts.get("info", 0)))
        st.dataframe(
            view[
                [
                    "created_at",
                    "batch_id",
                    "severity",
                    "category",
                    "metric",
                    "value",
                    "threshold",
                    "message",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "value": st.column_config.NumberColumn(format="%.4f"),
                "threshold": st.column_config.NumberColumn(format="%.4f"),
            },
        )

with tab_drift:
    choice = st.selectbox("Monitoring run", list(reversed(order)))
    summary = runs.iloc[order.index(choice)]["summary"]
    d, q, sc = summary["drift"], summary["quality"], summary["scores"]
    m = st.columns(4)
    m[0].metric("Features drifted", f"{d['n_drifted']} / {len(d['columns'])}")
    m[1].metric(
        "Score drift " + ("(detected)" if d["score_drift"]["detected"] else "(none)"),
        f"{d['score_drift']['value']:.3f}",
        help=f"{d['score_drift']['method']}, threshold {d['score_drift']['threshold']}",
    )
    m[2].metric(
        f"Review rate (expected {sc['expected_alert_rate']:.2%})", f"{sc['alert_rate']:.2%}"
    )
    m[3].metric("Status", STATUS_ICON[summary["status"]])

    cols = pd.DataFrame(
        [
            {
                "feature": c,
                **v,
                "ratio": v["value"] / v["threshold"]
                if "p_value" not in v["method"]
                else v["threshold"] / max(v["value"], 1e-12),
            }
            for c, v in d["columns"].items()
        ]
    ).sort_values("ratio", ascending=False)
    cols["drift"] = cols["detected"].map({True: "Drift detected", False: "No drift"})
    bars = (
        alt.Chart(cols)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("feature:N", sort=cols["feature"].tolist(), title=None),
            x=alt.X("ratio:Q", title="Drift statistic / threshold (≥ 1 means drift)"),
            color=alt.Color(
                "drift:N",
                scale=alt.Scale(domain=["Drift detected", "No drift"], range=[ORANGE, BLUE]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                "feature",
                "method",
                alt.Tooltip("value:Q", format=".4f"),
                alt.Tooltip("threshold:Q", format=".3f"),
                "drift",
            ],
        )
    )
    one = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(color="#0b0b0b", strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    st.altair_chart((bars + one).properties(height=alt.Step(20)), width="stretch")
    st.caption(
        "Monitored: hour of day (derived from Time), Amount and V1–V28 (anonymized PCA "
        "components). Raw Time is not monitored because it always increases. Reference: "
        f"{summary['reference']['reference_size']:,} training transactions (seed "
        f"{summary['reference']['random_state']}); score reference: validation split."
    )

    with st.expander("Data quality details"):
        st.json(q)
    with st.expander("Performance details"):
        st.json(summary["performance"])
    report_dir = Path(summary["report_dir"])
    for name, label in (
        ("data_drift.html", "Evidently data-drift report"),
        ("score_drift.html", "Evidently score-drift report"),
    ):
        path = report_dir / name
        with st.expander(label):
            if path.is_file():
                st.iframe(path, height=900)  # generated locally by Evidently
            else:
                st.warning(f"{path} not found")

with tab_models:
    if not reg:
        st.warning(f"Registry unavailable: {reg_error}")
    else:
        versions = pd.DataFrame(reg["versions"])
        versions["aliases"] = versions["aliases"].map(lambda a: ", ".join(a) or "—")
        st.markdown("**Registered versions**")
        st.dataframe(
            versions[
                [
                    "version",
                    "aliases",
                    "approval",
                    "source",
                    "snapshot",
                    "threshold",
                    "approved_by",
                    "approved_at",
                    "validation_pr_auc",
                    "test_pr_auc",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        pending = versions[versions["approval"] == "pending"]
        for row in pending.itertuples():
            report = cfg.paths.frozen_dir / row.snapshot / "reports" / "comparison.md"
            with st.expander(f"Pending approval: v{row.version} ({row.snapshot})", expanded=True):
                if report.is_file():
                    st.markdown(report.read_text())
                st.code(
                    f"fraud-detection promote --version {row.version} "
                    f'--approved-by "Your Name" --reason "..."',
                    language="bash",
                )
        st.markdown("**Governance log**")
        events = pd.DataFrame(reg["events"])
        if events.empty:
            st.caption("No governance events yet.")
        else:
            st.dataframe(
                events.sort_values("id", ascending=False)[
                    ["created_at", "action", "from_version", "to_version", "actor", "reason"]
                ],
                hide_index=True,
                width="stretch",
            )
        st.caption(
            'Rollback: `fraud-detection rollback --approved-by "Your Name" --reason '
            '"..."` (moves the champion to the previous approved version).'
        )
