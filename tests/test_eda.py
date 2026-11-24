import json

from fraud_detection.analysis.eda import (
    compute_eda_summary,
    generate_eda_figures,
    write_eda_report,
)


def test_eda_summary_values(synthetic_df):
    summary = compute_eda_summary(synthetic_df)
    cd = summary["class_distribution"]
    assert cd["fraud"] + cd["legitimate"] == len(synthetic_df)
    assert cd["fraud"] == int(synthetic_df["Class"].sum())
    assert summary["missing_values"]["total"] == 0
    assert summary["duplicates"]["duplicate_rows"] == 0
    assert len(summary["top_correlated_features"]) == 10
    # The synthetic data makes V1-V5 informative, so they should rank highest.
    assert set(list(summary["top_correlated_features"])[:5]) == {"V1", "V2", "V3", "V4", "V5"}
    json.dumps(summary)  # must be JSON-serialisable


def test_eda_figures_and_report(tmp_path, synthetic_df):
    figures = generate_eda_figures(synthetic_df, tmp_path / "figures")
    assert len(figures) == 5
    assert all(p.is_file() and p.stat().st_size > 0 for p in figures)
    report = write_eda_report(compute_eda_summary(synthetic_df), figures, tmp_path)
    assert "Fraud cases" in report.read_text()
    assert (tmp_path / "eda_summary.json").is_file()
