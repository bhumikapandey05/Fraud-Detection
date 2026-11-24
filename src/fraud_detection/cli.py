"""Command-line interface: ``fraud-detection <command>``.

Commands:
    download   Fetch the Kaggle dataset into data/raw (or print instructions).
    validate   Run data validation checks on the raw CSV.
    eda        Generate the EDA report and figures.
    prepare    Clean, split, validate and fit the preprocessor.
    verify     Re-check the prepared splits (counts, schema, leakage).
    train      Train, tune, select and test models; save the best detector.
    evaluate   Re-score the saved detector on the test split (reproducibility check).
    gap-report Document the validation -> test change for a saved model (read-only).
    freeze     Save the current model as an immutable, checksummed snapshot.
    verify-frozen  Check a snapshot's files, data and test metrics are unchanged.
    track-frozen   Log a frozen snapshot to MLflow and register it (no retraining).
    registry   List registered model versions and aliases.
    serve      Start the FastAPI prediction service.
    dashboard  Start the Streamlit dashboard.

Phase 5 - monitoring and controlled retraining:
    simulate-batch        Write a synthetic batch (+ labels) generated from the training split.
    ingest-batch          Validate + score a CSV with the champion and store it for monitoring.
    add-labels            Attach confirmed outcomes (transaction_id,label) to stored predictions.
    monitor               Run drift / quality / performance monitoring for a batch.
    retrain-candidate     Train a candidate on permitted data and compare it with the champion.
    promote               Promote an eligible candidate (explicit approval required).
    rollback              Move the champion back to an approved version (approval required).
    governance            Registry versions, approval states, aliases and the event log.
    monitoring-dashboard  Start the Streamlit monitoring dashboard.
    monitoring-demo       Run the whole lifecycle in an isolated sandbox.

Setup:
    bootstrap             Fresh clone -> data, splits, model, frozen release, champion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fraud_detection.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from fraud_detection.logging_utils import setup_logging

DEFAULT_SNAPSHOT = "phase2-rf-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud-detection",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download the dataset")
    dl.add_argument("--source", choices=["kaggle", "mirror"], default="kaggle")
    dl.add_argument("--force", action="store_true", help="Re-download even if present")

    sub.add_parser("validate", help="Validate the raw dataset")

    eda = sub.add_parser("eda", help="Run exploratory data analysis")
    eda.add_argument(
        "--on",
        choices=["raw", "train"],
        default="raw",
        help="Analyse the full raw file or only the training split",
    )

    sub.add_parser("prepare", help="Run the data preparation pipeline")
    sub.add_parser("verify", help="Verify prepared splits before training")
    train = sub.add_parser("train", help="Train and evaluate models, save the best one")
    train.add_argument("--no-mlflow", action="store_true", help="Do not log this run to MLflow")
    train.add_argument(
        "--register",
        action="store_true",
        help="Also register the result as a new model version (no alias change)",
    )

    evaluate = sub.add_parser("evaluate", help="Re-evaluate the saved model on the test split")
    evaluate.add_argument(
        "--model-dir",
        default=None,
        help="Model folder (default: working model; e.g. a frozen snapshot)",
    )

    gap = sub.add_parser("gap-report", help="Validation vs test analysis (read-only)")
    gap.add_argument("--model-dir", default=None)

    for cmd, text in (
        ("freeze", "Freeze the current model"),
        ("verify-frozen", "Verify a frozen snapshot"),
        ("track-frozen", "Log + register a frozen snapshot in MLflow"),
    ):
        p = sub.add_parser(cmd, help=text)
        p.add_argument("--name", default=DEFAULT_SNAPSHOT, help="Snapshot name")
    sub.choices["track-frozen"].add_argument(
        "--no-register", action="store_true", help="Log the run but do not register it"
    )

    sub.add_parser("registry", help="List registered model versions")

    serve = sub.add_parser("serve", help="Start the prediction API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    dash = sub.add_parser("dashboard", help="Start the Streamlit dashboard")
    dash.add_argument("--port", type=int, default=8501)
    dash.add_argument("--api-url", default=None, help="Prediction API base URL")
    dash.add_argument(
        "--address", default="127.0.0.1", help="Interface to listen on (0.0.0.0 inside Docker)"
    )

    boot = sub.add_parser("bootstrap", help="Fresh clone -> data, model, champion")
    boot.add_argument("--source", choices=["mirror", "kaggle"], default="mirror")

    # --- Phase 5 -------------------------------------------------------------------------
    from fraud_detection.monitoring.simulate import SCENARIOS

    sim = sub.add_parser("simulate-batch", help="Write a synthetic monitoring batch")
    sim.add_argument("--scenario", choices=sorted(SCENARIOS), default="stable")
    sim.add_argument("--rows", type=int, default=20000)
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument("--day", type=int, default=1)
    sim.add_argument("--out-dir", default=None, help="Default: <monitoring_dir>/incoming")

    ing = sub.add_parser("ingest-batch", help="Score a CSV with the champion and store it")
    ing.add_argument("--file", required=True)
    ing.add_argument("--batch-id", default=None)

    lab = sub.add_parser("add-labels", help="Attach confirmed outcomes to stored predictions")
    lab.add_argument("--file", required=True, help="CSV with transaction_id,label (0/1)")

    mon = sub.add_parser("monitor", help="Run monitoring for a batch (default: latest)")
    mon.add_argument("--batch-id", default=None)
    mon.add_argument("--no-mlflow", action="store_true")

    sub.add_parser("retrain-candidate", help="Train + compare a candidate (never uses test)")

    for cmd in ("promote", "rollback"):
        p = sub.add_parser(cmd, help=f"{cmd.title()} the champion (explicit approval required)")
        p.add_argument("--approved-by", required=True, help="Name of the person approving")
        p.add_argument("--reason", required=True, help="Why this change is approved")
        p.add_argument(
            "--yes",
            action="store_true",
            help="Skip the interactive confirmation (approval is still recorded)",
        )
    sub.choices["promote"].add_argument(
        "--version", required=True, help="Registered candidate version to promote"
    )
    sub.choices["rollback"].add_argument(
        "--to-version", default=None, help="Default: the previous-champion alias"
    )

    sub.add_parser("governance", help="Registry status and governance events")
    mdash = sub.add_parser("monitoring-dashboard", help="Start the monitoring dashboard")
    mdash.add_argument("--port", type=int, default=8502)
    mdash.add_argument("--address", default="127.0.0.1")

    demo = sub.add_parser("monitoring-demo", help="Run the full lifecycle in a sandbox")
    demo.add_argument("--workdir", default="artifacts/demo")
    demo.add_argument("--rows", type=int, default=20000)
    return parser


def _confirm_interactively(summary: dict) -> bool:
    """Show what will change and require the exact version number to be typed."""
    print(json.dumps({k: v for k, v in summary.items() if k != "comparison"}, indent=2))
    if "comparison" in summary:
        verdict = "ELIGIBLE" if summary["comparison"]["eligible"] else "NOT ELIGIBLE"
        print(f"Comparison verdict: {verdict}")
        for check in summary["comparison"]["checks"]:
            print(
                f"  [{'pass' if check['passed'] else 'FAIL'}] {check['description']}: "
                f"candidate={check['candidate']} champion={check['champion']}"
            )
    answer = input(
        f"Type the target version ({summary['to_version']}) to approve this {summary['action']}: "
    )
    return answer.strip() == str(summary["to_version"])


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    cfg = load_config(config_path)
    logger = setup_logging(cfg.logging.level, cfg.paths.log_dir)

    try:
        if args.command == "download":
            from fraud_detection.data.download import download_dataset

            download_dataset(cfg.paths.raw_data.parent, source=args.source, force=args.force)

        elif args.command == "validate":
            from fraud_detection.data.loader import load_raw_data
            from fraud_detection.data.validation import validate_raw_data

            report = validate_raw_data(
                load_raw_data(cfg.paths.raw_data), expected_rows=cfg.dataset.expected_rows
            )
            report.log()
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.passed else 1

        elif args.command == "eda":
            from fraud_detection.pipelines.eda import run_eda

            path = run_eda(cfg, on=args.on)
            print(f"EDA report: {path}")

        elif args.command == "prepare":
            from fraud_detection.pipelines.prepare import run_prepare

            report = run_prepare(cfg)
            print(json.dumps(report["splits"], indent=2))

        elif args.command == "verify":
            from fraud_detection.data.verification import verify_prepared_data

            print(json.dumps(verify_prepared_data(cfg).summary, indent=2))

        elif args.command == "train":
            from fraud_detection.pipelines.train import run_train

            result = run_train(cfg)
            print(json.dumps({k: result[k] for k in ("selected_model", "threshold", "report")}))
            if cfg.mlflow and cfg.mlflow.log_training_runs and not args.no_mlflow:
                from fraud_detection.pipelines.track import track_training

                tracked = track_training(cfg, config_path, register=args.register)
                print(json.dumps({"mlflow": tracked.to_dict()}))

        elif args.command == "evaluate":
            from fraud_detection.pipelines.evaluate import run_evaluate

            model_dir = Path(args.model_dir) if args.model_dir else None
            result = run_evaluate(cfg, model_dir=model_dir)
            print(json.dumps(result, indent=2, default=str))
            return 0 if result["reproducible"] else 1

        elif args.command == "gap-report":
            from fraud_detection.pipelines.freeze import run_gap_report

            model_dir = Path(args.model_dir) if args.model_dir else cfg.paths.model_dir
            result = run_gap_report(cfg, model_dir, cfg.paths.training_report_dir)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "difference_test_minus_validation",
                            "bootstrap_ci95_difference",
                            "fisher_exact_p_values",
                            "any_significant",
                        )
                    },
                    indent=2,
                )
            )

        elif args.command == "freeze":
            from fraud_detection.pipelines.freeze import freeze_model

            print(f"Frozen snapshot: {freeze_model(cfg, args.name, config_path)}")

        elif args.command == "verify-frozen":
            from fraud_detection.pipelines.freeze import verify_frozen

            print(json.dumps(verify_frozen(cfg, args.name), indent=2))

        elif args.command == "track-frozen":
            from fraud_detection.pipelines.track import track_frozen

            result = track_frozen(cfg, args.name, register=not args.no_register)
            print(json.dumps(result.to_dict(), indent=2))

        elif args.command == "registry":
            from fraud_detection.pipelines.track import registry_summary

            print(json.dumps(registry_summary(cfg), indent=2))

        elif args.command == "serve":
            import uvicorn

            from fraud_detection.serving.api import create_app

            uvicorn.run(
                create_app(cfg),
                host=args.host or cfg.serving.api_host,
                port=args.port or cfg.serving.api_port,
                log_level="warning",  # request logs come from the app's own logger
            )

        elif args.command == "dashboard":
            import os
            import subprocess

            app_path = Path(__file__).parent / "dashboard" / "app.py"
            env = {
                **os.environ,
                "FRAUD_CONFIG": str(config_path),
                "FRAUD_API_URL": args.api_url or cfg.serving.dashboard_api_url,
            }
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(app_path),
                    "--server.port",
                    str(args.port),
                    "--server.address",
                    args.address,
                ],
                env=env,
                cwd=PROJECT_ROOT,  # picks up .streamlit/config.toml (theme, privacy)
            )

        elif args.command == "simulate-batch":
            import pandas as pd

            from fraud_detection.monitoring.simulate import generate_batch

            train = pd.read_parquet(cfg.paths.processed_dir / "train.parquet")
            batch_id = f"{args.scenario}-day{args.day}"
            x, y = generate_batch(train, args.scenario, args.rows, args.seed, args.day, batch_id)
            out = Path(args.out_dir) if args.out_dir else cfg.paths.monitoring_dir / "incoming"
            out.mkdir(parents=True, exist_ok=True)
            x.to_csv(out / f"{batch_id}.csv", index=False)
            y.to_csv(out / f"{batch_id}_labels.csv", index=False)
            print(
                json.dumps(
                    {
                        "batch": str(out / f"{batch_id}.csv"),
                        "labels": str(out / f"{batch_id}_labels.csv"),
                        "rows": len(x),
                        "frauds": int(y["label"].sum()),
                    },
                    indent=2,
                )
            )

        elif args.command == "ingest-batch":
            from fraud_detection.pipelines.monitor import ingest_batch

            print(json.dumps(ingest_batch(cfg, Path(args.file), args.batch_id), indent=2))

        elif args.command == "add-labels":
            from fraud_detection.pipelines.monitor import add_labels

            print(json.dumps(add_labels(cfg, Path(args.file)), indent=2))

        elif args.command == "monitor":
            from fraud_detection.pipelines.monitor import run_monitoring

            s = run_monitoring(cfg, args.batch_id, log_to_mlflow=not args.no_mlflow)
            print(
                json.dumps(
                    {
                        "run_id": s["run_id"],
                        "batch_id": s["batch_id"],
                        "status": s["status"],
                        "alerts": s["alerts"],
                        "report_dir": s["report_dir"],
                    },
                    indent=2,
                )
            )
            return {"ok": 0, "warning": 0, "critical": 2}[s["status"]]

        elif args.command == "retrain-candidate":
            from fraud_detection.retraining.candidate import train_candidate
            from fraud_detection.retraining.governance import log_candidate

            result = train_candidate(cfg)
            logged = log_candidate(cfg, Path(result["snapshot_dir"]))
            print(
                json.dumps(
                    {
                        "candidate_id": result["candidate_id"],
                        "snapshot": result["snapshot_dir"],
                        **logged,
                        "report": str(Path(result["snapshot_dir"]) / "reports" / "comparison.md"),
                    },
                    indent=2,
                )
            )

        elif args.command in ("promote", "rollback"):
            from fraud_detection.retraining import governance

            confirm = (lambda _summary: True) if args.yes else _confirm_interactively
            if args.command == "promote":
                result = governance.promote(
                    cfg, args.version, args.approved_by, args.reason, confirm
                )
            else:
                result = governance.rollback(
                    cfg, args.approved_by, args.reason, confirm, to_version=args.to_version
                )
            print(json.dumps(result, indent=2))
            print("Restart the API (make api) to serve the new champion.")

        elif args.command == "governance":
            from fraud_detection.retraining.governance import registry_status

            print(json.dumps(registry_status(cfg), indent=2, default=str))

        elif args.command == "monitoring-dashboard":
            import os
            import subprocess

            app_path = Path(__file__).parent / "dashboard" / "monitoring_app.py"
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(app_path),
                    "--server.port",
                    str(args.port),
                    "--server.address",
                    args.address,
                ],
                env={**os.environ, "FRAUD_CONFIG": str(config_path)},
                cwd=PROJECT_ROOT,
            )

        elif args.command == "bootstrap":
            from fraud_detection.pipelines.bootstrap import run_bootstrap

            print(json.dumps(run_bootstrap(cfg, config_path, source=args.source), indent=2))

        elif args.command == "monitoring-demo":
            from fraud_detection.pipelines.demo import run_demo

            r = run_demo(Path(args.workdir), n_rows=args.rows)
            print(
                json.dumps(
                    {
                        "sandbox_config": r["config"],
                        "runs": [
                            {
                                "batch": s["batch_id"],
                                "model_version": s["model_version"],
                                "status": s["status"],
                                "alerts": [a["metric"] for a in s["alerts"]],
                            }
                            for s in r["runs"]
                        ],
                        "candidate": r["candidate"]["candidate_id"],
                        "eligible": r["candidate_logged"]["eligible"],
                        "promoted": r["promoted"],
                        "rolled_back": r["rolled_back"],
                        "aliases": r["registry"]["aliases"],
                    },
                    indent=2,
                )
            )

    except Exception as exc:  # surface a clean message instead of a traceback
        logger.error("%s failed: %s", args.command, exc)
        logger.debug("Traceback:", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
