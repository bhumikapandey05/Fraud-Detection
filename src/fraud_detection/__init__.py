"""End-to-end credit card fraud detection project."""

import os

# MLflow prints an assistant "hint" on import; it is noise in CLI / API logs.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

__version__ = "0.1.0"
