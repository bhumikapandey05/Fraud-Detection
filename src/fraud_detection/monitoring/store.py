"""Local monitoring store (SQLite): what the model saw, what happened, what we did.

Tables
------
predictions   one row per scored transaction: features, score, decision, model version
labels        confirmed outcome per transaction_id (arrives later, e.g. from chargebacks)
ingestions    batch files loaded for scoring, with rows rejected by validation
runs          one row per monitoring run (summary JSON + report paths)
alerts        alerts raised by monitoring runs
events        governance actions: candidate registered, promotion, rollback

The store lives on the local disk only. It holds transaction features, so treat
the file like the dataset itself (it is git-ignored).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fraud_detection.data.schema import FEATURE_COLUMNS

STORE_FILE = "monitoring.db"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    source TEXT NOT NULL,
    request_id TEXT,
    transaction_id TEXT NOT NULL,
    model_version TEXT,
    threshold REAL NOT NULL,
    fraud_score REAL NOT NULL,
    is_fraud INTEGER NOT NULL,
    {", ".join(f'"{c}" REAL NOT NULL' for c in FEATURE_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS ix_predictions_batch ON predictions(batch_id);
CREATE INDEX IF NOT EXISTS ix_predictions_tx ON predictions(transaction_id);
CREATE TABLE IF NOT EXISTS labels (
    transaction_id TEXT PRIMARY KEY,
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    labeled_at TEXT NOT NULL,
    source TEXT
);
CREATE TABLE IF NOT EXISTS ingestions (
    batch_id TEXT PRIMARY KEY,
    ingested_at TEXT NOT NULL,
    file TEXT,
    n_rows INTEGER, n_valid INTEGER, n_rejected INTEGER,
    rejections TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    batch_id TEXT,
    n_rows INTEGER,
    model_version TEXT,
    status TEXT,
    summary TEXT NOT NULL,
    report_dir TEXT,
    mlflow_run_id TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    batch_id TEXT,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    threshold REAL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    registered_model TEXT,
    from_version TEXT,
    to_version TEXT,
    actor TEXT,
    reason TEXT,
    details TEXT
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class MonitoringStore:
    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / STORE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # -- predictions ---------------------------------------------------------------------
    def log_predictions(
        self,
        features: pd.DataFrame,
        transaction_ids: list[str],
        scores,
        threshold: float,
        model_version: str | None,
        batch_id: str,
        source: str,
        request_id: str | None = None,
    ) -> int:
        frame = features[FEATURE_COLUMNS].astype(float).reset_index(drop=True)
        frame.insert(0, "is_fraud", (pd.Series(scores) >= threshold).astype(int).to_numpy())
        frame.insert(0, "fraud_score", pd.Series(scores, dtype=float).to_numpy())
        frame.insert(0, "threshold", float(threshold))
        frame.insert(0, "model_version", model_version)
        frame.insert(0, "transaction_id", [str(t) for t in transaction_ids])
        frame.insert(0, "request_id", request_id)
        frame.insert(0, "source", source)
        frame.insert(0, "batch_id", batch_id)
        frame.insert(0, "received_at", now())
        with self._lock, self._connect() as con:
            frame.to_sql("predictions", con, if_exists="append", index=False)
        return len(frame)

    def predictions(self, batch_id: str | None = None, labeled_only: bool = False) -> pd.DataFrame:
        query = (
            "SELECT p.*, l.label FROM predictions p "
            "LEFT JOIN labels l ON l.transaction_id = p.transaction_id"
        )
        clauses, params = [], []
        if batch_id is not None:
            clauses.append("p.batch_id = ?")
            params.append(batch_id)
        if labeled_only:
            clauses.append("l.label IS NOT NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY p.id"
        with self._connect() as con:
            return pd.read_sql_query(query, con, params=params)

    def batches(self) -> pd.DataFrame:
        query = """
            SELECT p.batch_id, MIN(p.source) AS source, COUNT(*) AS n_rows,
                   MIN(p.received_at) AS first_seen, MAX(p.received_at) AS last_seen,
                   SUM(p.is_fraud) AS n_flagged, COUNT(l.label) AS n_labeled,
                   SUM(COALESCE(l.label, 0)) AS n_fraud_labels, MAX(p.id) AS last_id
            FROM predictions p LEFT JOIN labels l ON l.transaction_id = p.transaction_id
            GROUP BY p.batch_id ORDER BY last_id
        """
        with self._connect() as con:
            return pd.read_sql_query(query, con)

    # -- labels ----------------------------------------------------------------------------
    def add_labels(self, labels: pd.DataFrame, source: str = "feedback") -> dict:
        if not {"transaction_id", "label"} <= set(labels.columns):
            raise ValueError("Labels need columns 'transaction_id' and 'label'")
        clean = labels[["transaction_id", "label"]].dropna().copy()
        clean["transaction_id"] = clean["transaction_id"].astype(str)
        clean["label"] = clean["label"].astype(int)
        if not clean["label"].isin([0, 1]).all():
            raise ValueError("Labels must be 0 (legitimate) or 1 (fraud)")
        clean = clean.drop_duplicates("transaction_id", keep="last")
        stamp = now()
        with self._lock, self._connect() as con:
            # A temp table + join scales to any number of labels (no huge IN (...) lists).
            con.execute("CREATE TEMP TABLE incoming (transaction_id TEXT PRIMARY KEY, label INT)")
            con.executemany(
                "INSERT INTO incoming VALUES (?, ?)", list(clean.itertuples(index=False, name=None))
            )
            stored = con.execute(
                "INSERT INTO labels(transaction_id, label, labeled_at, source) "
                "SELECT i.transaction_id, i.label, ?, ? FROM incoming i "
                "WHERE EXISTS "
                "(SELECT 1 FROM predictions p WHERE p.transaction_id = i.transaction_id) "
                "ON CONFLICT(transaction_id) DO UPDATE SET label=excluded.label, "
                "labeled_at=excluded.labeled_at, source=excluded.source",
                (stamp, source),
            ).rowcount
            con.execute("DROP TABLE incoming")
        return {"received": len(clean), "stored": stored, "unmatched": len(clean) - stored}

    # -- ingestions, runs, alerts, events ------------------------------------------------------
    def record_ingestion(
        self, batch_id: str, file: str, n_rows: int, n_valid: int, rejections: dict
    ) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO ingestions VALUES (?,?,?,?,?,?,?)",
                (batch_id, now(), file, n_rows, n_valid, n_rows - n_valid, json.dumps(rejections)),
            )

    def ingestion(self, batch_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM ingestions WHERE batch_id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        keys = ("batch_id", "ingested_at", "file", "n_rows", "n_valid", "n_rejected", "rejections")
        out = dict(zip(keys, row, strict=True))
        out["rejections"] = json.loads(out["rejections"] or "{}")
        return out

    def save_run(self, run: dict, alerts: list[dict]) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run["run_id"],
                    run["created_at"],
                    run.get("batch_id"),
                    run.get("n_rows"),
                    run.get("model_version"),
                    run.get("status"),
                    json.dumps(run, default=str),
                    run.get("report_dir"),
                    run.get("mlflow_run_id"),
                ),
            )
            con.executemany(
                "INSERT INTO alerts(run_id, created_at, batch_id, severity, category, metric, "
                "value, threshold, message) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        run["run_id"],
                        run["created_at"],
                        run.get("batch_id"),
                        a["severity"],
                        a["category"],
                        a["metric"],
                        a.get("value"),
                        a.get("threshold"),
                        a["message"],
                    )
                    for a in alerts
                ],
            )

    def runs(self) -> pd.DataFrame:
        with self._connect() as con:
            frame = pd.read_sql_query("SELECT * FROM runs ORDER BY created_at, rowid", con)
        frame["summary"] = frame["summary"].map(json.loads)
        return frame

    def alerts(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query("SELECT * FROM alerts ORDER BY id", con)

    def log_event(
        self,
        action: str,
        registered_model: str | None = None,
        from_version: str | None = None,
        to_version: str | None = None,
        actor: str | None = None,
        reason: str | None = None,
        details: dict | None = None,
    ) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO events(created_at, action, registered_model, from_version, "
                "to_version, actor, reason, details) VALUES (?,?,?,?,?,?,?,?)",
                (
                    now(),
                    action,
                    registered_model,
                    from_version,
                    to_version,
                    actor,
                    reason,
                    json.dumps(details or {}, default=str),
                ),
            )

    def events(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query("SELECT * FROM events ORDER BY id", con)
