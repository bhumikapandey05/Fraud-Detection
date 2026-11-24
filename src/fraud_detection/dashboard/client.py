"""Small HTTP client for the prediction API (kept separate so it is unit-testable).

The dashboard never loads the model itself: every score comes from the API, so
what you see in the dashboard is exactly what the service would return.
"""

from __future__ import annotations

import io
import math

import httpx
import pandas as pd

from fraud_detection.data.schema import FEATURE_COLUMNS

DEFAULT_TIMEOUT = 30.0


class ApiError(RuntimeError):
    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def make_http_client(base_url: str, timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout)


class ApiClient:
    def __init__(self, http: httpx.Client) -> None:
        self.http = http

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(
                None,
                f"Cannot reach the API ({type(exc).__name__}). "
                "Is it running? Start it with `make api`.",
            ) from exc
        if response.status_code >= 400:
            try:
                body = response.json()
                detail = body.get("detail") or body.get("reason") or body
            except ValueError:
                detail = response.text[:300]
            if isinstance(detail, list):  # validation errors
                detail = "; ".join(f"{d.get('field')}: {d.get('message')}" for d in detail[:10])
            raise ApiError(response.status_code, str(detail))
        return response

    # -- status / model ------------------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health").json()

    def ready(self) -> dict:
        """Readiness payload; does not raise on 503 (the reason is the useful part)."""
        try:
            return self._request("GET", "/health/ready").json()
        except ApiError as exc:
            return {"status": "not_ready", "reason": exc.message, "http_status": exc.status}

    def model_info(self) -> dict:
        return self._request("GET", "/model").json()

    def performance(self) -> dict:
        return self._request("GET", "/model/performance").json()

    def figure(self, name: str) -> bytes:
        return self._request("GET", f"/model/performance/figures/{name}").content

    # -- predictions ---------------------------------------------------------------------
    def predict(self, transaction: dict) -> dict:
        return self._request("POST", "/predict", json=transaction).json()

    def explain(self, transaction: dict, top_k: int = 5) -> dict:
        return self._request("POST", "/explain", params={"top_k": top_k}, json=transaction).json()

    def predict_frame(self, frame: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        """Score a DataFrame through /predict/batch in chunks; returns one row per input."""
        prepared = prepare_batch(frame)
        results = []
        for start in range(0, len(prepared), batch_size):
            chunk = prepared.iloc[start : start + batch_size]
            payload = {"transactions": chunk.to_dict(orient="records")}
            results.extend(
                self._request("POST", "/predict/batch", json=payload).json()["predictions"]
            )
        out = pd.DataFrame(results)
        out["warnings"] = out["warnings"].map(lambda w: " | ".join(w))
        return out


def prepare_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Check an uploaded table and keep only what the API accepts.

    Raises ``ValueError`` with a readable message for missing columns, non-numeric
    or empty values. A ``Class`` label column, if present, is ignored.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The file has no rows")
    features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    bad = features.isna() | ~features.map(lambda v: math.isfinite(v) if pd.notna(v) else False)
    if bad.any().any():
        rows = (bad.any(axis=1)[lambda s: s].index[:5] + 2).tolist()  # +2: header, 1-based
        raise ValueError(f"Non-numeric, empty or infinite values (first rows: {rows})")
    out = features.astype(float)
    if "transaction_id" in frame.columns:
        ids = frame["transaction_id"].astype(str).str.replace(r"[^A-Za-z0-9_.:-]", "_", regex=True)
        out.insert(0, "transaction_id", ids.str.slice(0, 64))
    else:
        out.insert(0, "transaction_id", [f"row-{i + 1}" for i in range(len(frame))])
    return out


def read_csv_upload(data: bytes, max_rows: int) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(data))
    if len(frame) > max_rows:
        raise ValueError(f"The file has {len(frame):,} rows; the dashboard limit is {max_rows:,}")
    return frame
