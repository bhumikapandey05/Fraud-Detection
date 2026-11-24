"""FastAPI prediction service.

Endpoints
---------
GET  /health                 liveness  - the process is up (always 200)
GET  /health/ready           readiness - approved model loaded and verified (200 / 503)
GET  /model                  served model: name, registry version, alias, threshold, checksum
GET  /model/performance      recorded evaluation results of the frozen release
GET  /model/performance/figures/{name}   evaluation charts (PNG)
POST /predict                score one transaction
POST /predict/batch          score up to ``serving.max_batch_size`` transactions
POST /explain                score one transaction + local SHAP explanation

Scored transactions from /predict and /predict/batch are stored in the local
monitoring store (``monitoring.capture_predictions``) for drift monitoring.

Every response carries an ``X-Request-ID`` header; errors are JSON with a short,
safe message (no stack traces, no echo of submitted values).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from fraud_detection import __version__
from fraud_detection.config import Config, load_config
from fraud_detection.logging_utils import get_logger
from fraud_detection.serving.model_loader import ServedModel, load_served_model
from fraud_detection.serving.reports import figure_path, load_performance
from fraud_detection.serving.schemas import (
    BatchRequest,
    BatchResponse,
    ExplanationMethod,
    ExplanationResponse,
    FeatureContribution,
    FeatureRanges,
    ModelRef,
    Prediction,
    PredictionResponse,
    Transaction,
    to_frame,
)

logger = get_logger(__name__)


class ServiceUnavailableError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BatchTooLargeError(Exception):
    pass


class ExplanationsUnavailableError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _error(status: int, error: str, detail, request: Request) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": error,
            "detail": detail,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def _default_explainer_factory(cfg: Config, model: ServedModel):
    from fraud_detection.explainability.service import build_explainer

    return build_explainer(cfg, model.detector, model.snapshot)


def create_app(
    cfg: Config | None = None,
    loader: Callable[[Config], ServedModel] = load_served_model,
    explainer_factory: Callable | None = _default_explainer_factory,
) -> FastAPI:
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.not_ready_reason = "model not loaded yet"
        try:
            app.state.model = loader(cfg)
            app.state.not_ready_reason = None
        except Exception as exc:  # fail closed: stay up, report not ready
            app.state.not_ready_reason = f"{type(exc).__name__}: {exc}"
            logger.error("Model failed to load; serving 503s. Reason: %s", exc)
        app.state.ranges = FeatureRanges.from_training_split(
            cfg.paths.processed_dir / "train.parquet"
        )
        # Prediction capture for monitoring (Phase 5). Failures never affect predictions.
        app.state.store = None
        if cfg.monitoring.capture_predictions:
            try:
                from fraud_detection.monitoring.store import MonitoringStore

                app.state.store = MonitoringStore(cfg.paths.monitoring_dir)
            except Exception as exc:
                logger.error("Prediction capture disabled: %s", exc)
        # Explanations are optional: a failure here never affects predictions.
        app.state.explainer = None
        app.state.explainer_reason = "model not loaded"
        if app.state.model is not None and explainer_factory is not None:
            try:
                app.state.explainer = explainer_factory(cfg, app.state.model)
                app.state.explainer_reason = None
            except Exception as exc:
                app.state.explainer_reason = f"{type(exc).__name__}: {exc}"
                logger.error("Explanations disabled: %s", exc)
        elif explainer_factory is None:
            app.state.explainer_reason = "explanations are disabled"
        yield

    app = FastAPI(
        title="Credit Card Fraud Detection API",
        version=__version__,
        description="Scores card transactions with the approved, frozen fraud model.",
        lifespan=lifespan,
    )

    # --- middleware & error handling ----------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "%s %s -> %d (%.1f ms) id=%s",
            request.method,
            request.url.path,
            response.status_code,
            1000 * (time.perf_counter() - start),
            request.state.request_id,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        details = [
            {"field": ".".join(str(p) for p in err["loc"] if p != "body"), "message": err["msg"]}
            for err in exc.errors()
        ]
        return _error(422, "validation_error", details[:50], request)

    @app.exception_handler(ServiceUnavailableError)
    async def on_unavailable(request: Request, exc: ServiceUnavailableError):
        return _error(503, "model_unavailable", exc.reason, request)

    @app.exception_handler(BatchTooLargeError)
    async def on_batch_too_large(request: Request, exc: BatchTooLargeError):
        return _error(413, "batch_too_large", str(exc), request)

    @app.exception_handler(ExplanationsUnavailableError)
    async def on_no_explanations(request: Request, exc: ExplanationsUnavailableError):
        return _error(503, "explanations_unavailable", exc.reason, request)

    @app.exception_handler(Exception)
    async def on_unexpected(request: Request, exc: Exception):
        logger.exception("Unhandled error (request id %s)", request.state.request_id)
        return _error(
            500,
            "internal_error",
            "Unexpected server error; see server logs with the request id.",
            request,
        )

    # --- helpers ----------------------------------------------------------------------
    def served(request: Request) -> ServedModel:
        model = request.app.state.model
        if model is None:
            raise ServiceUnavailableError(request.app.state.not_ready_reason)
        return model

    def model_ref(model: ServedModel) -> ModelRef:
        return ModelRef(
            name=model.model_name,
            version=model.version,
            source=model.source,
            threshold=model.threshold,
        )

    def score(request: Request, transactions: list) -> list[Prediction]:
        model = served(request)
        frame = to_frame(transactions)
        scores = model.score(frame)
        ranges: FeatureRanges | None = request.app.state.ranges
        rows = frame.to_dict(orient="records")
        out = []
        for tx, row, s in zip(transactions, rows, scores, strict=True):
            flagged = bool(s >= model.threshold)
            out.append(
                Prediction(
                    transaction_id=tx.transaction_id,
                    fraud_score=float(s),
                    is_fraud=flagged,
                    decision="REVIEW" if flagged else "APPROVE",
                    warnings=ranges.warnings_for(row) if ranges else [],
                )
            )
        return out

    def capture(request: Request, transactions: list, predictions: list[Prediction]) -> None:
        """Store what was served, for monitoring. Never fails the request."""
        store = request.app.state.store
        if store is None:
            return
        try:
            model = request.app.state.model
            rid = request.state.request_id
            store.log_predictions(
                to_frame(transactions),
                [p.transaction_id or f"{rid}-{i}" for i, p in enumerate(predictions)],
                [p.fraud_score for p in predictions],
                model.threshold,
                model.version,
                batch_id=f"api-{time.strftime('%Y-%m-%d', time.gmtime())}",
                source="api",
                request_id=rid,
            )
        except Exception:
            logger.exception(
                "Could not store predictions for monitoring (request %s)",
                request.state.request_id,
            )

    # --- endpoints ----------------------------------------------------------------------
    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "service": "fraud-detection-api", "version": __version__}

    @app.get("/health/ready", tags=["health"])
    def ready(request: Request):
        model = request.app.state.model
        if model is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": request.app.state.not_ready_reason},
            )
        return {"status": "ready", "model": model.info()}

    @app.get("/model", tags=["model"])
    def model_info(request: Request):
        info = served(request).info()
        info["max_batch_size"] = cfg.serving.max_batch_size
        info["explanations_available"] = request.app.state.explainer is not None
        if request.app.state.explainer is None:
            info["explanations_unavailable_reason"] = request.app.state.explainer_reason
        return info

    @app.get("/model/performance", tags=["model"])
    def performance(request: Request):
        served(request)
        # Reports of the release actually being served (follows promotions/rollbacks).
        return load_performance(cfg, served(request).snapshot)

    @app.get("/model/performance/figures/{name}", tags=["model"])
    def figure(name: str, request: Request):
        path = figure_path(cfg, served(request).snapshot, name)
        if path is None:
            return _error(404, "not_found", f"Unknown figure '{name}'", request)
        return FileResponse(path, media_type="image/png")

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(transaction: Transaction, request: Request):  # type: ignore[valid-type]
        model = served(request)
        (prediction,) = score(request, [transaction])
        capture(request, [transaction], [prediction])
        return PredictionResponse(**prediction.model_dump(), model=model_ref(model))

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
    def predict_batch(batch: BatchRequest, request: Request):
        model = served(request)
        if len(batch.transactions) > cfg.serving.max_batch_size:
            raise BatchTooLargeError(
                f"{len(batch.transactions)} transactions sent; the limit is "
                f"{cfg.serving.max_batch_size} per request"
            )
        predictions = score(request, batch.transactions)
        capture(request, batch.transactions, predictions)
        return BatchResponse(
            model=model_ref(model),
            n_transactions=len(predictions),
            n_flagged=sum(p.is_fraud for p in predictions),
            predictions=predictions,
        )

    @app.post("/explain", response_model=ExplanationResponse, tags=["prediction"])
    def explain(
        transaction: Transaction,  # type: ignore[valid-type]
        request: Request,
        top_k: int = Query(
            cfg.explainability.top_k,
            ge=1,
            le=30,
            description="Features listed toward / away from fraud",
        ),
    ):
        """Score one transaction and explain the score with SHAP.

        ``base_value + sum(contributions)`` equals ``fraud_score`` (up to
        ``additivity_error``). The score and decision are computed exactly as in
        ``/predict``.
        """
        model = served(request)
        explainer = request.app.state.explainer
        if explainer is None:
            raise ExplanationsUnavailableError(request.app.state.explainer_reason)
        (prediction,) = score(request, [transaction])
        (exp,) = explainer.explain(to_frame([transaction]))
        toward = exp.toward_fraud(top_k)
        away = exp.away_from_fraud(top_k)
        listed = {c.feature for c in toward + away}
        info = explainer.background_info
        return ExplanationResponse(
            **prediction.model_dump(),
            model=model_ref(model),
            base_value=exp.base_value,
            top_toward_fraud=[FeatureContribution(**c.to_dict()) for c in toward],
            top_away_from_fraud=[FeatureContribution(**c.to_dict()) for c in away],
            other_features_contribution=sum(
                c.contribution for c in exp.contributions if c.feature not in listed
            ),
            contributions=[FeatureContribution(**c.to_dict()) for c in exp.contributions],
            additivity_error=exp.additivity_error,
            explanation_method=ExplanationMethod(
                method=explainer.METHOD,
                output="probability of fraud (class 1), the same number as fraud_score",
                baseline_meaning=(
                    f"average fraud score of {info['size']} randomly sampled training "
                    "transactions (a typical transaction)"
                ),
                background_size=info["size"],
                background_random_state=info.get("random_state", -1),
                notes=[
                    "Positive contributions push the score toward fraud; negative ones away.",
                    "V1-V28 are anonymized PCA components; their business meaning is unknown.",
                    "Time contributes through the hour of day only (hour_sin + hour_cos).",
                    "Contributions describe how the model used these values, not causes.",
                ],
            ),
        )

    return app
