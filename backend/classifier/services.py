"""Model lifecycle and prediction service.

The artifact is loaded **once per worker** and reused.  Loading a joblib file
on every request would add hundreds of milliseconds and waste memory; loading
it in ``AppConfig.ready`` would slow down management commands, so the model is
created lazily and guarded by a lock.
"""

from __future__ import annotations

import copy
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from django.conf import settings

logger = logging.getLogger(__name__)

# Ensure ``ml`` is importable when the backend runs from /backend.
_PROJECT_ROOT = Path(settings.BASE_DIR).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml import predict as ml_predict  # noqa: E402  (path setup must run first)


class ModelNotReadyError(RuntimeError):
    """Raised when the artifact is missing or cannot be unpickled."""


def make_preview(text: str, limit: int | None = None) -> str:
    """Collapse whitespace and truncate for safe storage/display."""

    limit = limit or settings.MAX_PREVIEW_CHARS
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


class ModelService:
    """Thread-safe singleton wrapper around the saved scikit-learn pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipeline: Any | None = None
        self._metrics: dict[str, Any] | None = None
        self._metrics_mtime: float | None = None
        self._loaded_at: float | None = None
        self._load_error: str | None = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def loaded_at(self) -> float | None:
        return self._loaded_at

    def reset(self) -> None:
        """Drop cached state (used by tests after swapping MODEL_PATH)."""

        with self._lock:
            self._pipeline = None
            self._metrics = None
            self._metrics_mtime = None
            self._loaded_at = None
            self._load_error = None

    def _ensure_loaded(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:  # double-checked locking
                return self._pipeline

            started = time.perf_counter()
            try:
                pipeline = ml_predict.load_model(settings.MODEL_PATH)
            except Exception as exc:  # noqa: BLE001 - surface a clear 503
                self._load_error = str(exc)
                logger.exception("Could not load model from %s", settings.MODEL_PATH)
                raise ModelNotReadyError(
                    "The classifier is not available. Train the model with "
                    "`python -m ml.train` and restart the backend."
                ) from exc

            self._pipeline = pipeline
            self._loaded_at = time.time()
            self._load_error = None
            logger.info(
                "Loaded model %s from %s in %.0f ms",
                getattr(pipeline, "version_", "unknown"),
                settings.MODEL_PATH,
                (time.perf_counter() - started) * 1000,
            )
            return pipeline

    # -- metadata ----------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        """Read metrics.json, refreshing only when the file changes."""

        path = Path(settings.METRICS_PATH)
        if not path.exists():
            return {}
        mtime = path.stat().st_mtime
        with self._lock:
            if self._metrics is not None and self._metrics_mtime == mtime:
                return copy.deepcopy(self._metrics)
            try:
                metrics = ml_predict.load_metrics(path)
            except (OSError, ValueError):
                logger.warning("Could not read metrics file %s", path, exc_info=True)
                return {}
            self._metrics = metrics
            self._metrics_mtime = mtime
        return copy.deepcopy(metrics)

    def model_info(self) -> dict[str, Any]:
        """Public model metadata, with local filesystem paths removed."""

        metrics = self.metrics()
        dataset = metrics.get("dataset")
        if isinstance(dataset, dict):
            dataset.pop("source", None)
        artifact = metrics.get("artifact")
        if isinstance(artifact, dict):
            artifact.pop("path", None)

        pipeline = self._pipeline
        metrics["runtime"] = {
            "loaded": pipeline is not None,
            "loaded_at": self._loaded_at,
            "threshold": float(getattr(pipeline, "threshold_", 0.5)) if pipeline else None,
            "version": getattr(pipeline, "version_", None) if pipeline else None,
        }
        return metrics

    # -- inference ---------------------------------------------------------
    def predict_one(self, text: str) -> dict[str, Any]:
        pipeline = self._ensure_loaded()
        result = ml_predict.predict_text(pipeline, text, explain=True, top_n=8)
        result["model_version"] = result.get("model_version") or self._version_fallback()
        return result

    def predict_many(self, texts: Iterable[str], *, explain: bool = False) -> list[dict[str, Any]]:
        pipeline = self._ensure_loaded()
        results = ml_predict.predict_many(pipeline, texts, explain=explain)
        fallback = self._version_fallback()
        for result in results:
            result["model_version"] = result.get("model_version") or fallback
        return results

    def _version_fallback(self) -> str:
        return str(self.metrics().get("version", "unknown"))

    def status(self) -> dict[str, Any]:
        metrics = self.metrics()
        return {
            "loaded": self.is_loaded,
            "model_version": getattr(self._pipeline, "version_", None) or metrics.get("version", "unknown"),
            "threshold": float(
                getattr(self._pipeline, "threshold_", metrics.get("model", {}).get("threshold", 0.5))
            ),
            "loaded_at": self._loaded_at,
            "load_error": self._load_error,
        }


#: Imported by views; a single instance per process.
model_service = ModelService()
