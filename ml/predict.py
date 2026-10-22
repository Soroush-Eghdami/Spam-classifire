"""Framework-agnostic inference: load a pipeline, predict, and explain.

The Django backend imports only this module.  Keeping inference here means the
model can be exercised from a shell or a notebook without Django installed,
and any backend bug cannot masquerade as an ML bug.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import joblib

from ml.evaluate import top_feature_rows
from ml.features import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ml" / "artifacts" / "spam_pipeline.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "ml" / "artifacts" / "metrics.json"

#: Keep inference consistent with how the model was trained.
MAX_TEXT_CHARS = 20_000


def load_model(path: str | Path | None = None) -> Any:
    """Load the fitted pipeline once and reuse it for every prediction."""

    model_path = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Train it first with: python -m ml.train"
        )
    return joblib.load(model_path)


def load_metrics(path: str | Path | None = None) -> dict[str, Any]:
    """Read the metrics sidecar written by the training script."""

    metrics_path = Path(path) if path is not None else DEFAULT_METRICS_PATH
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _threshold(model: Any, override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(getattr(model, "threshold_", 0.5))


def _probabilities(model: Any, texts: list[str]) -> list[float]:
    if hasattr(model, "predict_proba"):
        return [float(value) for value in model.predict_proba(texts)[:, 1]]
    if hasattr(model, "decision_function"):
        import numpy as np

        margins = np.asarray(model.decision_function(texts), dtype=float)
        return (1.0 / (1.0 + np.exp(-margins))).tolist()
    raise TypeError(f"{type(model).__name__} cannot produce spam probabilities")


def predict_text(
    model: Any,
    text: str,
    *,
    threshold: float | None = None,
    top_n: int = 8,
    explain: bool = True,
) -> dict[str, Any]:
    """Classify one message and (optionally) explain the strongest signals."""

    cleaned = normalize_text(text, MAX_TEXT_CHARS)
    if not cleaned:
        raise ValueError("Text is empty after cleaning")

    cut = _threshold(model, threshold)
    probability = _probabilities(model, [cleaned])[0]
    label = "spam" if probability >= cut else "ham"

    result: dict[str, Any] = {
        "label": label,
        "spam_probability": round(probability, 6),
        "ham_probability": round(1.0 - probability, 6),
        "threshold": round(cut, 4),
        "confidence": round(max(probability, 1.0 - probability), 4),
        "model_version": getattr(model, "version_", "unknown"),
        "text_length": len(cleaned),
    }

    if explain:
        contributions = top_feature_rows(model, [cleaned], threshold=cut, top_n=top_n)[0]
        result["top_words"] = contributions
        result["signals"] = {
            "toward_spam": [row for row in contributions if row["weight"] > 0],
            "toward_ham": [row for row in contributions if row["weight"] < 0],
        }
    return result


def predict_many(
    model: Any,
    texts: Iterable[str],
    *,
    threshold: float | None = None,
    explain: bool = False,
    top_n: int = 6,
) -> list[dict[str, Any]]:
    """Vectorised batch inference (one transform per file, not per row)."""

    cleaned = [normalize_text(text, MAX_TEXT_CHARS) for text in texts]
    cut = _threshold(model, threshold)
    probabilities = _probabilities(model, cleaned)

    contributions: list[list[dict[str, Any]]] = []
    if explain and cleaned:
        contributions = top_feature_rows(model, cleaned, threshold=cut, top_n=top_n)

    results = []
    for index, (text, probability) in enumerate(zip(cleaned, probabilities)):
        row: dict[str, Any] = {
            "label": "spam" if probability >= cut else "ham",
            "spam_probability": round(probability, 6),
            "threshold": round(cut, 4),
            "text_length": len(text),
        }
        if explain:
            row["top_words"] = contributions[index]
        results.append(row)
    return results
