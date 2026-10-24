"""Evaluation helpers: metrics, threshold tuning, and feature inspection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_scores(model: Any, texts: Any) -> np.ndarray:
    """Return a spam score from either predict_proba or decision_function."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        return np.asarray(probabilities)[:, 1]
    if hasattr(model, "decision_function"):
        margins = np.asarray(model.decision_function(texts))
        return 1.0 / (1.0 + np.exp(-margins))
    raise TypeError(f"{type(model).__name__} exposes neither predict_proba nor decision_function")


def binary_metrics(y_true: Any, scores: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Return the main metrics for one threshold."""

    y_true = np.asarray(y_true).astype(int)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "f0_5": round(float(fbeta_score(y_true, predictions, beta=0.5, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "average_precision": round(float(average_precision_score(y_true, scores)), 4),
        "brier": round(float(brier_score_loss(y_true, np.clip(scores, 0, 1))), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) else 0.0, 4),
        "support": {"ham": int((y_true == 0).sum()), "spam": int((y_true == 1).sum())},
    }


def metrics_by_group(
    y_true: Any,
    scores: np.ndarray,
    groups: Any,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Calculate metrics for each length/source slice."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)

    result: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if not mask.any():
            continue
        result[str(group)] = binary_metrics(y_true[mask], scores[mask], threshold)
    return result


def length_bucket(length: int) -> str:
    """Group messages into short, medium, and long buckets."""

    if length < 160:
        return "short (<160 chars)"
    if length < 500:
        return "medium (160-499 chars)"
    return "long (500+ chars)"


def choose_threshold(
    y_true: Any,
    scores: np.ndarray,
    *,
    beta: float = 0.5,
    grid_size: int = 400,
    max_fpr: float | None = None,
) -> float:
    """Pick the best threshold, with an optional false-positive cap."""

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    # Use validation scores only; never tune on the test set.
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, grid_size))
    candidates = np.unique(np.concatenate(([0.5], np.round(candidates, 4))))

    best_threshold = 0.5
    best_score = -1.0
    fallback_threshold = float(candidates[-1])
    fallback_fpr = float("inf")
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        tn, fp, _, _ = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
        if max_fpr is not None and fpr > max_fpr:
            # Remember the safest option if the cap is too tight.
            if fpr < fallback_fpr:
                fallback_fpr = fpr
                fallback_threshold = float(threshold)
            continue
        score = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        if score > best_score or (score == best_score and threshold > best_threshold):
            best_score = float(score)
            best_threshold = float(threshold)

    if best_score < 0.0:
        return round(fallback_threshold, 4)
    return round(best_threshold, 4)


def _classifier(pipeline: Any) -> Any:
    if hasattr(pipeline, "named_steps"):
        for name in ("clf", "classifier", "model"):
            if name in pipeline.named_steps:
                return pipeline.named_steps[name]
    return pipeline


def _coefficients(pipeline: Any) -> np.ndarray | None:
    """Get coefficients from a linear model or its calibrated version."""

    classifier = _classifier(pipeline)
    if hasattr(classifier, "coef_"):
        return np.asarray(classifier.coef_).ravel()
    if hasattr(classifier, "calibrated_classifiers_"):
        coefs = []
        for calibrated in classifier.calibrated_classifiers_:
            estimator = getattr(calibrated, "estimator", None)
            if estimator is not None and hasattr(estimator, "coef_"):
                coefs.append(np.asarray(estimator.coef_).ravel())
        if coefs:
            return np.mean(coefs, axis=0)
    return None


def feature_names(pipeline: Any) -> np.ndarray | None:
    """Turn raw pipeline names into readable feature names."""

    try:
        features = pipeline.named_steps["features"]
        names = features.get_feature_names_out()
    except (AttributeError, KeyError):
        return None

    cleaned = []
    for name in names:
        text = str(name)
        if text.startswith("word__"):
            cleaned.append(text[len("word__") :])
        elif text.startswith("char__"):
            cleaned.append("~" + text[len("char__") :])
        elif text.startswith("stats__"):
            cleaned.append("stat:" + text[len("stats__") :])
        else:
            cleaned.append(text)
    return np.asarray(cleaned, dtype=object)


def top_features(pipeline: Any, n: int = 20) -> dict[str, list[dict[str, Any]]]:
    """Return the model's strongest spam and ham words."""

    coefficients = _coefficients(pipeline)
    names = feature_names(pipeline)
    if coefficients is None or names is None or len(names) != len(coefficients):
        return {"spam": [], "ham": []}

    order = np.argsort(coefficients)
    ham_indices = order[:n][::-1]
    spam_indices = order[-n:][::-1]

    def rows(indices: np.ndarray) -> list[dict[str, Any]]:
        result = []
        for index in indices:
            result.append(
                {
                    "word": str(names[index]),
                    "weight": round(float(coefficients[index]), 4),
                }
            )
        return result

    return {"spam": rows(spam_indices), "ham": rows(ham_indices)}


def top_feature_rows(
    pipeline: Any,
    texts: Any,
    *,
    threshold: float = 0.5,
    top_n: int = 8,
) -> list[list[dict[str, Any]]]:
    """Return the words that pushed each message toward spam or ham."""

    # Use the same preprocessing as the classifier.
    preprocessor = pipeline[:-1] if hasattr(pipeline, "__getitem__") else pipeline
    matrix = preprocessor.transform(texts)
    coefficients = _coefficients(pipeline)
    names = feature_names(pipeline)
    if coefficients is None or names is None:
        return [[] for _ in texts]

    matrix = matrix.tocsr()
    results: list[list[dict[str, Any]]] = []
    for row_index in range(matrix.shape[0]):
        start, end = matrix.indptr[row_index], matrix.indptr[row_index + 1]
        columns = matrix.indices[start:end]
        values = matrix.data[start:end]
        contributions = values * coefficients[columns]

        # Rank by absolute impact so negative (ham) signals can appear too.
        order = np.argsort(-np.abs(contributions))
        picked: list[int] = []
        for position in order:
            col = columns[position]
            name = str(names[col])
            if name.startswith("stat:"):
                continue
            if name.startswith("~") and len(picked) >= max(3, top_n // 2):
                continue
            picked.append(position)
            if len(picked) >= top_n:
                break

        results.append(
            [
                {
                    "word": str(names[columns[position]]),
                    "weight": round(float(contributions[position]), 4),
                }
                for position in picked
            ]
        )
    return results
