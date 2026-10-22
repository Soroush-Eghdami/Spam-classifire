"""Train, compare, tune, and export the Sift spam classifier.

Run from the repository root::

    python -m ml.train                     # full training
    python -m ml.train --sample 15000      # quick smoke run

The script is deliberately linear and verbose:

1. load + clean + de-duplicate + split
2. fit the shared feature space **on the training split only**
3. compare three classical classifiers on the validation split
4. tune the winning family's regularisation
5. choose a precision-weighted decision threshold on validation
6. refit on train+validation, score the untouched test split once
7. write ``spam_pipeline.joblib`` and ``metrics.json``

No deep learning: TF-IDF + linear models are faster, smaller, and easier to
explain, which matters for spam where a reviewer wants to see *why* a message
was blocked.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from ml import __version__
from ml.data import PROJECT_ROOT, load_dataset, split_dataset
from ml.evaluate import (
    binary_metrics,
    choose_threshold,
    predict_scores,
    top_features,
)
from ml.features import TextStats, build_pipeline
from ml.predict import predict_many

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "artifacts"
RANDOM_STATE = 42


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _candidate_specs() -> dict[str, dict[str, Any]]:
    """The three classical families compared on the validation split."""

    return {
        "LogisticRegression": {
            "params": {"C": 4.0, "solver": "liblinear", "max_iter": 2000},
            "build": lambda params: LogisticRegression(
                **params, random_state=RANDOM_STATE
            ),
        },
        "CalibratedLinearSVC": {
            "params": {"C": 0.5, "calibration_cv": 3},
            "build": lambda params: CalibratedClassifierCV(
                estimator=LinearSVC(C=params["C"], dual="auto", random_state=RANDOM_STATE),
                method="sigmoid",
                cv=params["calibration_cv"],
            ),
        },
        "ComplementNB": {
            "params": {"alpha": 0.3},
            "build": lambda params: ComplementNB(alpha=params["alpha"]),
        },
    }


def _tuning_grid(family: str) -> list[dict[str, Any]]:
    if family == "LogisticRegression":
        return [{"C": value, "solver": "liblinear", "max_iter": 2000} for value in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)]
    if family == "CalibratedLinearSVC":
        return [{"C": value, "calibration_cv": 3} for value in (0.1, 0.25, 0.5, 1.0)]
    if family == "ComplementNB":
        return [{"alpha": value} for value in (0.1, 0.3, 0.5, 1.0)]
    raise ValueError(f"Unknown model family: {family}")


def _fit_and_score(
    build,
    params: dict[str, Any],
    X_train_features,
    y_train,
    X_val_features,
    y_val,
) -> tuple[Any, dict[str, Any], np.ndarray, float]:
    started = time.perf_counter()
    model = build(params)
    model.fit(X_train_features, y_train)
    fit_seconds = time.perf_counter() - started

    scores = predict_scores(model, X_val_features)
    threshold = choose_threshold(y_val, scores, beta=0.5)
    metrics = binary_metrics(y_val, scores, threshold)
    metrics.update(
        {
            "average_precision": round(float(metrics["average_precision"]), 4),
            "fit_seconds": round(fit_seconds, 2),
        }
    )
    return model, metrics, scores, threshold


def train(
    *,
    data_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample: int | None = None,
    drop_near_duplicates: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Execute the whole training workflow and return the metrics document."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    _log("Loading and cleaning dataset")
    dataset = load_dataset(
        data_path,
        drop_near_duplicates=drop_near_duplicates,
        sample=sample,
        random_state=random_state,
    )
    split = split_dataset(dataset, random_state=random_state)
    _log(f"Split: {split.summary()}")

    _log("Fitting shared feature space on the training split only")
    feature_started = time.perf_counter()
    full_pipeline = build_pipeline(
        LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
    )
    features = full_pipeline.named_steps["features"]
    X_train_features = features.fit_transform(split.X_train, split.y_train)
    feature_seconds = time.perf_counter() - feature_started
    X_val_features = features.transform(split.X_val)
    _log(
        f"Feature matrix: {X_train_features.shape[0]:,} x "
        f"{X_train_features.shape[1]:,} ({X_train_features.nnz / 1e6:.0f}M non-zeros) "
        f"in {feature_seconds:.1f}s"
    )

    # Ablation baseline: the word view alone, to show what char n-grams and
    # hand-crafted stats add.  The union stores "word" first.
    word_names = features.transformer_list[0][1].get_feature_names_out()
    word_width = len(word_names)
    _log("Running ablation: word TF-IDF only")
    word_model = LogisticRegression(
        C=4.0, solver="liblinear", max_iter=2000, random_state=random_state
    )
    word_model.fit(X_train_features[:, :word_width], split.y_train)
    word_scores = predict_scores(word_model, X_val_features[:, :word_width])
    word_threshold = choose_threshold(split.y_val, word_scores, beta=0.5)
    ablation_word_only = binary_metrics(split.y_val, word_scores, word_threshold)
    ablation_word_only["feature_count"] = int(word_width)
    del word_model, word_scores
    gc.collect()

    specs = _candidate_specs()
    comparison: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    val_scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    _log("Comparing candidate classifiers on the validation split")
    for name, spec in specs.items():
        _log(f"  fitting {name} {spec['params']}")
        model, metrics, scores, threshold = _fit_and_score(
            spec["build"],
            spec["params"],
            X_train_features,
            split.y_train,
            X_val_features,
            split.y_val,
        )
        fitted[name] = model
        val_scores[name] = scores
        thresholds[name] = threshold
        comparison.append(
            {
                "name": name,
                "params": spec["params"],
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f0_5": metrics["f0_5"],
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "fit_seconds": metrics["fit_seconds"],
            }
        )

    comparison.sort(key=lambda row: (row["average_precision"], row["f0_5"]), reverse=True)
    family = comparison[0]["name"]
    _log(f"Best family by PR-AUC: {family}")

    _log(f"Tuning {family} on validation")
    tuning: list[dict[str, Any]] = []
    best_params = specs[family]["params"]
    best_model = fitted[family]
    best_scores = val_scores[family]
    best_threshold = thresholds[family]
    best_ap = -1.0

    for params in _tuning_grid(family):
        model, metrics, scores, threshold = _fit_and_score(
            specs[family]["build"],
            params,
            X_train_features,
            split.y_train,
            X_val_features,
            split.y_val,
        )
        tuning.append(
            {
                "params": params,
                "threshold": threshold,
                "average_precision": metrics["average_precision"],
                "f0_5": metrics["f0_5"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fit_seconds": metrics["fit_seconds"],
            }
        )
        if metrics["average_precision"] > best_ap:
            best_ap = metrics["average_precision"]
            best_params = params
            best_model = model
            best_scores = scores
            best_threshold = threshold
        _log(
            f"  {params} -> AP={metrics['average_precision']:.4f} "
            f"F1={metrics['f1']:.4f} (threshold={threshold:.3f})"
        )

    validation_metrics = binary_metrics(split.y_val, best_scores, best_threshold)
    validation_at_05 = binary_metrics(split.y_val, best_scores, 0.5)
    _log(
        f"Chosen threshold {best_threshold:.3f}: "
        f"P={validation_metrics['precision']:.4f} "
        f"R={validation_metrics['recall']:.4f} "
        f"F1={validation_metrics['f1']:.4f}"
    )

    # Free the large matrices before refitting the deployable pipeline on
    # train + validation (otherwise peak memory roughly doubles).
    del (
        X_train_features,
        X_val_features,
        features,
        full_pipeline,
        fitted,
        best_model,
        val_scores,
    )
    gc.collect()

    _log("Refitting final pipeline on train + validation")
    final_started = time.perf_counter()
    final_pipeline = build_pipeline(specs[family]["build"](best_params))
    X_fit = np.concatenate([split.X_train.to_numpy(), split.X_val.to_numpy()])
    y_fit = np.concatenate([split.y_train.to_numpy(), split.y_val.to_numpy()])
    final_pipeline.fit(X_fit, y_fit)
    final_fit_seconds = time.perf_counter() - final_started

    final_pipeline.threshold_ = float(best_threshold)
    final_pipeline.version_ = __version__
    final_pipeline.trained_at_ = _now()

    _log("Scoring the untouched test split")
    test_scores = predict_scores(final_pipeline, split.X_test)
    test_metrics = binary_metrics(split.y_test, test_scores, best_threshold)
    test_at_05 = binary_metrics(split.y_test, test_scores, 0.5)

    latency_started = time.perf_counter()
    predict_many(final_pipeline, split.X_test.iloc[:200].tolist())
    latency_ms = (time.perf_counter() - latency_started) * 1000 / 200

    top = top_features(final_pipeline, n=20)

    metrics_document: dict[str, Any] = {
        "version": __version__,
        "trained_at": _now(),
        "training_seconds": round(time.perf_counter() - started_at, 1),
        "dataset": dataset.stats,
        "split": split.summary(),
        "preprocessing": {
            "max_chars": 20_000,
            "deduplication": "exact + SimHash Hamming<=3 (LSH banded)",
            "stop_words": False,
            "lowercase": True,
        },
        "features": {
            "word_tfidf": {
                "ngram_range": [1, 2],
                "max_features": 150_000,
                "min_df": 2,
                "sublinear_tf": True,
            },
            "char_tfidf": {
                "analyzer": "char_wb",
                "ngram_range": [3, 5],
                "max_features": 80_000,
                "min_df": 5,
                "max_chars": 4_000,
            },
            "text_stats": [str(name) for name in TextStats.FEATURE_NAMES],
        },
        "model": {
            "family": family,
            "params": best_params,
            "threshold": float(best_threshold),
            "threshold_rule": "maximise F0.5 on validation (precision weighted)",
        },
        "validation": {
            "threshold": float(best_threshold),
            "at_tuned_threshold": validation_metrics,
            "at_0.5": validation_at_05,
        },
        "test": {
            "threshold": float(best_threshold),
            "at_tuned_threshold": test_metrics,
            "at_0.5": test_at_05,
        },
        "comparison": comparison,
        "tuning": tuning,
        "ablation": {
            "word_only": ablation_word_only,
            "full_feature_union": {
                "feature_count": int(
                    final_pipeline.named_steps["features"].transform(["placeholder"]).shape[1]
                ),
                "average_precision": validation_metrics["average_precision"],
                "f0_5": validation_metrics["f0_5"],
            },
        },
        "top_features": top,
        "confusion_matrix": test_metrics["confusion_matrix"],
        "latency_ms_per_email": round(latency_ms, 2),
        "feature_seconds": round(feature_seconds, 1),
        "final_fit_seconds": round(final_fit_seconds, 1),
        "libraries": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "notes": [
            "Models and thresholds were selected with training-fold vectorizers only; the final artifact was refit on train + validation. Test data was never seen during training or threshold tuning.",
            "The decision threshold maximises F0.5 on validation, deliberately trading a little recall for fewer false positives.",
            "Accuracy alone is misleading on imbalanced mail; precision, recall, and the confusion matrix are reported.",
        ],
    }

    model_path = output_dir / "spam_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"
    _log(f"Saving pipeline to {model_path}")
    joblib.dump(final_pipeline, model_path, compress=3)
    metrics_document["artifact"] = {
        "path": model_path.name,
        "size_mb": round(model_path.stat().st_size / (1024 * 1024), 2),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_document, handle, indent=2)

    _log(
        f"Done. Test F1={test_metrics['f1']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"FP={test_metrics['confusion_matrix']['fp']} "
        f"FN={test_metrics['confusion_matrix']['fn']}"
    )
    return metrics_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sift spam classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to combined_data.csv")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Artifact directory (default: ml/artifacts)",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional row cap for smoke tests")
    parser.add_argument(
        "--no-near-dedup",
        action="store_true",
        help="Skip SimHash near-duplicate removal (faster, but risks leakage)",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_path=args.data,
        output_dir=args.output,
        sample=args.sample,
        drop_near_duplicates=not args.no_near_dedup,
        random_state=args.random_state,
    )
