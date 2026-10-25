"""Train the spam model and save its metrics."""

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
    length_bucket,
    metrics_by_group,
    predict_scores,
    top_features,
)
from ml.external import (
    DEFAULT_SPAMASSASSIN_DIR,
    evaluate_external,
    load_spamassassin,
)
from ml.features import TextStats, build_pipeline
from ml.predict import predict_many

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "artifacts"
DEFAULT_SMS_PATH = PROJECT_ROOT / "data" / "sms_spam.zip"
RANDOM_STATE = 42
# Keep false positives below 0.5% on validation.
MAX_VALIDATION_FPR = 0.005
# Treat PR-AUC gaps under this as a tie and pick the simpler model.
SELECTION_TOLERANCE = 0.003


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _candidate_specs() -> dict[str, dict[str, Any]]:
    """Models compared on validation."""

    return {
        "LogisticRegression": {
            "params": {"C": 0.25, "solver": "liblinear", "max_iter": 2000},
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
        return [
            {"C": value, "solver": "liblinear", "max_iter": 2000}
            for value in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
        ]
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
    threshold = choose_threshold(
        y_val, scores, beta=0.5, max_fpr=MAX_VALIDATION_FPR
    )
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
    sms_data: str | Path | None = None,
    spamassassin_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    sample: int | None = None,
    drop_near_duplicates: bool = True,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Run training and return the metrics document."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()

    _log("Loading and cleaning dataset")
    extra_data: list[Path] = []
    if sms_data is not None:
        extra_data.append(Path(sms_data))
    elif DEFAULT_SMS_PATH.exists():
        extra_data.append(DEFAULT_SMS_PATH)
        _log(f"Including short-message corpus: {DEFAULT_SMS_PATH.name}")
    dataset = load_dataset(
        data_path,
        extra_data=extra_data,
        drop_near_duplicates=drop_near_duplicates,
        sample=sample,
        random_state=random_state,
    )
    split = split_dataset(dataset, random_state=random_state)
    _log(f"Split: {split.summary()}")

    _log("Fitting shared feature space on the training split only")
    feature_started = time.perf_counter()
    feature_pipeline = build_pipeline(
        LogisticRegression(C=1.0, solver="liblinear", max_iter=2000)
    )[:-1]
    features = feature_pipeline.named_steps["features"]
    X_train_features = feature_pipeline.fit_transform(split.X_train, split.y_train)
    feature_seconds = time.perf_counter() - feature_started
    X_val_features = feature_pipeline.transform(split.X_val)
    _log(
        f"Feature matrix: {X_train_features.shape[0]:,} x "
        f"{X_train_features.shape[1]:,} ({X_train_features.nnz / 1e6:.0f}M non-zeros) "
        f"in {feature_seconds:.1f}s"
    )

    # Compare word-only with word+char without rebuilding the features.
    word_width = len(features.transformer_list[0][1].vocabulary_)
    char_width = len(features.transformer_list[1][1].named_steps["tfidf"].vocabulary_)
    stats_width = (
        len(TextStats.FEATURE_NAMES)
        if any(name == "stats" for name, _ in features.transformer_list)
        else 0
    )

    def run_ablation(width: int, label: str) -> dict[str, Any]:
        _log(f"Running ablation: {label}")
        model = LogisticRegression(
            C=0.25, solver="liblinear", max_iter=2000, random_state=random_state
        )
        model.fit(X_train_features[:, :width], split.y_train)
        scores = predict_scores(model, X_val_features[:, :width])
        threshold = choose_threshold(
            split.y_val, scores, beta=0.5, max_fpr=MAX_VALIDATION_FPR
        )
        result = binary_metrics(split.y_val, scores, threshold)
        result["feature_count"] = int(width)
        return result

    ablation_word_only = run_ablation(word_width, "word TF-IDF only")
    ablation_word_char = run_ablation(
        word_width + char_width, "word + character TF-IDF"
    )
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
    best_ap = comparison[0]["average_precision"]
    # If scores are close, prefer the simpler and more stable model.
    preference = {
        "LogisticRegression": 0,
        "CalibratedLinearSVC": 1,
        "ComplementNB": 2,
    }
    eligible = [
        row
        for row in comparison
        if best_ap - row["average_precision"] <= SELECTION_TOLERANCE
    ]
    family = min(eligible, key=lambda row: preference.get(row["name"], 99))["name"]
    _log(
        f"Best family by PR-AUC: {comparison[0]['name']} "
        f"(selected {family} within {SELECTION_TOLERANCE:.3f} tolerance)"
    )

    _log(f"Tuning {family} on validation")
    tuning: list[dict[str, Any]] = []
    tuning_runs: list[dict[str, Any]] = []

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
        tuning_runs.append(
            {
                "params": params,
                "model": model,
                "scores": scores,
                "threshold": threshold,
                "metrics": metrics,
            }
        )
        _log(
            f"  {params} -> AP={metrics['average_precision']:.4f} "
            f"F1={metrics['f1']:.4f} (threshold={threshold:.3f})"
        )

    best_ap = max(run["metrics"]["average_precision"] for run in tuning_runs)
    eligible_runs = [
        run
        for run in tuning_runs
        if best_ap - run["metrics"]["average_precision"] <= SELECTION_TOLERANCE
    ]
    # Among tied runs, pick the simplest one.
    selected_run = min(
        eligible_runs,
        key=lambda run: (
            float(run["params"].get("C", run["params"].get("alpha", 0.0))),
            -float(run["metrics"]["f0_5"]),
        ),
    )
    best_params = selected_run["params"]
    best_model = selected_run["model"]
    best_scores = selected_run["scores"]
    best_threshold = selected_run["threshold"]
    _log(
        f"Selected {best_params} "
        f"(AP={selected_run['metrics']['average_precision']:.4f}, "
        f"threshold={best_threshold:.3f})"
    )

    validation_metrics = binary_metrics(split.y_val, best_scores, best_threshold)
    validation_at_05 = binary_metrics(split.y_val, best_scores, 0.5)
    validation_by_length = metrics_by_group(
        split.y_val,
        best_scores,
        split.X_val.str.len().map(length_bucket).tolist(),
        best_threshold,
    )
    total_feature_count = int(word_width + char_width + stats_width)
    _log(
        f"Chosen threshold {best_threshold:.3f}: "
        f"P={validation_metrics['precision']:.4f} "
        f"R={validation_metrics['recall']:.4f} "
        f"F1={validation_metrics['f1']:.4f}"
    )

    # Free memory before fitting the final model.
    del (
        X_train_features,
        X_val_features,
        feature_pipeline,
        features,
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
    test_lengths = split.X_test.str.len().map(length_bucket)
    test_by_length = metrics_by_group(
        split.y_test, test_scores, test_lengths.tolist(), best_threshold
    )
    test_by_source = (
        metrics_by_group(
            split.y_test,
            test_scores,
            split.source_test.tolist(),
            best_threshold,
        )
        if split.source_test is not None
        else {}
    )

    external_metrics: dict[str, Any] | None = None
    external_path: Path | None = None
    if spamassassin_path is not None:
        external_path = Path(spamassassin_path)
        if not external_path.exists():
            raise FileNotFoundError(f"SpamAssassin corpus not found: {external_path}")
    elif DEFAULT_SPAMASSASSIN_DIR.exists():
        external_path = DEFAULT_SPAMASSASSIN_DIR

    if external_path is not None:
        _log(f"Evaluating external SpamAssassin corpus: {external_path.name}")
        external_dataset = load_spamassassin(external_path)
        external_metrics = evaluate_external(
            final_pipeline,
            external_dataset,
            threshold=best_threshold,
        )
        external_test = external_metrics["at_tuned_threshold"]
        _log(
            f"SpamAssassin: P={external_test['precision']:.4f} "
            f"R={external_test['recall']:.4f} F1={external_test['f1']:.4f} "
            f"FP={external_test['confusion_matrix']['fp']} "
            f"FN={external_test['confusion_matrix']['fn']}"
        )
    else:
        _log("SpamAssassin corpus not found; skipping external evaluation")

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
            "number_masking": {
                "action": "removed",
                "patterns": ["escapenumber", "standalone digits/years/dates/phone numbers"],
                "preserves_digits_inside_words": True,
            },
            "url_masking": "URLs -> urltoken; email addresses -> emailtoken",
            "html": "tags stripped before feature extraction",
            "whitespace": "collapsed to single spaces",
            "length_style_features": "excluded",
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
                "weight": 0.2,
            },
            "text_stats": [],
            "stats_included": False,
            "stats_note": (
                "Dense content statistics are available in TextStats for ablation, "
                "but are excluded from the deployed model because they reduced "
                "cross-corpus precision."
            ),
            "excluded_stats": [
                "log_chars",
                "log_words",
                "avg_word_len",
                "max_word_len",
                "unique_word_ratio",
                "digit_ratio",
                "uppercase_ratio",
                "number_token_ratio",
                "caps_word_ratio",
                "log_lines",
            ],
        },
        "model": {
            "family": family,
            "params": best_params,
            "threshold": float(best_threshold),
            "threshold_rule": (
                "maximise F0.5 on validation subject to false-positive rate "
                f"<= {MAX_VALIDATION_FPR:.1%} (precision weighted)"
            ),
        },
        "validation": {
            "threshold": float(best_threshold),
            "at_tuned_threshold": validation_metrics,
            "at_0.5": validation_at_05,
            "by_length": validation_by_length,
        },
        "test": {
            "threshold": float(best_threshold),
            "at_tuned_threshold": test_metrics,
            "at_0.5": test_at_05,
            "by_length": test_by_length,
            "by_source": test_by_source,
        },
        "external": external_metrics,
        "comparison": comparison,
        "tuning": tuning,
        "ablation": {
            "word_only": ablation_word_only,
            "word_char": ablation_word_char,
            "full_feature_union": {
                "feature_count": total_feature_count,
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
            "The decision threshold maximises F0.5 on validation subject to a 0.5% validation false-positive-rate ceiling, deliberately trading some recall for fewer false positives.",
            "All inputs are lowercased; standalone numbers/years are removed, URLs and email addresses are replaced with neutral tokens, and HTML tags are stripped before feature extraction.",
            "Length, average word length, unique-word ratio, casing, line count, and URL/punctuation statistics are not features in the deployed model. The external check motivated dropping the dense statistics entirely.",
            "When PR-AUC differs by less than 0.003, the simpler/more strongly regularised model is preferred instead of chasing a noise-level gain.",
            "The external SpamAssassin result is reported separately and was not used for model selection or threshold tuning.",
            "Accuracy alone is misleading on imbalanced mail; precision, recall, the confusion matrix, length slices, and source slices are reported.",
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
        "--sms-data",
        type=str,
        default=None,
        help="Optional UCI SMS Spam Collection zip/TSV to add short messages",
    )
    parser.add_argument(
        "--spamassassin",
        type=str,
        default=None,
        help="SpamAssassin corpus directory/archive for external evaluation",
    )
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
        sms_data=args.sms_data,
        spamassassin_path=args.spamassassin,
        output_dir=args.output,
        sample=args.sample,
        drop_near_duplicates=not args.no_near_dedup,
        random_state=args.random_state,
    )
