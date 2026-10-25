"""Load, clean, de-duplicate, and split the email data."""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "combined_data.csv"

_WORD_RE = re.compile(r"[a-z0-9]+")
_STRIP_RE = re.compile(r"[^a-z0-9]+")

# Common label spellings in public datasets.
_HAM_LABELS = {"0", "ham", "not spam", "not-spam", "nonspam", "legit", "legitimate", "false"}
_SPAM_LABELS = {"1", "spam", "spammy", "true", "yes"}


@dataclass
class Split:
    """Train, validation, and test data."""

    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series
    source_train: pd.Series | None = None
    source_val: pd.Series | None = None
    source_test: pd.Series | None = None

    def summary(self) -> dict[str, int]:
        return {
            "train": int(len(self.X_train)),
            "validation": int(len(self.X_val)),
            "test": int(len(self.X_test)),
        }


@dataclass
class Dataset:
    """Cleaned data and a summary of what cleaning removed."""

    X: pd.Series
    y: pd.Series
    stats: dict[str, Any] = field(default_factory=dict)
    source: pd.Series | None = None

    def __len__(self) -> int:
        return len(self.X)


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Return an explicit dataset path, or find the default file."""

    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset not found: {candidate}")
        return candidate

    for candidate in (DEFAULT_DATA_PATH, PROJECT_ROOT / "data" / "combined_data.csv"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find combined_data.csv. Pass --data or place it at the "
        f"project root ({DEFAULT_DATA_PATH})."
    )


def _load_extra_frame(path: str | Path) -> pd.DataFrame:
    """Load an optional SMS/CSV file with label and text columns."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Extra dataset not found: {source_path}")

    if source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith("smsspamcollection")
            ]
            if not members:
                raise ValueError(
                    f"{source_path} does not contain an SMSSpamCollection file"
                )
            with archive.open(members[0]) as handle:
                return pd.read_csv(
                    handle,
                    sep="\t",
                    header=None,
                    names=["label", "text"],
                    quoting=csv.QUOTE_NONE,
                    encoding="utf-8",
                )

    frame = pd.read_csv(source_path)
    if {"label", "text"}.issubset(frame.columns):
        return frame[["label", "text"]]

    return pd.read_csv(
        source_path,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )


def _prepare_frame(frame: pd.DataFrame, source: str, max_chars: int) -> pd.DataFrame:
    """Normalize labels/text and drop empty rows."""

    frame = frame.dropna(subset=["label", "text"]).copy()
    frame["label"] = frame["label"].map(_normalise_label)
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(np.int8)
    frame["text"] = (
        frame["text"]
        .astype(str)
        .str.replace("\x00", " ", regex=False)
        .str.replace("\r\n", "\n", regex=False)
        .str.replace("\r", "\n", regex=False)
        .str.strip()
        .str.slice(0, max_chars)
    )
    frame = frame[frame["text"].str.len() >= 5].copy()
    frame["source"] = source
    return frame[["label", "text", "source"]]


def _normalise_label(value: object) -> int | None:
    text = str(value).strip().lower()
    if text in _SPAM_LABELS:
        return 1
    if text in _HAM_LABELS:
        return 0
    try:
        number = int(float(text))
    except (TypeError, ValueError):
        return None
    return number if number in (0, 1) else None


def _fingerprint(text: str) -> bytes:
    """Make a simple duplicate key that ignores case and punctuation."""

    collapsed = _STRIP_RE.sub("", text.lower())
    return hashlib.blake2b(collapsed.encode("utf-8", "ignore"), digest_size=16).digest()


def _simhash(text: str, max_words: int = 400) -> int:
    """Build a 64-bit SimHash so near-identical messages get similar values."""

    words = _WORD_RE.findall(text.lower())[:max_words]
    if not words:
        return 0
    if len(words) < 3:
        shingles = words
    else:
        shingles = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]

    hashes = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(shingle.encode("utf-8", "ignore"), digest_size=8).digest(),
                "big",
            )
            for shingle in shingles
        ),
        dtype=np.uint64,
        count=len(shingles),
    )
    bits = np.unpackbits(hashes.view(np.uint8).reshape(-1, 8), axis=1)
    signed = bits.astype(np.int16) * 2 - 1
    totals = signed.sum(axis=0)

    fingerprint = 0
    for position, positive in enumerate(totals > 0):
        if positive:
            fingerprint |= 1 << (63 - position)
    return fingerprint


def _drop_near_duplicates(
    texts: list[str],
    keep_mask: np.ndarray,
    *,
    max_distance: int = 3,
    bands: int = 4,
) -> np.ndarray:
    """Mark messages to keep; LSH makes near-duplicate checks fast."""

    band_bits = 64 // bands
    band_mask = (1 << band_bits) - 1
    buckets: list[dict[int, list[int]]] = [{} for _ in range(bands)]
    kept_fingerprints: list[int] = []
    keep = np.zeros(len(texts), dtype=bool)

    for index, text in enumerate(texts):
        if not keep_mask[index]:
            continue

        fingerprint = _simhash(text)
        keys = [(fingerprint >> (band * band_bits)) & band_mask for band in range(bands)]

        duplicate = False
        for band, key in enumerate(keys):
            for other in buckets[band].get(key, ()):  # only LSH collisions
                if (fingerprint ^ kept_fingerprints[other]).bit_count() <= max_distance:
                    duplicate = True
                    break
            if duplicate:
                break

        if duplicate:
            continue

        kept_index = len(kept_fingerprints)
        for band, key in enumerate(keys):
            buckets[band].setdefault(key, []).append(kept_index)
        kept_fingerprints.append(fingerprint)
        keep[index] = True

    return keep


def load_dataset(
    path: str | Path | None = None,
    *,
    extra_data: Iterable[str | Path] | None = None,
    max_chars: int = 20_000,
    drop_near_duplicates: bool = True,
    near_dup_distance: int = 3,
    sample: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Dataset:
    """Load the main dataset, optionally add SMS data, then clean it."""

    data_path = resolve_data_path(path)
    main_frame = pd.read_csv(data_path, usecols=["label", "text"])
    rows_loaded = len(main_frame)
    frame = _prepare_frame(main_frame, data_path.stem, max_chars)

    extra_paths = [Path(item) for item in (extra_data or [])]
    for extra_path in extra_paths:
        extra_frame = _load_extra_frame(extra_path)
        rows_loaded += len(extra_frame)
        frame = pd.concat(
            [frame, _prepare_frame(extra_frame, extra_path.stem, max_chars)],
            ignore_index=True,
        )

    # Drop exact duplicates first.
    fingerprints = frame["text"].map(_fingerprint)
    exact_duplicate_mask = fingerprints.duplicated(keep="first")
    exact_duplicates = int(exact_duplicate_mask.sum())
    frame = frame.loc[~exact_duplicate_mask].reset_index(drop=True)

    near_duplicates = 0
    if drop_near_duplicates and len(frame) > 1:
        texts = frame["text"].tolist()
        keep = _drop_near_duplicates(
            texts,
            np.ones(len(texts), dtype=bool),
            max_distance=near_dup_distance,
        )
        near_duplicates = int((~keep).sum())
        frame = frame.loc[keep].reset_index(drop=True)

    if sample is not None and sample < len(frame):
        frame = frame.sample(n=sample, random_state=random_state).reset_index(drop=True)

    label_counts = frame["label"].value_counts().to_dict()
    source_counts = frame["source"].value_counts().to_dict()
    lengths = frame["text"].str.len()

    stats: dict[str, Any] = {
        # Keep local paths out of the committed metrics file.
        "source": data_path.name,
        "sources": {str(name): int(count) for name, count in source_counts.items()},
        "extra_sources": [path.name for path in extra_paths],
        "rows_loaded": rows_loaded,
        "rows_clean": int(len(frame)),
        "exact_duplicates_removed": exact_duplicates,
        "near_duplicates_removed": near_duplicates,
        "label_counts": {"ham": int(label_counts.get(0, 0)), "spam": int(label_counts.get(1, 0))},
        "text_length": {
            "mean": round(float(lengths.mean()), 1),
            "median": int(lengths.median()),
            "p95": int(lengths.quantile(0.95)),
            "max": int(lengths.max()),
        },
        "max_chars": max_chars,
    }

    if verbose:
        print(
            f"Loaded {rows_loaded:,} rows -> {len(frame):,} clean rows "
            f"({exact_duplicates:,} exact and {near_duplicates:,} near duplicates removed)"
        )
        print(
            f"Labels: ham={stats['label_counts']['ham']:,} "
            f"spam={stats['label_counts']['spam']:,}"
        )
        print(f"Sources: {stats['sources']}")

    return Dataset(
        X=frame["text"],
        y=frame["label"],
        stats=stats,
        source=frame["source"],
    )


def split_dataset(
    dataset: Dataset,
    *,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Split:
    """Make a stratified 80/10/10 split before fitting anything."""

    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")

    # Split row numbers so every column stays aligned.
    indices = np.arange(len(dataset))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=dataset.y,
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_size,
        stratify=dataset.y.iloc[train_val_idx],
        random_state=random_state,
    )

    def take(indices_for_part: np.ndarray) -> tuple[pd.Series, pd.Series, pd.Series | None]:
        return (
            dataset.X.iloc[indices_for_part].reset_index(drop=True),
            dataset.y.iloc[indices_for_part].reset_index(drop=True),
            (
                dataset.source.iloc[indices_for_part].reset_index(drop=True)
                if dataset.source is not None
                else None
            ),
        )

    X_train, y_train, source_train = take(train_idx)
    X_val, y_val, source_val = take(val_idx)
    X_test, y_test, source_test = take(test_idx)

    return Split(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        source_train=source_train,
        source_val=source_val,
        source_test=source_test,
    )
