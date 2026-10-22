"""Data loading, cleaning, near-duplicate removal, and leak-free splitting.

Spam corpora are notorious for duplicate messages: the same campaign email
arrives in thousands of inboxes, and public datasets often contain the same
message more than once.  If a near-identical copy lands in both the training
and test set, the reported score is inflated and the model looks better than
it is.  For that reason this module:

1. normalises text,
2. removes exact duplicates (after stripping punctuation / casing),
3. removes *near* duplicates with a 64-bit SimHash and LSH banding,
4. performs a stratified train / validation / test split **before** any
   vectorizer is fitted.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "combined_data.csv"

_WORD_RE = re.compile(r"[a-z0-9]+")
_STRIP_RE = re.compile(r"[^a-z0-9]+")

#: Label spellings seen in public spam datasets, mapped to 0 / 1.
_HAM_LABELS = {"0", "ham", "not spam", "not-spam", "nonspam", "legit", "legitimate", "false"}
_SPAM_LABELS = {"1", "spam", "spammy", "true", "yes"}


@dataclass
class Split:
    """The three stratified folds used throughout training and evaluation."""

    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series

    def summary(self) -> dict[str, int]:
        return {
            "train": int(len(self.X_train)),
            "validation": int(len(self.X_val)),
            "test": int(len(self.X_test)),
        }


@dataclass
class Dataset:
    """Cleaned dataset plus the audit trail of what cleaning removed."""

    X: pd.Series
    y: pd.Series
    stats: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)


def resolve_data_path(path: str | Path | None = None) -> Path:
    """Find the dataset, accepting either an explicit path or the defaults."""

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
    """Cheap exact-duplicate key: case, punctuation and spacing insensitive."""

    collapsed = _STRIP_RE.sub("", text.lower())
    return hashlib.blake2b(collapsed.encode("utf-8", "ignore"), digest_size=16).digest()


def _simhash(text: str, max_words: int = 400) -> int:
    """64-bit SimHash over word 3-shingles, implemented with numpy.

    Documents with a Hamming distance of a few bits are near duplicates.
    """

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
    """Return a boolean mask of documents that are not near duplicates.

    Uses LSH banding so the comparison is close to linear instead of O(n^2).
    With 4 bands of 16 bits, any pair within a Hamming distance of 3 is
    guaranteed to share at least one band (pigeonhole principle).
    """

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
            for other in buckets[band].get(key, ()):  # candidate collisions only
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
    max_chars: int = 20_000,
    drop_near_duplicates: bool = True,
    near_dup_distance: int = 3,
    sample: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> Dataset:
    """Load ``combined_data.csv`` and return clean, de-duplicated text/labels.

    Parameters
    ----------
    max_chars:
        Hard cap per message.  Some emails in the corpus are ~600k characters;
        keeping them would waste memory without adding vocabulary signal.
    sample:
        Optional row cap for quick experiments (sampled after cleaning).
    """

    data_path = resolve_data_path(path)
    frame = pd.read_csv(data_path, usecols=["label", "text"])
    rows_loaded = len(frame)

    frame = frame.dropna(subset=["label", "text"])
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
    frame = frame[frame["text"].str.len() >= 5]

    # Exact duplicates: same text ignoring case, punctuation, and whitespace.
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
    lengths = frame["text"].str.len()

    stats: dict[str, Any] = {
        "source": str(data_path),
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

    return Dataset(X=frame["text"], y=frame["label"], stats=stats)


def split_dataset(
    dataset: Dataset,
    *,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Split:
    """Stratified 80/10/10 split created *before* fitting any transformer."""

    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        stratify=dataset.y,
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return Split(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
