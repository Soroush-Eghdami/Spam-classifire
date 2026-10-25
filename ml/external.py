"""Test the model on the public SpamAssassin email corpus."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import pandas as pd

from ml.data import PROJECT_ROOT
from ml.evaluate import (
    binary_metrics,
    length_bucket,
    metrics_by_group,
    predict_scores,
)
from ml.features import normalize_text

DEFAULT_SPAMASSASSIN_DIR = PROJECT_ROOT / "data" / "spamassassin"
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_ARCHIVES = (
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
    "20050311_spam_2.tar.bz2",
)

_SCRIPT_RE = re.compile(r"(?is)<(script|style).*?</\1>")
_TAG_RE = re.compile(r"(?s)<[^>]+>")


@dataclass
class ExternalDataset:
    """External messages, labels, and cleaning stats."""

    X: pd.Series
    y: pd.Series
    stats: dict[str, Any]


def download_spamassassin(
    destination: str | Path = DEFAULT_SPAMASSASSIN_DIR,
    *,
    force: bool = False,
) -> list[Path]:
    """Download the six SpamAssassin archives."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename in SPAMASSASSIN_ARCHIVES:
        target = destination / filename
        if force or not target.exists() or target.stat().st_size == 0:
            request = urllib.request.Request(
                SPAMASSASSIN_BASE_URL + filename,
                headers={"User-Agent": "sift-spam-classifier/2.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                target.write_bytes(response.read())
        downloaded.append(target)
    return downloaded


def _archive_label(filename: str) -> int:
    name = filename.lower()
    if "spam" in name and "ham" not in name:
        return 1
    if "ham" in name:
        return 0
    raise ValueError(f"Cannot infer a label from {filename}")


def _strip_html(text: str) -> str:
    text = _SCRIPT_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def _part_text(part: Any) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        text = payload.decode(charset, errors="replace")
    except LookupError:
        text = payload.decode("utf-8", errors="replace")
    if part.get_content_subtype() == "html":
        text = _strip_html(text)
    return text


def _message_text(raw: bytes) -> str:
    """Get the message body, without headers or attachments."""

    try:
        message = BytesParser(policy=policy.default).parsebytes(raw)
    except Exception:  # bad emails happen in real life
        return ""

    if not message.is_multipart():
        return _part_text(message).strip()

    plain: list[str] = []
    rich: list[str] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        if (part.get_content_disposition() or "").lower() == "attachment":
            continue
        subtype = part.get_content_subtype()
        text = _part_text(part)
        if not text.strip():
            continue
        if subtype == "plain":
            plain.append(text)
        elif subtype == "html":
            rich.append(text)
    return ("\n".join(plain) if plain else "\n".join(rich)).strip()


def _archive_paths(path: str | Path) -> list[Path]:
    candidate = Path(path)
    if candidate.is_dir():
        return sorted(candidate.glob("*.tar.bz2"))
    if candidate.is_file():
        return [candidate]
    raise FileNotFoundError(
        f"SpamAssassin corpus not found at {candidate}. Run `python -m ml.external --download`."
    )


def load_spamassassin(
    path: str | Path = DEFAULT_SPAMASSASSIN_DIR,
    *,
    max_chars: int = 20_000,
) -> ExternalDataset:
    """Read the archives into clean text and labels."""

    archives = _archive_paths(path)
    if not archives:
        raise FileNotFoundError(
            f"No .tar.bz2 archives found in {Path(path)}. "
            "Run `python -m ml.external --download`."
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    empty = 0
    duplicates = 0
    rows_loaded = 0

    for archive_path in archives:
        label = _archive_label(archive_path.name)
        with tarfile.open(archive_path, "r:bz2") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                rows_loaded += 1
                text = _message_text(handle.read())
                cleaned = normalize_text(text, max_chars)
                if len(cleaned) < 5:
                    empty += 1
                    continue
                fingerprint = hashlib.blake2b(
                    cleaned.encode("utf-8", "ignore"), digest_size=16
                ).hexdigest()
                if fingerprint in seen:
                    duplicates += 1
                    continue
                seen.add(fingerprint)
                records.append(
                    {
                        "text": cleaned,
                        "label": label,
                        "source": "spamassassin",
                        "archive": archive_path.name,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No usable messages were found in the SpamAssassin corpus")

    lengths = frame["text"].str.len()
    label_counts = frame["label"].value_counts().to_dict()
    stats: dict[str, Any] = {
        "source": "SpamAssassin public corpus",
        "archives": [archive.name for archive in archives],
        "rows_loaded": rows_loaded,
        "rows_clean": int(len(frame)),
        "duplicates_removed": duplicates,
        "empty_removed": empty,
        "label_counts": {
            "ham": int(label_counts.get(0, 0)),
            "spam": int(label_counts.get(1, 0)),
        },
        "text_length": {
            "mean": round(float(lengths.mean()), 1),
            "median": int(lengths.median()),
            "p95": int(lengths.quantile(0.95)),
            "max": int(lengths.max()),
        },
    }
    return ExternalDataset(X=frame["text"], y=frame["label"], stats=stats)


def evaluate_external(
    pipeline: Any,
    dataset: ExternalDataset,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Score the corpus with the deployed threshold and with 0.5."""

    scores = predict_scores(pipeline, dataset.X.tolist())
    groups = [length_bucket(int(length)) for length in dataset.X.str.len()]
    return {
        "dataset": dataset.stats,
        "threshold": round(float(threshold), 4),
        "at_tuned_threshold": binary_metrics(dataset.y, scores, threshold),
        "at_0.5": binary_metrics(dataset.y, scores, 0.5),
        "by_length": metrics_by_group(dataset.y, scores, groups, threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Sift on SpamAssassin")
    parser.add_argument("--download", action="store_true", help="Download the corpus archives")
    parser.add_argument("--evaluate", action="store_true", help="Score the corpus with a saved model")
    parser.add_argument("--force", action="store_true", help="Redownload existing archives")
    parser.add_argument(
        "--path",
        default=str(DEFAULT_SPAMASSASSIN_DIR),
        help="Corpus directory or one .tar.bz2 archive",
    )
    parser.add_argument("--model", default=None, help="Optional model artifact to evaluate")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path; defaults to stdout",
    )
    args = parser.parse_args()

    if args.download:
        downloaded = download_spamassassin(args.path, force=args.force)
        print(f"Downloaded {len(downloaded)} archives to {Path(args.path)}")

    if args.evaluate and not args.model:
        from ml.predict import DEFAULT_MODEL_PATH

        args.model = str(DEFAULT_MODEL_PATH)

    dataset = load_spamassassin(args.path)
    print(json.dumps(dataset.stats, indent=2))

    if args.model:
        from ml.predict import load_model

        model = load_model(args.model)
        payload = evaluate_external(
            model,
            dataset,
            threshold=float(getattr(model, "threshold_", 0.5)),
        )
        rendered = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
            print(f"Wrote {args.output}")
        else:
            print(rendered)


if __name__ == "__main__":
    main()
