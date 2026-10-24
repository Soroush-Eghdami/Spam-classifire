"""Build text features for the spam model."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

# Punctuation and other non-word characters.
_NON_WORD_RE = re.compile(r"[^\w\s]")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# The old corpus already split URLs into words like "http www site com".
_URL_TOKEN_RE = re.compile(
    r"(?<!\w)(?:https?|www|com|org|net|biz|edu|gov)(?!\w)",
    re.IGNORECASE,
)
# Turn a split hostname into one token.
_URL_SEQUENCE_RE = re.compile(
    r"(?<!\w)urltoken(?:\s+[a-z0-9-]+){0,2}\s+urltoken(?!\w)",
    re.IGNORECASE,
)
# Real HTML tags only. "<http://...>" is a URL, not a tag.
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]{0,300})?>")
_REPEATED_PUNCT_RE = re.compile(r"[!?]{2,}")
# The old dataset uses this placeholder for numbers.
_ESCAPENUMBER_RE = re.compile(r"(?<!\w)escapenumber(?!\w)", re.IGNORECASE)
# Standalone numbers, dates, and prices. Digits inside words (fr33) stay.
_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:/\-]\d+)*(?!\w)")
_WHITESPACE_RE = re.compile(r"\s+")

_URGENCY_WORDS = (
    "urgent", "urgently", "immediately", "act now", "limited time",
    "expires", "expire", "last chance", "hurry", "final notice",
    "don't miss", "dont miss", "asap", "right away", "today only",
)
_MONEY_WORDS = (
    "free", "cash", "money", "prize", "winner", "win", "credit", "loan",
    "investment", "guarantee", "guaranteed", "viagra", "dollar", "usd",
    "income", "profit", "million", "billion", "cheap", "discount",
    "offer", "bonus", "reward", "refund", "unclaimed", "lottery",
)
_MARKETING_WORDS = (
    "click", "subscribe", "unsubscribe", "promo", "promotion", "sale",
    "buy", "order", "trial", "deal", "marketing", "advertisement",
    "newsletter", "opt out", "opt-out", "call now", "visit our",
)


def normalize_text(
    text: object,
    max_chars: int = 20_000,
    *,
    mask_numbers: bool = True,
    lowercase: bool = True,
) -> str:
    """Clean raw text the same way for training and inference."""

    if text is None:
        return ""
    value = str(text).replace("\x00", " ")
    if lowercase:
        value = value.lower()
    value = _ESCAPENUMBER_RE.sub(" ", value)
    if mask_numbers:
        value = _NUMBER_RE.sub(" ", value)
    value = _URL_RE.sub(" urltoken ", value)
    value = _EMAIL_RE.sub(" emailtoken ", value)
    value = _URL_TOKEN_RE.sub(" urltoken ", value)
    value = _URL_SEQUENCE_RE.sub(" urltoken ", value)
    value = _HTML_TAG_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    if len(value) > max_chars:
        value = value[:max_chars]
    return value


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Clean text once before the feature steps see it."""

    def __init__(
        self,
        max_chars: int = 20_000,
        mask_numbers: bool = True,
        lowercase: bool = True,
    ):
        self.max_chars = max_chars
        self.mask_numbers = mask_numbers
        self.lowercase = lowercase

    def fit(self, X: Sequence[str], y: object = None) -> "TextNormalizer":
        return self

    def transform(self, X: Sequence[str]) -> list[str]:
        return [
            normalize_text(
                text,
                self.max_chars,
                mask_numbers=self.mask_numbers,
                lowercase=self.lowercase,
            )
            for text in X
        ]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(list(input_features), dtype=object)


class HeadText(BaseEstimator, TransformerMixin):
    """Keep only the start of a message for the character model."""

    def __init__(self, max_chars: int = 4_000):
        self.max_chars = max_chars

    def fit(self, X: Sequence[str], y: object = None) -> "HeadText":
        return self

    def transform(self, X: Sequence[str]) -> list[str]:
        return [str(text)[: self.max_chars] for text in X]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(list(input_features), dtype=object)


class TextStats(BaseEstimator, TransformerMixin):
    """Optional content stats such as links, punctuation, and spam words."""

    FEATURE_NAMES: tuple[str, ...] = (
        "punctuation_ratio",
        "non_ascii_ratio",
        "exclamation_per_kchar",
        "question_per_kchar",
        "dollar_per_kchar",
        "percent_per_kchar",
        "url_count",
        "email_address_count",
        "repeated_punctuation",
        "urgency_hits",
        "money_hits",
        "marketing_hits",
    )

    def __init__(self, max_chars: int = 20_000):
        self.max_chars = max_chars

    def fit(self, X: Sequence[str], y: object = None) -> "TextStats":
        return self

    def transform(self, X: Sequence[str]) -> np.ndarray:
        rows = [self._row(normalize_text(text, self.max_chars)) for text in X]
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def _row(text: str) -> list[float]:
        chars = len(text)
        lowers = text.lower()

        # Avoid dividing by zero for empty messages.
        safe_chars = chars + 1

        non_ascii = sum(ord(ch) > 127 for ch in text)
        punctuation = sum(bool(_NON_WORD_RE.match(ch)) for ch in text)

        urgency = sum(lowers.count(token) for token in _URGENCY_WORDS)
        money = sum(lowers.count(token) for token in _MONEY_WORDS)
        marketing = sum(lowers.count(token) for token in _MARKETING_WORDS)

        return [
            punctuation / safe_chars,
            non_ascii / safe_chars,
            text.count("!") * 1000.0 / safe_chars,
            text.count("?") * 1000.0 / safe_chars,
            text.count("$") * 1000.0 / safe_chars,
            text.count("%") * 1000.0 / safe_chars,
            float(lowers.split().count("urltoken")),
            float(lowers.split().count("emailtoken")),
            float(len(_REPEATED_PUNCT_RE.findall(text))),
            float(urgency),
            float(money),
            float(marketing),
        ]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(self.FEATURE_NAMES, dtype=object)


def build_feature_union(
    *,
    word_max_features: int = 150_000,
    word_min_df: int = 2,
    word_ngram_range: tuple[int, int] = (1, 2),
    char_max_features: int = 80_000,
    char_min_df: int = 5,
    char_ngram_range: tuple[int, int] = (3, 5),
    char_max_chars: int = 4_000,
    char_weight: float = 0.2,
    include_stats: bool = False,
    stats_weight: float = 0.1,
) -> FeatureUnion:
    """Build the feature views used by the model."""

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        min_df=word_min_df,
        max_df=0.95,
        max_features=word_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
    )
    char = Pipeline(
        steps=[
            ("head", HeadText(max_chars=char_max_chars)),
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=char_ngram_range,
                    min_df=char_min_df,
                    max_features=char_max_features,
                    sublinear_tf=True,
                    lowercase=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    transformer_list: list[tuple[str, BaseEstimator]] = [
        ("word", word),
        ("char", char),
    ]
    transformer_weights: dict[str, float] = {
        "word": 1.0,
        # Character patterns help with obfuscation, so keep them light.
        "char": char_weight,
    }

    if include_stats:
        stats = Pipeline(
            steps=[
                ("stats", TextStats()),
                # Scale the columns so they are comparable.
                ("scale", MaxAbsScaler()),
            ]
        )
        transformer_list.append(("stats", stats))
        transformer_weights["stats"] = stats_weight

    return FeatureUnion(
        transformer_list=transformer_list,
        n_jobs=None,
        transformer_weights=transformer_weights,
    )


def build_pipeline(
    classifier: BaseEstimator,
    **feature_kwargs: object,
) -> Pipeline:
    """Build the complete text -> classifier pipeline."""

    return Pipeline(
        steps=[
            ("normalize", TextNormalizer()),
            ("features", build_feature_union(**feature_kwargs)),
            ("clf", classifier),
        ]
    )
