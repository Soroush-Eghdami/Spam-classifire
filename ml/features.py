"""Feature engineering for the Sift spam classifier.

Three views of every message are combined:

1. **Word TF-IDF** (1-2 grams) captures phrases such as "act now", "click
   here", "unclaimed prize".  Sublinear term frequency stops a single word
   repeated a hundred times from dominating the vector.
2. **Character TF-IDF** (3-5 grams, word-boundary aware) captures obfuscated
   spam ("fr33 m0ney", "V1AGRA", "p@yout") and language-agnostic morphology.
   It only looks at the first ``char_max_chars`` characters of a message to
   keep the feature matrix small enough for a laptop.
3. **Dense text statistics** capture signals that bag-of-words misses: link
   density, shouting, digit runs, punctuation abuse, and keyword families
   (urgency / money / marketing).

Everything here is in a separate module (rather than inside the training
script) because the fitted pipeline is pickled with joblib.  The backend must
be able to import the exact same classes when it unpickles the artifact.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

#: Matches one or more characters that are not letters, digits, or whitespace.
_NON_WORD_RE = re.compile(r"[^\w\s]")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_HTML_RE = re.compile(r"<[^>]{1,80}>")
_REPEATED_PUNCT_RE = re.compile(r"[!?]{2,}")

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


def normalize_text(text: object, max_chars: int = 20_000) -> str:
    """Return a safe, whitespace-normalised string, truncated to a budget.

    Newlines are preserved because :class:`TextStats` measures line counts.
    Null bytes are removed because they break a few vectorizer regexes.
    """

    if text is None:
        return ""
    value = str(text).replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = value.strip()
    if len(value) > max_chars:
        value = value[:max_chars]
    return value


class HeadText(BaseEstimator, TransformerMixin):
    """Keep only the first ``max_chars`` characters of every document.

    Character n-grams are expensive on long email threads.  The opening of a
    message carries the subject line and the call to action, which is where
    most obfuscated spam lives, so trimming the tail is a good trade.
    """

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
    """Turn a message into a handful of dense, scale-free numeric signals.

    The transformer is stateless, so it works inside a scikit-learn pipeline
    without any fitting.  A :class:`StandardScaler` downstream makes the
    columns comparable to one another.
    """

    FEATURE_NAMES: tuple[str, ...] = (
        "log_chars",
        "log_words",
        "avg_word_len",
        "max_word_len",
        "unique_word_ratio",
        "digit_ratio",
        "uppercase_ratio",
        "punctuation_ratio",
        "non_ascii_ratio",
        "exclamation_per_kchar",
        "question_per_kchar",
        "dollar_per_kchar",
        "percent_per_kchar",
        "url_count",
        "email_address_count",
        "html_tag_count",
        "number_token_ratio",
        "caps_word_ratio",
        "log_lines",
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
        words = text.split()
        word_count = len(words)
        lowers = text.lower()

        # Ratios use a +1 denominator so an empty message returns 0, not NaN.
        safe_chars = chars + 1
        safe_words = word_count + 1

        digits = sum(ch.isdigit() for ch in text)
        uppers = sum(ch.isupper() for ch in text)
        non_ascii = sum(ord(ch) > 127 for ch in text)
        punctuation = sum(bool(_NON_WORD_RE.match(ch)) for ch in text)

        word_lengths = [len(word) for word in words]
        unique_words = {word.lower() for word in words}
        number_tokens = sum(any(ch.isdigit() for ch in word) for word in words)
        caps_words = sum(
            1 for word in words if len(word) > 2 and word.isupper()
        )

        urgency = sum(lowers.count(token) for token in _URGENCY_WORDS)
        money = sum(lowers.count(token) for token in _MONEY_WORDS)
        marketing = sum(lowers.count(token) for token in _MARKETING_WORDS)

        return [
            np.log1p(chars),
            np.log1p(word_count),
            float(np.mean(word_lengths)) if word_lengths else 0.0,
            float(max(word_lengths)) if word_lengths else 0.0,
            len(unique_words) / safe_words,
            digits / safe_chars,
            uppers / safe_chars,
            punctuation / safe_chars,
            non_ascii / safe_chars,
            text.count("!") * 1000.0 / safe_chars,
            text.count("?") * 1000.0 / safe_chars,
            text.count("$") * 1000.0 / safe_chars,
            text.count("%") * 1000.0 / safe_chars,
            float(len(_URL_RE.findall(text))),
            float(len(_EMAIL_RE.findall(text))),
            float(len(_HTML_RE.findall(text))),
            number_tokens / safe_words,
            caps_words / safe_words,
            np.log1p(text.count("\n") + 1),
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
) -> FeatureUnion:
    """Create the shared text feature space used by every candidate model."""

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
    stats = Pipeline(
        steps=[
            ("stats", TextStats()),
            # MaxAbs keeps every statistic non-negative, which ComplementNB
            # requires, while still putting the columns on a comparable scale.
            ("scale", MaxAbsScaler()),
        ]
    )

    return FeatureUnion(
        transformer_list=[
            ("word", word),
            ("char", char),
            ("stats", stats),
        ],
        n_jobs=None,
        transformer_weights={
            "word": 1.0,
            # Character n-grams are useful but correlated with the word view.
            "char": 0.65,
            # Hand-made signals should nudge, not dominate, the decision.
            "stats": 0.35,
        },
    )


def build_pipeline(
    classifier: BaseEstimator,
    **feature_kwargs: object,
) -> Pipeline:
    """Assemble the deployable ``features -> classifier`` pipeline."""

    return Pipeline(
        steps=[
            ("features", build_feature_union(**feature_kwargs)),
            ("clf", classifier),
        ]
    )
