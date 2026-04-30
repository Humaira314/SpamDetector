from __future__ import annotations

import re
from typing import Callable, List, Optional, Set

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:  # pragma: no cover - nltk may be unavailable
    nltk_stopwords = None

try:
    from nltk.stem import PorterStemmer
except Exception:  # pragma: no cover - nltk may be unavailable
    PorterStemmer = None

URL_RE = re.compile(r"(http\S+|www\.\S+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def get_stopwords(use_nltk: bool = True) -> Set[str]:
    if use_nltk and nltk_stopwords is not None:
        try:
            return set(nltk_stopwords.words("english"))
        except LookupError:
            return set(ENGLISH_STOP_WORDS)
    return set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" url ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


class TextTokenizer:
    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        use_stemming: bool = False,
    ) -> None:
        self.stop_words = stop_words or set()
        self.use_stemming = use_stemming

    def __call__(self, text: str) -> List[str]:
        text = clean_text(text)
        tokens = text.split()
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        if self.use_stemming and PorterStemmer is not None:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]
        return tokens


def build_tokenizer(
    use_stemming: bool = False,
    use_stopwords: bool = True,
    use_nltk_stopwords: bool = True,
) -> Callable[[str], List[str]]:
    stop_words = get_stopwords(use_nltk_stopwords) if use_stopwords else set()
    return TextTokenizer(stop_words=stop_words, use_stemming=use_stemming)


def build_pipeline(
    model: str = "nb",
    use_stemming: bool = False,
    use_stopwords: bool = True,
    use_nltk_stopwords: bool = True,
    calibrate_svm: bool = True,
) -> Pipeline:
    tokenizer = build_tokenizer(
        use_stemming=use_stemming,
        use_stopwords=use_stopwords,
        use_nltk_stopwords=use_nltk_stopwords,
    )

    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        preprocessor=None,
        lowercase=False,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
    )

    if model == "nb":
        classifier = MultinomialNB()
    elif model == "svm":
        base = LinearSVC()
        classifier = (
            CalibratedClassifierCV(base, method="sigmoid", cv=3)
            if calibrate_svm
            else base
        )
    else:
        raise ValueError(f"Unsupported model type: {model}")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])
