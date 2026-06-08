import re
from pathlib import Path

import pandas as pd

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover - fallback for minimal environments
    nltk = None
    stopwords = None
    PorterStemmer = None
    word_tokenize = None


LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true",
    "false",
    "half_true",
    "mostly_true",
    "pants_fire",
    "context",
]


class Preprocessor:
    def __init__(self, use_stemming=True, remove_stopwords=True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self._stemmer = PorterStemmer() if PorterStemmer and use_stemming else None
        self.stop_words = self._load_stopwords() if remove_stopwords else set()

    def _load_stopwords(self):
        if not nltk or not stopwords:
            return {
                "a", "an", "and", "are", "as", "at", "be", "by", "for",
                "from", "has", "he", "in", "is", "it", "its", "of", "on",
                "that", "the", "to", "was", "were", "will", "with",
            }
        try:
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("english"))

    def load_data(self, filepath):
        path = Path(filepath)
        return pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)

    def convert_label(self, label):
        return "REAL" if str(label).strip() in {"true", "mostly-true"} else "FAKE"

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text):
        if word_tokenize and nltk:
            try:
                tokens = word_tokenize(text)
            except LookupError:
                nltk.download("punkt", quiet=True)
                try:
                    nltk.download("punkt_tab", quiet=True)
                except Exception:
                    pass
                tokens = word_tokenize(text)
        else:
            tokens = text.split()

        processed = []
        for token in tokens:
            if token in self.stop_words or len(token) < 2:
                continue
            if self._stemmer:
                token = self._stemmer.stem(token)
            processed.append(token)
        return processed

    def preprocess_text(self, text):
        return self.tokenize(self.clean_text(text))

    def preprocess_dataframe(self, df):
        docs = [self.preprocess_text(text) for text in df["statement"]]
        labels = [self.convert_label(label) for label in df["label"]]
        return docs, labels

