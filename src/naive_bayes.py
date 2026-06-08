import math
from collections import defaultdict


class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words = defaultdict(int)
        self.vocabulary = set()
        self.total_documents = 0
        self.labels = []

    def fit(self, documents, labels):
        self.labels = sorted(set(labels))
        self.total_documents = len(documents)
        for tokens, label in zip(documents, labels):
            self.class_counts[label] += 1
            for word in tokens:
                self.vocabulary.add(word)
                self.word_counts[label][word] += 1
                self.total_words[label] += 1

    def predict_single(self, tokens):
        prediction, _ = self.predict_with_scores(tokens)
        return prediction

    def predict_with_scores(self, tokens):
        vocab_size = max(len(self.vocabulary), 1)
        scores = {}
        for label in self.labels:
            prior = math.log(self.class_counts[label] / self.total_documents)
            likelihood = 0.0
            denominator = self.total_words[label] + vocab_size
            for word in tokens:
                likelihood += math.log((self.word_counts[label].get(word, 0) + 1) / denominator)
            scores[label] = prior + likelihood
        return max(scores, key=scores.get), scores

    def predict(self, documents):
        return [self.predict_single(tokens) for tokens in documents]

