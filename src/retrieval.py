import math

from indexer import PositionalInvertedIndex


class TFIDFRetriever:
    def __init__(self):
        self.index = PositionalInvertedIndex()
        self.documents = []
        self.idf = {}
        self.doc_vectors = {}

    def build(self, documents):
        self.documents = documents
        self.index.build_index(documents)
        self._compute_idf()
        self.doc_vectors = {
            doc_id: self.vectorize_tokens(tokens)
            for doc_id, tokens in enumerate(documents)
        }

    def _compute_idf(self):
        total = max(self.index.total_docs, 1)
        for term in self.index.get_vocabulary():
            df = self.index.document_frequency(term)
            self.idf[term] = math.log((1 + total) / (1 + df)) + 1

    def vectorize_tokens(self, tokens):
        if not tokens:
            return {}
        counts = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        length = len(tokens)
        return {
            term: (freq / length) * self.idf[term]
            for term, freq in counts.items()
            if term in self.idf
        }

    def cosine_similarity(self, left, right):
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(term, 0.0) for term, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def retrieve_tokens(self, tokens, top_k=10):
        query_vector = self.vectorize_tokens(tokens)
        scores = [
            (doc_id, self.cosine_similarity(query_vector, vector))
            for doc_id, vector in self.doc_vectors.items()
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

