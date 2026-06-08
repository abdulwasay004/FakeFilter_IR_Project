from collections import Counter, defaultdict


class RocchioClassifier:
    def __init__(self, similarity):
        self.similarity = similarity
        self.centroids = {}

    def fit(self, doc_vectors, labels):
        sums = defaultdict(dict)
        counts = Counter(labels)
        for doc_id, vector in doc_vectors.items():
            label = labels[doc_id]
            for term, value in vector.items():
                sums[label][term] = sums[label].get(term, 0.0) + value
        self.centroids = {
            label: {term: value / counts[label] for term, value in vector.items()}
            for label, vector in sums.items()
        }

    def predict_single(self, vector):
        scores = {
            label: self.similarity(vector, centroid)
            for label, centroid in self.centroids.items()
        }
        return max(scores, key=scores.get)

    def predict(self, doc_vectors):
        return [self.predict_single(doc_vectors[doc_id]) for doc_id in sorted(doc_vectors)]


class KNNClassifier:
    def __init__(self, similarity, k=5):
        self.similarity = similarity
        self.k = k
        self.train_vectors = {}
        self.labels = []

    def fit(self, doc_vectors, labels):
        self.train_vectors = doc_vectors
        self.labels = labels

    def predict_single(self, vector):
        neighbors = [
            (doc_id, self.similarity(vector, train_vector))
            for doc_id, train_vector in self.train_vectors.items()
        ]
        neighbors.sort(key=lambda item: item[1], reverse=True)
        votes = Counter(self.labels[doc_id] for doc_id, _ in neighbors[: self.k])
        return votes.most_common(1)[0][0]

    def predict(self, doc_vectors):
        return [self.predict_single(doc_vectors[doc_id]) for doc_id in sorted(doc_vectors)]

