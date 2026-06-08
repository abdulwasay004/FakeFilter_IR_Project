from pathlib import Path

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation import cluster_purity


def _texts(documents):
    return [" ".join(tokens) for tokens in documents]


def run_kmeans(documents, labels, output_dir, max_k=10, k=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(_texts(documents))

    inertias = []
    k_values = range(2, min(max_k, len(documents) - 1) + 1)
    for candidate in k_values:
        model = KMeans(n_clusters=candidate, random_state=42, n_init=10)
        model.fit(x)
        inertias.append(model.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_values), inertias, marker="o")
    plt.xlabel("K")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("K-Means Elbow Method")
    plt.tight_layout()
    elbow_path = output_dir / "kmeans_elbow.png"
    plt.savefig(elbow_path, dpi=160)
    plt.close()

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    assignments = model.fit_predict(x)
    terms = vectorizer.get_feature_names_out()
    top_terms = []
    for cluster_id in range(k):
        center = model.cluster_centers_[cluster_id]
        indexes = center.argsort()[-10:][::-1]
        top_terms.append([terms[index] for index in indexes])

    reduced = TruncatedSVD(n_components=2, random_state=42).fit_transform(x)
    plt.figure(figsize=(7, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=assignments, s=12, cmap="tab10")
    plt.title("K-Means Clusters")
    plt.tight_layout()
    scatter_path = output_dir / "kmeans_clusters.png"
    plt.savefig(scatter_path, dpi=160)
    plt.close()

    return {
        "assignments": assignments,
        "purity": cluster_purity(assignments, labels),
        "top_terms": top_terms,
        "elbow_path": elbow_path,
        "scatter_path": scatter_path,
    }


def run_hierarchical(documents, labels, output_dir, sample_size=250, k=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs = documents[:sample_size]
    sample_labels = labels[:sample_size]
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(_texts(docs)).toarray()

    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    assignments = model.fit_predict(x)
    linked = linkage(x, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(linked, truncate_mode="lastp", p=30)
    plt.title("Agglomerative Hierarchical Clustering")
    plt.tight_layout()
    dendrogram_path = output_dir / "hierarchical_dendrogram.png"
    plt.savefig(dendrogram_path, dpi=160)
    plt.close()
    return {
        "assignments": assignments,
        "purity": cluster_purity(assignments, sample_labels),
        "dendrogram_path": dendrogram_path,
    }

