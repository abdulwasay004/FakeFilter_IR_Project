import argparse
from pathlib import Path

from evaluation import classifier_metrics, mean_average_precision
from naive_bayes import NaiveBayesClassifier
from preprocess import Preprocessor
from retrieval import TFIDFRetriever
from vector_classifiers import KNNClassifier, RocchioClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def load_project():
    processor = Preprocessor()
    train_df = processor.load_data(DATA / "train.tsv")
    valid_df = processor.load_data(DATA / "valid.tsv")
    test_df = processor.load_data(DATA / "test.tsv")
    train_docs, train_labels = processor.preprocess_dataframe(train_df)
    valid_docs, valid_labels = processor.preprocess_dataframe(valid_df)
    test_docs, test_labels = processor.preprocess_dataframe(test_df)
    retriever = TFIDFRetriever()
    retriever.build(train_docs)
    test_vectors = {i: retriever.vectorize_tokens(tokens) for i, tokens in enumerate(test_docs)}
    return {
        "processor": processor,
        "train_df": train_df,
        "test_df": test_df,
        "train_docs": train_docs,
        "train_labels": train_labels,
        "valid_docs": valid_docs,
        "valid_labels": valid_labels,
        "test_docs": test_docs,
        "test_labels": test_labels,
        "retriever": retriever,
        "test_vectors": test_vectors,
    }


def build_models(project, k=5):
    nb = NaiveBayesClassifier()
    nb.fit(project["train_docs"], project["train_labels"])
    rocchio = RocchioClassifier(project["retriever"].cosine_similarity)
    rocchio.fit(project["retriever"].doc_vectors, project["train_labels"])
    knn = KNNClassifier(project["retriever"].cosine_similarity, k=k)
    knn.fit(project["retriever"].doc_vectors, project["train_labels"])
    return nb, rocchio, knn


def cmd_evaluate(args):
    project = load_project()
    nb, rocchio, knn = build_models(project, args.k)
    models = {
        "Naive Bayes": nb.predict(project["test_docs"]),
        "Rocchio": rocchio.predict(project["test_vectors"]),
        f"KNN (k={args.k})": knn.predict(project["test_vectors"]),
    }
    REPORTS.mkdir(exist_ok=True)
    lines = ["FakeFilter Evaluation Report", "=" * 28, ""]
    for name, predictions in models.items():
        metrics = classifier_metrics(project["test_labels"], predictions)
        lines += [
            name,
            "-" * len(name),
            f"Accuracy: {metrics['accuracy']:.4f}",
            "Confusion Matrix labels: REAL, FAKE",
            str(metrics["confusion_matrix"]),
            metrics["report"],
            "",
        ]
        print("\n".join(lines[-7:]))

    query_terms = [
        project["processor"].preprocess_text("health care taxes election claims"),
        project["processor"].preprocess_text("obama economy jobs policy"),
        project["processor"].preprocess_text("crime immigration government spending"),
    ]
    map_score = mean_average_precision(query_terms, project["retriever"], project["train_labels"], top_k=args.top_k)
    lines += [f"MAP@{args.top_k} for fake-relevant sample queries: {map_score:.4f}", ""]
    report_path = REPORTS / "evaluation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report: {report_path}")


def cmd_query(args):
    project = load_project()
    nb, rocchio, knn = build_models(project, args.k)
    tokens = project["processor"].preprocess_text(args.text)
    results = project["retriever"].retrieve_tokens(tokens, top_k=args.top_k)
    print(f"\nQuery: {args.text}")
    print(f"NB prediction for query text: {nb.predict_single(tokens)}")
    print(f"Rocchio prediction for query text: {rocchio.predict_single(project['retriever'].vectorize_tokens(tokens))}")
    print(f"KNN prediction for query text: {knn.predict_single(project['retriever'].vectorize_tokens(tokens))}\n")
    for rank, (doc_id, score) in enumerate(results, start=1):
        row = project["train_df"].iloc[doc_id]
        print(f"{rank}. score={score:.4f} label={project['train_labels'][doc_id]} original={row['label']}")
        print(f"   {row['statement']}")


def cmd_boolean(args):
    project = load_project()
    terms = project["processor"].preprocess_text(args.text)
    if args.operator == "AND":
        matches = project["retriever"].index.boolean_and(terms)
    elif args.operator == "OR":
        matches = project["retriever"].index.boolean_or(terms)
    else:
        matches = project["retriever"].index.phrase_search(terms)
    print(f"Found {len(matches)} documents")
    for doc_id in list(matches)[: args.top_k]:
        print(f"{doc_id}: {project['train_df'].iloc[doc_id]['statement']}")


def cmd_cluster(args):
    from clustering import run_hierarchical, run_kmeans

    project = load_project()
    REPORTS.mkdir(exist_ok=True)
    subset_docs = project["train_docs"]
    subset_labels = project["train_labels"]
    if args.fake_only:
        pairs = [(doc, label) for doc, label in zip(subset_docs, subset_labels) if label == "FAKE"]
        subset_docs = [doc for doc, _ in pairs]
        subset_labels = [label for _, label in pairs]
    kmeans = run_kmeans(subset_docs, subset_labels, REPORTS, max_k=args.max_k, k=args.k)
    hierarchical = run_hierarchical(subset_docs, subset_labels, REPORTS, sample_size=args.sample_size, k=args.k)
    print(f"K-Means purity: {kmeans['purity']:.4f}")
    for index, terms in enumerate(kmeans["top_terms"], start=1):
        print(f"Cluster {index}: {', '.join(terms)}")
    print(f"Hierarchical purity on sample: {hierarchical['purity']:.4f}")
    print(f"Saved plots in: {REPORTS}")


def build_parser():
    parser = argparse.ArgumentParser(description="FakeFilter IR misinformation detection system")
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate NB, Rocchio, KNN and MAP")
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--top-k", type=int, default=25)
    evaluate.set_defaults(func=cmd_evaluate)
    query = sub.add_parser("query", help="Rank documents and classify the query")
    query.add_argument("text")
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--k", type=int, default=5)
    query.set_defaults(func=cmd_query)
    boolean = sub.add_parser("boolean", help="Boolean or phrase retrieval")
    boolean.add_argument("text")
    boolean.add_argument("--operator", choices=["AND", "OR", "PHRASE"], default="AND")
    boolean.add_argument("--top-k", type=int, default=10)
    boolean.set_defaults(func=cmd_boolean)
    cluster = sub.add_parser("cluster", help="Run K-Means and hierarchical clustering")
    cluster.add_argument("--k", type=int, default=5)
    cluster.add_argument("--max-k", type=int, default=10)
    cluster.add_argument("--sample-size", type=int, default=250)
    cluster.add_argument("--fake-only", action="store_true")
    cluster.set_defaults(func=cmd_cluster)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
