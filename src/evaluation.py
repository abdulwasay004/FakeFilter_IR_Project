from collections import defaultdict


def classifier_metrics(true_labels, predictions):
    labels = ["REAL", "FAKE"]
    total = len(true_labels)
    accuracy = sum(1 for true, pred in zip(true_labels, predictions) if true == pred) / total
    matrix = []
    report_lines = ["              precision    recall  f1-score   support"]
    for label in labels:
        tp = sum(1 for true, pred in zip(true_labels, predictions) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(true_labels, predictions) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(true_labels, predictions) if true == label and pred != label)
        support = sum(1 for true in true_labels if true == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        report_lines.append(f"{label:>12} {precision:>10.2f} {recall:>9.2f} {f1:>9.2f} {support:>9}")
        matrix.append([
            sum(1 for true, pred in zip(true_labels, predictions) if true == label and pred == predicted)
            for predicted in labels
        ])
    return {
        "accuracy": accuracy,
        "report": "\n".join(report_lines),
        "confusion_matrix": matrix,
    }


def average_precision(ranked_doc_ids, relevant_doc_ids):
    relevant = set(relevant_doc_ids)
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def mean_average_precision(queries, retriever, labels, relevant_label="FAKE", top_k=25):
    scores = []
    relevant = [doc_id for doc_id, label in enumerate(labels) if label == relevant_label]
    for query_tokens in queries:
        ranked = [doc_id for doc_id, _ in retriever.retrieve_tokens(query_tokens, top_k=top_k)]
        scores.append(average_precision(ranked, relevant))
    return sum(scores) / len(scores) if scores else 0.0


def cluster_purity(assignments, labels):
    clusters = defaultdict(list)
    for cluster_id, label in zip(assignments, labels):
        clusters[cluster_id].append(label)
    correct = 0
    for cluster_labels in clusters.values():
        correct += max(cluster_labels.count("REAL"), cluster_labels.count("FAKE"))
    return correct / len(labels) if labels else 0.0
