# FakeFilter Project Viva Guide

## 1. Project Title

**FakeFilter: Misinformation Detection via Text Classification and Clustering**

FakeFilter is an Information Retrieval project that detects whether a political/news statement is likely `REAL` or `FAKE`. It combines classical IR techniques such as inverted indexing, TF-IDF ranked retrieval, Naive Bayes classification, vector-space classification, KNN, and clustering.

## 2. Main Goal

The goal of this project is to build a complete misinformation detection system that can:

- Store and search a news/claim corpus using an inverted index.
- Retrieve ranked documents for a user query using TF-IDF and cosine similarity.
- Classify statements as `REAL` or `FAKE`.
- Compare multiple classifiers.
- Discover hidden groups/topics using clustering.
- Evaluate performance using IR and ML metrics.

## 3. Dataset Used

The project uses the LIAR-style dataset, stored in:

```bash
data/train.tsv
data/valid.tsv
data/test.tsv
```

Each row contains a political/news statement with metadata and an original truth label such as:

```text
true
mostly-true
half-true
barely-true
false
pants-fire
```

For this project, labels are converted into two classes:

```text
true, mostly-true -> REAL
all other labels -> FAKE
```

This makes the task a binary classification problem.

## 4. Project Folder Structure

```bash
FakeFilter_complete/
|
├── data/
|   ├── train.tsv
|   ├── valid.tsv
|   └── test.tsv
|
├── src/
|   ├── main.py
|   ├── preprocess.py
|   ├── indexer.py
|   ├── retrieval.py
|   ├── naive_bayes.py
|   ├── vector_classifiers.py
|   ├── clustering.py
|   └── evaluation.py
|
├── reports/
|   └── evaluation_report.txt
|
├── requirements.txt
└── README.md
```

## 5. File-by-File Explanation

### `main.py`

This is the main entry point of the project.

It provides command-line features using `argparse`.

Main commands:

```bash
evaluate
query
boolean
cluster
```

It connects all modules together: preprocessing, indexing, retrieval, classification, evaluation, and clustering.

### `preprocess.py`

This file handles text preparation.

It performs:

- Dataset loading
- Label conversion
- Lowercasing
- Removing punctuation and numbers
- Tokenization
- Stopword removal
- Optional stemming

Example:

```text
The government increased health care taxes!
```

becomes:

```text
["govern", "increas", "health", "care", "tax"]
```

This makes text cleaner and easier to index/classify.

### `indexer.py`

This file implements the positional inverted index.

An inverted index maps each term to the documents where it appears.

Example:

```text
health -> doc 3: [2, 8], doc 10: [5]
care   -> doc 3: [3], doc 15: [7]
```

Because positions are stored, the system supports:

- Term search
- Boolean AND search
- Boolean OR search
- Phrase search

Example phrase query:

```text
health care
```

The system checks whether `health` and `care` appear next to each other.

### `retrieval.py`

This file implements ranked retrieval using:

- TF-IDF weighting
- Vector Space Model
- Cosine similarity

TF-IDF gives higher importance to words that are frequent in a document but rare in the corpus.

Cosine similarity measures how close a query vector is to each document vector.

The output is a ranked list:

```text
1. score=0.3442 label=REAL
2. score=0.3419 label=FAKE
3. score=0.3182 label=REAL
```

### `naive_bayes.py`

This file implements Multinomial Naive Bayes from scratch.

It calculates:

- Prior probability of each class
- Word likelihood for each class
- Laplace smoothing for unseen words

Formula idea:

```text
score(class) = log P(class) + sum(log P(word | class))
```

The class with the highest score becomes the prediction.

This is useful for text classification because it works well with bag-of-words features.

### `vector_classifiers.py`

This file contains two vector-space classifiers.

#### Rocchio Classifier

Rocchio creates one centroid vector for each class:

```text
REAL centroid
FAKE centroid
```

A test document is assigned to the nearest centroid using cosine similarity.

#### KNN Classifier

KNN compares a test document with all training documents.

It selects the top `k` most similar documents and uses majority voting.

Example with `k=5`:

```text
FAKE, FAKE, REAL, FAKE, REAL
```

Prediction:

```text
FAKE
```

Both Rocchio and KNN are implemented from scratch.

### `clustering.py`

This file implements unsupervised clustering.

It includes:

- K-Means clustering
- Elbow method
- K-Means scatter plot
- Agglomerative hierarchical clustering
- Dendrogram generation

K-Means groups similar documents into `k` clusters.

Hierarchical clustering builds a tree-like structure showing document/topic relationships.

Generated files:

```bash
reports/kmeans_elbow.png
reports/kmeans_clusters.png
reports/hierarchical_dendrogram.png
```

### `evaluation.py`

This file calculates evaluation metrics.

Implemented metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- MAP
- Cluster purity

Accuracy tells how many predictions were correct overall.

Precision tells how reliable positive predictions are.

Recall tells how many true class items were found.

F1-score balances precision and recall.

MAP evaluates ranked retrieval quality.

Cluster purity checks how well clusters match actual labels.

## 6. Implemented Proposal Features

The project implements:

- Inverted index
- Positional index
- Boolean retrieval
- Phrase retrieval
- Ranked TF-IDF retrieval
- Naive Bayes from scratch
- Rocchio classifier
- KNN classifier
- K-Means clustering
- Agglomerative hierarchical clustering
- Elbow method
- Dendrogram
- Cluster purity
- Accuracy, precision, recall, F1-score
- Confusion matrix
- MAP
- Command-line query interface
- Evaluation report generation

## 7. Features Partially Implemented

Some features are implemented but with practical limitations:

- The validation set is loaded but not used for tuning.
- Elbow plot is generated, but best `K` is selected manually.
- Suspicious document retrieval is based on ranking plus labels/predictions, not a separate suspiciousness formula.
- The evaluation report is text-based, not a formatted PDF or DOCX.

## 8. Features Not Implemented

The following proposal ideas are not fully implemented:

- Automatic best-K detection from elbow curve.
- Separate scikit-learn KNN baseline comparison.
- Advanced class imbalance handling.
- Polished academic report with embedded charts.

## 9. How to Run the Project

### Step 1: Open the Project Folder

In PowerShell or Command Prompt:

```bash
cd "C:\Users\pakistanbusiness.biz\Documents\Codex\2026-06-08\files-mentioned-by-the-user-fakefilter\outputs\FakeFilter_complete"
```

### Step 2: Check Python

```bash
python --version
```

Use Python 3.10 or newer.

If Python is not installed, install it from:

```text
https://www.python.org/downloads/
```

During installation, enable:

```text
Add Python to PATH
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Required libraries:

```text
pandas
numpy
nltk
scikit-learn
matplotlib
scipy
```

## 10. Commands to Run Every Feature

### A. Run Full Evaluation

```bash
python src\main.py evaluate
```

This runs:

- Naive Bayes
- Rocchio
- KNN
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- MAP

Output report:

```bash
reports\evaluation_report.txt
```

### B. Run Evaluation with Custom KNN Value

```bash
python src\main.py evaluate --k 7
```

This changes KNN from default `k=5` to `k=7`.

### C. Run Ranked Retrieval and Classification

```bash
python src\main.py query "health care taxes election claims" --top-k 5
```

This shows:

- Query prediction by Naive Bayes
- Query prediction by Rocchio
- Query prediction by KNN
- Top 5 ranked documents
- Document scores
- Actual labels

Another example:

```bash
python src\main.py query "government spending and taxes" --top-k 10
```

### D. Run Boolean AND Search

```bash
python src\main.py boolean "health care" --operator AND
```

This returns documents containing both terms.

### E. Run Boolean OR Search

```bash
python src\main.py boolean "health care" --operator OR
```

This returns documents containing either term.

### F. Run Phrase Search

```bash
python src\main.py boolean "health care" --operator PHRASE
```

This returns documents where the exact phrase appears.

### G. Run K-Means and Hierarchical Clustering

```bash
python src\main.py cluster --k 5
```

This generates:

```bash
reports\kmeans_elbow.png
reports\kmeans_clusters.png
reports\hierarchical_dendrogram.png
```

### H. Run Clustering Only on Fake Documents

```bash
python src\main.py cluster --k 5 --fake-only
```

This focuses clustering on misinformation-style documents.

### I. Run Clustering with a Different K

```bash
python src\main.py cluster --k 6
```

### J. Run Clustering with Larger Hierarchical Sample

```bash
python src\main.py cluster --k 5 --sample-size 500
```

## 11. Recommended Viva Demo Flow

Use this order during your viva:

### 1. Show Evaluation

```bash
python src\main.py evaluate
```

Explain that this compares Naive Bayes, Rocchio, and KNN.

### 2. Show Ranked Retrieval

```bash
python src\main.py query "health care taxes election claims" --top-k 5
```

Explain TF-IDF, cosine similarity, and classifier predictions.

### 3. Show Boolean Search

```bash
python src\main.py boolean "health care" --operator AND
```

Explain inverted index retrieval.

### 4. Show Phrase Search

```bash
python src\main.py boolean "health care" --operator PHRASE
```

Explain positional index.

### 5. Show Clustering

```bash
python src\main.py cluster --k 5
```

Explain K-Means, elbow method, hierarchical clustering, dendrogram, and purity.

## 12. Viva Explanation of IR Concepts

### Inverted Index

An inverted index allows fast search by storing terms as keys and document IDs as values.

Instead of scanning every document, the system directly checks the postings list of query terms.

### Positional Index

A positional index stores word positions.

This allows phrase search because the system can check if words appear consecutively.

### TF-IDF

TF-IDF measures word importance.

A word gets a high score if:

- It appears often in one document.
- It does not appear in many documents.

This helps reduce the importance of common words.

### Cosine Similarity

Cosine similarity compares two vectors.

In this project, it compares:

```text
query vector vs document vector
```

or:

```text
test document vector vs class centroid
```

Higher cosine similarity means more similar text.

### Naive Bayes

Naive Bayes is a probabilistic classifier.

It assumes words are conditionally independent given the class.

Even though this assumption is simple, it works well for text classification.

### Rocchio

Rocchio is a vector-space classifier.

It calculates the average vector of each class and assigns a new document to the closest class centroid.

### KNN

KNN is instance-based classification.

It does not create a model like Naive Bayes. Instead, it compares a new document with training documents and chooses the majority label among the nearest neighbors.

### K-Means

K-Means is flat clustering.

It groups documents into `K` clusters based on similarity.

### Hierarchical Clustering

Hierarchical clustering creates a tree of document relationships.

The dendrogram helps visualize which documents/topics are closer to each other.

### Cluster Purity

Cluster purity measures how clean the clusters are.

If most documents in a cluster belong to the same class, purity is high.

### MAP

MAP stands for Mean Average Precision.

It evaluates ranked retrieval by checking whether relevant documents appear near the top of the result list.

## 13. Expected Output Summary

After running the full system, you should be able to show:

- Classifier comparison
- Ranked search results
- Boolean search results
- Phrase search results
- Evaluation report
- K-Means elbow plot
- K-Means cluster plot
- Hierarchical dendrogram

## 14. Final Viva Summary

FakeFilter is a complete IR-based misinformation detection system. It demonstrates indexing, retrieval, classification, clustering, and evaluation. The project does not rely on black-box AI APIs. Instead, it uses classical Information Retrieval and Machine Learning concepts such as inverted indexes, TF-IDF, cosine similarity, Naive Bayes, Rocchio, KNN, K-Means, hierarchical clustering, MAP, and cluster purity.

The project is suitable for an IR viva because every major component connects directly to the Information Retrieval syllabus.

