from collections import Counter, defaultdict


class PositionalInvertedIndex:
    def __init__(self):
        self.index = defaultdict(lambda: defaultdict(list))
        self.doc_lengths = {}
        self.total_docs = 0

    def build_index(self, documents):
        self.total_docs = len(documents)
        for doc_id, tokens in enumerate(documents):
            self.doc_lengths[doc_id] = len(tokens)
            for position, term in enumerate(tokens):
                self.index[term][doc_id].append(position)

    def search_term(self, term):
        return self.index.get(term, {})

    def term_frequency(self, term, doc_id):
        return len(self.index.get(term, {}).get(doc_id, []))

    def document_frequency(self, term):
        return len(self.index.get(term, {}))

    def get_vocabulary(self):
        return list(self.index.keys())

    def vocabulary_size(self):
        return len(self.index)

    def boolean_and(self, terms):
        if not terms:
            return set()
        result = set(self.search_term(terms[0]).keys())
        for term in terms[1:]:
            result &= set(self.search_term(term).keys())
        return result

    def boolean_or(self, terms):
        result = set()
        for term in terms:
            result |= set(self.search_term(term).keys())
        return result

    def phrase_search(self, terms):
        if not terms:
            return set()
        candidates = self.boolean_and(terms)
        matches = set()
        for doc_id in candidates:
            starts = set(self.index[terms[0]][doc_id])
            for offset, term in enumerate(terms[1:], start=1):
                starts &= {pos - offset for pos in self.index[term][doc_id]}
            if starts:
                matches.add(doc_id)
        return matches

    def collection_term_counts(self):
        counts = Counter()
        for term, postings in self.index.items():
            counts[term] = sum(len(positions) for positions in postings.values())
        return counts

