import re
import math
from collections import defaultdict, OrderedDict
from nltk.stem import PorterStemmer

class Node:
    def __init__(self, value=None, next_node=None):
        self.value = value
        self.next = next_node
        self.skip = None  # Skip pointer

class LinkedList:
    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.length = 0
        self.n_skips = 0
        self.idf = 0.0
        self.skip_length = None

    def traverse_list(self):
        traversal = []
        current_node = self.start_node
        while current_node:
            traversal.append(current_node.value)
            current_node = current_node.next
        return traversal

    def traverse_skips(self):
        traversal = []
        current_node = self.start_node
        while current_node:
            traversal.append(current_node.value)
            current_node = current_node.skip
        return traversal
    def insert(self, doc_id):
        new_node = Node(doc_id)
        if not self.start_node:
            self.start_node = new_node
            self.end_node = new_node
        else:
            self.end_node.next = new_node
            self.end_node = new_node
        self.length += 1

    def add_skip_connections(self):
        self.n_skips = math.floor(math.sqrt(self.length))
        if self.n_skips * self.n_skips == self.length:
            self.n_skips = self.n_skips - 1
        skip_length = math.floor(math.sqrt(self.length))
        current_node = self.start_node

        for i in range(self.n_skips):
            for _ in range(skip_length):
                if current_node:
                    current_node = current_node.next

        current_skip = self.start_node
        while current_node:
            current_skip.skip = current_node
            current_skip = current_node
            for _ in range(skip_length):
                if current_node:
                    current_node = current_node.next

    def insert_at_end(self, value):
        new_node = Node(value)
        if not self.start_node:
            self.start_node = new_node
            self.end_node = new_node
        else:
            self.end_node.next = new_node
            self.end_node = new_node
        self.length += 1
        
    def get_length(self):
        return self.length

class Preprocessor:
    def __init__(self):
        self.stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'and', 'to'])
        self.ps = PorterStemmer()

    def get_doc_id(self, doc):
        arr = doc.split("\t")
        return int(arr[0]), arr[1]

    def tokenizer(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        tokens = text.split()
        tokens = [token for token in tokens if token not in self.stop_words]
        tokens = [self.ps.stem(token) for token in tokens]
        return tokens

class Indexer:
    def __init__(self):
        self.inverted_index = OrderedDict({})
        self.stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'of', 'and', 'to'])
        self.ps = PorterStemmer()

    def get_index(self):
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        for t in tokenized_document:
            self.add_to_index(t, doc_id)

    def add_to_index(self, term_, doc_id_):
        if term_ not in self.inverted_index:
            self.inverted_index[term_] = LinkedList()

        self.inverted_index[term_].insert(doc_id_)

    def sort_terms(self):
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        for term, postings_list in self.inverted_index.items():
            L = len(postings_list)
            if L <= 1:
                continue

            num_skip_connections = int(math.floor(math.sqrt(L)))
            skip_length = int(round(math.sqrt(L)))
            skip_pointer = postings_list.start_node
            i = 0

            while skip_pointer and i < num_skip_connections:
                for _ in range(skip_length):
                    if skip_pointer:
                        skip_pointer = skip_pointer.next

                if skip_pointer:
                    skip_pointer.skip = skip_pointer
                    i += 1

    def calculate_tf_idf(self, total_num_docs):
        for term, postings_list in self.inverted_index.items():
            for doc_id in postings_list.traverse_list():
                tf = postings_list.get_document_frequency(doc_id) / len(postings_list)
                idf = total_num_docs / len(postings_list)
                tf_idf = tf * idf
                postings_list.idf = tf_idf

    def preprocess_query(self, query):
        query = query.lower()
        query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
        query = ' '.join(query.split())
        query_tokens = query.split()
        query_tokens = [token for token in query_tokens if token not in self.stop_words]
        query_tokens = [self.ps.stem(token) for token in query_tokens]
        return query_tokens

    def get_postings_lists(self, query_terms):
        postings_lists = {}
        for term in query_terms:
            if term in self.inverted_index:
                postings_lists[term] = self.inverted_index[term].traverse_list()
            else:
                postings_lists[term] = []
        return postings_lists

    def daat_and_query(self, query_terms):
        comparisons = 0
        result_docs = []

        postings_lists = self.get_postings_lists(query_terms)
        sorted_postings_lists = {term: sorted(postings) for term, postings in postings_lists.items()}

        doc_pointers = {term: 0 for term in query_terms}

        while all(doc_pointers[term] < len(sorted_postings_lists[term]) for term in query_terms):
            doc_ids = [sorted_postings_lists[term][doc_pointers[term]] for term in query_terms]
            min_doc_id = min(doc_ids)
            comparisons += len(query_terms) - 1

            if all(doc_id == min_doc_id for doc_id in doc_ids):
                result_docs.append(min_doc_id)
                for term in query_terms:
                    doc_pointers[term] += 1
            else:
                min_term = query_terms[doc_ids.index(min_doc_id)]
                doc_pointers[min_term] += 1

        return result_docs, comparisons

    def get_postings_lists_with_skip_pointers(self, query_terms):
        postings_lists = {}
        for term in query_terms:
            if term in inverted_index:
                postings_lists[term] = self.inverted_index[term].traverse_skips()
            else:
                postings_lists[term] = []
        return postings_lists
    
    def daat_and_query_sorted_by_tfidf(self, query_terms):
    # Get the postings lists and corresponding TF-IDF values
        postings_lists = self.get_postings_lists(query_terms)
        tfidf_values = {term: self.inverted_index[term].idf for term in query_terms}
    
    # Initialize result_docs as the first postings list
        result_docs = postings_lists[query_terms[0]]
        comparisons = 0
    
    # Merge postings lists by term using TF-IDF scores
        for term in query_terms[1:]:
            comparisons += len(result_docs)
            term_postings = postings_lists[term]
            tfidf_term = tfidf_values[term]
        
        # Perform the merge based on TF-IDF scores
            merged_result = []
            i = 0
            j = 0
            while i < len(result_docs) and j < len(term_postings):
                doc_id_1 = result_docs[i]
                doc_id_2 = term_postings[j]
            
            # Calculate the TF-IDF scores for both documents
                tfidf_1 = self.inverted_index[term].idf
                tfidf_2 = tfidf_term
            
                if doc_id_1 == doc_id_2:
                    merged_result.append(doc_id_1)
                    i += 1
                    j += 1
                elif doc_id_1 < doc_id_2:
                    i += 1
                else:
                    j += 1
        
            result_docs = merged_result
    
    # Sort the merged results by TF-IDF scores
        result_docs_sorted = sorted(result_docs, key=lambda doc_id: sum(tfidf_values[term] for term in query_terms), reverse=True)
    
        return result_docs_sorted, comparisons


    def daat_and_query_with_skip_pointers(self, query_terms):
        comparisons = 0
        result_docs = []

        postings_lists = self.get_postings_lists_with_skip_pointers(query_terms)
        doc_pointers = {term: 0 for term in query_terms}

        while all(doc_pointers[term] < len(postings_lists[term]) for term in query_terms):
            doc_ids = [postings_lists[term][doc_pointers[term]] for term in query_terms]
            min_doc_id = min(doc_ids)
            comparisons += len(query_terms) - 1

            if all(doc_id == min_doc_id for doc_id in doc_ids):
                result_docs.append(min_doc_id)
                for term in query_terms:
                    doc_pointers[term] += 1
            else:
                min_term = query_terms[doc_ids.index(min_doc_id)]
                doc_pointers[min_term] += 1

        return result_docs, comparisons

def build_inverted_index(corpus_file):
    preprocessor = Preprocessor()
    inverted_index = {}

    with open(corpus_file, 'r') as file:
        for line in file:
            doc_id, text = preprocessor.get_doc_id(line)
            tokens = preprocessor.tokenizer(text)
            for token in tokens:
                    # Check if the term is already in the inverted index
                if token not in inverted_index:
                        # If not, create a new LinkedList instance
                    inverted_index[token] = LinkedList()
                    # Add the document ID to the LinkedList for the term
                inverted_index[token].insert(doc_id)

    return inverted_index

if __name__ == '__main__':
    corpus_file = 'input_corpus.txt'
    inverted_index = build_inverted_index(corpus_file)

    total_num_docs = len(inverted_index)
    for term in inverted_index:
        postings_list = inverted_index[term]
        postings_list.idf = total_num_docs / postings_list.get_length()
        postings_list.add_skip_connections()

    indexer = Indexer()
    indexer.inverted_index = inverted_index

    
    query = "the novel coronavirus"
    query_terms = indexer.preprocess_query(query)
    
    
    result_docs, comparisons = indexer.daat_and_query(query_terms)
    
    print("Query:", query)
    print("Matching Documents (DAAT):", result_docs)
    print("Number of Comparisons (DAAT):", comparisons)

    
    result_docs_sorted_by_tfidf = indexer.daat_and_query_sorted_by_tfidf(query_terms)
    
    print("Matching Documents (DAAT sorted by TF-IDF):", result_docs_sorted_by_tfidf)

    
    result_docs_skip, comparisons_skip = indexer.daat_and_query_with_skip_pointers(query_terms)

    print("Matching Documents (DAAT with Skip Pointers):", result_docs_skip)
    print("Number of Comparisons (DAAT with Skip Pointers):", comparisons_skip)

   
    result_docs_skip_sorted_by_tfidf = indexer.daat_and_query_sorted_by_tfidf(query_terms)
    
    print("Matching Documents (DAAT with Skip Pointers sorted by TF-IDF):", result_docs_skip_sorted_by_tfidf)

   