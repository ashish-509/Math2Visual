# Retrieval using TF-IDF
# No heavy dependencies needed, works well for technical docs

import re
import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

class SimpleRetriever:
    def __init__(self):
        self.chunks = []
        self.idf_scores = {}
        self.doc_vectors = []
        self.query_cache = {}  # cache for query results to speed up repeated queries
        
    def index_chunks(self, chunks):
        """
        Build TF-IDF index from chunks.
        chunks: list of dicts from DocumentChunker
        """
        self.chunks = chunks
        
        # calculate IDF for each term
        doc_count = len(chunks)
        term_doc_freq = defaultdict(int)
        
        for chunk in chunks:
            terms = self._tokenize(chunk['text'])
            unique_terms = set(terms)
            for term in unique_terms:
                term_doc_freq[term] += 1
        
        # compute IDF: log(total_docs / docs_with_term)
        for term, freq in term_doc_freq.items():
            self.idf_scores[term] = math.log(doc_count / freq) if freq > 0 else 0
        
        # build TF-IDF vector for each chunk
        self.doc_vectors = []
        for chunk in chunks:
            vec = self._compute_tfidf(chunk['text'])
            self.doc_vectors.append(vec)
    
    def retrieve(self, query, top_k=3):
        """
        Find top_k most relevant chunks for the query
        Uses caching for speed on repeated queries
        Returns list of (chunk, score) tuples
        """
        if not self.chunks:
            return []
        
        # check cache first
        cache_key = (query, top_k)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        query_vec = self._compute_tfidf(query)
        scores = []
        
        for idx, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_sim(query_vec, doc_vec)
            scores.append((idx, score))
        
        # sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            results.append({
                'chunk': self.chunks[idx],
                'score': score,
                'rank': len(results) + 1
            })
        
        # cache the result
        self.query_cache[cache_key] = results
        
        return results
        scores = []
        
        for idx, doc_vec in enumerate(self.doc_vectors):
            score = self._cosine_sim(query_vec, doc_vec)
            scores.append((idx, score))
        
        # sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            results.append({
                'chunk': self.chunks[idx],
                'score': score,
                'rank': len(results) + 1
            })
        
        return results
    
    def _tokenize(self, text):
        # Simple tokenization - lowercase and split on non-alphanumeric
        text = text.lower()
        # keep important programming terms intact
        text = re.sub(r'[^\w\s\+\-]', ' ', text)
        tokens = text.split()
        
        # filter out very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        return tokens
    
    def _compute_tfidf(self, text):
        # Compute TF-IDF vector for text
        terms = self._tokenize(text)
        term_freq = Counter(terms)
        doc_len = len(terms)
        
        tfidf_vec = {}
        for term, freq in term_freq.items():
            tf = freq / doc_len if doc_len > 0 else 0
            idf = self.idf_scores.get(term, 0)
            tfidf_vec[term] = tf * idf
        
        return tfidf_vec
    
    def _cosine_sim(self, vec1, vec2):
        # Compute cosine similarity between two TF-IDF vectors
        # get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        # dot product
        dot_prod = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # magnitudes
        mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_prod / (mag1 * mag2)
    
    def get_stats(self):
        # Return index statistics
        return {
            'num_chunks': len(self.chunks),
            'vocab_size': len(self.idf_scores),
            'avg_chunk_size': sum(c['size'] for c in self.chunks) / len(self.chunks) if self.chunks else 0
        }
