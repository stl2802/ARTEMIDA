import json
import math
import time
import numpy as np
import re
import hnswlib
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Пути к файлам
VOCABULARY_FILE = "json/vocabulary.json"
INVERTED_INDEX_FILE = "json/inverted_index.json"
SVD_VT_FILE = "npy/svd_vt.npy"
SVD_VECTORS_FILE = "npy/doc_vectors_svd.npy"
HNSW_INDEX_FILE = "hnsw_index.bin"
DOC_IDS_FILE = "json/doc_ids.json"
QUERIES_FILE = "json/queries.json"

class BenchmarkRunner:
    def __init__(self):
        print("Загрузка данных для бенчмарка...")
        
        raw_vectors = np.load(SVD_VECTORS_FILE)

        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        # Защита от деления на ноль
        norms[norms == 0] = 1e-10
        self.doc_vectors = raw_vectors / norms
        
        with open(DOC_IDS_FILE, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
            
        _, dim = self.doc_vectors.shape
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.load_index(HNSW_INDEX_FILE, max_elements=len(self.doc_ids))
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        with open(VOCABULARY_FILE, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)["vocabulary"]
        self.vocab_size = len(self.vocabulary)
        
        with open(INVERTED_INDEX_FILE, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        self.total_docs = index_data["doc_count"]
        self.doc_freq = {term: len(postings) for term, postings in index_data["inverted_index"].items()}
        
        self.V_k = np.load(SVD_VT_FILE)
        
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        self.test_queries = (
            [q["en"] for q in queries_data["queries"]["exact_terms"][:3]] +
            [q["en"] for q in queries_data["queries"]["detailed_queries"][:3]] +
            [q["en"] for q in queries_data["queries"]["short_queries"][:3]] +
            [q["en"] for q in queries_data["queries"]["error_queries"][:3]]
        )
        print(f"Загружено {len(self.test_queries)} тестовых запросов (EN).")

    def _preprocess_text(self, text: str) -> list:
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def _calculate_idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        if df == 0: return 0.0
        return math.log((self.total_docs + 1) / (df + 1)) + 1

    def query_to_svd_vector(self, query: str) -> np.ndarray:
        tokens = self._preprocess_text(query)
        if not tokens: return np.zeros(self.doc_vectors.shape[1])
        
        token_counts = Counter(tokens)
        query_length = len(tokens)
        tfidf_vector = np.zeros(self.vocab_size)
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                tf = count / query_length
                idf = self._calculate_idf(token)
                tfidf_vector[self.vocabulary[token]] = tf * idf
        
        svd_vector = tfidf_vector @ self.V_k.T
        norm = np.linalg.norm(svd_vector)
        return svd_vector / norm if norm > 0 else svd_vector

    def exact_search(self, query_vector: np.ndarray, top_k: int = 5) -> list:
        """Точный поиск. Т.к. векторы нормализованы, dot product = cosine similarity"""
        similarities = self.doc_vectors @ query_vector
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.doc_ids[idx] for idx in top_indices]

    def approx_search(self, index, query_vector: np.ndarray, top_k: int = 5) -> list:
        """Приближенный поиск через HNSW"""
        labels, _ = index.knn_query(query_vector.reshape(1, -1), k=top_k)
        return [self.doc_ids[label] for label in labels[0]]

    def run_benchmark(self):
        ef_values = [10, 25, 50, 100]
        num_runs = 20
        
        print("\n" + "="*85)
        print("БЕНЧМАРК: Сравнение точного и приближенного поиска (SVD + HNSW)")
        print("="*85)
        print(f"{'efSearch':<10} | {'Recall@5':<10} | {'Время векториз. (мс)':<22} | {'Время HNSW (мс)':<15}")
        print("-" * 85)
        
        for ef in ef_values:
            self.index.set_ef(ef)
            
            total_overlap = 0.0
            total_vec_time = 0.0
            total_search_time = 0.0
            
            for query in self.test_queries:
                start_vec = time.perf_counter()
                q_vec = self.query_to_svd_vector(query)
                vec_time = time.perf_counter() - start_vec
                total_vec_time += vec_time
                
                exact_docs = self.exact_search(q_vec, top_k=5)
                
                start_search = time.perf_counter()
                for _ in range(num_runs):
                    approx_docs = self.approx_search(self.index, q_vec, top_k=5)
                search_time = (time.perf_counter() - start_search) / num_runs
                total_search_time += search_time
                
                overlap = len(set(exact_docs) & set(approx_docs)) / 5.0
                total_overlap += overlap
            
            avg_overlap = total_overlap / len(self.test_queries)
            avg_vec_time = (total_vec_time / len(self.test_queries)) * 1000
            avg_search_time = (total_search_time / len(self.test_queries)) * 1000
            
            print(f"efSearch = {ef:<3} | "
                  f"{avg_overlap*100:>6.2f}%   | "
                  f"{avg_vec_time:>20.3f} | "
                  f"{avg_search_time:>13.4f}")


def main():
    runner = BenchmarkRunner()
    runner.run_benchmark()

if __name__ == "__main__":
    main()