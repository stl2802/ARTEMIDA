import json
import math
import numpy as np
import re
import sys
import hnswlib
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

import os
import time

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Пути к файлам
VOCABULARY_FILE = "json/vocabulary.json"
INVERTED_INDEX_FILE = "json/inverted_index.json"
SVD_VT_FILE = "npy/svd_vt.npy"
SVD_VECTORS_FILE = "npy/doc_vectors_svd.npy"
HNSW_INDEX_FILE = "bin/hnsw_index.bin"
DOC_IDS_FILE = "json/doc_ids.json"
QUERIES_FILE = "json/queries.json"

# Параметры HNSW
M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 50


class HnswSearcher:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        with open(VOCABULARY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocabulary = data["vocabulary"]
        self.vocab_size = len(self.vocabulary)
        
        with open(INVERTED_INDEX_FILE, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        self.total_docs = index_data["doc_count"]
        self.doc_freq = {
            term: len(postings)
            for term, postings in index_data["inverted_index"].items()
        }
        self.forward_index = index_data["forward_index"]

        self.V_k = np.load(SVD_VT_FILE)

        self.doc_vectors = np.load(SVD_VECTORS_FILE)
        _, dim = self.doc_vectors.shape
        
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.load_index(HNSW_INDEX_FILE, max_elements=self.doc_vectors.shape[0])
        self.index.set_ef(EF_SEARCH)
        
        with open(DOC_IDS_FILE, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
            
        print(f"HNSW Searcher инициализирован. Размерность: {dim}")

    def _preprocess_text(self, text: str) -> list:
        """Предобработка текста (идентична заданию 4)"""
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def _calculate_idf(self, token: str) -> float:
        """IDF с сглаживанием"""
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        return math.log((self.total_docs + 1) / (df + 1)) + 1

    def _query_to_svd_vector(self, query: str) -> np.ndarray:
        """
        Преобразует запрос в вектор с помощью модели из задания 4 
        (TF-IDF + проекция в SVD-пространство).
        """
        tokens = self._preprocess_text(query)
        if not tokens:
            return np.zeros(self.V_k.shape[0])
        
        token_counts = Counter(tokens)
        query_length = len(tokens)
        
        tfidf_vector = np.zeros(self.vocab_size)
        for token, count in token_counts.items():
            if token not in self.vocabulary:
                continue
            tf = count / query_length
            idf = self._calculate_idf(token)
            tfidf_vector[self.vocabulary[token]] = tf * idf
        
        svd_vector = tfidf_vector @ self.V_k.T
        
        norm = np.linalg.norm(svd_vector)
        if norm > 0:
            svd_vector = svd_vector / norm
            
        return svd_vector

    def approximate_search(self, query: str, top_k: int = 5) -> list:
        """
        Пункт 4: Приближенный поиск документов через HNSW.
        """
        query_vector = self._query_to_svd_vector(query)
        
        if np.linalg.norm(query_vector) == 0:
            return []
        
        labels, distances = self.index.knn_query(
            query_vector.reshape(1, -1), 
            k=top_k
        )
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            real_doc_id = self.doc_ids[label]
            cosine_similarity = 1.0 - distance
            results.append((real_doc_id, float(cosine_similarity)))
            
        return results


def build_index():
    """
    Режим построения индекса: создает HNSW индекс на основе SVD-векторов.
    С замера времени, размера и параметров для отчета.
    """
    print("=" * 60)
    print("РЕЖИМ ПОСТРОЕНИЯ ИНДЕКСА")
    print("=" * 60)
    
    doc_vectors = np.load(SVD_VECTORS_FILE)
    
    num_vectors, dim = doc_vectors.shape
    print(f"Загружено {num_vectors} векторов размерности {dim}")
    
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    doc_vectors_normalized = doc_vectors / norms
    print(f"Векторы нормализованы для косинусного сходства")
    
    index = hnswlib.Index(space='cosine', dim=dim)
    
    index.init_index(
        max_elements=num_vectors,
        ef_construction=EF_CONSTRUCTION,
        M=M
    )
    
    print(f"\nПостроение индекса...")
    start_time = time.perf_counter()
    
    index.add_items(doc_vectors_normalized, ids=list(range(num_vectors)))
    
    end_time = time.perf_counter()
    build_time = end_time - start_time
    
    index.set_ef(EF_SEARCH)
    
    index.save_index(HNSW_INDEX_FILE)
    
    #ПОЛУЧЕНИЕ ПАРАМЕТРОВ ИНДЕКСА
    max_elements = num_vectors
    current_count = num_vectors
    ef_construction = EF_CONSTRUCTION
    M_param = M
    
    file_size_bytes = os.path.getsize(HNSW_INDEX_FILE)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"\nHNSW индекс построен и сохранен в '{HNSW_INDEX_FILE}'")
    print(f"\n--- Параметры индекса ---")
    print(f"  M (количество связей на узел): {M_param}")
    print(f"  efConstruction (ширина при построении): {ef_construction}")
    print(f"  efSearch (ширина при поиске): {EF_SEARCH}")
    print(f"  Метрика: косинусное сходство (cosine)")
    print(f"  Размерность векторов: {dim}")
    print(f"  Максимальное кол-во элементов: {max_elements}")
    print(f"  Текущее кол-во элементов: {current_count}")
    
    print(f"\n--- Статистика ---")
    print(f"  Время построения: {build_time:.4f} сек ({build_time*1000:.2f} мс)")
    print(f"  Размер файла индекса: {file_size_bytes:,} байт ({file_size_mb:.2f} МБ)")
    print(f"  Размер исходных SVD-векторов: {doc_vectors.nbytes:,} байт ({doc_vectors.nbytes/(1024*1024):.2f} МБ)")
    print(f"  Коэффициент сжатия: {doc_vectors.nbytes / file_size_bytes:.2f}x")

def search_mode():
    """
    Режим поиска: пользователь вводит запрос и получает топ-5 документов.
    """
    print("=" * 60)
    print("РЕЖИМ ПОИСКА")
    print("=" * 60)
    
    searcher = HnswSearcher()
    
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        test_queries = (
            [q["en"] for q in queries_data["queries"]["exact_terms"][:2]] +
            [q["en"] for q in queries_data["queries"]["detailed_queries"][:2]] +
            [q["en"] for q in queries_data["queries"]["short_queries"][:2]] +
            [q["en"] for q in queries_data["queries"]["error_queries"][:2]]
        )
        print(f"\nЗагружено {len(test_queries)} тестовых запросов (EN)\n")
    except Exception as e:
        print(f"Ошибка загрузки queries.json: {e}")
        test_queries = []

    while True:
        if test_queries:
            query = test_queries.pop(0)
            print(f"Запрос: '{query}'")
        else:
            query = input("Введите поисковый запрос (или 'exit' для выхода): ")
            if query.lower() == 'exit':
                break
        
        results = searcher.approximate_search(query, top_k=5)
        
        if not results:
            print("Документов не найдено\n")
        else:
            print(f"Топ-5 результатов (приближенный поиск HNSW):")
            for rank, (doc_id, similarity) in enumerate(results, 1):
                title = searcher.forward_index[doc_id]["title"]
                summary = searcher.forward_index[doc_id]["summary"]
                
                print(f"{rank}. [{doc_id}] (similarity: {similarity:.4f})")
                print(f"   Title: {title}")
                print(f"   Summary: {summary[:150]}...")
                print()


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python hnsw_search.py build  - построение HNSW индекса")
        print("  python hnsw_search.py search - режим поиска")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "build":
        build_index()
    elif mode == "search":
        search_mode()
    else:
        print(f"Неизвестный режим: {mode}")
        print("Используйте 'build' или 'search'")


if __name__ == "__main__":
    main()