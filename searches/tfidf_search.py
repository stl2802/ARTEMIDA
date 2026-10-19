import json
import numpy as np
import re
import os
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

CORPUS_FILE = "generated_data/json/preprocessed_corpus.json"
VOCABULARY_FILE = "generated_data/json/vocabulary.json"
INVERTED_INDEX_FILE = "generated_data/json/inverted_index.json"

DOC_VECTORS_FILE = "generated_data/npy/doc_vectors.npy"
DOC_IDS_FILE = "generated_data/json/doc_ids.json"

DOC_VECTORS_SVD_FILE = "generated_data/npy/doc_vectors_svd.npy"
SVD_VT_FILE = "generated_data/npy/svd_vt.npy"


class TfidfSearcher:
    def __init__(self):
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
        
        # Загружаем прямой индекс для title и summary
        self.forward_index = {}
        for doc_id, entry_data in index_data["forward_index"].items():
            self.forward_index[doc_id] = {
                "title": entry_data["title"],
                "summary": entry_data["summary"]
            }
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        self.doc_vectors = None  # shape: (num_docs, vocab_size)
        self.doc_ids = None

        # Атрибуты для SVD
        self.doc_vectors_svd = None  # shape: (num_docs, k) — сжатые векторы
        self.V_k = None              # shape: (k, vocab_size) — для проекции запроса
        self.S_k = None
    
    def _preprocess_text(self, text: str) -> list:
        """Предобработка текста (идентична предобработке запроса)"""
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens
    
    def _calculate_idf(self, token: str) -> float:
        """IDF с сглаживанием"""
        import math
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        return math.log((self.total_docs + 1) / (df + 1)) + 1
    
    def _text_to_tfidf_vector(self, text: str) -> np.ndarray:
        """Превращает текст документа в TF-IDF вектор"""
        tokens = self._preprocess_text(text)
        if not tokens:
            return np.zeros(self.vocab_size)
        
        token_counts = Counter(tokens)
        text_length = len(tokens)
        
        vector = np.zeros(self.vocab_size)
        for token, count in token_counts.items():
            if token not in self.vocabulary:
                continue
            tf = count / text_length
            idf = self._calculate_idf(token)
            tfidf = tf * idf
            token_idx = self.vocabulary[token]
            vector[token_idx] = tfidf
        
        return vector
    
    def build(self, corpus_path=CORPUS_FILE):
        """Режим обучения: строит TF-IDF векторы для всех документов"""
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
        self.doc_ids = [doc["id"] for doc in corpus]
        vectors = []
        
        for i, doc in enumerate(corpus):
            vector = self._text_to_tfidf_vector(doc["text"])
            vectors.append(vector)
            if (i + 1) % 100 == 0:
                print(f"   Векторизовано {i + 1}/{len(corpus)} документов")
        
        self.doc_vectors = np.array(vectors)
        
        np.save(DOC_VECTORS_FILE, self.doc_vectors)
        with open(DOC_IDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.doc_ids, f)
        
        print(f"Векторы документов построены:")
        print(f"Размер матрицы: {self.doc_vectors.shape}")
        print(f"Ненулевых элементов: {np.count_nonzero(self.doc_vectors):,}")
        print(f"Сохранено в '{DOC_VECTORS_FILE}' и '{DOC_IDS_FILE}'")
    
    def load(self):
        """Загружает готовые векторы документов"""
        self.doc_vectors = np.load(DOC_VECTORS_FILE)
        with open(DOC_IDS_FILE, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
        print(f"Загружены векторы: {self.doc_vectors.shape}, документов: {len(self.doc_ids)}")
        
        # Пытаемся загрузить SVD-данные, если они есть
        if os.path.exists(DOC_VECTORS_SVD_FILE) and os.path.exists(SVD_VT_FILE):
            self.doc_vectors_svd = np.load(DOC_VECTORS_SVD_FILE)
            self.V_k = np.load(SVD_VT_FILE)
            print(f"Загружены SVD-векторы: {self.doc_vectors_svd.shape}")
        else:
            print("SVD-данные не найдены. Сначала выполните: python tfidf_search.py svd")
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное сходство между двумя векторами"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
    
    def _query_to_vector(self, query: str) -> np.ndarray:
        """Вектор запроса (та же логика, что и в QueryVectorizer)"""
        tokens = self._preprocess_text(query)
        if not tokens:
            return np.zeros(self.vocab_size)
        
        token_counts = Counter(tokens)
        query_length = len(tokens)
        
        vector = np.zeros(self.vocab_size)
        for token, count in token_counts.items():
            if token not in self.vocabulary:
                continue
            tf = count / query_length
            idf = self._calculate_idf(token)
            tfidf = tf * idf
            token_idx = self.vocabulary[token]
            vector[token_idx] = tfidf
        return vector
    
    def apply_svd(self, k: int = 100):
        """Применяет усеченное SVD для уменьшения размерности"""
        #"ох, напрашиваются ницензурные слова, но я не могу их удалить, потому что они нужны для работы программы" - это мне сказал квен когда помогал мне написать этот метод
        if self.doc_vectors is None:
            raise RuntimeError("Сначала загрузите или постройте векторы документов")
        
        print(f"Применяем SVD с k={k}...")
        
        # SVD разложение
        U, S, Vt = np.linalg.svd(self.doc_vectors, full_matrices=False)
        
        # Усечение до k измерений
        U_k = U[:, :k]
        self.S_k = S[:k]
        self.V_k = Vt[:k, :]  # shape: (k, vocab_size) — сохраняем для проекции запроса
        
        # Новые векторы документов
        self.doc_vectors_svd = U_k * self.S_k  # Broadcasting
        

        np.save(DOC_VECTORS_SVD_FILE, self.doc_vectors_svd)
        np.save(SVD_VT_FILE, self.V_k)

        total_variance = np.sum(S**2)
        kept_variance = np.sum(self.S_k**2)
        explained_ratio = kept_variance / total_variance * 100
        
        print(f"SVD применено:")
        print(f"  Исходная размерность: {self.doc_vectors.shape}")
        print(f"  Новая размерность: {self.doc_vectors_svd.shape}")
        print(f"  Сохранено информации: {explained_ratio:.2f}%")
        print(f"  Сохранено в '{DOC_VECTORS_SVD_FILE}' и '{SVD_VT_FILE}'")
    
    def search(self, query: str, top_k: int = 5, use_svd: bool = None) -> list:
        """
        Поиск документов по косинусному сходству.
        
        Returns:
            Список кортежей (doc_id, similarity), отсортированных по убыванию
        """
        if self.doc_vectors is None:
            raise RuntimeError("Сначала загрузите или постройте векторы документов")
        
        if use_svd is None:
            use_svd = (self.doc_vectors_svd is not None and self.V_k is not None)
        
        query_vector = self._query_to_vector(query)
        
        if use_svd:
            if self.doc_vectors_svd is None or self.V_k is None:
                raise RuntimeError("SVD-данные не загружены. Сначала выполните: python tfidf_search.py svd")
            query_vector_reduced = query_vector @ self.V_k.T
            doc_vectors = self.doc_vectors_svd
            qv = query_vector_reduced
        else:
            doc_vectors = self.doc_vectors
            qv = query_vector
        
        if np.linalg.norm(qv) == 0:
            return []
        
        # Векторизованное вычисление сходства со всеми документами сразу
        # Это намного быстрее, чем цикл по 500 документам
        #короче тут нанейронил немного, потомучто возникали большие проблемы с производительностьюю \
            # ну и что я полтора месяца учу нейронки, а тут сделайте поиск гибридный, я что доктор манхетен?
        #обьсняю как работает 
        # self.doc_vectors  # shape: (500, 3421) — матрица, где каждая строка = вектор документа
        # query_vector      # shape: (3421,) — вектор запроса
        # `@` — это матричное умножение (einsum для векторов)
        # `doc_vectors @ query_vector` даст массив из 500 чисел — косинусных сходств
        dot_products = doc_vectors @ qv
        doc_norms = np.linalg.norm(doc_vectors, axis=1)#тут тоже самое как с квери но для документов
        query_norm = np.linalg.norm(qv)#считает длину (норму) каждого вектора документа.
        
        # Защита от деления на ноль
        denominators = doc_norms * query_norm
        similarities = np.zeros_like(dot_products)
        nonzero_mask = denominators > 0
        similarities[nonzero_mask] = dot_products[nonzero_mask] / denominators[nonzero_mask]
        
        # Сортировка по убыванию
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.doc_ids[idx], float(similarities[idx])))
        
        return results


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python tfidf_search.py build   - построение TF-IDF векторов")
        print("  python tfidf_search.py svd [k] - применение SVD (по умолчанию k=100)")
        print("  python tfidf_search.py search  - режим поиска")
        return
    
    mode = sys.argv[1].lower()
    searcher = TfidfSearcher()
    
    if mode == "build":
        searcher.build()
    
    elif mode == "svd":
        searcher.load()
        k = 100
        if len(sys.argv) >= 3:
            try:
                k = int(sys.argv[2])
            except ValueError:
                pass
        searcher.apply_svd(k=k)
    
    elif mode == "search":
        searcher.load()
        
        try:
            with open("json/queries.json", 'r', encoding='utf-8') as f:
                queries_data = json.load(f)
            queries = (
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["exact_terms"][:3]] +
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["detailed_queries"][:2]] +
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["short_queries"][:3]] +
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["error_queries"][:2]]
            )
            print(f"Загружено {len(queries)} тестовых запросов\n")
        except Exception:
            queries = []
        
        while True:
            if queries:
                query = queries.pop(0)
                print(f"Запрос: '{query}'")
            else:
                query = input("Введите поисковый запрос (или 'exit' для выхода): ")
                if query.lower() == 'exit':
                    break
            
            results = searcher.search(query, top_k=5)
            
            if not results:
                print("Документов не найдено\n")
            else:
                print(f"Топ-5 результатов (косинусное сходство):")
                for rank, (doc_id, similarity) in enumerate(results, 1):
                    title = searcher.forward_index[doc_id]["title"]
                    summary = searcher.forward_index[doc_id]["summary"]
                    print(f"{rank}. [{doc_id}] - {title} - {summary}")
                print()

def visualize_matrix():
    """Визуализирует матрицу TF-IDF для отчета"""
    import matplotlib.pyplot as plt
    
    # Загружаем матрицу
    doc_vectors = np.load(DOC_VECTORS_FILE)
    
    print(f"Размер матрицы: {doc_vectors.shape}")
    print(f"Тип данных: {doc_vectors.dtype}")
    print(f"Минимум: {doc_vectors.min():.4f}")
    print(f"Максимум: {doc_vectors.max():.4f}")
    print(f"Среднее: {doc_vectors.mean():.4f}")
    print(f"Ненулевых элементов: {np.count_nonzero(doc_vectors)}")
    print(f"\nПервые 10x10 элементов матрицы:")
    print(doc_vectors[:10, :10])

if __name__ == "__main__":
    visualize_matrix()