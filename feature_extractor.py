import json
import math
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

from lecsika_search import load_index, calculate_bm25, tokenize

INVERTED_INDEX_FILE = "json/inverted_index.json"
VOCABULARY_FILE = "json/vocabulary.json"
NEURAL_VECTORS_FILE = "npy/neural_doc_vectors.npy"
NEURAL_DOC_IDS_FILE = "json/neural_doc_ids.json"


class FeatureExtractor:
    def __init__(self):
        print("Инициализация извлекателя признаков...")
        
        self.forward_index, self.inverted_index, self.avg_doc_length, self.doc_count = load_index(INVERTED_INDEX_FILE)
        
        with open(VOCABULARY_FILE, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)["vocabulary"]
        self.vocab_size = len(self.vocabulary)
        
        self.doc_freq = {term: entry.doc_freq() for term, entry in self.inverted_index.items()}
        
        self.neural_vectors = np.load(NEURAL_VECTORS_FILE)
        with open(NEURAL_DOC_IDS_FILE, 'r', encoding='utf-8') as f:
            self.neural_doc_ids = json.load(f)
        self.neural_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.neural_doc_ids)}
        
        print("Загрузка модели Sentence-Transformer...")
        self.neural_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        print("--Извлекатель признаков готов.\n")

    def _calculate_tfidf_vector(self, tokens, doc_length=None):
        """Вспомогательная функция для расчета TF-IDF вектора"""
        tfidf_vector = np.zeros(self.vocab_size)
        if not tokens:
            return tfidf_vector
            
        token_counts = Counter(tokens)
        length = doc_length if doc_length else len(tokens)
        
        for token, count in token_counts.items():
            if token not in self.vocabulary:
                continue
                
            tf = count / length
            
            df = self.doc_freq.get(token, 0)
            if df == 0:
                idf = 0.0
            else:
                idf = math.log((self.doc_count + 1) / (df + 1)) + 1
                
            tfidf_vector[self.vocabulary[token]] = tf * idf
            
        return tfidf_vector

    def get_features(self, query_text: str, doc_id: str) -> list:
        """
        Вычисление 6 признаков для пары (запрос, документ).
        
        Возвращает список из 6 float значений.
        """
        # tokenize из lecsika_search.py
        query_tokens = tokenize(query_text)
        
        if not query_tokens or doc_id not in self.forward_index:
            return [0.0] * 6
            
        entry = self.forward_index[doc_id]
        
        # --- Признак 1: Оценка BM25 (из lecsika_search.py) ---
        bm25_score = calculate_bm25(
            query_tokens, doc_id, 
            self.forward_index, self.inverted_index, 
            self.avg_doc_length
        )
        
        # --- Признак 2: Косинусное сходство TF-IDF-векторов (задание 3) ---
        q_tfidf = self._calculate_tfidf_vector(query_tokens)
        
        # word_freq - это Counter, поэтому разворачиваем его в список токенов
        doc_tokens_list = []
        for word, freq in entry.word_freq.items():
            doc_tokens_list.extend([word] * freq)
        d_tfidf = self._calculate_tfidf_vector(doc_tokens_list, doc_length=entry.doc_length)
        
        q_norm = np.linalg.norm(q_tfidf)
        d_norm = np.linalg.norm(d_tfidf)
        if q_norm > 0 and d_norm > 0:
            tfidf_cosine = float(np.dot(q_tfidf, d_tfidf) / (q_norm * d_norm))
        else:
            tfidf_cosine = 0.0
            
        # --- Признак 3: Косинусное сходство нейросетевых векторов (из neural_search.py) ---
        # Вектор документа уже нормализован
        doc_neural_idx = self.neural_id_to_idx.get(doc_id)
        if doc_neural_idx is not None:
            doc_neural_vec = self.neural_vectors[doc_neural_idx]
        else:
            doc_neural_vec = np.zeros(self.neural_vectors.shape[1])
            
        query_neural_vec = self.neural_model.encode(
            query_text, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        neural_cosine = float(np.dot(query_neural_vec, doc_neural_vec))
        
        # --- Признак 4: Доля уникальных слов запроса в тексте документа ---
        unique_query_words = set(query_tokens)
        doc_text_words = set(entry.word_freq.keys())
        if len(unique_query_words) > 0:
            overlap_text = len(unique_query_words & doc_text_words) / len(unique_query_words)
        else:
            overlap_text = 0.0
            
        # --- Признак 5: Доля уникальных слов запроса в заголовке документа ---
        title_tokens = tokenize(entry.title)
        unique_title_words = set(title_tokens)
        if len(unique_query_words) > 0:
            overlap_title = len(unique_query_words & unique_title_words) / len(unique_query_words)
        else:
            overlap_title = 0.0
            
        # --- Признак 6: Длина документа в токенах (из forward_index) ---
        doc_length = float(entry.doc_length)
        
        # Целевая переменная relevance НЕ включается
        return [
            bm25_score,          # 1. BM25 Score
            tfidf_cosine,        # 2. TF-IDF Cosine Similarity
            neural_cosine,       # 3. Neural Cosine Similarity
            overlap_text,        # 4. Word Overlap (Text)
            overlap_title,       # 5. Word Overlap (Title)
            doc_length           # 6. Document Length
        ]


#ТЕСТОВЫЙ ЗАПУСК
if __name__ == "__main__":
    extractor = FeatureExtractor()
    
    test_query = "neural networks"
    test_doc_id = "doc_0003"
    
    print(f"\nТестирование извлечения признаков:")
    print(f"Запрос: '{test_query}'")
    print(f"Документ: {test_doc_id} ({extractor.forward_index[test_doc_id].title})\n")
    
    features = extractor.get_features(test_query, test_doc_id)
    
    feature_names = [
        "1. BM25 Score",
        "2. TF-IDF Cosine Similarity",
        "3. Neural Cosine Similarity",
        "4. Word Overlap (Text)",
        "5. Word Overlap (Title)",
        "6. Document Length"
    ]
    
    for name, value in zip(feature_names, features):
        print(f"{name:<35}: {value:.4f}")