import json
import math
import numpy as np
from collections import Counter
import re

from utils.text_processor import preprocess_text
##

VOCABULARY_FILE = "generated_data/json/vocabulary.json"
INVERTED_INDEX_FILE = "generated_data/json/inverted_index.json"

class QueryVectorizer:
    def __init__(self, inverted_index_path=INVERTED_INDEX_FILE):
        with open(VOCABULARY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocabulary = data["vocabulary"]
        self.vocab_size = len(self.vocabulary)
        
        with open(inverted_index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        self.total_docs = index_data["doc_count"]

        self.doc_freq = {
            term: len(postings) 
            for term, postings in index_data["inverted_index"].items()
        }
        
    def preprocess_query(self, query: str) -> list:
        """Предобработка запроса с использованием общего процессора"""
        return preprocess_text(query, language='auto')
    
    def calculate_idf(self, token: str) -> float:
        """Рассчитывает IDF для токена"""
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        # формула IDF с сглаживанием
        return math.log((self.total_docs + 1) / (df + 1)) + 1
    
    def query_to_tfidf_vector(self, query: str) -> np.ndarray:
        """
        Преобразует запрос в TF-IDF вектор.
        
        Args:
            query: строка запроса
            
        Returns:
            numpy array размером vocab_size с TF-IDF весами
        """
        tokens = self.preprocess_query(query)
        
        if not tokens:
            return np.zeros(self.vocab_size)
        
        token_counts = Counter(tokens)
        query_length = len(tokens)
        
        vector = np.zeros(self.vocab_size)
        
        for token, count in token_counts.items():
            if token not in self.vocabulary:
                continue 
            
            tf = count / query_length
            
            idf = self.calculate_idf(token)
            
            tfidf = tf * idf
            
            token_idx = self.vocabulary[token]
            vector[token_idx] = tfidf
        
        return vector
    
    def query_to_tfidf_dict(self, query: str) -> dict:
        """
        Альтернативный формат: возвращает словарь {токен: tfidf_weight}
        Только для ненулевых значений (разреженное представление)
        """
        tokens = self.preprocess_query(query)
        
        if not tokens:
            return {}
        
        token_counts = Counter(tokens)
        query_length = len(tokens)
        
        result = {}
        for token, count in token_counts.items():
            if token not in self.vocabulary:
                continue
            
            tf = count / query_length
            idf = self.calculate_idf(token)
            tfidf = tf * idf
            
            if tfidf > 0:
                result[token] = tfidf
        
        return result



def main():
    vectorizer = QueryVectorizer()
    
    queries = [
        "neural networks",
        "deep learning optimization",
        "transformer architecture"
    ]
    
    for query in queries:
        print(f"\nЗапрос: '{query}'")
        
        vector = vectorizer.query_to_tfidf_vector(query)
        print(f"Размер вектора: {len(vector)}")
        print(f"Ненулевых элементов: {np.count_nonzero(vector)}")
        print(f"Максимальный TF-IDF: {vector.max():.4f}")
        
        sparse_vector = vectorizer.query_to_tfidf_dict(query)
        print(f"Ненулевые веса:")
        for token, weight in sorted(sparse_vector.items(), key=lambda x: x[1], reverse=True):
            print(f"  {token}: {weight:.4f}")


if __name__ == "__main__":
    main()