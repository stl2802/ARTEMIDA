import json
import numpy as np
import sys
from sentence_transformers import SentenceTransformer

CORPUS_FILE = "generated_data/json/preprocessed_corpus.json"
INVERTED_INDEX_FILE = "generated_data/json/inverted_index.json"
NEURAL_DOC_IDS_FILE = "generated_data/json/neural_doc_ids.json"
NEURAL_VECTORS_FILE = "generated_data/npy/neural_doc_vectors.npy"

class NeuralSearcher:
    def __init__(self):
        print("Загрузка модели paraphrase-multilingual-MiniLM-L12-v2...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        with open(INVERTED_INDEX_FILE, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        self.forward_index = index_data["forward_index"]
        
        self.doc_vectors = None
        self.doc_ids = None

    def build(self):
        """Режим обучения: векторизация корпуса"""
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
        self.doc_ids = [doc["id"] for doc in corpus]
        texts = [doc["text"] for doc in corpus]
        
        print(f"Векторизация {len(texts)} документов...")
        vectors = self.model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True 
        )
        
        np.save(NEURAL_VECTORS_FILE, vectors)
        with open(NEURAL_DOC_IDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.doc_ids, f)
            
        print(f"Готово! Матрица {vectors.shape} сохранена в '{NEURAL_VECTORS_FILE}'")

    def load(self):
        """Загрузка готовых векторов"""
        self.doc_vectors = np.load(NEURAL_VECTORS_FILE)
        with open(NEURAL_DOC_IDS_FILE, 'r', encoding='utf-8') as f:
            self.doc_ids = json.load(f)
        print(f"Загружено {len(self.doc_ids)} документов, размерность векторов: {self.doc_vectors.shape[1]}")

    def encode_query(self, query: str) -> np.ndarray:
        """Пункт 4: Преобразование запроса в вектор"""
        return self.model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 5) -> list:
        """Пункт 5: Точный поиск по сходству запроса со всеми векторами коллекции"""
        query_vector = self.encode_query(query)
        
        similarities = self.doc_vectors @ query_vector
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            title = self.forward_index[doc_id]["title"]
            summary = self.forward_index[doc_id]["summary"]
            score = float(similarities[idx])
            results.append((doc_id, title, summary, score))
            
        return results

def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python neural_search.py build  - векторизация корпуса (режим обучения)")
        print("  python neural_search.py search - режим поиска")
        return

    mode = sys.argv[1].lower()
    searcher = NeuralSearcher()

    if mode == "build":
        searcher.build()
    elif mode == "search":
        searcher.load()
        
        try:
            with open("generated_data/json/queries.json", 'r', encoding='utf-8') as f:
                queries_data = json.load(f)
            # БЕРЕМ РУССКИЕ ЗАПРОСЫ для демонстрации кросс-языкового поиска!
            queries = (
                [q["ru"] if isinstance(q, dict) else q for q in queries_data["queries"]["exact_terms"][:3]] +
                [q["ru"] if isinstance(q, dict) else q for q in queries_data["queries"]["detailed_queries"][:2]] +
                [q["ru"] if isinstance(q, dict) else q for q in queries_data["queries"]["short_queries"][:3]] +
                [q["ru"] if isinstance(q, dict) else q for q in queries_data["queries"]["error_queries"][:2]]
            )
            print(f"Загружено {len(queries)} тестовых запросов (RU)\n")
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
                for rank, (doc_id, title, summary, score) in enumerate(results, 1):
                    print(f"{rank}. [{doc_id}] (score: {score:.4f})")
                    print(f"   Title: {title}")
                    print(f"   Summary: {summary[:150]}...")
                    print()

if __name__ == "__main__":
    main()