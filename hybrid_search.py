import sys
import json
import numpy as np

from lecsika_search import load_index, calculate_bm25, tokenize
from hnsw_search import HnswSearcher
#TODO:пути
INDEX_FILE = "json/inverted_index.json"
CORPUS_FILE = "json/preprocessed_corpus.json"
QUERIES_FILE = "json/queries.json"

# Значение α по умолчанию (вес лексического поиска)
DEFAULT_ALPHA = 0.5


def calculate_hybrid_score(bm25_norm: float, cosine_norm: float, alpha: float = DEFAULT_ALPHA) -> float:
    """
    Пункт 4: Функция вычисления итоговой оценки релевантности документа.
    
    Формула: score = α × normalized_BM25 + (1 − α) × normalized_cosine
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Параметр α должен быть в диапазоне [0, 1], получено: {alpha}")
    
    return alpha * bm25_norm + (1.0 - alpha) * cosine_norm


class HybridSearcher:
    def __init__(self, index_path=INDEX_FILE, corpus_path=CORPUS_FILE):
        print("Инициализация гибридного поисковика...")
        
        self.forward_index, self.inverted_index, self.avg_doc_length, self.doc_count = load_index(index_path)
        print(f"Лексический индекс загружен: {self.doc_count} документов")
        
        self.hnsw_searcher = HnswSearcher()
        
        self.doc_id_to_vec_idx = {
            doc_id: idx for idx, doc_id in enumerate(self.hnsw_searcher.doc_ids)
        }
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        self.corpus_text = {doc["id"]: doc["text"] for doc in corpus}
        print(f" Загружено {len(self.corpus_text)} полных текстов документов")
        
        print("Гибридный поисковик полностью готов\n")

    def get_candidates_with_scores(self, query: str, top_k_per_method: int = 20) -> list:
        """Получение кандидатов с оценками."""
        query_tokens = tokenize(query)
        
        bm25_all = {}
        for doc_id in self.forward_index:
            score = calculate_bm25(
                query_tokens, doc_id, 
                self.forward_index, self.inverted_index, 
                self.avg_doc_length
            )
            if score > 0:
                bm25_all[doc_id] = score
        
        bm25_top_ids = set(sorted(bm25_all.keys(), key=lambda x: bm25_all[x], reverse=True)[:top_k_per_method])
        
        hnsw_results = self.hnsw_searcher.approximate_search(query, top_k=top_k_per_method)
        hnsw_all = {doc_id: score for doc_id, score in hnsw_results}
        
        candidate_ids = list(bm25_top_ids | set(hnsw_all.keys()))
        
        results_structure = []
        for doc_id in candidate_ids:
            results_structure.append({
                "doc_id": doc_id,
                "bm25_score": bm25_all.get(doc_id, 0.0),
                "cosine_score": hnsw_all.get(doc_id, 0.0)
            })
        
        return results_structure

    def _normalize_scores(self, scores: list, key: str) -> list:
        """ Нормализация Min-Max в [0, 1]."""
        if not scores:
            return scores
        
        values = [item[key] for item in scores]
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            for item in scores:
                item[f"{key}_norm"] = 0.0
        else:
            for item in scores:
                item[f"{key}_norm"] = (item[key] - min_val) / (max_val - min_val)
        
        return scores

    def hybrid_search(self, query: str, top_k: int = 5, top_k_per_method: int = 20, alpha: float = DEFAULT_ALPHA) -> list:
        """ Финальное гибридное ранжирование."""
        candidates = self.get_candidates_with_scores(query, top_k_per_method)
        
        if not candidates:
            return []
        
        candidates = self._normalize_scores(candidates, "bm25_score")
        candidates = self._normalize_scores(candidates, "cosine_score")
        
        for item in candidates:
            item["hybrid_score"] = calculate_hybrid_score(
                bm25_norm=item["bm25_score_norm"],
                cosine_norm=item["cosine_score_norm"],
                alpha=alpha
            )
        
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        results = []
        for item in candidates[:top_k]:
            doc_id = item["doc_id"]
            entry = self.forward_index[doc_id]
            full_text = self.corpus_text.get(doc_id, "Текст недоступен")
            
            results.append({
                "doc_id": doc_id,
                "title": entry.title,
                "summary": entry.summary,
                "text": full_text,
                "bm25_score": item["bm25_score"],
                "cosine_score": item["cosine_score"],
                "bm25_norm": item["bm25_score_norm"],
                "cosine_norm": item["cosine_score_norm"],
                "hybrid_score": item["hybrid_score"]
            })
        
        return results


def build_mode():
    """
    Режим подготовки индексов: проверяет загрузку всех необходимых индексов.
    """
    print("=" * 70)
    print("РЕЖИМ ПОДГОТОВКИ ИНДЕКСОВ")
    print("=" * 70)
    
    try:
        searcher = HybridSearcher()
        print("\nВсе индексы успешно загружены и готовы к поиску!")
        print(f"   - Лексический индекс (BM25): {searcher.doc_count} документов")
        print(f"   - Векторный индекс (HNSW): размерность {len(searcher.hnsw_searcher.doc_ids)}")
        print(f"   - Корпус текстов: {len(searcher.corpus_text)} документов")
    except Exception as e:
        print(f"\n!!!Ошибка при загрузке индексов: {e}")
        print("Убедитесь, что все необходимые файлы существуют:")
        print("  - json/inverted_index.json")
        print("  - json/preprocessed_corpus.json")
        print("  - npy/doc_vectors_svd.npy")
        print("  - bin/hnsw_index.bin")
        print("  - json/doc_ids.json")


def search_mode():
    """
    Режим поиска: интерактивный поиск с указанием параметра α.
    """
    print("=" * 70)
    print("РЕЖИМ ПОИСКА")
    print("=" * 70)
    
    searcher = HybridSearcher()
    
    test_queries = []
    try:
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        test_queries = (
            [q["en"] for q in queries_data["queries"]["exact_terms"][:2]] +
            [q["en"] for q in queries_data["queries"]["detailed_queries"][:2]] +
            [q["en"] for q in queries_data["queries"]["short_queries"][:2]] +
            [q["en"] for q in queries_data["queries"]["error_queries"][:2]]
        )
        print(f"\nЗагружено {len(test_queries)} тестовых запросов\n")
    except Exception as e:
        print(f"Ошибка загрузки queries.json: {e}")
    
    while True:
        if test_queries:
            query = test_queries.pop(0)
            print(f"\n{'─' * 70}")
            print(f"Запрос: '{query}'")
        else:
            query = input("\nВведите поисковый запрос (или 'exit' для выхода): ")
            if query.lower() == 'exit':
                break
        if test_queries:
            alpha = DEFAULT_ALPHA
            print(f"Параметр α: {alpha:.2f} (по умолчанию)")
        else:
            alpha_input = input("Введите значение α от 0 до 1 (или Enter для 0.5): ")
            if alpha_input.strip() == '':
                alpha = DEFAULT_ALPHA
            else:
                try:
                    alpha = float(alpha_input)
                    if not (0.0 <= alpha <= 1.0):
                        print(f"!α должно быть в диапазоне [0, 1]. Использую α=0.5")
                        alpha = DEFAULT_ALPHA
                except ValueError:
                    print(f"!Неверное значение. Использую α=0.5")
                    alpha = DEFAULT_ALPHA
        
        results = searcher.hybrid_search(query, top_k=5, alpha=alpha)
        
        if not results:
            print("\n!Документов не найдено\n")
        else:
            print(f"\n Топ-5 результатов (α={alpha:.2f}, вес BM25={alpha:.2f}, вес Cosine={1-alpha:.2f}):")
            print(f"{'─' * 70}")
            
            for rank, res in enumerate(results, 1):
                print(f"\n{rank}. [{res['doc_id']}] (hybrid score: {res['hybrid_score']:.4f})")
                print(f"   Title: {res['title']}")
                print(f"   Summary: {res['summary']}")
                print(f"   Text: {res['text'][:300]}...") 
                print(f"   [BM25: {res['bm25_score']:.4f} | Cosine: {res['cosine_score']:.4f}]")


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python hybrid_search.py build  - подготовка индексов")
        print("  python hybrid_search.py search - режим поиска")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "build":
        build_mode()
    elif mode == "search":
        search_mode()
    else:
        print(f"Неизвестный режим: {mode}")
        print("Используйте 'build' или 'search'")


if __name__ == "__main__":
    main()