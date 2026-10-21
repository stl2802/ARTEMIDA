import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
from typing import Tuple, Optional

try:
    from searches.neural_search import NeuralSearcher
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("Внимание: neural_search недоступен (sentence_transformers не установлен)")

try:
    from searches.ltr_search import LTRSearcher
    LTR_AVAILABLE = os.path.exists("generated_data/model/ltr_model.pkl")
except ImportError:
    LTR_AVAILABLE = False
    print("Внимание: ltr_search недоступен")

try:
    from searches.hnsw_search import HnswSearcher
    HNSW_AVAILABLE = os.path.exists("generated_data/bin/hnsw_index.bin")
except ImportError:
    HNSW_AVAILABLE = False
    print("Внимание: hnsw_search недоступен")

from searches.lecsika_search import load_index, calculate_bm25, tokenize
from searches.hybrid_search import HybridSearcher


class QueryAnalyzer:
    """Анализатор запросов для определения типа поиска"""

    def __init__(self):
        # Русские буквы
        self.russian_pattern = re.compile(r'[а-яёА-ЯЁ]')
        # Паттерн для термина
        self.term_pattern = re.compile(r'^[a-zA-Z\s]{1,20}$')

    def detect_language(self, query: str) -> str:
        """Определение языка запроса"""
        if self.russian_pattern.search(query):
            return 'russian'
        return 'english'

    def is_term_query(self, query: str) -> bool:
        """Проверка, является ли запрос термином"""
        tokens = query.strip().split()
        if len(tokens) > 3:
            return False
        if len(query) < 30 and not any(c in '.,!?;:' for c in query):
            return True
        return False

    def is_conceptual_query(self, query: str) -> bool:
        """Проверка, является ли запрос смысловым (вопрос, описание)"""
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'describe'}
        if any(word in query.lower().split() for word in question_words):
            return True
        if len(query.split()) > 3:
            return True
        return False

    def analyze(self, query: str, neural_available: bool = True, ltr_available: bool = True) -> Tuple[str, str]:
        """
        Анализ запроса и определение типа поиска

        Возвращает: (search_type, reason)
        search_type: 'neural', 'lexical', 'hybrid', 'ltr', 'hnsw'
        """
        language = self.detect_language(query)

        if language == 'russian' and neural_available:
            return 'neural', 'russian_language'
        elif language == 'russian' and not neural_available:
            return 'hybrid', 'russian_language_fallback'

        if self.is_term_query(query):
            return 'lexical', 'short_term'

        if self.is_conceptual_query(query) and ltr_available:
            return 'ltr', 'conceptual_query_with_ltr'

        if self.is_conceptual_query(query):
            return 'hybrid', 'conceptual_query'
        

        return 'hybrid', 'default'


class SmartHybridSearcher:
    """Умный гибридный поисковик с автоматическим выбором метода"""

    def __init__(self):
        print("Инициализация умного гибридного поисковика...")
        self.analyzer = QueryAnalyzer()

        if NEURAL_AVAILABLE:
            print("Загрузка нейронного поисковика...")
            self.neural_searcher = NeuralSearcher()
            self.neural_searcher.load()
        else:
            self.neural_searcher = None
            print("Нейронный поисковик недоступен")

        if LTR_AVAILABLE:
            print("Загрузка LTR поисковика...")
            self.ltr_searcher = LTRSearcher()
        else:
            self.ltr_searcher = None
            print("LTR поисковик недоступен")

        if HNSW_AVAILABLE:
            print("Загрузка HNSW поисковика...")
            self.hnsw_searcher = HnswSearcher()
        else:
            self.hnsw_searcher = None
            print("HNSW поисковик недоступен")

        print("Загрузка лексического индекса...")
        self.forward_index, self.inverted_index, self.avg_doc_length, self.doc_count = \
            load_index("generated_data/json/inverted_index.json")

        print("Загрузка гибридного поисковика...")
        self.hybrid_searcher = HybridSearcher()

        print("Умный гибридный поисковик готов!\n")

    def search(self, query: str, top_k: int = 5, force_method: Optional[str] = None) -> dict:
        """
        Умный поиск с автоматическим выбором метода

        Args:
            query: поисковый запрос
            top_k: количество результатов
            force_method: принудительно указать метод ('neural', 'lexical', 'hybrid', 'ltr', 'hnsw')

        Returns:
            dict с результатами и информацией о выбранном методе
        """
        if force_method:
            search_type = force_method
            reason = 'forced'
        else:
            search_type, reason = self.analyzer.analyze(query, neural_available=NEURAL_AVAILABLE, ltr_available=LTR_AVAILABLE)

        print(f"Анализ запроса: '{query}'")
        print(f"Выбранный метод: {search_type} (причина: {reason})")

        if search_type == 'neural':
            if not self.neural_searcher:
                print("Нейронный поисковик недоступен, используем гибридный")
                results = self._hybrid_search(query, top_k)
                search_type = 'hybrid'
                reason = 'neural_unavailable_fallback'
            else:
                results = self._neural_search(query, top_k)
        elif search_type == 'lexical':
            results = self._lexical_search(query, top_k)
        elif search_type == 'hybrid':
            results = self._hybrid_search(query, top_k)
        elif search_type == 'ltr':
            if not self.ltr_searcher:
                print("LTR поисковик недоступен, используем гибридный")
                results = self._hybrid_search(query, top_k)
                search_type = 'hybrid'
                reason = 'ltr_unavailable_fallback'
            else:
                results = self._ltr_search(query, top_k)
        elif search_type == 'hnsw':
            if not self.hnsw_searcher:
                print("HNSW поисковик недоступен, используем гибридный")
                results = self._hybrid_search(query, top_k)
                search_type = 'hybrid'
                reason = 'hnsw_unavailable_fallback'
            else:
                results = self._hnsw_search(query, top_k)
        else:
            raise ValueError(f"Неизвестный тип поиска: {search_type}")

        return {
            'method': search_type,
            'reason': reason,
            'results': results
        }

    def _neural_search(self, query: str, top_k: int) -> list:
        """Нейронный поиск"""
        results = self.neural_searcher.search(query, top_k=top_k)
        formatted = []
        for doc_id, title, summary, score in results:
            formatted.append({
                'doc_id': doc_id,
                'title': title,
                'summary': summary,
                'score': score,
                'score_type': 'cosine'
            })
        return formatted

    def _lexical_search(self, query: str, top_k: int) -> list:
        """Лексический поиск (BM25)"""
        query_tokens = tokenize(query)
        scores = []
        for doc_id in self.forward_index:
            score = calculate_bm25(
                query_tokens, doc_id,
                self.forward_index, self.inverted_index,
                self.avg_doc_length
            )
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        formatted = []
        for doc_id, score in top_results:
            entry = self.forward_index[doc_id]
            formatted.append({
                'doc_id': doc_id,
                'title': entry.title,
                'summary': entry.summary,
                'score': score,
                'score_type': 'bm25'
            })
        return formatted

    def _hybrid_search(self, query: str, top_k: int) -> list:
        """Гибридный поиск"""
        results = self.hybrid_searcher.hybrid_search(query, top_k=top_k, alpha=0.5)
        formatted = []
        for res in results:
            formatted.append({
                'doc_id': res['doc_id'],
                'title': res['title'],
                'summary': res['summary'],
                'score': res['hybrid_score'],
                'score_type': 'hybrid',
                'bm25_score': res['bm25_score'],
                'cosine_score': res['cosine_score']
            })
        return formatted

    def _ltr_search(self, query: str, top_k: int) -> list:
        """LTR поиск (LightGBM на деревьях)"""
        results = self.ltr_searcher.search(query, top_k=top_k)
        formatted = []
        for res in results:
            entry = self.ltr_searcher.feature_extractor.forward_index[res["doc_id"]]
            formatted.append({
                'doc_id': res['doc_id'],
                'title': entry.title,
                'summary': entry.summary,
                'score': res['ltr_score'],
                'score_type': 'ltr',
                'bm25_score': res['bm25_score'],
                'tfidf_cosine': res['tfidf_cosine'],
                'neural_cosine': res['neural_cosine']
            })
        return formatted

    def _hnsw_search(self, query: str, top_k: int) -> list:
        """HNSW поиск (приближенный векторный поиск)"""
        results = self.hnsw_searcher.approximate_search(query, top_k=top_k)
        formatted = []
        for doc_id, score in results:
            entry = self.hnsw_searcher.forward_index[doc_id]
            formatted.append({
                'doc_id': doc_id,
                'title': entry['title'],
                'summary': entry['summary'],
                'score': score,
                'score_type': 'hnsw'
            })
        return formatted


def search_mode():
    """Интерактивный режим умного поиска"""
    print("=" * 70)
    print("УМНЫЙ ГИБРИДНЫЙ ПОИСК")
    print("=" * 70)
    print("\nПоисковик автоматически выберет метод:")
    print("  - Русский текст          -> Нейронный (multilingual)")
    print("  - Короткий термин        -> Лексический (BM25)")
    print("  - Смысловой запрос       -> LTR (LightGBM) или Гибридный")
    print("  - Принудительный метод   -> neural/lexical/hybrid/ltr/hnsw")
    print("=" * 70)

    searcher = SmartHybridSearcher()

    test_queries = []
    try:
        with open("generated_data/json/queries.json", 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        test_queries = (
            [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["exact_terms"][:2]] +
            [q["ru"] if isinstance(q, dict) else q for q in queries_data["queries"]["exact_terms"][:2]] +
            [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["detailed_queries"][:2]]
        )
        print(f"\nЗагружено {len(test_queries)} тестовых запросов\n")
    except Exception as e:
        print(f"Ошибка загрузки queries.json: {e}\n")

    while True:
        if test_queries:
            query = test_queries.pop(0)
            print(f"\n" + "-" * 70)
            print(f"Автоматический запрос: '{query}'")
            force_method = None
        else:
            query = input("\nВведите поисковый запрос (или 'exit' для выхода): ")
            if query.lower() == 'exit':
                break

            method_input = input("Метод [Enter=auto, neural/lexical/hybrid/ltr/hnsw]: ").strip().lower()
            force_method = method_input if method_input in ['neural', 'lexical', 'hybrid', 'ltr', 'hnsw'] else None

        try:
            result = searcher.search(query, top_k=5, force_method=force_method)

            if not result['results']:
                print("\nДокументов не найдено\n")
                continue

            print(f"\nТоп-5 результатов (метод: {result['method']}):")
            print("-" * 70)

            for rank, res in enumerate(result['results'], 1):
                print(f"\n{rank}. [{res['doc_id']}] ({res['score_type']}: {res['score']:.4f})")
                print(f"   Title: {res['title']}")
                print(f"   Summary: {res['summary'][:150]}...")
                if 'bm25_score' in res and 'cosine_score' in res:
                    print(f"   BM25: {res['bm25_score']:.4f} | Cosine: {res['cosine_score']:.4f}")
                elif 'tfidf_cosine' in res and 'neural_cosine' in res:
                    print(f"   BM25: {res['bm25_score']:.4f} | TF-IDF: {res['tfidf_cosine']:.4f} | Neural: {res['neural_cosine']:.4f}")

        except Exception as e:
            print(f"\nОшибка поиска: {e}\n")


def main():
    import sys
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python smart_hybrid_search.py search - режим умного поиска")
        return

    mode = sys.argv[1].lower()

    if mode == "search":
        search_mode()
    else:
        print(f"Неизвестный режим: {mode}")
        print("Используйте 'search'")


if __name__ == "__main__":
    main()
