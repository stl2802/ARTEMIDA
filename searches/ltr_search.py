import sys
import os
import json
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dotenv import load_dotenv
from pathlib import Path


env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_train_process_data.feature_extractor import FeatureExtractor
from searches.hybrid_search import HybridSearcher


TRAIN_FILE = Path(os.getenv('TRAIN_FILE', 'generated_data/json/ltr_train.json'))
TEST_FILE = Path(os.getenv('TEST_FILE', 'generated_data/json/ltr_test.json'))

MODEL_DIR = Path(os.getenv('MODEL_DIR', 'generated_data/model'))
MODEL_FILE = MODEL_DIR / os.getenv('MODEL_FILE', 'ltr_model.pkl')
FEATURE_ORDER_FILE = MODEL_DIR / os.getenv('FEATURE_ORDER_FILE', 'feature_order.json')


N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', '100'))
MAX_DEPTH = int(os.getenv('MAX_DEPTH', '5'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.1'))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', '42'))


FEATURE_NAMES = [
    "bm25_score",           # 1. Оценка BM25
    "tfidf_cosine",         # 2. TF-IDF Cosine Similarity
    "neural_cosine",        # 3. Neural Cosine Similarity
    "word_overlap_text",    # 4. Word Overlap (Text)
    "word_overlap_title",   # 5. Word Overlap (Title)
    "doc_length"            # 6. Document Length
]


class LTRSearcher:
    """
    Поисковая система на основе Learning to Rank.
    Переупорядочивает результаты гибридного поиска с помощью ML-модели.
    """
    
    def __init__(self):
        """Загружаем модель и инициализируем извлекатель признаков."""
        print("Инициализация LTR-поисковика...")
        
        self.model = joblib.load(MODEL_FILE)
        print(f"Модель загружена: {MODEL_FILE}")
        
        with open(FEATURE_ORDER_FILE, 'r', encoding='utf-8') as f:
            self.feature_order = json.load(f)
        
        self.feature_extractor = FeatureExtractor()
        self.hybrid_searcher = HybridSearcher()
        
        print("LTR-поисковик готов.\n")
    
    def _extract_features(self, query: str, doc_id: str) -> list:
        """
        Извлекаем признаки для пары (запрос, документ).
        
        Возвращаем список из 6 признаков:
        1. bm25_score
        2. tfidf_cosine
        3. neural_cosine
        4. word_overlap_text
        5. word_overlap_title
        6. doc_length
        """
        return self.feature_extractor.get_features(query, doc_id)
    
    def search(self, query: str, top_k: int = 10, top_k_candidates: int = 30) -> list:
        """
        Переупорядочиваем результаты поиска с помощью LTR-модели.
        
        Шаги:
        1. Получаем кандидатов из гибридного поиска
        2. Извлекаем признаки для каждого кандидата
        3. Предсказываем релевантность с помощью модели
        4. Сортируем по убыванию предсказания
        
        Возвращаем список словарей с doc_id и предсказанием.
        """
        candidates = self.hybrid_searcher.get_candidates_with_scores(query, top_k_per_method=15)
        
        if not candidates:
            return []
        
        candidates = self.hybrid_searcher._normalize_scores(candidates, "bm25_score")
        candidates = self.hybrid_searcher._normalize_scores(candidates, "cosine_score")
        
        for item in candidates:
            item["hybrid_score"] = 0.5 * item["bm25_score_norm"] + 0.5 * item["cosine_score_norm"]
        
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        candidates = candidates[:top_k_candidates]
        
        results = []
        for item in candidates:
            doc_id = item["doc_id"]
            
            features = self._extract_features(query, doc_id)

            features_vec = np.array([features])

            prediction = self.model.predict(features_vec)[0]
            
            results.append({
                "doc_id": doc_id,
                "ltr_score": prediction,
                "bm25_score": features[0],
                "tfidf_cosine": features[1],
                "neural_cosine": features[2],
                "word_overlap_text": features[3],
                "word_overlap_title": features[4],
                "doc_length": features[5]
            })

        results.sort(key=lambda x: x["ltr_score"], reverse=True)
        
        return results[:top_k]


def load_data(filepath):
    """Загружаем данные из JSON и разделяем на признаки (X) и метки (y)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    X = []
    y = []
    
    for item in data:
        features = [
            item["features"]["bm25_score"],
            item["features"]["tfidf_cosine"],
            item["features"]["neural_cosine"],
            item["features"]["word_overlap_text"],
            item["features"]["word_overlap_title"],
            item["features"]["doc_length"]
        ]
        X.append(features)
        y.append(item["relevance_label"])
    
    return np.array(X), np.array(y)


def compute_ndcg(y_true, y_pred, k=10):
    """Вычисляем NDCG@k — метрику качества ранжирования."""
    order = np.argsort(y_pred)[::-1]
    top_k_true = y_true[order][:k]
    
    gains = top_k_true
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)
    
    ideal_gains = np.sort(y_true)[::-1][:k]
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = np.sum(ideal_gains / ideal_discounts)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def train_mode():
    """Режим обучения модели."""
    print("=" * 70)
    print("РЕЖИМ ОБУЧЕНИЯ")
    print("=" * 70)
    
    print("\n[1] Загрузка данных...")
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    print(f"   Обучающая выборка: {len(X_train)} примеров")
    print(f"   Тестовая выборка: {len(X_test)} примеров")
    print(f"   Количество признаков: {X_train.shape[1]}")
    
    print("\n[2] Обучение модели LightGBM...")
    print("   Алгоритм: Gradient Boosting Regressor")
    print("   Параметры:")
    print(f"   - n_estimators: {N_ESTIMATORS}")
    print(f"   - max_depth: {MAX_DEPTH}")
    print(f"   - learning_rate: {LEARNING_RATE}")
    
    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        objective='regression',
        random_state=RANDOM_STATE,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    print("\n[3] Оценка качества:")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print(f"\n   Обучающая выборка:")
    print(f"   - MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
    print(f"   - MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
    
    print(f"\n   Тестовая выборка:")
    print(f"   - MSE: {mean_squared_error(y_test, y_pred_test):.4f}")
    print(f"   - MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
    
    ndcg_train = compute_ndcg(y_train, y_pred_train, k=10)
    ndcg_test = compute_ndcg(y_test, y_pred_test, k=10)
    
    print(f"\n   NDCG@10:")
    print(f"   - Train: {ndcg_train:.4f}")
    print(f"   - Test: {ndcg_test:.4f}")
    
    print("\n[4] Важность признаков:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"   - {name}: {imp:.2f}")
    
    print("\n[5] Сохранение модели...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"   Модель: {MODEL_FILE}")
    
    with open(FEATURE_ORDER_FILE, 'w', encoding='utf-8') as f:
        json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)
    print(f"   Порядок признаков: {FEATURE_ORDER_FILE}")
    
    print("\n" + "=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    
    return model


def search_mode():
    """Режим интерактивного поиска."""
    print("=" * 70)
    print("РЕЖИМ ПОИСКА")
    print("=" * 70)
    
    try:
        searcher = LTRSearcher()
    except FileNotFoundError as e:
        print(f"\nОшибка: {e}")
        return
    
    queries_file = Path("generated_data") / "json" / "queries.json"
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        test_queries = (
            [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["exact_terms"][:3]] +
            [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["detailed_queries"][:2]] +
            [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["short_queries"][:3]] +
            [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["error_queries"][:2]]
        )
        print(f"Загружено {len(test_queries)} тестовых запросов (EN)\n")
    except FileNotFoundError:
        print(f"\nФайл запросов не найден: {queries_file}")
        print("Используем тестовые запросы по умолчанию...\n")
        test_queries = [
            # Точные термины
            "neural network optimization",
            "quantum computing algorithms",
            "machine learning classification",
            # Описательные запросы
            "how to improve model accuracy in deep learning",
            "methods for natural language processing",
            # Короткие запросы
            "transformer architecture",
            "gradient descent",
            "attention mechanism",
            # Сложные/ошибочные запросы
            "optimization techniques for training large scale models",
            "comparison of supervised and unsupervised learning"
        ]
    
    while True:
        if test_queries:
            query = test_queries.pop(0)
            print(f"Запрос: '{query}'")
        else:
            try:
                query = input("\nВведите запрос (или 'exit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nДо свидания!")
                break
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nДо свидания!")
                break
            
            if not query:
                continue
        
        results = searcher.search(query, top_k=5)
        
        if not results:
            print("\nДокументов не найдено.\n")
            continue
        
        print(f"\nРезультаты поиска по запросу '{query}':\n")
        print("-" * 70)
        
        for i, res in enumerate(results, 1):
            entry = searcher.feature_extractor.forward_index[res["doc_id"]]
            full_text = searcher.hybrid_searcher.corpus_text.get(res["doc_id"], "")
            text_preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
            
            print(f"\n{i}. {res['doc_id']} — {entry.title}")
            print(f"   {entry.summary}")
            print(f"   {text_preview}")
        
        print("\n" + "-" * 70 + "\n")


def main():
    if len(sys.argv) < 2:
        print("ARTEMIDA - Поисковая система с Learning to Rank\n")
        print("Использование:")
        print("  python ltr_search.py train   - обучение модели")
        print("  python ltr_search.py search  - интерактивный поиск")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "train":
        train_mode()
    elif mode == "search":
        search_mode()
    else:
        print(f"Неизвестный режим: {mode}")
        print("Используйте: train или search")


if __name__ == "__main__":
    main()