import json
import os
import sys
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TRAIN_FILE = "generated_data/json/ltr_train.json"
TEST_FILE = "generated_data/json/ltr_test.json"


MODEL_DIR = "generated_data/model"
MODEL_FILE = os.path.join(MODEL_DIR, "ltr_model.pkl")
FEATURE_ORDER_FILE = os.path.join(MODEL_DIR, "feature_order.json")

FEATURE_NAMES = [
    "bm25_score",           # 1. Оценка BM25
    "tfidf_cosine",         # 2. TF-IDF Cosine Similarity
    "neural_cosine",        # 3. Neural Cosine Similarity
    "word_overlap_text",    # 4. Word Overlap (Text)
    "word_overlap_title",   # 5. Word Overlap (Title)
    "doc_length"            # 6. Document Length
]


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


def train_model():
    """Обучаем модель градиентного бустинга."""
    print("-" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ LTR")
    print("-" * 60)
    
    print("\n[1] Загрузка данных...")
    X_train, y_train = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    
    print(f"   Обучающая выборка: {len(X_train)} примеров")
    print(f"   Тестовая выборка: {len(X_test)} примеров")
    print(f"   Количество признаков: {X_train.shape[1]}")
    
    print("\n[2] Создание модели...")
    print("   Алгоритм: LightGBM (Gradient Boosting)")
    print("   Параметры:")
    print("   - n_estimators: 100")
    print("   - max_depth: 5")
    print("   - learning_rate: 0.1")
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='regression',
        random_state=42,
        verbose=-1
    )
    
    print("\n[3] Обучение модели...")
    model.fit(X_train, y_train)
    print("   Готово!")
    
    print("\n[4] Оценка качества:")
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\n   Обучающая выборка:")
    print(f"   - MSE: {mse_train:.4f}")
    print(f"   - MAE: {mae_train:.4f}")
    
    print(f"\n   Тестовая выборка:")
    print(f"   - MSE: {mse_test:.4f}")
    print(f"   - MAE: {mae_test:.4f}")
    
    ndcg_train = compute_ndcg(y_train, y_pred_train, k=10)
    ndcg_test = compute_ndcg(y_test, y_pred_test, k=10)
    
    print(f"\n   NDCG@10:")
    print(f"   - Train: {ndcg_train:.4f}")
    print(f"   - Test: {ndcg_test:.4f}")

    print("\n[5] Важность признаков:")
    importances = model.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True):
        print(f"   - {name}: {imp:.2f}")

    print("\n[6] Сохранение модели...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"   Модель: {MODEL_FILE}")

    with open(FEATURE_ORDER_FILE, 'w', encoding='utf-8') as f:
        json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)
    print(f"   Порядок признаков: {FEATURE_ORDER_FILE}")
    
    print("\n" + "-" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("-" * 60)
    
    return model


if __name__ == "__main__":
    train_model()
