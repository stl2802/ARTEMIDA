import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from neural_train_process_data.feature_extractor import FeatureExtractor
from searches.hybrid_search import HybridSearcher

QUERIES_FILE = Path("generated_data") / "json" / "queries.json"
OUTPUT_DATASET_FILE = Path("generated_data") / "json" / "ltr_dataset.json"


def prepare_dataset(top_k_candidates=30):
    """
    Подготовка набора данных для обучения модели ранжирования.
    
    6 признаков:
    1. BM25 Score
    2. TF-IDF Cosine Similarity
    3. Neural Cosine Similarity
    4. Word Overlap (Text)
    5. Word Overlap (Title)
    6. Document Length
    
    Метки релевантности:
    3 - высокая (топ 5)
    2 - средняя (места 6-15)
    1 - низкая (места 16-30)
    """
    print("-" * 60)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ LTR")
    print("-" * 60)
    
    print("\n[1] Инициализация FeatureExtractor...")
    extractor = FeatureExtractor()
    
    print("\n[2] Инициализация HybridSearcher...")
    searcher = HybridSearcher()
    QUERIES_FILE.parent.mkdir(exist_ok=True)
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    all_queries = []
    for category in ["exact_terms", "detailed_queries", "short_queries", "error_queries"]:
        for query_obj in queries_data["queries"][category]:
            all_queries.append(query_obj["en"])
    
    print(f"\n[3] Всего запросов: {len(all_queries)}")
    
    dataset = []
    
    print("\n[4] Обработка запросов...")
    
    for query_idx, query_text in enumerate(all_queries):
        query_id = f"q_{query_idx + 1:03d}"
        
        candidates = searcher.get_candidates_with_scores(query_text, top_k_per_method=15)
        
        candidates = searcher._normalize_scores(candidates, "bm25_score")
        candidates = searcher._normalize_scores(candidates, "cosine_score")
        
        for item in candidates:
            item["hybrid_score"] = 0.5 * item["bm25_score_norm"] + 0.5 * item["cosine_score_norm"]
        
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        for rank, item in enumerate(candidates[:top_k_candidates]):
            doc_id = item["doc_id"]
            
            features = extractor.get_features(query_text, doc_id)

            # Метки релевантности:
            # 1 - полностью соответствует (топ-5)
            # 2 - частично соответствует (места 6-15)
            # 3 - не соответствует (места 16-30)
            if rank < 5:
                relevance_label = 1
            elif rank < 15:
                relevance_label = 2
            else:
                relevance_label = 3
            
            data_pair = {
                "query_id": query_id,
                "query_text": query_text,
                "doc_id": doc_id,
                "relevance_label": relevance_label,
                "features": {
                    "bm25_score": features[0],
                    "tfidf_cosine": features[1],
                    "neural_cosine": features[2],
                    "word_overlap_text": features[3],
                    "word_overlap_title": features[4],
                    "doc_length": features[5]
                }
            }
            dataset.append(data_pair)
        
        if (query_idx + 1) % 10 == 0:
            print(f"   Обработано {query_idx + 1}/{len(all_queries)} запросов")
    
    print(f"\n[5] Сохранение датасета...")
    OUTPUT_DATASET_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n   Файл: {OUTPUT_DATASET_FILE}")
    print(f"   Всего пар: {len(dataset)}")
    
    labels_count = {1: 0, 2: 0, 3: 0}
    for item in dataset:
        labels_count[item["relevance_label"]] += 1
    
    print(f"\n   Распределение меток:")
    print(f"   - Метка 1 (полностью соответствует): {labels_count[1]}")
    print(f"   - Метка 2 (частично соответствует): {labels_count[2]}")
    print(f"   - Метка 3 (не соответствует): {labels_count[3]}")
    
    print("\n" + "-" * 60)
    print("ГОТОВО")
    print("-" * 60)


if __name__ == "__main__":
    prepare_dataset()