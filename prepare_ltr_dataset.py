import json
import sys
from hybrid_search import HybridSearcher

QUERIES_FILE = "json/queries.json"
OUTPUT_DATASET_FILE = "json/ltr_dataset.json"

def prepare_dataset(top_k_candidates=30):
    """
    Подготовка набора данных для обучения модели ранжирования.
    Шкала релевантности: 3 (полное соответствие), 2 (частичное), 1 (не соответствует)
    """
    print("Инициализация поисковика для сбора данных...")
    searcher = HybridSearcher()
    
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    all_queries = []
    for category in ["exact_terms", "detailed_queries", "short_queries", "error_queries"]:
        for query_obj in queries_data["queries"][category]:
            all_queries.append(query_obj["en"])
    
    dataset = []
    
    print(f"Обработка {len(all_queries)} запросов...")
    
    for query_idx, query_text in enumerate(all_queries):
        query_id = f"q_{query_idx + 1:03d}"
        
        #(берем с запасом, например, 30)
        #alpha=0.5 - база )))
        candidates_with_scores = searcher.get_candidates_with_scores(
            query_text, top_k_per_method=15 # 15+15 = ~25-30 уникальных
        )
        
        candidates_with_scores = searcher._normalize_scores(candidates_with_scores, "bm25_score")
        candidates_with_scores = searcher._normalize_scores(candidates_with_scores, "cosine_score")
        
        for item in candidates_with_scores:
            item["hybrid_score"] = 0.5 * item["bm25_score_norm"] + 0.5 * item["cosine_score_norm"]
            
        candidates_with_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # метки релевантности
        # 3 -  (Топ 5)
        # 2 -  (Места 6-15)
        # 1 -  (Места 16-30)
        for rank, item in enumerate(candidates_with_scores[:top_k_candidates]):
            if rank < 5:
                relevance_label = 3 
            elif rank < 15:
                relevance_label = 2
            else:
                relevance_label = 1
                
            data_pair = {
                "query_id": query_id,
                "query_text": query_text,
                "doc_id": item["doc_id"],
                "relevance_label": relevance_label,
                "features": {
                    "bm25_score": item["bm25_score"],
                    "cosine_score": item["cosine_score"],
                    "bm25_norm": item["bm25_score_norm"],
                    "cosine_norm": item["cosine_score_norm"],
                    "hybrid_score": item["hybrid_score"]
                }
            }
            dataset.append(data_pair)
            
    with open(OUTPUT_DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Набор данных успешно сохранен в '{OUTPUT_DATASET_FILE}'")
    print(f"   Всего пар (запрос-документ): {len(dataset)}")
    
    labels_count = {1: 0, 2: 0, 3: 0}
    for item in dataset:
        labels_count[item["relevance_label"]] += 1
        
    print(f"   Распределение меток (шкала 1-3, где 3 - наивысшая релевантность):")
    print(f"   - Метка 3 (полное соответствие): {labels_count[3]}")
    print(f"   - Метка 2 (частичное соответствие): {labels_count[2]}")
    print(f"   - Метка 1 (не соответствует): {labels_count[3]}")


if __name__ == "__main__":
    prepare_dataset()