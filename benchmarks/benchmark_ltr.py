import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
from searches.hybrid_search import HybridSearcher
from searches.ltr_search import LTRSearcher

TEST_FILE = Path("generated_data") / "json" / "ltr_test.json"


def dcg_at_k(relevances, k):
    """DCG@k для списка релевантностей."""
    relevances = np.array(relevances[:k])
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k, ideal_relevances):
    """nDCG@k для списка релевантностей."""
    dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def precision_at_k(relevances, k):
    """Precision@k - доля релевантных (оценка 1 или 2) среди первых k."""
    if k == 0:
        return 0.0
    relevant_count = sum(1 for r in relevances[:k] if r in [1, 2])
    return relevant_count / k


def main():
    print("=" * 70)
    print("СРАВНЕНИЕ ГИБРИДНОГО ПОИСКА И LTR")
    print("=" * 70)
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    query_groups = {}
    for item in test_data:
        qid = item["query_id"]
        if qid not in query_groups:
            query_groups[qid] = {
                "query_text": item["query_text"],
                "docs": {}
            }
        query_groups[qid]["docs"][item["doc_id"]] = item["relevance_label"]
    
    print(f"\nТестовых запросов: {len(query_groups)}")
    
    print("\nИнициализация поисковиков...")
    print("\n[1/2] Гибридный поиск (без LTR):")
    hybrid = HybridSearcher()
    
    print("\n[2/2] LTR-поиск:")
    ltr = LTRSearcher()
    
    ALPHA = 0.5
    
    results_hybrid = {"precision": [], "ndcg": [], "times": []}
    results_ltr = {"precision": [], "ndcg": [], "times": []}
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПО ЗАПРОСАМ")
    print("=" * 70)
    
    for query_id, query_data in query_groups.items():
        query_text = query_data["query_text"]
        docs_rel = query_data["docs"]
        
        start = time.perf_counter()
        hybrid_results = hybrid.hybrid_search(query_text, top_k=5, alpha=ALPHA)
        time_hybrid = time.perf_counter() - start
        
        start = time.perf_counter()
        ltr_results = ltr.search(query_text, top_k=5, top_k_candidates=30)
        time_ltr = time.perf_counter() - start
        
        hybrid_doc_ids = [r["doc_id"] for r in hybrid_results]
        ltr_doc_ids = [r["doc_id"] for r in ltr_results]
        
        hybrid_rels = [docs_rel.get(doc_id, 3) for doc_id in hybrid_doc_ids]
        ltr_rels = [docs_rel.get(doc_id, 3) for doc_id in ltr_doc_ids]
        
        ideal_rels = sorted(docs_rel.values())
        
        p_hybrid = precision_at_k(hybrid_rels, 5)
        p_ltr = precision_at_k(ltr_rels, 5)
        
        ndcg_hybrid = ndcg_at_k(hybrid_rels, 5, ideal_rels)
        ndcg_ltr = ndcg_at_k(ltr_rels, 5, ideal_rels)
        
        results_hybrid["precision"].append(p_hybrid)
        results_hybrid["ndcg"].append(ndcg_hybrid)
        results_hybrid["times"].append(time_hybrid)
        
        results_ltr["precision"].append(p_ltr)
        results_ltr["ndcg"].append(ndcg_ltr)
        results_ltr["times"].append(time_ltr)
        
        print(f"\n{'─' * 70}")
        print(f"Запрос: '{query_text}'")
        print(f"{'─' * 70}")
        print(f"  Hybrid: P@5={p_hybrid:.2f} nDCG@5={ndcg_hybrid:.3f} time={time_hybrid*1000:.1f}ms")
        print(f"     LTR: P@5={p_ltr:.2f} nDCG@5={ndcg_ltr:.3f} time={time_ltr*1000:.1f}ms")
        print(f"  Hybrid docs: {hybrid_doc_ids}")
        print(f"     LTR docs: {ltr_doc_ids}")
    
    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ МЕТРИКИ (среднее по запросам)")
    print("=" * 70)
    
    avg_p_hybrid = np.mean(results_hybrid["precision"])
    avg_p_ltr = np.mean(results_ltr["precision"])
    avg_ndcg_hybrid = np.mean(results_hybrid["ndcg"])
    avg_ndcg_ltr = np.mean(results_ltr["ndcg"])
    avg_time_hybrid = np.mean(results_hybrid["times"]) * 1000
    avg_time_ltr = np.mean(results_ltr["times"]) * 1000
    
    print(f"\n{'Метрика':<20} {'Hybrid':<15} {'LTR':<15} {'Изменение':<15}")
    print(f"{'-' * 65}")
    print(f"{'Precision@5':<20} {avg_p_hybrid:<15.3f} {avg_p_ltr:<15.3f} {(avg_p_ltr - avg_p_hybrid) / max(avg_p_hybrid, 0.001) * 100:+.1f}%")
    print(f"{'nDCG@5':<20} {avg_ndcg_hybrid:<15.3f} {avg_ndcg_ltr:<15.3f} {(avg_ndcg_ltr - avg_ndcg_hybrid) / max(avg_ndcg_hybrid, 0.001) * 100:+.1f}%")
    print(f"{'Время (мс)':<20} {avg_time_hybrid:<15.1f} {avg_time_ltr:<15.1f} {(avg_time_ltr - avg_time_hybrid) / avg_time_hybrid * 100:+.1f}%")
    
    print("\n" + "=" * 70)
    print("ВЫВОД")
    print("=" * 70)
    
    if avg_p_ltr > avg_p_hybrid:
        print(f"Precision@5 улучшен на {(avg_p_ltr - avg_p_hybrid) / max(avg_p_hybrid, 0.001) * 100:.1f}%")
    else:
        print(f"Precision@5 снижен на {(avg_p_hybrid - avg_p_ltr) / max(avg_p_hybrid, 0.001) * 100:.1f}%")
    
    if avg_ndcg_ltr > avg_ndcg_hybrid:
        print(f"nDCG@5 улучшен на {(avg_ndcg_ltr - avg_ndcg_hybrid) / max(avg_ndcg_hybrid, 0.001) * 100:.1f}%")
    else:
        print(f"nDCG@5 снижен на {(avg_ndcg_hybrid - avg_ndcg_ltr) / max(avg_ndcg_hybrid, 0.001) * 100:.1f}%")
    
    print(f"\nДополнительное время на LTR: {avg_time_ltr - avg_time_hybrid:.1f} мс")


if __name__ == "__main__":
    main()
