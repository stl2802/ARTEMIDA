import json
import random

DATASET_FILE = "generated_data/json/ltr_dataset.json"
TRAIN_FILE = "generated_data/json/ltr_train.json"
TEST_FILE = "generated_data/json/ltr_test.json"

def split_dataset(train_queries=30, test_queries=10, seed=42):
    """
    Разделение датасета на обучающую и тестовую выборки.
    Все пары одного запроса остаются в одной выборке.
    Запросы разных категорий равномерно распределены.
    """
    random.seed(seed)
    
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    query_groups = {}
    for item in dataset:
        qid = item["query_id"]
        if qid not in query_groups:
            query_groups[qid] = []
        query_groups[qid].append(item)
    
    all_query_ids = list(query_groups.keys())
    print(f"Всего уникальных запросов: {len(all_query_ids)}")
    
    categories = {
        "exact_terms": [],
        "detailed_queries": [],
        "short_queries": [],
        "error_queries": []
    }
    
    for qid in all_query_ids:
        num = int(qid.split("_")[1])
        if num <= 10:
            categories["exact_terms"].append(qid)
        elif num <= 20:
            categories["detailed_queries"].append(qid)
        elif num <= 30:
            categories["short_queries"].append(qid)
        else:
            categories["error_queries"].append(qid)
    
    print(f"\nЗапросов по категориям:")
    for cat, qids in categories.items():
        print(f"  {cat}: {len(qids)}")
    
    train_ids = []
    test_ids = []
    
    for cat, qids in categories.items():
        random.shuffle(qids)
        test_ids.extend(qids[:2])   # 2 из каждой категории в test
        train_ids.extend(qids[2:])  # остальные в train
    
    print(f"\nРаспределение:")
    print(f"  Обучающая выборка: {len(train_ids)} запросов")
    print(f"  Тестовая выборка: {len(test_ids)} запросов")
    
    train_data = []
    test_data = []
    
    for qid in train_ids:
        train_data.extend(query_groups[qid])
    
    for qid in test_ids:
        test_data.extend(query_groups[qid])
    
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(TEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nСохранено:")
    print(f"  {TRAIN_FILE}: {len(train_data)} пар")
    print(f"  {TEST_FILE}: {len(test_data)} пар")
    
    print(f"\nРаспределение меток в обучающей выборке:")
    labels = {1: 0, 2: 0, 3: 0}
    for item in train_data:
        labels[item["relevance_label"]] += 1
    print(f"  3 (высокая): {labels[3]}, 2 (средняя): {labels[2]}, 1 (низкая): {labels[1]}")
    
    print(f"\nРаспределение меток в тестовой выборке:")
    labels = {1: 0, 2: 0, 3: 0}
    for item in test_data:
        labels[item["relevance_label"]] += 1
    print(f"  3 (высокая): {labels[3]}, 2 (средняя): {labels[2]}, 1 (низкая): {labels[1]}")


if __name__ == "__main__":
    split_dataset()
