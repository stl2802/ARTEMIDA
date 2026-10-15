import json
from collections import Counter

#TODO: пути
INPUT_FILE = "json/preprocessed_corpus.json"
OUTPUT_FILE = "json/vocabulary.json"

def build_vocabulary():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    all_tokens = []
    for doc in corpus:
        tokens = doc["text"].split()
        all_tokens.extend(tokens)
    
    token_counts = Counter(all_tokens)
    
    # Сортируем токены по алфавиту (для воспроизводимости)
    sorted_tokens = sorted(token_counts.keys())
    
    vocabulary = {token: idx for idx, token in enumerate(sorted_tokens)}
    
    data = {
        "vocabulary": vocabulary,
        "stats": {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(vocabulary),
            "total_documents": len(corpus),
            "avg_tokens_per_doc": len(all_tokens) / len(corpus)
        }
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Словарь терминов построен:")
    print(f"   Всего токенов в корпусе: {len(all_tokens):,}")
    print(f"   Уникальных токенов: {len(vocabulary):,}")
    print(f"   Документов: {len(corpus)}")
    print(f"   Среднее токенов на документ: {len(all_tokens) / len(corpus):.2f}")
    print(f"Словарь сохранен в '{OUTPUT_FILE}'")

if __name__ == "__main__":
    build_vocabulary()