import json
from utils.text_processor import preprocess_text_to_string

#TODO: пути кросплатформенность
INPUT_FILE:str = "generated_data/json/corpus.json"
OUTPUT_FILE:str = "generated_data/json/preprocessed_corpus.json"

def preprocess_text(text: str) -> str:
    """Обработка текста с использованием мультиязычного процессора"""
    return preprocess_text_to_string(text, language='multilingual')

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    preprocessed_corpus:list = []

    for doc in corpus:
        preprocessed_doc = {
            "id": doc["id"],
            "title": doc["title"],
            "summary": preprocess_text(doc["summary"]),
            "text": preprocess_text(doc["text"])
        }
        preprocessed_corpus.append(preprocessed_doc)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_corpus, f, ensure_ascii=False, indent=2)

    print(f"Обработано {len(preprocessed_corpus)} документов.")
    print(f"Предобработанный корпус сохранен в '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
