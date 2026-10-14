import json
import re
from pathlib import Path

FOLDER_NAME = "literature"
OUTPUT_FILE = "corpus.json"

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_document(file_path: Path, doc_id: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    parts = raw_content.split("ABSTRACT:", 1)
    
    if len(parts) < 2:
        raise ValueError(f"Битый файл: не найден маркер 'ABSTRACT:' в {file_path}")
    
    title_raw = parts[0].replace("TITLE:", "").strip()
    abstract_raw = parts[1].strip()
    
    title = clean_text(title_raw)
    summary = clean_text(abstract_raw)
    full_text = clean_text(raw_content)
    
    return {
        "id": doc_id,
        "title": title,
        "summary": summary,
        "text": full_text
    }

def main():
    folder:Path = Path(FOLDER_NAME)
    if not folder.exists():
        print(f"Папка '{FOLDER_NAME}' не найдена. Сначала выполните literature_installer.py")
        return

    corpus: dict = []
    #TODO: сделать альтернативный путь чтоб не забивать память при обработке более 2к документов
    txt_files = sorted(folder.glob("doc_*.txt"))
    
    for file_path in txt_files:
        doc_id = file_path.stem
        try:
            doc_data = parse_document(file_path, doc_id)
            corpus.append(doc_data)
        except ValueError as e:
            print(f"{e} — файл пропущен")
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print(f"Обработано {len(corpus)} документов.")
    print(f"Структурированный корпус сохранен в '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()