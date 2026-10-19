import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#необходимые данные NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

#TODO: пути кросплатформенность
INPUT_FILE:str = "generated_data/json/corpus.json"
OUTPUT_FILE:str = "generated_data/json/preprocessed_corpus.json"

def preprocess_text(text: str) -> str:
    text:str = text.lower()
    tokens = word_tokenize(text)
    #re на только слова без цифр
    tokens = [token for token in tokens if re.match(r'^[a-z]+$', token)]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    #Нормализация (стемминг) #TODO: посмотреть как на русском нормализовывать
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

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