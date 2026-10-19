import os
import re
import sys
import json
import math
from dataclasses import dataclass, field
from collections import Counter, defaultdict


## посмотреть может лучше импортировать токенезацию из процессинга
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Инициализация (один раз при старте)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
##

@dataclass
class ForwardEntry:
    doc_id: str
    title: str
    summary: str
    word_freq: Counter = field(default_factory=Counter)
    doc_length: int = 0

    def add(self, word, freq):
        self.word_freq[word] += freq

    def top_words(self, n=10):
        return self.word_freq.most_common(n)


@dataclass
class InvertedEntry:
    term: str
    postings: Counter = field(default_factory=Counter)#

    def add(self, doc_id, freq):
        self.postings[doc_id] += freq

    def top_documents(self, n=5):
        return self.postings.most_common(n)

    def doc_freq(self):
        return len(self.postings)


def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def calculate_bm25(query_tokens, doc_id, forward_index, inverted_index, avg_doc_length, k1=1.5, b=0.75):
    if doc_id not in forward_index:
        return 0.0

    entry = forward_index[doc_id]
    doc_length = entry.doc_length
    N = len(forward_index)
    score = 0.0

    for token in query_tokens:
        if token not in inverted_index:
            continue

        df = inverted_index[token].doc_freq()
        # IDF по формуле BM25
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        tf = entry.word_freq.get(token, 0)
        if tf == 0:
            continue

        # Нормализация TF с учетом длины документа
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))

        score += idf * (numerator / denominator)

    return score


def build_mode(corpus_path, output_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    forward_index = {}
    inverted_index = {}
    total_length = 0

    for doc in corpus:
        doc_id = doc["id"]
        tokens = doc["text"].split()
        unique_words = set(tokens)

        forward_index[doc_id] = ForwardEntry(
            doc_id=doc_id,
            title=doc["title"],
            summary=doc["summary"],
            doc_length=len(tokens)
        )
        total_length += len(tokens)

        word_counts = Counter(tokens)
        for word in unique_words:
            freq = word_counts[word]
            #пояснение для меня тут метод встроеный в класс  а именно .add ну типо у нас сначало возвращается обьект классса поэтому можно записать это как entry = forward_index["doc_0000"] entry.add(word, freq)
            forward_index[doc_id].add(word, freq)

            if word not in inverted_index:
                inverted_index[word] = InvertedEntry(term=word)
            inverted_index[word].add(doc_id, freq)

    avg_doc_length = total_length / len(forward_index) if forward_index else 0

    # Сохранение индекса
    # тоже прикол, короче тут я юзах хитрую штуку котарая генерирует прямые индексы 1 строкой в иманно for doc_id, entry in forward_index.items()
    data = {
        "forward_index": {
            doc_id: {
                "title": entry.title,
                "summary": entry.summary,
                #dict для нормальной работы сериализатора епт
                "word_freq": dict(entry.word_freq),
                "doc_length": entry.doc_length
            }
            for doc_id, entry in forward_index.items()
        },
        "inverted_index": {
            #dict для нормальной работы сериализатора епт
            term: dict(entry.postings)
            for term, entry in inverted_index.items()
        },
        "avg_doc_length": avg_doc_length,
        "doc_count": len(forward_index)
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Индекс построен: {len(inverted_index)} уникальных токенов, {len(forward_index)} документов")
    print(f"Средняя длина документа: {avg_doc_length:.2f} токенов")
    print(f"Индекс сохранен в '{output_path}'")


def load_index(index_path):
    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    forward_index = {}
    for doc_id, entry_data in data["forward_index"].items():
        forward_index[doc_id] = ForwardEntry(
            doc_id=doc_id,
            title=entry_data["title"],
            summary=entry_data["summary"],
            word_freq=Counter(entry_data["word_freq"]),
            doc_length=entry_data["doc_length"]
        )

    inverted_index = {}
    for term, postings in data["inverted_index"].items():
        inverted_index[term] = InvertedEntry(
            term=term,
            postings=Counter(postings)
        )

    return forward_index, inverted_index, data["avg_doc_length"], data["doc_count"]

#TODO: тоже кросбплатформенность путей
def search_mode(index_path, queries_path='generated_data/json/queries.json'):
    forward_index, inverted_index, avg_doc_length, doc_count = load_index(index_path)
    print(f"Индекс загружен: {doc_count} документов, средняя длина {avg_doc_length:.2f}\n")

    queries = []
    if queries_path and os.path.exists(queries_path):
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
            queries = (
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["exact_terms"][:3]] +
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["detailed_queries"][:2]] +
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["short_queries"][:3]] +
                [q["en"] if isinstance(q, dict) else q for q in queries_data["queries"]["error_queries"][:2]]
            )
        print(f"Загружено {len(queries)} тестовых запросов из '{queries_path}'\n")

    while True:
        if queries:
            query = queries.pop(0)
            print(f"Запрос: '{query}'")
        else:
            query = input("Введите поисковый запрос (или 'exit' для выхода): ")
            if query.lower() == 'exit':
                break

        query_tokens = tokenize(query)
        if not query_tokens:
            print("После предобработки запрос пуст\n")
            continue

        # расчет BM25
        scores = []
        for doc_id in forward_index:
            score = calculate_bm25(query_tokens, doc_id, forward_index, inverted_index, avg_doc_length)
            if score > 0:
                scores.append((doc_id, score))

        # сортировка по убыванию BM25
        scores.sort(key=lambda x: x[1], reverse=True)
        top_5 = scores[:5]

        if not top_5:
            print("Документов не найдено\n")
        else:
            print(f"Топ-5 результатов (BM25):")
            for rank, (doc_id, score) in enumerate(top_5, 1):
                entry = forward_index[doc_id]
                print(f"{rank}. [{doc_id}] (score: {score:.4f})")
                print(f"   Title: {entry.title}")
                print(f"   Summary: {entry.summary[:150]}...")
                print()


def main():
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python search_engine.py build   - построение индекса")
        print("  python search_engine.py search  - режим поиска")
        return

    mode = sys.argv[1].lower()

    if mode == "build":
        #TODO: кросплафторменность путей
        build_mode("generated_data/json/preprocessed_corpus.json", "generated_data/json/inverted_index.json")
    elif mode == "search":
        search_mode("generated_data/json/inverted_index.json", "generated_data/json/queries.json")
    else:
        print(f"Неизвестный режим: {mode}")


if __name__ == "__main__":
    main()