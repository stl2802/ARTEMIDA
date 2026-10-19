ARTEMIDA - гибридная поисковая система

В качестве текста документа использована аннотация (abstract) научной статьи, так как полные тексты статей на arXiv распространяются по лицензии CC-BY и требуют дополнительной обработки PDF. Аннотация содержит ключевые термины и семантическое ядро работы, что достаточно для задач информационного поиска.

## Быстрый старт

### Полная инициализация проекта с нуля

Если у вас есть пустой проект (нет файлов json/, npy/, model/, hnsw_index.bin), выполните:

```bash
python main.py init
```

Эта команда автоматически выполнит все 10 шагов:

1. Обработка корпуса из literature/ → json/corpus.json
2. Предобработка корпуса → json/preprocessed_corpus.json
3. Построение словаря → json/vocabulary.json
4. Лексический индекс (BM25) → json/inverted_index.json
5. TF-IDF векторы + SVD → npy/doc_vectors_svd.npy, npy/svd_vt.npy
6. HNSW индекс → hnsw_index.bin, json/doc_ids.json
7. Нейронные векторы (multilingual) → npy/neural_doc_vectors.npy, json/neural_doc_ids.json
8. Подготовка LTR датасета → json/ltr_dataset.json
9. Разделение LTR датасета → json/ltr_train.json, json/ltr_test.json
10. Обучение LTR модели (LightGBM) → model/ltr_model.pkl, model/feature_order.json

После завершения все файлы будут сгенерированы и можно использовать все методы поиска.

## Команды CLI

### Основные команды

```bash
python main.py init                  # Полная инициализация проекта с нуля
python main.py index-all             # Построение всех индексов (требует json/corpus.json)
python main.py smart-hybrid          # Умный гибридный поиск (автовыбор метода)
```

### Построение индексов

```bash
python main.py corpus                # Обработка корпуса из literature/
python main.py preprocess            # Предобработка корпуса
python main.py vocabulary            # Построение словаря
python main.py lexical build         # Лексический индекс (BM25)
python main.py tfidf build           # TF-IDF векторы
python main.py tfidf svd -k 100      # Применение SVD
python main.py neural build          # Нейронные векторы
python main.py hnsw build            # HNSW индекс
python main.py ltr prepare           # Подготовка LTR датасета
python main.py ltr train             # Обучение LTR модели
```

### Поиск

```bash
python main.py lexical search        # Лексический поиск (BM25)
python main.py tfidf search          # TF-IDF поиск
python main.py neural search         # Нейронный поиск (multilingual)
python main.py hnsw search           # HNSW поиск
python main.py hybrid search         # Гибридный поиск (BM25 + HNSW)
python main.py hybrid search -a 0.7  # Гибридный поиск с alpha=0.7
python main.py smart-hybrid          # Умный гибридный поиск (автовыбор)
python main.py ltr search            # LTR поиск (LightGBM)
```

### Бенчмарки

```bash
python main.py benchmark alpha       # Бенчмарк влияния параметра alpha
python main.py benchmark hnsw        # Бенчмарк HNSW vs точный поиск
python main.py benchmark ltr         # Бенчмарк LTR vs гибридный поиск
```

## Умный гибридный поиск

Команда `smart-hybrid` автоматически выбирает оптимальный метод поиска на основе анализа запроса:

- **Русский текст** → Neural (мультиязычная модель paraphrase-multilingual-MiniLM-L12-v2)
- **Короткий термин (1-3 слова)** → Lexical (BM25)
- **Смысловой запрос** → LTR (LightGBM) или Hybrid (BM25 + HNSW)
- **Принудительный выбор** → neural/lexical/hybrid/ltr/hnsw

Примеры:
```bash
python main.py smart-hybrid
# В интерактивном режиме:
# - Введите запрос
# - Метод [Enter=auto, neural/lexical/hybrid/ltr/hnsw]
```

## Структура проекта

```
.
├── literature/              # Исходные документы (doc_*.txt)
├── json/                   # JSON файлы индексов и данных
│   ├── corpus.json
│   ├── preprocessed_corpus.json
│   ├── inverted_index.json
│   ├── vocabulary.json
│   ├── doc_ids.json
│   ├── neural_doc_ids.json
│   ├── queries.json
│   ├── ltr_dataset.json
│   ├── ltr_train.json
│   └── ltr_test.json
├── npy/                    # NumPy массивы векторов
│   ├── doc_vectors_svd.npy
│   ├── svd_vt.npy
│   └── neural_doc_vectors.npy
├── model/                  # ML модели
│   ├── ltr_model.pkl
│   └── feature_order.json
├── hnsw_index.bin          # HNSW индекс
└── main.py                 # CLI интерфейс
```

## Требования

Установите зависимости:

```bash
pip install -r requirements.txt
```

Основные зависимости:
- nltk
- sentence-transformers
- hnswlib
- lightgbm
- datasets
- matplotlib
