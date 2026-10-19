# Инструкция по запуску проекта ARTEMIDA

Полное руководство по установке и запуску гибридной поисковой системы с нуля.

## Содержание

1. [Требования](#требования)
2. [Установка зависимостей](#установка-зависимостей)
3. [Подготовка исходных данных](#подготовка-исходных-данных)
4. [Полная инициализация проекта](#полная-инициализация-проекта)
5. [Использование поисковой системы](#использование-поисковой-системы)
6. [Тестирование всех методов поиска](#тестирование-всех-методов-поиска)

---

## Требования

### Системные требования

- Python 3.8 или выше
- Операционная система: Windows, Linux, macOS
- Минимум 4 GB RAM (рекомендуется 8 GB)
- Минимум 2 GB свободного места на диске

### Программные зависимости

Все зависимости перечислены в файле `requirements.txt`:

```
datasets
nltk
matplotlib
Sentence-Transformers
hnswlib
lightgbm
```

---

## Установка зависимостей

### Шаг 1: Создание виртуального окружения (рекомендуется)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Шаг 2: Установка пакетов

```bash
pip install -r requirements.txt
```

### Шаг 3: Загрузка NLTK данных

При первом запуске NLTK автоматически загрузит необходимые данные:
- `punkt` - токенизатор
- `stopwords` - стоп-слова

---

## Подготовка исходных данных
если нужно для обучение, можно просто сгенерировать тестовые данные при помощи команды create-data

### Структура директории `literature/`

Исходные документы должны находиться в директории `literature/` в формате `doc_XXXX.txt`, где XXXX - номер документа.

Пример структуры:
```
literature/
├── doc_0000.txt
├── doc_0001.txt
├── doc_0002.txt
└── ...
```

### Формат документа

Каждый файл должен содержать:
```
TITLE: Название статьи
ABSTRACT: Аннотация статьи...
```

### Получение данных

Если у вас нет данных, они должны быть предоставлены отдельно или находятся в директории `literature/`.

---

## Полная инициализация проекта

### Шаг 1: Проверка структуры проекта

Убедитесь, что директория `literature/` существует и содержит файлы документов:
или же необходимо сгенерировать тестовые данные при помощи команды create-data

```bash
# Windows
dir literature

# Linux/macOS
ls literature
```

### Шаг 2: Запуск полной инициализации

Выполните команду для автоматической генерации всех файлов:

```bash
python main.py init
```

### Что происходит во время инициализации

Команда `init` выполняет следующие шаги:
#### 0. Если необходимо сощздать тестовые данные(1 мин)
- Выгрузка данных из библиотеки datasets
- Сохранение их в файлы в папке literature/
#### 1. Обработка корпуса (1-2 минуты)
- Чтение файлов из `literature/`
- Парсинг TITLE и ABSTRACT
- Сохранение в `json/corpus.json`

#### 2. Предобработка корпуса (1-2 минуты)
- Токенизация
- Удаление стоп-слов
- Стемминг (Porter Stemmer)
- Сохранение в `json/preprocessed_corpus.json`

#### 3. Построение словаря (< 1 минута)
- Создание словаря из предобработанного корпуса
- Сохранение в `json/vocabulary.json`

#### 4. Лексический индекс BM25 (< 1 минута)
- Построение инвертированного индекса
- Вычисление BM25 параметров
- Сохранение в `json/inverted_index.json`

#### 5. TF-IDF векторы + SVD (1-2 минуты)
- Вычисление TF-IDF матрицы
- Применение SVD для снижения размерности (по умолчанию k=100)
- Сохранение в `npy/doc_vectors_svd.npy` и `npy/svd_vt.npy`

#### 6. HNSW индекс (1-2 минуты)
- Построение HNSW индекса на SVD векторах
- Параметры: M=16, ef_construction=200, ef_search=50
- Сохранение в `hnsw_index.bin` и `json/doc_ids.json`

#### 7. Нейронные векторы (3-5 минут)
- Загрузка модели `paraphrase-multilingual-MiniLM-L12-v2` (первый раз)
- Векторизация корпуса
- Сохранение в `npy/neural_doc_vectors.npy` и `json/neural_doc_ids.json`

#### 8. Подготовка LTR датасета (2-3 минуты)
- Извлечение признаков для всех пар (запрос, документ)
- Присвоение меток релевантности (1-3)
- Сохранение в `json/ltr_dataset.json`

#### 9. Разделение LTR датасета (< 1 минута)
- Разделение на train/test (80/20)
- Сохранение в `json/ltr_train.json` и `json/ltr_test.json`

#### 10. Обучение LTR модели (1-2 минуты)
- Обучение LightGBM (Gradient Boosting)
- Параметры: n_estimators=100, max_depth=5, learning_rate=0.1
- Сохранение в `model/ltr_model.pkl` и `model/feature_order.json`

**Общее время: ~15-20 минут**

### Проверка успешной инициализации

После завершения проверьте наличие всех файлов:

```bash
# Windows
dir json
dir npy
dir model
dir bin/hnsw_index.bin

# Linux/macOS
ls json
ls npy
ls model
ls bin/hnsw_index.bin
```

Ожидаемые файлы:

**json/**
- corpus.json
- preprocessed_corpus.json
- inverted_index.json
- vocabulary.json
- doc_ids.json
- neural_doc_ids.json
- queries.json
- ltr_dataset.json
- ltr_train.json
- ltr_test.json

**npy/**
- doc_vectors_svd.npy
- svd_vt.npy
- neural_doc_vectors.npy

**model/**
- ltr_model.pkl
- feature_order.json

**bin**
- hnsw_index.bin

---

## Использование поисковой системы

### Умный гибридный поиск (рекомендуется)

Команда автоматически выбирает оптимальный метод поиска:

```bash
python main.py smart-hybrid
```

**Логика выбора метода:**
- Русский текст → Neural (мультиязычная модель)
- Короткий термин (1-3 слова) → Lexical (BM25)
- Смысловой запрос → LTR (LightGBM) или Hybrid

**Примеры запросов:**

```
Введите поисковый запрос (или 'exit' для выхода): machine learning
Метод [Enter=auto, neural/lexical/hybrid/ltr/hnsw]: 

Анализ запроса: 'machine learning'
Выбранный метод: lexical (причина: short_term)
Топ-5 результатов:
...
```

```
Введите поисковый запрос (или 'exit' для выхода): оптимизация гиперпараметров модели
Метод [Enter=auto, neural/lexical/hybrid/ltr/hnsw]: 

Анализ запроса: 'оптимизация гиперпараметров модели'
Выбранный метод: neural (причина: russian_language)
Топ-5 результатов:
...
```

```
Введите поисковый запрос (или 'exit' для выхода): How does deep learning improve image recognition
Метод [Enter=auto, neural/lexical/hybrid/ltr/hnsw]: 

Анализ запроса: 'How does deep learning improve image recognition'
Выбранный метод: ltr (причина: conceptual_query_with_ltr)
Топ-5 результатов:
...
```

### Принудительный выбор метода

В интерактивном режиме можно указать метод:

```
Метод [Enter=auto, neural/lexical/hybrid/ltr/hnsw]: ltr
```

### Другие методы поиска

#### Лексический поиск (BM25)
```bash
python main.py lexical search
```

#### Нейронный поиск (multilingual)
```bash
python main.py neural search
```

#### HNSW поиск (приближенный векторный)
```bash
python main.py hnsw search
```

#### Гибридный поиск (BM25 + HNSW)
```bash
python main.py hybrid search
python main.py hybrid search -a 0.7  # с alpha=0.7
```

#### LTR поиск (LightGBM)
```bash
python main.py ltr search
```

---

## Тестирование всех методов поиска

### Автоматическое тестирование

Для тестирования всех методов используйте скрипт `test_smart_hybrid.py`:

```bash
python test_smart_hybrid.py
```

Этот скрипт автоматически протестирует:

1. **Lexical** - короткие термины
2. **Neural** - русские запросы
3. **LTR** - смысловые запросы
4. **HNSW** - принудительное использование
5. **Hybrid** - fallback

Пример вывода:

```
======================================================================
ДЕМОНСТРАЦИЯ УМНОГО ГИБРИДНОГО ПОИСКА
======================================================================

----------------------------------------------------------------------
Запрос: 'machine learning'
Принудительный метод: auto
Выбранный метод: lexical (причина: short_term)

Топ-3 результатов:

1. [doc_0266] (bm25: 4.8636)
   Title: Considerations upon the Machine Learning Technologies
   Summary: artifici intellig offer superior techniqu method problem...

----------------------------------------------------------------------
Запрос: 'оптимизация гиперпараметров модели'
Принудительный метод: auto
Выбранный метод: neural (причина: russian_language)

Топ-3 результатов:

1. [doc_0211] (cosine: 0.5320)
   Title: On the Dual Formulation of Boosting Algorithms
   Summary: studi boost algorithm new perspect show lagrang...
```

---

## Частые проблемы и решения

### Ошибка: "Папка 'literature' не найдена"

**Решение:** Убедитесь, что директория `literature/` существует в корне проекта и содержит файлы документов.

### Ошибка: "ModuleNotFoundError: No module named 'sentence_transformers'"

**Решение:** Установите зависимости:
```bash
pip install -r requirements.txt
```

### Ошибка: "FileNotFoundError: model/ltr_model.pkl"

**Решение:** Запустите полную инициализацию:
```bash
python main.py init
```

### Медленная работа нейронного поиска

**Причина:** Первая загрузка модели `paraphrase-multilingual-MiniLM-L12-v2` загружает модель из интернета (~400 MB).

**Решение:** Модель кэшируется локально, последующие запуски будут быстрее.

### UnicodeEncodeError (Windows)

**Решение:** Установите кодировку UTF-8:
```bash
set PYTHONIOENCODING=utf-8
python main.py smart-hybrid
```

---

## Дополнительные команды

### Построение отдельных индексов

Если нужно перестроить только определенный индекс:

```bash
python main.py corpus                # Обработка корпуса
python main.py preprocess            # Предобработка
python main.py vocabulary            # Словарь
python main.py lexical build         # BM25 индекс
python main.py tfidf build           # TF-IDF
python main.py tfidf svd -k 100      # SVD
python main.py neural build          # Нейронные векторы
python main.py hnsw build            # HNSW индекс
python main.py ltr prepare           # LTR датасет
python main.py ltr train             # LTR модель
```

### Бенчмарки

Сравнение производительности разных методов:

```bash
python main.py benchmark alpha       # Влияние параметра alpha
python main.py benchmark hnsw        # HNSW vs точный поиск
python main.py benchmark ltr         # LTR vs гибридный поиск
```

---

## Структура проекта после инициализации

```
.
├── literature/              # Исходные документы
│   ├── doc_0000.txt
│   ├── doc_0001.txt
│   └── ...
├── json/                   # JSON файлы
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
├── npy/                    # NumPy массивы
│   ├── doc_vectors_svd.npy
│   ├── svd_vt.npy
│   └── neural_doc_vectors.npy
├── model/                  # ML модели
│   ├── ltr_model.pkl
│   └── feature_order.json
├── hnsw_index.bin          # HNSW индекс
├── main.py                 # CLI интерфейс
├── smart_hybrid_search.py  # Умный гибридный поиск
├── test_smart_hybrid.py    # Тестовый скрипт
├── requirements.txt        # Зависимости
├── README.md               # Документация
└── SETUP.md                # Эта инструкция
├──benchmarks
   ├── benchmark_alpha.py
   ├── benchmark_hnsw.py
   ├── benchmark_ltr.py
└── feature_extractor.py
└── hybrid_search.py
└── lecsika_search.py
└── neural_search.py
└── prepare_ltr_dataset.py
└── preprocess_corpus.py
└── split_dataset.py
└── tfidf_search.py
└── train_ltr.py
```

---

## Следующие шаги

После успешной инициализации:

1. **Попробуйте умный поиск:**
   ```bash
   python main.py smart-hybrid
   ```

2. **Протестируйте все методы:**
   ```bash
   python test_smart_hybrid.py
   ```

3. **Запустите бенчмарки:**
   ```bash
   python main.py benchmark ltr
   ```

4. **Изучите документацию:**
   ```bash
   cat README.md
   ```

---

## Поддержка

При возникновении проблем:

1. Проверьте наличие всех файлов после инициализации
2. Убедитесь, что все зависимости установлены
3. Проверьте версию Python (требуется 3.8+)
4. Ознакомьтесь с разделом "Частые проблемы"

---

**Удачи в использовании ARTEMIDA!** 🚀
