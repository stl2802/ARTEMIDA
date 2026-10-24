#генератор тестовых моих поисковых запросов для тестового кластера данных

import json
import os
from pathlib import Path

OUTPUT_FILE: Path = Path("generated_data") / "json" / "queries.json"

QUERIES_DATA: dict = {
    "domain": "Machine Learning and Natural Language Processing",
    "queries": {
        "exact_terms": [
            {"en": "Neural Networks", "ru": "Нейронные сети"},
            {"en": "Backpropagation", "ru": "Обратное распространение ошибки"},
            {"en": "Transformer architecture", "ru": "Архитектура трансформера"},
            {"en": "Gradient descent", "ru": "Градиентный спуск"},
            {"en": "Natural Language Processing", "ru": "Обработка естественного языка"},
            {"en": "Overfitting", "ru": "Переобучение"},
            {"en": "Convolutional Neural Network", "ru": "Сверточная нейронная сеть"},
            {"en": "Reinforcement Learning", "ru": "Обучение с подкреплением"},
            {"en": "Stochastic optimization", "ru": "Стохастическая оптимизация"},
            {"en": "Attention mechanism", "ru": "Механизм внимания"},
        ],
        "detailed_queries": [
            {"en": "How does the transformer architecture improve natural language processing tasks significantly",
             "ru": "Как архитектура трансформера значительно улучшает задачи обработки естественного языка"},
            {"en": "What are the main causes of overfitting in deep neural networks training",
             "ru": "Каковы основные причины переобучения при обучении глубоких нейронных сетей"},
            {"en": "Compare the performance of stochastic gradient descent and Adam optimizer algorithms",
             "ru": "Сравните производительность алгоритмов стохастического градиентного спуска и оптимизатора Adam"},
            {"en": "Applications of reinforcement learning in autonomous driving systems and modern robotics",
             "ru": "Применение обучения с подкреплением в системах автономного вождения и современной робототехнике"},
            {"en": "How to prevent vanishing gradient problem in recurrent neural networks effectively",
             "ru": "Как эффективно предотвратить проблему затухающего градиента в рекуррентных нейронных сетях"},
            {"en": "The impact of attention mechanisms on machine translation accuracy and speed",
             "ru": "Влияние механизмов внимания на точность и скорость машинного перевода"},
            {"en": "Techniques for reducing computational complexity in large language models deployment",
             "ru": "Методы снижения вычислительной сложности при развертывании больших языковых моделей"},
            {"en": "Evaluating the effectiveness of convolutional neural networks in image classification tasks",
             "ru": "Оценка эффективности сверточных нейронных сетей в задачах классификации изображений"},
            {"en": "Methods for handling imbalanced datasets in supervised machine learning algorithms properly",
             "ru": "Методы правильной обработки несбалансированных наборов данных в алгоритмах обучения с учителем"},
            {"en": "The role of dropout regularization in improving generalization of deep learning models",
             "ru": "Роль регуляризации dropout в улучшении обобщающей способности моделей глубокого обучения"},
        ],
        "short_queries": [
            {"en": "Deep learning optimization techniques", "ru": "Методы оптимизации глубокого обучения"},
            {"en": "Transformer model architecture", "ru": "Архитектура модели трансформера"},
            {"en": "NLP text classification", "ru": "Классификация текста NLP"},
            {"en": "Prevent neural network overfitting", "ru": "Предотвращение переобучения нейронной сети"},
            {"en": "Reinforcement learning basics", "ru": "Основы обучения с подкреплением"},
            {"en": "Gradient descent algorithm", "ru": "Алгоритм градиентного спуска"},
            {"en": "Attention mechanism explained", "ru": "Объяснение механизма внимания"},
            {"en": "CNN image recognition", "ru": "Распознавание изображений CNN"},
            {"en": "Machine learning datasets", "ru": "Наборы данных машинного обучения"},
            {"en": "Backpropagation in neural networks", "ru": "Обратное распространение в нейронных сетях"},
        ],
        "error_queries": [
            {"en": "how to fix overfiting in nueral networks", "ru": "как исправить переобучение в нейронных сетях (с опечатками)"},
            {"en": "tranformer attention mecanism explained", "ru": "объяснение механизма внимания трансформера (с опечатками)"},
            {"en": "gradiant descent vs adam optimiser", "ru": "градиентный спуск vs оптимизатор adam (с опечатками)"},
            {"en": "convolutional neural netwrok for images", "ru": "сверточная нейронная сеть для изображений (с опечатками)"},
            {"en": "natural langage processing basics", "ru": "основы обработки естественного языка (с опечатками)"},
            {"en": "machine learning algorithms for predicting the past", "ru": "алгоритмы машинного обучения для предсказания прошлого (логическая ошибка)"},
            {"en": "how to make neural networks run without electricity", "ru": "как заставить нейронные сети работать без электричества (абсурдный запрос)"},
            {"en": "best database for storing human thoughts directly", "ru": "лучшая база данных для прямого хранения человеческих мыслей (абсурдный запрос)"},
            {"en": "using sql queries to train a convolutional neural network", "ru": "использование SQL-запросов для обучения сверточной нейронной сети (логическая ошибка)"},
            {"en": "why python is slower than manual calculation in ai", "ru": "почему python медленнее ручных вычислений в ИИ (абсурдный запрос)"},
        ],
    },
}


def generate_queries(output_file: str = str(OUTPUT_FILE)) -> None:
    """Создаёт файл queries.json"""
    output_dir = Path(output_file).parent
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(QUERIES_DATA, f, ensure_ascii=False, indent=4)

    total = sum(len(v) for v in QUERIES_DATA["queries"].values())
    print(f"Сгенерировано {total} запросов в '{output_file}'")
    for category, items in QUERIES_DATA["queries"].items():
        print(f"  - {category}: {len(items)}")


if __name__ == "__main__":
    generate_queries()