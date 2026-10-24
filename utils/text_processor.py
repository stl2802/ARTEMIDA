import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functools import lru_cache
from typing import List, Optional, Dict, Tuple
import time
import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # Если .env не найден, загрузить из .env.example
    env_example_path = Path('.') / '.env.example'
    if env_example_path.exists():
        load_dotenv(dotenv_path=env_example_path)

ENGLISH_PATTERN = r'^[a-z]+$'
RUSSIAN_PATTERN = r'^[а-яё]+$'
MIXED_PATTERN = r'^[a-zа-яё]+$'

# Кэш для морфологического анализа
MORPH_CACHE_SIZE = int(os.getenv('MORPH_CACHE_SIZE', '10000'))

class TextProcessor:
    """
    Класс для обработки текста с поддержкой мультиязычности.
    Оптимизированная версия с кэшированием результатов морфологического анализа.
    """
    
    def __init__(self):
        self._stemmer_en = None
        self._stemmer_ru = None
        self._morph_analyzer = None
        self._stop_words_en = None
        self._stop_words_ru = None
        self._nltk_downloaded = False
        self._morph_cache: Dict[str, Optional[str]] = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        self._morph_available = False
        self._default_language = os.getenv('DEFAULT_LANGUAGE', 'auto').lower()
        self._use_stemming = os.getenv('USE_STEMMING', 'true').lower() == 'true'
        self._remove_stopwords = os.getenv('REMOVE_STOPWORDS', 'true').lower() == 'true'
        self._min_length = int(os.getenv('MIN_TOKEN_LENGTH', '2'))
        
        try:
            import pymorphy3
            self._morph_analyzer = pymorphy3.MorphAnalyzer()
            self._morph_available = True
        except Exception as e:
            # fallback
            self._morph_available = False
            print("Предупреждение: pymorphy3 недоступен. Используется простой стемминг для русского языка.")
            print(f"Детали: {e}")
            print("Для полной функциональности установите: pip install pymorphy3[fast]")
    
    def _ensure_nltk_downloaded(self):
        """Убедиться, что необходимые ресурсы NLTK загружены"""
        if not self._nltk_downloaded:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            self._nltk_downloaded = True
    
    @property
    def stemmer_en(self):
        """Ленивая инициализация английского стеммера"""
        if self._stemmer_en is None:
            self._stemmer_en = PorterStemmer()
        return self._stemmer_en
    
    @property
    def stemmer_ru(self):
        """Ленивая инициализация русского стеммера"""
        if self._stemmer_ru is None:
            from snowballstemmer import RussianStemmer
            self._stemmer_ru = RussianStemmer()
        return self._stemmer_ru
    
    @property
    def morph_analyzer(self):
        """Ленивая инициализация морфологического анализатора"""
        if self._morph_analyzer is None and self._morph_available:
            try:
                import pymorphy3
                self._morph_analyzer = pymorphy3.MorphAnalyzer()
            except Exception:
                self._morph_available = False
        return self._morph_analyzer
    
    @property
    def stop_words_en(self):
        """Ленивая загрузка английских стоп-слов"""
        if self._stop_words_en is None:
            self._ensure_nltk_downloaded()
            self._stop_words_en = set(stopwords.words('english'))
        return self._stop_words_en
    
    @property
    def stop_words_ru(self):
        """Ленивая загрузка русских стоп-слов"""
        if self._stop_words_ru is None:
            self._ensure_nltk_downloaded()
            self._stop_words_ru = set(stopwords.words('russian'))
        return self._stop_words_ru
    
    def detect_language_word(self, word: str) -> str:
        """
        Определяет язык слова на основе алфавита.
        
        Args:
            word: Слово для анализа (уже в нижнем регистре)
        
        Returns:
            'en' - английское слово
            'ru' - русское слово
            'other' - другой язык или смешанный
        """
        if re.match(ENGLISH_PATTERN, word):
            return 'en'
        elif re.match(RUSSIAN_PATTERN, word):
            return 'ru'
        else:
            return 'other'
    
    def tokenize(self, text: str) -> List[str]:
        """
        Базовый токенизатор текста.
        
        Args:
            text: Исходный текст
        
        Returns:
            Список токенов в нижнем регистре
        """
        self._ensure_nltk_downloaded()
        tokens = word_tokenize(text.lower())
        return tokens
    
    def lemmatize_token(self, token: str, language: str) -> str:
        """
        Оптимизированная лемматизация с кэшированием результатов.
        Значительно ускоряет обработку повторяющихся слов.
        
        Args:
            token: Слово для лемматизации
            language: Язык ('en' или 'ru')
        
        Returns:
            Лемматизированная форма слова
        """
        # Создаем ключ кэша с учетом языка
        cache_key = f"{language}:{token}"
        
        # Проверяем кэш
        if cache_key in self._morph_cache:
            self._cache_stats['hits'] += 1
            result = self._morph_cache[cache_key]
            return result if result is not None else token
        
        self._cache_stats['misses'] += 1
        
        # Применяем лемматизацию
        if language == 'en':
            result = self.stemmer_en.stem(token)
        elif language == 'ru':
            # Проверяем доступность pymorphy3
            if self._morph_available and self._morph_analyzer:
                try:
                    parsed = self._morph_analyzer.parse(token)
                    if parsed:
                        result = parsed[0].normal_form
                    else:
                        result = token
                except Exception:
                    # Fallback на стемминг если pymorphy3 не работает
                    result = self.stemmer_ru.stemWord(token)
            else:
                # Fallback на стемминг
                result = self.stemmer_ru.stemWord(token)
        else:
            result = token
        
        # Сохраняем в кэш
        if len(self._morph_cache) >= MORPH_CACHE_SIZE:
            # Очищаем 10% кэша при переполнении
            keys_to_remove = list(self._morph_cache.keys())[:1000]
            for key in keys_to_remove:
                del self._morph_cache[key]
        
        self._morph_cache[cache_key] = result
        return result
    
    def clear_morph_cache(self):
        """Очистка кэша морфологического анализа (для освобождения памяти)"""
        self._morph_cache.clear()
        self._cache_stats = {'hits': 0, 'misses': 0}
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Получение статистики кэширования"""
        total = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = self._cache_stats['hits'] / total * 100 if total > 0 else 0
        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': round(hit_rate, 2),
            'cache_size': len(self._morph_cache)
        }
    
    def preprocess_text(self, text: str, language: str = None,
                        use_stemming: bool = None,
                        remove_stopwords: bool = None,
                        min_length: int = None) -> List[str]:
        """
        Основная функция препроцессинга текста с оптимизированной лемматизацией.
        
        Args:
            text: Исходный текст
            language: 
                - 'en': только английские слова
                - 'ru': только русские слова  
                - 'multilingual': оба языка
                - 'auto': автоматическое определение языка для каждого слова
                - None: использовать значение из .env файла
            use_stemming: Применять стемминг/лемматизацию
            remove_stopwords: Удалять стоп-слова
            min_length: Минимальная длина слова
        
        Returns:
            Список обработанных токенов
        """
        # Используем значения из .env если не указаны в аргументах
        lang = language if language is not None else self._default_language
        stem = use_stemming if use_stemming is not None else self._use_stemming
        stop = remove_stopwords if remove_stopwords is not None else self._remove_stopwords
        min_len = min_length if min_length is not None else self._min_length
        
        tokens = self.tokenize(text)
        processed_tokens = []
        
        for token in tokens:
            if len(token) < min_len:
                continue
                
            if lang == 'en':
                lang_type = 'en'
            elif lang == 'ru':
                lang_type = 'ru'
            elif lang in ['multilingual', 'auto']:
                lang_type = self.detect_language_word(token)
            else:
                raise ValueError(f"Неподдерживаемый язык: {lang}")

            if stop:
                if lang_type == 'en' and token in self.stop_words_en:
                    continue
                elif lang_type == 'ru' and token in self.stop_words_ru:
                    continue
            
            if stem:
                processed_token = self.lemmatize_token(token, lang_type)
            else:
                processed_token = token
            
            processed_tokens.append(processed_token)
        
        return processed_tokens
    
    def preprocess_text_to_string(self, text: str, language: str = None, **kwargs) -> str:
        """
        Версия препроцессинга, возвращающая строку.
        
        Args:
            text: Исходный текст
            language: Язык обработки (см. preprocess_text)
            **kwargs: Дополнительные аргументы для preprocess_text
        
        Returns:
            Обработанный текст в виде строки
        """
        tokens = self.preprocess_text(text, language=language, **kwargs)
        return ' '.join(tokens)
    
    def preprocess_query(self, query: str, language: str = None, **kwargs) -> List[str]:
        """
        Специальная версия для обработки поисковых запросов.
        
        Args:
            query: Поисковый запрос
            language: Язык обработки
            
        Returns:
            Список обработанных токенов
        """
        return self.preprocess_text(query, language=language, **kwargs)

processor = TextProcessor()

def preprocess_text(text: str, language: str = None) -> List[str]:
    """API-совместимая функция (возвращает список токенов)"""
    return processor.preprocess_text(text, language=language)

def tokenize(text: str) -> List[str]:
    """Настоящая базовая токенизация: без стоп-слов и стемминга."""
    return processor.tokenize(text)

def preprocess_text_to_string(text: str, language: str = None) -> str:
    """API-совместимая функция (возвращает строку)"""
    return processor.preprocess_text_to_string(text, language=language)

def get_stemmer(language: str = 'en'):
    """Получить стеммер для указанного языка"""
    if language == 'en':
        return processor.stemmer_en
    elif language == 'ru':
        return processor.stemmer_ru
    else:
        raise ValueError(f"Неподдерживаемый язык: {language}")

def clear_morph_cache():
    """Очистка кэша морфологического анализа"""
    processor.clear_morph_cache()

def get_cache_stats() -> Dict[str, int]:
    """Получение статистики кэширования"""
    return processor.get_cache_stats()

DEFAULT_LANGUAGE = 'multilingual'
SUPPORTED_LANGUAGES = ['en', 'ru', 'multilingual', 'auto']

# что другие модели могут импортироватть, вот это ялогику пидумал капец
__all__ = [
    'TextProcessor',
    'processor',
    'preprocess_text',
    'tokenize', 
    'preprocess_text_to_string',
    'get_stemmer',
    'clear_morph_cache',
    'get_cache_stats',
    'DEFAULT_LANGUAGE',
    'SUPPORTED_LANGUAGES'
]

if __name__ == "__main__":
    # Тест скорости и эффективности кэширования
    test_texts = [
        "Neural networks and machine learning",
        "Нейронные сети и машинное обучение",
        "Neural networks и нейронные сети",
        "Gradient boosting algorithm градиентный бустинг",
    ]
    
    print("=" * 70)
    print("ТЕСТ ОПТИМИЗИРОВАННОГО ТОКЕНИЗАТОРА С КЭШИРОВАНИЕМ")
    print("=" * 70)
    
    # Показываем текущие настройки
    print(f"\nТекущие настройки из .env файла:")
    print(f"  DEFAULT_LANGUAGE: {processor._default_language}")
    print(f"  USE_STEMMING: {processor._use_stemming}")
    print(f"  REMOVE_STOPWORDS: {processor._remove_stopwords}")
    print(f"  MIN_TOKEN_LENGTH: {processor._min_length}")
    print(f"  MORPH_CACHE_SIZE: {MORPH_CACHE_SIZE}")
    
    # Первый проход - заполнение кэша
    print("\n[1/3] Первый проход (заполнение кэша)...")
    start = time.time()
    for text in test_texts:
        tokens = processor.preprocess_text(text, 'multilingual')
        print(f"  '{text[:30]}...' -> {len(tokens)} токенов")
    elapsed1 = time.time() - start
    print(f"  Время: {elapsed1:.4f} сек")
    
    # Второй проход - использование кэша
    print("\n[2/3] Второй проход (использование кэша)...")
    start = time.time()
    for text in test_texts:
        tokens = processor.preprocess_text(text, 'multilingual')
        print(f"  '{text[:30]}...' -> {len(tokens)} токенов")
    elapsed2 = time.time() - start
    print(f"  Время: {elapsed2:.4f} сек")
    
    # Статистика кэша
    print("\n[3/3] Статистика кэширования:")
    stats = processor.get_cache_stats()
    print(f"  Попадания: {stats['hits']}")
    print(f"  Промахи: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']}%")
    print(f"  Размер кэша: {stats['cache_size']}")
    
    # Ускорение
    if elapsed2 > 0:
        speedup = elapsed1 / elapsed2
        print(f"\nУскорение благодаря кэшированию: {speedup:.2f}x")
    
    print("\nТесты мультиязычности:")
    for text in test_texts:
        print(f"\nИсходный текст: {text}")
        print(f"  Токены (мультиязычный): {processor.preprocess_text(text, 'multilingual')}")
        print(f"  Токены (английский): {processor.preprocess_text(text, 'en')}")
        print(f"  Токены (русский): {processor.preprocess_text(text, 'ru')}")
        print(f"  Строка (мультиязычный): {processor.preprocess_text_to_string(text, 'multilingual')}")
