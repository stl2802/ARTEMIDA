import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import pymorphy3
from typing import List

# Глобальные константы для паттернов
ENGLISH_PATTERN = r'^[a-z]+$'
RUSSIAN_PATTERN = r'^[а-яё]+$'
MIXED_PATTERN = r'^[a-zа-яё]+$'

class TextProcessor:
    """
    Класс для обработки текста с поддержкой мультиязычности.
    """
    
    def __init__(self):
        self._stemmer_en = None
        self._stemmer_ru = None
        self._morph_analyzer = None
        self._stop_words_en = None
        self._stop_words_ru = None
        self._nltk_downloaded = False
        
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
        if self._morph_analyzer is None:
            self._morph_analyzer = pymorphy3.MorphAnalyzer()
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
    
    def preprocess_text(self, text: str, language: str = 'multilingual',use_stemming: bool = True, \
        remove_stopwords: bool = True,min_length: int = 2) -> List[str]:
        """
        Основная функция препроцессинга текста.
        
        Args:
            text: Исходный текст
            language: 
                - 'en': только английские слова
                - 'ru': только русские слова  
                - 'multilingual': оба языка
                - 'auto': автоматическое определение языка для каждого слова
            use_stemming: Применять стемминг/лемматизацию
            remove_stopwords: Удалять стоп-слова
            min_length: Минимальная длина слова
        
        Returns:
            Список обработанных токенов
        """
        tokens = self.tokenize(text)
        processed_tokens = []
        
        for token in tokens:
            if len(token) < min_length:
                continue
                
            if language == 'en':
                lang = 'en'
            elif language == 'ru':
                lang = 'ru'
            elif language == 'multilingual':
                lang = self.detect_language_word(token)
            elif language == 'auto':
                lang = self.detect_language_word(token)
            else:
                raise ValueError(f"Неподдерживаемый язык: {language}")

            if remove_stopwords:
                if lang == 'en' and token in self.stop_words_en:
                    continue
                elif lang == 'ru' and token in self.stop_words_ru:
                    continue
            
            if use_stemming:
                if lang == 'en':
                    processed_token = self.stemmer_en.stem(token)
                elif lang == 'ru':
                    parsed_word = self.morph_analyzer.parse(token)[0]
                    processed_token = parsed_word.normal_form
                else:
                    processed_token = token
            else:
                processed_token = token
            
            processed_tokens.append(processed_token)
        
        return processed_tokens
    
    def preprocess_text_to_string(self, text: str, language: str = 'multilingual',**kwargs) -> str:
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
    
    def preprocess_query(self,query: str,language: str = 'multilingual',**kwargs) -> List[str]:
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

def preprocess_text(text: str, language: str = 'multilingual') -> List[str]:
    """API-совместимая функция (возвращает список токенов)"""
    return processor.preprocess_text(text, language)

def tokenize(text: str) -> List[str]:
    """Настоящая базовая токенизация: без стоп-слов и стемминга."""
    return processor.tokenize(text)

def preprocess_text_to_string(text: str, language: str = 'multilingual') -> str:
    """API-совместимая функция (возвращает строку)"""
    return processor.preprocess_text_to_string(text, language)

def get_stemmer(language: str = 'en'):
    """Получить стеммер для указанного языка"""
    if language == 'en':
        return processor.stemmer_en
    elif language == 'ru':
        return processor.stemmer_ru
    else:
        raise ValueError(f"Неподдерживаемый язык: {language}")

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
    'DEFAULT_LANGUAGE',
    'SUPPORTED_LANGUAGES'
]

if __name__ == "__main__":
    #TODO: импорт из json/queries.json
    test_texts = [
        "Neural networks and machine learning",
        "Нейронные сети и машинное обучение",
        "Neural networks и нейронные сети",
        "Gradient boosting algorithm градиентный бустинг",
    ]
    
    for text in test_texts:
        print(f"\nИсходный текст: {text}")
        print(f"Токены (мультиязычный): {processor.preprocess_text(text, 'multilingual')}")
        print(f"Токены (английский): {processor.preprocess_text(text, 'en')}")
        print(f"Токены (русский): {processor.preprocess_text(text, 'ru')}")
        print(f"Строка (мультиязычный): {processor.preprocess_text_to_string(text, 'multilingual')}")