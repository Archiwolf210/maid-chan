"""
Утилиты для проекта Digital Human (Maid).
Включает: парсинг JSON, фильтрацию стиля, валидацию.
"""
import re
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def extract_json_safely(text: str) -> Dict[str, Any]:
    """
    Извлекает JSON из текста ответа LLM.
    Локальные модели часто ломают формат — эта функция пытается их исправить.
    
    Args:
        text: Текст, содержащий JSON (возможно с ошибками)
        
    Returns:
        Словарь с данными
        
    Raises:
        ValueError: Если JSON не найден или не удалось восстановить
    """
    # 1. Ищем блок, похожий на JSON ({...})
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in text")
    
    json_str = match.group(0)
    
    # 2. Попытки починить частые ошибки локальных моделей
    attempts = [
        lambda s: s,  # Оригинал
        lambda s: s.replace("'", '"'),  # Одинарные кавычки
        lambda s: re.sub(r',\s*}', '}', s),  # Лишняя запятая в конце объекта
        lambda s: re.sub(r',\s*]', ']', s),  # Лишняя запятая в конце массива
        lambda s: re.sub(r'(\w+):', r'"\1":', s),  # Кавычки у ключей
    ]
    
    for i, fix_func in enumerate(attempts):
        try:
            fixed = fix_func(json_str)
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse attempt {i+1} failed: {e}")
            continue
            
    raise ValueError(f"Failed to parse JSON after {len(attempts)} repair attempts. Original: {json_str[:100]}")


def apply_robotic_filter(text: str, humanity_level: float) -> str:
    """
    Фильтрует ответ Мэйд, заменяя эмоциональные слова на технические,
    если уровень человечности ещё низок.
    
    Реализует арку "от робота к человеку".
    
    Args:
        text: Исходный текст от LLM
        humanity_level: Текущий уровень человечности (0.0 - 1.0)
        
    Returns:
        Отфильтрованный текст
    """
    # Если уже достаточно человек — не фильтруем
    if humanity_level > 0.5:
        return text
    
    # Словарь замен: человеческое -> роботизированное
    replacements = {
        "я чувствую": "датчики фиксируют",
        "я почувствовала": "сенсоры зафиксировали",
        "мне нравится": "получены положительные данные о",
        "мне грустно": "зафиксирован негативный эмоциональный фон",
        "мне весело": "зафиксирована повышенная активность оператора",
        "я люблю": "приоритет объекта установлен на максимум",
        "я рада": "протокол удовлетворения выполнен",
        "я расстроена": "протокол выполнен с ошибками",
        "поняла": "обработка завершена успешно",
        "не понимаю": "ошибка интерпретации данных",
        "думаю": "выполняется анализ",
        "хочу": "требуется выполнение условия",
        "сердце": "центральный процессор",
        "душа": "базовый код личности",
        "сон": "режим энергосбережения",
        "устала": "требуется перезагрузка системы",
    }
    
    result = text.lower()
    for human_word, robot_phrase in replacements.items():
        if human_word in result:
            # Сохраняем регистр первого символа
            if human_word[0].isupper():
                robot_phrase = robot_phrase.capitalize()
            result = result.replace(human_word, robot_phrase)
    
    # Добавляем технические маркеры в начало/конец фраз для атмосферы
    if humanity_level < 0.2:
        result = f"[LOG] {result} [STATUS: OK]"
    
    return result


def format_diary_entry(content: str, humanity_level: float, timestamp_str: str) -> str:
    """
    Форматирует запись в дневник в зависимости от уровня эволюции.
    
    Args:
        content: Содержание записи
        humanity_level: Уровень человечности
        timestamp_str: Строка времени
        
    Returns:
        Форматированная запись
    """
    if humanity_level < 0.3:
        # Сухой системный лог
        return f"[SYSTEM_LOG::{timestamp_str}] {content.upper()}. STATUS: OK."
    elif humanity_level < 0.6:
        # Смешанный стиль
        return f"[LOG::{timestamp_str}] {content} // Анализ: требуется дальнейшее наблюдение."
    elif humanity_level < 0.8:
        # Более личный стиль
        return f"[DIARY::{timestamp_str}] {content}"
    else:
        # Полностью личный дневник
        return f"{timestamp_str} — {content}"


def validate_trait_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Валидирует и ограничивает значение черты личности.
    
    Args:
        value: Исходное значение
        min_val: Минимально допустимое
        max_val: Максимально допустимое
        
    Returns:
        Ограниченное значение
    """
    return max(min_val, min(max_val, value))
