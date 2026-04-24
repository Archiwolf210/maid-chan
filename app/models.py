"""
Модели данных для проекта Digital Human (Maid).
Используется Pydantic для валидации и типизации.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum


class PersonalityTrait(BaseModel):
    """Отдельная черта личности с метаданными."""
    value: float = Field(ge=0.0, le=1.0, description="Текущее значение черты")
    seed_value: float = Field(ge=0.0, le=1.0, description="Базовое значение (якорь)")
    delta: float = Field(default=0.0, description="Накопленное изменение")


class UserState(BaseModel):
    """
    Полное состояние пользователя.
    Заменяет собой разрозненные словари _USER_STATE.
    """
    uid: str
    username: str
    avatar_path: Optional[str] = None
    
    # Личность и эволюция
    traits: Dict[str, PersonalityTrait] = Field(default_factory=dict)
    humanity_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Уровень 'очеловечивания'")
    software_version: str = Field(default="1.0.0", description="Версия ПО личности")
    
    # Отношения
    relation_level: float = Field(default=0.0, ge=0.0, le=100.0)
    trust_level: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Режимы
    nsfw_mode: bool = False
    immersive_mode: bool = True
    
    # Контекст сессии
    last_interaction: Optional[datetime] = None
    current_mood: str = "neutral"
    
    class Config:
        arbitrary_types_allowed = True


class ChatRequest(BaseModel):
    """Запрос на сообщение в чат."""
    message: str = Field(..., min_length=1, max_length=5000)
    uid: Optional[str] = None  # Если не передано, берется из сессии/cookie


class ChatResponse(BaseModel):
    """Структура ответа (для логирования, не для SSE потока)."""
    text: str
    mood: str
    traits_updated: bool = False
    scene_generated: bool = False


class DiaryEntry(BaseModel):
    """Структура записи в дневник."""
    timestamp: datetime
    content: str
    log_type: str = Field(description="SYSTEM_LOG, EMOTIONAL_NOTE, etc.")
    relevance_score: float = Field(ge=0.0, le=1.0)


class AppConfig(BaseModel):
    """Типизированная конфигурация приложения."""
    llm_base_url: str
    llm_model: str
    embedding_model: str
    db_path: str
    immersive: Dict[str, Any] = Field(default_factory=dict)
    evolution_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    class Config:
        extra = "allow"  # Разрешаем лишние поля из config.json
