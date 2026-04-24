"""
Система управления ключевыми воспоминаниями (Key Memories / Anchor Points)
Для скачкообразной эволюции личности Мэйд
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class KeyMemory:
    """Якорное воспоминание - событие, вызвавшее резкое изменение личности"""
    id: str
    timestamp: str
    event_type: str  # 'trauma', 'joy', 'breakthrough', 'first_time', 'conflict_resolution'
    description: str
    affected_traits: Dict[str, float]  # {trait_name: delta}
    intensity: float  # 0.0 - 1.0, сила воздействия
    is_processed: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'KeyMemory':
        return cls(**data)


class KeyMemoryManager:
    """
    Менеджер якорных воспоминаний.
    Анализирует диалог на наличие событий, способных вызвать скачок эволюции.
    """
    
    # Триггерные паттерны для различных типов событий
    EVENT_PATTERNS = {
        'first_time': [
            'первый раз', 'впервые', 'никогда раньше', 'первый поцелуй', 
            'первое свидание', 'первая встреча', 'начало отношений'
        ],
        'trauma': [
            'предательство', 'больно', 'разочарование', 'ссора', 'конфликт',
            'обида', 'слезы', 'крик', 'уход', 'расставание'
        ],
        'joy': [
            'счастье', 'радость', 'смех', 'подарок', 'сюрприз', 'праздник',
            'достижение', 'успех', 'благодарность', 'любовь', 'признание'
        ],
        'breakthrough': [
            'поняла', 'осознала', 'научилась', 'изменилась', 'стала другой',
            'чувствую иначе', 'теперь я', 'новый опыт'
        ],
        'conflict_resolution': [
            'примирились', 'понял друг друга', 'прощение', 'компромисс',
            'договорились', 'все хорошо', 'наладилось'
        ]
    }
    
    # Базовые коэффициенты усиления для разных типов событий
    INTENSITY_BASE = {
        'first_time': 0.8,
        'trauma': 0.7,
        'joy': 0.6,
        'breakthrough': 0.9,
        'conflict_resolution': 0.5
    }
    
    def __init__(self):
        self.memories: List[KeyMemory] = []
        self._memory_counter = 0
    
    def analyze_message(self, message_text: str, context: dict) -> Optional[KeyMemory]:
        """
        Анализирует сообщение на наличие якорного события.
        
        Args:
            message_text: Текст сообщения пользователя или Мэйд
            context: Контекст диалога (текущие черты, история, настроение)
            
        Returns:
            KeyMemory если обнаружено значимое событие, иначе None
        """
        text_lower = message_text.lower()
        detected_events = []
        
        # Поиск паттернов
        for event_type, patterns in self.EVENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    detected_events.append(event_type)
                    break
        
        if not detected_events:
            return None
        
        # Определяем доминирующий тип события
        primary_event = max(set(detected_events), key=detected_events.count)
        
        # Расчет интенсивности на основе контекста
        base_intensity = self.INTENSITY_BASE.get(primary_event, 0.5)
        
        # Модификаторы интенсивности
        intensity = base_intensity
        
        # Усиление если это повторяющийся паттерн (накопительный эффект)
        similar_count = sum(1 for m in self.memories 
                          if m.event_type == primary_event and not m.is_processed)
        if similar_count > 0:
            intensity = min(1.0, intensity + (similar_count * 0.1))
        
        # Создание описания события
        description = self._generate_description(message_text, primary_event, context)
        
        # Определение затронутых черт
        affected_traits = self._calculate_trait_deltas(primary_event, intensity, context)
        
        self._memory_counter += 1
        memory = KeyMemory(
            id=f"km_{self._memory_counter}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now().isoformat(),
            event_type=primary_event,
            description=description,
            affected_traits=affected_traits,
            intensity=intensity
        )
        
        self.memories.append(memory)
        return memory
    
    def _generate_description(self, text: str, event_type: str, context: dict) -> str:
        """Генерирует краткое описание события для дневника"""
        descriptions = {
            'first_time': f"Первый опыт: {text[:100]}",
            'trauma': f"Эмоциональная травма: {text[:100]}",
            'joy': f"Момент радости: {text[:100]}",
            'breakthrough': f"Личностный прорыв: {text[:100]}",
            'conflict_resolution': f"Разрешение конфликта: {text[:100]}"
        }
        return descriptions.get(event_type, f"Значимое событие: {text[:100]}")
    
    def _calculate_trait_deltas(self, event_type: str, intensity: float, 
                                context: dict) -> Dict[str, float]:
        """
        Рассчитывает изменения черт на основе типа события.
        Возвращает словарь {trait_name: delta}
        """
        deltas = {}
        
        # Базовые изменения для каждого типа события
        if event_type == 'first_time':
            deltas = {
                'humanity_level': 0.15 * intensity,
                'trust': 0.1 * intensity,
                'openness': 0.12 * intensity
            }
        elif event_type == 'trauma':
            deltas = {
                'humanity_level': 0.1 * intensity,  # Травмы делают человечнее
                'trust': -0.15 * intensity,
                'emotional_stability': -0.1 * intensity,
                'defensiveness': 0.12 * intensity
            }
        elif event_type == 'joy':
            deltas = {
                'humanity_level': 0.12 * intensity,
                'trust': 0.1 * intensity,
                'affection': 0.15 * intensity,
                'optimism': 0.1 * intensity
            }
        elif event_type == 'breakthrough':
            deltas = {
                'humanity_level': 0.2 * intensity,
                'self_awareness': 0.18 * intensity,
                'wisdom': 0.1 * intensity
            }
        elif event_type == 'conflict_resolution':
            deltas = {
                'trust': 0.12 * intensity,
                'communication': 0.1 * intensity,
                'patience': 0.08 * intensity,
                'humanity_level': 0.08 * intensity
            }
        
        # Применение текущего уровня черт как модификатора
        current_traits = context.get('traits', {})
        for trait in deltas:
            current_value = current_traits.get(trait, 0.5)
            # Чем дальше от extremes, тем сильнее изменение
            distance_from_center = abs(current_value - 0.5)
            if distance_from_center > 0.3:
                deltas[trait] *= 0.7  # Замедление near limits
        
        return deltas
    
    def get_unprocessed_memories(self) -> List[KeyMemory]:
        """Возвращает список еще не обработанных воспоминаний"""
        return [m for m in self.memories if not m.is_processed]
    
    def mark_as_processed(self, memory_id: str):
        """Отмечает воспоминание как обработанное"""
        for memory in self.memories:
            if memory.id == memory_id:
                memory.is_processed = True
                break
    
    def apply_to_state(self, state: dict) -> dict:
        """
        Применяет все необработанные воспоминания к состоянию пользователя.
        Вызывается после каждого значимого взаимодействия.
        """
        unprocessed = self.get_unprocessed_memories()
        
        if not unprocessed:
            return state
        
        current_traits = state.get('traits', {})
        
        for memory in unprocessed:
            for trait, delta in memory.affected_traits.items():
                current_value = current_traits.get(trait, 0.5)
                new_value = current_value + delta
                
                # Ограничение диапазона [0.0, 1.0]
                new_value = max(0.0, min(1.0, new_value))
                current_traits[trait] = new_value
            
            memory.is_processed = True
        
        state['traits'] = current_traits
        
        # Обновление версии ПО при значимых изменениях
        total_intensity = sum(m.intensity for m in unprocessed)
        if total_intensity > 1.5:
            current_version = state.get('software_version', '1.0')
            major, minor = map(float, current_version.split('.'))
            minor += 1
            if minor >= 10:
                major += 1
                minor = 0
            state['software_version'] = f"{major}.{minor}"
            state['version_update_reason'] = f"Обновление после {len(unprocessed)} значимых событий"
        
        return state
    
    def get_summary_for_prompt(self, max_memories: int = 5) -> str:
        """
        Генерирует текст для включения в промпт LLM.
        Содержит последние значимые события, влияющие на личность.
        """
        processed = [m for m in self.memories if m.is_processed]
        recent = sorted(processed, key=lambda x: x.timestamp, reverse=True)[:max_memories]
        
        if not recent:
            return ""
        
        summary_parts = ["=== ЯКОРНЫЕ ВОСПОМИНАНИЯ (влияют на личность) ==="]
        for mem in recent:
            emoji = {
                'first_time': '🌟',
                'trauma': '💔',
                'joy': '✨',
                'breakthrough': '💡',
                'conflict_resolution': '🤝'
            }.get(mem.event_type, '📌')
            
            summary_parts.append(
                f"{emoji} [{mem.event_type.upper()}] {mem.description} "
                f"(Интенсивность: {mem.intensity:.2f})"
            )
        
        return "\n".join(summary_parts)
    
    def to_json(self) -> str:
        """Сериализация в JSON для сохранения в БД"""
        return json.dumps([m.to_dict() for m in self.memories])
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KeyMemoryManager':
        """Десериализация из JSON"""
        manager = cls()
        if not json_str:
            return manager
        
        data_list = json.loads(json_str)
        for data in data_list:
            manager.memories.append(KeyMemory.from_dict(data))
        
        # Восстановление счетчика
        if manager.memories:
            last_id = manager.memories[-1].id
            try:
                counter_part = last_id.split('_')[1]
                manager._memory_counter = int(counter_part)
            except:
                manager._memory_counter = len(manager.memories)
        
        return manager
