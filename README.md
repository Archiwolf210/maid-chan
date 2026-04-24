# Digital Human v9.1 — Мэйд

Локальный AI-компаньон на FastAPI + llama-server (CUDA) с Qwen3-32B Q4_K_M и спекулятивным декодированием через Qwen3-0.6B Q8_0.

## Структура

```
DigitalHuman_v8\
├── main.py              ← бэкенд (FastAPI + SSE + SQLite WAL)
├── check_server.py      ← watchdog
├── start.bat            ← запуск
├── requirements.txt
├── config.json
├── web\
│   └── index.html       ← веб-интерфейс (SPA)
├── models\
│   ├── qwen3.gguf         ← основная модель
│   └── qwen3-draft.gguf   ← draft для speculative decoding
├── llm\
│   └── llama-server.exe ← из llama.cpp с CUDA
└── logs\                ← error.log + debug.log
```

## Железо

- GPU: RTX 3090 24GB
- CPU: Xeon E5-2666 v3 (или сопоставимый)
- RAM: 32GB
- Все слои модели (`gpu_layers=999`) — в VRAM

## Установка

1. Скопируй `llama-server.exe` (llama.cpp build с CUDA) в `llm\`
2. Положи `qwen3.gguf` (Qwen3-32B Q4_K_M) в `models\`
3. Положи `qwen3-draft.gguf` (Qwen3-0.6B Q8_0) в `models\` — draft для speculative decoding
4. Запусти `start.bat`

## Ключевые параметры (`config.json`)

| Секция | Ключ | Значение | Комментарий |
|---|---|---|---|
| `inference.profiles.daily` | `ctx_size` | `32768` | полный 32k контекст Qwen3 |
| | `gpu_layers` | `999` | все слои в VRAM |
| | `draft_gpu_layers` | `99` | draft целиком на GPU |
| | `draft_max` | `8` | окно спекуляции |
| | `ubatch` | `512` | batch size |
| | `cache_type_k/v` | `q8_0` | половинный KV-cache |
| | `temperature` | `0.72` | основной ответ |
| `memory` | `short_term_limit` | `40` | окно STM |
| | `compress_every` | `80` | частота LTM-компрессии |
| | `embedding_model` | `intfloat/multilingual-e5-large` | fastembed CPU |
| `session` | `gap_seconds` | `10800` | 3ч до сброса сессии |
| `immersive` | `enabled` | `true` | live-scene второго прохода |
| | `temperature` | `0.85` | чуть живее основного |
| | `max_tokens` | `-1` | без ограничения |
| `server` | `remote_mode` | `local_trusted` | доверие локалхосту |
| `nsfw_mode` | — | `true` | NSFW-ветка разблокирована |

## API

| Метод | URL | Описание |
|---|---|---|
| GET | `/` | веб-интерфейс |
| GET | `/health` | статус бэка + LLM (watchdog) |
| GET | `/api/status` | расширенный статус (версия, embeddings, режим) |
| POST | `/api/chat` | SSE streaming чат (основной путь) |
| GET | `/api/live_scene` | текущая immersive-сцена + `since` |
| POST | `/api/immersive/cancel` | отменить in-flight immersive |
| POST | `/api/immersive/resume` | снять авто-паузу |
| GET | `/memory` | окно STM |
| DELETE | `/api/memory/clear` | сбросить сессию (STM + счётчик + rp_scene) |
| GET | `/api/ltm` | факты долгосрочной памяти |
| POST | `/api/ltm/compress` | извлечь факты из последнего окна |
| POST | `/api/ltm/reindex` | бэкфилл эмбеддингов для старых записей |
| GET | `/api/cognitive` | последние 10 когнитивных кадров |
| GET | `/api/reflections` | автоматические саморефлексии Мэйд |
| POST | `/api/reflections/trigger` | форсировать рефлексию |
| GET | `/api/scene_summary` / POST / DELETE | ручная настройка сцены |
| GET | `/api/daily_summary` | сводка за день (по запросу) |
| GET/POST/PATCH | `/api/tasks` | задачи |
| GET/POST/DELETE | `/api/notes` | заметки |
| GET | `/stats` | эмоциональное состояние + черты + счётчики |
| GET | `/api/users` | список профилей |
| GET | `/config` / POST | конфигурация (горячее чтение/запись) |

## Что под капотом

- **SQLite WAL** с 14 таблицами (memory, user_state, long_term_memory, memory_links, cognitive_log, character_traits, self_reflections, tasks, notes, users, trait_intent_counts, pending_topics, rp_scene, daily_summaries) + автомиграции через `_MIGRATIONS`.
- **Hybrid LTM retrieval**: fastembed cosine (L2-normalized float32 BLOB) + keyword fallback.
- **Dual-counter**: `msg_count` (сессия, сбрасывается по `session.gap_seconds`) + `total_msg_count` (lifetime, управляет LTM-компрессией, рефлексиями, эволюцией черт).
- **KV-cache-friendly prompt split**: `build_prompt → (system_static, system_dynamic)`.
- **Per-user `threading.Lock`** для сериализации state.
- **Pinned background tasks** через `_track(asyncio.create_task(...))` — GC-safe.
- **Dedicated 16-worker ThreadPoolExecutor** для блокирующего IO.
- **Immersive live-scene** — второй проход LLM после основного ответа, генерирует action/atmosphere/inner-thought как JSON; только READ по памяти, пишет только в `rp_scene` + process-cache.
- **RP/NSFW state machine**: `normal` / `rp` / `nsfw` через `_detect_rp_mode_change`.

## Версия

Единая константа `VERSION` в [main.py](main.py). Отдаётся на `/health`, `/api/status`, FastAPI title, startup-лог. Фронт подтягивает её через `/health` в header.
