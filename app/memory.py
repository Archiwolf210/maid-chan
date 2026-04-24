"""Memory subsystems: STM / LTM / embeddings / scene / topics.

Consolidates disk- and model-backed memory utilities that are conceptually
"what Maid knows" but distinct from cognition and state logic:

  STM (short-term messages)
    get_memory, get_memory_for_prompt, clear_memory, delete_last_exchange

  Embeddings (fastembed, CPU, optional)
    get_embedder, encode_text, _decode_vec, _cosine_topk
    _ltm_backfill_embeddings  -- attach vectors to legacy LTM rows

  LTM (hybrid recall: semantic + keyword)
    get_ltm_relevant

  Scene (cosmetic RP state + free-text scene summary)
    load_rp_scene, save_rp_scene, _detect_rp_mode_change
    save_scene_summary, get_scene_summary, _update_scene_summary_async

  Pending topics (open conversational loops)
    add_pending_topic, get_open_topics, close_topic, _auto_extract_topics

All cross-module references go via LATE imports from `main` inside functions —
avoids circular import at module load.
"""
from __future__ import annotations
import re
import threading
from typing import Optional

import httpx

from app.db import db


def _log():
    from main import log
    return log


def _log_exc(msg, exc):
    from main import _log_exc as _le
    _le(msg, exc)


# ─────────────────────────────────────────────────────────────────────────────
#  SHORT-TERM MEMORY (messages table)
# ─────────────────────────────────────────────────────────────────────────────
def get_memory(uid, limit=20):
    """Returns only completed messages (no pending/in-flight)."""
    try:
        with db() as c:
            rows = c.execute(
                "SELECT role,content,importance,emotion_tag,intent_tag FROM memory "
                "WHERE user_id=? AND turn_status='completed' ORDER BY id DESC LIMIT ?",
                (uid, limit)).fetchall()
        return [{"role": r[0], "content": r[1], "importance": r[2],
                 "emotion_tag": r[3], "intent_tag": r[4]} for r in reversed(rows)]
    except Exception as e:
        _log_exc("get_memory", e); return []


def get_memory_for_prompt(uid, limit=20):
    return [{"role": m["role"], "content": m["content"]} for m in get_memory(uid, limit)]


def clear_memory(uid):
    """v9.1 semantics:
      - Wipes short-term memory + topic-links.
      - RESETS `msg_count` (session counter) and `last_activity_ts`.
      - KEEPS `total_msg_count` — lifetime milestones stay consistent.
      - RESETS rp_scene to 'normal'/empty.
      - Does NOT touch long_term_memory, character_traits, reflections, or
        emotional state — user has dedicated buttons for those.
    """
    try:
        with db() as c:
            c.execute("DELETE FROM memory WHERE user_id=?", (uid,))
            c.execute("DELETE FROM memory_links WHERE user_id=?", (uid,))
            c.execute("UPDATE user_state SET msg_count=0, last_activity_ts=0 WHERE user_id=?", (uid,))
            c.execute("DELETE FROM rp_scene WHERE user_id=?", (uid,))
    except Exception as e:
        _log_exc("clear_memory", e)


def delete_last_exchange(uid):
    """Delete last user->assistant pair by role — not just two most recent rows."""
    try:
        with db() as c:
            ids = []
            last_asst = c.execute(
                "SELECT id FROM memory WHERE user_id=? AND role='assistant' AND turn_status='completed' ORDER BY id DESC LIMIT 1",
                (uid,)).fetchone()
            if last_asst:
                asst_id = last_asst[0]; ids.append(asst_id)
                last_user = c.execute(
                    "SELECT id FROM memory WHERE user_id=? AND role='user' AND id < ? ORDER BY id DESC LIMIT 1",
                    (uid, asst_id)).fetchone()
                if last_user: ids.append(last_user[0])
            else:
                last_user = c.execute(
                    "SELECT id FROM memory WHERE user_id=? AND role='user' ORDER BY id DESC LIMIT 1",
                    (uid,)).fetchone()
                if last_user: ids.append(last_user[0])
            if ids:
                ph = ",".join("?" * len(ids))
                c.execute(f"DELETE FROM memory WHERE id IN ({ph})", ids)
                c.execute(f"DELETE FROM memory_links WHERE from_id IN ({ph}) OR to_id IN ({ph})", ids + ids)
            return len(ids)
    except Exception as e:
        _log_exc("delete_last_exchange", e); return 0


# ─────────────────────────────────────────────────────────────────────────────
#  SEMANTIC EMBEDDINGS (v8.3) — CPU-only, lazy, fastembed
# ─────────────────────────────────────────────────────────────────────────────
#  Degrades gracefully: if fastembed / numpy are missing, encode_text returns
#  None and LTM recall falls back to keyword path.
#  Defaults to intfloat/multilingual-e5-large; overridable via config.memory.
#  Vectors are L2-normalized float32 little-endian BLOBs (cosine = dot product).
_EMBEDDER_STATE = {"model": None, "tried": False, "name": None, "dim": None}
_EMBED_LOCK = threading.Lock()


def _embedding_cfg() -> dict:
    from main import load_config
    mem = load_config().get("memory", {}) or {}
    return {
        "enabled": bool(mem.get("embedding_enabled", True)),
        "model":   mem.get("embedding_model",   "intfloat/multilingual-e5-large"),
        "threads": int(mem.get("embedding_threads", 4) or 4),
    }


def get_embedder():
    """Return (model, name, dim) tuple or (None, None, None) if unavailable."""
    log = _log()
    st = _EMBEDDER_STATE
    if st["model"] is not None: return st["model"], st["name"], st["dim"]
    if st["tried"]:              return None, None, None
    with _EMBED_LOCK:
        if st["model"] is not None: return st["model"], st["name"], st["dim"]
        if st["tried"]:              return None, None, None
        cfg = _embedding_cfg()
        if not cfg["enabled"]:
            st["tried"] = True
            log.info("Embeddings disabled via config.memory.embedding_enabled=false")
            return None, None, None
        try:
            from fastembed import TextEmbedding  # type: ignore
            import numpy as _np  # type: ignore
            log.info("Loading embedding model: %s (threads=%d) -- first run may download ~1 GB",
                     cfg["model"], cfg["threads"])
            m = TextEmbedding(model_name=cfg["model"], threads=cfg["threads"])
            probe = list(m.embed(["passage: hello"]))
            dim = int(_np.asarray(probe[0]).shape[0]) if probe else 0
            st["model"], st["name"], st["dim"], st["tried"] = m, cfg["model"], dim, True
            log.info("Embedding model ready: dim=%d", dim)
            return m, cfg["model"], dim
        except Exception as e:
            st["tried"] = True
            log.warning("Embeddings unavailable (%s). LTM recall will use keyword fallback.", e)
            return None, None, None


def encode_text(text: str, is_query: bool = False) -> Optional[bytes]:
    """Encode `text` to L2-normalized float32 bytes. Returns None if embedder unavailable."""
    if not text: return None
    m, _name, _dim = get_embedder()
    if m is None: return None
    try:
        import numpy as _np  # type: ignore
        prefix = "query: " if is_query else "passage: "
        vecs = list(m.embed([prefix + text[:2000]]))
        if not vecs: return None
        v = _np.asarray(vecs[0], dtype=_np.float32)
        n = float(_np.linalg.norm(v)) or 1.0
        v = (v / n).astype(_np.float32, copy=False)
        return v.tobytes()
    except Exception as e:
        _log_exc("encode_text", e); return None


def _decode_vec(blob: Optional[bytes]):
    if not blob: return None
    try:
        import numpy as _np  # type: ignore
        return _np.frombuffer(blob, dtype=_np.float32)
    except Exception:
        return None


def _cosine_topk(qblob: bytes, candidates: list, k: int) -> list:
    """Return [(score, candidate_dict), ...] sorted desc.
    Cosine collapses to dot-product because all vectors are L2-normalized."""
    try:
        import numpy as _np  # type: ignore
        q = _np.frombuffer(qblob, dtype=_np.float32)
        scored = []
        for r in candidates:
            v = _decode_vec(r.get("embedding"))
            if v is None or v.shape != q.shape: continue
            scored.append((float(_np.dot(q, v)), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]
    except Exception as e:
        _log_exc("_cosine_topk", e); return []


def _ltm_backfill_embeddings(max_rows: int = 200) -> int:
    """Background job: attach embeddings to old LTM rows that lack them.
    Small batches to avoid locking the DB. Returns rows updated."""
    log = _log()
    m, _name, _dim = get_embedder()
    if m is None: return 0
    try:
        with db() as c:
            rows = c.execute(
                "SELECT id, fact FROM long_term_memory "
                "WHERE embedding IS NULL AND fact IS NOT NULL "
                "ORDER BY id ASC LIMIT ?", (int(max_rows),)
            ).fetchall()
        if not rows: return 0
        updated = 0
        batch_texts = [("passage: " + (r["fact"] or ""))[:2000] for r in rows]
        try:
            import numpy as _np  # type: ignore
            vecs = list(m.embed(batch_texts))
            blobs = []
            for v in vecs:
                arr = _np.asarray(v, dtype=_np.float32)
                n = float(_np.linalg.norm(arr)) or 1.0
                blobs.append((arr / n).astype(_np.float32, copy=False).tobytes())
        except Exception as e:
            _log_exc("backfill encode", e); return 0
        with db() as c:
            for r, blob in zip(rows, blobs):
                c.execute("UPDATE long_term_memory SET embedding=? WHERE id=?", (blob, r["id"]))
                updated += 1
        log.info("LTM backfill: embedded %d row(s)", updated)
        return updated
    except Exception as e:
        _log_exc("_ltm_backfill_embeddings", e); return 0


# ─────────────────────────────────────────────────────────────────────────────
#  LTM HYBRID RECALL (semantic + keyword)
# ─────────────────────────────────────────────────────────────────────────────
_LTM_COS_MIN = 0.55   # ignore weak semantic hits (noise floor for e5 on RU)


def get_ltm_relevant(uid, text, limit=8):
    from main import _detect_emotion
    try:
        with db() as c:
            rows = c.execute(
                "SELECT id,fact,category,importance,emotion_tag,access_count,embedding "
                "FROM long_term_memory WHERE user_id=? ORDER BY importance DESC",
                (uid,)).fetchall()
        if not rows: return []
        rows_d = [dict(r) for r in rows]
        tl = text.lower(); ue, _ = _detect_emotion(text)

        # 1) Semantic path
        chosen_ids = set(); ranked = []
        qvec = encode_text(text, is_query=True)
        if qvec:
            cand = [r for r in rows_d if r.get("embedding")]
            top = _cosine_topk(qvec, cand, limit)
            for sim, r in top:
                if sim < _LTM_COS_MIN: continue
                blended = sim * 0.75 + float(r["importance"]) * 0.20 + min(int(r.get("access_count") or 0), 10) * 0.005
                ranked.append((blended, r))
                chosen_ids.add(r["id"])

        # 2) Keyword fallback
        for r in rows_d:
            if r["id"] in chosen_ids: continue
            sc = float(r["importance"]) + len(set(re.findall(r"\w+", r["fact"].lower())) & set(re.findall(r"\w+", tl))) * 0.08
            if r["emotion_tag"] == ue and ue != "neutral": sc += 0.15
            ranked.append((sc * 0.5, r))

        ranked.sort(key=lambda x: x[0], reverse=True)
        result = []
        seen = set()
        for _sc, r in ranked:
            if r["id"] in seen: continue
            seen.add(r["id"])
            r2 = {k: v for k, v in r.items() if k != "embedding"}
            result.append(r2)
            if len(result) >= limit: break

        if result:
            with db() as c:
                for r in result:
                    c.execute("UPDATE long_term_memory SET access_count=access_count+1,last_accessed=unixepoch() WHERE id=?", (r["id"],))
        return result
    except Exception as e:
        _log_exc("get_ltm_relevant", e); return []


# ─────────────────────────────────────────────────────────────────────────────
#  SCENE SUMMARY (persistent, LTM-backed)
# ─────────────────────────────────────────────────────────────────────────────
def save_scene_summary(uid: str, summary: str) -> None:
    """Store current RP scene context in LTM so it survives restarts."""
    if not summary or not summary.strip():
        return
    try:
        with db() as c:
            c.execute("DELETE FROM long_term_memory WHERE user_id=? AND category='scene_summary'", (uid,))
            emb = encode_text(summary[:400], is_query=False)
            c.execute(
                "INSERT INTO long_term_memory(user_id,fact,category,importance,emotion_tag,embedding) "
                "VALUES(?,?,?,?,?,?)",
                (uid, summary[:400], "scene_summary", 1.0, "neutral", emb))
    except Exception as e:
        _log_exc("save_scene_summary", e)


def get_scene_summary(uid: str) -> str:
    """Retrieve stored RP scene context for prompt injection."""
    try:
        with db() as c:
            row = c.execute(
                "SELECT fact FROM long_term_memory WHERE user_id=? AND category='scene_summary' "
                "ORDER BY ts DESC LIMIT 1", (uid,)).fetchone()
        return row[0] if row else ""
    except Exception as e:
        _log_exc("get_scene_summary", e); return ""


async def _update_scene_summary_async(uid: str, last_exchange: str) -> None:
    """Ask LLM to distill current scene into a short summary for next session."""
    from main import _get_http_client, _llm_url, _clean
    log = _log()
    prompt = (
        "Опиши текущую сцену/атмосферу диалога одним абзацем (до 120 слов).\n"
        "Включи: место, настроение, эмоциональный контекст, незавершённые моменты.\n"
        "Пиши от третьего лица, как краткое авторское описание.\n\n"
        f"Последний обмен:\n{last_exchange}\n\nОписание сцены:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 180, "stream": False},
            timeout=60.0)
        if r.status_code != 200:
            log.warning("Scene summary LLM %d", r.status_code); return
        text = _clean(r.json()["choices"][0]["message"]["content"])
        if text:
            save_scene_summary(uid, text)
            log.debug("Scene summary saved uid=%s", uid)
    except httpx.ReadTimeout:
        log.warning("Scene summary timeout uid=%s", uid)
    except Exception as e:
        _log_exc("_update_scene_summary_async", e)


# ─────────────────────────────────────────────────────────────────────────────
#  PENDING TOPICS (open conversational loops)
# ─────────────────────────────────────────────────────────────────────────────
def add_pending_topic(uid: str, topic: str, context: str = "", importance: float = 0.6,
                      ttl_days: int = 3) -> int:
    """Save an open conversational loop for later follow-up."""
    try:
        with db() as c:
            existing = c.execute(
                "SELECT id FROM pending_topics WHERE user_id=? AND topic LIKE ? AND status='open'",
                (uid, f"%{topic[:30]}%")).fetchone()
            if existing:
                return existing[0]
            cur = c.execute(
                "INSERT INTO pending_topics(user_id,topic,context,importance,expires_at) "
                "VALUES(?,?,?,?,unixepoch()+?)",
                (uid, topic[:200], context[:500], importance, ttl_days * 86400))
            return cur.lastrowid
    except Exception as e:
        _log_exc("add_pending_topic", e); return -1


def get_open_topics(uid: str, limit: int = 3) -> list[dict]:
    """Return open topics that haven't expired yet."""
    try:
        with db() as c:
            rows = c.execute(
                "SELECT id,topic,context,importance FROM pending_topics "
                "WHERE user_id=? AND status='open' AND expires_at > unixepoch() "
                "ORDER BY importance DESC, id DESC LIMIT ?",
                (uid, limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        _log_exc("get_open_topics", e); return []


def close_topic(uid: str, topic_id: int) -> None:
    try:
        with db() as c:
            c.execute("UPDATE pending_topics SET status='closed' WHERE id=? AND user_id=?",
                      (topic_id, uid))
    except Exception as e:
        _log_exc("close_topic", e)


def _auto_extract_topics(uid: str, user_text: str, cog) -> None:
    """Automatically detect open loops from user message and save them."""
    log = _log()
    tl = user_text.lower()
    open_loop_signals = [
        "потом расскажу", "скоро узнаю", "завтра", "послезавтра", "на следующей неделе",
        "жду ответа", "надо решить", "думаю об этом", "беспокоит", "не знаю как",
        "важная встреча", "важный разговор", "сдать", "защитить", "собеседование",
        "врач", "результаты", "операция", "экзамен", "дедлайн",
    ]
    for signal in open_loop_signals:
        if signal in tl:
            topic = user_text[:100].strip()
            add_pending_topic(uid, topic, context=user_text[:300],
                              importance=max(0.6, cog.intensity), ttl_days=4)
            log.debug("Auto-extracted pending topic uid=%s", uid)
            break

    closed_signals = ["всё хорошо", "решилось", "прошло", "сдал", "прошёл", "получилось",
                      "разобрался", "договорились", "уже не важно"]
    if any(s in tl for s in closed_signals):
        open_topics = get_open_topics(uid, 5)
        if open_topics:
            close_topic(uid, open_topics[0]["id"])
            log.debug("Auto-closed topic uid=%s id=%d", uid, open_topics[0]["id"])


# ─────────────────────────────────────────────────────────────────────────────
#  RP SCENE STATE MACHINE (cosmetic mode/location/atmosphere)
# ─────────────────────────────────────────────────────────────────────────────
def load_rp_scene(uid: str) -> dict:
    try:
        with db() as c:
            row = c.execute(
                "SELECT mode,location,atmosphere FROM rp_scene WHERE user_id=?", (uid,)
            ).fetchone()
        return dict(row) if row else {"mode": "normal", "location": "", "atmosphere": ""}
    except Exception as e:
        _log_exc("load_rp_scene", e)
        return {"mode": "normal", "location": "", "atmosphere": ""}


def save_rp_scene(uid: str, mode: str, location: str = "", atmosphere: str = "") -> None:
    if mode not in ("normal", "rp", "nsfw"):
        mode = "normal"
    try:
        with db() as c:
            c.execute(
                "INSERT INTO rp_scene(user_id,mode,location,atmosphere,last_updated) "
                "VALUES(?,?,?,?,unixepoch()) ON CONFLICT(user_id) DO UPDATE SET "
                "mode=excluded.mode,location=excluded.location,atmosphere=excluded.atmosphere,"
                "last_updated=excluded.last_updated",
                (uid, mode, location[:200], atmosphere[:300]))
    except Exception as e:
        _log_exc("save_rp_scene", e)


def _detect_rp_mode_change(user_text: str, current_mode: str, cfg: dict) -> str:
    """Detect if user wants to enter/exit RP mode."""
    tl = user_text.lower()
    exit_signals = ["/выход", "/стоп", "/обычно", "выйди из роли", "без роли", "просто поговорим"]
    if any(s in tl for s in exit_signals):
        return "normal"
    if cfg.get("nsfw_mode", False):
        nsfw_enter = ["/нсфв", "/nsfw", "/сцена18", "начнём сцену", "войди в роль", "/rp nsfw"]
        if any(s in tl for s in nsfw_enter):
            return "nsfw"
    rp_enter = ["/рп", "/rp", "/сцена", "начни сцену", "давай разыграем", "представь что"]
    if any(s in tl for s in rp_enter):
        return "rp"
    return current_mode


# ─────────────────────────────────────────────────────────────────────────────
#  CONTEXT SUMMARIZATION (Long-term Coherence)
# ─────────────────────────────────────────────────────────────────────────────
def summarize_old_messages(uid: str, days_old: int = 7, max_summaries: int = 5) -> str:
    """
    Генерирует суммаризацию старых сообщений (старше days_old дней).
    Вместо хранения каждого слова годовой давности, храним краткое саммари.
    
    Args:
        uid: ID пользователя
        days_old: Минимальный возраст сообщений для суммаризации
        max_summaries: Максимальное количество саммари для включения в контекст
        
    Returns:
        Текст саммари для включения в промпт LLM
    """
    from datetime import datetime, timedelta
    from app.llm import generate_text
    
    cutoff_date = datetime.now() - timedelta(days=days_old)
    cutoff_ts = cutoff_date.timestamp()
    
    try:
        with db() as c:
            # Получаем старые сообщения, которые еще не суммаризированы
            rows = c.execute("""
                SELECT id, role, content, timestamp 
                FROM messages 
                WHERE user_id = ? AND timestamp < ? AND is_summarized = 0
                ORDER BY timestamp ASC
                LIMIT 100
            """, (uid, cutoff_ts)).fetchall()
            
            if not rows:
                # Возвращаем уже существующие саммари
                summaries = c.execute("""
                    SELECT summary_content, summary_date
                    FROM message_summaries
                    WHERE user_id = ?
                    ORDER BY summary_date DESC
                    LIMIT ?
                """, (uid, max_summaries)).fetchall()
                
                if not summaries:
                    return ""
                
                result_parts = ["=== ИСТОРИЯ ВЗАИМОДЕЙСТВИЙ (краткое содержание) ==="]
                for summ_content, summ_date in summaries:
                    date_str = datetime.fromtimestamp(summ_date).strftime('%Y-%m-%d')
                    result_parts.append(f"[{date_str}] {summ_content}")
                
                return "\n".join(result_parts)
            
            # Группируем сообщения по дням для суммаризации
            messages_by_day = {}
            for row in rows:
                day_key = datetime.fromtimestamp(row['timestamp']).date()
                if day_key not in messages_by_day:
                    messages_by_day[day_key] = []
                messages_by_day[day_key].append({
                    'role': row['role'],
                    'content': row['content']
                })
            
            # Генерируем саммари для каждой группы
            new_summaries = []
            for day, messages in messages_by_day.items():
                # Формируем компактный текст для LLM
                day_text = f"Дата: {day}\nДиалог:\n"
                for msg in messages[:20]:  # Ограничиваем количество сообщений за день
                    role_ru = "Пользователь" if msg['role'] == 'user' else "Мэйд"
                    day_text += f"{role_ru}: {msg['content'][:150]}\n"
                
                # Запрос к LLM для генерации саммари
                summary_prompt = f"""
Кратко суммаризируй следующие сообщения диалога за один день (2-3 предложения):
- Какие важные события произошли?
- Какие темы обсуждались?
- Были ли значимые эмоциональные моменты?

{day_text}

Саммари (только факты, без эмоций):
"""
                try:
                    summary_text = generate_text(summary_prompt, max_tokens=150, temperature=0.3)
                    summary_text = summary_text.strip()[:400]  # Ограничение длины
                    
                    new_summaries.append({
                        'date': day.isoformat(),
                        'timestamp': datetime.combine(day, datetime.min.time()).timestamp(),
                        'content': summary_text
                    })
                except Exception as e:
                    _log_exc(f"Failed to summarize day {day}", e)
                    continue
            
            # Сохраняем саммари в БД
            if new_summaries:
                with db() as c:
                    for summ in new_summaries:
                        c.execute("""
                            INSERT INTO message_summaries(user_id, summary_date, summary_content, created_at)
                            VALUES(?, ?, ?, unixepoch())
                            ON CONFLICT(user_id, summary_date) DO UPDATE SET
                            summary_content=excluded.summary_content,
                            created_at=excluded.created_at
                        """, (uid, summ['timestamp'], summ['content']))
                    
                    # Помечаем сообщения как суммаризированные
                    msg_ids = [r['id'] for r in rows]
                    if msg_ids:
                        placeholders = ','.join('?' * len(msg_ids))
                        c.execute(f"""
                            UPDATE messages SET is_summarized = 1
                            WHERE id IN ({placeholders})
                        """, msg_ids)
            
            # Возвращаем все саммари (новые + старые)
            all_summaries = c.execute("""
                SELECT summary_content, summary_date
                FROM message_summaries
                WHERE user_id = ?
                ORDER BY summary_date DESC
                LIMIT ?
            """, (uid, max_summaries)).fetchall()
            
            if not all_summaries:
                return ""
            
            result_parts = ["=== ИСТОРИЯ ВЗАИМОДЕЙСТВИЙ (краткое содержание) ==="]
            for summ_content, summ_date in all_summaries:
                date_str = datetime.fromtimestamp(summ_date).strftime('%Y-%m-%d')
                result_parts.append(f"[{date_str}] {summ_content}")
            
            return "\n".join(result_parts)
            
    except Exception as e:
        _log_exc("summarize_old_messages", e)
        return ""


def get_context_with_summary(uid: str, recent_limit: int = 20, days_for_summary: int = 7) -> str:
    """
    Комбинирует недавние сообщения (полные) со старыми (суммаризированными).
    Оптимизирует использование контекстного окна LLM.
    
    Args:
        uid: ID пользователя
        recent_limit: Количество последних сообщений для включения полностью
        days_for_summary: Сообщения старше скольких дней суммаризировать
        
    Returns:
        Полный контекст для промпта
    """
    parts = []
    
    # 1. Добавляем суммаризированную историю
    summary = summarize_old_messages(uid, days_old=days_for_summary)
    if summary:
        parts.append(summary)
        parts.append("")  # Пустая строка для разделения
    
    # 2. Добавляем недавние сообщения полностью
    recent = get_memory(uid, limit=recent_limit)
    if recent:
        parts.append("=== ПОСЛЕДНИЕ СООБЩЕНИЯ (полностью) ===")
        for msg in recent:
            timestamp = datetime.fromtimestamp(msg['timestamp']).strftime('%H:%M')
            role = "Вы" if msg['role'] == 'user' else "Мэйд"
            parts.append(f"[{timestamp}] {role}: {msg['content']}")
    
    return "\n".join(parts)
