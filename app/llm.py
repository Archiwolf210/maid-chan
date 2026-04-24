"""Background LLM tasks: self-reflection, LTM compression, daily summary.

These three async jobs run OUTSIDE the chat hot path. They all POST to the
same llama-server used for main replies, but with tuned temperature / max_tokens
for their specific analytic purpose, and they write quietly to the DB.

  _reflection_task(uid)          -- short self-insights, stored in self_reflections
  _compress_ltm(uid)             -- fact extraction, stored in long_term_memory
  _build_daily_summary_async(uid) -- 24h narrative, stored in daily_summaries

All cross-module dependencies resolve via LATE imports from `main` inside
functions — avoids circular import at module load.
"""
from __future__ import annotations
import re
from datetime import datetime, timedelta

import httpx

from app.db import db
from app.memory import get_memory, encode_text


def _log():
    from main import log
    return log


def _log_exc(msg, exc):
    from main import _log_exc as _le
    _le(msg, exc)


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-REFLECTIONS (LLM insights)
# ─────────────────────────────────────────────────────────────────────────────
def _save_reflection(uid: str, text: str) -> None:
    """Insert one line of self-reflection, pruning to last 20 per user."""
    try:
        with db() as c:
            c.execute("INSERT INTO self_reflections(user_id,text) VALUES(?,?)", (uid, text[:300]))
            c.execute(
                "DELETE FROM self_reflections WHERE user_id=? AND id NOT IN "
                "(SELECT id FROM self_reflections WHERE user_id=? ORDER BY id DESC LIMIT 20)",
                (uid, uid))
    except Exception as e:
        _log_exc("_save_reflection", e)


async def _reflection_task(uid: str) -> None:
    """Ask LLM to produce 2-3 short self-observations about Maid's own behavior."""
    from main import _get_http_client, _llm_url, _clean
    log = _log()
    log.info("Reflection uid=%s", uid)
    history = get_memory(uid, 30)
    if len(history) < 15:
        return
    conv = "\n".join(f"{m['role'].upper()}: {m['content'][:200]}" for m in history[-20:])
    prompt = (
        "Ты -- Мэйд, AI-компаньонка. Проанализируй разговоры и сделай 2-3 "
        "осторожных вывода о своём поведении.\n"
        "Формат: каждый вывод начинается с '- я ', до 60 символов.\n\n"
        f"Разговоры:\n{conv}\n\nВыводы:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.2, "max_tokens": 300, "stream": False},
            timeout=120.0)
        if r.status_code != 200:
            log.warning("Reflection LLM %d", r.status_code); return
        raw = _clean(r.json()["choices"][0]["message"]["content"])
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("- я ") and len(line) > 5:
                _save_reflection(uid, line)
        log.info("Reflection done uid=%s", uid)
    except httpx.ReadTimeout:
        log.warning("Reflection timeout uid=%s -- model busy, skipping", uid)
    except Exception as e:
        _log_exc("_reflection_task", e)


# ─────────────────────────────────────────────────────────────────────────────
#  LTM COMPRESSION (extract facts from recent dialog)
# ─────────────────────────────────────────────────────────────────────────────
async def _compress_ltm(uid: str) -> None:
    """Extract 4-6 facts about the user from the last ~24 messages into LTM."""
    from main import _get_http_client, _llm_url, _clean, _detect_emotion
    log = _log()
    log.info("LTM compress uid=%s", uid)
    history = get_memory(uid, 40)
    if len(history) < 8:
        return
    conv = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-24:])
    prompt = (
        "Из диалога извлеки 4-6 важных фактов о ПОЛЬЗОВАТЕЛЕ.\n"
        "Категории: что радует, что тревожит, проекты, предпочтения, чего избегать.\n"
        "Формат: '- факт [категория]', категория: preference|event|goal|feeling|habit|avoid.\n\n"
        f"Диалог:\n{conv}\n\nФакты:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.15, "max_tokens": 400, "stream": False},
            timeout=120.0)
        if r.status_code != 200:
            log.warning("LTM LLM %d", r.status_code); return
        text = _clean(r.json()["choices"][0]["message"]["content"])
        lines = [ln.lstrip("-\u2022\u00b7 ").strip() for ln in text.splitlines()
                 if ln.strip().startswith(("-", "\u2022", "\u00b7"))]
        CATS = {"preference", "event", "goal", "feeling", "habit", "avoid", "general"}
        with db() as c:
            for raw in lines[:6]:
                if not raw:
                    continue
                m = re.search(r"\[(\w+)\]$", raw)
                cat = m.group(1) if m and m.group(1) in CATS else "general"
                fact = raw[:m.start()].strip() if m else raw
                etag, _ = _detect_emotion(fact)
                if not c.execute(
                    "SELECT id FROM long_term_memory WHERE user_id=? AND fact LIKE ?",
                    (uid, f"%{fact[:30]}%")).fetchone():
                    emb = encode_text(fact[:200], is_query=False)
                    c.execute(
                        "INSERT INTO long_term_memory(user_id,fact,category,emotion_tag,embedding) "
                        "VALUES(?,?,?,?,?)",
                        (uid, fact[:200], cat, etag, emb))
        log.info("LTM done uid=%s", uid)
    except httpx.ReadTimeout:
        log.warning("LTM compress timeout uid=%s -- model busy, will retry", uid)
    except Exception as e:
        _log_exc("_compress_ltm", e)


# ─────────────────────────────────────────────────────────────────────────────
#  DAILY SUMMARY (24h narrative, cached per-day)
# ─────────────────────────────────────────────────────────────────────────────
def _load_daily_summary(uid: str, day: str) -> str:
    try:
        with db() as c:
            row = c.execute(
                "SELECT summary FROM daily_summaries WHERE user_id=? AND day=?",
                (uid, day)).fetchone()
        return row[0] if row else ""
    except Exception as e:
        _log_exc("_load_daily_summary", e); return ""


def _save_daily_summary(uid: str, day: str, summary: str) -> None:
    try:
        with db() as c:
            c.execute(
                "INSERT INTO daily_summaries(user_id,day,summary) VALUES(?,?,?) "
                "ON CONFLICT(user_id,day) DO UPDATE SET summary=excluded.summary,ts=unixepoch()",
                (uid, day, summary[:1200]))
    except Exception as e:
        _log_exc("_save_daily_summary", e)


async def _build_daily_summary_async(uid: str) -> str:
    """
    Build a short narrative summary of the last 24 h of conversation.
    Cached per (uid, YYYY-MM-DD) — regenerates at most every 30 min.
    """
    from main import _get_http_client, _llm_url, _clean
    log = _log()
    day = datetime.now().strftime("%Y-%m-%d")
    cached = _load_daily_summary(uid, day)
    try:
        with db() as c:
            row = c.execute(
                "SELECT ts FROM daily_summaries WHERE user_id=? AND day=?",
                (uid, day)).fetchone()
        if cached and row and (int(datetime.now().timestamp()) - int(row[0])) < 1800:
            return cached
    except Exception:
        pass

    since_ts = int((datetime.now() - timedelta(hours=24)).timestamp())
    try:
        with db() as c:
            rows = c.execute(
                "SELECT role,content FROM memory "
                "WHERE user_id=? AND turn_status='completed' AND ts>=? "
                "ORDER BY id ASC LIMIT 120",
                (uid, since_ts)).fetchall()
    except Exception as e:
        _log_exc("daily_summary fetch", e); return cached

    if len(rows) < 6:
        return cached

    convo = "\n".join(
        f"{'Хозяин' if r['role']=='user' else 'Мэйд'}: {r['content'][:240]}"
        for r in rows[-80:]
    )
    prompt = (
        "Ты — Мэйд. Напиши короткую (до 140 слов) личную сводку за сегодня — "
        "о чём говорили с Хозяином, что ты заметила, что тебя тронуло. "
        "Пиши от первого лица, тепло, без списков. Не упоминай, что ты ИИ.\n\n"
        f"Диалог:\n{convo}\n\nСводка:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.4, "max_tokens": 260, "stream": False},
            timeout=120.0)
        if r.status_code != 200:
            log.warning("Daily summary LLM %d", r.status_code); return cached
        text = _clean(r.json()["choices"][0]["message"]["content"])
        if text:
            _save_daily_summary(uid, day, text)
            return text
    except httpx.ReadTimeout:
        log.warning("Daily summary timeout uid=%s", uid)
    except Exception as e:
        _log_exc("_build_daily_summary_async", e)
    return cached
