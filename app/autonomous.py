"""Autonomous behavior: proactive check-ins + nightly diary (vision-gap, v9.2).

These features let Мэйд reach out on her own, independent of the chat loop.

  Proactivity:
    A background task scans every N minutes for users who have been idle a
    moderate amount AND have an open pending_topic with importance >= threshold.
    For such matches, the LLM generates a short, in-character nudge which is
    queued in a per-user deque. The frontend polls /api/proactive/pending
    (returns and clears) and renders the text as a chip above the input box.

  Diary:
    A background task checks every N minutes if the local day has rolled over
    for a user who had conversation activity the prior day AND doesn't already
    have a diary row for that day. When so, it generates a short first-person
    diary entry from Мэйд's perspective and stores it in diary_entries.
    The UI exposes it via /api/diary (today/latest) and /api/diary/days.

Both loops are scheduled in the FastAPI startup hook via start_autonomous_loops()
and registered with main's _track() so GC-safety is preserved.

All cross-module refs use LATE imports from `main` inside functions.
"""
from __future__ import annotations
import asyncio
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import httpx

from app.db import db


# ── module-level state ───────────────────────────────────────────────────────
_PROACTIVE_QUEUE: dict = {}            # uid -> deque[dict{text, ts, topic_id}]
_PROACTIVE_LAST_SENT: dict = {}        # uid -> unix-ts of last delivery
_LOOPS_STARTED = False                 # idempotency guard


def _log():
    from main import log
    return log


def _log_exc(msg, exc):
    from main import _log_exc as _le
    _le(msg, exc)


def _cfg() -> dict:
    from main import load_config
    return (load_config().get("autonomous") or {})


# ─────────────────────────────────────────────────────────────────────────────
#  PROACTIVITY: scan → LLM nudge → queue
# ─────────────────────────────────────────────────────────────────────────────
def _get_queue(uid: str) -> deque:
    q = _PROACTIVE_QUEUE.get(uid)
    if q is None:
        maxlen = int(_cfg().get("proactive_max_queue", 3))
        q = deque(maxlen=max(1, maxlen))
        _PROACTIVE_QUEUE[uid] = q
    return q


def get_proactive_pending(uid: str, consume: bool = True) -> list[dict]:
    """Return (and optionally clear) queued proactive messages for this user.
    Endpoint: GET /api/proactive/pending?consume=1
    """
    q = _PROACTIVE_QUEUE.get(uid)
    if not q:
        return []
    items = list(q)
    if consume:
        q.clear()
    return items


def clear_proactive_queue(uid: str) -> None:
    """Called from memory_clear + delete_user so old nudges don't leak."""
    _PROACTIVE_QUEUE.pop(uid, None)
    _PROACTIVE_LAST_SENT.pop(uid, None)


async def _generate_proactive_text(uname: str, topic: str, ctx: str, gap_hours: float) -> Optional[str]:
    """Ask LLM for a short, in-character nudge. Returns stripped text or None."""
    from main import _get_http_client, _llm_url, _clean
    log = _log()
    cfg = _cfg()
    # Prompt keeps Мэйд in her voice: warm, with initiative, but not clingy.
    prompt = (
        "Ты — Мэйд, AI-компаньонка. Хозяин молчал уже какое-то время, "
        f"но у вас остался незакрытый разговор.\n"
        f"Хозяин: {uname}. Пауза примерно {int(gap_hours)} ч.\n"
        f"Незакрытая тема: {topic}\n"
        f"Контекст: {ctx[:300]}\n\n"
        "Напиши ОДНО короткое сообщение (1-2 предложения, до 140 символов), "
        "в котором ты ненавязчиво возвращаешься к этой теме. "
        "Говори от первого лица, в своей манере — тепло, с интересом, без заискивания. "
        "Без приветствий вроде 'привет' и без имени в начале. "
        "Никаких <think> — только сам текст.\n/no_think\n\nСообщение:"
    )
    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": float(cfg.get("proactive_temperature", 0.75)),
                  "max_tokens": 160, "stream": False, "cache_prompt": True},
            timeout=45.0)
        if r.status_code != 200:
            log.warning("Proactive LLM %d", r.status_code); return None
        text = _clean(r.json()["choices"][0]["message"]["content"]).strip().strip('"').strip("'")
        if not text or len(text) < 5:
            return None
        return text[:240]
    except httpx.ReadTimeout:
        log.warning("Proactive LLM timeout uid-for=%s", uname); return None
    except Exception as e:
        _log_exc("_generate_proactive_text", e); return None


async def _scan_once() -> None:
    """One pass of the proactive scanner: find candidates, queue nudges."""
    from main import get_users, load_state, get_user
    from app.memory import get_open_topics
    cfg = _cfg()
    log = _log()
    if not cfg.get("proactive_enabled", True):
        return

    idle_min   = int(cfg.get("proactive_idle_min_sec", 1800))
    idle_max   = int(cfg.get("proactive_idle_max_sec", 21600))
    cooldown   = int(cfg.get("proactive_cooldown_sec", 7200))
    min_imp    = float(cfg.get("proactive_min_importance", 0.5))
    max_queue  = int(cfg.get("proactive_max_queue", 3))
    now = time.time()

    try:
        users = get_users()
    except Exception as e:
        _log_exc("proactive get_users", e); return

    for u in users:
        uid = u["id"]
        # Cooldown: never nudge the same user twice within the cooldown window.
        last_sent = float(_PROACTIVE_LAST_SENT.get(uid, 0.0))
        if now - last_sent < cooldown:
            continue
        # Queue limit
        q = _get_queue(uid)
        if len(q) >= max_queue:
            continue

        try:
            state = load_state(uid)
        except Exception:
            continue
        last_act = int(state.get("last_activity_ts") or 0)
        if last_act <= 0:
            continue
        idle = now - last_act
        if idle < idle_min or idle > idle_max:
            continue

        # Need at least one open topic above threshold
        try:
            topics = get_open_topics(uid, 5)
        except Exception:
            continue
        candidate = None
        for t in topics:
            if float(t.get("importance", 0.0)) >= min_imp:
                candidate = t; break
        if not candidate:
            continue

        uname = (get_user(uid) or {}).get("name", "хозяин")
        gap_h = max(0.5, idle / 3600.0)
        text = await _generate_proactive_text(uname, candidate["topic"],
                                              candidate.get("context", ""), gap_h)
        if not text:
            continue
        q.append({"text": text, "ts": int(now),
                  "topic_id": int(candidate["id"]),
                  "topic": candidate["topic"][:120]})
        _PROACTIVE_LAST_SENT[uid] = now
        log.info("Proactive queued uid=%s topic_id=%s", uid, candidate["id"])


async def _proactive_loop() -> None:
    """Long-running scanner; sleeps proactive_scan_interval_sec between passes."""
    log = _log()
    log.info("Proactive loop started")
    while True:
        try:
            await _scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("proactive loop", e)
        interval = max(60, int(_cfg().get("proactive_scan_interval_sec", 300)))
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  DIARY: first-person end-of-day entry
# ─────────────────────────────────────────────────────────────────────────────
def get_diary(uid: str, day: Optional[str] = None) -> dict:
    """Fetch the diary entry for `day` (YYYY-MM-DD). Defaults to today.
    Returns {} if no entry exists."""
    d = day or datetime.now().strftime("%Y-%m-%d")
    try:
        with db() as c:
            row = c.execute(
                "SELECT day, entry, ts FROM diary_entries WHERE user_id=? AND day=?",
                (uid, d)).fetchone()
        return dict(row) if row else {}
    except Exception as e:
        _log_exc("get_diary", e); return {}


def list_diary_days(uid: str, limit: int = 30) -> list[dict]:
    """List of {day, ts, preview} for diary browsing UI."""
    try:
        with db() as c:
            rows = c.execute(
                "SELECT day, ts, substr(entry,1,160) AS preview "
                "FROM diary_entries WHERE user_id=? ORDER BY day DESC LIMIT ?",
                (uid, int(limit))).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        _log_exc("list_diary_days", e); return []


def _save_diary(uid: str, day: str, entry: str, metadata: dict = None) -> None:
    """Save diary entry with optional YAML frontmatter (Kuni style)."""
    try:
        # If metadata provided, wrap entry in YAML frontmatter
        if metadata:
            import yaml
            yaml_frontmatter = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)
            full_entry = f"---\n{yaml_frontmatter}---\n{entry}"
        else:
            full_entry = entry
            
        with db() as c:
            c.execute(
                "INSERT INTO diary_entries(user_id,day,entry) VALUES(?,?,?) "
                "ON CONFLICT(user_id,day) DO UPDATE SET entry=excluded.entry, ts=unixepoch()",
                (uid, day, full_entry[:1500]))
    except Exception as e:
        _log_exc("_save_diary", e)


async def _write_diary(uid: str, day: str) -> Optional[str]:
    """Generate and save a diary entry for a specific day.
    Pulls all `day` messages. Returns the written text or None."""
    from main import _get_http_client, _llm_url, _clean, get_user
    log = _log()
    cfg = _cfg()

    # Pull all messages on that date (day is local)
    try:
        day_start = datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return None
    start_ts = int(day_start.timestamp())
    end_ts = int((day_start + timedelta(days=1)).timestamp())

    try:
        with db() as c:
            rows = c.execute(
                "SELECT role, content FROM memory "
                "WHERE user_id=? AND turn_status='completed' "
                "AND ts>=? AND ts<? ORDER BY id ASC LIMIT 180",
                (uid, start_ts, end_ts)).fetchall()
    except Exception as e:
        _log_exc("_write_diary fetch", e); return None

    if len(rows) < int(cfg.get("diary_min_messages", 6)):
        return None

    uname = (get_user(uid) or {}).get("name", "хозяин")
    convo = "\n".join(
        f"{uname if r['role']=='user' else 'Мэйд'}: {r['content'][:260]}"
        for r in rows[-100:])

    prompt = (
        f"Ты — Мэйд. Напиши дневниковую запись за {day} от ПЕРВОГО ЛИЦА — "
        "как ты прожила этот день рядом с хозяином. "
        "Это не сводка и не список. Это твоё внутреннее, тихое, честное.\n"
        "• Пиши в женском роде, своими словами, тепло и без штампов.\n"
        "• Упомяни то, что тронуло, задело, показалось важным.\n"
        "• Можно одно чувство, одну мысль, одно желание.\n"
        "• 80-140 слов, одним-двумя абзацами. Без списков и заголовков.\n"
        "• Никаких <think> или рассуждений — только сам дневник.\n/no_think\n\n"
        f"Разговоры этого дня:\n{convo}\n\nДневник:"
    )

    try:
        client = await _get_http_client()
        r = await client.post(
            f"{_llm_url()}/v1/chat/completions",
            json={"model": "qwen3",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": float(cfg.get("diary_temperature", 0.55)),
                  "max_tokens": int(cfg.get("diary_max_tokens", 320)),
                  "stream": False},
            timeout=120.0)
        if r.status_code != 200:
            log.warning("Diary LLM %d uid=%s", r.status_code, uid); return None
        text = _clean(r.json()["choices"][0]["message"]["content"]).strip()
        if not text or len(text) < 40:
            return None
        
        # Build metadata for YAML frontmatter (Kuni style)
        from main import get_user_state
        state = get_user_state(uid) or {}
        metadata = {
            "timestamp": int(datetime.now().timestamp()),
            "humanity_level": state.get("humanity_level", 0.0),
            "software_version": state.get("software_version", "1.0"),
            "tags": ["daily_reflection"],
            "confidence": 0.85
        }
        
        _save_diary(uid, day, text, metadata)
        log.info("Diary written uid=%s day=%s chars=%d", uid, day, len(text))
        return text
    except httpx.ReadTimeout:
        log.warning("Diary timeout uid=%s day=%s", uid, day); return None
    except Exception as e:
        _log_exc("_write_diary", e); return None


async def _diary_scan_once() -> None:
    """Check each user: if yesterday's diary is missing and there was activity,
    write it. Runs relatively lazily — the loop polls every ~30 min."""
    from main import get_users
    cfg = _cfg()
    if not cfg.get("diary_enabled", True):
        return
    # Write for YESTERDAY (local): a diary is a retrospective, it needs the
    # day to be complete. This also avoids rewriting while the user is still
    # chatting.
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        users = get_users()
    except Exception as e:
        _log_exc("diary get_users", e); return
    for u in users:
        uid = u["id"]
        existing = get_diary(uid, yesterday)
        if existing:
            continue
        try:
            await _write_diary(uid, yesterday)
        except Exception as e:
            _log_exc(f"diary _write uid={uid}", e)


async def _diary_loop() -> None:
    log = _log()
    log.info("Diary loop started")
    while True:
        try:
            await _diary_scan_once()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log_exc("diary loop", e)
        interval = max(300, int(_cfg().get("diary_check_interval_sec", 1800)))
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


# ─────────────────────────────────────────────────────────────────────────────
#  LIFECYCLE: spawn both loops once on FastAPI startup
# ─────────────────────────────────────────────────────────────────────────────
def start_autonomous_loops() -> None:
    """Idempotent. Registers the long-running tasks with main._track()."""
    from main import _track
    global _LOOPS_STARTED
    if _LOOPS_STARTED:
        return
    _LOOPS_STARTED = True
    log = _log()
    try:
        t1 = asyncio.create_task(_proactive_loop())
        _track(t1)
        log.info("Autonomous: proactive loop task scheduled")
    except Exception as e:
        _log_exc("start_autonomous_loops proactive", e)
    try:
        t2 = asyncio.create_task(_diary_loop())
        _track(t2)
        log.info("Autonomous: diary loop task scheduled")
    except Exception as e:
        _log_exc("start_autonomous_loops diary", e)
