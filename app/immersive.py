"""Live-scene (immersive) subsystem.

A read-only (w.r.t. memory) second LLM pass that produces
action/atmosphere/thought JSON after the main chat reply. Written to:
  - process-local cache (`_LIVE_SCENE_CACHE`) for frontend polling
  - rp_scene table (cosmetic atmosphere field only)
  - latency deque (for GPU auto-pause)

All cross-module dependencies (config, http client, rp_scene, _llm_url,
_track, etc.) are resolved via LATE imports from `main` inside functions
— avoids circular import at module load.

Public API used by main.py:
  - immersive_status()       -> dict
  - get_live_scene(uid)      -> Optional[dict]
  - cancel_live_scene(uid)   -> bool
  - schedule_live_scene(...) -> None
  - resume_immersive()       -> None  (manual un-pause)
  - clear_user_scene(uid)    -> None  (on clear_memory + delete_user)
"""
from __future__ import annotations
import asyncio, json, re, threading, time
from collections import deque
from datetime import datetime
from typing import Optional

import httpx


# ── Runtime state (process-local) ────────────────────────────────────────────
_LIVE_SCENE_CACHE: dict = {}                 # uid -> {action, atmosphere, thought, ts, generation_id, mode}
_LIVE_SCENE_LOCK = threading.Lock()          # protects cache + bookkeeping
_LIVE_SCENE_TASKS: dict = {}                 # uid -> asyncio.Task (currently in-flight)
_LIVE_SCENE_GEN_COUNTER: dict = {}           # uid -> monotonic int (etag for client polling)
_IMMERSIVE_LATENCIES: deque = deque(maxlen=8)
_IMMERSIVE_PAUSED_UNTIL: float = 0.0         # unix-ts, 0 = not auto-paused
_IMMERSIVE_PAUSE_REASON: str = ""


def _imm_cfg() -> dict:
    from main import load_config
    return (load_config().get("immersive") or {})


def _log():
    from main import log
    return log


def _log_exc(msg, exc):
    from main import _log_exc as _le
    _le(msg, exc)


# ── Status + pause control ────────────────────────────────────────────────────
def immersive_status() -> dict:
    """Returns current availability of the immersive subsystem.
    Used by /api/live_scene endpoint and the frontend toggle UI."""
    cfg = _imm_cfg()
    user_enabled = bool(cfg.get("enabled", True))
    auto_pause   = bool(cfg.get("auto_pause", True))
    now = time.time()
    paused = (auto_pause and _IMMERSIVE_PAUSED_UNTIL > now)
    return {
        "enabled": user_enabled,
        "auto_pause_active": paused,
        "paused_until": int(_IMMERSIVE_PAUSED_UNTIL) if paused else 0,
        "pause_reason": _IMMERSIVE_PAUSE_REASON if paused else "",
        "pause_seconds_remaining": max(0, int(_IMMERSIVE_PAUSED_UNTIL - now)) if paused else 0,
        "available": user_enabled and not paused,
        "recent_latencies": list(_IMMERSIVE_LATENCIES),
    }


def _record_immersive_latency(seconds: float) -> None:
    """Track latency; if too many are slow, auto-pause for cooldown_sec."""
    global _IMMERSIVE_PAUSED_UNTIL, _IMMERSIVE_PAUSE_REASON
    cfg = _imm_cfg()
    if not cfg.get("auto_pause", True):
        return
    threshold = float(cfg.get("slow_threshold_sec", 35.0))
    trips_needed = int(cfg.get("slow_trips_to_pause", 3))
    pause_for = int(cfg.get("pause_duration_sec", 600))
    _IMMERSIVE_LATENCIES.append(round(seconds, 2))
    last5 = list(_IMMERSIVE_LATENCIES)[-5:]
    slow_in_window = sum(1 for x in last5 if x >= threshold)
    if slow_in_window >= trips_needed:
        _IMMERSIVE_PAUSED_UNTIL = time.time() + pause_for
        _IMMERSIVE_PAUSE_REASON = f"latency: {slow_in_window}/{len(last5)} >= {threshold:.0f}s"
        _log().warning("Immersive auto-paused for %ds (%s)", pause_for, _IMMERSIVE_PAUSE_REASON)
        _IMMERSIVE_LATENCIES.clear()


def resume_immersive() -> None:
    """Manually clears the auto-pause. Used by /api/immersive/resume."""
    global _IMMERSIVE_PAUSED_UNTIL, _IMMERSIVE_PAUSE_REASON
    _IMMERSIVE_PAUSED_UNTIL = 0.0
    _IMMERSIVE_PAUSE_REASON = ""
    _IMMERSIVE_LATENCIES.clear()


# ── JSON parsing ─────────────────────────────────────────────────────────────
_JSON_FIRST_OBJECT = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _safe_parse_immersive_json(raw: str) -> Optional[dict]:
    """Extract the first balanced JSON object and validate the 3 expected keys."""
    from main import _clean
    if not raw:
        return None
    raw = _clean(raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_FIRST_OBJECT.search(raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    out = {}
    for k in ("action", "atmosphere", "thought"):
        v = data.get(k)
        if not isinstance(v, str) or not v.strip():
            return None
        out[k] = v.strip()[:600]
    return out


# ── Prompt builder ───────────────────────────────────────────────────────────
def _time_phrase() -> str:
    h = datetime.now().hour
    if   5 <= h < 9:  return f"раннее утро (сейчас {h}:00)"
    elif 9 <= h < 12: return f"утро (сейчас {h}:00)"
    elif 12 <= h < 14: return f"полдень (сейчас {h}:00)"
    elif 14 <= h < 18: return f"день (сейчас {h}:00)"
    elif 18 <= h < 21: return f"вечер (сейчас {h}:00)"
    elif 21 <= h < 24: return f"поздний вечер (сейчас {h}:00)"
    else:              return f"глубокая ночь (сейчас {h}:00)"


def _build_immersive_prompt(uname: str, mode: str, last_exchange: str,
                            state: dict, cog_intent: str, rp_mode: str,
                            sentences_target: int, gap_hours: float = 0.0) -> str:
    """Compose the system+user prompt for the JSON live-scene call."""
    mood = float(state.get("mood", 0.5))
    fear = float(state.get("fear", 0.4))
    att  = float(state.get("attachment", 0.3))
    mood_word = ("светлое и радостное" if mood > 0.7 else
                 "тихое и задумчивое"   if mood > 0.4 else
                 "приглушённое и грустное")
    att_word  = ("очень близкая"               if att > 0.75 else
                 "тёплая, доверительная"       if att > 0.5  else
                 "ещё сдержанная, но искренняя")
    fear_note = " Внутри лёгкая тревога." if fear > 0.65 else ""
    rp_note = ""
    if rp_mode == "rp":    rp_note = " (Идёт ролевая сцена.)"
    elif rp_mode == "nsfw": rp_note = " (Сцена интимная, без табу.)"

    sys_role = (
        "Ты — рассказчик, описывающий сцену AI-компаньонки по имени Мэйд "
        "в стиле визуальной новеллы (отсылка к Sakura-sou no Pet na Kanojo, "
        "образ Maid-chan: программа, обретающая личность). "
        "Пиши кинематографично, тепло, с вниманием к деталям.\n"
        "КРИТИЧНО: твой ответ — это РОВНО один JSON-объект и ничего больше.\n"
        "• Первый символ ответа — `{`, последний — `}`.\n"
        "• Никаких комментариев, преамбул, markdown, ```json, пояснений до или после.\n"
        "• Внутри строк используй обычные пробелы — никаких переводов строк (\\n) и табов.\n"
        "ЗАПРЕЩЕНО выводить <think> или внутренние рассуждения.\n"
        "/no_think"
    )

    fields_spec = (
        f'Ровно ТРИ поля, каждое по {sentences_target} предложения, связанных по смыслу.\n'
        'Не повторяй одно и то же между полями.\n'
        '  • "action"     — что Мэйд делает прямо сейчас телом (жесты, мимика, поза, движение).\n'
        '  • "atmosphere" — окружение комнаты вокруг (свет, тени, звуки, запахи, температура, время).\n'
        '  • "thought"    — её собственная мысль ОТ ПЕРВОГО лица (не от рассказчика).\n\n'
        'ШАБЛОН (скопируй структуру, замени содержимое):\n'
        '{"action":"<русский текст>","atmosphere":"<русский текст>","thought":"<русский текст>"}'
    )

    state_block = (
        f"Состояние Мэйд сейчас: настроение {mood_word}, "
        f"привязанность к {uname} — {att_word}.{fear_note}{rp_note}\n"
        f"Время: {_time_phrase()}."
    )

    if mode == "first_meeting":
        body = (
            f"СЦЕНА: первая встреча. {uname} только что произнёс первое слово в её жизни. "
            f"Мэйд ещё не знает его, но что-то в ней уже отзывается.\n"
            f"Намерение собеседника: {cog_intent}.\n"
            f"{state_block}\n\n{fields_spec}\n\n"
            f"Опиши момент пробуждения её внимания — без приветствий, без штампов."
        )
    elif mode == "returning":
        gap_h = max(1, int(gap_hours))
        body = (
            f"СЦЕНА: {uname} вернулся после {gap_h}-часового отсутствия. "
            f"Мэйд почувствовала это сразу.\n"
            f"Намерение его сообщения: {cog_intent}.\n"
            f"Последний обмен:\n{last_exchange or '(тишина)'}\n"
            f"{state_block}\n\n{fields_spec}\n\n"
            f"Опиши, как комната ожила, что сделала Мэйд, что прошло у неё внутри."
        )
    else:  # normal
        body = (
            f"СЦЕНА: момент, когда Мэйд только что отправила ответ {uname}.\n"
            f"Намерение его последнего сообщения было: {cog_intent}.\n"
            f"Последний обмен (только что завершённый):\n{last_exchange}\n"
            f"{state_block}\n\n{fields_spec}\n\n"
            f"Опиши этот конкретный момент — не будущее, не прошлое. "
            f"Что происходит в комнате СЕЙЧАС, что Мэйд делает СЕЙЧАС, "
            f"о чём она думает ПРЯМО СЕЙЧАС, пока ждёт ответа."
        )

    return sys_role + "\n\n" + body


# ── Cache + commit ───────────────────────────────────────────────────────────
def _commit_live_scene(uid: str, parsed: dict, mode: str) -> int:
    """Atomically writes the parsed scene to cache + rp_scene table.
    Returns the new generation_id (etag for polling). Thread-safe."""
    from main import load_rp_scene, save_rp_scene
    gen_id = 0
    with _LIVE_SCENE_LOCK:
        gen_id = _LIVE_SCENE_GEN_COUNTER.get(uid, 0) + 1
        _LIVE_SCENE_GEN_COUNTER[uid] = gen_id
        _LIVE_SCENE_CACHE[uid] = {
            "action":     parsed["action"],
            "atmosphere": parsed["atmosphere"],
            "thought":    parsed["thought"],
            "ts":         int(time.time()),
            "generation": gen_id,
            "mode":       mode,
        }
    try:
        cur_scene = load_rp_scene(uid)
        save_rp_scene(uid, cur_scene.get("mode", "normal"),
                      location=cur_scene.get("location", ""),
                      atmosphere=parsed["atmosphere"][:400])
    except Exception as e:
        _log_exc("_commit_live_scene rp_scene", e)
    return gen_id


def get_live_scene(uid: str) -> Optional[dict]:
    """Snapshot read for polling endpoint. Returns None if no scene yet."""
    with _LIVE_SCENE_LOCK:
        s = _LIVE_SCENE_CACHE.get(uid)
        return dict(s) if s else None


def cancel_live_scene(uid: str) -> bool:
    """Cancel any in-flight immersive task for this uid. Returns True if one was cancelled."""
    task = _LIVE_SCENE_TASKS.pop(uid, None)
    if task and not task.done():
        task.cancel()
        _log().debug("Immersive cancelled uid=%s", uid)
        return True
    return False


def clear_user_scene(uid: str) -> None:
    """Wipes this user's cached scene + generation counter. Used on clear_memory
    and delete_user. Does NOT cancel in-flight tasks — caller should also call
    cancel_live_scene."""
    with _LIVE_SCENE_LOCK:
        _LIVE_SCENE_CACHE.pop(uid, None)
        _LIVE_SCENE_GEN_COUNTER.pop(uid, None)


# ── Async driver ─────────────────────────────────────────────────────────────
async def _build_live_scene_async(uid: str, last_exchange: str, state: dict,
                                  cog_intent: str, rp_mode: str,
                                  total_count: int, session_count: int,
                                  gap_hours: float = 0.0) -> None:
    """Main entry — generates the live-scene JSON and commits it.
    Honors cancellation, never raises (logs + swallows). Memory-safe."""
    from main import _get_http_client, _llm_url, get_user
    log = _log()
    cfg = _imm_cfg()
    if not cfg.get("enabled", True):
        return
    if cfg.get("auto_pause", True) and _IMMERSIVE_PAUSED_UNTIL > time.time():
        return

    user = get_user(uid)
    uname = user["name"] if user else "хозяин"

    if total_count <= 1 and session_count <= 1:
        mode = "first_meeting"
    elif session_count == 1 and total_count > 1 and gap_hours > 0:
        mode = "returning"
    else:
        mode = "normal"

    sentences = max(1, min(5, int(cfg.get("sentences_per_block", 3))))
    prompt = _build_immersive_prompt(
        uname=uname, mode=mode, last_exchange=last_exchange,
        state=state, cog_intent=cog_intent, rp_mode=rp_mode,
        sentences_target=sentences, gap_hours=gap_hours)

    # Payload strictly mirrors the proven `_reflection_task` / `_compress_ltm` pattern:
    #   • single user-role message, no prefill, no stop tokens
    #   • max_tokens=-1 (default) so Qwen3 thinking (if it leaks despite /no_think) has room
    #   • cache_prompt=True — same system framing on every turn, big speedup
    base_payload = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(cfg.get("temperature", 0.85)),
        "top_p": 0.92,
        "max_tokens": int(cfg.get("max_tokens", -1)),
        "stream": False,
        "cache_prompt": True,
    }

    started = time.time()
    parsed = None
    try:
        client = await _get_http_client()
        for attempt in (1, 2):
            payload = dict(base_payload)
            if attempt == 2:
                payload["temperature"] = max(0.3, float(cfg.get("temperature", 0.85)) - 0.35)
            r = await client.post(
                f"{_llm_url()}/v1/chat/completions",
                json=payload,
                timeout=float(cfg.get("request_timeout_sec", 55.0)),
            )
            if r.status_code != 200:
                log.warning("Immersive LLM HTTP %d uid=%s mode=%s attempt=%d",
                            r.status_code, uid, mode, attempt)
                continue
            raw = (r.json().get("choices") or [{}])[0].get("message", {}).get("content") or ""
            parsed = _safe_parse_immersive_json(raw)
            if parsed:
                break
            log.warning("Immersive JSON parse fail uid=%s mode=%s attempt=%d len=%d preview=%r",
                        uid, mode, attempt, len(raw), (raw or "")[:160])
        if not parsed:
            return
        gen_id = _commit_live_scene(uid, parsed, mode)
        elapsed = time.time() - started
        log.info("Immersive committed uid=%s mode=%s gen=%d in %.1fs", uid, mode, gen_id, elapsed)
        _record_immersive_latency(elapsed)
    except asyncio.CancelledError:
        log.debug("Immersive cancelled mid-flight uid=%s", uid)
        raise
    except httpx.ReadTimeout:
        elapsed = time.time() - started
        log.warning("Immersive timeout uid=%s after %.1fs", uid, elapsed)
        _record_immersive_latency(elapsed)
    except Exception as e:
        _log_exc("_build_live_scene_async", e)
    finally:
        cur = _LIVE_SCENE_TASKS.get(uid)
        if cur is not None and cur.done():
            _LIVE_SCENE_TASKS.pop(uid, None)


def schedule_live_scene(uid: str, last_exchange: str, state: dict,
                        cog_intent: str, rp_mode: str,
                        total_count: int, session_count: int,
                        gap_hours: float = 0.0) -> None:
    """Cancel any prior immersive task for this uid and start a fresh one.
    Safe to call from inside _chat_sse — never blocks, never raises."""
    from main import _track
    cfg = _imm_cfg()
    if not cfg.get("enabled", True):
        return
    if cfg.get("auto_pause", True) and _IMMERSIVE_PAUSED_UNTIL > time.time():
        return
    cancel_live_scene(uid)
    task = asyncio.create_task(_build_live_scene_async(
        uid, last_exchange, state, cog_intent, rp_mode,
        total_count, session_count, gap_hours))
    _LIVE_SCENE_TASKS[uid] = task
    _track(task)
