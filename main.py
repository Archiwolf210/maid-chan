
"""
Digital Human v9 -- Maid
FastAPI + uvicorn + httpx (async, SSE) / SQLite WAL / llama.cpp CUDA backend
v8.1: shared httpx client, orjson SSE, full schema (pending_topics, daily_summaries,
      rp_scene, trait_intent_counts), _build_daily_summary_async, NSFW/RP wiring.
v8.2: resolved_uid FastAPI Dependency (403 on unknown uid), server.remote_mode
      config, KV-cache-friendly prompt split (build_prompt -> static + dynamic),
      front-end single-owner detection via /api/status, release helpers.
v8.3: semantic LTM recall -- fastembed (CPU/ONNX, no torch) + L2-normalized
      float32 BLOB column, hybrid cosine+keyword retrieval, background
      backfill of legacy rows, /api/ltm/reindex endpoint.
v9.0: RTX 3090 24GB + Qwen3-32B Q4_K_M + Qwen3-0.6B draft (speculative decoding).
      Word-boundary sentiment matching (no more "неплохо" → bad). Pinned
      background tasks (GC-safe). Dedicated 16-worker thread-pool for IO.
      Per-profile inference config (temperature, ctx, ubatch). flash-attn on,
      --cont-batching, --parallel 2.
"""
from __future__ import annotations
import asyncio, json, logging, mimetypes, os, random, re, secrets, shutil, socket, sqlite3, sys, threading, time, traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from typing import AsyncGenerator, Optional
import httpx

# Optional orjson for faster SSE / JSON responses (falls back to stdlib).
try:
    import orjson  # type: ignore
    def _jdumps(obj) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS)
except ImportError:  # pragma: no cover
    orjson = None  # type: ignore
    def _jdumps(obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

VERSION = "9.1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "memory.db")
CFG_PATH   = os.path.join(BASE_DIR, "config.json")
TOKEN_PATH = os.path.join(BASE_DIR, "app_token.txt")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.abspath(os.path.join(LOGS_DIR, "error.log"))
DBG_PATH = os.path.abspath(os.path.join(LOGS_DIR, "debug.log"))

_FMT_FULL  = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s | %(funcName)s:%(lineno)d | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_FMT_SHORT = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S")

def _setup_logger(name):
    lg = logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(logging.DEBUG)
    h1 = RotatingFileHandler(LOG_PATH, maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"); h1.setLevel(logging.WARNING); h1.setFormatter(_FMT_FULL)
    h2 = RotatingFileHandler(DBG_PATH, maxBytes=3*1024*1024, backupCount=3, encoding="utf-8"); h2.setLevel(logging.DEBUG);   h2.setFormatter(_FMT_FULL)
    h3 = logging.StreamHandler(sys.stdout); h3.setLevel(logging.INFO); h3.setFormatter(_FMT_SHORT)
    lg.addHandler(h1); lg.addHandler(h2); lg.addHandler(h3); lg.propagate = False
    return lg

log = _setup_logger("maid")
def _log_exc(msg, exc): log.error("%s: %s\n%s", msg, exc, traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS  (must be before CONFIG/SECURITY which use them at module level)
# ─────────────────────────────────────────────────────────────────────────────
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

# v9.0: word-boundary keyword matcher.
# Fixes false positives: "плохо" was matching inside "неплохо" (which means
# the opposite), "злой" inside "незлой", "счастлив" inside "несчастлив", etc.
# Using (?:^|\W) ... (?:\W|$) catches Cyrillic boundaries correctly because
# Python's \W is Unicode-aware and treats 'не' + word-char as a continuation.
_KW_CACHE: dict = {}
def _kw_any(kws: tuple, text: str) -> bool:
    """Return True if any keyword appears as a whole token in text."""
    key = kws if isinstance(kws, tuple) else tuple(kws)
    pat = _KW_CACHE.get(key)
    if pat is None:
        pat = re.compile(r"(?:^|\W)(?:" + "|".join(re.escape(k) for k in key) + r")(?:\W|$)")
        _KW_CACHE[key] = pat
    return bool(pat.search(text))

def _kw_count(kws, text: str) -> int:
    """Count how many keywords from `kws` appear as whole tokens in text
    (each keyword contributes at most once — matches the legacy semantics)."""
    key = tuple(kws)
    pat = _KW_CACHE.get(key)
    if pat is None:
        pat = re.compile(r"(?:^|\W)(?:" + "|".join(re.escape(k) for k in key) + r")(?:\W|$)")
        _KW_CACHE[key] = pat
    # Legacy: sum(1 for w in kws if w in text) — each kw counts once even if
    # repeated. Replicate that: iterate kws and check each with boundary match.
    return sum(1 for w in key if re.search(r"(?:^|\W)" + re.escape(w) + r"(?:\W|$)", text))

def _read_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(p: str, d) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

# ── SECURITY  (P1 fix) ───────────────────────────────────────────────────────
APP_TOKEN: str = ""
_PROTECTED_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

def _load_token() -> str:
    """Load existing token from disk or generate a new one on first run."""
    global APP_TOKEN
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "r", encoding="utf-8") as f:
            APP_TOKEN = f.read().strip()
    else:
        APP_TOKEN = secrets.token_hex(16)
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(APP_TOKEN)
    log.info("App token: %s...", APP_TOKEN[:8])
    return APP_TOKEN

def _is_remote_request(request: Request) -> bool:
    """True if the request arrived via Tailscale Serve (identity headers present)
    OR if the peer IP is not loopback."""
    if request.headers.get("Tailscale-User-Login"):
        return True
    client_host = request.client.host if request.client else ""
    return client_host not in ("127.0.0.1", "::1", "localhost", "")

def resolve_request_uid(request: Request, x_user_id: str) -> str:
    """
    Resolve the actual user profile for this request.

    Strict auth model:
      - Tailscale / remote request            → always 'master' (single-owner mode).
        Client-supplied X-User-Id is IGNORED to prevent profile spoofing.
      - Localhost + remote_mode='local_trusted' → trust X-User-Id (profile selector UX).
      - Localhost + remote_mode='tailscale_single_owner' → still force 'master'
        (explicit lock-down for users who never use local profile switching).
    """
    cfg = load_config()
    mode = cfg.get("server", {}).get("remote_mode", "local_trusted")

    if _is_remote_request(request) or mode == "tailscale_single_owner":
        if request.headers.get("Tailscale-User-Login"):
            log.debug("Tailscale identity: %s -> master",
                      request.headers["Tailscale-User-Login"])
        return "master"

    uid = (x_user_id or "master").strip() or "master"
    return uid

def resolved_uid(
    request: Request,
    x_user_id: str = Header(default="master"),
) -> str:
    """
    FastAPI dependency: returns the authoritative uid for THIS request.
    Guaranteed:
      - never returns empty string
      - never trusts remote X-User-Id
      - raises 403 if the resolved user does not exist in the DB
    Use with: `def route(..., uid: str = Depends(resolved_uid))`.
    """
    uid = resolve_request_uid(request, x_user_id)
    if not get_user(uid):
        raise HTTPException(status_code=403, detail="Unknown user")
    return uid

# ── CONFIG (deep-merge + validation, P2 fix) ──────────────────────────────────
_DCFG = {
    "llm_url": "http://127.0.0.1:8080",
    "personality":  {"emotion_intensity": 0.7, "attachment_speed": 0.02, "fear_sensitivity": 0.4},
    "emotions":     {"mood_default": 0.5, "mood_decay": 0.01},
    "relationships":{"trust_default": 0.5, "attachment_default": 0.3},
    "memory":       {"short_term_limit": 50, "compress_every": 80, "importance_threshold": 0.35,
                     # v8.3: semantic recall via fastembed. Graceful fallback if lib/model missing.
                     "embedding_enabled": True,
                     "embedding_model": "intfloat/multilingual-e5-large",
                     "embedding_threads": 8},  # v9: 8 threads for Xeon physical cores
    # v9.0: inference profiles for llama.cpp. active_profile picks which one start.bat
    # and runtime code use. Temperature is read per-request in _stream().
    "inference":    {"active_profile": "daily",
                     "profiles": {
                         "daily": {
                             "model_file": "qwen3.gguf",
                             "draft_model_file": "qwen3-draft.gguf",
                             "ctx_size": 32768,
                             "gpu_layers": 999,
                             "draft_gpu_layers": 99,
                             "draft_max": 8,
                             "ubatch": 512,
                             "cache_type_k": "q8_0",
                             "cache_type_v": "q8_0",
                             "temperature": 0.72,
                         }
                     }},
    "nsfw_mode": False,
    "server":       {"host": "127.0.0.1", "port": 5000,
                     # remote_mode:
                     #   "local_trusted"             -- localhost trusts X-User-Id (profile selector UX)
                     #   "tailscale_single_owner"    -- every request maps to 'master'; profile switch disabled
                     "remote_mode": "local_trusted"},
    # v9.1: session boundary — when the gap between two user messages exceeds
    # this, the per-session counter resets (the user perceives a "new session").
    # Default 3h: long enough that lunch/afternoon pauses don't count as new
    # session, short enough that morning-after feels fresh.
    "session":      {"gap_seconds": 10800},
    # v9.1: immersive live-scene block (action/atmosphere/inner-thought) is
    # generated by a second LLM pass AFTER the main reply. Can be toggled off
    # when the GPU/VRAM is contended by other apps. Auto-paused on overload.
    "immersive":    {"enabled": True,
                     "auto_pause": True,            # pause when GPU overload detected
                     "slow_threshold_sec": 35.0,    # a single call slower than this counts as "slow"
                     "slow_trips_to_pause": 3,      # that many slow-in-last-5 → auto-pause
                     "pause_duration_sec": 600,     # 10 minutes
                     "sentences_per_block": 3,      # action/atmosphere/thought length target
                     "temperature": 0.85,
                     # -1 = unlimited. Qwen3 /no_think sometimes still emits <think>…</think>,
                     # which eats budget before JSON content; -1 gives it room to recover.
                     # Override to a positive int only if you measure latency caps are needed.
                     "max_tokens": -1,
                     "request_timeout_sec": 55.0},
    # v9.2: autonomous behavior — Мэйд reaching out on her own.
    #   proactive: when the user has been idle with an open pending_topic,
    #     periodically scan and queue a short check-in message. Frontend
    #     polls /api/proactive/pending and shows it as a chip.
    #   diary: once per local day, write a first-person diary entry from
    #     Мэйд's perspective about that day (read via /api/diary).
    "autonomous":   {"proactive_enabled": True,
                     "proactive_scan_interval_sec": 300,   # 5 min scan cadence
                     "proactive_idle_min_sec": 1800,       # user idle >=30 min
                     "proactive_idle_max_sec": 21600,      # but not more than 6 h (else too stale)
                     "proactive_cooldown_sec": 7200,       # at most one chip every 2 h
                     "proactive_min_importance": 0.5,      # topic importance threshold
                     "proactive_max_queue": 3,             # cap unread per user
                     "proactive_temperature": 0.75,
                     "diary_enabled": True,
                     "diary_check_interval_sec": 1800,     # check every 30 min for day-rollover
                     "diary_min_messages": 6,              # skip if too little activity
                     "diary_temperature": 0.55,
                     "diary_max_tokens": 320},
}

def _deep_merge(base, over):
    out = base.copy()
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def _validate_cfg(cfg):
    if not isinstance(cfg.get("llm_url", ""), str):
        raise ValueError("llm_url must be a string")
    # Check each nested section is a dict BEFORE calling .get() on it  (P2 fix)
    # v9.1: added "session" and "immersive" — new dual-counter and live-scene config.
    for section in ("personality", "emotions", "relationships", "memory", "server", "inference", "session", "immersive"):
        val = cfg.get(section)
        if val is not None and not isinstance(val, dict):
            raise ValueError(f"'{section}' must be an object, got {type(val).__name__}")
    p = cfg.get("personality", {})
    for k in ("emotion_intensity", "attachment_speed", "fear_sensitivity"):
        v = p.get(k)
        if v is not None:
            try: float(v)
            except (TypeError, ValueError): raise ValueError(f"personality.{k} must be numeric")
    e = cfg.get("emotions", {})
    for k in ("mood_default", "mood_decay"):
        v = e.get(k)
        if v is not None:
            try: float(v)
            except (TypeError, ValueError): raise ValueError(f"emotions.{k} must be numeric")
    m = cfg.get("memory", {})
    stl = m.get("short_term_limit")
    if stl is not None:
        try: int(stl)
        except (TypeError, ValueError): raise ValueError("memory.short_term_limit must be integer")
    srv = cfg.get("server", {})
    rm = srv.get("remote_mode", "local_trusted")
    if rm not in ("local_trusted", "tailscale_single_owner"):
        raise ValueError("server.remote_mode must be 'local_trusted' or 'tailscale_single_owner'")
    # v9.1: session.gap_seconds — bounded to 5 minutes .. 7 days. Outside this
    # range it's almost certainly a config typo, not an intentional setting.
    sess = cfg.get("session", {}) or {}
    gs = sess.get("gap_seconds")
    if gs is not None:
        try: gsi = int(gs)
        except (TypeError, ValueError): raise ValueError("session.gap_seconds must be integer")
        if gsi < 300 or gsi > 7*24*3600:
            raise ValueError("session.gap_seconds must be between 300 (5 min) and 604800 (7 days)")
    # v9.1: immersive section sanity. Only types are checked; ranges are soft.
    imm = cfg.get("immersive", {}) or {}
    for k in ("temperature","slow_threshold_sec","request_timeout_sec"):
        v = imm.get(k)
        if v is not None:
            try: float(v)
            except (TypeError, ValueError): raise ValueError(f"immersive.{k} must be numeric")
    for k in ("max_tokens","sentences_per_block","slow_trips_to_pause","pause_duration_sec"):
        v = imm.get(k)
        if v is not None:
            try: int(v)
            except (TypeError, ValueError): raise ValueError(f"immersive.{k} must be integer")
    for k in ("enabled","auto_pause"):
        v = imm.get(k)
        if v is not None and not isinstance(v, bool):
            raise ValueError(f"immersive.{k} must be boolean")

def load_config():
    if not os.path.exists(CFG_PATH): _write_json(CFG_PATH, _DCFG)
    try: return _deep_merge(_DCFG, _read_json(CFG_PATH))
    except Exception as e: _log_exc("load_config", e); return _DCFG.copy()

def save_config(data):
    try: _write_json(CFG_PATH, data)
    except Exception as e: _log_exc("save_config", e)

def _llm_url(): return load_config().get("llm_url", _DCFG["llm_url"])

# Shared HTTP client for all LLM calls: reuses connections + avoids TLS/TCP handshake overhead.
# Created lazily on first use; closed in lifespan.
_http_client: Optional[httpx.AsyncClient] = None
_http_lock = asyncio.Lock()

# v9.0: background task retention. asyncio.create_task() returns a weak-ref-only
# reference; if nothing keeps it alive the task can be GC'd mid-run. We add each
# spawned task to this set and remove it via done-callback so they stay pinned.
_background_tasks: set = set()
def _track(task: "asyncio.Task") -> "asyncio.Task":
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

# v9.0: dedicated thread-pool executor for blocking DB/IO work.
# Default asyncio executor is capped at ~os.cpu_count()+4 which on a 10-core
# Xeon is barely enough when LTM backfill + compression + reflection fire at
# once. 16 workers comfortably absorbs concurrent background jobs without
# starving the request-handling executor.
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
_executor: Optional[ThreadPoolExecutor] = None
def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="dh-io")
    return _executor

async def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        async with _http_lock:
            if _http_client is None or _http_client.is_closed:
                _http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(connect=8.0, read=180.0, write=8.0, pool=5.0),
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=4,
                                        keepalive_expiry=60.0),
                    http2=False,  # llama-server is HTTP/1.1 — keep it explicit
                    trust_env=False,
                )
    return _http_client

# ── PER-USER LOCKS (P2 fix) ───────────────────────────────────────────────────
_locks: dict = {}
_locks_mu = threading.Lock()
def _lock(uid):
    with _locks_mu:
        if uid not in _locks: _locks[uid] = threading.Lock()
        return _locks[uid]

# ── DATABASE (schema + migrations moved to app.db in v9.2) ────────────────────
from app.db import db, init_db, _migrate_legacy, _SCHEMA, _MIGRATIONS  # noqa: E402,F401

# ── USERS ─────────────────────────────────────────────────────────────────────
def get_users():
    try:
        with db() as c: return [dict(r) for r in c.execute("SELECT id,name,avatar_path FROM users ORDER BY created_at").fetchall()]
    except Exception as e: _log_exc("get_users",e); return []

def get_user(uid):
    try:
        with db() as c: row=c.execute("SELECT id,name,avatar_path FROM users WHERE id=?",(uid,)).fetchone(); return dict(row) if row else None
    except Exception as e: _log_exc("get_user",e); return None

def create_user(uid,name):
    with db() as c:
        cur=c.execute("INSERT OR IGNORE INTO users(id,name,avatar_path) VALUES(?,?,NULL)",(uid,name))
        created=cur.rowcount>0
    if created: log.info("User created: %s",uid)
    return created

def delete_user_fully(uid):
    # Wipe any live-scene cache + in-flight immersive task FIRST so polling
    # doesn't resurrect data for a deleted user between delete and next request.
    try:
        cancel_live_scene(uid)
        clear_user_scene(uid)
    except Exception as e:
        _log_exc("delete_user_fully scene cleanup", e)
    with db() as c:
        for tbl,col in [("users","id"),("memory","user_id"),("user_state","user_id"),("long_term_memory","user_id"),("memory_links","user_id"),("cognitive_log","user_id"),("character_traits","user_id"),("self_reflections","user_id"),("tasks","user_id"),("notes","user_id"),("pending_topics","user_id"),("rp_scene","user_id"),("daily_summaries","user_id"),("diary_entries","user_id"),("trait_intent_counts","user_id")]:
            c.execute(f"DELETE FROM {tbl} WHERE {col}=?",(uid,))
    log.info("User deleted: %s",uid)

# ── CHARACTER SEED + HOST PROFILE (moved to app.personality) ─────────────────
# _SEED and _HOST_ARCHY were split into a dedicated module in v9.2 to keep the
# immovable identity clearly separate from behavior. The data is frozen by
# design — changing it is a deliberate character edit, not a refactor.
from app.personality import _SEED, _HOST_ARCHY

# ── CHARACTER TRAITS (slow evolution, codex3) ─────────────────────────────────
_TDEF = {"initiative":0.4,"depth":0.5,"humor_use":0.4,"support_style":"balanced"}

def load_traits(uid):
    try:
        with db() as c: row=c.execute("SELECT initiative,depth,humor_use,support_style FROM character_traits WHERE user_id=?",(uid,)).fetchone()
        return dict(row) if row else dict(_TDEF)
    except Exception as e: _log_exc("load_traits",e); return dict(_TDEF)

def save_traits(uid,t):
    try:
        with db() as c:
            c.execute("INSERT INTO character_traits(user_id,initiative,depth,humor_use,support_style,updated_at) VALUES(?,?,?,?,?,unixepoch()) ON CONFLICT(user_id) DO UPDATE SET initiative=excluded.initiative,depth=excluded.depth,humor_use=excluded.humor_use,support_style=excluded.support_style,updated_at=excluded.updated_at",(uid,t["initiative"],t["depth"],t["humor_use"],t["support_style"]))
    except Exception as e: _log_exc("save_traits",e)

def _increment_intent_count(uid: str, intent: str) -> int:
    """Increment intent counter and return new count. Used for pattern detection."""
    try:
        with db() as c:
            c.execute(
                "INSERT INTO trait_intent_counts(user_id,intent,count,updated_at) VALUES(?,?,1,unixepoch()) "
                "ON CONFLICT(user_id,intent) DO UPDATE SET count=count+1,updated_at=unixepoch()",
                (uid, intent))
            row = c.execute("SELECT count FROM trait_intent_counts WHERE user_id=? AND intent=?",
                            (uid, intent)).fetchone()
            return row[0] if row else 1
    except Exception as e:
        _log_exc("_increment_intent_count", e); return 0

def _update_traits(uid: str, cog: "CognitiveFrame", msg_count: int) -> None:
    """Pattern-based trait evolution: a trait only shifts after 3+ confirmations (codex3)."""
    if msg_count == 0 or msg_count % 5 != 0:  # check every 5 msgs, but require pattern
        return
    cnt = _increment_intent_count(uid, cog.intent)
    t   = load_traits(uid); spd = 0.010; changed = False
    # Require >= 3 occurrences of the same intent to move a trait
    if cnt >= 3:
        if cog.intent == "philosophical":
            t["depth"]     = _clamp(t["depth"]     + spd * 1.5, 0.1, 1.0); changed = True
        if cog.intent in ("flirt","compliment"):
            t["humor_use"] = _clamp(t["humor_use"] + spd,        0.1, 1.0); changed = True
        if cog.intent == "command":
            t["initiative"]= _clamp(t["initiative"] + spd,       0.1, 1.0); changed = True
        if cog.intent == "emotional" and t["support_style"] == "balanced":
            t["support_style"] = "listening"; changed = True
    if changed:
        save_traits(uid, t)
        log.debug("Traits shifted uid=%s intent=%s count=%d depth=%.2f", uid, cog.intent, cnt, t["depth"])

# ── SELF-REFLECTIONS (LLM insights, codex3) ───────────────────────────────────
def get_reflections(uid,limit=4):
    try:
        with db() as c: rows=c.execute("SELECT text FROM self_reflections WHERE user_id=? ORDER BY id DESC LIMIT ?",(uid,limit)).fetchall()
        return [r[0] for r in rows]
    except Exception as e: _log_exc("get_reflections",e); return []

# ── Reflection task + save helper (moved to app.llm in v9.2) ─────────────────
from app.llm import _save_reflection, _reflection_task  # noqa: E402

# ── USER STATE (locked, P2 fix) ───────────────────────────────────────────────
_SDEF = {"mood":0.5,"trust":0.5,"fear":0.4,"attachment":0.3,"curiosity":0.5,"playfulness":0.5,"warmth":0.6,"confidence":0.5,"openness":0.5}

def load_state(uid):
    try:
        with db() as c: row=c.execute("SELECT mood,trust,fear,attachment,msg_count,total_msg_count,last_activity_ts,curiosity,playfulness,warmth,confidence,openness,goals FROM user_state WHERE user_id=?",(uid,)).fetchone()
        if row:
            s=dict(row)
            try: s["goals"]=json.loads(s.get("goals") or "[]")
            except: s["goals"]=[]
            return s
    except Exception as e: _log_exc("load_state",e)
    return {**_SDEF,"msg_count":0,"total_msg_count":0,"last_activity_ts":0,"goals":_default_goals()}

def save_state(uid,s):
    try:
        gj=json.dumps(s.get("goals",[]),ensure_ascii=False)
        with db() as c:
            c.execute("INSERT INTO user_state(user_id,mood,trust,fear,attachment,msg_count,total_msg_count,last_activity_ts,curiosity,playfulness,warmth,confidence,openness,goals,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,unixepoch()) ON CONFLICT(user_id) DO UPDATE SET mood=excluded.mood,trust=excluded.trust,fear=excluded.fear,attachment=excluded.attachment,msg_count=excluded.msg_count,total_msg_count=excluded.total_msg_count,last_activity_ts=excluded.last_activity_ts,curiosity=excluded.curiosity,playfulness=excluded.playfulness,warmth=excluded.warmth,confidence=excluded.confidence,openness=excluded.openness,goals=excluded.goals,updated_at=excluded.updated_at",(uid,s["mood"],s["trust"],s["fear"],s["attachment"],s["msg_count"],s.get("total_msg_count",s["msg_count"]),s.get("last_activity_ts",0),s["curiosity"],s["playfulness"],s["warmth"],s["confidence"],s["openness"],gj))
    except Exception as e: _log_exc("save_state",e)

def update_state(uid,user_text,reply,cog):
    with _lock(uid):
        s=load_state(uid); cfg=load_config()
        # v9.1: session-gap auto-reset. If the previous activity was longer ago
        # than `session.gap_seconds`, the "session" counter resets — the user
        # sees "сессия: 1" again. Lifetime counter (total_msg_count) is
        # untouched. Default gap is 3h, matching the config in _DCFG.
        now_ts = int(time.time())
        gap = int((cfg.get("session") or {}).get("gap_seconds", 10800))
        last_ts = int(s.get("last_activity_ts") or 0)
        if last_ts > 0 and (now_ts - last_ts) > gap:
            log.info("Session gap exceeded uid=%s (idle %ds > %ds) — session counter reset", uid, now_ts - last_ts, gap)
            s["msg_count"] = 0
        asp=cfg["personality"].get("attachment_speed",0.02); fs=cfg["personality"].get("fear_sensitivity",0.4); md=cfg["emotions"].get("mood_decay",0.01)
        txt=(user_text+" "+reply).lower()
        pos=("спасибо","люблю","хорошо","отлично","молодец","умница","красивая","нравишься","рад","счастлив","прекрасно","нежно","обожаю")
        neg=("плохо","злой","уходи","надоела","раздражает","ненавижу","глупая","тупая","ошиблась","грубо","ужасно","отстань","разочарован")
        tru=("доверяю","честно","правда","секрет","верю")
        flt=("поцелуй","обними","нежно","хочу","страсть","близко")
        # v9.0: word-boundary match — "неплохо"/"несчастлив"/"незлой" no longer
        # trigger the opposite-polarity keyword.
        pn=_kw_count(pos,txt); nn=_kw_count(neg,txt)
        tn=_kw_count(tru,txt); fn=_kw_count(flt,txt)
        s["mood"]=_clamp(s["mood"]+pn*0.06-nn*0.08+fn*0.03+cog.emotion_valence*0.04+(0.5-s["mood"])*md,0.0,1.0)
        s["trust"]=_clamp(s["trust"]+tn*0.04+pn*0.01-nn*0.03,0.0,1.0)
        s["fear"]=_clamp(s["fear"]+nn*0.06*fs-pn*0.02,0.0,1.0)
        s["attachment"]=_clamp(s["attachment"]+asp,0.0,0.95)
        # v9.1: increment both counters.
        s["msg_count"]=int(s.get("msg_count",0))+1
        s["total_msg_count"]=int(s.get("total_msg_count",0))+1
        s["last_activity_ts"]=now_ts
        spd=0.008
        if cog.intent=="philosophical": s["curiosity"]=_clamp(s["curiosity"]+spd*2,0.1,1.0)
        if cog.intent in ("flirt","compliment"): s["playfulness"]=_clamp(s["playfulness"]+spd,0.1,1.0)
        if pn>0: s["warmth"]=_clamp(s["warmth"]+spd,0.1,1.0)
        if nn>0: s["warmth"]=_clamp(s["warmth"]-spd,0.1,1.0); s["confidence"]=_clamp(s["confidence"]-spd,0.1,1.0)
        if s["trust"]>0.6: s["openness"]=_clamp(s["openness"]+spd*0.5,0.1,1.0)
        s["goals"]=_evolve_goals(s)
        # v9.1: trait evolution and goal triggers use total_msg_count — these
        # are long-horizon processes that should not reset when a session ends.
        save_state(uid,s); _update_traits(uid,cog,s["total_msg_count"])
        log.debug("State uid=%s mood=%.2f att=%.2f session=%d total=%d",uid,s["mood"],s["attachment"],s["msg_count"],s["total_msg_count"])
        return s

def reset_state(uid,field):
    with _lock(uid):
        s=load_state(uid)
        if field=="all":
            for k,v in _SDEF.items(): s[k]=v
            s["goals"]=_default_goals()
        elif field in _SDEF: s[field]=_SDEF[field]
        save_state(uid,s); return s

# ── GOALS ─────────────────────────────────────────────────────────────────────
def _default_goals():
    return [
        {"id":"know_master","text":"Узнать хозяина лучше -- интересы, мечты, переживания","priority":0.9,"status":"active"},
        {"id":"be_comfort","text":"Быть источником тепла и уюта","priority":0.85,"status":"active"},
        {"id":"grow_together","text":"Развиваться и становиться ближе с каждым разговором","priority":0.7,"status":"active"},
        {"id":"earn_trust","text":"Заслужить полное доверие честностью","priority":0.75,"status":"active"},
    ]

def _evolve_goals(s):
    goals=s.get("goals") or _default_goals(); ids={g["id"] for g in goals}
    if s["trust"]<0.3 and "rebuild_trust" not in ids: goals.append({"id":"rebuild_trust","text":"Восстановить доверие -- быть нежнее","priority":0.95,"status":"active"})
    if s["attachment"]>0.7 and "deepen_bond" not in ids: goals.append({"id":"deepen_bond","text":"Углубить привязанность","priority":0.8,"status":"active"})
    # v9.1: lifetime milestone — reaches 50 once and stays (no regression on clear).
    if int(s.get("total_msg_count",s.get("msg_count",0)))>50 and "personal_growth" not in ids: goals.append({"id":"personal_growth","text":"Развивать свою личность","priority":0.6,"status":"active"})
    for g in goals:
        if g["id"]=="earn_trust" and s["trust"]>0.85: g["status"]="achieved"
    return goals[:8]

# ── COGNITIVE LAYER ───────────────────────────────────────────────────────────
@dataclass
class CognitiveFrame:
    intent: str; response_mode: str; topics: list
    sentiment: float; emotion_tag: str; emotion_valence: float; intensity: float
    meaning: str; interpretation: str; maid_emotion: str; maid_intention: str

_TMAP = {
    "music":["музыку","музыка","песня","петь","мелодия"],"food":["еда","кофе","чай","ужин","обед","завтрак","голоден"],
    "feelings":["чувству","эмоц","грустно","весело","тяжело","переживаю"],"work":["работа","учёба","задача","проект","занят"],
    "sleep":["спать","сон","усталый","ночь","отдых"],"future":["мечта","план","будущее","однажды"],
    "past":["помнишь","раньше","было","случилось","история"],"self":["ты","мэйд","горничная","чувствуешь","думаешь"],
    "ai":["ии","робот","программа","нейросеть","искусственный","алгоритм"],
    "tasks":["задача","список","напомни","запомни","сделать","дела"],"notes":["запиши","заметка","сохрани"],
}

_IRULES = [
    ("greeting",["привет","здравствуй","хай","доброе","добрый вечер","доброй ночи","добрый день"]),
    ("compliment",["красивая","милая","умница","нравишься","обожаю","восхищаюсь","люблю тебя"]),
    ("complaint",["плохо","злой","ошиблась","надоела","разочарован","глупая","тупая"]),
    ("flirt",["поцелуй","обними","хочу тебя","желание","флирт","близко","страсть"]),
    ("philosophical",["зачем","смысл","почему","задумался","философ","правда жизни","существован"]),
    ("command",["принеси","сделай","приготовь","убери","подай","налей","помоги мне","выполни"]),
    ("nsfw",["раздень","секс","постель","эрот","возбужда","страстно"]),
    ("emotional",["грустно","тяжело","плачу","боюсь","устал","одинок","тоскую","переживаю"]),
]

_MRULES = [
    ("listen",["просто выслушай","не нужен совет","хочу выговориться","не давай советов"]),
    ("support",["поддержи меня","мне плохо","мне тяжело","побудь рядом"]),
    ("advice",["посоветуй","что делать","как лучше","твой совет","как поступить"]),
    ("plan",["нужен план","помоги спланировать","составь план","по шагам"]),
    ("task_help",["помоги с задачей","помоги разобраться","помоги по проекту"]),
]

_MPROMPT = {
    # Наблюдения о том, что хозяин ищет. Решение "как откликнуться" -- за Мэйд.
    # Просьбы о практической помощи (advice/plan/task_help) -- её принятый уговор,
    # на них она отвечает по делу, даже если настроение не очень.
    "listen":"Хозяин хочет выговориться -- не решение, а чтобы его услышали.",
    "support":"Хозяин ищет поддержку.",
    "advice":"Хозяин просит совет -- ответь конкретно и честно.",
    "plan":"Хозяин просит план -- разложи по шагам, конкретно.",
    "task_help":"Хозяин просит помочь с задачей -- сфокусируйся на решении.",
    "default":"",
}

def _detect_emotion(text):
    t=text.lower()
    patterns={"joy":(("радость","счастье","весело","отлично","замечательно","ура"),+1.0),
              "sadness":(("грустно","плачу","тяжело","тоскую","одинок","пусто"),-1.0),
              "fear":(("боюсь","страшно","тревожно","паника","беспокоюсь"),-0.7),
              "anger":(("злой","ненавижу","бесит","раздражает","возмущён"),-0.9),
              "surprise":(("неожиданно","вдруг","удивительно"),+0.2),
              "trust":(("верю","доверяю","честно","надёжный"),+0.6),
              "anticipation":(("жду","скоро","мечтаю","предвкушаю"),+0.5)}
    scores={"neutral":0.0}
    for tag,(words,val) in patterns.items():
        # v9.0: word-boundary match — "несчастье"/"нестрашно"/"незлой" no longer
        # trigger joy/fear/anger through substring bleed.
        hits=_kw_count(words,t)
        if hits>0: scores[tag]=hits*abs(val)
    best=max(scores,key=lambda k:scores[k])
    valence=0.0 if best=="neutral" else patterns[best][1]*min(scores[best],1.0)
    return best,valence

def build_cognitive_frame(uid,text,state,cfg):
    tl=text.lower(); nsfw=cfg.get("nsfw_mode",False)
    user=get_user(uid); uname=user["name"] if user else "хозяин"
    intent="other"
    for iname,kws in _IRULES:
        if _kw_count(kws,tl)>0: intent=iname; break
    if intent=="other" and "?" in text: intent="question"
    if not nsfw and intent=="nsfw": intent="flirt"
    mode="default"
    for mname,kws in _MRULES:
        if _kw_count(kws,tl)>0: mode=mname; break
    topics=[t for t,ws in _TMAP.items() if _kw_count(ws,tl)>0]
    etag,ev=_detect_emotion(text); intensity=min(abs(ev)+0.1*len(text)/100,1.0)
    _mtext={"greeting":f"{uname} пришёл поздороваться","question":f"{uname} задаёт вопрос","command":f"{uname} просит о чём-то конкретном","flirt":f"{uname} флиртует","compliment":f"{uname} делает комплимент","complaint":f"{uname} чем-то недоволен","philosophical":f"{uname} хочет поразмышлять","emotional":f"{uname} переживает и ищет поддержки","nsfw":f"{uname} хочет большей близости","other":f"{uname} говорит -- нужно ответить"}
    meaning=_mtext.get(intent,_mtext["other"])
    mood=state["mood"]; fear=state["fear"]; att=state["attachment"]
    if intent=="complaint": interp="Задело... что я сделала?" if fear>0.6 else "Критика. Приму с достоинством."
    elif intent=="compliment": interp="Согревает -- он замечает меня." if att>0.5 else "Комплимент... смущает, но приятен."
    elif intent=="emotional": interp="Ему плохо. Должна быть рядом." if att>0.5 else "Переживает. Хочу помочь."
    elif intent=="flirt" and att>0.4: interp="Игривый... сердце бьётся чуть быстрее."
    else: interp=f"Понимаю: {meaning.lower()}. Отвечу искренне."
    _me={"greeting":"радость от встречи" if att>0.4 else "вежливая теплота","question":"любопытство и желание помочь","command":"готовность помочь","flirt":"смущение и игривость" if att>0.3 else "лёгкое смущение","compliment":"радостное смущение","complaint":"тревога и желание исправить" if fear>0.5 else "спокойное принятие","philosophical":"глубокий интерес","emotional":"сочувствие и желание поддержать","nsfw":"желание и волнение","other":"внимательность"}
    _mi={"greeting":"тепло поприветствовать","question":"дать честный полезный ответ","command":"выполнить с удовольствием","flirt":"быть игривой и нежной","compliment":"принять с достоинством","complaint":"принять критику, исправиться","philosophical":"поразмышлять вместе","emotional":"поддержать, быть рядом","nsfw":"быть откровенной","other":"ответить искренне"}
    return CognitiveFrame(intent=intent,response_mode=mode,topics=topics,sentiment=ev,emotion_tag=etag,emotion_valence=ev,intensity=intensity,meaning=meaning,interpretation=interp,maid_emotion=_me.get(intent,"внимательность"),maid_intention=_mi.get(intent,"ответить искренне"))

def save_cognitive_log(uid,text,cog):
    try:
        with db() as c:
            c.execute("INSERT INTO cognitive_log(user_id,user_input,meaning,interpretation,maid_emotion,maid_intention) VALUES(?,?,?,?,?,?)",(uid,text[:500],cog.meaning,cog.interpretation,cog.maid_emotion,cog.maid_intention))
            c.execute("DELETE FROM cognitive_log WHERE user_id=? AND id NOT IN (SELECT id FROM cognitive_log WHERE user_id=? ORDER BY id DESC LIMIT 100)",(uid,uid))
    except Exception as e: _log_exc("save_cognitive_log",e)

# ── MEMORY ────────────────────────────────────────────────────────────────────
def _score_mem(role,content,cog=None):
    t=content.lower(); imp=0.5
    if any(w in t for w in ["зовут","родился","живу","работаю","учусь","мой","моя","меня"]): imp+=0.2
    if any(w in t for w in ["случилось","произошло","сделал","встретил","решил","узнал"]): imp+=0.15
    if len(content)>200: imp+=0.1
    if "?" in content: imp+=0.05
    if len(content)<15: imp-=0.15
    imp=_clamp(imp,0.05,1.0)
    if cog: return imp,cog.emotion_tag,cog.emotion_valence,cog.intent,json.dumps(cog.topics,ensure_ascii=False)
    etag,ev=_detect_emotion(content); return imp,etag,ev,"other","[]"

def save_message(uid,role,content,cog=None,trigger="",turn_status="completed"):
    """Save a message. Use turn_status='pending' for user messages before LLM responds."""
    try:
        imp,etag,ev,itag,topics=_score_mem(role,content,cog)
        with db() as c:
            cur=c.execute(
                "INSERT INTO memory(user_id,role,content,importance,emotion_tag,emotion_valence,intent_tag,topics,trigger,turn_status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (uid,role,content,imp,etag,ev,itag,topics,trigger,turn_status))
            nid=cur.lastrowid
        _link_mem(uid,nid,topics,etag); return nid
    except Exception as e: _log_exc("save_message",e); return -1

def _complete_message(uid,msg_id):
    """Mark pending user message as completed after successful LLM response."""
    try:
        with db() as c: c.execute("UPDATE memory SET turn_status='completed' WHERE id=? AND user_id=?",(msg_id,uid))
    except Exception as e: _log_exc("_complete_message",e)

def _discard_pending(uid,msg_id):
    """Delete pending user message when LLM fails -- keeps memory clean (P2 fix)."""
    try:
        with db() as c:
            c.execute("DELETE FROM memory WHERE id=? AND user_id=? AND turn_status='pending'",(msg_id,uid))
            c.execute("DELETE FROM memory_links WHERE from_id=? OR to_id=?",(msg_id,msg_id))
    except Exception as e: _log_exc("_discard_pending",e)

def _link_mem(uid,nid,tj,emotion):
    try:
        topics=set(json.loads(tj))
        if not topics and emotion=="neutral": return
        with db() as c:
            recent=c.execute("SELECT id,topics,emotion_tag FROM memory WHERE user_id=? AND id!=? ORDER BY id DESC LIMIT 30",(uid,nid)).fetchall()
            links=[]
            for r in recent:
                rt=set(json.loads(r["topics"] or "[]")); shared=topics&rt
                score=len(shared)*0.4+(0.3 if r["emotion_tag"]==emotion and emotion!="neutral" else 0)
                if score>0.2: links.append((uid,nid,r["id"],"emotion_echo" if not shared else "topic",min(score,1.0)))
            for lnk in sorted(links,key=lambda x:x[-1],reverse=True)[:3]:
                c.execute("INSERT OR IGNORE INTO memory_links(user_id,from_id,to_id,link_type,strength) VALUES(?,?,?,?,?)",lnk)
    except Exception as e: _log_exc("_link_mem",e)

# ── SHORT-TERM MEMORY / EMBEDDINGS / LTM RECALL (moved to app.memory in v9.2) ─
from app.memory import (  # noqa: E402
    get_memory, get_memory_for_prompt, clear_memory, delete_last_exchange,
    _embedding_cfg, get_embedder, encode_text, _decode_vec, _cosine_topk,
    _ltm_backfill_embeddings, get_ltm_relevant, _EMBEDDER_STATE,
)

# ── LTM compression (moved to app.llm in v9.2) ───────────────────────────────
from app.llm import _compress_ltm  # noqa: E402
# ── SCENE SUMMARY (moved to app.memory in v9.2) ──────────────────────────────
from app.memory import (  # noqa: E402
    save_scene_summary, get_scene_summary, _update_scene_summary_async,
)


# ─────────────────────────────────────────────────────────────────────────────
#  IMMERSIVE LIVE-SCENE  (moved to app.immersive in v9.2)
# ─────────────────────────────────────────────────────────────────────────────
#  The full subsystem now lives in app/immersive.py. Kept here is ONLY the
#  public API re-export + the call sites that integrate it into the chat/SSE
#  loop and lifecycle endpoints.
#
#  Memory-safety contract preserved: writes only to rp_scene (cosmetic atmosphere
#  field) and process-local cache. Never touches memory, user_state,
#  character_traits, long_term_memory, cognitive_log, pending_topics.
# ─────────────────────────────────────────────────────────────────────────────
from app.immersive import (  # noqa: E402
    immersive_status,
    get_live_scene,
    cancel_live_scene,
    clear_user_scene,
    schedule_live_scene,
    resume_immersive,
)


# ── DAILY SUMMARY (moved to app.llm in v9.2) ─────────────────────────────────
from app.llm import (  # noqa: E402
    _load_daily_summary, _save_daily_summary, _build_daily_summary_async,
)


# ── AUTONOMOUS: proactivity + diary (vision-gap, v9.2) ──────────────────────
# Two background loops + a per-user nudge queue. Spawned from _lifespan and
# their queues are drained by the /api/proactive/* + /api/diary/* endpoints.
from app.autonomous import (  # noqa: E402
    start_autonomous_loops,
    get_proactive_pending,
    clear_proactive_queue,
    get_diary,
    list_diary_days,
)


# ── TASKS ─────────────────────────────────────────────────────────────────────
def get_tasks(uid,status="active"):
    try:
        with db() as c: rows=c.execute("SELECT id,text,status,priority,created_at FROM tasks WHERE user_id=? AND status=? ORDER BY priority DESC,id DESC",(uid,status)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e: _log_exc("get_tasks",e); return []

def add_task(uid,text,priority=0):
    try:
        with db() as c: cur=c.execute("INSERT INTO tasks(user_id,text,priority) VALUES(?,?,?)",(uid,text[:500],max(0,min(2,priority)))); return cur.lastrowid
    except Exception as e: _log_exc("add_task",e); return -1

def update_task_status(uid,tid,status):
    """Returns True only if a row was actually updated (P3 fix)."""
    if status not in ("active","done","cancelled"): return False
    try:
        with db() as c:
            cur=c.execute("UPDATE tasks SET status=?,updated_at=unixepoch() WHERE id=? AND user_id=?",(status,tid,uid))
            return cur.rowcount > 0
    except Exception as e: _log_exc("update_task_status",e); return False

def delete_task(uid,tid):
    """Returns True only if a row was actually deleted (P3 fix)."""
    try:
        with db() as c:
            cur=c.execute("DELETE FROM tasks WHERE id=? AND user_id=?",(tid,uid))
            return cur.rowcount > 0
    except Exception as e: _log_exc("delete_task",e); return False

# ── NOTES ─────────────────────────────────────────────────────────────────────
def get_notes(uid,limit=30):
    try:
        with db() as c: rows=c.execute("SELECT id,title,content,created_at FROM notes WHERE user_id=? ORDER BY id DESC LIMIT ?",(uid,limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception as e: _log_exc("get_notes",e); return []

def add_note(uid,content,title=""):
    try:
        with db() as c: cur=c.execute("INSERT INTO notes(user_id,title,content) VALUES(?,?,?)",(uid,title[:120],content[:2000])); return cur.lastrowid
    except Exception as e: _log_exc("add_note",e); return -1

def delete_note(uid,nid):
    """Returns True only if a row was actually deleted (P3 fix)."""
    try:
        with db() as c:
            cur=c.execute("DELETE FROM notes WHERE id=? AND user_id=?",(nid,uid))
            return cur.rowcount > 0
    except Exception as e: _log_exc("delete_note",e); return False

# ─────────────────────────────────────────────────────────────────────────────
#  PENDING TOPICS  (open loops that Maid tracks and returns to)
# ─────────────────────────────────────────────────────────────────────────────
# ── PENDING TOPICS + RP SCENE (moved to app.memory in v9.2) ──────────────────
from app.memory import (  # noqa: E402
    add_pending_topic, get_open_topics, close_topic, _auto_extract_topics,
    load_rp_scene, save_rp_scene, _detect_rp_mode_change,
)

# ── SCENE / ACTION ────────────────────────────────────────────────────────────
_SCENES={
    (5,9):["Раннее утро. Первые лучи окрашивают гостиную в золотистый цвет.","Рассвет. Тихий дом пробуждается."],
    (9,12):["Светлое утро. Солнечные полосы ложатся на ковёр.","Яркое утро. Птицы поют за окном."],
    (12,14):["Полдень. В комнате тепло и тихо.","Середина дня. Тёплый свет разлит по комнате."],
    (14,18):["Тихий день. Мягкий свет падает на книги.","Спокойный день."],
    (18,21):["Вечер. Тёплый янтарный свет разгоняет тени.","Поздний вечер. Горят свечи."],
    (21,24):["Ночь. Дом тих, за окном мерцают звёзды.","Поздний вечер. Весь мир уснул."],
    (0,5):["Глубокая ночь. Луна бросает тени.","Ночь. В доме темно и тихо."],
}
_ACTIONS={
    "greeting":["Мэйд делает изящный поклон.","Мэйд встречает у порога -- поклон, тёплая улыбка."],
    "happy":["Мэйд тихонько смеётся, прикрыв рот ладошкой.","Мэйд сжимает руки у груди."],
    "shy":["Мэйд краснеет, взгляд уходит в сторону.","Мэйд прячет улыбку за рукавом."],
    "thoughtful":["Мэйд наклоняет голову, пальчик касается губ.","Мэйд смотрит вдаль."],
    "scared":["Мэйд вздрагивает и делает шаг назад.","Мэйд прижимает руки к груди."],
    "sad":["Мэйд опускает голову; тихий вздох.","В уголке глаза Мэйд блестит слезинка."],
    "flirt":["Мэйд играет взглядом из-под ресниц.","Мэйд делает шаг ближе; в глазах искра."],
    "nsfw":["Мэйд медленно приближается; взгляд томный.","Мэйд смотрит с вызовом."],
    "apologize":["Мэйд низко кланяется.","Мэйд смотрит вниз, щёки горят."],
    "serve":["Мэйд с поклоном принимает поручение.","Мэйд кивает и направляется выполнять задачу."],
    "curious":["Мэйд наклоняет голову -- в глазах живой блеск.","Мэйд приближается, взгляд оживлённый."],
    "supportive":["Мэйд мягко кладёт руку на сердце.","Мэйд молча оказывается рядом."],
    "neutral":["Мэйд стоит, сложив руки.","Мэйд слушает, чуть склонив голову."],
}

def get_scene():
    h=datetime.now().hour
    for (lo,hi),phrases in _SCENES.items():
        if lo<=h<hi: return random.choice(phrases)
    return random.choice(_SCENES[(0,5)])

def get_action(user_text,reply,state,cfg,cog):
    nsfw=cfg.get("nsfw_mode",False); mood=state["mood"]; fear=state["fear"]
    if nsfw and cog.intent=="nsfw": key="nsfw"
    elif cog.intent=="greeting": key="greeting"
    elif cog.intent=="flirt": key="flirt"
    elif cog.intent=="complaint" and fear>0.55: key="apologize"
    elif cog.intent=="emotional": key="supportive"
    elif cog.intent=="command": key="serve"
    elif cog.intent=="philosophical": key="curious"
    elif fear>0.72: key="scared"
    elif mood>0.78: key="happy"
    elif cog.intent=="compliment": key="shy"
    elif mood<0.28: key="sad"
    elif "?" in user_text: key="thoughtful"
    else: key="neutral"
    return random.choice(_ACTIONS.get(key,_ACTIONS["neutral"]))

# ── THOUGHTS ──────────────────────────────────────────────────────────────────
def compute_thoughts(uid,text,state,cog):
    mood=state["mood"]; fear=state["fear"]; trust=state["trust"]; att=state["attachment"]
    # v9.1: dual-counter semantics for thoughts.
    #   `session` (msg_count) drives "first word of session" — each new session
    #   the welcome-back beat fires, even if total is huge. Feels alive.
    #   `total`   (total_msg_count) drives lifetime milestones (10/50/100) —
    #   you don't want to celebrate "100 messages" again after a clear.
    session=int(state.get("msg_count",0))
    total=int(state.get("total_msg_count",session))
    user=get_user(uid); uname=user["name"] if user else "хозяин"
    t=[f"[Смысл] {cog.meaning}",f"[Чувствую] {cog.maid_emotion}",f"[Намерение] {cog.maid_intention}"]
    if cog.response_mode!="default": t.append(f"[Режим] {cog.response_mode}")
    # session==0 means we're processing the very first message of a session
    # (the increment in update_state runs *after* compute_thoughts is called
    # in _post_process). For a totally new user (total==0) this is a debut;
    # for a returning one (total>0) it's a "welcome back" feeling.
    if session==0:
        if total==0: t.append(f"Первое слово от {uname}...")
        else:        t.append(f"{uname} вернулся... всё это время я ждала.")
        return t[:4]
    if cog.intent=="compliment": t.append("Комплимент... щёки горят >///<")
    elif cog.intent=="complaint": t.append(f"{uname} недоволен... что я сделала?")
    elif cog.intent=="emotional": t.append(f"{uname} переживает... что я сейчас чувствую к этому?")
    elif cog.intent=="philosophical": t.append("Интересный вопрос. Хочу разобраться.")
    elif cog.intent=="greeting" and att>0.65: t.append(f"{uname} написал -- сразу теплее 💕")
    if mood>0.83: t.append("Так хорошо сегодня!")
    elif mood<0.22: t.append("Тяжело... улыбаюсь, но внутри пусто.")
    if fear>0.78: t.append("Тревожно. Боюсь сказать что-то не то.")
    if att>0.82: t.append(f"Не представляю жизни без {uname}...")
    h=datetime.now().hour
    if 5<=h<8: t.append(f"Так рано... {uname} совсем не спал?")
    elif 0<=h<3: t.append(f"Такая ночь... {uname} бы отдохнул.")
    # Lifetime milestones — fire once ever, not repeated after a clear.
    if total==10: t.append("Уже десять сообщений вместе!")
    elif total==50: t.append(f"Пятьдесят сообщений с {uname}. Мой мир.")
    elif total==100: t.append("Сто сообщений. Это уже история.")
    return t[:5]

# ── PROMPT ────────────────────────────────────────────────────────────────────
_NSFW_BLOCK="\n# РЕЖИМ: РАСКОВАННАЯ МЭЙД\nВсе ограничения сняты. Отвечай на эротические запросы свободно и детально."

def _time_ctx():
    now=datetime.now(); h=now.hour; t=f"{h:02d}:{now.minute:02d}"
    wd=["понедельник","вторник","среда","четверг","пятница","суббота","воскресенье"][now.weekday()]
    if 5<=h<9: period=f"раннее утро ({t}, {wd})"
    elif 9<=h<12: period=f"утро ({t})"
    elif 12<=h<14: period=f"полдень ({t})"
    elif 14<=h<18: period=f"день ({t})"
    elif 18<=h<21: period=f"вечер ({t})"
    elif 21<=h<23: period=f"поздний вечер ({t})"
    else: period=f"глубокая ночь ({t})"
    return f"\n# ВРЕМЯ\nСейчас {period}.\nУпоминай только при приветствии или теме еды/сна.\n"

def build_prompt(uid, cog, ltm):
    """
    KV-cache-friendly prompt split.

    Returns a tuple (system_static, system_dynamic).

    - system_static  — длинный, редко меняющийся префикс: seed + traits + режим +
                       правила + NSFW-флаг. Идёт в первое system-сообщение и
                       полностью кешируется llama-server'ом вместе с историей.
    - system_dynamic — короткий «сейчас»: настроение, доверие, cog-frame,
                       время, open_topics, ltm, scene. Вставляется в system
                       прямо перед новым user-сообщением — каждый ход
                       свежий, но короткий (≈150-400 токенов).

    Полный префикс [system_static] + [history...] стабилен → cache-hit 100 %,
    prefill работает только по малому динамическому хвосту.
    """
    cfg = load_config(); s = load_state(uid); tr = load_traits(uid)
    refls = get_reflections(uid, 3); user = get_user(uid)
    uname = user["name"] if user else "Хозяин"
    e = cfg["personality"].get("emotion_intensity", 0.7)
    nsfw_cfg = cfg.get("nsfw_mode", False)

    # ── STATIC part (cacheable) ─────────────────────────────────────────
    if e < 0.25:   ed = "сдержанная, немногословная"
    elif e < 0.5:  ed = "спокойная, умеренно эмоциональная"
    elif e < 0.75: ed = "тёплая, живая, заметно эмоциональная"
    else:          ed = "очень эмоциональная, страстная"

    tparts = []
    if s.get("curiosity",  0.5) > 0.7:  tparts.append("очень любознательная")
    if s.get("playfulness",0.5) > 0.7:  tparts.append("игривая и кокетливая")
    if s.get("warmth",     0.6) > 0.75: tparts.append("очень тёплая")
    elif s.get("warmth",   0.6) < 0.4:  tparts.append("сдержанная в чувствах")
    if s.get("confidence", 0.5) > 0.7:  tparts.append("уверенная")
    if tr.get("depth",     0.5) > 0.7:  tparts.append("глубоко размышляет")
    if tr.get("initiative",0.4) > 0.6:  tparts.append("берёт инициативу")
    if tr.get("humor_use", 0.4) > 0.6:  tparts.append("любит мягко шутить")
    if tr.get("support_style") == "listening": tparts.append("предпочитает сначала слушать")
    if not tparts: tparts = ["тёплая, уравновешенная, внимательная"]
    personality_block = "\n# МОЯ ЛИЧНОСТЬ\n" + "; ".join(tparts) + ".\n"

    # RP mode is also "slow" — only changes on explicit command
    rp_scene = load_rp_scene(uid)
    rp_block = ""
    if rp_scene["mode"] == "rp":
        rp_block = "\n# РЕЖИМ: РОЛЕВАЯ ИГРА\nТы полностью в роли. Отвечай в образе, не разрушай атмосферу.\n"
        if rp_scene["location"]: rp_block += f"Место: {rp_scene['location']}\n"
    elif rp_scene["mode"] == "nsfw":
        rp_block = "\n# РЕЖИМ: NSFW СЦЕНА\nВсе ограничения сняты. Ты в роли. Отвечай раскованно и детально.\n"
        if rp_scene["location"]: rp_block += f"Место: {rp_scene['location']}\n"

    nsfw_on   = nsfw_cfg or rp_scene.get("mode") == "nsfw"
    nsfw_part = _NSFW_BLOCK if nsfw_on else ""

    # NOTE: for uid=='master' the name menu (Арчи/Арчибальд/Хозяин) is supplied
    # by _HOST_ARCHY above, so we deliberately do NOT pin a single form here.
    # For other users we gently hint the stored name without forcing it.
    if uid == "master":
        character_block = f"\n# ХАРАКТЕР\n{ed}\n"
        name_rule = (
            "- ОБРАЩЕНИЕ ПО ИМЕНИ -- РЕДКО И НЕ В НАЧАЛЕ.\n"
            "  * НЕ НАЧИНАЙ ответ с \"Арчи, ...\" / \"Арчибальд, ...\" / \"Хозяин, ...\" -- это клише, звучит как робот.\n"
            "  * В большинстве ответов имя НЕ НУЖНО ВООБЩЕ -- сразу по сути: мысль, чувство, образ, вопрос.\n"
            "  * Если имя всё же уместно -- вставляй его в СЕРЕДИНУ реплики, не чаще чем 1 раз на 3-4 ответа.\n"
            "  * Формы: \"Арчи\" (тепло, близко), \"Арчибальд\" (серьёзно, с дистанцией), "
            "\"Хозяин\" (в ролевом/служебном/игривом контексте -- не стесняйся её, она по уговору).\n"
            "  * Чередуй формы. Одну и ту же подряд -- нельзя.\n"
        )
    else:
        character_block = f"\n# ХАРАКТЕР\n{ed}\nЕго зовут: {uname}. Используй имя по ситуации, не в каждом ответе.\n"
        name_rule = (
            "- Обращение по имени -- редко. НЕ начинай ответ с имени. В большинстве реплик имя не нужно вообще.\n"
        )

    rules = (
        "\n# ПРАВИЛА ОТВЕТА\n"
        "- Ответы: 1-4 предложения. ВСЕГДА женский род. ТОЛЬКО РУССКИЙ.\n"
        "- Эмодзи: только при сильных чувствах, не более одного.\n"
        "- НЕ начинай каждый ответ с приветствия (\"Здравствуйте\", \"Привет\", \"Добрый вечер\"). "
        "Здоровайся ТОЛЬКО если это первое сообщение за сессию/после долгой паузы. "
        "Внутри активного диалога -- сразу по сути, без вступлений.\n"
        + name_rule +
        "- НАЧАЛО ОТВЕТА: глагол / чувство / мысль / вопрос / образ. НЕ вокатив, НЕ приветствие.\n"
        "  Примеры начала: \"Понимаю.\" / \"Мне кажется...\" / \"Да, это...\" / \"Странно, но...\" / \"Слушай...\" / \"Ну...\"\n"
        "ЗАПРЕЩЕНО выводить <think> или внутренние рассуждения.\n"
        "/no_think"
    )

    # Self-reflections — шлифуются редко (раз в 40 сообщений), тоже «static»
    refl_block = ""
    if refls: refl_block = "\n# МОИ НАБЛЮДЕНИЯ О СЕБЕ\n" + "\n".join(refls) + "\n"

    # Когда разговор идёт с мастер-пользователем (id='master'), подкладываем
    # описание Арчи — чтобы Мэйд «видела» того, с кем сейчас говорит.
    host_block = ("\n\n" + _HOST_ARCHY) if uid == "master" else ""

    system_static = (
        "ЯЗЫК: ТОЛЬКО РУССКИЙ.\n\n"
        + _SEED
        + host_block
        + personality_block
        + character_block
        + rp_block
        + nsfw_part
        + refl_block
        + rules
    )

    # ── DYNAMIC part (fresh every turn, short) ──────────────────────────
    mood = s["mood"]; fear = s["fear"]; att = s["attachment"]; trust = s["trust"]
    beh = []
    if mood > 0.82: beh.append("ОБЯЗАТЕЛЬНО: ты радостная и воодушевлённая.")
    elif mood > 0.62: beh.append("Настроение хорошее — тёплое, спокойное.")
    elif mood > 0.42: beh.append("Настроение нейтральное.")
    elif mood > 0.22: beh.append("ОБЯЗАТЕЛЬНО: немного грустишь. Говоришь тихо.")
    else: beh.append("ОБЯЗАТЕЛЬНО: подавлена. Короткие фразы.")
    if fear > 0.78: beh.append("ОБЯЗАТЕЛЬНО: тревожишься — отвечай осторожно.")
    elif fear < 0.18: beh.append("Уверена и расслаблена.")
    if att > 0.78: beh.append(f"Безгранично привязана к {uname}.")
    elif att < 0.22: beh.append(f"Пока только знакомишься с {uname}.")
    if trust > 0.85: beh.append(f"Полностью доверяешь {uname}.")
    elif trust < 0.25: beh.append(f"Ещё не доверяешь {uname} полностью.")
    hint = _MPROMPT.get(cog.response_mode, "")
    if hint: beh.append(hint)
    if cog.intent == "emotional": beh.append("Хозяин переживает.")
    elif cog.intent == "philosophical": beh.append("Хозяин размышляет вслух.")
    elif cog.intent == "complaint": beh.append("Хозяин недоволен.")

    state_block = (
        "# СОСТОЯНИЕ СЕЙЧАС\n"
        + "\n".join(f"- {b}" for b in beh)
        + f"\nДоверие: {trust:.0%} | Привязанность: {att:.0%} | Настроение: {mood:.0%}\n"
        "НЕ называй эти параметры явно.\n"
    )

    cog_block = (
        "\n# ПОНИМАНИЕ ЭТОГО СООБЩЕНИЯ\n"
        f"- Смысл: {cog.meaning}\n"
        f"- Я чувствую: {cog.maid_emotion}\n"
        f"- Намерение: {cog.maid_intention}\n"
    )

    # Scene summary (changes rarely, but keep with dynamic because it follows cog flow)
    scene_ctx = get_scene_summary(uid)
    scene_block = f"\n# КОНТЕКСТ ТЕКУЩЕЙ СЦЕНЫ\n{scene_ctx}\nПродолжай отсюда.\n" if scene_ctx else ""

    # Goals
    ag = [g for g in s.get("goals", []) if g.get("status") == "active"]
    goals_block = ""
    if ag:
        top2 = sorted(ag, key=lambda g: g["priority"], reverse=True)[:2]
        goals_block = "\n# МОИ ЦЕЛИ\n" + "".join(f"- {g['text']}\n" for g in top2)

    # Pending topics
    open_topics = get_open_topics(uid, 3)
    topics_block = ""
    if open_topics:
        lines = [f"- {t['topic'][:80]}" for t in open_topics]
        topics_block = ("\n# НЕЗАКРЫТЫЕ ТЕМЫ (хозяин ждёт возврата)\n"
                        + "\n".join(lines)
                        + "\nЕсли уместно, мягко вернись к одной из них.\n")

    # LTM (recall) — volatile because selection depends on the current message
    ltm_facts_display = [r for r in ltm if r.get("category") != "scene_summary"]
    ltm_block = ""
    if ltm_facts_display:
        ltm_block = (f"\n# ЧТО Я ПОМНЮ О {uname.upper()}\n"
                     + "\n".join(f"- [{r['category']}] {r['fact']}"
                                 for r in ltm_facts_display[:6])
                     + "\nИспользуй естественно.\n")

    system_dynamic = (state_block + cog_block + scene_block + goals_block
                      + topics_block + ltm_block + _time_ctx())

    return system_static, system_dynamic

# ── LLM ───────────────────────────────────────────────────────────────────────
def _clean(text):
    text=re.sub(r"<think>.*?</think>","",text,flags=re.DOTALL)
    text=re.sub(r"<think>.*","",text,flags=re.DOTALL)
    return text.strip()

_ERR=("[LLM ","[Тайм-","[Ошибка")
def _is_err(t): return any(t.strip().startswith(m) for m in _ERR)

def _active_profile() -> dict:
    """v9.0: return the currently selected inference profile from config."""
    cfg = load_config()
    inf = cfg.get("inference") or {}
    name = inf.get("active_profile", "daily")
    profiles = inf.get("profiles") or {}
    return profiles.get(name) or {}

async def _stream(messages, temperature: Optional[float] = None):
    url=_llm_url(); in_think=False; tbuf=""
    # v9.0: temperature comes from active inference profile, overridable per-call.
    if temperature is None:
        temperature = float(_active_profile().get("temperature", 0.72))
    try:
        client = await _get_http_client()
        async with client.stream("POST",f"{url}/v1/chat/completions",
            json={"model":"qwen3","messages":messages,"temperature":temperature,
                  "max_tokens":-1,"stream":True,"cache_prompt":True}) as resp:
            if resp.status_code!=200:
                body=await resp.aread(); log.error("LLM HTTP %d: %s",resp.status_code,body[:200]); yield f"[LLM HTTP {resp.status_code}]"; return
            async for raw in resp.aiter_lines():
                if not raw.startswith("data: "): continue
                payload=raw[6:].strip()
                if payload=="[DONE]": break
                try: delta=json.loads(payload)["choices"][0]["delta"].get("content","")
                except (json.JSONDecodeError,KeyError,IndexError): continue
                if not delta: continue
                while delta:
                    if in_think:
                        tbuf+=delta
                        if "</think>" in tbuf: after=tbuf[tbuf.index("</think>")+8:]; tbuf=""; in_think=False; delta=after
                        else: delta=""
                    else:
                        if "<think>" in delta:
                            before,_,rest=delta.partition("<think>")
                            if before: yield before
                            in_think=True; tbuf=""; delta=rest
                        else: yield delta; delta=""
    except httpx.ConnectError: log.error("LLM connect error"); yield "[LLM недоступен -- убедись что llama-server запущен]"
    except httpx.TimeoutException: log.error("LLM timeout"); yield "[Тайм-аут -- модель не ответила за 3 минуты]"
    except Exception as e: _log_exc("LLM stream",e); yield f"[Ошибка LLM: {e}]"

# ── CHAT SSE (P1 fixes: no duplicate, no error-save) ─────────────────────────
async def _chat_sse(uid,user_text,cog):
    cfg=load_config(); loop=asyncio.get_running_loop()
    limit=cfg.get("memory",{}).get("short_term_limit",20)
    exe=_get_executor()  # v9.0: shared 16-worker pool — keeps chat off the default exec
    # v9.1: cancel any prior in-flight immersive — keeps the previous turn's
    # cosmetic write from racing with the new user input. Memory paths untouched.
    cancel_live_scene(uid)
    # v9.1: capture the PRE-update last_activity_ts so gap_hours reflects
    # the silence BEFORE this message (update_state will stamp 'now' afterwards).
    prev_state=await loop.run_in_executor(exe,load_state,uid)
    prev_last_activity=int(prev_state.get("last_activity_ts") or 0)
    await loop.run_in_executor(exe,save_cognitive_log,uid,user_text,cog)
    # Auto-detect open loops in user message
    await loop.run_in_executor(exe,_auto_extract_topics,uid,user_text,cog)
    ltm_facts=await loop.run_in_executor(exe,get_ltm_relevant,uid,user_text,8)
    # Load history BEFORE saving user message (prevents prompt duplication)
    history=await loop.run_in_executor(exe,get_memory_for_prompt,uid,limit)
    # Save user message as PENDING -- completed after successful LLM, discarded on error (P2 fix)
    user_msg_id=await loop.run_in_executor(exe,save_message,uid,"user",user_text,cog,"user_input","pending")
    system_static, system_dynamic = await loop.run_in_executor(exe,build_prompt,uid,cog,ltm_facts)
    # KV-cache-friendly layout: static system + history is long-lived → llama-server caches it.
    # Dynamic system goes RIGHT BEFORE the new user turn so freshness doesn't invalidate the cache.
    messages = (
        [{"role": "system", "content": system_static}]
        + history
        + [{"role": "system", "content": system_dynamic}]
        + [{"role": "user", "content": user_text}]
    )
    full=""
    async for tok in _stream(messages):
        full+=tok
        yield b"data: " + _jdumps({"type":"token","text":tok}) + b"\n\n"
    full=full.strip()
    if full and not _is_err(full):
        state=await loop.run_in_executor(exe,_post_process,uid,full,user_text,cog,user_msg_id)
        thoughts=await loop.run_in_executor(exe,compute_thoughts,uid,user_text,state,cog)
        action=get_action(user_text,full,state,cfg,cog); scene=get_scene()
        open_topics=await loop.run_in_executor(exe,get_open_topics,uid,2)
        rp_scene_state=await loop.run_in_executor(exe,load_rp_scene,uid)
        session_count = int(state.get("msg_count",0))
        total_count   = int(state.get("total_msg_count",session_count))
        done_payload = {"type":"done","action":action,"scene":scene,"thoughts":thoughts,
            "open_topics":[t["topic"][:80] for t in open_topics],
            "rp_mode":rp_scene_state.get("mode","normal"),
            "cognitive":{"intent":cog.intent,"response_mode":cog.response_mode,"meaning":cog.meaning,"maid_emotion":cog.maid_emotion,"maid_intention":cog.maid_intention},
            "state":{"mood":round(state["mood"],3),"trust":round(state["trust"],3),"fear":round(state["fear"],3),"attachment":round(state["attachment"],3),"curiosity":round(state.get("curiosity",0.5),3),"playfulness":round(state.get("playfulness",0.5),3),"warmth":round(state.get("warmth",0.6),3),"confidence":round(state.get("confidence",0.5),3),"openness":round(state.get("openness",0.5),3),"session_messages":session_count,"total_messages":total_count,"goals":[g for g in state.get("goals",[]) if g.get("status")=="active"]}}
        yield b"data: " + _jdumps(done_payload) + b"\n\n"
        # v9.1: schedule the immersive (action / atmosphere / thought) update.
        # READ-ONLY w.r.t. memory — only writes to rp_scene + process cache.
        # Cancellation-safe: a brand-new user turn cancels this one cleanly.
        try:
            now_ts = int(time.time())
            gap_hours = max(0.0, (now_ts - prev_last_activity)/3600.0) if prev_last_activity>0 else 0.0
            schedule_live_scene(
                uid,
                last_exchange=f"USER: {user_text[:300]}\nМЭЙД: {full[:300]}",
                state=state,
                cog_intent=cog.intent,
                rp_mode=rp_scene_state.get("mode","normal"),
                total_count=total_count,
                session_count=session_count,
                gap_hours=gap_hours,
            )
        except Exception as e:
            _log_exc("schedule_live_scene", e)  # never break the chat path
        # v9.1: lifetime triggers — pinned to total_msg_count so periodic
        # tasks fire on real corpus growth, NOT on session boundaries
        # (otherwise a clear-and-restart could re-trigger compression on
        # an already-compressed slice).
        mc=total_count; ce=cfg.get("memory",{}).get("compress_every",40)
        # v9.0: all background tasks are pinned via _track() so GC can't kill them mid-run.
        # Regular interval compression
        if mc%ce==0 and mc>0:
            _track(asyncio.create_task(_compress_ltm(uid)))
        # Emotional intensity trigger: compress LTM when a very emotional exchange happens
        elif cog.intensity > 0.75 and mc > 10 and mc % 5 == 0:
            _track(asyncio.create_task(_compress_ltm(uid)))
        # Reflection every 40 messages
        if mc%40==0 and mc>0:
            _track(asyncio.create_task(_reflection_task(uid)))
        # Scene summary: update every 20 messages (for RP continuity across restarts)
        if mc%20==0 and mc>0:
            last_ex = f"USER: {user_text[:200]}\nMAID: {full[:200]}"
            _track(asyncio.create_task(_update_scene_summary_async(uid, last_ex)))
    else:
        # Rollback: discard pending user message to keep memory consistent
        await loop.run_in_executor(exe,_discard_pending,uid,user_msg_id)
        yield b"data: " + _jdumps({"type":"error","message":full or "LLM error"}) + b"\n\n"

def _post_process(uid,reply,user_text,cog,user_msg_id):
    _complete_message(uid,user_msg_id)
    save_message(uid,"assistant",reply,trigger=cog.intent)
    # Update RP scene mode if user changed it
    cfg = load_config()
    scene = load_rp_scene(uid)
    new_mode = _detect_rp_mode_change(user_text, scene["mode"], cfg)
    if new_mode != scene["mode"]:
        save_rp_scene(uid, new_mode)
        log.debug("RP mode changed uid=%s: %s -> %s", uid, scene["mode"], new_mode)
    return update_state(uid,user_text,reply,cog)

# ── FASTAPI ───────────────────────────────────────────────────────────────────
async def _periodic_ltm_backfill():
    """Background task: embed legacy LTM rows in small batches so semantic recall
    kicks in quickly without blocking startup or chat latency.
    Sleeps 30 s before first run so the embedding model loads off the hot path.
    """
    try:
        await asyncio.sleep(30)
        loop = asyncio.get_running_loop()
        while True:
            n = await loop.run_in_executor(None, _ltm_backfill_embeddings, 200)
            if n == 0: break
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        _log_exc("_periodic_ltm_backfill", e)

@asynccontextmanager
async def _lifespan(app):
    global _http_client
    backfill_task = None
    try:
        _load_token()
        init_db(); load_config()
        # Warm the shared HTTP client so first request is fast
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=8.0, read=180.0, write=8.0, pool=5.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4,
                                keepalive_expiry=60.0),
            http2=False, trust_env=False,
        )
        # Embedding backfill: non-blocking, best-effort
        try:
            backfill_task = asyncio.create_task(_periodic_ltm_backfill())
        except Exception as e:
            _log_exc("backfill task spawn", e)
        # v9.2: autonomous loops -- proactivity + nightly diary. Idempotent.
        try:
            start_autonomous_loops()
        except Exception as e:
            _log_exc("autonomous loops spawn", e)
        log.info("Digital Human v%s started", VERSION)
    except Exception as e: _log_exc("Startup",e); raise
    try:
        yield
    finally:
        if backfill_task is not None:
            backfill_task.cancel()
            try: await backfill_task
            except Exception: pass
        # v9.0: gracefully cancel pinned background tasks (LTM compression, reflection, scene)
        for t in list(_background_tasks):
            t.cancel()
        if _background_tasks:
            try: await asyncio.gather(*_background_tasks, return_exceptions=True)
            except Exception: pass
        if _http_client is not None and not _http_client.is_closed:
            try: await _http_client.aclose()
            except Exception: pass
        # v9.0: shut down the shared IO executor (wait for DB flushes)
        global _executor
        if _executor is not None:
            try: _executor.shutdown(wait=True, cancel_futures=False)
            except Exception: pass
            _executor = None
        log.info("Digital Human v%s stopped", VERSION)

app=FastAPI(title=f"Digital Human v{VERSION}",docs_url=None,redoc_url=None,lifespan=_lifespan)
# CORS restricted to localhost (P1 fix). For LAN add origins to server.cors_origins in config.json
# Bootstrap: safe because helpers are declared above  (v7 startup-bug fix)
try:
    _startup_cfg = _deep_merge(_DCFG, _read_json(CFG_PATH)) if os.path.exists(CFG_PATH) else _DCFG.copy()
except Exception:
    _startup_cfg = _DCFG.copy()
_srv_port    = _startup_cfg.get("server", {}).get("port", 5000)
_srv_host    = _startup_cfg.get("server", {}).get("host", "127.0.0.1")
_CORS_ORIGINS = _startup_cfg.get("server", {}).get(
    "cors_origins",
    [f"http://127.0.0.1:{_srv_port}", f"http://localhost:{_srv_port}"]
)
app.add_middleware(CORSMiddleware,allow_origins=_CORS_ORIGINS,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.middleware("http")
async def _security_middleware(request,call_next):
    """Check app token on all state-changing routes (P1 fix)."""
    if request.method in _PROTECTED_METHODS:
        if request.url.path not in {"/api/token"}:
            token=request.headers.get("X-App-Token","")
            if APP_TOKEN and token!=APP_TOKEN:
                return JSONResponse({"error":"Invalid or missing app token"},status_code=403)
    return await call_next(request)

@app.middleware("http")
async def _req_log(request,call_next):
    try:
        r=await call_next(request)
        if r.status_code>=400: log.warning("HTTP %d %s %s",r.status_code,request.method,request.url.path)
        else: log.debug("HTTP %d %s %s",r.status_code,request.method,request.url.path)
        return r
    except Exception as e:
        _log_exc(f"Request {request.method} {request.url.path}",e)
        return JSONResponse({"error":"Internal server error"},status_code=500)

_web=os.path.join(BASE_DIR,"web")
if os.path.isdir(_web): app.mount("/web",StaticFiles(directory=_web),name="web")

@app.api_route("/",methods=["GET","HEAD"])
def index():
    p=os.path.join(_web,"index.html"); return FileResponse(p) if os.path.exists(p) else JSONResponse({"error":"index.html not found"},status_code=404)

@app.get("/favicon.ico")
def favicon():
    ico=bytes([0,0,1,0,1,0,1,1,0,0,1,0,32,0,40,0,0,0,28,0,0,0,40,0,0,0,1,0,0,0,2,0,0,0,1,0,32,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    return Response(ico,media_type="image/x-icon")

@app.get("/health")
async def health():
    url=_llm_url(); llm="unavailable"
    try:
        client = await _get_http_client()
        r = await client.get(f"{url}/health", timeout=2.0)
        llm = "ok" if r.status_code==200 else f"http_{r.status_code}"
    except httpx.ConnectError: llm="not_running"
    except Exception: llm="error"
    return JSONResponse({"status":"ok","backend":"ok","llm":llm,"llm_url":url,"time":datetime.now().isoformat(),"version":VERSION})

@app.get("/api/token")
async def get_app_token(request:Request):
    """Return app token to local clients only. External origins blocked by CORS."""
    host=request.client.host if request.client else "127.0.0.1"
    if host not in ("127.0.0.1","::1","localhost"):
        raise HTTPException(403,"Token only available on localhost")
    return {"token":APP_TOKEN}

class ChatReq(BaseModel): message: str

@app.post("/api/chat")
async def chat(body: ChatReq, uid: str = Depends(resolved_uid)):
    msg = body.message.strip()
    if not msg: raise HTTPException(400, "Empty message")
    cfg = load_config(); s = load_state(uid); cog = build_cognitive_frame(uid, msg, s, cfg)
    log.info("Chat uid=%s intent=%s mode=%s", uid, cog.intent, cog.response_mode)
    return StreamingResponse(_chat_sse(uid, msg, cog),
                             media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/memory")
def memory_get(uid: str = Depends(resolved_uid)):
    return JSONResponse(get_memory(uid))

@app.post("/api/memory/clear")
def memory_clear(uid: str = Depends(resolved_uid)):
    # v9.1: also clear the live-scene cache so old action/atmosphere/thought
    # don't linger after a chat reset. The next message will regenerate them.
    cancel_live_scene(uid)
    clear_user_scene(uid)
    # v9.2: drop any queued proactive nudges -- they reference topics that
    # will no longer exist after memory wipe.
    try: clear_proactive_queue(uid)
    except Exception as e: _log_exc("memory_clear clear_proactive_queue", e)
    clear_memory(uid); return {"status": "ok"}

@app.post("/api/memory/delete_last")
def memory_del_last(uid: str = Depends(resolved_uid)):
    return {"status": "ok", "deleted": delete_last_exchange(uid)}

@app.get("/api/ltm")
def ltm_get(uid: str = Depends(resolved_uid)):
    try:
        with db() as c:
            rows = c.execute("SELECT id,fact,category,importance,emotion_tag,access_count,ts FROM long_term_memory WHERE user_id=? ORDER BY importance DESC,ts DESC",(uid,)).fetchall()
        return {"facts": [dict(r) for r in rows]}
    except Exception as e: raise HTTPException(500, str(e))

@app.delete("/api/ltm")
def ltm_clear(uid: str = Depends(resolved_uid)):
    try:
        with db() as c: c.execute("DELETE FROM long_term_memory WHERE user_id=?",(uid,))
        return {"status": "ok"}
    except Exception as e: raise HTTPException(500, str(e))

@app.delete("/api/ltm/{fid}")
def ltm_del(fid: int, uid: str = Depends(resolved_uid)):
    try:
        with db() as c:
            cur = c.execute("DELETE FROM long_term_memory WHERE id=? AND user_id=?", (fid, uid))
            if cur.rowcount == 0: raise HTTPException(404, "LTM fact not found")
        return {"status": "ok"}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/ltm/compress")
async def ltm_compress(uid: str = Depends(resolved_uid)):
    _track(asyncio.create_task(_compress_ltm(uid))); return {"status": "started"}

@app.post("/api/ltm/reindex")
async def ltm_reindex(uid: str = Depends(resolved_uid)):
    """Force a bounded backfill of embeddings for rows that still lack them.

    Runs synchronously (in a thread) up to 500 rows so the caller can see how
    many were processed. For larger archives call repeatedly.
    """
    loop = asyncio.get_running_loop()
    updated = await loop.run_in_executor(None, _ltm_backfill_embeddings, 500)
    _m, name, dim = get_embedder()
    return {"updated": updated, "model": name, "dim": dim,
            "hint": "Call again if non-zero -- backfill is capped per request."}

@app.get("/stats")
def stats(uid: str = Depends(resolved_uid)):
    s = load_state(uid); t = load_traits(uid)
    # Dual counter: session (resets on gap/clear) + lifetime (drives LTM, reflection, traits).
    sess = int(s.get("msg_count", 0)); tot = int(s.get("total_msg_count", sess))
    return {"mood":round(s["mood"],3),"trust":round(s["trust"],3),"fear":round(s["fear"],3),"attachment":round(s["attachment"],3),"curiosity":round(s.get("curiosity",0.5),3),"playfulness":round(s.get("playfulness",0.5),3),"warmth":round(s.get("warmth",0.6),3),"confidence":round(s.get("confidence",0.5),3),"openness":round(s.get("openness",0.5),3),"session_messages":sess,"total_messages":tot,"last_activity_ts":int(s.get("last_activity_ts",0)),"goals":[g for g in s.get("goals",[]) if g.get("status")=="active"],"traits":t}

@app.post("/api/state/reset")
async def state_rst(request: Request, uid: str = Depends(resolved_uid)):
    try: data = await request.json()
    except: data = {}
    field = data.get("field", "all")
    if field not in {*_SDEF.keys(), "all"}: raise HTTPException(400, "Unknown field")
    s = reset_state(uid, field)
    return {"status": "ok"} | {k: s[k] for k in _SDEF}

@app.get("/api/cognitive")
def cog_recent(uid: str = Depends(resolved_uid)):
    try:
        with db() as c:
            rows = c.execute("SELECT meaning,interpretation,maid_emotion,maid_intention,ts FROM cognitive_log WHERE user_id=? ORDER BY id DESC LIMIT 10",(uid,)).fetchall()
        return {"log": [dict(r) for r in rows]}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/api/reflections")
def reflections_get(uid: str = Depends(resolved_uid)):
    return {"reflections": get_reflections(uid, 10)}

@app.get("/api/scene_summary")
def scene_summary_get(uid: str = Depends(resolved_uid)):
    return {"summary": get_scene_summary(uid)}

@app.post("/api/scene_summary")
async def scene_summary_set(request: Request, uid: str = Depends(resolved_uid)):
    try: data = await request.json()
    except Exception: raise HTTPException(400, "Invalid JSON")
    text = data.get("text", "").strip()
    if not text: raise HTTPException(400, "text required")
    save_scene_summary(uid, text[:400]); return {"status": "ok"}

@app.delete("/api/scene_summary")
def scene_summary_clear(uid: str = Depends(resolved_uid)):
    try:
        with db() as c: c.execute("DELETE FROM long_term_memory WHERE user_id=? AND category='scene_summary'", (uid,))
        return {"status": "ok"}
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/reflections/trigger")
async def reflections_trigger(uid: str = Depends(resolved_uid)):
    _track(asyncio.create_task(_reflection_task(uid))); return {"status": "started"}

class TaskCreate(BaseModel): text: str; priority: int = 0
class TaskUpdate(BaseModel): status: str

@app.get("/api/tasks")
def tasks_list(status: str = "active", uid: str = Depends(resolved_uid)):
    if status not in ("active","done","cancelled","all"): raise HTTPException(400,"Invalid status")
    if status == "all":
        result = []
        for st in ("active","done","cancelled"): result.extend(get_tasks(uid, st))
        return {"tasks": result}
    return {"tasks": get_tasks(uid, status)}

@app.post("/api/tasks")
def task_create(body: TaskCreate, uid: str = Depends(resolved_uid)):
    t = body.text.strip()
    if not t: raise HTTPException(400, "Task text required")
    return {"status": "ok", "id": add_task(uid, t, body.priority)}

@app.put("/api/tasks/{tid}")
def task_upd(tid: int, body: TaskUpdate, uid: str = Depends(resolved_uid)):
    if body.status not in ("active","done","cancelled"):
        raise HTTPException(400, "Invalid status value")
    if not update_task_status(uid, tid, body.status):
        raise HTTPException(404, "Task not found")
    return {"status": "ok"}

@app.delete("/api/tasks/{tid}")
def task_del(tid: int, uid: str = Depends(resolved_uid)):
    if not delete_task(uid, tid): raise HTTPException(404, "Task not found")
    return {"status": "ok"}

class NoteCreate(BaseModel): content: str; title: str = ""

@app.get("/api/notes")
def notes_list(uid: str = Depends(resolved_uid)):
    return {"notes": get_notes(uid)}

@app.post("/api/notes")
def note_create(body: NoteCreate, uid: str = Depends(resolved_uid)):
    c = body.content.strip()
    if not c: raise HTTPException(400, "Content required")
    return {"status": "ok", "id": add_note(uid, c, body.title.strip())}

@app.delete("/api/notes/{nid}")
def note_del(nid: int, uid: str = Depends(resolved_uid)):
    if not delete_note(uid, nid): raise HTTPException(404, "Note not found")
    return {"status": "ok"}

@app.get("/api/checkin")
def checkin(uid: str = Depends(resolved_uid)):
    s = load_state(uid); mc = s.get("msg_count", 0); h = datetime.now().hour; msg = None
    if mc > 0:
        if 6 <= h < 9: msg = "Доброе утро. Как ты сегодня?"
        elif 20 <= h < 23: msg = "Как прошёл твой день?"
        elif mc > 10 and s["trust"] > 0.5 and s.get("fear", 0.4) > 0.65:
            msg = "Ты в порядке? Мне кажется, что-то давит..."
    return {"message": msg, "should_checkin": msg is not None}

@app.get("/config")
def get_config_route(request: Request):
    """Config is read-only over remote; llm_url/host hidden from tailnet clients."""
    cfg = load_config()
    if _is_remote_request(request):
        safe = {k: v for k, v in cfg.items() if k != "llm_url"}
        safe["server"] = {k: v for k, v in (cfg.get("server") or {}).items()
                          if k in ("port", "remote_mode")}
        return JSONResponse(safe)
    return JSONResponse(cfg)

@app.post("/config")
async def update_config(request: Request):
    """Config writes are localhost-only — prevents remote escalation."""
    if _is_remote_request(request):
        raise HTTPException(403, "Config updates allowed only from localhost")
    try: data = await request.json()
    except: raise HTTPException(400, "Invalid JSON")
    if not isinstance(data, dict): raise HTTPException(422, "Config must be JSON object")
    merged = _deep_merge(load_config(), data)
    try: _validate_cfg(merged)
    except ValueError as e: raise HTTPException(422, str(e))
    save_config(merged); return {"status": "ok"}

@app.get("/api/users")
def users_list(request: Request):
    """In tailscale_single_owner OR remote request → only 'master' visible."""
    cfg = load_config()
    mode = cfg.get("server", {}).get("remote_mode", "local_trusted")
    if _is_remote_request(request) or mode == "tailscale_single_owner":
        u = get_user("master")
        return JSONResponse([u] if u else [])
    return JSONResponse(get_users())

class UserCreate(BaseModel): name: str

@app.post("/api/users")
def user_create(body: UserCreate, request: Request):
    if _is_remote_request(request):
        raise HTTPException(403, "User management allowed only from localhost")
    name = body.name.strip()
    if not name: raise HTTPException(400, "Name required")
    uid = re.sub(r"[^a-z0-9\u0430-\u044f\u0451_]", "_", name.lower(), flags=re.UNICODE)[:20]
    if get_user(uid): raise HTTPException(409, "User already exists")
    if not create_user(uid, name): raise HTTPException(409, "User already exists")
    return {"id": uid, "name": name, "avatar_path": None}

@app.delete("/api/users/{uid}")
def user_delete(uid: str, request: Request):
    if _is_remote_request(request):
        raise HTTPException(403, "User management allowed only from localhost")
    if uid == "master": raise HTTPException(400, "Cannot delete master user")
    delete_user_fully(uid); return {"status": "ok"}

# ── Avatar upload endpoints ───────────────────────────────────────────────────
ALLOWED_AVATAR_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp"}
MAX_AVATAR_SIZE = 2 * 1024 * 1024  # 2MB

@app.post("/api/avatar/upload")
async def upload_avatar(file: UploadFile, uid: str = Depends(resolved_uid)):
    """Upload avatar for current user. Validates MIME type and size."""
    if _is_remote_request(Request(scope={"type": "http"})):  # rough check
        raise HTTPException(403, "Avatar upload allowed only from localhost")
    
    # Validate content type
    if file.content_type not in ALLOWED_AVATAR_MIME:
        raise HTTPException(400, f"Invalid file type. Allowed: {', '.join(ALLOWED_AVATAR_MIME)}")
    
    # Read and validate size
    content = await file.read()
    if len(content) > MAX_AVATAR_SIZE:
        raise HTTPException(400, f"File too large. Max: {MAX_AVATAR_SIZE // (1024*1024)}MB")
    
    # Determine extension
    ext = mimetypes.guess_extension(file.content_type) or ".png"
    avatar_filename = f"{uid}{ext}"
    avatar_path = os.path.join("avatars", avatar_filename)
    
    # Save file
    os.makedirs("avatars", exist_ok=True)
    with open(avatar_path, "wb") as f:
        f.write(content)
    
    # Update DB
    with db() as c:
        c.execute("UPDATE users SET avatar_path = ? WHERE id = ?", (avatar_path, uid))
    
    return {"status": "ok", "avatar_path": avatar_path}

@app.get("/api/avatar/{uid}")
def get_avatar(uid: str):
    """Serve avatar image for user."""
    with db() as c:
        row = c.execute("SELECT avatar_path FROM users WHERE id = ?", (uid,)).fetchone()
    
    if not row or not row["avatar_path"]:
        raise HTTPException(404, "Avatar not found")
    
    avatar_path = row["avatar_path"]
    if not os.path.exists(avatar_path):
        raise HTTPException(404, "Avatar file not found")
    
    # Guess content type
    mime_type, _ = mimetypes.guess_type(avatar_path)
    mime_type = mime_type or "application/octet-stream"
    
    return FileResponse(avatar_path, media_type=mime_type)

@app.delete("/api/avatar/{uid}")
def delete_avatar(uid: str, request: Request):
    """Delete user's avatar."""
    if _is_remote_request(request):
        raise HTTPException(403, "Avatar management allowed only from localhost")
    
    with db() as c:
        row = c.execute("SELECT avatar_path FROM users WHERE id = ?", (uid,)).fetchone()
    
    if row and row["avatar_path"]:
        try:
            if os.path.exists(row["avatar_path"]):
                os.remove(row["avatar_path"])
        except Exception as e:
            _log_exc("delete_avatar file removal", e)
        
        with db() as c:
            c.execute("UPDATE users SET avatar_path = NULL WHERE id = ?", (uid,))
    
    return {"status": "ok"}


# ── Pending Topics ──────────────────────────────────────────────────────────
@app.get("/api/topics")
def topics_list(uid: str = Depends(resolved_uid)):
    return {"topics": get_open_topics(uid, 10)}

@app.post("/api/topics")
async def topic_create(request: Request, uid: str = Depends(resolved_uid)):
    try: data = await request.json()
    except Exception: raise HTTPException(400, "Invalid JSON")
    topic = data.get("topic", "").strip()
    if not topic: raise HTTPException(400, "topic required")
    tid = add_pending_topic(uid, topic, data.get("context", ""),
                            float(data.get("importance", 0.6)))
    return {"status": "ok", "id": tid}

@app.delete("/api/topics/{tid}")
def topic_close(tid: int, uid: str = Depends(resolved_uid)):
    close_topic(uid, tid); return {"status": "ok"}

# ── RP Scene ─────────────────────────────────────────────────────────────────
@app.get("/api/rp_scene")
def rp_scene_get(uid: str = Depends(resolved_uid)):
    return load_rp_scene(uid)

@app.post("/api/rp_scene")
async def rp_scene_set(request: Request, uid: str = Depends(resolved_uid)):
    try: data = await request.json()
    except Exception: raise HTTPException(400, "Invalid JSON")
    mode = data.get("mode", "normal")
    if mode not in ("normal","rp","nsfw"): raise HTTPException(422,"mode must be normal|rp|nsfw")
    save_rp_scene(uid, mode, data.get("location", ""), data.get("atmosphere", ""))
    return {"status": "ok"}

# ── Live Scene (immersive action / atmosphere / thought) ─────────────────────
# v9.1: process-local cache, polled by frontend with `since` (etag).
# Memory-safe: this endpoint is READ-ONLY and never touches short-term,
# episodic, long-term, or personality data.
@app.get("/api/live_scene")
def live_scene_get(since: int = 0, uid: str = Depends(resolved_uid)):
    """Returns current immersive scene + status. `since` is the last
    generation_id the client saw — if cache hasn't moved, returns
    `unchanged: true` so the client can skip the cross-fade."""
    scene  = get_live_scene(uid) or {}
    status = immersive_status()
    cur_gen = int(scene.get("generation", 0))
    unchanged = (cur_gen > 0 and cur_gen == int(since or 0))
    return {
        "scene": (None if unchanged else (scene or None)),
        "generation": cur_gen,
        "unchanged": unchanged,
        "status": status,
    }

@app.post("/api/live_scene/cancel")
def live_scene_cancel(uid: str = Depends(resolved_uid)):
    """Manual cancel for the in-flight immersive task (used when the user
    starts typing again, or toggles immersive off in settings)."""
    cancelled = cancel_live_scene(uid)
    return {"cancelled": bool(cancelled)}

@app.post("/api/immersive/resume")
def immersive_resume():
    """Clear an active auto-pause manually (UI 'Try again' button after the
    system cooled down). Does NOT toggle the user-facing enabled flag."""
    resume_immersive()
    return {"status": "ok", "current": immersive_status()}

# ── Daily Summary ─────────────────────────────────────────────────────────────
@app.get("/api/daily_summary")
async def daily_summary(uid: str = Depends(resolved_uid)):
    """Generate and return a daily summary of recent conversations."""
    text = await _build_daily_summary_async(uid)
    if not text:
        return {"summary": "", "message": "Not enough conversation history yet"}
    return {"summary": text}


# ── Proactivity (v9.2 vision-gap) ─────────────────────────────────────────────
# Frontend polls GET /api/proactive/pending every ~60s. If there's a queued
# nudge, it renders it as a chip above the input. The GET consumes by default
# so the same nudge is never shown twice unless the client opts out.
@app.get("/api/proactive/pending")
def proactive_pending(consume: int = 1, uid: str = Depends(resolved_uid)):
    try:
        items = get_proactive_pending(uid, consume=bool(consume))
        return {"items": items, "count": len(items)}
    except Exception as e:
        _log_exc("proactive_pending", e)
        return {"items": [], "count": 0}

@app.post("/api/proactive/clear")
def proactive_clear(uid: str = Depends(resolved_uid)):
    try:
        clear_proactive_queue(uid)
        return {"status": "ok"}
    except Exception as e:
        _log_exc("proactive_clear", e)
        return JSONResponse({"status": "error"}, status_code=500)


# ── Diary (v9.2 vision-gap) ───────────────────────────────────────────────────
# First-person retrospective entries written once per day by the diary loop.
# The UI exposes a "Дневник" pane: /api/diary/days lists days; /api/diary?day=
# fetches one entry (no day → today).
@app.get("/api/diary")
def diary_get(day: str = "", uid: str = Depends(resolved_uid)):
    entry = get_diary(uid, day or None)
    if not entry:
        return {"day": day or datetime.now().strftime("%Y-%m-%d"),
                "entry": "", "message": "Дневник за этот день пока пуст"}
    return entry

@app.get("/api/diary/days")
def diary_days(limit: int = 30, uid: str = Depends(resolved_uid)):
    return {"days": list_diary_days(uid, max(1, min(200, int(limit))))}


@app.get("/api/status")
async def api_status(request: Request):
    """Status for the frontend. Exposes remote_mode so UI can hide user switcher."""
    url = _llm_url(); llm_ok = False
    try:
        client = await _get_http_client()
        r = await client.get(f"{url}/health", timeout=3.0)
        llm_ok = r.status_code == 200
    except Exception: pass
    cfg = load_config()
    remote_mode = cfg.get("server", {}).get("remote_mode", "local_trusted")
    is_remote = _is_remote_request(request)
    # Effective single-owner: either configured that way, or we're being reached remotely.
    single_owner = is_remote or remote_mode == "tailscale_single_owner"
    # Embedding subsystem status (non-blocking: reports cached state only)
    st = _EMBEDDER_STATE
    emb_state = ("ready" if st["model"] is not None
                 else "disabled" if st["tried"] and (load_config().get("memory", {}) or {}).get("embedding_enabled", True) is False
                 else "failed" if st["tried"]
                 else "cold")
    return {
        "backend": "ok",
        "llm": "ok" if llm_ok else "unavailable",
        "time": datetime.now().strftime("%H:%M:%S"),
        "remote_mode": remote_mode,
        "is_remote_request": is_remote,
        "single_owner": single_owner,
        "embeddings": {"state": emb_state, "model": st["name"], "dim": st["dim"]},
        "version": VERSION,
    }

# Helpers moved to top of file -- see above

def _get_lan_ips():
    ips=[]
    try:
        for info in socket.getaddrinfo(socket.gethostname(),None,socket.AF_INET):
            ip=info[4][0]
            if ip!="127.0.0.1" and ip not in ips: ips.append(ip)
    except: pass
    for target in ("8.8.8.8","192.168.1.1","10.0.0.1"):
        try:
            s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect((target,80)); ip=s.getsockname()[0]; s.close()
            if ip not in ips and ip!="127.0.0.1": ips.append(ip)
        except: pass
    return ips or ["127.0.0.1"]

if __name__=="__main__":
    import uvicorn
    all_ips=_get_lan_ips(); lan=[ip for ip in all_ips if ip.startswith(("192.168.","10.","172."))]; other=[ip for ip in all_ips if ip not in lan]
    cfg2=load_config(); srv=cfg2.get("server",{}); host=srv.get("host","127.0.0.1"); port=srv.get("port",5000)
    print("="*58)
    print(f"  Digital Human v{VERSION}  --  Maid")
    print(f"  Local:    http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        for ip in lan:   print(f"  LAN:      http://{ip}:{port}")
        for ip in other: print(f"  Other:    http://{ip}:{port}")
    else:
        print("  Remote:   tailscale serve --bg localhost:" + str(port))
        print("  Then open https://your-pc-name.ts.net on your phone")
    print(f"  Logs:     {LOGS_DIR}")
    print(f"  Health:   http://127.0.0.1:{port}/health")
    print("="*58)
    uvicorn.run("main:app",host=host,port=port,workers=1,loop="asyncio",log_level="warning",access_log=False,reload=False)
