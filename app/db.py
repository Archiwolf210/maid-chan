"""SQLite schema, migrations, and db() context manager.

Single source of truth for the disk layout. Everything stateful Maid knows
about her users — short-term memory, long-term memory, cognitive log,
traits, pending topics, rp_scene, daily summaries, tasks, notes — is
declared here.

All cross-module dependencies resolve via LATE imports from `main` inside
functions — avoids circular import at module load.

Public API used by main.py:
  - DB_PATH           -> path of the SQLite file
  - db()              -> contextmanager yielding a sqlite3.Connection
  - init_db()         -> create schema + run column-add migrations
  - _migrate_legacy() -> one-time JSON→SQLite backfill (idempotent)
  - _SCHEMA / _MIGRATIONS exposed for inspection by /api/status
"""
from __future__ import annotations
import os
import sqlite3
from contextlib import contextmanager


def _log():
    from main import log
    return log


def _log_exc(msg, exc):
    from main import _log_exc as _le
    _le(msg, exc)


def _base_dir() -> str:
    from main import BASE_DIR
    return BASE_DIR


# DB_PATH resolves lazily on first db() call — but we also expose it as an
# attribute callers can read. Using a function here because BASE_DIR depends
# on main being importable.
def _db_path() -> str:
    from main import DB_PATH
    return DB_PATH


# ── SCHEMA ───────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL DEFAULT 'master',role TEXT NOT NULL,content TEXT NOT NULL,ts INTEGER NOT NULL DEFAULT(unixepoch()),importance REAL NOT NULL DEFAULT 0.5,emotion_tag TEXT NOT NULL DEFAULT 'neutral',emotion_valence REAL NOT NULL DEFAULT 0.0,intent_tag TEXT NOT NULL DEFAULT 'other',topics TEXT NOT NULL DEFAULT '[]',trigger TEXT NOT NULL DEFAULT '',
    turn_status TEXT NOT NULL DEFAULT 'completed');
CREATE INDEX IF NOT EXISTS idx_mem ON memory(user_id,id DESC);
CREATE TABLE IF NOT EXISTS user_state(user_id TEXT PRIMARY KEY,mood REAL NOT NULL DEFAULT 0.5,trust REAL NOT NULL DEFAULT 0.5,fear REAL NOT NULL DEFAULT 0.4,attachment REAL NOT NULL DEFAULT 0.3,msg_count INTEGER NOT NULL DEFAULT 0,total_msg_count INTEGER NOT NULL DEFAULT 0,last_activity_ts INTEGER NOT NULL DEFAULT 0,curiosity REAL NOT NULL DEFAULT 0.5,playfulness REAL NOT NULL DEFAULT 0.5,warmth REAL NOT NULL DEFAULT 0.6,confidence REAL NOT NULL DEFAULT 0.5,openness REAL NOT NULL DEFAULT 0.5,goals TEXT NOT NULL DEFAULT '[]',updated_at INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE TABLE IF NOT EXISTS long_term_memory(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,fact TEXT NOT NULL,category TEXT NOT NULL DEFAULT 'general',importance REAL NOT NULL DEFAULT 0.7,emotion_tag TEXT NOT NULL DEFAULT 'neutral',access_count INTEGER NOT NULL DEFAULT 0,last_accessed INTEGER,ts INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE INDEX IF NOT EXISTS idx_ltm ON long_term_memory(user_id,importance DESC);
CREATE TABLE IF NOT EXISTS memory_links(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,from_id INTEGER NOT NULL,to_id INTEGER NOT NULL,link_type TEXT NOT NULL DEFAULT 'topic',strength REAL NOT NULL DEFAULT 0.5);
CREATE INDEX IF NOT EXISTS idx_ml ON memory_links(user_id,from_id);
CREATE TABLE IF NOT EXISTS cognitive_log(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,ts INTEGER NOT NULL DEFAULT(unixepoch()),user_input TEXT NOT NULL,meaning TEXT NOT NULL DEFAULT '',interpretation TEXT NOT NULL DEFAULT '',maid_emotion TEXT NOT NULL DEFAULT 'neutral',maid_intention TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_clog ON cognitive_log(user_id,id DESC);
CREATE TABLE IF NOT EXISTS character_traits(user_id TEXT PRIMARY KEY,initiative REAL NOT NULL DEFAULT 0.4,depth REAL NOT NULL DEFAULT 0.5,humor_use REAL NOT NULL DEFAULT 0.4,support_style TEXT NOT NULL DEFAULT 'balanced',updated_at INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE TABLE IF NOT EXISTS self_reflections(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,text TEXT NOT NULL,ts INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE INDEX IF NOT EXISTS idx_refl ON self_reflections(user_id,id DESC);
CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,text TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'active',priority INTEGER NOT NULL DEFAULT 0,created_at INTEGER NOT NULL DEFAULT(unixepoch()),updated_at INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE INDEX IF NOT EXISTS idx_tasks ON tasks(user_id,status);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,title TEXT NOT NULL DEFAULT '',content TEXT NOT NULL,created_at INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE INDEX IF NOT EXISTS idx_notes ON notes(user_id,id DESC);
CREATE TABLE IF NOT EXISTS trait_intent_counts(user_id TEXT NOT NULL,intent TEXT NOT NULL,count INTEGER NOT NULL DEFAULT 0,updated_at INTEGER NOT NULL DEFAULT(unixepoch()),PRIMARY KEY(user_id,intent));
CREATE TABLE IF NOT EXISTS pending_topics(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,topic TEXT NOT NULL,context TEXT NOT NULL DEFAULT '',importance REAL NOT NULL DEFAULT 0.6,status TEXT NOT NULL DEFAULT 'open',created_at INTEGER NOT NULL DEFAULT(unixepoch()),expires_at INTEGER NOT NULL DEFAULT(unixepoch()+259200));
CREATE INDEX IF NOT EXISTS idx_pt ON pending_topics(user_id,status,expires_at);
CREATE TABLE IF NOT EXISTS rp_scene(user_id TEXT PRIMARY KEY,mode TEXT NOT NULL DEFAULT 'normal',location TEXT NOT NULL DEFAULT '',atmosphere TEXT NOT NULL DEFAULT '',last_updated INTEGER NOT NULL DEFAULT(unixepoch()));
CREATE TABLE IF NOT EXISTS daily_summaries(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,day TEXT NOT NULL,summary TEXT NOT NULL,ts INTEGER NOT NULL DEFAULT(unixepoch()),UNIQUE(user_id,day));
CREATE INDEX IF NOT EXISTS idx_ds ON daily_summaries(user_id,day);
CREATE TABLE IF NOT EXISTS diary_entries(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id TEXT NOT NULL,day TEXT NOT NULL,entry TEXT NOT NULL,ts INTEGER NOT NULL DEFAULT(unixepoch()),UNIQUE(user_id,day));
CREATE INDEX IF NOT EXISTS idx_diary ON diary_entries(user_id,day);
"""


# ── MIGRATIONS (idempotent column-adds) ──────────────────────────────────────
_MIGRATIONS = {
    "memory": [("ts","INTEGER NOT NULL DEFAULT(unixepoch())"),("importance","REAL NOT NULL DEFAULT 0.5"),
               ("emotion_tag","TEXT NOT NULL DEFAULT 'neutral'"),("emotion_valence","REAL NOT NULL DEFAULT 0.0"),
               ("intent_tag","TEXT NOT NULL DEFAULT 'other'"),("topics","TEXT NOT NULL DEFAULT '[]'"),
               ("trigger","TEXT NOT NULL DEFAULT ''"),
        ("turn_status","TEXT NOT NULL DEFAULT 'completed'")],  # v6 transaction support
    "user_state":[("curiosity","REAL NOT NULL DEFAULT 0.5"),("playfulness","REAL NOT NULL DEFAULT 0.5"),
                  ("warmth","REAL NOT NULL DEFAULT 0.6"),("confidence","REAL NOT NULL DEFAULT 0.5"),
                  ("openness","REAL NOT NULL DEFAULT 0.5"),("goals","TEXT NOT NULL DEFAULT '[]'"),
                  # v9.1: dual counters. `msg_count` is now the session counter
                  # (resets on /api/memory/clear and after session_gap idle).
                  # `total_msg_count` is lifetime — drives LTM compression,
                  # reflection, trait evolution milestones. `last_activity_ts`
                  # is the last message epoch; used to detect session gaps.
                  ("total_msg_count","INTEGER NOT NULL DEFAULT 0"),
                  ("last_activity_ts","INTEGER NOT NULL DEFAULT 0")],
    "long_term_memory":[("category","TEXT NOT NULL DEFAULT 'general'"),("importance","REAL NOT NULL DEFAULT 0.7"),
                        ("emotion_tag","TEXT NOT NULL DEFAULT 'neutral'"),("access_count","INTEGER NOT NULL DEFAULT 0"),
                        ("last_accessed","INTEGER"),
                        # v8.3: semantic recall -- raw little-endian float32 L2-normalized vector
                        ("embedding","BLOB")],
    "pending_topics":[("context","TEXT NOT NULL DEFAULT ''"),("importance","REAL NOT NULL DEFAULT 0.6"),
                      ("status","TEXT NOT NULL DEFAULT 'open'"),
                      ("expires_at","INTEGER NOT NULL DEFAULT(unixepoch()+259200)")],
    "rp_scene":[("location","TEXT NOT NULL DEFAULT ''"),("atmosphere","TEXT NOT NULL DEFAULT ''"),
                ("last_updated","INTEGER NOT NULL DEFAULT(unixepoch())")],
}


@contextmanager
def db():
    """SQLite connection context manager. WAL, synchronous=NORMAL, FK on.
    Commits on exit; rollbacks + logs on any exception. Closes always."""
    conn = sqlite3.connect(_db_path(), check_same_thread=False, timeout=15)
    for p in ("PRAGMA journal_mode=WAL","PRAGMA synchronous=NORMAL","PRAGMA cache_size=-8192",
              "PRAGMA foreign_keys=ON","PRAGMA temp_store=MEMORY"):
        conn.execute(p)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback(); _log_exc("DB", e); raise
    except Exception as e:
        conn.rollback(); _log_exc("DB unexpected", e); raise
    finally:
        conn.close()


def init_db():
    """Create schema + run column-add migrations + ensure master user exists."""
    log = _log()
    log.info("DB init: %s", _db_path())
    try:
        with db() as c:
            c.executescript(_SCHEMA)
        with db() as c:
            for table, cols in _MIGRATIONS.items():
                try:
                    existing = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
                    for col, defn in cols:
                        if col not in existing:
                            try:
                                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
                                log.info("Migrated %s.%s", table, col)
                            except sqlite3.OperationalError:
                                pass
                except Exception as e:
                    _log_exc(f"migration {table}", e)
        with db() as c:
            c.execute("INSERT OR IGNORE INTO users(id,name) VALUES('master','Арчибальд')")
            # Idempotent rename: legacy installs called the master 'Хозяин'.
            c.execute("UPDATE users SET name='Арчибальд' WHERE id='master' AND name='Хозяин'")
            # v9.1: one-time backfill — on the migration from single-counter to
            # dual-counter, existing users' lifetime count is in `msg_count`.
            # Copy it into `total_msg_count` so we don't lose history.
            # Idempotent: only touches rows where total is still zero but session has progress.
            c.execute("UPDATE user_state SET total_msg_count=msg_count "
                      "WHERE total_msg_count=0 AND msg_count>0")
        _migrate_legacy()
        log.info("DB ready")
    except Exception as e:
        _log_exc("init_db", e); raise


def _migrate_legacy():
    """One-time JSON→SQLite backfill for legacy DigitalHuman installs.
    Idempotent — uses INSERT OR IGNORE."""
    from main import _read_json
    log = _log()
    base = _base_dir()
    uj = os.path.join(base, "users.json")
    if os.path.exists(uj):
        try:
            for u in _read_json(uj):
                with db() as c:
                    c.execute("INSERT OR IGNORE INTO users(id,name) VALUES(?,?)", (u["id"], u["name"]))
        except Exception as e:
            _log_exc("users.json migration", e)
    for fn in os.listdir(base):
        if fn.startswith("state_") and fn.endswith(".json"):
            uid = fn[6:-5]
            try:
                s = _read_json(os.path.join(base, fn))
                with db() as c:
                    c.execute(
                        "INSERT OR IGNORE INTO user_state(user_id,mood,trust,fear,attachment,msg_count) VALUES(?,?,?,?,?,?)",
                        (uid, s.get("mood", 0.5), s.get("trust", 0.5), s.get("fear", 0.4),
                         s.get("attachment", 0.3), s.get("message_count", s.get("msg_count", 0))))
            except Exception as e:
                _log_exc(f"state migration {fn}", e)
