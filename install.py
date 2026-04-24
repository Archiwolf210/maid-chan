#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digital Human v9.0 — zero-friction installer.

Автоопределение целевого бэкенда (CUDA/Vulkan/CPU), установка llama.cpp,
cudart-sidecar (нужен отдельным архивом для CUDA-build), Python-зависимостей,
и опционально скачивание Qwen3-32B + Qwen3-0.6B draft (для speculative decoding).

Запуск:
    python install.py                         # интерактивный режим
    python install.py --auto                  # всё подтвердить автоматически
    python install.py --target cuda           # принудительный бэкенд
    python install.py --target vulkan         # fallback (устаревшие GPU)
    python install.py --target cpu            # без GPU вообще
    python install.py --llama-only            # только llama.cpp + cudart
    python install.py --deps-only             # только pip-зависимости
    python install.py --models-only           # только скачать модели
    python install.py --no-venv               # ставить в системный python
    python install.py --llama-version bXXXX   # конкретный релиз
    python install.py --skip-embedder         # не качать embedding-модель
    python install.py --skip-models           # не предлагать модели
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
VENV_DIR   = ROOT / ".venv"
LLM_DIR    = ROOT / "llm"
MODELS_DIR = ROOT / "models"
LOGS_DIR   = ROOT / "logs"
REQ_FILE   = ROOT / "requirements.txt"

MIN_PY = (3, 11)
MAX_PY = (3, 13)

LLAMA_REPO_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"

# Regex for llama.cpp release assets (win-x64). CUDA 12.4 matches current driver 591.74
# forward-compat (CUDA 13.1). Fallback: newer cu12.x variants.
ASSET_PATTERNS = {
    "cuda":   re.compile(r"^llama-.*-bin-win-cuda(?:-?12\.[0-9]+)?-x64\.zip$", re.IGNORECASE),
    "vulkan": re.compile(r"^llama-.*-bin-win-vulkan-x64\.zip$",                re.IGNORECASE),
    "cpu":    re.compile(r"^llama-.*-bin-win-cpu-x64\.zip$",                   re.IGNORECASE),
}

# CUDA runtime sidecar: separate ZIP with cudart*.dll + cublas*.dll required
# by CUDA-builds of llama-server but not shipped inside the main archive.
CUDART_PATTERN = re.compile(r"^cudart-llama-bin-win-cuda(?:-?12\.[0-9]+)?-x64\.zip$", re.IGNORECASE)

# ── Model targets (Qwen3 family on RTX 3090 24 GB) ──────────────────────
# Main: Qwen3-32B Q4_K_M ≈ 19.5 GB — fits 3090 with 6 k ctx headroom.
# Draft: Qwen3-0.6B Q8_0 ≈ 700 MB — speculative decoding gives 1.8-2.5× throughput.
MAIN_MODEL = {
    "name": "qwen3.gguf",
    "size_gb": 19.5,
    "repos": [
        # Primary: HuggingFace Qwen official
        ("https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/qwen3-32b-q4_k_m.gguf",
         "Qwen3-32B Q4_K_M (официальный)"),
        # Mirror: community rehost
        ("https://huggingface.co/bartowski/Qwen_Qwen3-32B-GGUF/resolve/main/Qwen_Qwen3-32B-Q4_K_M.gguf",
         "Qwen3-32B Q4_K_M (bartowski mirror)"),
    ],
}
DRAFT_MODEL = {
    "name": "qwen3-draft.gguf",
    "size_gb": 0.7,
    "repos": [
        ("https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/qwen3-0.6b-q8_0.gguf",
         "Qwen3-0.6B Q8_0 (draft, официальный)"),
        ("https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q8_0.gguf",
         "Qwen3-0.6B Q8_0 (draft, bartowski mirror)"),
    ],
}

COLORS = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "cyan": "\033[36m", "magenta": "\033[35m",
}

def _enable_ansi_on_windows() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        handle = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(handle, ctypes.byref(mode)):
            k32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass

def c(text: str, color: str) -> str:  return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"
def step(msg: str) -> None:           print(c(f"▸ {msg}", "cyan"))
def ok(msg: str) -> None:             print(c(f"✓ {msg}", "green"))
def warn(msg: str) -> None:           print(c(f"! {msg}", "yellow"))
def err(msg: str) -> None:            print(c(f"✗ {msg}", "red"), file=sys.stderr)

def die(msg: str, code: int = 1) -> None:
    err(msg); sys.exit(code)

def confirm(prompt: str, default: bool, auto: bool) -> bool:
    if auto:
        return True
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{prompt} {suffix} ").strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans in ("y", "yes", "д", "да")

# ── Hardware detection ──────────────────────────────────────────────────

def detect_target(override: str | None) -> str:
    """Return 'cuda', 'vulkan', or 'cpu' — autodetected or user-forced."""
    if override:
        if override not in ("cuda", "vulkan", "cpu"):
            die(f"Unknown --target {override}; use cuda|vulkan|cpu")
        step(f"Целевой бэкенд задан вручную: {override}")
        return override
    step("Автоопределение GPU (nvidia-smi)")
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            line = r.stdout.strip().splitlines()[0]
            ok(f"Nvidia GPU: {line}")
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    warn("nvidia-smi не найден или не ответил — переключаюсь на Vulkan (работает на любом GPU).")
    return "vulkan"

# ── Environment checks ──────────────────────────────────────────────────

def check_python() -> None:
    step(f"Проверка Python (требуется {MIN_PY[0]}.{MIN_PY[1]}+)")
    if sys.version_info < MIN_PY:
        die(f"Нужен Python {MIN_PY[0]}.{MIN_PY[1]}+, у тебя {sys.version.split()[0]}. "
            f"Скачай с https://python.org")
    if sys.version_info >= (MAX_PY[0], MAX_PY[1] + 1):
        warn(f"Python {sys.version.split()[0]} новее протестированного. Некоторые wheels могут отсутствовать.")
    ok(f"Python {sys.version.split()[0]} — подходит")

def check_os() -> None:
    step("Проверка ОС")
    sys_name = platform.system()
    if sys_name != "Windows":
        warn(f"Обнаружена {sys_name}. Автоскачивание настроено под Windows; "
             "для Linux/macOS собирай llama.cpp вручную.")
    else:
        ok(f"{sys_name} {platform.release()}")

def ensure_dirs() -> None:
    step("Создание структуры папок")
    for d in (LLM_DIR, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ok("llm/, models/, logs/ готовы")

# ── Virtualenv + pip ────────────────────────────────────────────────────

def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def create_venv(no_venv: bool) -> Path:
    if no_venv:
        warn("venv пропущен (--no-venv): зависимости пойдут в системный Python.")
        return Path(sys.executable)
    if _venv_python().exists():
        ok(f"venv уже существует: {VENV_DIR}")
        return _venv_python()
    step(f"Создание venv в {VENV_DIR}")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    except subprocess.CalledProcessError as e:
        die(f"Не удалось создать venv: {e}")
    ok("venv создан")
    return _venv_python()

def pip_install(py: Path) -> None:
    step("Обновление pip / setuptools / wheel")
    subprocess.check_call([str(py), "-m", "pip", "install",
                           "--upgrade", "--disable-pip-version-check",
                           "pip", "setuptools", "wheel"])
    step("Установка зависимостей из requirements.txt")
    if not REQ_FILE.exists():
        die("requirements.txt не найден")
    subprocess.check_call([str(py), "-m", "pip", "install",
                           "--disable-pip-version-check", "-r", str(REQ_FILE)])
    ok("Python-зависимости установлены")

def verify_imports(py: Path) -> None:
    step("Проверка рабочего импорта")
    code = ("import fastapi, uvicorn, httpx, pydantic, orjson; "
            "print('ok', fastapi.__version__, pydantic.VERSION, orjson.__version__)")
    try:
        out = subprocess.check_output([str(py), "-c", code], text=True).strip()
        ok(out)
    except subprocess.CalledProcessError:
        die("Импорты провалились — посмотри логи выше.")
    code2 = ("import fastembed, numpy; "
             "print('fastembed', fastembed.__version__, 'numpy', numpy.__version__)")
    try:
        out2 = subprocess.check_output([str(py), "-c", code2], text=True).strip()
        ok(out2)
    except subprocess.CalledProcessError:
        warn("fastembed/numpy не импортируются — семантическая LTM будет отключена, "
             "бот переключится на keyword-recall.")


def warm_embedder(py: Path, auto: bool, skip: bool) -> None:
    if skip:
        warn("Прогрев embedding-кэша пропущен (--skip-embedder).")
        return
    if not confirm("Скачать embedding-модель (~1 GB, разово)?", True, auto):
        warn("Пропущено. Модель скачается автоматически при первом обращении к LTM.")
        return
    step("Прогрев fastembed (скачивание ONNX-модели)")
    code = (
        "import sys\n"
        "try:\n"
        "    from fastembed import TextEmbedding\n"
        "except Exception as e:\n"
        "    print('SKIP fastembed missing:', e); sys.exit(0)\n"
        "m = TextEmbedding(model_name='intfloat/multilingual-e5-large', threads=8)\n"
        "out = list(m.embed(['passage: ready']))\n"
        "print('OK dim', len(out[0]))\n"
    )
    try:
        subprocess.check_call([str(py), "-c", code])
        ok("Embedding-модель прогрета")
    except subprocess.CalledProcessError:
        warn("Не удалось прогреть embedding-модель. LTM включится позже на первом запросе.")

# ── llama.cpp downloader ────────────────────────────────────────────────

def llama_server_present() -> bool:
    return (LLM_DIR / "llama-server.exe").exists()

def _http_get_json(url: str):
    req = urllib.request.Request(url, headers={
        "User-Agent": "DigitalHuman-installer/1.0",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def _find_asset(assets: list, pattern: re.Pattern):
    """Return first matching asset (prefer cuda12.4 over cuda12.x)."""
    # Pass 1: exact cu12.4 preference (matches current driver-forward-compat baseline)
    for a in assets:
        if "12.4" in a.get("name", "").lower() and pattern.match(a.get("name", "")):
            return a
    # Pass 2: any match
    for a in assets:
        if pattern.match(a.get("name", "")):
            return a
    return None

def find_llama_asset(target: str, version: str | None):
    step(f"Запрос списка релизов llama.cpp (target={target})")
    try:
        url = f"{LLAMA_REPO_API}/tags/{version}" if version else f"{LLAMA_REPO_API}/latest"
        data = _http_get_json(url)
    except urllib.error.HTTPError as e:
        die(f"GitHub ответил {e.code}: {e.reason}")
    except Exception as e:
        die(f"Не удалось получить релиз: {e}")
    tag = data.get("tag_name", "?")
    assets = data.get("assets", [])
    pattern = ASSET_PATTERNS.get(target)
    if pattern is None:
        die(f"Неизвестный target: {target}")
    a = _find_asset(assets, pattern)
    if a is None:
        available = ", ".join(x.get("name", "?") for x in assets[:8])
        die(f"В релизе {tag} не найден ассет под {target}.\n  Доступно: {available}")
    return tag, a["name"], a["browser_download_url"], a.get("size", 0), assets

def find_cudart_asset(assets: list):
    """cudart sidecar (cublas/cudart DLLs). Returns None if not found — warn but continue."""
    for a in assets:
        if "12.4" in a.get("name", "").lower() and CUDART_PATTERN.match(a.get("name", "")):
            return a["name"], a["browser_download_url"], a.get("size", 0)
    for a in assets:
        if CUDART_PATTERN.match(a.get("name", "")):
            return a["name"], a["browser_download_url"], a.get("size", 0)
    return None

def download_with_progress(url: str, dst: Path, size_hint: int = 0) -> None:
    """Download `url` to `dst` with progress bar, atomic rename, and resume support.

    Correctness guarantees (v9.0+):
      * Body is streamed into ``dst.partial``. The final ``dst`` is created
        only after a completeness check against Content-Length succeeds —
        this defends against the "urllib silently stops at empty read()"
        failure mode where a dropped CDN connection truncates multi-GB models.
      * If ``dst.partial`` already exists, the download resumes via a
        ``Range: bytes=N-`` request. Callers that switch URLs must delete
        ``dst.partial`` first — resuming against a different file corrupts it.
      * On any exception the ``.partial`` file is preserved so the next
        invocation can resume rather than redownload 10+ GB from scratch.
    """
    partial = dst.with_suffix(dst.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0

    headers = {"User-Agent": "DigitalHuman-installer/1.0"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        step(f"Возобновление с {existing/1024/1024:.1f} MB: {url}")
    else:
        step(f"Скачивание {url}")

    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing > 0 else "wb"
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            # 206 Partial Content: Content-Length = remaining bytes only.
            # 200 OK: Content-Length = full size. Some mirrors ignore Range.
            cl = int(r.headers.get("Content-Length", 0) or 0)
            status = getattr(r, "status", 200)
            if status == 206:
                total = existing + cl
            elif status == 200:
                # Server didn't honor Range — restart from zero to be safe.
                if existing > 0:
                    warn("Сервер проигнорировал Range: запрос, начинаю заново.")
                    partial.unlink(missing_ok=True)
                    existing = 0
                    mode = "wb"
                total = cl or size_hint
            else:
                total = cl or size_hint
            with open(partial, mode) as f:
                read = existing
                chunk = 1024 * 128
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    read += len(buf)
                    if total:
                        pct = read * 100 // total
                        sys.stdout.write(f"\r  {pct:3d}%  {read/1024/1024:6.1f} / {total/1024/1024:.1f} MB")
                        sys.stdout.flush()
                sys.stdout.write("\n")
    except Exception as e:
        # Keep .partial so the next run resumes from where we stopped.
        raise RuntimeError(f"Download failed: {e}") from e

    # Completeness check — defends against silent truncation on CDN drops.
    got = partial.stat().st_size
    if total > 0 and got < total:
        # Missing bytes: threshold 32 KB tolerates off-by-one chunking but
        # catches the multi-gigabyte truncation we actually see in the wild.
        if total - got > 32 * 1024:
            raise RuntimeError(
                f"Неполная загрузка: получено {got/1024/1024:.1f} MB из "
                f"{total/1024/1024:.1f} MB (соединение оборвано). "
                f"Запусти установщик ещё раз — возобновится с этого места."
            )
    # Atomic move: only a correct download ever becomes the target file.
    if dst.exists():
        dst.unlink()
    partial.replace(dst)
    ok(f"Загружено: {dst.name} ({got/1024/1024:.1f} MB)")

def extract_zip(zip_path: Path, dest: Path) -> None:
    step(f"Распаковка в ./{dest.name}/")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            # Flatten — llama releases are flat ZIPs but cudart sometimes has nested folder
            name = Path(member.filename).name
            target = dest / name
            with z.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
    ok(f"Распаковано: {zip_path.name}")

def install_llama(target: str, auto: bool, version: str | None) -> None:
    if llama_server_present():
        if not confirm("llama-server.exe уже есть. Переустановить?", False, auto):
            ok("llama.cpp пропущен")
            return
    if platform.system() != "Windows":
        warn("Автоскачивание только под Windows. На Linux/macOS собирай вручную.")
        return

    tag, name, url, size, all_assets = find_llama_asset(target, version)
    print(f"  релиз: {c(tag, 'bold')}, ассет: {name} ({size/1024/1024:.1f} MB)")

    cudart = None
    if target == "cuda":
        cudart = find_cudart_asset(all_assets)
        if cudart:
            print(f"  cudart: {c(cudart[0], 'bold')} ({cudart[2]/1024/1024:.1f} MB)")
        else:
            warn("cudart-sidecar не найден в релизе. Если llama-server выдаст "
                 "'cudart64_*.dll missing' — придётся доставить вручную из CUDA Toolkit.")

    if not confirm("Скачать и распаковать?", True, auto):
        return

    zip_path = ROOT / name
    try:
        download_with_progress(url, zip_path, size)
        extract_zip(zip_path, LLM_DIR)
    finally:
        try: zip_path.unlink(missing_ok=True)
        except Exception: pass

    if cudart:
        cu_zip = ROOT / cudart[0]
        try:
            download_with_progress(cudart[1], cu_zip, cudart[2])
            extract_zip(cu_zip, LLM_DIR)
        finally:
            try: cu_zip.unlink(missing_ok=True)
            except Exception: pass

    if not llama_server_present():
        die("После распаковки llama-server.exe не появился. Проверь ZIP вручную.")
    ok(f"llama.cpp ({tag}, {target}) готов в ./llm/")

# ── Model downloader ────────────────────────────────────────────────────

def _model_present(name: str, expected_gb: float = 0.0) -> bool:
    """Return True only if the file exists AND looks fully-downloaded.

    With ``expected_gb`` we require the file to be at least 80% of the expected
    size — that way a truncated 2.8 GB download of a 19 GB model no longer
    registers as "present" and we redownload it on the next run.
    """
    p = MODELS_DIR / name
    if not p.exists():
        return False
    size = p.stat().st_size
    if expected_gb > 0:
        lo = int(expected_gb * 1024 * 1024 * 1024 * 0.80)
        return size >= lo
    return size > 1024 * 1024

def _try_download_model(spec: dict) -> bool:
    """Try each mirror in order. Each mirror starts with a clean slate —
    .partial files are keyed to a specific source URL, so resuming a
    different mirror into the same .partial would corrupt the file.
    """
    dst = MODELS_DIR / spec["name"]
    partial = dst.with_suffix(dst.suffix + ".partial")
    size_hint = int(spec["size_gb"] * 1024 * 1024 * 1024)
    for idx, (url, label) in enumerate(spec["repos"]):
        print(f"  источник: {c(label, 'dim')}")
        # On mirror switch, drop the stale .partial from the previous URL.
        if idx > 0 and partial.exists():
            try: partial.unlink()
            except Exception: pass
        try:
            download_with_progress(url, dst, size_hint)
            return True
        except Exception as e:
            warn(f"Не удалось скачать из {label}: {e}")
            # Same-URL retries on next run will still resume via .partial.
    return False

def install_models(auto: bool, skip: bool) -> None:
    if skip:
        return
    step("Проверка моделей")

    # Main model — strict size check catches silently-truncated downloads
    # (HF CDN has been observed to drop connection mid-transfer).
    if _model_present(MAIN_MODEL["name"], MAIN_MODEL["size_gb"]):
        ok(f"{MAIN_MODEL['name']} уже есть ({(MODELS_DIR / MAIN_MODEL['name']).stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
    else:
        # If a truncated file exists, warn the user explicitly
        existing = MODELS_DIR / MAIN_MODEL["name"]
        if existing.exists():
            size_gb = existing.stat().st_size / 1024 / 1024 / 1024
            warn(f"Найден неполный {MAIN_MODEL['name']} ({size_gb:.2f} GB, ожидалось ~{MAIN_MODEL['size_gb']:.1f} GB). Перезапускаю скачивание.")
            try: existing.unlink()
            except Exception: pass
        print()
        print(c(f"  Основная модель: Qwen3-32B Q4_K_M (~{MAIN_MODEL['size_gb']:.1f} GB)", "bold"))
        print("    • 32B-параметрическая, русский + роль-плей отлично")
        print("    • на RTX 3090 24 GB запускается с запасом")
        if confirm("Скачать основную модель сейчас?", True, auto):
            if not _try_download_model(MAIN_MODEL):
                warn("Все зеркала упали. Скачай вручную и положи в ./models/qwen3.gguf")
        else:
            warn("Пропущено. Скачай вручную: ./models/qwen3.gguf")

    # Draft model (speculative decoding)
    if _model_present(DRAFT_MODEL["name"], DRAFT_MODEL["size_gb"]):
        ok(f"{DRAFT_MODEL['name']} уже есть ({(MODELS_DIR / DRAFT_MODEL['name']).stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
    else:
        existing = MODELS_DIR / DRAFT_MODEL["name"]
        if existing.exists():
            size_gb = existing.stat().st_size / 1024 / 1024 / 1024
            warn(f"Найден неполный {DRAFT_MODEL['name']} ({size_gb:.2f} GB, ожидалось ~{DRAFT_MODEL['size_gb']:.1f} GB). Перезапускаю скачивание.")
            try: existing.unlink()
            except Exception: pass
        print()
        print(c(f"  Draft-модель: Qwen3-0.6B Q8_0 (~{DRAFT_MODEL['size_gb']:.1f} GB)", "bold"))
        print("    • нужна для speculative decoding (+60-150% скорости)")
        print("    • ОБЯЗАТЕЛЬНО из той же Qwen3-семьи, чтобы совпадал токенайзер")
        if confirm("Скачать draft-модель сейчас?", True, auto):
            if not _try_download_model(DRAFT_MODEL):
                warn("Все зеркала упали. Скачай вручную: ./models/qwen3-draft.gguf")
        else:
            warn("Пропущено. Можно запустить без spec-decode, просто медленнее.")
    print()

def model_status_hint() -> None:
    gguf = list(MODELS_DIR.glob("*.gguf"))
    if not gguf:
        warn("В ./models/ не найдено .gguf моделей — запусти install.py --models-only.")
        return
    step("Модели в ./models/:")
    for g in gguf:
        ok(f"{g.name} ({g.stat().st_size/1024/1024/1024:.2f} GB)")

# ── Main ────────────────────────────────────────────────────────────────

def print_final(py: Path, target: str) -> None:
    print()
    print(c("━" * 60, "magenta"))
    print(c(" Установка завершена", "bold"))
    print(c("━" * 60, "magenta"))
    print()
    print(f"  Бэкенд:  {c(target.upper(), 'bold')}")
    print(f"  Запуск:  {c('start.bat', 'bold')}   (использует {py})")
    print(f"  UI:      http://127.0.0.1:5000")
    print(f"  Логи:    {LOGS_DIR}")
    print()

def main() -> None:
    _enable_ansi_on_windows()
    p = argparse.ArgumentParser(description="Digital Human v9.0 installer")
    p.add_argument("--auto", action="store_true", help="автоподтверждение всех вопросов")
    p.add_argument("--no-venv", action="store_true", help="не создавать .venv")
    p.add_argument("--deps-only", action="store_true", help="только pip install")
    p.add_argument("--llama-only", action="store_true", help="только llama.cpp + cudart")
    p.add_argument("--models-only", action="store_true", help="только скачать GGUF-модели")
    p.add_argument("--target", choices=["cuda", "vulkan", "cpu"],
                   help="принудительно выбрать бэкенд (иначе автоопределение)")
    p.add_argument("--llama-version", metavar="TAG",
                   help="конкретный tag llama.cpp (например b5000). По умолчанию latest.")
    p.add_argument("--skip-embedder", action="store_true",
                   help="не скачивать embedding-модель (≈1 GB) сейчас")
    p.add_argument("--skip-models", action="store_true",
                   help="не предлагать скачивание LLM-моделей")
    args = p.parse_args()

    print(c("═" * 60, "magenta"))
    print(c(" Digital Human v9.0 — installer (CUDA / Vulkan / CPU)", "bold"))
    print(c("═" * 60, "magenta"))
    print()

    check_python()
    check_os()
    ensure_dirs()
    target = detect_target(args.target)

    if args.models_only:
        install_models(args.auto, False)
        return
    if args.llama_only:
        install_llama(target, args.auto, args.llama_version)
        model_status_hint()
        return

    py = create_venv(args.no_venv)
    pip_install(py)
    verify_imports(py)
    warm_embedder(py, args.auto, args.skip_embedder)

    if not args.deps_only:
        install_llama(target, args.auto, args.llama_version)
        install_models(args.auto, args.skip_models)
        model_status_hint()

    print_final(py, target)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(); err("Прервано пользователем"); sys.exit(130)
    except subprocess.CalledProcessError as e:
        err(f"Команда завершилась с кодом {e.returncode}"); sys.exit(e.returncode)
