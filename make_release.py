"""
make_release.py -- build a clean distributable ZIP of Digital Human v8.

Produces: dist/DigitalHuman_v8.3_YYYYMMDD.zip

Rules:
 * Source code + docs + install.py + start.bat are included.
 * User state (memory.db, app_token.txt, logs/) is EXCLUDED -- shipping it
   would leak the developer's personal chat memory and CSRF token.
 * Large binaries (llm/*.exe, llm/*.dll) are EXCLUDED -- the end user runs
   `python install.py` which downloads the latest Vulkan build of
   llama.cpp (~200 MB) from GitHub releases.
 * GGUF models are EXCLUDED -- user downloads their own (licensing + size).
 * __pycache__, .venv, dist/, editor junk are EXCLUDED.

Usage:
    python make_release.py                 # default output to ./dist/
    python make_release.py --out build/    # custom output dir
    python make_release.py --name foo.zip  # custom archive name
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Files / directories that form the shippable project.
INCLUDE_FILES = [
    "main.py",
    "install.py",
    "check_server.py",
    "make_release.py",
    "start.bat",
    "run_llama.bat",
    "requirements.txt",
    "config.json",
    "README.md",
    "AUDIT.md",
    "TAILSCALE_SETUP.md",
    ".gitignore",
]
INCLUDE_DIRS = ["web", "llm", "models"]

# Glob patterns to drop even if they live inside INCLUDE_DIRS.
EXCLUDE_GLOBS = [
    # state / secrets
    "app_token.txt",
    "memory.db",
    "memory.db-*",
    # logs
    "logs",
    "logs/*",
    "*.log",
    # python junk
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    # virtualenvs
    ".venv",
    "venv",
    "env",
    # editor / OS
    ".vscode",
    ".idea",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "*.swp",
    "*.swo",
    # release output itself
    "dist",
    "release",
    "*.zip",
    # heavy binaries (user installs via install.py)
    "llm/*.exe",
    "llm/*.dll",
    "llm/*.so",
    "llm/*.dylib",
    # models (user supplies their own GGUF)
    "models/*.gguf",
    "models/*.bin",
    "models/*.safetensors",
]

# Small placeholder files we DO want shipped so the layout is obvious.
FORCE_INCLUDE = {
    "llm/PLACE_LLAMA_SERVER_HERE.txt",
    "models/PLACE_MODEL_HERE.txt",
}


def _rel(path: Path) -> str:
    """Posix-style relative path from ROOT -- used for matching and archive names."""
    return path.relative_to(ROOT).as_posix()


def _excluded(relpath: str) -> bool:
    if relpath in FORCE_INCLUDE:
        return False
    for pat in EXCLUDE_GLOBS:
        # Match against full relative path AND against each segment.
        if fnmatch.fnmatch(relpath, pat):
            return True
        for seg in relpath.split("/"):
            if fnmatch.fnmatch(seg, pat):
                return True
    return False


def _gather_files() -> list[Path]:
    chosen: list[Path] = []

    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.is_file() and not _excluded(_rel(p)):
            chosen.append(p)

    for d in INCLUDE_DIRS:
        root = ROOT / d
        if not root.is_dir():
            continue
        for cur, dirs, files in os.walk(root):
            cur_path = Path(cur)
            # prune excluded dirs in-place so os.walk doesn't descend into them
            dirs[:] = [
                x for x in dirs
                if not _excluded(_rel(cur_path / x))
            ]
            for f in files:
                fp = cur_path / f
                if _excluded(_rel(fp)):
                    continue
                chosen.append(fp)

    return chosen


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a clean release ZIP.")
    ap.add_argument("--out", default="dist", help="output directory (default: dist)")
    ap.add_argument("--name", default=None,
                    help="archive filename (default: DigitalHuman_v8.2_YYYYMMDD.zip)")
    ap.add_argument("--list-only", action="store_true",
                    help="print what would be included and exit")
    args = ap.parse_args()

    files = _gather_files()
    files.sort()

    if args.list_only:
        for f in files:
            print(_rel(f))
        print(f"\n-- {len(files)} files --", file=sys.stderr)
        return 0

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.name:
        archive_name = args.name
    else:
        stamp = datetime.now().strftime("%Y%m%d")
        archive_name = f"DigitalHuman_v8.3_{stamp}.zip"
    archive_path = out_dir / archive_name

    if archive_path.exists():
        archive_path.unlink()

    total_bytes = 0
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for f in files:
            arcname = f"DigitalHuman_v8/{_rel(f)}"
            zf.write(f, arcname)
            total_bytes += f.stat().st_size

    size_on_disk = archive_path.stat().st_size
    print(f"OK  -- {archive_path}")
    print(f"     {len(files)} files, raw {_format_size(total_bytes)}, "
          f"archive {_format_size(size_on_disk)}")
    print("     Excluded: memory.db, app_token.txt, logs/, llm/*.exe, models/*.gguf, __pycache__, .venv")
    print("     End user runs:  python install.py  (fetches llama.cpp + sets up venv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
