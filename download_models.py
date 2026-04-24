"""
Fast model downloader for Digital Human v9.0
==============================================

Uses huggingface_hub.hf_hub_download() with hf_transfer (Rust backend,
parallel chunks). Typically 5-10x faster than urllib on throttled HF
connections, because HF rate-limits per-TCP-connection, not aggregate.

Idempotent: if final files already exist with correct size (verified
against HF API when reachable), exits clean.

Sizes are reported in **decimal GB / MB** (1 GB = 10^9 bytes) to match
the tqdm progress bar and the file sizes shown on huggingface.co.
This avoids the GB-vs-GiB confusion where a fully-downloaded 19.76 GB
file looks like "18.40 GiB" and seems truncated.

Run via download_models.bat (which finds the venv python).
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

MAIN_TARGET = MODELS / "qwen3.gguf"
DRAFT_TARGET = MODELS / "qwen3-draft.gguf"

MAIN_REPO = "bartowski/Qwen_Qwen3-32B-GGUF"
MAIN_FILE = "Qwen_Qwen3-32B-Q4_K_M.gguf"
# Loose lower bound used only when HF API is unreachable.
# Real check is exact-byte comparison against HF API.
MAIN_MIN_BYTES = 15_000_000_000  # 15 GB; full size ~19.76 GB

DRAFT_REPO = "bartowski/Qwen_Qwen3-0.6B-GGUF"
DRAFT_FILE = "Qwen_Qwen3-0.6B-Q8_0.gguf"
DRAFT_MIN_BYTES = 500_000_000  # 500 MB; full size ~750 MB


# ---------- formatting ----------

def fmt_size(n: int) -> str:
    """Human-readable size in decimal units (matches tqdm and HF site)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


# ---------- pip helpers ----------

def pip_install(pkg: str) -> None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--quiet", "--disable-pip-version-check", pkg,
    ])


def ensure_huggingface_hub() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("[INFO] Installing huggingface_hub...")
        pip_install("huggingface_hub>=0.24")


def ensure_hf_transfer() -> bool:
    """Returns True if hf_transfer is usable. Falls back to urllib otherwise."""
    try:
        import hf_transfer  # noqa: F401
        return True
    except ImportError:
        pass
    print("[INFO] Installing hf_transfer (Rust backend, parallel chunks)...")
    try:
        pip_install("hf_transfer")
        import hf_transfer  # noqa: F401
        return True
    except Exception as e:
        print(f"[WARN] hf_transfer install failed: {e}")
        print("       Will fall back to single-threaded download (slower).")
        return False


# ---------- HF API: get expected file size ----------

def get_expected_size(repo: str, filename: str) -> Optional[int]:
    """
    Ask HF API for the exact byte size of `filename` in `repo`.
    Returns None if the API call fails (offline, rate-limited, etc.) —
    caller should fall back to a coarse min-bytes check.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.get_paths_info(repo_id=repo, paths=[filename])
        if not info:
            return None
        sz = getattr(info[0], "size", None)
        return int(sz) if sz else None
    except Exception as e:
        print(f"[WARN] Could not query HF API for expected size: {e}")
        return None


def verify_size(target: Path, expected: Optional[int], min_bytes: int) -> tuple[bool, str]:
    """
    Returns (ok, message). If `expected` known, requires exact match.
    Otherwise falls back to >= min_bytes (loose check).
    """
    actual = target.stat().st_size
    if expected is not None:
        if actual == expected:
            return True, f"{fmt_size(actual)} (exact match with HF)"
        if actual < expected:
            return False, (f"{fmt_size(actual)} on disk, HF says {fmt_size(expected)} "
                           f"(missing {fmt_size(expected - actual)})")
        # actual > expected: file is somehow larger than HF reports; weird but not "incomplete".
        return True, f"{fmt_size(actual)} (HF expected {fmt_size(expected)} — accepting larger file)"
    # No HF data; fall back to coarse threshold.
    if actual >= min_bytes:
        return True, f"{fmt_size(actual)} (HF API unreachable, accepted by min-size check)"
    return False, f"{fmt_size(actual)} < min {fmt_size(min_bytes)} (HF API unreachable)"


# ---------- main download routine ----------

def fetch(repo: str, filename: str, target: Path, min_bytes: int) -> None:
    expected = get_expected_size(repo, filename)
    if expected is not None:
        print(f"[INFO] HF reports expected size: {fmt_size(expected)}")

    # Idempotency: skip if already complete.
    if target.exists():
        ok, msg = verify_size(target, expected, min_bytes)
        if ok:
            print(f"[OK] {target.name} already present — {msg}")
            return
        print(f"[WARN] {target.name} is incomplete: {msg}")
        print(f"       Removing and redownloading...")
        target.unlink()

    from huggingface_hub import hf_hub_download

    print(f"[INFO] Downloading {filename}")
    print(f"       from {repo}")
    print(f"       hf_transfer={'on' if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1' else 'off'}")

    cached = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(target.parent),
    )

    src = Path(cached)
    # File arrives with HF filename; rename to our convention.
    if src.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(src), str(target))

    ok, msg = verify_size(target, expected, min_bytes)
    if not ok:
        raise RuntimeError(f"Download incomplete: {msg}")
    print(f"[OK] {target.name} ready — {msg}")


def main() -> None:
    ensure_huggingface_hub()
    if ensure_hf_transfer():
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    print()
    print("=" * 60)
    print("  Downloading Qwen3-32B Q4_K_M (~19.76 GB)")
    print("=" * 60)
    fetch(MAIN_REPO, MAIN_FILE, MAIN_TARGET, MAIN_MIN_BYTES)

    print()
    print("=" * 60)
    print("  Downloading Qwen3-0.6B Q8_0 draft (~750 MB)")
    print("=" * 60)
    fetch(DRAFT_REPO, DRAFT_FILE, DRAFT_TARGET, DRAFT_MIN_BYTES)

    print()
    print("=" * 60)
    print("  Done. Run start.bat next.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Re-run download_models.bat to resume.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print("Re-run download_models.bat to retry.")
        sys.exit(1)
