@echo off
setlocal enabledelayedexpansion
title Digital Human v9.0

rem === Tuned for RTX 3090 24 GB + Xeon E5-2666v3 (10C/20T) + 32 GB DDR4 ===
rem === Model: Qwen3-32B Q4_K_M + Qwen3-0.6B draft (speculative decoding) ===

rem -- Main inference parameters (can be overridden by user) --------------
set "MODEL_FILE=qwen3.gguf"
set "DRAFT_MODEL=qwen3-draft.gguf"
set "LLM_PORT=8080"
set "APP_PORT=5000"

rem 999 = offload ALL layers to GPU. Qwen3-32B Q4_K_M has 64 layers;
rem llama.cpp clamps to actual count, and the whole model fits in 24 GB.
set "GPU_LAYERS=999"
set "DRAFT_GPU_LAYERS=99"
rem draft-max=4 picked because measured acceptance rate is ~0.40-0.55.
rem At 0.5 acc, expected accepted run = 2 tokens; max=4 keeps wasted
rem draft compute low while still capturing good runs. Max=8 was greedy.
set "DRAFT_MAX=4"

rem 16384 chosen so main(q8_0 KV) + draft(f16 KV) fit fully in 24 GB
rem WITHOUT spilling into shared system memory (the silent 30x slowdown).
rem Math: main 18424 + main_KV 2176 + draft 604 + draft_KV 1792 + ~700 MB
rem compute buffers ~= 23.7 GB out of 24.5 GB. ~800 MB headroom.
rem Drop to 12288 if you still see <10 t/s on long contexts.
set "CTX_SIZE=16384"
set "UBATCH=512"

rem q8_0 KV-cache: halves VRAM footprint of context at ~0.1% quality loss.
set "CACHE_TYPE_K=q8_0"
set "CACHE_TYPE_V=q8_0"

cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem -- Autodetect physical cores (not logical). Xeon E5-2666v3 = 10 phys.
rem Fallback to 10 if WMIC returns garbage (Windows 11 sometimes deprecates wmic).
set "THREADS=10"
for /f "skip=1 tokens=*" %%A in ('wmic cpu get NumberOfCores 2^>nul') do (
    set "tmp=%%A"
    if defined tmp (
        set "tmp=!tmp: =!"
        if not "!tmp!"=="" (
            set /a _n=!tmp! 2>nul
            if !_n! GTR 0 if !_n! LSS 64 set "THREADS=!_n!"
        )
    )
)
echo [INFO] Physical CPU cores detected: !THREADS!

rem -- 1. Python ----------------------------------------------------------
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PY=%ROOT%\.venv\Scripts\python.exe"
    echo [INFO] Using venv: !PY!
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python 3.11+ from https://python.org
        echo [HINT]  Then run: python install.py
        goto END
    )
    set "PY=python"
    echo [WARN] venv not found. Recommended: python install.py
)

rem -- 2. Check Python 3.11+ ---------------------------------------------
"!PY!" -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.11+ required. Version: !PY! --version
    goto END
)

rem -- 3. Files -----------------------------------------------------------
if not exist "models\%MODEL_FILE%" (
    echo [ERROR] Model not found: models\%MODEL_FILE%
    echo [HINT]  Download .gguf into models/ or run: python install.py --models-only
    goto END
)

if not exist "llm\llama-server.exe" (
    echo [ERROR] llm\llama-server.exe not found
    echo [HINT]  Run: python install.py --llama-only
    goto END
)

if not exist "run_llama.bat" (
    echo [ERROR] run_llama.bat missing next to start.bat
    goto END
)

rem Draft model is optional -- spec-decode just disables itself.
if not exist "models\%DRAFT_MODEL%" (
    echo [WARN] Draft model not found: models\%DRAFT_MODEL%
    echo        Speculative decoding will be OFF. Run install.py --models-only to fix.
    set "DRAFT_MODEL="
)

rem -- 4. Deps ------------------------------------------------------------
"!PY!" -c "import fastapi,uvicorn,httpx,pydantic,orjson" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    "!PY!" -m pip install -r requirements.txt --quiet --disable-pip-version-check
    if errorlevel 1 (
        echo [ERROR] pip install failed
        goto END
    )
)

rem -- 5. Ports -----------------------------------------------------------
netstat -ano | findstr ":%LLM_PORT% " | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 (
    echo [WARN] Port %LLM_PORT% is already in use. llama-server may fail to start.
)

if not exist "logs" mkdir logs

rem -- 6. Start llama-server via separate bat ---------------------------
echo [INFO] Starting llama-server (CUDA) -- this takes ~30-90 seconds with spec-decode...
echo [INFO] A separate window will open with llama-server. Do not close it.
echo [INFO] If that window closes instantly -- look at logs\llama-server.log.

start "llama-server" cmd /k call "%ROOT%\run_llama.bat" %MODEL_FILE% %LLM_PORT% %GPU_LAYERS% !THREADS! %CTX_SIZE% %UBATCH% %CACHE_TYPE_K% %CACHE_TYPE_V% "%DRAFT_MODEL%" %DRAFT_GPU_LAYERS% %DRAFT_MAX%

"!PY!" check_server.py llama %LLM_PORT% 420
if errorlevel 1 (
    echo.
    echo [ERROR] llama-server did not start within 7 minutes.
    echo.
    echo [HINT]  Look at the llama-server window -- the reason is printed there.
    echo         Or see logs\llama-server.log -- last 40 lines:
    echo ------------------------------------------------------------
    if exist "logs\llama-server.log" (
        powershell -NoProfile -Command "Get-Content 'logs\llama-server.log' -Tail 40"
    ) else (
        echo [no log -- llama-server crashed before it could write anything.]
        echo [Most likely: no VC++ Redistributable x64 or no CUDA runtime DLLs.]
    )
    echo ------------------------------------------------------------
    echo.
    echo [HINT]  Common issues:
    echo         - "cudart64_*.dll not found"       -^> run: python install.py --llama-only
    echo         - "no CUDA device found"           -^> update GPU driver, reboot
    echo         - "failed to load model"           -^> decrease GPU_LAYERS in start.bat
    echo         - "out of memory" / "ggml_cuda"    -^> decrease CTX_SIZE or GPU_LAYERS
    echo         - "VCRUNTIME140.dll not found"     -^> install Microsoft VC++ Redist x64
    echo         - draft/main vocab mismatch        -^> draft MUST be from the same family
    goto END
)

rem -- 7. Start backend --------------------------------------------------
echo [INFO] Starting Python backend...
start "Digital Human" /MIN "!PY!" main.py

"!PY!" check_server.py backend %APP_PORT% 60
if errorlevel 1 (
    echo [WARNING] Backend is not responding -- check logs\error.log
)

echo.
echo ============================================================
echo   Digital Human v9.0 ready
echo   UI: http://localhost:%APP_PORT%
echo ============================================================
timeout /t 2 /nobreak >nul
start "" "http://localhost:%APP_PORT%"

:END
echo.
pause
exit /b 0
