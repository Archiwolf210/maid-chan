@echo off
title llama-server [Digital Human v9.0]
setlocal
cd /d "%~dp0llm"

rem =============================================================
rem  Args (all passed by start.bat):
rem   %1  MODEL_FILE          e.g. qwen3.gguf
rem   %2  LLM_PORT            e.g. 8080
rem   %3  GPU_LAYERS          e.g. 999 (all onto GPU)
rem   %4  THREADS             e.g. 10  (physical cores, not logical)
rem   %5  CTX_SIZE            e.g. 32768
rem   %6  UBATCH              e.g. 512
rem   %7  CACHE_K             e.g. q8_0
rem   %8  CACHE_V             e.g. q8_0
rem   %9  DRAFT_MODEL         draft .gguf name (empty => no spec-decode)
rem  %10  DRAFT_GPU_LAYERS    e.g. 99 (all draft onto GPU)  -- via shift
rem  %11  DRAFT_MAX           e.g. 8  (speculative lookahead) -- via shift
rem =============================================================

set "MODEL_FILE=%~1"
set "LLM_PORT=%~2"
set "GPU_LAYERS=%~3"
set "THREADS=%~4"
set "CTX_SIZE=%~5"
set "UBATCH=%~6"
set "CACHE_K=%~7"
set "CACHE_V=%~8"
set "DRAFT_MODEL=%~9"
shift
shift
set "DRAFT_GPU_LAYERS=%~8"
set "DRAFT_MAX=%~9"

set "LOG=%~dp0logs\llama-server.log"
rem Ensure log directory exists -- otherwise PowerShell tee spams "path not found".
if not exist "%~dp0logs" mkdir "%~dp0logs" >nul 2>nul
if exist "%LOG%" del /q "%LOG%" >nul 2>nul

echo [run_llama] model=%MODEL_FILE% port=%LLM_PORT% gpu_layers=%GPU_LAYERS% ctx=%CTX_SIZE%
if defined DRAFT_MODEL if exist "..\models\%DRAFT_MODEL%" (
    echo [run_llama] spec-decode: draft=%DRAFT_MODEL% draft_layers=%DRAFT_GPU_LAYERS% draft_max=%DRAFT_MAX%
) else (
    echo [run_llama] spec-decode: OFF ^(draft model not found^)
    set "DRAFT_MODEL="
)
echo [run_llama] cache_k=%CACHE_K% cache_v=%CACHE_V% ubatch=%UBATCH% threads=%THREADS%
echo [run_llama] log: %LOG%
echo ============================================================
echo.

rem Keep stderr+stdout together and tee to a UTF-8 log file via PowerShell.
rem Two branches -- with/without speculative decoding.

if defined DRAFT_MODEL (
    llama-server.exe ^
        --model "..\models\%MODEL_FILE%" ^
        --port %LLM_PORT% ^
        --host 127.0.0.1 ^
        --n-gpu-layers %GPU_LAYERS% ^
        --main-gpu 0 ^
        --threads %THREADS% ^
        --parallel 1 ^
        --cont-batching ^
        --ctx-size %CTX_SIZE% ^
        --n-predict -1 ^
        --ubatch-size %UBATCH% ^
        --cache-type-k %CACHE_K% ^
        --cache-type-v %CACHE_V% ^
        --flash-attn on ^
        --mlock ^
        --model-draft "..\models\%DRAFT_MODEL%" ^
        --gpu-layers-draft %DRAFT_GPU_LAYERS% ^
        --draft-max %DRAFT_MAX% ^
        --draft-min 1 ^
        2>&1 | powershell -NoProfile -Command "$input | ForEach-Object { Write-Output $_; Add-Content -LiteralPath '%LOG%' -Value $_ -Encoding UTF8 }"
) else (
    llama-server.exe ^
        --model "..\models\%MODEL_FILE%" ^
        --port %LLM_PORT% ^
        --host 127.0.0.1 ^
        --n-gpu-layers %GPU_LAYERS% ^
        --main-gpu 0 ^
        --threads %THREADS% ^
        --parallel 1 ^
        --cont-batching ^
        --ctx-size %CTX_SIZE% ^
        --n-predict -1 ^
        --ubatch-size %UBATCH% ^
        --cache-type-k %CACHE_K% ^
        --cache-type-v %CACHE_V% ^
        --flash-attn on ^
        --mlock ^
        2>&1 | powershell -NoProfile -Command "$input | ForEach-Object { Write-Output $_; Add-Content -LiteralPath '%LOG%' -Value $_ -Encoding UTF8 }"
)

set EXITCODE=%ERRORLEVEL%
echo.
echo ============================================================
echo [llama-server exited with code %EXITCODE%]
echo Reason see in log: %LOG%
echo Window stays open so you can copy the text.
echo ============================================================
pause
