@echo off
setlocal
title Fast model download [Digital Human v9.0]

rem ================================================================
rem  Thin launcher for download_models.py
rem  All logic lives in Python to avoid:
rem    - cmd UTF-8 console quirks
rem    - 32-bit signed overflow in `set /a` for files > 2 GB
rem    - huggingface-cli versioning across hub releases
rem ================================================================

cd /d "%~dp0"
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo.
    echo [ERROR] venv not found at "%ROOT%\.venv".
    echo         First run: python install.py --deps-only
    echo.
    pause
    exit /b 1
)

if not exist "%ROOT%\download_models.py" (
    echo.
    echo [ERROR] download_models.py not found next to this .bat.
    echo.
    pause
    exit /b 1
)

"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\download_models.py"
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
