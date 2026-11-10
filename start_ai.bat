@echo off
setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"

echo ================================
echo          My Local AI
echo ================================

if not exist "%PY%" (
    echo Python virtual enviroment not found.
    pause
    exit /b 1
)

REM Check Ollama API
"%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=2)" >nul 2>&1

if errorlevel 1 (
    echo Starting Ollama...

    where ollama >nul 2>&1
    if errorlevel 1 (
        echo Ollama not found in PATH.
        echo Open Ollama from the Windows Start menu.
        pause
        exit /b 1
    )

    start "Ollama Server" /min cmd /k "ollama serve"
    timeout /t 5 /nobreak >nul

    "%PY%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:11434/api/tags', timeout=3)" >nul 2>&1

    if errorlevel 1 (
        echo Ollama could not start.
        pause
        exit /b 1
    )
)

echo Starting My Local AI...

if exist "web_app.py" (
    "%PY%" -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false
) else (
    REM Web UI not built yet - run terminal chat
    chcp 65001 >nul
    set PYTHONIOENCODING=utf-8
    "%PY%" main.py
)

pause