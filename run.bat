@echo off
rem Start the overlay without keeping a console window open.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo Run "uv sync" and "uv run jpreplace-setup" first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m jpreplace
