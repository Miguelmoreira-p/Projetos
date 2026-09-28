@echo off
cd /d "%~dp0"
echo Iniciando EduPhone Guard...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
