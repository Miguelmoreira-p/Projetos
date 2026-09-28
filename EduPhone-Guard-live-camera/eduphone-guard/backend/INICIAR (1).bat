@echo off
chcp 65001 >nul
cd /d "%~dp0"
title EduPhone Guard
set PY=

py -3 --version >nul 2>nul
if not errorlevel 1 set PY=py -3
if defined PY goto found

python --version >nul 2>nul
if not errorlevel 1 set PY=python
if defined PY goto found

for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%D\python.exe" set PY="%%D\python.exe"
)
if defined PY goto found

echo Python nao encontrado. Instale em python.org/downloads marcando "Add python.exe to PATH".
pause
exit /b 1

:found
echo Usando Python:
%PY% --version

if not exist .env copy .env.example .env >nul
if not exist data\samples mkdir data\samples
if not exist data\clips mkdir data\clips

%PY% -c "import cv2, ultralytics, pydantic, pydantic_settings, numpy, uvicorn, fastapi, slowapi, jose, multipart" >nul 2>nul
if errorlevel 1 (
    echo Instalando dependencias na primeira vez. Isso pode demorar alguns minutos...
    %PY% -m pip install --upgrade pip
    %PY% -m pip install fastapi "uvicorn[standard]" pydantic pydantic-settings email-validator python-dotenv supabase opencv-python-headless ultralytics numpy python-jose python-multipart slowapi httpx pytest
    if errorlevel 1 (
        echo.
        echo Falha ao instalar dependencias. Copie a mensagem acima e envie para suporte.
        pause
        exit /b 1
    )
)

echo.
echo Abrindo o navegador em http://127.0.0.1:8000/monitor ...
start "" "http://127.0.0.1:8000/monitor"

%PY% -m uvicorn app.main:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo.
    echo O servidor fechou com erro. Copie a mensagem acima.
    pause
)
