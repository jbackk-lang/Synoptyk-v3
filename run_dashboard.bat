@echo off
title Synoptyk-v3 -- Dashboard (appka lokalna)
color 0B
cls

cd /d "%~dp0"

echo ============================================================
echo   Synoptyk-v3: membrana pogodowa (FastAPI)
echo   Katalog roboczy: %cd%
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" (
    echo [OK] Aktywacja venv...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [OK] Aktywacja .venv...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] Uzywanie systemowej instalacji Pythona.
)

echo.
echo [1/2] Instalacja pakietow pip...
python -m pip install --upgrade pip --disable-pip-version-check
python -m pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo [BLAD] Instalacja pakietow nie powiodla sie.
    pause
    exit /b 1
)

echo.
echo [2/2] Uruchamianie dashboardu na http://127.0.0.1:8010 ...
echo.
echo (Wymaga polaczenia z internetem - appka pobiera dane na zywo
echo  z Open-Meteo. Zamknij to okno (Ctrl+C), zeby zatrzymac serwer.)
echo.

start "" http://127.0.0.1:8010
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8010

pause
