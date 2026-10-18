@echo off
chcp 65001 >nul 2>&1
REM ============================================================
REM  Image2Music Demo - start web server
REM
REM  Usage: double-click this file, or run start_web.cmd
REM  Then open http://127.0.0.1:5000
REM  Stop: Ctrl+C
REM ============================================================
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" goto :no_venv

REM The venv may come from another machine: python.exe exists but pyvenv.cfg
REM points to a missing base interpreter. Only a real run proves it works.
"%VENV_PY%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :broken_venv

"%VENV_PY%" -c "import flask" >nul 2>nul
if errorlevel 1 goto :no_flask

echo [INFO] interpreter: %VENV_PY%
echo [INFO] loading CLIP model and material vectors (about 10 seconds)...
echo [INFO] open http://127.0.0.1:5000 after startup
echo.

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
"%VENV_PY%" app.py
set "RC=%ERRORLEVEL%"
echo.
echo [INFO] server stopped (exit code %RC%).
if not "%RC%"=="0" (
    echo [HINT] If port 5000 is already in use, close the old python process first.
    pause
)
exit /b %RC%

:no_venv
echo.
echo [ERROR] .venv\Scripts\python.exe not found.
echo.
echo First-time setup (needs network, run once):
echo   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
echo.
pause
exit /b 1

:broken_venv
echo.
echo [ERROR] .venv exists but cannot start: its base interpreter is missing on this PC.
echo         (Common cause: the project was copied from another PC or folder.)
echo.
echo Two ways to fix, pick one:
echo   1. Rebuild (needs network):
echo        rmdir /s /q .venv
echo        powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
echo   2. Edit one line (no network, keeps installed packages):
echo        open .venv\pyvenv.cfg and point home / executable to a real
echo        Python on this PC, for example:
echo        %%LOCALAPPDATA%%\Programs\DeepSeek Harness\resources\runtime\primary-runtime\dependencies\python
echo.
pause
exit /b 1

:no_flask
echo.
echo [ERROR] .venv starts, but Flask and other packages are missing.
echo.
echo Please run (needs network):
echo   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
echo Or, inside the venv:
echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
pause
exit /b 1
