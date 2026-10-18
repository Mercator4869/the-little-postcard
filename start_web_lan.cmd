@echo off
chcp 65001 >nul 2>&1
REM ============================================================
REM  Image2Music Demo - start web server for LAN access
REM
REM  Binds 0.0.0.0 so other devices on the same network can open
REM  http://<this-PC-LAN-IP>:5000
REM
REM  Windows may ask to allow Python through the firewall once.
REM  Stop: Ctrl+C
REM ============================================================
setlocal
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" goto :no_venv

"%VENV_PY%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :broken_venv

set "SERVER_HOST=0.0.0.0"
set "SERVER_PORT=5000"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo [INFO] server bind: 0.0.0.0:5000
echo [INFO] on this PC:      http://127.0.0.1:5000
echo [INFO] from LAN:        http://^<this-PC-LAN-IP^>:5000
echo [INFO] check LAN IP with: ipconfig ^| findstr IPv4
echo.

"%VENV_PY%" app.py
set "RC=%ERRORLEVEL%"
echo.
echo [INFO] server stopped (exit code %RC%).
if not "%RC%"=="0" pause
exit /b %RC%

:no_venv
echo [ERROR] .venv\Scripts\python.exe not found.
echo Run setup_env.ps1 first.
pause
exit /b 1

:broken_venv
echo [ERROR] .venv exists but cannot start (base interpreter missing).
echo Rebuild with setup_env.ps1, or fix .venv\pyvenv.cfg.
pause
exit /b 1
