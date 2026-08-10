@echo off
setlocal
cd /d "%~dp0"

set "CONDA_ENV=E:\conda_envs\flashhead"
set "PYTHON=%CONDA_ENV%\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Conda env not found: %CONDA_ENV%
    pause
    exit /b 1
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONUNBUFFERED=1"
if not defined FLASHHEAD_COMPILE set "FLASHHEAD_COMPILE=1"

echo ============================================
echo  SoulX-FlashHead PCM WebSocket API
echo  ws://0.0.0.0:8765/ws  (LAN phones use this PC's IP)
echo  Startup: preload + warmup model (default lite @512)
echo  New aspect ratios warm on first start (status: loading)
echo  Wait until you see "Model ready" / WS ready before connecting
echo ============================================

"%PYTHON%" pcm_ws_server.py --host 0.0.0.0 --port 8765 --model-type lite --preload --warmup
if errorlevel 1 pause
endlocal
