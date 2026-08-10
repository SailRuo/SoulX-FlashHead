@echo off
setlocal
cd /d "%~dp0"

set "CONDA_ENV=E:\conda_envs\flashhead"
set "PYTHON=%CONDA_ENV%\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Conda env not found: %CONDA_ENV%
    echo Create it first, or edit CONDA_ENV in this script.
    pause
    exit /b 1
)

if not exist "models\SoulX-FlashHead-1_3B\Model_Lite\diffusion_pytorch_model.safetensors" (
    echo [ERROR] Model weights missing under models\SoulX-FlashHead-1_3B
    pause
    exit /b 1
)

if not exist "models\wav2vec2-base-960h\config.json" (
    echo [ERROR] wav2vec2 missing under models\wav2vec2-base-960h
    pause
    exit /b 1
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONUNBUFFERED=1"
if not defined FLASHHEAD_COMPILE set "FLASHHEAD_COMPILE=1"

"%PYTHON%" -c "import triton" 1>nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing triton-windows ^(required by torch.compile^)...
    "%PYTHON%" -m pip install -U "triton-windows<3.4"
)

echo ============================================
echo  SoulX-FlashHead Gradio STREAMING
echo  Env: %CONDA_ENV%
echo  GPU: %CUDA_VISIBLE_DEVICES%
echo  COMPILE: %FLASHHEAD_COMPILE%
echo  Open: http://127.0.0.1:7860
echo  Tip: UI default is Lite; plays while generating
echo ============================================

"%PYTHON%" gradio_app_streaming.py
if errorlevel 1 (
    echo.
    echo [ERROR] Gradio exited with an error.
    pause
)
endlocal
