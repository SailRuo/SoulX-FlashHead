@echo off
REM =====================================================================
REM  Launcher for the onedir-packaged SoulX-FlashHead PCM WebSocket API.
REM  This file is copied into dist\FlashHead_PCM_Server\Launch_PCM_Server.bat
REM  Run: double-click, or open CMD and call this file to preserve logs.
REM =====================================================================
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion

set "DIR=%~dp0"
cd /d "%DIR%"

set "EXE=%DIR%FlashHead_PCM_Server.exe"
if not exist "%EXE%" (
    echo [ERROR] FlashHead_PCM_Server.exe not found next to this launcher.
    echo         Expected: %EXE%
    echo         Make sure you copied the ENTIRE dist folder, not just this .bat.
    pause
    exit /b 1
)

REM --- Check for models folder (can be in exe dir or one level up) -----------
set "MODELS_DIR="
if exist "%DIR%models\SoulX-FlashHead-1_3B\Model_Lite\diffusion_pytorch_model.safetensors" set "MODELS_DIR=%DIR%models"
if not defined MODELS_DIR (
    if exist "%DIR%..\models\SoulX-FlashHead-1_3B\Model_Lite\diffusion_pytorch_model.safetensors" set "MODELS_DIR=%DIR%..\models"
)
if not defined MODELS_DIR (
    echo [WARN] Model weights not auto-detected under:
    echo        %DIR%models\
    echo        %DIR%..\models\
    echo        You MUST provide --ckpt_dir and --wav2vec_dir CLI arguments manually.
    echo.
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONUNBUFFERED=1"
REM torch.compile in a frozen exe would need a full MSVC+SDK+CUDA+Python header
REM toolchain at runtime (triton JIT), which is not portable -> use eager mode.
REM Set FLASHHEAD_COMPILE explicitly (1=compile will fail without dev toolchain).
set "FLASHHEAD_COMPILE=0"

echo ============================================
echo  SoulX-FlashHead PCM WebSocket API
echo  EXE : %EXE%
echo  ws://0.0.0.0:8765/ws
echo  Default args: --host 0.0.0.0 --port 8765 --model-type lite --preload --warmup
echo ============================================
echo.

"%EXE%" --host 0.0.0.0 --port 8765 --model-type lite --preload --warmup
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] PCM Server exited with code %EXITCODE%. See messages above.
    echo         If the window closes before you can read this, re-run this .bat
    echo         from inside a CMD prompt so the output stays visible.
    pause
)

endlocal
exit /b %EXITCODE%
