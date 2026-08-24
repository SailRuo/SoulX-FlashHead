@echo off
REM =====================================================================
REM  Launcher for the onedir-packaged SoulX-FlashHead Gradio (streaming).
REM  This file is copied into dist\FlashHead_Gradio\Launch_Gradio.bat
REM  Run: double-click, or open CMD and call this file to preserve logs.
REM =====================================================================
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion

set "DIR=%~dp0"
cd /d "%DIR%"

set "EXE=%DIR%FlashHead_Gradio.exe"
if not exist "%EXE%" (
    echo [ERROR] FlashHead_Gradio.exe not found next to this launcher.
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
    echo        The program will still start; you can point to models inside the Gradio UI.
    echo.
)

set "CUDA_VISIBLE_DEVICES=0"
set "PYTHONUNBUFFERED=1"
if not defined FLASHHEAD_COMPILE set "FLASHHEAD_COMPILE=1"

echo ============================================
echo  SoulX-FlashHead Gradio (Streaming)
echo  EXE : %EXE%
echo  GPU : %CUDA_VISIBLE_DEVICES%
echo  COMPILE: %FLASHHEAD_COMPILE%
echo  Open: http://127.0.0.1:7860
echo ============================================
echo.

"%EXE%"
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] Gradio exited with code %EXITCODE%. See messages above.
    echo         If the window closes before you can read this, re-run this .bat
    echo         from inside a CMD prompt so the output stays visible.
    pause
)

endlocal
exit /b %EXITCODE%
