@echo off
REM =====================================================================
REM  SoulX-FlashHead Windows exe build script - PCM Server ONLY
REM  Output: dist\FlashHead_PCM_Server\
REM  Double-click to run, or call:  build_exe.bat --no-pause  (CI / pipe mode)
REM =====================================================================
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion

set "NO_PAUSE=0"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
if /I "%~1"=="-y"          set "NO_PAUSE=1"

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo ============================================
echo  SoulX-FlashHead exe build (onedir, PCM only)
echo  ROOT: %ROOT%
echo ============================================
echo.

REM --- 1) Locate python (prefer CONDA_ENV if set, else current python) ---
set "PYTHON="
if defined CONDA_ENV (
    if exist "%CONDA_ENV%\python.exe" set "PYTHON=%CONDA_ENV%\python.exe"
)
if not defined PYTHON (
    if exist "E:\conda_envs\flashhead\python.exe" set "PYTHON=E:\conda_envs\flashhead\python.exe"
)
if not defined PYTHON (
    for /f "delims=" %%i in ('where python 2^>nul') do if not defined PYTHON set "PYTHON=%%i"
)
if not defined PYTHON (
    echo [ERROR] python.exe not found. Activate your env or set PYTHON explicitly.
    echo   Example: set "PYTHON=E:\conda_envs\flashhead\python.exe" ^&^& build_exe.bat
    call :PAUSE_OR_EXIT 1
    exit /b 1
)
echo [INFO] Using Python: %PYTHON%

REM --- 2) Ensure PyInstaller is installed ---
"%PYTHON%" -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller ...
    "%PYTHON%" -m pip install -U pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller.
        call :PAUSE_OR_EXIT 1
        exit /b 1
    )
)

REM --- 3) PyInstaller version sanity ---
"%PYTHON%" -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python -m PyInstaller not available after install - abort.
    call :PAUSE_OR_EXIT 1
    exit /b 1
)

REM --- 4) Wipe previous build artifacts for clean rebuild ---
if exist "build" (
    echo [INFO] Removing previous build\ ...
    rmdir /s /q "build"
)
if exist "dist\FlashHead_PCM_Server" (
    echo [INFO] Removing previous dist\FlashHead_PCM_Server\ ...
    rmdir /s /q "dist\FlashHead_PCM_Server"
)

REM --- 5) Make sure the launcher and spec exist ---
if not exist "%ROOT%flashhead_pcm_server.spec" (
    echo [ERROR] flashhead_pcm_server.spec missing. Expected: %ROOT%flashhead_pcm_server.spec
    call :PAUSE_OR_EXIT 1
    exit /b 1
)
if not exist "%ROOT%pcm_ws_server.py" (
    echo [ERROR] pcm_ws_server.py missing. Expected: %ROOT%pcm_ws_server.py
    call :PAUSE_OR_EXIT 1
    exit /b 1
)

REM --- 6) Run PyInstaller (NOTE: --workpath / --specpath are makespec-only, do NOT use with a .spec) ---
echo.
echo ===== Building FlashHead_PCM_Server (pcm_ws_server.py) =====
"%PYTHON%" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --distpath "%ROOT%dist" ^
    "%ROOT%flashhead_pcm_server.spec"
if errorlevel 1 (
    echo.
    echo [ERROR] PCM Server exe build FAILED. See PyInstaller output above for traceback.
    call :PAUSE_OR_EXIT 1
    exit /b 1
)

REM --- 7) Verify the produced .exe exists before claiming success ---
if not exist "%ROOT%dist\FlashHead_PCM_Server\FlashHead_PCM_Server.exe" (
    echo [ERROR] PyInstaller returned 0 but FlashHead_PCM_Server.exe NOT found under dist\FlashHead_PCM_Server\.
    dir /b "%ROOT%dist\FlashHead_PCM_Server" 2>nul
    call :PAUSE_OR_EXIT 1
    exit /b 1
)

REM --- 8) Copy runtime launcher into dist folder ---
echo.
echo [INFO] Copying launchers into dist directories ...
if exist "%ROOT%launch_pcm_server.bat" (
    copy /y "%ROOT%launch_pcm_server.bat" "%ROOT%dist\FlashHead_PCM_Server\Launch_PCM_Server.bat" >nul
) else (
    echo [WARN] launch_pcm_server.bat not found - not copied.
)

REM --- 9) Copy configs/ if PyInstaller hook missed the local subpackage data ---
if exist "%ROOT%flash_head\configs" (
    if not exist "%ROOT%dist\FlashHead_PCM_Server\_internal\flash_head\configs" (
        echo [INFO] Copying flash_head\configs into _internal\flash_head\configs ...
        xcopy /e /i /h /y "%ROOT%flash_head\configs" "%ROOT%dist\FlashHead_PCM_Server\_internal\flash_head\configs" >nul
    )
)

REM --- 10) Write deploy note so user knows models/ must be placed next to exe ---
(
echo SoulX-FlashHead - Deploy Notes (onedir build, PCM Server)
echo ==========================================================
echo 1. This folder is PORTABLE. Copy the WHOLE folder to any Windows PC.
echo 2. Requirements on target machine:
echo    - NVIDIA GPU with CUDA 12.x compatible driver (4090 / 5090 recommended)
echo    - At least 16 GB VRAM for Lite model, 24 GB+ for Pro model
echo    - 64 GB RAM or more recommended
echo 3. Put model weights next to this .txt, directory layout:
echo    models\
echo      SoulX-FlashHead-1_3B\
echo        Model_Lite\   diffusion_pytorch_model.safetensors ...
echo        Model_Pro\    diffusion_pytorch_model.safetensors ...
echo      wav2vec2-base-960h\  config.json, preprocessor_config.json, model.safetensors ...
echo 4. Start: double-click Launch_PCM_Server.bat
echo    URL: ws://0.0.0.0:8765/ws   (use this PC's IP for LAN phones)
echo 5. If the window closes instantly: re-run Launch_PCM_Server.bat from a CMD
echo    window so the error text stays visible.
) > "%ROOT%dist\FlashHead_PCM_Server\DEPLOY_NOTES.txt"

echo.
echo ============================================
echo  BUILD SUCCESS (PCM Server only)
echo  Output: %ROOT%dist\FlashHead_PCM_Server\
echo ============================================
echo.
echo IMPORTANT: Before shipping, copy your models\ folder next to the exe.
echo            See DEPLOY_NOTES.txt in the dist folder.
echo.
call :PAUSE_OR_EXIT 0
endlocal
exit /b 0

REM ------------------------------------------------------------------
REM  Subroutine: pause only in interactive mode (when --no-pause not set)
REM  %1 = exit code hint (0 = success banner, other = error banner)
REM ------------------------------------------------------------------
:PAUSE_OR_EXIT
if "%NO_PAUSE%"=="1" exit /b %~1
echo Press any key to exit ...
pause >nul
exit /b
