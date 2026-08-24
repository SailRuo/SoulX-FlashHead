# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller onedir spec for SoulX-FlashHead Gradio (gradio_app_streaming.py).
Produces dist/FlashHead_Gradio/ containing FlashHead_Gradio.exe + _internal/.
Run via:  pyinstaller flashhead_gradio.spec   (or build_exe.bat)
"""
import os
import sys
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    collect_all,
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(SPEC)) if 'SPEC' in globals() else os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

ENTRY_SCRIPT = os.path.join(PROJECT_ROOT, 'gradio_app_streaming.py')
APP_NAME = 'FlashHead_Gradio'

# ---- Collect heavy ML / UI packages (submodules + data + binaries) ---------
hiddenimports: list[str] = []
datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []

PKGS_TO_COLLECT_ALL = [
    'flash_head',
    'ltx_video',
    'wan',
    'diffusers',
    'transformers',
    'tokenizers',
    'accelerate',
    'gradio',
    'gradio_client',
    'mediapipe',
    'xformers',
    'xfuser',
    'websockets',
    'loguru',
    'librosa',
    'decord',
    'imageio',
    'imageio_ffmpeg',
    'skimage',
    'flask',
    'easydict',
    'ftfy',
    'pyloudnorm',
    'soundfile',
    'resampy',
    'pydub',
    'audioread',
    'sndfile',
    'numpy',
    'scipy',
    'PIL',
    'cv2',
]
for pkg in PKGS_TO_COLLECT_ALL:
    try:
        _subs, _datas, _bins = collect_all(pkg)
        hiddenimports.extend(_subs)
        datas.extend(_datas)
        binaries.extend(_bins)
    except Exception as exc:  # pragma: no cover - packaging tolerance
        print(f'[WARN] collect_all({pkg!r}) skipped: {exc}')

# ---- Torch (big, but must include CUDA runtime libs + C extensions) --------
hiddenimports += collect_submodules('torch')
hiddenimports += collect_submodules('torchvision')
datas += collect_data_files('torch', include_py_files=False, excludes=['**/*.pyi'])
datas += collect_data_files('torchvision', include_py_files=False, excludes=['**/*.pyi'])
binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('torchvision')

# ---- triton / nvidia vendor bits -------------------------------------------
for _vendor in ('triton', 'triton-windows', 'nvidia', 'nvidia.nccl-cu12'):
    try:
        hiddenimports += collect_submodules(_vendor)
        datas += collect_data_files(_vendor)
        binaries += collect_dynamic_libs(_vendor)
    except Exception:
        pass

# ---- realesrgan-ncnn-py optional -------------------------------------------
for _opt in ('realesrgan_ncnn_py',):
    try:
        hiddenimports += collect_submodules(_opt)
        datas += collect_data_files(_opt)
        binaries += collect_dynamic_libs(_opt)
    except Exception:
        pass

# ---- Local configs (infer_params.yaml etc.) --------------------------------
CFG_SRC = os.path.join(PROJECT_ROOT, 'flash_head', 'configs')
if os.path.isdir(CFG_SRC):
    datas.append((CFG_SRC, os.path.join('flash_head', 'configs')))

# ---- Gradio needs its frontend assets (they normally auto-download) -------
# Already collected via collect_all(gradio), but belt-and-braces:
try:
    datas += collect_data_files('gradio', subdir='templates')
    datas += collect_data_files('gradio', subdir='themes')
except Exception:
    pass

# ---- Deduplicate -----------------------------------------------------------
hiddenimports = sorted(set(hiddenimports))
datas = list({k: v for k, v in datas}.items())
binaries = list({k: v for k, v in binaries}.items())

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'tests',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
