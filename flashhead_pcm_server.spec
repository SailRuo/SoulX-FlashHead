# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller onedir spec for SoulX-FlashHead PCM WebSocket server (pcm_ws_server.py).
Produces dist/FlashHead_PCM_Server/ containing FlashHead_PCM_Server.exe + _internal/.
Run via:  pyinstaller flashhead_pcm_server.spec   (or build_exe.bat)
"""
import os
import sys
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
    collect_all,
    copy_metadata,
)

PROJECT_ROOT = os.path.abspath(os.path.dirname(SPEC)) if 'SPEC' in globals() else os.path.abspath('.')
sys.path.insert(0, PROJECT_ROOT)

ENTRY_SCRIPT = os.path.join(PROJECT_ROOT, 'pcm_ws_server.py')
APP_NAME = 'FlashHead_PCM_Server'

# ---- Helpers: strictly type-filter lists to avoid mixed-type crashes --------
def _as_str_list(xs, label='hiddenimports'):
    """Accept arbitrary iterable, return list of unique strings (non-str dropped)."""
    out = []
    seen = set()
    if not xs:
        return out
    for x in xs:
        if isinstance(x, str):
            if x not in seen:
                seen.add(x)
                out.append(x)
        # silently drop tuples / anything else - those belong to datas/binaries
    return out


def _as_pair_list(xs, label='datas/binaries'):
    """Accept arbitrary iterable, return list of 2-tuples of str, deduped by first element."""
    out = []
    seen = set()
    if not xs:
        return out
    for x in xs:
        try:
            if isinstance(x, (list, tuple)) and len(x) == 2:
                src, dst = x
                if isinstance(src, str) and isinstance(dst, str):
                    key = (os.path.normcase(os.path.abspath(src)), dst) if os.path.isabs(src) else (src, dst)
                    if key not in seen:
                        seen.add(key)
                        out.append((src, dst))
        except Exception:
            continue
    return out


# ---- Collect heavy ML / WS packages ----------------------------------------
hiddenimports = []
datas = []
binaries = []

# NOTE: 'ltx_video' and 'wan' are subpackages under flash_head/ -> collect via flash_head only.
PKGS_TO_COLLECT_ALL = [
    'flash_head',
    'diffusers',
    'transformers',
    'tokenizers',
    'accelerate',
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
    'numpy',
    'scipy',
    'PIL',
    'cv2',
]
for pkg in PKGS_TO_COLLECT_ALL:
    try:
        _result = collect_all(pkg)
        if _result is None:
            continue
        if len(_result) < 3:
            print(f'[WARN] collect_all({pkg!r}) returned {len(_result)} items, expected 3; skipping.')
            continue
        _subs, _datas, _bins = _result[0], _result[1], _result[2]
        hiddenimports.extend(_as_str_list(_subs))
        datas.extend(_as_pair_list(_datas))
        binaries.extend(_as_pair_list(_bins))
    except Exception as exc:
        print(f'[WARN] collect_all({pkg!r}) skipped: {exc.__class__.__name__}: {exc}')

# ---- Torch / torchvision (separate hooks - they ship native PyInstaller hooks) ----
try:
    hiddenimports.extend(_as_str_list(collect_submodules('torch')))
except Exception as exc:
    print(f'[WARN] collect_submodules(torch): {exc}')
try:
    hiddenimports.extend(_as_str_list(collect_submodules('torchvision')))
except Exception as exc:
    print(f'[WARN] collect_submodules(torchvision): {exc}')
try:
    datas.extend(_as_pair_list(collect_data_files('torch', include_py_files=False, excludes=['**/*.pyi'])))
except Exception as exc:
    print(f'[WARN] collect_data_files(torch): {exc}')
try:
    datas.extend(_as_pair_list(collect_data_files('torchvision', include_py_files=False, excludes=['**/*.pyi'])))
except Exception as exc:
    print(f'[WARN] collect_data_files(torchvision): {exc}')
try:
    binaries.extend(_as_pair_list(collect_dynamic_libs('torch')))
except Exception as exc:
    print(f'[WARN] collect_dynamic_libs(torch): {exc}')
try:
    binaries.extend(_as_pair_list(collect_dynamic_libs('torchvision')))
except Exception as exc:
    print(f'[WARN] collect_dynamic_libs(torchvision): {exc}')

# ---- nvidia vendor bits (use only valid Python identifiers) -------
for _vendor in ('nvidia',):
    try:
        hiddenimports.extend(_as_str_list(collect_submodules(_vendor)))
    except Exception:
        pass
    try:
        datas.extend(_as_pair_list(collect_data_files(_vendor)))
    except Exception:
        pass
    try:
        binaries.extend(_as_pair_list(collect_dynamic_libs(_vendor)))
    except Exception:
        pass

# ---- triton: MUST be included (torch._inductor needs it for CUDA compiles) ----
# FlashHead uses @torch.compile -> torch._inductor -> Triton kernel backend. Without
# a working triton the warmup forward raises TritonMissing. The official
# hook-triton.py keeps triton's source on disk and collects the backend files, which
# is exactly what fixes the original "triton/backends/amd/compiler.py missing" crash
# (that crash was due to triton modules being bundled into PYZ without their .py).
try:
    hiddenimports.extend(_as_str_list(collect_submodules('triton')))
    hiddenimports.extend(_as_str_list(collect_submodules('triton.backends')))
    hiddenimports.extend(_as_str_list(collect_submodules('triton.backends.amd')))
    binaries.extend(_as_pair_list(collect_dynamic_libs('triton')))
    print('[INFO] triton modules + backends collected')
except Exception as exc:
    print(f'[WARN] collect triton: {exc.__class__.__name__}: {exc}')

# triton also reads NON-.py files at runtime to compile CUDA kernels: backends'
# driver.c, include/*.h, libdevice*.bc, bin/ptxas(.exe), cuda.lib, etc. These are
# excluded by collect_data_files()'s default suffix filter, so walk the whole package
# tree and add every non-.pyc file as data to be safe.
try:
    import importlib.util as _ilu
    _tspec = _ilu.find_spec('triton')
    _tpath = _tspec.submodule_search_locations[0]
    for _dp, _dn, _fn in os.walk(_tpath):
        if '__pycache__' in _dp.split(os.sep):
            continue
        for _f in _fn:
            if _f.endswith('.pyc'):
                continue
            _src = os.path.join(_dp, _f)
            _rel = os.path.relpath(_src, os.path.dirname(_tpath))
            datas.append((_src, os.path.dirname(_rel)))
    print('[INFO] triton package tree added as data (driver.c/.h/.bc/exe/etc)')
except Exception as exc:
    print(f'[WARN] triton data walk: {exc.__class__.__name__}: {exc}')

# ---- realesrgan-ncnn-py optional -------------------------------------------
for _opt in ('realesrgan_ncnn_py',):
    try:
        hiddenimports.extend(_as_str_list(collect_submodules(_opt)))
    except Exception:
        pass
    try:
        datas.extend(_as_pair_list(collect_data_files(_opt)))
    except Exception:
        pass
    try:
        binaries.extend(_as_pair_list(collect_dynamic_libs(_opt)))
    except Exception:
        pass

# ---- xformers: materialize package dir on disk -----------------------------
# xformers/_cpp_lib.py calls os.add_dll_directory(os.path.dirname(__file__)).
# Under PyInstaller onedir the pure modules live in PYZ, so `_internal\xformers`
# never exists -> add_dll_directory raises an UNCAUGHT WinError 2 at import time.
# Force the package dir (with .py + cpp_lib.json) onto disk to fix it.
try:
    datas.extend(_as_pair_list(collect_data_files('xformers', include_py_files=True)))
    print('[INFO] xformers package data collected to disk')
except Exception as exc:
    print(f'[WARN] collect_data_files(xformers): {exc.__class__.__name__}: {exc}')

# ---- yunchang: source must be on disk for torch.jit.script ------------------
# yunchang uses @torch.jit.script which needs the .py source (via inspect/linecache).
# In frozen mode co_filename points to a relative PYZ path with no real file.
# We (a) collect the modules normally, (b) materialise the .py on disk, and
# (c) run flashhead_rt_hook_torch_source so linecache is seeded -> TorchScript works.
try:
    hiddenimports.extend(_as_str_list(collect_submodules('yunchang')))
    datas.extend(_as_pair_list(collect_data_files('yunchang', include_py_files=True)))
    print('[INFO] yunchang modules + source collected')
except Exception as exc:
    print(f'[WARN] collect yunchang: {exc.__class__.__name__}: {exc}')

# ---- sageattention: source must be on disk for @triton.jit -----------------
# sageattention/triton/*.py decorate kernels with @triton.jit, which reads the
# function source via inspect.getsourcelines(). Same PYZ-source problem as yunchang:
# materialise the .py to disk so the runtime hook can seed linecache for it too.
try:
    binaries.extend(_as_pair_list(collect_dynamic_libs('sageattention')))
    datas.extend(_as_pair_list(collect_data_files('sageattention', include_py_files=True)))
    print('[INFO] sageattention source collected to disk')
except Exception as exc:
    print(f'[WARN] collect sageattention: {exc.__class__.__name__}: {exc}')

RTH = os.path.join(PROJECT_ROOT, 'flashhead_rt_hook_torch_source.py')

# ---- Local configs ---------------------------------------------------------
CFG_SRC = os.path.join(PROJECT_ROOT, 'flash_head', 'configs')
if os.path.isdir(CFG_SRC):
    datas.append((CFG_SRC, os.path.join('flash_head', 'configs')))

# ---- dist-info metadata (fixes importlib.metadata.PackageNotFoundError) -----
# Many packages call importlib.metadata.version("<pkg>") at import time
# (imageio, transformers, diffusers, librosa, mediapipe, etc.).
# PyInstaller collect_all() does NOT always copy *.dist-info/, so add them explicitly.
METADATA_PKGS = [
    'imageio',
    'imageio_ffmpeg',
    'transformers',
    'diffusers',
    'accelerate',
    'tokenizers',
    'xformers',
    'xfuser',
    'mediapipe',
    'numpy',
    'scipy',
    'librosa',
    'soundfile',
    'resampy',
    'pydub',
    'audioread',
    'pyloudnorm',
    'sndfile',
    'easydict',
    'ftfy',
    'loguru',
    'decord',
    'websockets',
    'flask',
    'skimage',
    'opencv_python_headless',
    'opencv_python',
    'Pillow',
    'torch',
    'torchvision',
    'sageattention',
    'realesrgan_ncnn_py',
    'gradio',
    'gradio_client',
]
for pkg in METADATA_PKGS:
    try:
        # copy_metadata returns list[(src, dst)] pointing to <pkg>-<ver>.dist-info/
        datas.extend(_as_pair_list(copy_metadata(pkg)))
        print(f'[INFO] copy_metadata({pkg!r}) OK')
    except Exception as exc:
        print(f'[WARN] copy_metadata({pkg!r}) skipped: {exc.__class__.__name__}: {exc}')

# ---- Sanitize and dedupe ---------------------------------------------------
hiddenimports = sorted(_as_str_list(hiddenimports))
datas = _as_pair_list(datas)
binaries = _as_pair_list(binaries)

print(f'[INFO] hiddenimports count: {len(hiddenimports)}  datas: {len(datas)}  binaries: {len(binaries)}')

a = Analysis(
    [ENTRY_SCRIPT],
    pathex=[PROJECT_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(PROJECT_ROOT, 'pyi_hooks')],
    hooksconfig={},
    runtime_hooks=[RTH],
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
