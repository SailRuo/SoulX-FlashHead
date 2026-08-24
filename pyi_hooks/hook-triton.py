# ------------------------------------------------------------------
# Custom triton hook for FlashHead_PCM_Server (overrides std hook-triton.py)
#
# torch._inductor needs triton to compile CUDA kernels (FlashHead uses
# @torch.compile). triton's own triton_key() hashes triton.compiler.* and
# triton.backends.* source files via find_spec(name).origin. Modules bundled into
# the PYZ archive report origin=None, which crashes torch's FxGraphCache with
# TypeError. To fix it every triton module must live on disk as real .py source,
# so we force the WHOLE triton package into 'py' (source-on-disk) collection mode.
# ------------------------------------------------------------------
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, is_module_satisfies

hiddenimports = []
datas = []

# Keep all of triton's source .py on disk (whole-tree: submodules inherit 'py').
module_collection_mode = {'triton': 'py'}

# Ensure triton/_C/libtriton.pyd (and any other native libs) are bundled.
binaries = collect_dynamic_libs('triton')

# triton 3.0.0+ has triton.backends with backend-specific source files (#discover).
if is_module_satisfies('triton >= 3.0.0'):
    hiddenimports += collect_submodules('triton.backends')
    hiddenimports += collect_submodules('triton.backends.amd')
    datas += collect_data_files('triton.backends')
else:
    datas += collect_data_files('triton.third_party.cuda')