"""
PyInstaller run-time hook for FlashHead_PCM_Server.

torch.jit.script(fn) requires the original .py source, resolved via
inspect.getsourcefile(fn) -> fn.__code__.co_filename -> linecache.  In a frozen
onedir bundle the pure-Python modules live inside the PYZ archive, so PyInstaller
refers to them by a RELATIVE name like `yunchang\\ring\\utils.py` and there is no
real file at that path -> inspect.getsourcelines() raises OSError.

We materialise the package source (yunchang) on disk via collect_data_files(...)
and seed `linecache.cache` for every such .py so that TorchScript can parse the
function source when it needs to.
"""
import linecache
import os
import sys


_SOURCED_PACKAGES = ("yunchang", "sageattention")


def _seed_source_for_package(pkg_name: str) -> None:
    _MEIPASS = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    root = os.path.join(_MEIPASS, pkg_name)
    if not os.path.isdir(root):
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".py"):
                continue
            full = os.path.normpath(os.path.join(dirpath, name))
            try:
                with open(full, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            # Cache under every spelling of the path that inspect might compute as
            # __code__.co_filename (relative to _MEIPASS with either slash style,
            # plus the absolute path).
            rel = os.path.relpath(full, _MEIPASS).replace("\\", "/")
            for key in (
                os.path.relpath(full, _MEIPASS),  # backslash relative (co_filename)
                full,  # absolute path
                rel,  # forward-slash relative
                "/" + rel,
                f"{_MEIPASS}\\{rel}",
            ):
                linecache.cache[key] = (len(full), None, lines, full)


for _pkg in _SOURCED_PACKAGES:
    _seed_source_for_package(_pkg)