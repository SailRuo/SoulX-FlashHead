"""
Optional Real-ESRGAN post-enhance for FlashHead frame batches.

Preferred backend: realesrgan-ncnn-py (Vulkan) — does not steal FlashHead CUDA VRAM.
Fallback: OpenCV Lanczos 2x + unsharp (always available, no extra deps).

Wire: RGB uint8 (T,H,W,3) → enhance → JPEG encode.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

logger = logging.getLogger("flash_head.enhance")


@contextmanager
def _silence_native_stdio() -> Iterator[None]:
    """Mute ncnn C++ progress on stderr (0.00% / 16.67% / …). Also mute stdout."""
    # Progress is fprintf(stderr) — stdout-only redirect does nothing.
    # Process-wide: hold lock so concurrent pack threads don't stomp fds.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    saved_out = saved_err = devnull_fd = -1
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_out = os.dup(1)
        saved_err = os.dup(2)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    except Exception:
        if saved_out >= 0:
            try:
                os.close(saved_out)
            except Exception:
                pass
        if saved_err >= 0:
            try:
                os.close(saved_err)
            except Exception:
                pass
        if devnull_fd >= 0:
            try:
                os.close(devnull_fd)
            except Exception:
                pass
        yield
        return
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
        finally:
            try:
                os.close(saved_out)
            except Exception:
                pass
            try:
                os.close(saved_err)
            except Exception:
                pass
            try:
                os.close(devnull_fd)
            except Exception:
                pass


_SILENCE_LOCK = threading.Lock()


@contextmanager
def _silence_native_stdio_locked() -> Iterator[None]:
    with _SILENCE_LOCK:
        with _silence_native_stdio():
            yield


# realesrgan_ncnn_py model ids (see package docs)
NCNN_MODELS = {
    "animevideov3-x2": 0,  # realtime, 2x — recommended for live DH
    "animevideov3-x3": 1,
    "animevideov3-x4": 2,
    "x4plus-anime": 3,
    "x4plus": 4,  # best quality, slower
}


@dataclass
class EnhanceConfig:
    enabled: bool = False
    backend: str = "auto"  # auto | ncnn | opencv
    model: str = "animevideov3-x2"
    """Target long-side after enhance. 0 = keep model native scale (e.g. 2x)."""
    out_long_side: int = 1024
    """NCNN tile size; 0 = auto."""
    tile: int = 0
    gpuid: int = 0
    """Run Real-ESRGAN every Nth frame; in-betweens are blended. 1 = all frames."""
    stride: int = 2


class FrameEnhancer:
    """Process one FrameBatch's RGB frames. Thread-safe for multi-worker NCNN."""

    def __init__(self, config: EnhanceConfig):
        self.config = config
        self._backend = "off"
        self._ncnn_pool: list[Any] = []
        self._worker_locks: list[threading.Lock] = []
        self._lock = threading.Lock()
        self._model_scale = 2
        if not config.enabled:
            return
        backend = (config.backend or "auto").strip().lower()
        if backend in ("auto", "ncnn"):
            if self._try_init_ncnn():
                return
            if backend == "ncnn":
                logger.warning(
                    "enhance_backend=ncnn requested but realesrgan-ncnn-py unavailable; "
                    "falling back to opencv"
                )
        self._backend = "opencv"
        logger.info("FrameEnhancer backend=opencv (Lanczos+unsharp)")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def active(self) -> bool:
        return self.config.enabled and self._backend != "off"

    def info(self) -> dict:
        return {
            "enhance": self.config.enabled,
            "enhance_backend": self._backend if self.config.enabled else "off",
            "enhance_model": self.config.model,
            "enhance_out_long_side": int(self.config.out_long_side),
            "enhance_workers": len(self._ncnn_pool) if self._ncnn_pool else 1,
            "enhance_stride": int(getattr(self.config, "stride", 1) or 1),
        }

    def _try_init_ncnn(self) -> bool:
        try:
            from realesrgan_ncnn_py import Realesrgan  # type: ignore
        except Exception as e:
            logger.info(f"realesrgan-ncnn-py not available: {e}")
            return False
        model_key = (self.config.model or "animevideov3-x2").strip().lower()
        # aliases
        aliases = {
            "x4fast": "animevideov3-x2",
            "fast": "animevideov3-x2",
            "realesrgan-x4fast": "animevideov3-x2",
            "realesrgan-x4plus": "x4plus",
            "quality": "x4plus",
        }
        model_key = aliases.get(model_key, model_key)
        model_id = NCNN_MODELS.get(model_key, 0)
        scale_by_id = {0: 2, 1: 3, 2: 4, 3: 4, 4: 4}
        self._model_scale = scale_by_id.get(model_id, 2)
        # Default 1 worker: FlashHead CUDA + NCNN Vulkan share one GPU; 2 workers
        # make denoise steps jump ~52ms → 200–400ms under overlap.
        try:
            workers = int(os.environ.get("FLASHHEAD_ENHANCE_WORKERS", "1") or "1")
        except ValueError:
            workers = 1
        workers = max(1, min(4, workers))
        pool: list[Any] = []
        try:
            for i in range(workers):
                pool.append(
                    Realesrgan(
                        gpuid=int(self.config.gpuid),
                        tta_mode=False,
                        tilesize=int(self.config.tile or 0),
                        model=int(model_id),
                    )
                )
            self._ncnn_pool = pool
            self._worker_locks = [threading.Lock() for _ in pool]
            self._backend = "ncnn"
            logger.info(
                f"FrameEnhancer backend=ncnn model={model_key}(id={model_id}) "
                f"scale={self._model_scale} out_long_side={self.config.out_long_side} "
                f"workers={workers} stride={max(1, int(self.config.stride or 1))}"
            )
            return True
        except Exception as e:
            logger.warning(f"failed to init realesrgan-ncnn-py: {e}")
            self._ncnn_pool = []
            self._worker_locks = []
            return False

    def enhance_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        frames: (T,H,W,3) uint8 RGB
        returns same layout, possibly larger H/W
        """
        if not self.active:
            return frames
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected (T,H,W,3) RGB, got {frames.shape}")
        if self._backend == "ncnn":
            return self._enhance_ncnn(frames)
        return self._enhance_opencv(frames)

    def _key_indices(self, t: int) -> list[int]:
        stride = max(1, int(self.config.stride or 1))
        if stride <= 1 or t <= 2:
            return list(range(t))
        keys = list(range(0, t, stride))
        if keys[-1] != t - 1:
            keys.append(t - 1)
        return keys

    def _fill_stride_gaps(self, out: np.ndarray, keys: list[int]) -> None:
        """Linear-blend missing frames between enhanced keyframes (in-place)."""
        t = out.shape[0]
        if len(keys) >= t:
            return
        key_set = set(keys)
        for i in range(t):
            if i in key_set:
                continue
            # find surrounding keys
            prev_k = keys[0]
            next_k = keys[-1]
            for k in keys:
                if k <= i:
                    prev_k = k
                if k >= i:
                    next_k = k
                    break
            if next_k == prev_k:
                out[i] = out[prev_k]
                continue
            a = float(i - prev_k) / float(next_k - prev_k)
            # int blend is fine for speech; avoids float alloc churn
            out[i] = (
                out[prev_k].astype(np.float32) * (1.0 - a)
                + out[next_k].astype(np.float32) * a
            ).astype(np.uint8)

    def _target_size(self, h: int, w: int) -> tuple[int, int]:
        """Compute output (nh, nw) from out_long_side or native model scale."""
        out_long = int(self.config.out_long_side or 0)
        if out_long > 0:
            long = max(h, w)
            scale = float(out_long) / float(long)
            nw = max(16, int(round(w * scale)) // 2 * 2)
            nh = max(16, int(round(h * scale)) // 2 * 2)
            return nh, nw
        # native model scale
        s = float(self._model_scale)
        nw = max(16, int(round(w * s)) // 2 * 2)
        nh = max(16, int(round(h * s)) // 2 * 2)
        return nh, nw

    def _one_ncnn_frame(
        self,
        worker_idx: int,
        rgb: np.ndarray,
        nh: int,
        nw: int,
    ) -> np.ndarray:
        import cv2

        ncnn = self._ncnn_pool[worker_idx]
        lock = self._worker_locks[worker_idx]
        with lock:
            with _silence_native_stdio_locked():
                bgr = rgb[:, :, ::-1].copy()
                try:
                    if hasattr(ncnn, "process_cv2"):
                        up = ncnn.process_cv2(bgr)
                    else:
                        from PIL import Image

                        pil = Image.fromarray(rgb)
                        up_pil = ncnn.process_pil(pil)
                        up_rgb = np.asarray(up_pil)
                        if up_rgb.shape[0] != nh or up_rgb.shape[1] != nw:
                            up_rgb = cv2.resize(up_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
                        return up_rgb
                except Exception as e:
                    logger.warning(f"ncnn enhance frame failed: {e}; opencv fallback")
                    return self._one_opencv(rgb, nh, nw)

                if up is None:
                    return self._one_opencv(rgb, nh, nw)
                up_rgb = up[:, :, ::-1]
                if up_rgb.shape[0] != nh or up_rgb.shape[1] != nw:
                    interp = cv2.INTER_AREA if (up_rgb.shape[0] > nh) else cv2.INTER_LANCZOS4
                    up_rgb = cv2.resize(up_rgb, (nw, nh), interpolation=interp)
                return up_rgb

    def _enhance_ncnn(self, frames: np.ndarray) -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time as _time

        t, h, w, _ = frames.shape
        nh, nw = self._target_size(h, w)
        out = np.empty((t, nh, nw, 3), dtype=np.uint8)
        keys = self._key_indices(t)
        n_workers = max(1, len(self._ncnn_pool))
        t0 = _time.time()
        if len(keys) == 1 or n_workers == 1:
            for i in keys:
                out[i] = self._one_ncnn_frame(0, frames[i], nh, nw)
        else:
            with ThreadPoolExecutor(max_workers=min(n_workers, len(keys))) as ex:
                futs = {
                    ex.submit(
                        self._one_ncnn_frame, j % n_workers, frames[i], nh, nw
                    ): i
                    for j, i in enumerate(keys)
                }
                for fut in as_completed(futs):
                    i = futs[fut]
                    out[i] = fut.result()
        self._fill_stride_gaps(out, keys)
        logger.info(
            f"ncnn enhance T={t} keys={len(keys)}/{t} stride={max(1, int(self.config.stride or 1))} "
            f"{w}x{h}->{nw}x{nh} workers={n_workers} cost={_time.time() - t0:.3f}s"
        )
        return out

    def _enhance_opencv(self, frames: np.ndarray) -> np.ndarray:
        t, h, w, _ = frames.shape
        nh, nw = self._target_size(h, w)
        out = np.empty((t, nh, nw, 3), dtype=np.uint8)
        keys = self._key_indices(t)
        for i in keys:
            out[i] = self._one_opencv(frames[i], nh, nw)
        self._fill_stride_gaps(out, keys)
        return out

    @staticmethod
    def _one_opencv(rgb: np.ndarray, nh: int, nw: int) -> np.ndarray:
        import cv2

        h, w = rgb.shape[:2]
        if h != nh or w != nw:
            interp = cv2.INTER_LANCZOS4 if (nh > h or nw > w) else cv2.INTER_AREA
            rgb = cv2.resize(rgb, (nw, nh), interpolation=interp)
        # Mild unsharp — recovers lite-model softness without haloing
        blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(rgb, 1.35, blur, -0.35, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)


_enhancer: Optional[FrameEnhancer] = None


def parse_enhance_config(msg: dict) -> EnhanceConfig:
    """Parse WS `start` message fields into EnhanceConfig."""
    enabled = bool(msg.get("enhance", msg.get("enable_enhance", False)))
    # Accept nested dict too
    nested = msg.get("enhance_options") if isinstance(msg.get("enhance_options"), dict) else {}
    backend = str(msg.get("enhance_backend") or nested.get("backend") or "auto")
    model = str(msg.get("enhance_model") or nested.get("model") or "animevideov3-x2")
    out_long = msg.get("enhance_out_long_side", nested.get("out_long_side", 1024))
    try:
        out_long_side = int(out_long if out_long is not None else 1024)
    except (TypeError, ValueError):
        out_long_side = 1024
    tile = msg.get("enhance_tile", nested.get("tile", 0))
    try:
        tile_i = int(tile or 0)
    except (TypeError, ValueError):
        tile_i = 0
    gpuid = msg.get("enhance_gpuid", nested.get("gpuid", 0))
    try:
        gpuid_i = int(gpuid if gpuid is not None else 0)
    except (TypeError, ValueError):
        gpuid_i = 0
    # Default stride=2 from env; client may override via enhance_stride.
    env_stride = os.environ.get("FLASHHEAD_ENHANCE_STRIDE", "4")
    raw_stride = msg.get("enhance_stride", nested.get("stride", env_stride))
    try:
        stride_i = int(raw_stride if raw_stride is not None else 4)
    except (TypeError, ValueError):
        stride_i = 4
    stride_i = max(1, min(8, stride_i))
    return EnhanceConfig(
        enabled=enabled,
        backend=backend,
        model=model,
        out_long_side=out_long_side,
        tile=tile_i,
        gpuid=gpuid_i,
        stride=stride_i,
    )


def create_enhancer(config: EnhanceConfig) -> FrameEnhancer:
    return FrameEnhancer(config)
