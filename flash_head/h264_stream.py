"""
Encode RGB frame batches to H.264 for low-latency WS streaming.

Wire format for mobile: short self-contained MP4 (frag moov) — Android WebView
can decode via <video>, unlike raw Annex-B + WebCodecs.

Encoder: libx264 ultrafast by default; FLASHHEAD_H264_ENCODER=nvenc for NVENC.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Optional

import numpy as np

logger = logging.getLogger("flash_head.h264")

_FFMPEG = shutil.which("ffmpeg")
_PREFERRED_ENCODER: Optional[str] = None  # h264_nvenc | libx264 | ""


def h264_available() -> bool:
    return bool(_FFMPEG) and _resolve_encoder() is not None


def _resolve_encoder() -> Optional[str]:
    global _PREFERRED_ENCODER
    if _PREFERRED_ENCODER is not None:
        return _PREFERRED_ENCODER or None
    if not _FFMPEG:
        _PREFERRED_ENCODER = ""
        return None
    try:
        probe = subprocess.run(
            [_FFMPEG, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        text = probe.stdout or ""
    except Exception as e:
        logger.warning(f"ffmpeg encoder probe failed: {e}")
        _PREFERRED_ENCODER = ""
        return None
    forced = (os.environ.get("FLASHHEAD_H264_ENCODER") or "").strip().lower()
    if forced in ("h264_nvenc", "nvenc"):
        if "h264_nvenc" in text:
            _PREFERRED_ENCODER = "h264_nvenc"
            logger.info("H.264 encoder=h264_nvenc (forced)")
            return _PREFERRED_ENCODER
        logger.warning("FLASHHEAD_H264_ENCODER=nvenc but h264_nvenc missing; trying libx264")
    if "libx264" in text:
        _PREFERRED_ENCODER = "libx264"
        logger.info("H.264 encoder=libx264")
        return _PREFERRED_ENCODER
    if "h264_nvenc" in text:
        _PREFERRED_ENCODER = "h264_nvenc"
        logger.info("H.264 encoder=h264_nvenc")
        return _PREFERRED_ENCODER
    _PREFERRED_ENCODER = ""
    logger.warning("no H.264 encoder found in ffmpeg")
    return None


def _encoder_args(enc: str, crf_i: int, n_frames: int) -> list[str]:
    if enc == "h264_nvenc":
        return [
            "-preset",
            "p1",
            "-tune",
            "ll",
            "-rc",
            "vbr",
            "-cq",
            str(crf_i),
            "-b:v",
            "0",
            "-bf",
            "0",
            "-g",
            str(n_frames),
        ]
    return [
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-bf",
        "0",
        "-g",
        str(n_frames),
        "-crf",
        str(crf_i),
    ]


def _prepare_frames(frames: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected (T,H,W,3) RGB, got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    t, h, w, _ = frames.shape
    if t <= 0:
        raise ValueError("empty frame batch")
    if (h % 2) or (w % 2):
        raise ValueError(f"H.264 requires even HxW, got {h}x{w}")
    return np.ascontiguousarray(frames), t, h, w


def encode_frames_mp4(
    frames: np.ndarray,
    fps: int,
    *,
    crf: int = 26,
) -> bytes:
    """
    frames: (T,H,W,3) uint8 RGB → fragmented MP4 (H.264, no audio).
    Mobile-friendly: starts with moov, decodable by HTML <video>.
    """
    frames, t, h, w = _prepare_frames(frames)
    enc = _resolve_encoder()
    if not enc or not _FFMPEG:
        raise RuntimeError("ffmpeg H.264 encoder unavailable")

    fps_i = max(1, int(fps))
    crf_i = int(np.clip(crf, 16, 40))
    cmd = [
        _FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps_i),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        enc,
        *_encoder_args(enc, crf_i, t),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "mp4",
        "-movflags",
        "frag_keyframe+empty_moov+default_base_moof",
        "pipe:1",
    ]
    raw = frames.tobytes()
    try:
        proc = subprocess.run(
            cmd,
            input=raw,
            capture_output=True,
            timeout=max(8.0, t * 0.5),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("ffmpeg H.264/mp4 encode timed out") from e
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg mp4 encode failed rc={proc.returncode}: {err}")
    if proc.stdout[:4] != b"\x00\x00\x00" and b"ftyp" not in proc.stdout[:64]:
        # fragmented mp4 still usually starts with size+ftyp or styp
        if b"ftyp" not in proc.stdout[:128] and b"styp" not in proc.stdout[:128] and b"moof" not in proc.stdout[:128]:
            raise RuntimeError("ffmpeg mp4 output missing ftyp/moof")
    return proc.stdout


def encode_frames_annexb(
    frames: np.ndarray,
    fps: int,
    *,
    crf: int = 26,
) -> bytes:
    """Legacy Annex-B bitstream (prefer encode_frames_mp4 for mobile)."""
    frames, t, h, w = _prepare_frames(frames)
    enc = _resolve_encoder()
    if not enc or not _FFMPEG:
        raise RuntimeError("ffmpeg H.264 encoder unavailable")

    fps_i = max(1, int(fps))
    crf_i = int(np.clip(crf, 16, 40))
    cmd = [
        _FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps_i),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        enc,
        *_encoder_args(enc, crf_i, t),
        "-pix_fmt",
        "yuv420p",
        "-f",
        "h264",
        "pipe:1",
    ]
    raw = frames.tobytes()
    try:
        proc = subprocess.run(
            cmd,
            input=raw,
            capture_output=True,
            timeout=max(8.0, t * 0.5),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("ffmpeg H.264 encode timed out") from e
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg H.264 encode failed rc={proc.returncode}: {err}")
    return proc.stdout
