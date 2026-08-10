"""
WebSocket realtime API: image upload + PCM in, video chunks out.

URL: ws://127.0.0.1:8765/ws

Typical flow:
  1) text  {"type":"upload_image","filename":"face.png"}
     binary <image bytes>
     text  {"type":"image_ready","cond_image":"stream_uploads/..."}
  2) text  {"type":"start","cond_image":"<path or omit to use last upload>",
            "model_type":"lite","sample_rate":24000,"fmt":"s16le",
            "send_mp4_binary":true}
     text  {"type":"ready", ...}
  3) binary PCM frames ...
     text  {"type":"video", ...} + binary mp4
  4) text  {"type":"flush"} -> videos -> {"type":"done"}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from loguru import logger
from PIL import Image
import numpy as np
from websockets.asyncio.server import ServerConnection, serve

from flash_head.pcm_stream import (
    FlashHeadPCMSession,
    FrameBatch,
    VideoChunk,
    preload_and_warmup,
)
from flash_head.frame_enhance import create_enhancer, parse_enhance_config
import struct

UPLOAD_DIR = "stream_uploads"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PendingBinary = Optional[Literal["image"]]


def _safe_filename(name: str) -> str:
    base = os.path.basename(name or "face.png")
    base = re.sub(r"[^\w.\-]+", "_", base)
    root, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in ALLOWED_IMAGE_EXT:
        ext = ".png"
    if not root:
        root = "face"
    return root + ext


def _save_uploaded_image(data: bytes, filename: str) -> str:
    if not data:
        raise ValueError("empty image binary")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image too large (>{MAX_IMAGE_BYTES} bytes)")

    # Validate it's a real image
    with Image.open(BytesIO(data)) as im:
        im = im.convert("RGB")
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = _safe_filename(filename)
        root, ext = os.path.splitext(safe)
        out_name = f"{stamp}_{uuid4().hex[:8]}_{root}.png"
        out_path = os.path.join(UPLOAD_DIR, out_name)
        im.save(out_path, format="PNG")
    return out_path.replace("\\", "/")


def _session_info(session: FlashHeadPCMSession) -> Dict[str, Any]:
    return {
        "session_id": session.session_id,
        "model_type": session.model_type,
        "input_sample_rate": session.input_sample_rate,
        "model_sample_rate": session.model_sample_rate,
        "fps": session.fps,
        "chunk_samples": session.chunk_samples,
        "input_chunk_samples": session.input_chunk_samples,
        "chunk_seconds": session.chunk_samples / session.model_sample_rate,
        "pending_samples": session.pending_samples,
        "samples_needed_for_next_chunk": session.samples_needed_for_next_chunk,
        "closed": session._closed,
        "mux_audio": bool(getattr(session, "mux_audio", True)),
        "chunks_per_segment": int(getattr(session, "chunks_per_segment", 1)),
        "first_segment_chunks": int(getattr(session, "first_segment_chunks", 1)),
        "emit_on_flush_only": bool(getattr(session, "emit_on_flush_only", False)),
        "stream_mode": getattr(session, "stream_mode", "mp4"),
        "aspect_ratio": getattr(session, "aspect_ratio", "1:1"),
        "height": int(getattr(session, "out_height", 512)),
        "width": int(getattr(session, "out_width", 512)),
        "max_long_side": int(getattr(session, "max_long_side", 1024)),
        "timeline_pts": float(getattr(session, "_timeline_pts", 0.0)),
        "segment_seconds": (
            session.chunk_samples / session.model_sample_rate
        )
        * int(getattr(session, "chunks_per_segment", 1)),
    }


def _video_meta(
    chunk: VideoChunk,
    byte_length: Optional[int] = None,
    *,
    subtitle: Optional[str] = None,
    subtitle_id: Optional[str] = None,
) -> Dict[str, Any]:
    meta = {
        "type": "video",
        "chunk_idx": chunk.chunk_idx,
        "n_frames": int(chunk.frames.shape[0]),
        "height": int(chunk.frames.shape[1]),
        "width": int(chunk.frames.shape[2]),
        "fps": chunk.fps,
        "elapsed_sec": chunk.elapsed_sec,
        "video_path": chunk.video_path,
        "has_audio": True,
        "n_micro_chunks": int(getattr(chunk, "n_micro_chunks", 1)),
    }
    if byte_length is not None:
        meta["byte_length"] = int(byte_length)
    if subtitle:
        meta["subtitle"] = subtitle
        if subtitle_id:
            meta["subtitle_id"] = subtitle_id
    return meta


async def _send_json(
    ws: ServerConnection,
    payload: Dict[str, Any],
    lock: Optional[asyncio.Lock] = None,
) -> None:
    data = json.dumps(payload, ensure_ascii=False)
    if lock is None:
        await ws.send(data)
    else:
        async with lock:
            await ws.send(data)


async def _send_bytes(
    ws: ServerConnection,
    data: bytes,
    lock: Optional[asyncio.Lock] = None,
) -> None:
    if lock is None:
        await ws.send(data)
    else:
        async with lock:
            await ws.send(data)


async def _emit_videos(
    ws: ServerConnection,
    chunks: list[VideoChunk],
    send_mp4_binary: bool,
    lock: Optional[asyncio.Lock] = None,
    *,
    subtitle: Optional[str] = None,
    subtitle_id: Optional[str] = None,
) -> None:
    for chunk in chunks:
        mp4_bytes: Optional[bytes] = None
        if send_mp4_binary and chunk.video_path and os.path.exists(chunk.video_path):
            with open(chunk.video_path, "rb") as f:
                mp4_bytes = f.read()
        await _send_json(
            ws,
            _video_meta(
                chunk,
                byte_length=len(mp4_bytes) if mp4_bytes else None,
                subtitle=subtitle,
                subtitle_id=subtitle_id,
            ),
            lock,
        )
        if mp4_bytes is not None:
            await _send_bytes(ws, mp4_bytes, lock)


def _pack_frame_batch_binary(
    batch: FrameBatch,
    jpeg_quality: int,
    stream_max_long_side: Optional[int] = None,
    enhancer: Optional[Any] = None,
    *,
    video_codec: str = "jpeg",
    h264_crf: int = 26,
) -> tuple[bytes, int, int, str]:
    """Pack audio + video frames.

    jpeg:  u32be audio_len | pcm | u32be n | (u32be len | jpeg)*n
    h264:  u32be audio_len | pcm | u32be annexb_len | annexb

    Returns (payload, out_height, out_width, wire_codec).
    """
    import time as _time
    from dataclasses import replace

    t0 = _time.time()
    work = batch
    enhance_tag = "off"
    codec = (video_codec or "jpeg").strip().lower()
    if codec not in ("jpeg", "jpg", "h264", "avc"):
        codec = "jpeg"
    if codec in ("jpg",):
        codec = "jpeg"
    if codec in ("avc",):
        codec = "h264"

    # Speech frames may go through Real-ESRGAN; final_ref must stay the true
    # FlashHead cond (crop/resize). Lanczos-only upscale keeps size continuous.
    if getattr(batch, "is_final_ref", False):
        stream_max_long_side = None
        enhance_tag = "final-ref"
        out_long = 0
        if enhancer is not None and getattr(enhancer, "config", None) is not None:
            try:
                out_long = int(getattr(enhancer.config, "out_long_side", 0) or 0)
            except (TypeError, ValueError):
                out_long = 0
        if out_long > 0:
            import cv2

            h, w = int(work.frames.shape[1]), int(work.frames.shape[2])
            if max(h, w) < out_long:
                scale = float(out_long) / float(max(h, w))
                nw = max(16, int(round(w * scale)) // 2 * 2)
                nh = max(16, int(round(h * scale)) // 2 * 2)
                resized = np.stack(
                    [
                        cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
                        for f in work.frames
                    ],
                    axis=0,
                )
                work = replace(batch, frames=resized)
                enhance_tag = f"final-ref-lanczos:{nw}x{nh}"
    elif enhancer is not None and getattr(enhancer, "active", False):
        t_e = _time.time()
        enhanced = enhancer.enhance_frames(batch.frames)
        work = replace(batch, frames=enhanced)
        enhance_tag = f"{getattr(enhancer, 'backend', '?')}:{_time.time() - t_e:.3f}s"
        # Enhanced frames already target out_long_side — do not crush back to 512.
        stream_max_long_side = None

    # Optional wire downscale (jpeg path / preview). H.264 keeps full enhance size.
    out_h, out_w = int(work.frames.shape[1]), int(work.frames.shape[2])
    if stream_max_long_side and max(out_h, out_w) > int(stream_max_long_side):
        import cv2
        from dataclasses import replace as _replace

        scale = float(stream_max_long_side) / float(max(out_h, out_w))
        nw = max(16, int(round(out_w * scale)) // 2 * 2)
        nh = max(16, int(round(out_h * scale)) // 2 * 2)
        resized = np.stack(
            [cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in work.frames],
            axis=0,
        )
        work = _replace(work, frames=resized)
        out_h, out_w = nh, nw

    audio = work.audio_s16le()
    wire_codec = "jpeg"
    if codec == "h264":
        try:
            from flash_head.h264_stream import encode_frames_mp4, h264_available

            if not h264_available():
                raise RuntimeError("ffmpeg H.264 unavailable")
            # yuv420p needs even dims; many phones refuse >1920 long-side HW decode
            h, w = int(work.frames.shape[1]), int(work.frames.shape[2])
            max_wire = 1920
            if max(h, w) > max_wire or (h % 2) or (w % 2):
                import cv2
                from dataclasses import replace as _replace

                scale = min(1.0, float(max_wire) / float(max(h, w)))
                nw = max(16, int(round(w * scale)) // 2 * 2)
                nh = max(16, int(round(h * scale)) // 2 * 2)
                if nw != w or nh != h:
                    resized = np.stack(
                        [
                            cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
                            for f in work.frames
                        ],
                        axis=0,
                    )
                    work = _replace(work, frames=resized)
                    out_h, out_w = nh, nw
            mp4 = encode_frames_mp4(work.frames, int(work.fps), crf=int(h264_crf))
            parts = [
                struct.pack(">I", len(audio)),
                audio,
                struct.pack(">I", len(mp4)),
                mp4,
            ]
            payload = b"".join(parts)
            wire_codec = "mp4"
            logger.info(
                f"pack frame_batch#{batch.chunk_idx} mp4/h264 frames={work.n_frames} "
                f"{out_w}x{out_h} bytes={len(payload)} mp4={len(mp4)} "
                f"cost={_time.time() - t0:.3f}s crf={h264_crf} enhance={enhance_tag}"
            )
            return payload, out_h, out_w, wire_codec
        except Exception as e:
            logger.warning(f"H.264/mp4 pack failed, falling back to JPEG: {e}")

    jpegs = work.encode_jpegs(quality=jpeg_quality, max_long_side=None)
    out_h, out_w = work.height, work.width
    parts = [struct.pack(">I", len(audio)), audio, struct.pack(">I", len(jpegs))]
    for jpg in jpegs:
        parts.append(struct.pack(">I", len(jpg)))
        parts.append(jpg)
    payload = b"".join(parts)
    logger.info(
        f"pack frame_batch#{batch.chunk_idx} jpegs={len(jpegs)} "
        f"{out_w}x{out_h} bytes={len(payload)} cost={_time.time() - t0:.3f}s "
        f"q={jpeg_quality} stream_max={stream_max_long_side} enhance={enhance_tag}"
    )
    return payload, out_h, out_w, wire_codec


async def _emit_frame_batches(
    ws: ServerConnection,
    batches: list[FrameBatch],
    jpeg_quality: int,
    stream_max_long_side: Optional[int] = None,
    lock: Optional[asyncio.Lock] = None,
    *,
    subtitle: Optional[str] = None,
    subtitle_id: Optional[str] = None,
    enhancer: Optional[Any] = None,
    video_codec: str = "jpeg",
    h264_crf: int = 26,
) -> None:
    for batch in batches:
        # CPU/Vulkan pack overlaps other work; only the socket write is serialized.
        payload, out_h, out_w, wire_codec = await asyncio.to_thread(
            _pack_frame_batch_binary,
            batch,
            jpeg_quality,
            stream_max_long_side,
            enhancer,
            video_codec=video_codec,
            h264_crf=h264_crf,
        )
        meta: Dict[str, Any] = {
            "type": "frame_batch",
            "chunk_idx": batch.chunk_idx,
            "n_frames": batch.n_frames,
            "pts0": batch.pts0,
            "fps": batch.fps,
            "height": out_h,
            "width": out_w,
            "audio_sample_rate": batch.audio_sample_rate,
            "audio_fmt": "s16le",
            "elapsed_sec": batch.elapsed_sec,
            "byte_length": len(payload),
            "duration_sec": float(batch.n_frames) / float(batch.fps or 25),
            "video_codec": wire_codec,
        }
        if getattr(batch, "is_final_ref", False):
            meta["final_ref"] = True
        if subtitle and not getattr(batch, "is_final_ref", False):
            meta["subtitle"] = subtitle
            if subtitle_id:
                meta["subtitle_id"] = subtitle_id
        await _send_json(ws, meta, lock)
        await _send_bytes(ws, payload, lock)


async def _emit_stream_items(
    ws: ServerConnection,
    items: list,
    *,
    send_mp4_binary: bool,
    jpeg_quality: int,
    stream_max_long_side: Optional[int] = None,
    lock: Optional[asyncio.Lock] = None,
    subtitle: Optional[str] = None,
    subtitle_id: Optional[str] = None,
    enhancer: Optional[Any] = None,
    video_codec: str = "jpeg",
    h264_crf: int = 26,
) -> None:
    videos: list[VideoChunk] = []
    frames: list[FrameBatch] = []
    for item in items:
        if isinstance(item, FrameBatch):
            if videos:
                await _emit_videos(
                    ws,
                    videos,
                    send_mp4_binary,
                    lock,
                    subtitle=subtitle,
                    subtitle_id=subtitle_id,
                )
                videos = []
            frames.append(item)
        elif isinstance(item, VideoChunk):
            if frames:
                await _emit_frame_batches(
                    ws,
                    frames,
                    jpeg_quality,
                    stream_max_long_side,
                    lock,
                    subtitle=subtitle,
                    subtitle_id=subtitle_id,
                    enhancer=enhancer,
                    video_codec=video_codec,
                    h264_crf=h264_crf,
                )
                frames = []
            videos.append(item)
    if frames:
        await _emit_frame_batches(
            ws,
            frames,
            jpeg_quality,
            stream_max_long_side,
            lock,
            subtitle=subtitle,
            subtitle_id=subtitle_id,
            enhancer=enhancer,
            video_codec=video_codec,
            h264_crf=h264_crf,
        )
    if videos:
        await _emit_videos(
            ws,
            videos,
            send_mp4_binary,
            lock,
            subtitle=subtitle,
            subtitle_id=subtitle_id,
        )


async def handle_connection(ws: ServerConnection) -> None:
    peer = getattr(ws, "remote_address", None)
    logger.info(f"WS connected: {peer}")

    session: Optional[FlashHeadPCMSession] = None
    fmt = "s16le"
    send_mp4_binary = True
    jpeg_quality = 82
    stream_max_long_side: Optional[int] = 512
    enhancer: Optional[Any] = None
    video_codec = "jpeg"
    h264_crf = 26
    pending_binary: PendingBinary = None
    pending_upload_filename = "face.png"
    uploaded_cond_image: Optional[str] = None
    emit_task: Optional[asyncio.Task] = None
    send_lock = asyncio.Lock()
    status_every_n = 8
    batches_since_status = 0
    # Client-provided caption stamped onto subsequent video / frame_batch metas.
    current_subtitle = ""
    current_subtitle_id = ""

    async def _drain_emit() -> None:
        nonlocal emit_task
        if emit_task is not None:
            try:
                await emit_task
            except asyncio.CancelledError:
                pass
            emit_task = None

    def _queue_emit(items: list) -> None:
        """Pipeline: pack/send previous batch overlaps next GPU generate."""
        nonlocal emit_task, batches_since_status
        if not items:
            return
        prev = emit_task
        n_new = sum(1 for it in items if isinstance(it, FrameBatch))
        # Snapshot so a later subtitle update does not rewrite in-flight emits.
        sub_text = current_subtitle
        sub_id = current_subtitle_id
        enh = enhancer

        async def _ordered() -> None:
            nonlocal batches_since_status
            try:
                if prev is not None:
                    await prev
                await _emit_stream_items(
                    ws,
                    items,
                    send_mp4_binary=send_mp4_binary,
                    jpeg_quality=jpeg_quality,
                    stream_max_long_side=stream_max_long_side,
                    lock=send_lock,
                    subtitle=sub_text or None,
                    subtitle_id=sub_id or None,
                    enhancer=enh,
                    video_codec=video_codec,
                    h264_crf=h264_crf,
                )
                if session is not None and n_new:
                    batches_since_status += n_new
                    if batches_since_status >= status_every_n:
                        batches_since_status = 0
                        await _send_json(
                            ws, {"type": "status", **_session_info(session)}, send_lock
                        )
            except Exception as e:
                logger.exception("emit failed")
                try:
                    await _send_json(
                        ws, {"type": "error", "error": f"emit failed: {e}"}, send_lock
                    )
                except Exception:
                    pass

        emit_task = asyncio.create_task(_ordered())

    async def sj(payload: Dict[str, Any]) -> None:
        await _send_json(ws, payload, send_lock)

    try:
        async for message in ws:
            if isinstance(message, bytes):
                # 1) image upload binary
                if pending_binary == "image":
                    pending_binary = None
                    try:
                        path = await asyncio.to_thread(
                            _save_uploaded_image, message, pending_upload_filename
                        )
                        uploaded_cond_image = path
                        wh = ""
                        try:
                            from PIL import Image

                            with Image.open(path) as im:
                                wh = f" {im.size[0]}x{im.size[1]}"
                        except Exception:
                            pass
                        await sj(
                            {
                                "type": "image_ready",
                                "cond_image": path,
                                "bytes": len(message),
                            }
                        )
                        logger.info(f"uploaded cond_image -> {path}{wh}")
                    except Exception as e:
                        logger.exception("image upload failed")
                        await sj({"type": "error", "error": str(e)})
                    continue

                # 2) PCM binary (after start/ready)
                if session is None:
                    await sj(
                        {
                            "type": "error",
                            "error": "unexpected binary: send upload_image (then image) or start first, then PCM",
                        }
                    )
                    continue
                if session._closed:
                    await sj({"type": "error", "error": "session already flushed/closed"})
                    continue
                try:
                    cancel_at = int(getattr(session, "_cancel_gen", 0))
                    items = await asyncio.to_thread(session.feed_pcm_bytes, message, fmt)
                    # Barge-in may have landed while this GPU chunk was running.
                    if int(getattr(session, "_cancel_gen", 0)) != cancel_at:
                        logger.info(
                            f"discard feed result after cancel "
                            f"gen={getattr(session, '_cancel_gen', 0)}"
                        )
                        continue
                    if items:
                        _queue_emit(items)
                        # Do not send status here — it raced with emit_task on the same WS.
                except Exception as e:
                    logger.exception("feed_pcm failed")
                    await sj({"type": "error", "error": str(e)})
                continue

            # text JSON control
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                await sj({"type": "error", "error": "invalid JSON text message"})
                continue

            mtype = msg.get("type")
            if mtype == "ping":
                await sj({"type": "pong"})
                continue

            if mtype == "upload_image":
                if session is not None:
                    await sj(
                        {
                            "type": "error",
                            "error": "upload_image only before start (or after reconnect)",
                        }
                    )
                    continue
                pending_upload_filename = str(msg.get("filename") or "face.png")
                pending_binary = "image"
                await sj(
                    {
                        "type": "upload_image_ack",
                        "filename": pending_upload_filename,
                        "message": "send next WebSocket binary frame as image bytes",
                    }
                )
                continue

            if mtype == "start":
                if session is not None and not session._closed:
                    await sj(
                        {
                            "type": "error",
                            "error": "session already started on this connection",
                        }
                    )
                    continue
                # Previous flush/done left a closed session: allow a new start on same WS.
                session = None
                enhancer = None
                current_subtitle = ""
                current_subtitle_id = ""
                if pending_binary is not None:
                    await sj(
                        {
                            "type": "error",
                            "error": "finish image upload binary before start",
                        }
                    )
                    continue

                cond_image = msg.get("cond_image") or uploaded_cond_image
                if not cond_image:
                    await sj(
                        {
                            "type": "error",
                            "error": "cond_image required (path or upload_image first)",
                        }
                    )
                    continue
                if not os.path.exists(cond_image):
                    await sj(
                        {
                            "type": "error",
                            "error": f"cond_image not found: {cond_image}",
                        }
                    )
                    continue

                fmt = str(msg.get("fmt", "s16le"))
                send_mp4_binary = bool(msg.get("send_mp4_binary", True))
                jpeg_quality = int(msg.get("jpeg_quality", 82))
                stream_mode = str(msg.get("stream_mode") or "mp4").strip().lower()
                # Wire preview long-side; None/0 = full native resolution JPEGs.
                raw_stream_max = msg.get("stream_max_long_side", 512)
                if raw_stream_max in (None, "", 0, "0"):
                    stream_max_long_side = None
                else:
                    stream_max_long_side = int(raw_stream_max)

                enhance_cfg = parse_enhance_config(msg)
                # Realtime DH default: if client asks enhance but leaves jpeg soft, bump it.
                if enhance_cfg.enabled and "jpeg_quality" not in msg:
                    jpeg_quality = max(jpeg_quality, 85)
                # When enhancing to 1024, do not downscale the wire preview to 512.
                if enhance_cfg.enabled and "stream_max_long_side" not in msg:
                    stream_max_long_side = None
                enhancer = create_enhancer(enhance_cfg)

                raw_codec = str(
                    msg.get("frame_video_codec")
                    or msg.get("video_codec")
                    or "jpeg"
                ).strip().lower()
                if raw_codec in ("h264", "avc", "h.264"):
                    video_codec = "h264"
                else:
                    video_codec = "jpeg"
                try:
                    h264_crf = int(msg.get("h264_crf", msg.get("video_crf", 26)))
                except (TypeError, ValueError):
                    h264_crf = 26
                h264_crf = max(16, min(40, h264_crf))
                if video_codec == "h264":
                    from flash_head.h264_stream import h264_available

                    if not h264_available():
                        logger.warning("frame_video_codec=h264 requested but ffmpeg H.264 missing; jpeg")
                        video_codec = "jpeg"
                    else:
                        # H.264 is already small — never crush enhance output for wire.
                        stream_max_long_side = None

                kwargs = {
                    "cond_image": cond_image,
                    "ckpt_dir": msg.get("ckpt_dir", "models/SoulX-FlashHead-1_3B"),
                    "wav2vec_dir": msg.get("wav2vec_dir", "models/wav2vec2-base-960h"),
                    "model_type": msg.get("model_type", "lite"),
                    "seed": int(msg.get("seed", 9999)),
                    "use_face_crop": bool(msg.get("use_face_crop", False)),
                    "save_dir": msg.get("save_dir", "stream_pcm_results"),
                    "save_mp4": bool(msg.get("save_mp4", stream_mode == "mp4")),
                    "mux_audio": bool(msg.get("mux_audio", True)),
                    "chunks_per_segment": int(msg.get("chunks_per_segment", 2)),
                    "first_segment_chunks": int(msg.get("first_segment_chunks", 1)),
                    "emit_on_flush_only": bool(msg.get("emit_on_flush_only", False)),
                    "stream_mode": stream_mode,
                    "jpeg_quality": jpeg_quality,
                    "sample_rate": int(msg.get("sample_rate", 16000)),
                    "aspect_ratio": msg.get("aspect_ratio") or msg.get("aspect") or "1:1",
                    "max_long_side": int(msg.get("max_long_side", 1024)),
                }
                if msg.get("height") is not None:
                    kwargs["height"] = int(msg["height"])
                if msg.get("width") is not None:
                    kwargs["width"] = int(msg["width"])
                await sj({"type": "loading", "message": "loading model..."})
                try:
                    session = await asyncio.to_thread(FlashHeadPCMSession, **kwargs)
                except Exception as e:
                    logger.exception("failed to start session")
                    await sj({"type": "error", "error": str(e)})
                    enhancer = None
                    continue

                h, w = int(session.out_height), int(session.out_width)
                await sj(
                    {
                        "type": "loading",
                        "message": (
                            f"warming {h}x{w} — first time at this resolution may take 1–2 min"
                        ),
                    }
                )
                try:
                    await asyncio.to_thread(session.warmup_inference)
                except Exception as e:
                    logger.exception("resolution warmup failed")
                    await sj({"type": "error", "error": f"warmup failed: {e}"})
                    session = None
                    enhancer = None
                    continue

                await sj(
                    {
                        "type": "ready",
                        "fmt": fmt,
                        "send_mp4_binary": send_mp4_binary,
                        "cond_image": cond_image,
                        "jpeg_quality": jpeg_quality,
                        "stream_max_long_side": stream_max_long_side,
                        "video_codec": video_codec,
                        "h264_crf": h264_crf,
                        **(enhancer.info() if enhancer is not None else {"enhance": False}),
                        **_session_info(session),
                    }
                )
                continue

            if mtype == "subtitle":
                # Stamp captions onto subsequent video / frame_batch metas.
                # FlashHead only sees PCM; the client owns the spoken text.
                text = str(msg.get("text") or "")
                sid = str(msg.get("id") or msg.get("subtitle_id") or "")
                current_subtitle = text.strip()
                current_subtitle_id = sid.strip()
                await sj(
                    {
                        "type": "subtitle_ack",
                        "subtitle": current_subtitle,
                        "subtitle_id": current_subtitle_id,
                    }
                )
                continue

            if mtype in ("cancel", "interrupt"):
                # Barge-in: drop buffered PCM + pending emits; keep session for next reply.
                current_subtitle = ""
                current_subtitle_id = ""
                if emit_task is not None:
                    emit_task.cancel()
                    try:
                        await emit_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        logger.exception("emit_task cancel drain failed")
                    emit_task = None
                cancel_gen = 0
                if session is not None and not session._closed:
                    try:
                        cancel_gen = int(session.cancel_pending())
                    except Exception as e:
                        logger.exception("cancel_pending failed")
                        await sj({"type": "error", "error": f"cancel failed: {e}"})
                        continue
                    # Snap UI back to the processed FlashHead reference frame.
                    try:
                        ref = session.make_final_ref_batch()
                        if ref is not None:
                            _queue_emit([ref])
                            await _drain_emit()
                    except Exception:
                        logger.exception("final_ref after cancel failed")
                logger.info(f"WS cancel/interrupt gen={cancel_gen} peer={peer}")
                await sj(
                    {
                        "type": "cancelled",
                        "cancel_gen": cancel_gen,
                        "final_ref": True,
                        **(
                            _session_info(session)
                            if session is not None and not session._closed
                            else {}
                        ),
                    }
                )
                continue

            if mtype == "flush":
                if session is None:
                    await sj({"type": "error", "error": "no active session"})
                    continue
                pad_silence = bool(msg.get("pad_silence", True))
                try:
                    items = await asyncio.to_thread(session.flush, pad_silence)
                    if items:
                        _queue_emit(items)
                    await _drain_emit()
                    await sj(
                        {
                            "type": "done",
                            "subtitle": current_subtitle or None,
                            "subtitle_id": current_subtitle_id or None,
                            **_session_info(session),
                        }
                    )
                    session = None
                    enhancer = None
                    current_subtitle = ""
                    current_subtitle_id = ""
                except Exception as e:
                    logger.exception("flush failed")
                    await sj({"type": "error", "error": str(e)})
                    session = None
                    enhancer = None
                continue

            if mtype == "close":
                await _drain_emit()
                await sj({"type": "bye"})
                break

            await sj({"type": "error", "error": f"unknown type: {mtype}"})
    except Exception as e:
        logger.exception(f"WS connection error: {e}")
    finally:
        try:
            await _drain_emit()
        except Exception:
            pass
        logger.info(f"WS disconnected: {peer}")


async def _run(
    host: str,
    port: int,
    *,
    preload: bool,
    warmup: bool,
    model_type: str,
    ckpt_dir: str,
    wav2vec_dir: str,
    warmup_image: Optional[str],
) -> None:
    if preload:
        logger.info(
            f"Preloading model_type={model_type} warmup={warmup} "
            f"(first launch may take several minutes with FLASHHEAD_COMPILE=1) ..."
        )
        await asyncio.to_thread(
            preload_and_warmup,
            ckpt_dir=ckpt_dir,
            model_type=model_type,
            wav2vec_dir=wav2vec_dir,
            warmup=warmup,
            warmup_image=warmup_image or None,
        )
        logger.info("Model ready — accepting WebSocket clients")
    else:
        logger.warning("Preload disabled; first client start will load the model")

    logger.info(f"FlashHead PCM WebSocket server ws://{host}:{port}/ws")
    # Long GPU + large binary frames can delay websocket keepalive pongs.
    # Default ping_timeout=20s caused 1011 "keepalive ping timeout" mid-stream.
    async with serve(
        handle_connection,
        host,
        port,
        max_size=32 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=None,
        close_timeout=10,
    ):
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="FlashHead PCM WebSocket streaming server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (0.0.0.0 = LAN reachable)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-type", default="lite", choices=["lite", "pro", "pretrained"])
    parser.add_argument("--ckpt-dir", default="models/SoulX-FlashHead-1_3B")
    parser.add_argument("--wav2vec-dir", default="models/wav2vec2-base-960h")
    parser.add_argument(
        "--warmup-image",
        default="",
        help="Optional face image for warmup; default creates a solid placeholder",
    )
    parser.add_argument("--preload", dest="preload", action="store_true", default=True)
    parser.add_argument("--no-preload", dest="preload", action="store_false")
    parser.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    args = parser.parse_args()
    asyncio.run(
        _run(
            args.host,
            args.port,
            preload=args.preload,
            warmup=args.warmup,
            model_type=args.model_type,
            ckpt_dir=args.ckpt_dir,
            wav2vec_dir=args.wav2vec_dir,
            warmup_image=args.warmup_image.strip() or None,
        )
    )


if __name__ == "__main__":
    main()
