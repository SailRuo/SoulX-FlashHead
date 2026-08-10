"""
HTTP API: stream PCM in, get video chunks out.

Audio requirements (preferred):
  - raw PCM body, NOT wav container mid-stream
  - mono
  - any sample rate (declare sample_rate on session create; optional override per push)
  - model always runs at 16 kHz internally (auto-resample)
  - s16le (int16 LE) default, or f32le (float32 LE in [-1,1])

Examples:
  # create session (TTS at 24000 Hz)
  curl -X POST http://127.0.0.1:8765/v1/sessions -H "Content-Type: application/json" -d "{
    \"cond_image\": \"examples/girl.png\",
    \"model_type\": \"lite\",
    \"sample_rate\": 24000
  }"

  # push PCM (s16le @ session sample_rate, or override with &sample_rate=48000)
  curl -X POST "http://127.0.0.1:8765/v1/sessions/<id>/pcm?fmt=s16le" \\
    -H "Content-Type: application/octet-stream" --data-binary @chunk.pcm

  # flush tail
  curl -X POST http://127.0.0.1:8765/v1/sessions/<id>/flush
"""
from __future__ import annotations

import argparse
import base64
import os
import threading
from typing import Dict, Optional

from flask import Flask, jsonify, request
from loguru import logger

from flash_head.pcm_stream import FlashHeadPCMSession, VideoChunk

app = Flask(__name__)
_lock = threading.Lock()
_sessions: Dict[str, FlashHeadPCMSession] = {}
_session_locks: Dict[str, threading.Lock] = {}


def _get_session_lock(session_id: str) -> threading.Lock:
    with _lock:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def _chunk_to_dict(chunk: VideoChunk, include_b64: bool = False) -> dict:
    item = {
        "chunk_idx": chunk.chunk_idx,
        "n_frames": int(chunk.frames.shape[0]),
        "height": int(chunk.frames.shape[1]),
        "width": int(chunk.frames.shape[2]),
        "fps": chunk.fps,
        "elapsed_sec": chunk.elapsed_sec,
        "video_path": chunk.video_path,
    }
    if include_b64 and chunk.video_path and os.path.exists(chunk.video_path):
        with open(chunk.video_path, "rb") as f:
            item["video_mp4_base64"] = base64.b64encode(f.read()).decode("ascii")
    return item


def _session_info(session: FlashHeadPCMSession) -> dict:
    return {
        "session_id": session.session_id,
        "model_type": session.model_type,
        "sample_rate": session.input_sample_rate,
        "input_sample_rate": session.input_sample_rate,
        "model_sample_rate": session.model_sample_rate,
        "fps": session.fps,
        "chunk_samples": session.chunk_samples,
        "input_chunk_samples": session.input_chunk_samples,
        "chunk_seconds": session.chunk_samples / session.model_sample_rate,
        "pending_samples": session.pending_samples,
        "samples_needed_for_next_chunk": session.samples_needed_for_next_chunk,
        "closed": session._closed,
    }


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/v1/sessions")
def create_session():
    body = request.get_json(force=True, silent=True) or {}
    cond_image = body.get("cond_image")
    if not cond_image:
        return jsonify({"error": "cond_image is required"}), 400

    kwargs = {
        "cond_image": cond_image,
        "ckpt_dir": body.get("ckpt_dir", "models/SoulX-FlashHead-1_3B"),
        "wav2vec_dir": body.get("wav2vec_dir", "models/wav2vec2-base-960h"),
        "model_type": body.get("model_type", "lite"),
        "seed": int(body.get("seed", 9999)),
        "use_face_crop": bool(body.get("use_face_crop", False)),
        "save_dir": body.get("save_dir", "stream_pcm_results"),
        "save_mp4": bool(body.get("save_mp4", True)),
        "sample_rate": int(body.get("sample_rate", 16000)),
    }

    try:
        session = FlashHeadPCMSession(**kwargs)
    except Exception as e:
        logger.exception("Failed to create PCM session")
        return jsonify({"error": str(e)}), 500

    with _lock:
        _sessions[session.session_id] = session
        _session_locks[session.session_id] = threading.Lock()
    return jsonify(_session_info(session))


@app.get("/v1/sessions/<session_id>")
def get_session(session_id: str):
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify(_session_info(session))


@app.post("/v1/sessions/<session_id>/pcm")
def push_pcm(session_id: str):
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    if session._closed:
        return jsonify({"error": "session already flushed/closed"}), 400

    fmt = request.args.get("fmt", request.headers.get("X-PCM-Format", "s16le"))
    include_b64 = request.args.get("include_b64", "0") in ("1", "true", "True")
    sr_arg = request.args.get("sample_rate", request.headers.get("X-Sample-Rate"))
    sample_rate = int(sr_arg) if sr_arg else None
    data = request.get_data(cache=False)
    if not data:
        return jsonify({"error": "empty body; send raw PCM bytes"}), 400

    try:
        with _get_session_lock(session_id):
            chunks = session.feed_pcm_bytes(data, fmt=fmt, sample_rate=sample_rate)
    except Exception as e:
        logger.exception("feed_pcm failed")
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            **_session_info(session),
            "produced": [_chunk_to_dict(c, include_b64=include_b64) for c in chunks],
        }
    )


@app.post("/v1/sessions/<session_id>/flush")
def flush_session(session_id: str):
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        return jsonify({"error": "session not found"}), 404

    include_b64 = request.args.get("include_b64", "0") in ("1", "true", "True")
    pad_silence = request.args.get("pad_silence", "1") in ("1", "true", "True")
    try:
        with _get_session_lock(session_id):
            chunks = session.flush(pad_silence=pad_silence)
    except Exception as e:
        logger.exception("flush failed")
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            **_session_info(session),
            "produced": [_chunk_to_dict(c, include_b64=include_b64) for c in chunks],
        }
    )


@app.delete("/v1/sessions/<session_id>")
def delete_session(session_id: str):
    with _lock:
        session = _sessions.pop(session_id, None)
        _session_locks.pop(session_id, None)
    if session is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"ok": True, "session_id": session_id})


def main():
    parser = argparse.ArgumentParser(description="FlashHead PCM streaming server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logger.info(f"PCM stream server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
