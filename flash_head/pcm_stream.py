"""
PCM streaming session for SoulX-FlashHead.

Model audio is always 16 kHz internally.
You may push PCM at any sample rate; this session resamples to 16 kHz.

Preferred live TTS input:
  - mono raw PCM (no WAV headers mid-stream)
  - any sample_rate (declare it when creating the session / on each push)
  - format: s16le (int16 LE) or f32le (float32 LE in [-1, 1])
"""
from __future__ import annotations

import os
import subprocess
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Union

import imageio
import librosa
import numpy as np
import torch
from loguru import logger

from flash_head.inference import (
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)


BytesLike = Union[bytes, bytearray, memoryview]
MODEL_SAMPLE_RATE = 16000

# width:height → (height, width), long side ≈ 512, aligned to 32 (Lite/Pro safe).
ASPECT_SIZE_PRESETS: dict[str, tuple[int, int]] = {
    "1:1": (512, 512),
    "3:4": (512, 384),  # portrait
    "4:3": (384, 512),  # landscape
    "9:16": (512, 288),
    "16:9": (288, 512),
}

ORIGINAL_ASPECT_KEYS = frozenset({"original", "origin", "native", "source", "原图", "原图尺寸"})


def _align_dim(value: int, align: int = 32) -> int:
    return max(align, int(round(int(value) / align) * align))


def size_from_cond_image(
    cond_image: str,
    *,
    align: int = 32,
    max_long_side: int = 1024,
) -> tuple[int, int]:
    """Read image WxH, optionally downscale, align to 32 → return (height, width)."""
    from PIL import Image

    if not cond_image or not os.path.isfile(cond_image):
        raise ValueError(f"cond_image not found for original size: {cond_image}")
    with Image.open(cond_image) as im:
        src_w, src_h = im.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid image size {src_w}x{src_h}")

    w, h = float(src_w), float(src_h)
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        w *= scale
        h *= scale
        logger.info(
            f"original size {src_w}x{src_h} exceeds max_long_side={max_long_side}, "
            f"scaled to ~{int(w)}x{int(h)}"
        )

    out_w = _align_dim(int(round(w)), align)
    out_h = _align_dim(int(round(h)), align)
    return out_h, out_w


def resolve_output_size(
    *,
    aspect_ratio: Optional[str] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    cond_image: Optional[str] = None,
    align: int = 32,
    max_long_side: int = 1024,
) -> tuple[int, int]:
    """Return (height, width) for prepare_params."""
    if height is not None and width is not None:
        h, w = int(height), int(width)
    elif aspect_ratio:
        key = str(aspect_ratio).strip().replace("：", ":")
        key_l = key.lower()
        if key in ORIGINAL_ASPECT_KEYS or key_l in ORIGINAL_ASPECT_KEYS:
            h, w = size_from_cond_image(
                cond_image or "",
                align=align,
                max_long_side=max_long_side,
            )
        elif key in ASPECT_SIZE_PRESETS:
            h, w = ASPECT_SIZE_PRESETS[key]
        else:
            raise ValueError(
                f"unsupported aspect_ratio={aspect_ratio!r}; "
                f"use one of {sorted(ASPECT_SIZE_PRESETS) + ['original']}"
            )
    else:
        h, w = 512, 512

    if h <= 0 or w <= 0:
        raise ValueError("height/width must be positive")
    if h % align != 0 or w % align != 0:
        raise ValueError(f"height/width must be divisible by {align}, got {h}x{w}")
    return h, w


# Shared pipelines across sessions (Gradio-style). Keyed by (ckpt, model_type, wav2vec).
_PIPELINE_CACHE: dict[tuple[str, str, str], object] = {}
# (height, width, model_type, sampling_steps) already warmed for generate().
_WARMED_SIZES: set[tuple[int, int, str, int]] = set()


def _pipeline_cache_key(ckpt_dir: str, model_type: str, wav2vec_dir: str) -> tuple[str, str, str]:
    return (
        os.path.abspath(ckpt_dir),
        str(model_type).strip().lower(),
        os.path.abspath(wav2vec_dir),
    )


def get_shared_pipeline(
    *,
    ckpt_dir: str = "models/SoulX-FlashHead-1_3B",
    model_type: str = "lite",
    wav2vec_dir: str = "models/wav2vec2-base-960h",
):
    key = _pipeline_cache_key(ckpt_dir, model_type, wav2vec_dir)
    pipe = _PIPELINE_CACHE.get(key)
    if pipe is not None:
        return pipe
    logger.info(
        f"Loading FlashHead pipeline into cache model_type={model_type} "
        f"ckpt={ckpt_dir} wav2vec={wav2vec_dir}"
    )
    pipe = get_pipeline(
        world_size=1,
        ckpt_dir=ckpt_dir,
        model_type=model_type,
        wav2vec_dir=wav2vec_dir,
    )
    _PIPELINE_CACHE[key] = pipe
    logger.info(f"Pipeline cached key={key[1]} ({len(_PIPELINE_CACHE)} entries)")
    return pipe


def ensure_warmup_image(path: str = "stream_uploads/_warmup_face.png") -> str:
    if os.path.isfile(path):
        return path.replace("\\", "/")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    from PIL import Image

    Image.new("RGB", (512, 512), (180, 140, 120)).save(path, format="PNG")
    logger.info(f"Created warmup face image: {path}")
    return path.replace("\\", "/")


def warmup_pipeline(
    pipeline,
    *,
    cond_image: Optional[str] = None,
    seed: int = 9999,
) -> None:
    """Run one silent chunk so torch.compile / CUDA kernels are warm before first client."""
    image = cond_image or ensure_warmup_image()
    logger.info(f"Warming up pipeline with image={image} ...")
    get_base_data(
        pipeline,
        cond_image_path_or_dir=image,
        base_seed=int(seed) if seed >= 0 else 9999,
        use_face_crop=False,
    )
    params = get_infer_params()
    fps = int(params["tgt_fps"])
    frame_num = int(params["frame_num"])
    motion_frames_num = int(params["motion_frames_num"])
    slice_len = frame_num - motion_frames_num
    chunk_samples = slice_len * MODEL_SAMPLE_RATE // fps
    cached_len = MODEL_SAMPLE_RATE * int(params["cached_audio_duration"])
    audio_end_idx = int(params["cached_audio_duration"]) * fps
    audio_start_idx = audio_end_idx - frame_num
    audio_array = np.zeros((cached_len,), dtype=np.float32)
    # One chunk of low-level noise so the audio path is non-trivial.
    if chunk_samples > 0:
        audio_array[-chunk_samples:] = (
            np.random.randn(chunk_samples).astype(np.float32) * 0.01
        )
    emb = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
    torch.cuda.synchronize()
    t0 = time.time()
    frames = run_pipeline(pipeline, emb)
    torch.cuda.synchronize()
    logger.info(
        f"Warmup inference done frames={tuple(frames.shape)} cost={time.time() - t0:.2f}s"
    )
    try:
        h = int(getattr(pipeline, "target_h", 512))
        w = int(getattr(pipeline, "target_w", 512))
        mt = str(getattr(pipeline, "model_type", "lite")).lower()
        _WARMED_SIZES.add((h, w, mt, int(params["sample_steps"])))
    except Exception:
        pass


def preload_and_warmup(
    *,
    ckpt_dir: str = "models/SoulX-FlashHead-1_3B",
    model_type: str = "lite",
    wav2vec_dir: str = "models/wav2vec2-base-960h",
    warmup: bool = True,
    warmup_image: Optional[str] = None,
) -> object:
    pipe = get_shared_pipeline(
        ckpt_dir=ckpt_dir, model_type=model_type, wav2vec_dir=wav2vec_dir
    )
    if warmup:
        warmup_pipeline(pipe, cond_image=warmup_image)
    return pipe


@dataclass
class VideoChunk:
    chunk_idx: int
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    fps: int
    elapsed_sec: float
    video_path: Optional[str] = None
    n_micro_chunks: int = 1


@dataclass
class FrameBatch:
    """Continuous stream unit: RGB frames + driving PCM on a shared timeline (pts)."""

    chunk_idx: int
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    fps: int
    pts0: float
    audio_f32: np.ndarray  # mono float32 @ audio_sample_rate
    audio_sample_rate: int
    elapsed_sec: float
    """True when this batch is the processed FlashHead reference (end settle)."""
    is_final_ref: bool = False

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def height(self) -> int:
        return int(self.frames.shape[1])

    @property
    def width(self) -> int:
        return int(self.frames.shape[2])

    def audio_s16le(self) -> bytes:
        samples = (np.clip(self.audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        return samples.tobytes()

    def encode_jpegs(
        self,
        quality: int = 80,
        *,
        max_long_side: Optional[int] = None,
    ) -> list[bytes]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python required for stream_mode=frames") from exc
        from concurrent.futures import ThreadPoolExecutor

        q = int(np.clip(quality, 40, 95))
        frames = self.frames
        h, w = int(frames.shape[1]), int(frames.shape[2])
        scale = 1.0
        if max_long_side and max(h, w) > int(max_long_side):
            scale = float(max_long_side) / float(max(h, w))

        def _one(i: int) -> bytes:
            rgb = frames[i]
            if scale < 1.0:
                nw = max(16, int(round(w * scale)) // 2 * 2)
                nh = max(16, int(round(h * scale)) // 2 * 2)
                bgr = cv2.resize(rgb[:, :, ::-1], (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                bgr = rgb[:, :, ::-1]
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise RuntimeError(f"jpeg encode failed at frame {i}")
            return buf.tobytes()

        n = int(frames.shape[0])
        workers = min(8, max(1, n))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_one, range(n)))


StreamItem = Union[VideoChunk, FrameBatch]


def pcm_bytes_to_float32(data: BytesLike, fmt: str = "s16le") -> np.ndarray:
    fmt = fmt.lower().strip()
    buf = np.frombuffer(data, dtype=np.uint8)
    if fmt in ("s16le", "int16", "pcm_s16le"):
        if len(buf) % 2 != 0:
            raise ValueError("s16le PCM length must be a multiple of 2 bytes")
        samples = np.frombuffer(memoryview(buf), dtype="<i2").astype(np.float32) / 32768.0
    elif fmt in ("f32le", "float32", "pcm_f32le"):
        if len(buf) % 4 != 0:
            raise ValueError("f32le PCM length must be a multiple of 4 bytes")
        samples = np.frombuffer(memoryview(buf), dtype="<f4").astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported PCM format: {fmt}. Use s16le or f32le.")
    return np.ascontiguousarray(samples).reshape(-1)


def _resample_to_model_rate(samples: np.ndarray, input_sr: int) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return samples
    if int(input_sr) == MODEL_SAMPLE_RATE:
        return samples
    try:
        return librosa.resample(
            samples,
            orig_sr=int(input_sr),
            target_sr=MODEL_SAMPLE_RATE,
            res_type="kaiser_fast",
        ).astype(np.float32, copy=False)
    except Exception as exc:
        # librosa default backend needs `resampy`; fall back so PCM sessions keep working.
        logger.warning(f"librosa.resample failed ({exc}); using linear fallback")
        n_out = max(1, int(round(samples.size * MODEL_SAMPLE_RATE / float(input_sr))))
        x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
        return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.shape[0] == length:
        return samples
    if samples.shape[0] > length:
        return samples[:length]
    return np.concatenate([samples, np.zeros((length - samples.shape[0],), dtype=np.float32)])


def _write_mp4(frames: np.ndarray, path: str, fps: int) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with imageio.get_writer(
        path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="libx264",
        # imageio already passes -pix_fmt yuv420p; don't duplicate it (ffmpeg warning).
        ffmpeg_params=["-bf", "0", "-movflags", "+faststart"],
    ) as writer:
        for i in range(frames.shape[0]):
            writer.append_data(frames[i])
    return path


def _save_wav(audio_f32: np.ndarray, wav_path: str, sample_rate: int = MODEL_SAMPLE_RATE) -> str:
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    samples = (np.clip(audio_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(samples.tobytes())
    return wav_path


def _write_mp4_with_audio(
    frames: np.ndarray,
    path: str,
    fps: int,
    audio_f32: np.ndarray,
    sample_rate: int = MODEL_SAMPLE_RATE,
) -> str:
    """Gradio-style: silent video temp + chunk WAV → mux AAC into final mp4."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp_video = path.replace(".mp4", "_temp.mp4")
    temp_wav = path.replace(".mp4", "_temp.wav")
    _write_mp4(frames, temp_video, fps)
    _save_wav(audio_f32, temp_wav, sample_rate=sample_rate)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video,
            "-i",
            temp_wav,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            path,
        ]
        proc = subprocess.run(cmd, check=False, capture_output=True)
        if proc.returncode != 0 or not os.path.isfile(path):
            err = proc.stderr.decode("utf-8", errors="replace")[:400]
            logger.warning(f"ffmpeg mux audio failed, falling back to silent mp4: {err}")
            if os.path.isfile(temp_video):
                os.replace(temp_video, path)
            else:
                _write_mp4(frames, path, fps)
    finally:
        for p in (temp_video, temp_wav):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    return path


class FlashHeadPCMSession:
    """
    Feed PCM at any sample rate; internally resample to 16 kHz and generate video chunks.

    Model chunk length (@16kHz):
      lite ≈ 15360 samples (~0.96s)
      pro  ≈ 17920 samples (~1.12s)
    """

    def __init__(
        self,
        cond_image: str,
        ckpt_dir: str = "models/SoulX-FlashHead-1_3B",
        wav2vec_dir: str = "models/wav2vec2-base-960h",
        model_type: str = "lite",
        seed: int = 9999,
        use_face_crop: bool = False,
        save_dir: Optional[str] = "stream_pcm_results",
        save_mp4: bool = True,
        mux_audio: bool = True,
        chunks_per_segment: int = 2,
        first_segment_chunks: int = 1,
        # Lab/short TTS: hold all micros until flush → one mp4 (no mid-play hard cuts).
        emit_on_flush_only: bool = False,
        stream_mode: str = "mp4",  # "mp4" | "frames"
        jpeg_quality: int = 80,
        sample_rate: int = 16000,
        aspect_ratio: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_long_side: int = 1024,
        sampling_steps: Optional[int] = None,
        color_correction_strength: Optional[float] = None,
    ):
        input_sr = int(sample_rate)
        if input_sr <= 0:
            raise ValueError("sample_rate must be a positive integer")

        self.model_type = model_type
        self.save_dir = save_dir
        self.save_mp4 = save_mp4
        self.mux_audio = bool(mux_audio)
        self.chunks_per_segment = max(1, int(chunks_per_segment))
        # Emit the first playable segment ASAP unless flush-only mode.
        self.first_segment_chunks = max(1, min(int(first_segment_chunks), self.chunks_per_segment))
        self.emit_on_flush_only = bool(emit_on_flush_only)
        mode = str(stream_mode or "mp4").strip().lower()
        if mode not in ("mp4", "frames"):
            raise ValueError("stream_mode must be 'mp4' or 'frames'")
        self.stream_mode = mode
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))
        self.input_sample_rate = input_sr
        self.model_sample_rate = MODEL_SAMPLE_RATE
        # backward-compatible alias: "sample_rate" means the rate YOU push
        self.sample_rate = input_sr
        self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        self.aspect_ratio = (aspect_ratio or "1:1").strip()
        self.max_long_side = max(256, int(max_long_side))
        params = get_infer_params()
        default_steps = int(params["sample_steps"])
        self.sampling_steps = int(sampling_steps) if sampling_steps is not None else default_steps
        if self.sampling_steps not in (2, 4):
            raise ValueError("sampling_steps must be 2 or 4")
        default_color = float(params["color_correction_strength"])
        self.color_correction_strength = (
            float(color_correction_strength)
            if color_correction_strength is not None
            else default_color
        )
        self.color_correction_strength = float(
            np.clip(self.color_correction_strength, 0.0, 1.0)
        )
        self.out_height, self.out_width = resolve_output_size(
            aspect_ratio=aspect_ratio,
            height=height,
            width=width,
            cond_image=cond_image,
            max_long_side=self.max_long_side,
        )

        self.pipeline = get_shared_pipeline(
            ckpt_dir=ckpt_dir,
            model_type=model_type,
            wav2vec_dir=wav2vec_dir,
        )
        get_base_data(
            self.pipeline,
            cond_image_path_or_dir=cond_image,
            base_seed=int(seed) if seed >= 0 else 9999,
            use_face_crop=use_face_crop,
            height=self.out_height,
            width=self.out_width,
            sampling_steps=self.sampling_steps,
            color_correction_strength=self.color_correction_strength,
        )

        if int(params["sample_rate"]) != MODEL_SAMPLE_RATE:
            raise ValueError(
                f"Unexpected model sample_rate in infer_params: {params['sample_rate']}"
            )

        self.fps = int(params["tgt_fps"])
        self.frame_num = int(params["frame_num"])
        self.motion_frames_num = int(params["motion_frames_num"])
        self.slice_len = self.frame_num - self.motion_frames_num
        # chunk size in MODEL sample rate (16k)
        self.chunk_samples = self.slice_len * self.model_sample_rate // self.fps
        # how many INPUT samples equal one model chunk duration
        self.input_chunk_samples = max(
            1,
            int(round(self.chunk_samples * self.input_sample_rate / self.model_sample_rate)),
        )
        self.cached_audio_duration = int(params["cached_audio_duration"])
        self.cached_audio_length_sum = self.model_sample_rate * self.cached_audio_duration
        self.audio_end_idx = self.cached_audio_duration * self.fps
        self.audio_start_idx = self.audio_end_idx - self.frame_num

        self._pcm_buf = np.zeros((0,), dtype=np.float32)  # buffered at input_sample_rate
        # Keep the model audio history as a fixed NumPy buffer.  The old deque
        # path converted every 16k PCM chunk to a Python list and then rebuilt a
        # NumPy array, creating tens of thousands of Python objects per chunk.
        self._audio_buf = np.zeros(
            (self.cached_audio_length_sum,), dtype=np.float32
        )
        self._micro_idx = 0
        self._segment_idx = 0
        self._seg_frames: List[np.ndarray] = []
        self._seg_audio: List[np.ndarray] = []
        self._seg_elapsed = 0.0
        self._timeline_pts = 0.0
        self._closed = False
        # Bumped by cancel_pending(); in-flight feed loops abandon after current micro-chunk.
        self._cancel_gen = 0
        # Exact FlashHead-processed reference (face-crop + resize/centercrop), HxWx3 RGB.
        self._ref_frame_rgb = self._capture_reference_rgb()

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        logger.info(
            f"PCM session ready id={self.session_id} model={model_type} "
            f"stream_mode={self.stream_mode} "
            f"input_sr={self.input_sample_rate} model_sr={self.model_sample_rate} "
            f"input_chunk_samples={self.input_chunk_samples} "
            f"model_chunk_samples={self.chunk_samples} "
            f"sampling_steps={self.sampling_steps} "
            f"color_correction={self.color_correction_strength:g} "
            f"(~{self.chunk_samples / self.model_sample_rate:.3f}s) fps={self.fps} "
            f"first_segment_chunks={self.first_segment_chunks} "
            f"chunks_per_segment={self.chunks_per_segment} "
            f"emit_on_flush_only={self.emit_on_flush_only} "
            f"size={self.out_height}x{self.out_width} aspect={self.aspect_ratio} "
            f"max_long_side={self.max_long_side}"
        )

    @property
    def pending_samples(self) -> int:
        return int(self._pcm_buf.shape[0])

    @property
    def samples_needed_for_next_chunk(self) -> int:
        return max(0, self.input_chunk_samples - self.pending_samples)

    def feed_pcm_bytes(
        self,
        data: BytesLike,
        fmt: str = "s16le",
        sample_rate: Optional[int] = None,
    ) -> List[StreamItem]:
        return self.feed_samples(
            pcm_bytes_to_float32(data, fmt=fmt),
            sample_rate=sample_rate,
        )

    def feed_samples(
        self,
        samples: Sequence[float],
        sample_rate: Optional[int] = None,
    ) -> List[StreamItem]:
        if self._closed:
            raise RuntimeError("Session already flushed/closed")
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return []

        start_gen = int(self._cancel_gen)
        sr = int(sample_rate) if sample_rate is not None else self.input_sample_rate
        if sr <= 0:
            raise ValueError("sample_rate must be a positive integer")
        if sr != self.input_sample_rate:
            # Convert this packet into session input rate first, so the buffer stays consistent.
            if sr != MODEL_SAMPLE_RATE and self.input_sample_rate == MODEL_SAMPLE_RATE:
                arr = _resample_to_model_rate(arr, sr)
            elif sr == MODEL_SAMPLE_RATE and self.input_sample_rate != MODEL_SAMPLE_RATE:
                arr = librosa.resample(
                    arr,
                    orig_sr=MODEL_SAMPLE_RATE,
                    target_sr=self.input_sample_rate,
                    res_type="kaiser_fast",
                ).astype(np.float32, copy=False)
            elif sr != self.input_sample_rate:
                arr = librosa.resample(
                    arr,
                    orig_sr=sr,
                    target_sr=self.input_sample_rate,
                    res_type="kaiser_fast",
                ).astype(np.float32, copy=False)

        self._pcm_buf = np.concatenate([self._pcm_buf, arr])
        out: List[StreamItem] = []
        while self._pcm_buf.shape[0] >= self.input_chunk_samples:
            if int(self._cancel_gen) != start_gen:
                self._pcm_buf = np.zeros((0,), dtype=np.float32)
                logger.info(
                    f"PCM feed aborted by cancel gen={self._cancel_gen} "
                    f"(dropped mid-buffer)"
                )
                return []
            chunk_in = self._pcm_buf[: self.input_chunk_samples]
            self._pcm_buf = self._pcm_buf[self.input_chunk_samples :]
            chunk_16k = _fit_length(
                _resample_to_model_rate(chunk_in, self.input_sample_rate),
                self.chunk_samples,
            )
            item = self._generate_micro_chunk(chunk_16k)
            if int(self._cancel_gen) != start_gen:
                logger.info(
                    f"PCM feed discarded chunk after cancel gen={self._cancel_gen}"
                )
                return []
            if item is not None:
                out.append(item)
        return out

    def cancel_pending(self) -> int:
        """
        Barge-in / interrupt: drop buffered PCM and unfinished segment state.
        Keeps the session open for the next utterance (does not set _closed).

        In-flight GPU micro-chunk may still finish (~1s); its result should be
        discarded by the caller via the returned generation id.
        """
        self._cancel_gen = int(self._cancel_gen) + 1
        dropped = int(self._pcm_buf.shape[0])
        self._pcm_buf = np.zeros((0,), dtype=np.float32)
        self._seg_frames = []
        self._seg_audio = []
        self._seg_elapsed = 0.0
        # Snap motion latents back to the reference face for the next utterance.
        try:
            person = getattr(self.pipeline, "person_name", None)
            self.pipeline.reset_person_name(person)
        except Exception:
            logger.exception("reset_person_name after cancel failed")
        logger.info(
            f"PCM cancel_pending gen={self._cancel_gen} dropped_input_samples={dropped} "
            f"session={self.session_id}"
        )
        return int(self._cancel_gen)

    def _capture_reference_rgb(self) -> Optional[np.ndarray]:
        """RGB uint8 (H,W,3) matching the tensor FlashHead actually conditions on."""
        try:
            pipe = self.pipeline
            person = getattr(pipe, "person_name", None)
            tensor = None
            if person and getattr(pipe, "cond_image_tensor_dict", None):
                tensor = pipe.cond_image_tensor_dict.get(person)
            if tensor is None:
                tensor = getattr(pipe, "original_color_reference", None)
            if tensor is None:
                logger.warning("no cond_image tensor available for reference frame")
                return None
            # (1, C, 1, H, W) in [-1, 1] → (H, W, 3) uint8 RGB
            x = tensor.detach().float().cpu()
            if x.ndim != 5:
                logger.warning(f"unexpected cond tensor shape {tuple(x.shape)}")
                return None
            x = ((x * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).byte()
            rgb = x[0, :, 0].permute(1, 2, 0).contiguous().numpy()
            logger.info(
                f"captured FlashHead reference frame {rgb.shape[1]}x{rgb.shape[0]} "
                f"person={person}"
            )
            return rgb
        except Exception:
            logger.exception("capture reference frame failed")
            return None

    def make_final_ref_batch(self, hold_sec: float = 0.28) -> Optional[FrameBatch]:
        """
        Insert the processed FlashHead reference as the last frames of the stream
        so the client settles on the true cond image (not the raw mobile avatar).
        """
        if self.stream_mode != "frames":
            return None
        if self._ref_frame_rgb is None:
            self._ref_frame_rgb = self._capture_reference_rgb()
        if self._ref_frame_rgb is None:
            return None
        try:
            person = getattr(self.pipeline, "person_name", None)
            self.pipeline.reset_person_name(person)
        except Exception:
            logger.exception("reset_person_name before final_ref failed")

        fps = max(1, int(self.fps))
        n_hold = max(1, int(round(float(hold_sec) * fps)))
        frames = np.stack([self._ref_frame_rgb] * n_hold, axis=0)
        n_audio = max(1, int(round(n_hold * self.model_sample_rate / float(fps))))
        audio = np.zeros((n_audio,), dtype=np.float32)
        pts0 = float(self._timeline_pts)
        self._timeline_pts += float(n_hold) / float(fps)
        idx = self._micro_idx
        self._micro_idx += 1
        logger.info(
            f"PCM final_ref batch-{idx} frames={n_hold} pts0={pts0:.3f}s "
            f"size={frames.shape[2]}x{frames.shape[1]}"
        )
        return FrameBatch(
            chunk_idx=idx,
            frames=frames,
            fps=fps,
            pts0=pts0,
            audio_f32=audio,
            audio_sample_rate=self.model_sample_rate,
            elapsed_sec=0.0,
            is_final_ref=True,
        )

    def flush_next(self, pad_silence: bool = False) -> Optional[StreamItem]:
        """Generate at most one remaining micro-chunk. Used by WS to emit
        mid-flush so the client can resume instead of freezing until all
        trailing PCM is generated (enhance makes that multi-second)."""
        if self._closed:
            return None
        if self._pcm_buf.shape[0] > 0 and self._pcm_buf.shape[0] < self.input_chunk_samples:
            if pad_silence:
                pad = self.input_chunk_samples - self._pcm_buf.shape[0]
                self._pcm_buf = np.concatenate(
                    [self._pcm_buf, np.zeros((pad,), dtype=np.float32)]
                )
            else:
                return None
        if self._pcm_buf.shape[0] < self.input_chunk_samples:
            return None
        chunk_in = self._pcm_buf[: self.input_chunk_samples]
        self._pcm_buf = self._pcm_buf[self.input_chunk_samples :]
        chunk_16k = _fit_length(
            _resample_to_model_rate(chunk_in, self.input_sample_rate),
            self.chunk_samples,
        )
        return self._generate_micro_chunk(chunk_16k)

    def flush(self, pad_silence: bool = True) -> List[StreamItem]:
        """Flush remaining PCM. Pads with silence to a full chunk if pad_silence=True."""
        if self._closed:
            return []
        out: List[StreamItem] = []
        first = True
        while True:
            item = self.flush_next(pad_silence=pad_silence and first)
            first = False
            if item is None:
                break
            out.append(item)

        if self.stream_mode == "mp4":
            rem = self._emit_segment()
            if rem is not None:
                out.append(rem)
        else:
            # Settle on the exact FlashHead-processed reference (cropped/resized).
            ref = self.make_final_ref_batch()
            if ref is not None:
                out.append(ref)
        self._closed = True
        return out

    def warmup_inference(self) -> bool:
        """
        Run one silent generate at this session's HxW so torch.compile finishes
        before the client starts streaming. Returns True if a warm run executed.
        """
        key = (
            int(self.out_height),
            int(self.out_width),
            str(self.model_type).lower(),
            int(self.sampling_steps),
        )
        if key in _WARMED_SIZES:
            logger.info(f"Skip warm: {key[0]}x{key[1]} already warmed")
            return False

        logger.info(
            f"Warming resolution {key[0]}x{key[1]} steps={key[3]} "
            f"(torch.compile first time at this size can take 1–2 minutes) ..."
        )
        audio_array = np.zeros((self.cached_audio_length_sum,), dtype=np.float32)
        if self.chunk_samples > 0:
            audio_array[-self.chunk_samples :] = (
                np.random.randn(self.chunk_samples).astype(np.float32) * 0.01
            )
        emb = get_audio_embedding(
            self.pipeline,
            audio_array,
            self.audio_start_idx,
            self.audio_end_idx,
        )
        torch.cuda.synchronize()
        t0 = time.time()
        _ = run_pipeline(self.pipeline, emb)
        torch.cuda.synchronize()
        # Restore motion state so real PCM starts from the reference face.
        person = getattr(self.pipeline, "person_name", None)
        self.pipeline.reset_person_name(person)
        _WARMED_SIZES.add(key)
        logger.info(
            f"Resolution warm done {key[0]}x{key[1]} cost={time.time() - t0:.2f}s"
        )
        return True

    def _emit_segment(self) -> Optional[VideoChunk]:
        """Merge buffered micro-chunks into one playable mp4 (Gradio-style)."""
        if not self._seg_frames:
            return None
        n_micro = len(self._seg_frames)
        frames = np.concatenate(self._seg_frames, axis=0)
        audio = np.concatenate(self._seg_audio, axis=0)
        elapsed = self._seg_elapsed
        self._seg_frames = []
        self._seg_audio = []
        self._seg_elapsed = 0.0

        path = None
        if self.save_mp4 and self.save_dir:
            path = os.path.join(
                self.save_dir, f"{self.session_id}_seg_{self._segment_idx:06d}.mp4"
            )
            if self.mux_audio:
                _write_mp4_with_audio(
                    frames,
                    path,
                    self.fps,
                    audio,
                    sample_rate=self.model_sample_rate,
                )
            else:
                _write_mp4(frames, path, self.fps)

        item = VideoChunk(
            chunk_idx=self._segment_idx,
            frames=frames,
            fps=self.fps,
            elapsed_sec=elapsed,
            video_path=path,
            n_micro_chunks=n_micro,
        )
        dur = frames.shape[0] / float(self.fps) if self.fps else 0.0
        logger.info(
            f"PCM segment-{self._segment_idx} done micros={n_micro} frames={frames.shape[0]} "
            f"dur={dur:.2f}s gen={elapsed:.3f}s mux_audio={self.mux_audio} path={path}"
        )
        self._segment_idx += 1
        return item

    def _generate_micro_chunk(self, chunk_pcm_16k: np.ndarray) -> Optional[StreamItem]:
        chunk_pcm_16k = np.ascontiguousarray(chunk_pcm_16k, dtype=np.float32).reshape(-1)
        history_len = int(self._audio_buf.shape[0])
        n_new = int(chunk_pcm_16k.shape[0])
        if n_new >= history_len:
            # Match deque(maxlen=...) semantics: retain the most recent samples.
            self._audio_buf[:] = chunk_pcm_16k[-history_len:]
        elif n_new > 0:
            self._audio_buf[:-n_new] = self._audio_buf[n_new:]
            self._audio_buf[-n_new:] = chunk_pcm_16k

        t_total = time.perf_counter()
        t_audio = time.perf_counter()
        audio_embedding = get_audio_embedding(
            self.pipeline,
            self._audio_buf,
            self.audio_start_idx,
            self.audio_end_idx,
        )
        torch.cuda.synchronize()
        audio_elapsed = time.perf_counter() - t_audio

        t_infer = time.perf_counter()
        video = run_pipeline(self.pipeline, audio_embedding)
        video = video[self.motion_frames_num :]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_infer

        t_download = time.perf_counter()
        frames = video.detach().cpu().numpy().astype(np.uint8)
        download_elapsed = time.perf_counter() - t_download
        total_elapsed = time.perf_counter() - t_total
        audio = np.asarray(chunk_pcm_16k, dtype=np.float32).reshape(-1)
        idx = self._micro_idx
        self._micro_idx += 1

        logger.info(
            f"PCM timing micro-{idx} audio_embed={audio_elapsed:.3f}s "
            f"gpu_infer={elapsed:.3f}s d2h={download_elapsed:.3f}s "
            f"total={total_elapsed:.3f}s"
        )

        if self.stream_mode == "frames":
            pts0 = float(self._timeline_pts)
            self._timeline_pts += float(frames.shape[0]) / float(self.fps)
            batch = FrameBatch(
                chunk_idx=idx,
                frames=frames,
                fps=self.fps,
                pts0=pts0,
                audio_f32=audio,
                audio_sample_rate=self.model_sample_rate,
                elapsed_sec=elapsed,
            )
            logger.info(
                f"PCM frame-batch-{idx} frames={frames.shape[0]} "
                f"pts0={pts0:.3f}s cost={elapsed:.3f}s"
            )
            return batch

        self._seg_frames.append(frames)
        self._seg_audio.append(audio)
        self._seg_elapsed += elapsed
        logger.info(
            f"PCM micro-{idx} frames={frames.shape[0]} "
            f"cost={elapsed:.3f}s buf={len(self._seg_frames)}/{self._segment_need()}"
        )
        if len(self._seg_frames) < self._segment_need():
            return None
        return self._emit_segment()

    def _segment_need(self) -> int:
        if self.emit_on_flush_only:
            # Never emit mid-stream; flush() will call _emit_segment().
            return 10**9
        if self._segment_idx == 0:
            return self.first_segment_chunks
        return self.chunks_per_segment
