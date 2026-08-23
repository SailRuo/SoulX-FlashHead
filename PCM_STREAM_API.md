# FlashHead PCM 流式接口文档（WebSocket 实时）

将 TTS / 实时语音以 **PCM** 经 **WebSocket** 推入，服务端按块回推数字人视频（meta JSON + mp4 二进制）。

相关文件：

| 文件 | 说明 |
|------|------|
| `pcm_ws_server.py` | **WebSocket 主服务（推荐）** |
| `start_pcm_server.bat` | 一键启动 WS |
| `flash_head/pcm_stream.py` | 核心会话 / 重采样 / 推理 |
| `pcm_stream_server.py` | 旧版 HTTP REST（可选备用） |

默认地址：`ws://127.0.0.1:8765/ws`（连接 `/` 也可）

---

## 1. 启动

```bash
start_pcm_server.bat

# 或
conda activate E:\conda_envs\flashhead
cd E:\Project\SoulX-FlashHead
set CUDA_VISIBLE_DEVICES=0
set FLASHHEAD_COMPILE=1
python pcm_ws_server.py --host 127.0.0.1 --port 8765
```

依赖：`websockets`（已在 flashhead 环境安装）。

---

## 2. 音频格式

| 项目 | 说明 |
|------|------|
| 传输 | WebSocket **binary** 帧 = 裸 PCM（不要 WAV 头） |
| 声道 | mono |
| 采样率 | 任意；`start` 里声明 `sample_rate`，服务端自动转到模型 **16 kHz** |
| 格式 | `s16le`（推荐）或 `f32le` |

一块视频时长（模型侧 16 kHz）：

- lite ≈ **0.96 s**（15360 samples @16k）
- pro ≈ **1.12 s**

---

## 3. 协议总览（一条连接 = 一个会话）

```
Client                         Server
  |--- text: upload_image ----->|
  |<-- text: upload_image_ack --|
  |--- binary: image bytes ---->|
  |<-- text: image_ready -------|
  |--- text: start ------------>|
  |<-- text: loading -----------|
  |<-- text: ready -------------|
  |--- binary: PCM ... -------->|
  |<-- text: video (meta) ------|
  |<-- binary: mp4 -------------|
  |--- text: flush ------------>|
  |<-- text/binary video(s) ----|
  |<-- text: done --------------|
```

也可跳过上传，直接 `start.cond_image` 填服务端已有路径（如 `examples/girl.png`）。

控制消息一律 **UTF-8 JSON 文本**；图片 / PCM / mp4 一律 **binary**。

---

## 4. 消息定义

### 4.0 Client → Server：上传人像（推荐，guiv1 用这个）

**文本**

```json
{ "type": "upload_image", "filename": "face.png" }
```

Server：

```json
{
  "type": "upload_image_ack",
  "filename": "face.png",
  "message": "send next WebSocket binary frame as image bytes"
}
```

**下一帧必须是 binary**：整张图片文件（png/jpg/jpeg/webp/bmp，≤20MB）。

Server 校验后保存到 `stream_uploads/`，回复：

```json
{
  "type": "image_ready",
  "cond_image": "stream_uploads/20260810-xxxx_face.png",
  "bytes": 123456
}
```

然后 `start` 可省略 `cond_image`（自动用刚上传的路径），或显式传入该路径。

约束：`upload_image` 必须在 `start` **之前**；`start` 之后需重连才能再换图。

---

### 4.1 Client → Server：`start`（文本）

```json
{
  "type": "start",
  "cond_image": "stream_uploads/xxx.png",
  "model_type": "lite",
  "sample_rate": 24000,
  "fmt": "s16le",
  "stream_mode": "frames",
  "jpeg_quality": 85,
  "stream_max_long_side": 0,
  "enhance": true,
  "enhance_backend": "auto",
  "enhance_model": "animevideov3-x2",
  "enhance_out_long_side": 1024,
  "send_mp4_binary": true,
  "ckpt_dir": "models/SoulX-FlashHead-1_3B",
  "wav2vec_dir": "models/wav2vec2-base-960h",
  "seed": 9999,
  "use_face_crop": false,
  "save_dir": "stream_pcm_results",
  "save_mp4": true,
  "sampling_steps": 4,
  "color_correction_strength": 1.0
}
```

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `cond_image` | 条件必填 | — | 服务端路径；本连接已 `image_ready` 时可省略 |
| `model_type` | 否 | `lite` | `lite` / `pro` |
| `sample_rate` | 否 | `16000` | **你推送 PCM 的采样率** |
| `fmt` | 否 | `s16le` | `s16le` / `f32le` |
| `stream_mode` | 否 | `mp4` | `mp4`（短片）或 `frames`（连续 JPEG+PCM，无硬切） |
| `jpeg_quality` | 否 | `82` | 仅 `frames`：JPEG 质量 40–95（开 `enhance` 时建议 ≥85） |
| `stream_max_long_side` | 否 | `512` | 预览最长边；`0`/`null`=不下采样。开 `enhance` 且未显式传时自动关掉 |
| `enhance` | 否 | `false` | 是否对每帧做 Real-ESRGAN 清晰化（JPEG 编码前） |
| `enhance_backend` | 否 | `auto` | `auto` → 优先 `ncnn`(Vulkan，不抢 CUDA) → `opencv` 兜底 |
| `enhance_model` | 否 | `animevideov3-x2` | 实时：`animevideov3-x2` / `x4fast`；画质：`x4plus` |
| `enhance_out_long_side` | 否 | `1024` | 超分后目标最长边（512→1024≈2×） |
| `send_mp4_binary` | 否 | `true` | 仅 `mp4`：meta 后是否紧跟 mp4 二进制 |
| `sampling_steps` | 否 | `4` | `4` 为默认画质；`2` 为极速档，约减半扩散去噪时间，但口型/细节稳定性会下降 |
| `color_correction_strength` | 否 | `1.0` | 0–1；设为 `0` 可关闭色彩校正并略降延迟 |
| 其它 | 否 | 见上 | 权重路径、种子等 |

### 画质增强（Real-ESRGAN，推荐 lite + 超分）

Lite 模型原生约 512，JPEG 再压会显得糊；Pro 又太慢。可在 **FlashHead 出帧之后、JPEG 之前** 串 Real-ESRGAN：

```json
{
  "type": "start",
  "stream_mode": "frames",
  "enhance": true,
  "enhance_backend": "auto",
  "enhance_model": "animevideov3-x2",
  "enhance_out_long_side": 1024,
  "jpeg_quality": 85,
  "stream_max_long_side": 0
}
```

- **依赖**：`pip install realesrgan-ncnn-py`（Vulkan，与 FlashHead CUDA 并行，几乎不占 CUDA 显存）
- **实测**（RTX 4080S）：24 帧 512→1024，`animevideov3-x2` ≈ **47 FPS**，可与下一块 GPU 推理重叠
- 未装 ncnn 时自动退化为 OpenCV Lanczos+锐化（仍比裸 512 JPEG 清晰）
- `ready` 会回传：`enhance` / `enhance_backend` / `enhance_model` / `enhance_out_long_side`

### 结束定格：`final_ref`（自动）

`stream_mode=frames` 时，`flush` 与 `cancel` 会在末尾再插入一块 **FlashHead 内部已裁剪/缩放的参考图**（不是手机原始头像）：

```json
{
  "type": "frame_batch",
  "final_ref": true,
  "n_frames": 7,
  "pts0": 3.84,
  "...": "..."
}
```

用途：播放结束或打断后，画面准确回到数字人参考脸。客户端收到后应定格该帧。

随后 Server：

1. `{"type":"loading","message":"loading model..."}`  
2. `{"type":"ready", ..., "cond_image":"...", "fmt":"s16le", "stream_mode":"frames"}`

`ready` 里常见字段：`session_id`、`cond_image`、`input_sample_rate`、`model_sample_rate`、`chunk_seconds`、`input_chunk_samples`、`samples_needed_for_next_chunk`、`stream_mode`。

---

### 4.2 Client → Server：PCM（binary）

在 `ready` 之后，持续发送 **binary** 帧，内容为原始 PCM 字节。

每凑满一块，Server 按 `stream_mode` 推送：

#### `stream_mode: "mp4"`（默认）

**文本 meta**

```json
{
  "type": "video",
  "chunk_idx": 0,
  "n_frames": 24,
  "height": 512,
  "width": 512,
  "fps": 25,
  "elapsed_sec": 0.41,
  "video_path": "stream_pcm_results/....mp4",
  "byte_length": 123456
}
```

**紧接着 binary**：长度为 `byte_length` 的 mp4 文件字节（当 `send_mp4_binary=true`）。

客户端必须按「先收 text video，再收下一帧 binary mp4」解析。

#### `stream_mode: "frames"`（连续流，推荐实时预览）

**文本 meta**

```json
{
  "type": "frame_batch",
  "chunk_idx": 0,
  "n_frames": 24,
  "pts0": 0.0,
  "fps": 25,
  "height": 1024,
  "width": 1024,
  "audio_sample_rate": 16000,
  "audio_fmt": "s16le",
  "elapsed_sec": 0.41,
  "byte_length": 123456,
  "video_codec": "h264"
}
```

`video_codec`：`mp4`（H.264 装进短 MP4，推荐手机端）或 `jpeg`（兼容）。客户端 `start.frame_video_codec=h264` 时服务端输出 `mp4`；不可用则回退 `jpeg`。长边 wire 上限约 **1920**（超分 2× 后若更大会压到 1920，避免部分 Android 硬件解码黑屏）。

**紧接着 binary**（大端 `uint32` 长度）：

**`video_codec=jpeg`**

```
u32be audio_len | s16le PCM @ 16kHz | u32be n_frames | (u32be jpeg_len | jpeg)*N
```

**`video_codec=mp4`（由 frame_video_codec=h264 产生）**

```
u32be audio_len | s16le PCM @ 16kHz | u32be mp4_len | fragmented MP4 (H.264, no audio)
```

- 每批独立可播的短 MP4（`frag_keyframe+empty_moov`）；客户端用 `<video>` / `createImageBitmap` 抽帧，再按 `pts0` 播放
- `pts0`：本批第 0 帧在会话时间轴上的秒数；第 `i` 帧 pts = `pts0 + i/fps`
- 音频与视频共用该时间轴，客户端可用 canvas + Web Audio 连续播放

`start` 相关字段：

| 字段 | 说明 |
|------|------|
| `frame_video_codec` / `video_codec` | `h264` \| `jpeg`，默认 `jpeg`（App 数字人默认发 `h264` → 实际 wire=`mp4`） |
| `h264_crf` / `video_crf` | 16–40，默认 26（x264 CRF / NVENC CQ） |
| `FLASHHEAD_H264_ENCODER` | 环境变量：`nvenc` 强制 `h264_nvenc`，否则优先 `libx264` ultrafast |

- 若客户端此前发过 `subtitle`，meta 里会带上当前字幕：

```json
{
  "type": "frame_batch",
  "chunk_idx": 0,
  "pts0": 0.0,
  "duration_sec": 0.96,
  "subtitle": "接着聊张居正",
  "subtitle_id": "seg-2",
  "...": "..."
}
```

可选还会跟一条：

```json
{ "type": "status", "pending_samples": 0, "samples_needed_for_next_chunk": 23040, ... }
```

---

### 4.2.1 Client → Server：`subtitle`（文本，可选）

FlashHead 只吃 PCM，**不认字**。字幕由客户端在推 PCM 前/中下发，服务端原样盖到后续 `frame_batch` / `video` meta 上，方便前端与口型时间轴对齐。

```json
{ "type": "subtitle", "text": "接着聊张居正，还是换个话题？", "id": "seg-2" }
```

- `text`：当前要显示的字幕；空字符串表示清空
- `id`：可选，句段 id（便于前端去重）

Server 立刻回：

```json
{ "type": "subtitle_ack", "subtitle": "接着聊张居正，还是换个话题？", "subtitle_id": "seg-2" }
```

之后每一块视频 meta 都会带上最新的 `subtitle` / `subtitle_id`，直到再次 `subtitle` 更新或 `flush`/`start` 清空。

推荐时机：TTS 每一句开始推 PCM 前发一次；整段结束 `flush` 前可再发 `{"type":"subtitle","text":""}`。

---

### 4.2.2 Client → Server：`cancel` / `interrupt`（文本）

用户打断（说话 barge-in / 点击打断）时立刻发，**不要等 flush**：

```json
{ "type": "cancel" }
```

效果：

- 清空未消费 PCM 缓冲与未发出的 segment
- 取消排队中的 `frame_batch` / `video` 发送
- **不关闭** WebSocket，也不卸载模型；下一句可继续推 PCM
- 正在跑的一块 GPU 推理最多再跑完约 1 秒后丢弃结果（CUDA 无法硬中断）

Server 回：

```json
{ "type": "cancelled", "cancel_gen": 3, ... }
```

---

### 4.3 Client → Server：`flush`（文本）

```json
{ "type": "flush", "pad_silence": true }
```

冲刷尾部；可能再产生若干 `video`，最后：

```json
{ "type": "done", "closed": true, ... }
```

之后本连接不能再推 PCM（可再发 `close` 或断开）。

---

### 4.4 其它

| 方向 | 消息 |
|------|------|
| C→S | `{"type":"ping"}` |
| S→C | `{"type":"pong"}` |
| C→S | `{"type":"close"}` |
| S→C | `{"type":"bye"}` |
| S→C | `{"type":"error","error":"..."}` |

---

## 5. Python 客户端示例

```python
import asyncio
import json
import websockets

async def main():
    uri = "ws://127.0.0.1:8765/ws"
    async with websockets.connect(uri, max_size=32 * 1024 * 1024) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "cond_image": "examples/girl.png",
            "model_type": "lite",
            "sample_rate": 24000,
            "fmt": "s16le",
            "send_mp4_binary": True,
        }))

        # wait ready
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                continue
            data = json.loads(msg)
            print("<<", data.get("type"), data)
            if data.get("type") == "ready":
                break
            if data.get("type") == "error":
                raise RuntimeError(data["error"])

        # push PCM from your TTS (example: read a raw s16le file in chunks)
        with open("tts_24k_s16le.pcm", "rb") as f:
            while True:
                pcm = f.read(4096)
                if not pcm:
                    break
                await ws.send(pcm)
                # drain any immediate responses without blocking forever
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    except asyncio.TimeoutError:
                        break
                    await handle_server_msg(ws, msg)

        await ws.send(json.dumps({"type": "flush", "pad_silence": True}))
        while True:
            msg = await ws.recv()
            done = await handle_server_msg(ws, msg)
            if done:
                break

async def handle_server_msg(ws, msg):
    if isinstance(msg, bytes):
        # previous video meta told us byte_length; here just save
        with open("out_chunk.mp4", "ab") as f:
            # better: save per-chunk with your own counter after meta
            pass
        return False

    data = json.loads(msg)
    t = data.get("type")
    print("<<", t)
    if t == "video":
        mp4 = await ws.recv()  # next frame must be binary mp4
        assert isinstance(mp4, bytes)
        path = f"recv_chunk_{data['chunk_idx']:06d}.mp4"
        with open(path, "wb") as f:
            f.write(mp4)
        print("saved", path, "frames", data["n_frames"])
    if t == "done":
        return True
    if t == "error":
        raise RuntimeError(data.get("error"))
    return False

asyncio.run(main())
```

更稳妥的收包方式：看到 `type=="video"` 后**立刻** `recv()` 下一帧作为 mp4（不要和 PCM 发送抢乱序逻辑）。

---

## 6. 与 Gradio / 旧 HTTP 的区别

| | Gradio 流式 | HTTP REST（旧） | **WebSocket（本接口）** |
|--|------------|----------------|------------------------|
| 音频输入 | 整文件 | POST 分块 | **binary 实时推** |
| 视频输出 | 攒 3 chunk 再播 | JSON 里给 path | **meta + mp4 即时推** |
| 连接 | 网页 | 短请求 | **长连接双工** |

---

## 7. 注意

1. 首包可能慢（模型加载 + `torch.compile` 预热）；之后按块接近实时（建议 `lite`）。  
2. 单连接单会话；PCM 顺序发送即可。  
3. 帧大小上限服务端设为 **32 MiB**（避免默认 1 MiB 装不下 mp4）。  
4. 输出默认 512×512。  
5. 旧 HTTP 接口仍在 `pcm_stream_server.py`，需要时单独启动（注意端口勿冲突）。
