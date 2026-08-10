"""Quick Real-ESRGAN enhance bench (24 frames @ 512)."""
import time
import numpy as np
from flash_head.frame_enhance import EnhanceConfig, FrameEnhancer

cfg = EnhanceConfig(enabled=True, backend="ncnn", model="animevideov3-x2", out_long_side=1024)
e = FrameEnhancer(cfg)
print("backend", e.backend, e.info())
frames = np.random.randint(0, 255, (24, 512, 512, 3), dtype=np.uint8)
# warmup
e.enhance_frames(frames[:2])
t0 = time.time()
out = e.enhance_frames(frames)
dt = time.time() - t0
print("shape", out.shape, "cost", round(dt, 3), "s", "fps", round(24 / dt, 1))
