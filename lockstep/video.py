"""Frame-level checks on LeRobot v3 video files, decoded with ffmpeg (AV1 via libdav1d).

Per frame we keep a few cheap, explainable signals:
  luma   mean brightness (black / blown-out frames)
  motion mean absolute difference to the previous frame (frozen frames, motion energy)
  dhash  64-bit difference hash (exact / near-exact repeats)
  sharp  variance of the Laplacian (motion blur, lens covered)
  flow   mean optical-flow magnitude to the previous frame (camera ego-motion, used for sync)
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 112  # decode at half resolution: enough for these signals, 4x cheaper


@dataclass
class FrameSignals:
    luma: np.ndarray    # float32 [n]
    motion: np.ndarray  # float32 [n] (frame 0 = 0)
    dhash: np.ndarray   # uint64  [n]
    sharp: np.ndarray   # float32 [n]
    flow: np.ndarray    # float32 [n] (frame 0 = 0)

    def slice(self, a: int, b: int) -> "FrameSignals":
        return FrameSignals(self.luma[a:b], self.motion[a:b], self.dhash[a:b], self.sharp[a:b], self.flow[a:b])


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH")
    return exe


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def decode_signals(path: str, start_s: float = 0.0, duration_s: float | None = None) -> FrameSignals:
    """Stream grey frames out of ffmpeg and reduce each one to four numbers."""
    cmd = [_ffmpeg(), "-v", "error", "-ss", f"{start_s:.3f}", "-i", path]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-vf", f"scale={SIZE}:{SIZE}:flags=area", "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = SIZE * SIZE
    luma, motion, hashes, sharp, flow = [], [], [], [], []
    prev = prev_u8 = None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        g = np.frombuffer(buf, np.uint8).reshape(SIZE, SIZE)
        gf = g.astype(np.float32)
        luma.append(float(gf.mean()))
        motion.append(0.0 if prev is None else float(np.abs(gf - prev).mean()))
        hashes.append(_dhash(g))
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        if prev_u8 is None:
            flow.append(0.0)
        else:
            f = cv2.calcOpticalFlowFarneback(prev_u8, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow.append(float(np.linalg.norm(f, axis=2).mean()))
        prev, prev_u8 = gf, g
    proc.wait()
    if proc.returncode not in (0, None):
        err = proc.stderr.read().decode(errors="replace")[:400]
        raise RuntimeError(f"ffmpeg failed: {err}")
    return FrameSignals(np.array(luma, np.float32), np.array(motion, np.float32),
                        np.array(hashes, np.uint64), np.array(sharp, np.float32), np.array(flow, np.float32))


def frame_at(path: str, t_s: float, size: int = 224) -> np.ndarray:
    """One RGB frame at time t (for evidence thumbnails)."""
    cmd = [_ffmpeg(), "-v", "error", "-ss", f"{t_s:.3f}", "-i", path, "-frames:v", "1",
           "-vf", f"scale={size}:{size}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out[: size * size * 3], np.uint8).reshape(size, size, 3)
