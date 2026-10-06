"""Physics and integrity checks for hand-held / glove demonstration data.

Every finding carries a reason code and a severity:
  QUARANTINE  the episode should not train as-is (kept, never deleted)
  WARN        usable, but it changes something downstream (e.g. normalisation)
  INFO        recorded for the passport; nothing to fix
Thresholds live in CONFIG so the model team can see and argue with them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONFIG = {
    "fps": 10.0,
    "teleport_speed_mps": 3.0,        # a hand-held gripper does not move 30 cm in 0.1 s
    "flipflop_cos": -0.8,             # consecutive jumps pointing back at each other
    "rot_speed_radps": 10.0,          # ~570 deg/s sustained between frames
    "wrap_naive_rad": 1.0,            # naive axis-angle step that is really a wrap at pi
    "runaway_extent_m": 1.5,          # tabletop task: normal episodes span 0.45 m (median), 0.9 m (99.9th pct)
    "frame_offset_m": 1.0,            # episode centre this far from the dataset centre
    "gripper_range_m": (0.0, 0.1),
    "gripper_jump_m": 0.02,
    "frozen_pose_frames": 5,
    "sync_max_lag_frames": 10,
    "sync_min_peak": 0.5,             # below this the lag estimate is not trusted
    "sync_outlier_frames": 2.0,       # |lag| this large (200 ms at 10 fps) with a trusted peak
}


@dataclass
class Finding:
    code: str
    severity: str
    detail: str
    frames: list[int] = field(default_factory=list)


def rotvec_to_mat(r: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(r, axis=1, keepdims=True)
    k = np.divide(r, th, out=np.zeros_like(r), where=th > 1e-12)
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th[:, :, None]
    return np.eye(3)[None] + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def geodesic_step(R: np.ndarray) -> np.ndarray:
    """Angle (rad) of the rotation between consecutive frames; immune to axis-angle wrap."""
    rel = np.einsum("nij,nik->njk", R[:-1], R[1:])
    return np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))


def rotation_speed(state: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame rotation speed (rad/s). For a camera mounted on the hand, image motion is
    dominated by rotation, so this is the signal to align video against."""
    return np.r_[0.0, geodesic_step(rotvec_to_mat(state[:, 3:6])) * fps]


def pose_speed(state: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame motion of the hand: translation (m/s) plus rotation (rad/s, scaled to ~metres)."""
    v = np.r_[0.0, np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1) * fps]
    w = np.r_[0.0, geodesic_step(rotvec_to_mat(state[:, 3:6])) * fps]
    return v + 0.1 * w  # 0.1 m lever arm: a wrist camera sees rotation as large image motion


def episode_checks(state: np.ndarray, ts: np.ndarray, fidx: np.ndarray, dataset_centre: np.ndarray,
                   cfg: dict = CONFIG) -> tuple[list[Finding], dict]:
    fps = cfg["fps"]
    out: list[Finding] = []
    n = len(state)
    m: dict = {"frames": int(n), "duration_s": round(n / fps, 2)}

    # --- structure ---------------------------------------------------------
    dt = np.diff(ts.astype(np.float64))
    bad_dt = np.flatnonzero(np.abs(dt - 1 / fps) > 1e-3)
    if len(bad_dt):
        out.append(Finding("TIMESTAMP_GAP", "QUARANTINE", f"{len(bad_dt)} gaps or repeats in timestamps", bad_dt.tolist()[:20]))
    if np.any(np.diff(fidx) != 1):
        out.append(Finding("FRAME_INDEX_BREAK", "QUARANTINE", "frame_index is not contiguous"))
    if not np.isfinite(state).all():
        out.append(Finding("NON_FINITE", "QUARANTINE", "NaN or inf in state"))
        return out, m

    P, rv, grip = state[:, :3], state[:, 3:6], state[:, 6]

    # --- translation: teleports and flip-flops ------------------------------
    step = np.diff(P, axis=0)
    speed = np.linalg.norm(step, axis=1) * fps
    m["max_speed_mps"] = round(float(speed.max()), 3) if len(speed) else 0.0
    jumps = np.flatnonzero(speed > cfg["teleport_speed_mps"])
    if len(jumps):
        flips = 0
        for a, b in zip(jumps[:-1], jumps[1:]):
            va, vb = step[a], step[b]
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
            flips += cos < cfg["flipflop_cos"]
        code = "POSE_FLIPFLOP" if flips >= 1 else "TELEPORT"
        detail = (f"{len(jumps)} jumps over {cfg['teleport_speed_mps']} m/s (max {speed.max():.1f} m/s)"
                  + (f"; {flips} pairs jump out and back between two tracks" if flips else ""))
        out.append(Finding(code, "QUARANTINE", detail, (jumps + 1).tolist()))

    # --- workspace: slow runaways that never trip a speed check -------------
    spread = float(np.linalg.norm(P.std(0)))
    extent = float(np.linalg.norm(P.max(0) - P.min(0)))  # bounding-box diagonal: catches slow drift a std hides
    centre = np.median(P, axis=0)
    path = float(np.linalg.norm(step, axis=1).sum())
    m.update(spread_m=round(spread, 3), extent_m=round(extent, 3), path_m=round(path, 2), centre=[round(float(c), 3) for c in centre])
    if extent > cfg["runaway_extent_m"]:
        out.append(Finding("POSE_RUNAWAY", "QUARANTINE",
                           f"the hand covers a {extent:.1f} m box and travels {path:.1f} m in {n / fps:.0f} s on a tabletop task"))
    elif np.linalg.norm(centre - dataset_centre) > cfg["frame_offset_m"]:
        out.append(Finding("FRAME_OFFSET", "WARN",
                           f"motion looks normal but the episode sits {np.linalg.norm(centre - dataset_centre):.2f} m from the dataset centre; "
                           "harmless for relative actions, but it skews absolute-position normalisation"))

    # --- rotation: true speed vs representation wrap ------------------------
    ang = geodesic_step(rotvec_to_mat(rv))
    naive = np.linalg.norm(np.diff(rv, axis=0), axis=1)
    wraps = int(np.sum((naive > cfg["wrap_naive_rad"]) & (ang < 0.3)))
    m["max_rot_speed_radps"] = round(float(ang.max() * fps), 3) if len(ang) else 0.0
    if wraps:
        out.append(Finding("ROTVEC_WRAP", "INFO", f"{wraps} axis-angle wraps at pi (not real motion; geodesic speed is normal)"))
    fast_rot = np.flatnonzero(ang * fps > cfg["rot_speed_radps"])
    if len(fast_rot):
        out.append(Finding("ROT_SPIKE", "QUARANTINE", f"{len(fast_rot)} rotation steps over {cfg['rot_speed_radps']} rad/s", (fast_rot + 1).tolist()))

    # --- gripper -------------------------------------------------------------
    lo, hi = cfg["gripper_range_m"]
    if np.any((grip < lo) | (grip > hi)):
        out.append(Finding("GRIPPER_RANGE", "QUARANTINE", f"gripper width outside [{lo}, {hi}] m"))
    gj = np.flatnonzero(np.abs(np.diff(grip)) > cfg["gripper_jump_m"])
    if len(gj):
        out.append(Finding("GRIPPER_JUMP", "WARN", f"{len(gj)} gripper steps over {cfg['gripper_jump_m'] * 100:.0f} mm per frame", (gj + 1).tolist()))

    # --- frozen pose (tracker stopped updating) -----------------------------
    same = np.all(np.diff(state[:, :6], axis=0) == 0, axis=1)
    run = best = 0
    for s in same:
        run = run + 1 if s else 0
        best = max(best, run)
    if best + 1 >= cfg["frozen_pose_frames"]:
        out.append(Finding("FROZEN_POSE", "QUARANTINE", f"pose identical for {best + 1} frames"))
    return out, m


def estimate_lag(video_motion: np.ndarray, pose_motion: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Lag (frames) that best aligns video motion energy with pose speed.

    Positive lag = the video lags the pose. Normalised cross-correlation over +-max_lag frames,
    refined to sub-frame with a parabola through the peak. Returns (lag, peak_correlation).
    """
    a = np.asarray(video_motion, np.float64)
    b = np.asarray(pose_motion, np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.array([np.mean(a[max(0, k):n + min(0, k)] * b[max(0, -k):n - max(0, k)]) for k in lags])
    i = int(np.argmax(cc))
    lag = float(lags[i])
    if 0 < i < len(cc) - 1:
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            lag += 0.5 * (y0 - y2) / den
    return lag, float(cc[i])
