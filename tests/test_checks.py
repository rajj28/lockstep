import numpy as np

from lockstep.checks import episode_checks, estimate_lag, geodesic_step, rotvec_to_mat

FPS = 10.0
RNG = np.random.default_rng(0)


def smooth_episode(n=300):
    """A plausible hand-held trajectory: slow smooth motion, gentle rotation, open-ish gripper."""
    t = np.arange(n) / FPS
    pos = np.c_[0.1 * np.sin(0.4 * t), 0.08 * np.cos(0.3 * t), 0.1 + 0.03 * np.sin(0.5 * t)]
    rot = np.c_[2.9 + 0.1 * np.sin(0.2 * t), 0.05 * np.cos(0.3 * t), 0.02 * t / t[-1]]
    grip = np.full((n, 1), 0.07)
    ts = t.astype(np.float32)
    return np.c_[pos, rot, grip], ts, np.arange(n)


def codes(state, ts, fi):
    return {f.code for f in episode_checks(state, ts, fi, np.zeros(3))[0]}


def test_clean_episode_passes():
    assert not {c for c in codes(*smooth_episode()) if c not in ("ROTVEC_WRAP",)}


def test_axis_angle_wrap_is_not_motion():
    r = np.array([[np.pi - 0.01, 0, 0]])
    r_equiv = r - 2 * np.pi * r / np.linalg.norm(r)  # same rotation, written the other way round
    R = rotvec_to_mat(np.vstack([r, r_equiv]))
    assert geodesic_step(R)[0] < 1e-6
    assert np.linalg.norm(r - r_equiv) > 6  # a naive subtraction sees a huge "spike"


def test_flipflop_is_quarantined():
    st, ts, fi = smooth_episode()
    st[100:105, :3] += [0.4, 0, 0]
    assert "POSE_FLIPFLOP" in codes(st, ts, fi)


def test_slow_drift_is_caught_without_any_fast_frame():
    st, ts, fi = smooth_episode()
    st[:, :3] += np.linspace(0, 3.0, len(st))[:, None] * np.array([1.0, 0, 0])
    out = codes(st, ts, fi)
    assert "POSE_RUNAWAY" in out and "TELEPORT" not in out


def test_small_shift_is_not_flagged():
    st, ts, fi = smooth_episode()
    st[:, :3] += [0.3, 0, 0]
    assert not {c for c in codes(st, ts, fi) if c != "ROTVEC_WRAP"}


def test_dropped_rows_break_timestamps():
    st, ts, fi = smooth_episode()
    keep = np.r_[0:50, 53:len(st)]
    out = codes(st[keep], ts[keep], fi[keep])
    assert {"TIMESTAMP_GAP", "FRAME_INDEX_BREAK"} <= out


def test_lag_estimator_recovers_planted_shift():
    base = np.convolve(RNG.random(400), np.ones(5) / 5, mode="same")
    for k in (-4, -1, 2, 5):
        shifted = np.roll(base, k)
        lag, peak = estimate_lag(shifted, base, 10)
        assert abs(lag - k) < 0.3 and peak > 0.8
