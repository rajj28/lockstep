"""Check the checker: plant known faults into clean episodes and measure what the checks catch,
then apply harmless changes and measure what they wrongly flag.

usage: python validate.py lerobot/umi_cup_in_the_wild   (run scan.py first)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from lockstep.checks import CONFIG, episode_checks, estimate_lag, rotation_speed
from scan import load

RNG = np.random.default_rng(7)
N = 200  # clean episodes per test


def quarantine_codes(findings) -> set[str]:
    return {f.code for f in findings if f.severity == "QUARANTINE"}


def main(repo: str) -> None:
    root, df, eps, info, state = load(repo)
    fps = float(info["fps"])
    E, T, FI = df["episode_index"].to_numpy(), df["timestamp"].to_numpy().astype(np.float64), df["frame_index"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(E)) + 1]
    ends = np.r_[starts[1:], len(E)]
    centres = np.array([np.median(state[a:b, :3], axis=0) for a, b in zip(starts, ends)])
    dataset_centre = np.median(centres, axis=0)
    passports = {int(f.split("_")[1][:4]): json.load(open(os.path.join("out/passports", f))) for f in os.listdir("out/passports")}
    clean = [e for e, p in passports.items() if p["verdict"] == "PASS"]
    pick = RNG.choice(clean, size=min(N, len(clean)), replace=False)

    def run(st, ts, fi):
        return quarantine_codes(episode_checks(st, ts, fi, dataset_centre)[0])

    faults = {
        "teleport (0.4 m for 5 frames)": ("POSE_FLIPFLOP", "TELEPORT"),
        "slow drift (3 m over the episode)": ("POSE_RUNAWAY",),
        "frozen pose (8 frames)": ("FROZEN_POSE",),
        "dropped rows (3 frames)": ("TIMESTAMP_GAP", "FRAME_INDEX_BREAK"),
        "NaN in one value": ("NON_FINITE",),
    }
    harmless = ["same rotation, other axis-angle form", "sensor noise (1 mm, 0.2 deg)", "whole episode shifted 0.3 m"]
    caught = {k: 0 for k in faults}
    false_alarm = {k: 0 for k in harmless}

    for e in pick:
        a, b = starts[e], ends[e]
        st0, ts0, fi0 = state[a:b].copy(), T[a:b].copy(), FI[a:b].copy()
        n = len(st0)
        base = run(st0, ts0, fi0)
        assert not base, (e, base)

        st = st0.copy(); i = RNG.integers(5, n - 10); d = RNG.normal(size=3); d = 0.4 * d / np.linalg.norm(d)
        st[i:i + 5, :3] += d
        caught["teleport (0.4 m for 5 frames)"] += bool(run(st, ts0, fi0) & set(faults["teleport (0.4 m for 5 frames)"]))

        st = st0.copy(); d = RNG.normal(size=3); d = d / np.linalg.norm(d)
        st[:, :3] += np.linspace(0, 3.0, n)[:, None] * d
        caught["slow drift (3 m over the episode)"] += bool(run(st, ts0, fi0) & set(faults["slow drift (3 m over the episode)"]))

        st = st0.copy(); i = RNG.integers(5, n - 10); st[i:i + 8] = st[i]
        caught["frozen pose (8 frames)"] += bool(run(st, ts0, fi0) & set(faults["frozen pose (8 frames)"]))

        i = RNG.integers(5, n - 10); keep = np.r_[0:i, i + 3:n]
        caught["dropped rows (3 frames)"] += bool(run(st0[keep], ts0[keep], fi0[keep]) & set(faults["dropped rows (3 frames)"]))

        st = st0.copy(); st[RNG.integers(0, n), RNG.integers(0, 7)] = np.nan
        caught["NaN in one value"] += bool(run(st, ts0, fi0) & set(faults["NaN in one value"]))

        st = st0.copy(); r = st[:, 3:6]; th = np.linalg.norm(r, axis=1, keepdims=True)
        st[:, 3:6] = r - 2 * np.pi * r / np.maximum(th, 1e-9)  # identical rotations, different numbers
        false_alarm["same rotation, other axis-angle form"] += bool(run(st, ts0, fi0))

        st = st0.copy(); st[:, :3] += RNG.normal(0, 0.001, (n, 3)); st[:, 3:6] += RNG.normal(0, np.radians(0.2), (n, 3))
        false_alarm["sensor noise (1 mm, 0.2 deg)"] += bool(run(st, ts0, fi0))

        st = st0.copy(); d = RNG.normal(size=3); st[:, :3] += 0.3 * d / np.linalg.norm(d)
        false_alarm["whole episode shifted 0.3 m"] += bool(run(st, ts0, fi0))

    # --- sync: plant known offsets between video and pose, see if the estimator recovers them ---
    from scan import video_signals_for_file
    key = next(k for k, v in info["features"].items() if v.get("dtype") == "video")
    fcol = f"videos/{key}/file_index"
    need = {int(f): int(round(float(g[f"videos/{key}/to_timestamp"].max()) * fps)) for f, g in eps.groupby(fcol)}
    sync_err, sync_n, shifts = [], 0, [-5, -3, -2, -1, 1, 2, 3, 5]
    frozen_caught = frozen_n = 0
    for fidx in sorted(need):
        sig, _ = video_signals_for_file(root, key, fidx, need[fidx])
        if sig is None:
            continue
        for e in eps.index[eps[fcol] == fidx]:
            p = passports.get(int(e))
            if not p or p["verdict"] != "PASS" or p["metrics"].get("sync_peak", 0) < CONFIG["sync_min_peak"]:
                continue
            a, b = starts[e], ends[e]
            f0 = int(round(float(eps.loc[e, f"videos/{key}/from_timestamp"]) * fps))
            flow = sig.flow[f0:f0 + (b - a)]
            w = rotation_speed(state[a:b], fps)
            base, _ = estimate_lag(flow[1:], w[1:], 10)
            for k in shifts:
                wk = np.roll(w, k)  # k > 0: pose arrives k frames late
                if k > 0: wk[:k] = w[0]
                else: wk[k:] = w[-1]
                est, _ = estimate_lag(flow[1:], wk[1:], 10)
                sync_err.append((base - est) - k)
            sync_n += 1
            # frozen video: repeat one frame 4 times inside the episode
            h, mo = sig.dhash[f0:f0 + (b - a)].copy(), sig.motion[f0:f0 + (b - a)].copy()
            i = RNG.integers(5, len(h) - 10)
            h[i:i + 4] = h[i]; mo[i + 1:i + 4] = 0.0
            same = np.r_[False, h[1:] == h[:-1]] & (mo < 0.5)
            run_, best = 0, 0
            for s in same:
                run_ = run_ + 1 if s else 0; best = max(best, run_)
            frozen_caught += best >= 2; frozen_n += 1

    err = np.abs(np.array(sync_err))
    out = {
        "episodes_tested": int(len(pick)),
        "faults": {k: {"caught": v, "of": int(len(pick)), "recall": round(v / len(pick), 3)} for k, v in caught.items()},
        "harmless": {k: {"flagged": v, "of": int(len(pick)), "false_alarm_rate": round(v / len(pick), 3)} for k, v in false_alarm.items()},
        "frozen_video": {"caught": int(frozen_caught), "of": int(frozen_n)},
        "sync": {"episodes": sync_n, "trials": int(len(err)), "shifts_frames": shifts,
                 "mean_abs_error_frames": round(float(err.mean()), 3) if len(err) else None,
                 "within_half_frame": round(float(np.mean(err <= 0.5)), 3) if len(err) else None,
                 "within_one_frame": round(float(np.mean(err <= 1.0)), 3) if len(err) else None},
    }
    json.dump(out, open("out/validation.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lerobot/umi_cup_in_the_wild")
