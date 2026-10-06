"""Scan a LeRobot v3 dataset: per-episode passports, quarantine list, release manifest, planted-fault validation.

usage: python scan.py lerobot/umi_cup_in_the_wild
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from lockstep.checks import CONFIG, Finding, episode_checks, estimate_lag, rotation_speed
from lockstep.video import decode_signals

PIPELINE_VERSION = "lockstep-0.1.0"
RNG = np.random.default_rng(20261006)


def load(repo: str):
    root = os.path.join("data", repo.replace("/", "__"))
    df = pd.read_parquet(os.path.join(root, "data/chunk-000/file-000.parquet"))
    eps = pd.read_parquet(os.path.join(root, "meta/episodes/chunk-000/file-000.parquet"))
    info = json.load(open(os.path.join(root, "meta/info.json")))
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    return root, df, eps, info, state


def episode_hash(state: np.ndarray, ts: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(state.astype(np.float32)).tobytes())
    h.update(np.ascontiguousarray(ts.astype(np.float32)).tobytes())
    return h.hexdigest()[:16]


def video_signals_for_file(root: str, key: str, file_index: int, expected_frames: int):
    """Decode (or load cached) per-frame signals. Refuses files shorter than the episodes need,
    so a half-downloaded video can never masquerade as a data problem."""
    path = os.path.join(root, f"videos/{key}/chunk-000/file-{file_index:03d}.mp4")
    if not os.path.exists(path):
        return None, path
    cache = path + ".signals.npz"
    from lockstep.video import FrameSignals
    if os.path.exists(cache):
        z = np.load(cache)
        sig = FrameSignals(z["luma"], z["motion"], z["dhash"], z["sharp"], z["flow"])
    else:
        t0 = time.time()
        sig = decode_signals(path)
        np.savez(cache, luma=sig.luma, motion=sig.motion, dhash=sig.dhash, sharp=sig.sharp, flow=sig.flow)
        print(f"  decoded {path}: {len(sig.luma)} frames in {time.time() - t0:.1f}s", flush=True)
    if len(sig.luma) < expected_frames:
        print(f"  SKIP {path}: {len(sig.luma)} frames < {expected_frames} needed (incomplete download?)", flush=True)
        return None, path
    return sig, path


def video_checks(sig, n_expected: int, meta_frames: int) -> tuple[list[Finding], dict]:
    out, m = [], {"video_frames": int(len(sig.luma))}
    if meta_frames != n_expected:
        out.append(Finding("VIDEO_FRAME_COUNT", "QUARANTINE", f"video segment spans {meta_frames} frames, signals have {n_expected}"))
    same = np.r_[False, sig.dhash[1:] == sig.dhash[:-1]] & (sig.motion < 0.5)
    run = best = 0
    for s in same:
        run = run + 1 if s else 0
        best = max(best, run)
    m["longest_frozen_run"] = int(best + 1 if best else 0)
    if best >= 2:
        out.append(Finding("FROZEN_VIDEO", "QUARANTINE", f"{best + 1} identical frames in a row"))
    dark = int(np.sum(sig.luma < 12))
    if dark:
        out.append(Finding("DARK_FRAMES", "WARN", f"{dark} near-black frames (lens covered or exposure failure)"))
    return out, m


def main(repo: str) -> None:
    t0 = time.time()
    root, df, eps, info, state = load(repo)
    fps = float(info["fps"])
    E, T, FI = df["episode_index"].to_numpy(), df["timestamp"].to_numpy(), df["frame_index"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(E)) + 1]
    ends = np.r_[starts[1:], len(E)]
    centres = np.array([np.median(state[a:b, :3], axis=0) for a, b in zip(starts, ends)])
    dataset_centre = np.median(centres, axis=0)
    key = next(k for k, v in info["features"].items() if v.get("dtype") == "video")

    files: dict[int, tuple] = {}
    fcol = f"videos/{key}/file_index"
    need = {int(f): int(round(float(g[f"videos/{key}/to_timestamp"].max()) * fps)) for f, g in eps.groupby(fcol)}
    passports, codes = [], Counter()
    lag_rows = []
    for e, (a, b) in enumerate(zip(starts, ends)):
        st, ts, fi = state[a:b], T[a:b], FI[a:b]
        findings, metrics = episode_checks(st, ts, fi, dataset_centre)
        row = eps.iloc[e]
        fidx = int(row[f"videos/{key}/file_index"])
        if fidx not in files:
            files[fidx] = video_signals_for_file(root, key, fidx, need[fidx])
        sig, _ = files[fidx]
        if sig is not None:
            f0 = int(round(float(row[f"videos/{key}/from_timestamp"]) * fps))
            f1 = int(round(float(row[f"videos/{key}/to_timestamp"]) * fps))
            esig = sig.slice(f0, f0 + (b - a))
            vf, vm = video_checks(esig, b - a, f1 - f0)
            findings += vf
            metrics.update(vm)
            if len(esig.flow) == b - a and b - a > 60:
                lag, peak = estimate_lag(esig.flow[1:], rotation_speed(st, fps)[1:], CONFIG["sync_max_lag_frames"])
                metrics["sync_lag_ms"], metrics["sync_peak"] = round(lag / fps * 1000, 1), round(peak, 3)
                lag_rows.append((e, lag, peak))
                if peak >= CONFIG["sync_min_peak"] and abs(lag) >= CONFIG["sync_outlier_frames"]:
                    findings.append(Finding("SYNC_OUTLIER", "WARN", f"camera motion and hand rotation line up best at {lag / fps * 1000:+.0f} ms (r={peak:.2f}); needs a look"))
        sev = {f.severity for f in findings}
        verdict = "QUARANTINE" if "QUARANTINE" in sev else ("PASS_WITH_WARNINGS" if "WARN" in sev else "PASS")
        for f in findings:
            codes[f.code] += 1
        passports.append({
            "episode": e, "verdict": verdict, "content_hash": episode_hash(st, ts),
            "video_file": fidx, "video_checked": sig is not None,
            "metrics": metrics,
            "findings": [{"code": f.code, "severity": f.severity, "detail": f.detail, "frames": f.frames[:30]} for f in findings],
        })

    os.makedirs("out/passports", exist_ok=True)
    for p in passports:
        json.dump(p, open(f"out/passports/episode_{p['episode']:04d}.json", "w"), indent=1)
    verdicts = Counter(p["verdict"] for p in passports)
    release = {
        "dataset": repo, "pipeline_version": PIPELINE_VERSION, "config": CONFIG,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "episodes_in": len(passports),
        "included": [{"episode": p["episode"], "hash": p["content_hash"], "verdict": p["verdict"]} for p in passports if p["verdict"] != "QUARANTINE"],
        "quarantined": [{"episode": p["episode"], "hash": p["content_hash"], "codes": sorted({f["code"] for f in p["findings"] if f["severity"] == "QUARANTINE"})} for p in passports if p["verdict"] == "QUARANTINE"],
    }
    release["release_hash"] = hashlib.sha256(json.dumps([r["hash"] for r in release["included"]]).encode()).hexdigest()[:16]
    json.dump(release, open("out/release_manifest.json", "w"), indent=1)

    # normalisation impact of quarantined + offset episodes
    flagged = {p["episode"] for p in passports if any(f["code"] in ("POSE_RUNAWAY", "FRAME_OFFSET") for f in p["findings"])}
    keep = ~np.isin(E, list(flagged))
    norm = {
        "std_all": state[:, :3].std(0).round(4).tolist(), "std_without": state[keep, :3].std(0).round(4).tolist(),
        "min_all": state[:, :3].min(0).round(3).tolist(), "max_all": state[:, :3].max(0).round(3).tolist(),
        "min_without": state[keep, :3].min(0).round(3).tolist(), "max_without": state[keep, :3].max(0).round(3).tolist(),
        "frames_flagged": int((~keep).sum()), "frames_total": int(len(keep)), "episodes_flagged": sorted(flagged),
    }
    summary = {"repo": repo, "fps": fps, "episodes": len(passports), "frames": int(len(E)), "hours": round(len(E) / fps / 3600, 2),
               "verdicts": dict(verdicts), "codes": dict(codes), "normalisation": norm,
               "video_files_checked": sorted(k for k, v in files.items() if v[0] is not None),
               "sync": {"n": len(lag_rows), "lags": [[e, round(l, 3), round(p, 3)] for e, l, p in lag_rows]},
               "runtime_s": round(time.time() - t0, 1)}
    json.dump(summary, open("out/summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("episodes", "frames", "hours", "verdicts", "codes", "video_files_checked", "runtime_s")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lerobot/umi_cup_in_the_wild")
