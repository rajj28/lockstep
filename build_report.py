"""Build a single self-contained HTML report from out/ (scan + validation + evidence).

Every number on the page is read from out/*.json or recomputed from the dataset here; nothing is typed in by hand.
"""
from __future__ import annotations

import base64
import html
import json

import cv2
import numpy as np

from lockstep.checks import geodesic_step, rotvec_to_mat
from scan import load

REPO = "lerobot/umi_cup_in_the_wild"
root, df, eps, info, S = load(REPO)
E = df["episode_index"].to_numpy()
F = df["frame_index"].to_numpy()
FPS = float(info["fps"])
summary = json.load(open("out/summary.json"))
val = json.load(open("out/validation.json"))
release = json.load(open("out/release_manifest.json"))
evidence = json.load(open("out/evidence/evidence.json"))
p136 = json.load(open("out/passports/episode_0136.json"))


def img_uri(path: str, width: int = 1200, q: int = 74) -> str:
    im = cv2.imread(path)
    h, w = im.shape[:2]
    if w > width:
        im = cv2.resize(im, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------- tiny SVG chart helpers (theme-aware through CSS classes) ----------
def scale(v, a, b, c, d):
    return c + (v - a) * (d - c) / (b - a if b != a else 1)


def line_chart(series, x0, x1, y0, y1, w=760, h=240, xlabel="", ylabel="", marks=(), yticks=None, xticks=None, band=None):
    L, R, T, B = 56, 14, 14, 34
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    if band:
        by0, by1, label = band
        ya, yb = scale(by1, y0, y1, h - B, T), scale(by0, y0, y1, h - B, T)
        out.append(f'<rect x="{L}" y="{ya:.1f}" width="{w - L - R}" height="{yb - ya:.1f}" class="band"/>')
        out.append(f'<text x="{w - R - 4}" y="{ya - 5:.1f}" class="lbl" text-anchor="end">{html.escape(label)}</text>')
    for t in (yticks or []):
        y = scale(t, y0, y1, h - B, T)
        out.append(f'<line x1="{L}" x2="{w - R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/><text x="{L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>')
    for t in (xticks or []):
        x = scale(t, x0, x1, L, w - R)
        out.append(f'<text x="{x:.1f}" y="{h - B + 16}" class="tick" text-anchor="middle">{t:g}</text>')
    for xm in marks:
        x = scale(xm, x0, x1, L, w - R)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T}" y2="{h - B}" class="mark"/>')
    for xs, ys, cls in series:
        pts = " ".join(f"{scale(a, x0, x1, L, w - R):.1f},{scale(b, y0, y1, h - B, T):.1f}" for a, b in zip(xs, ys))
        out.append(f'<polyline points="{pts}" class="{cls}" fill="none"/>')
    out.append(f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" class="axis"/>')
    out.append(f'<text x="{(L + w - R) / 2:.0f}" y="{h - 4}" class="lbl" text-anchor="middle">{html.escape(xlabel)}</text>')
    out.append(f'<text x="14" y="{(T + h - B) / 2:.0f}" class="lbl" text-anchor="middle" transform="rotate(-90 14 {(T + h - B) / 2:.0f})">{html.escape(ylabel)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---------- chart 1: episode 136, two tracks interleaved ----------
m = E == 136
P136, f136 = S[m, :3], F[m]
sel = (f136 >= 60) & (f136 <= 170)
d0 = P136[0]
dist136 = np.linalg.norm(P136 - d0, axis=1)
jumps136 = p136["findings"][0]["frames"]
chart136 = line_chart([(f136[sel], dist136[sel], "s1")], 60, 170, 0, 0.6, xlabel="frame (10 per second)",
                      ylabel="hand distance from start (m)", marks=[j for j in jumps136 if 60 <= j <= 170],
                      yticks=[0, 0.2, 0.4, 0.6], xticks=[60, 80, 100, 120, 140, 160])

# ---------- chart 2: episode 125 runaway ----------
m = E == 125
P125 = S[m, :3]
t125 = np.arange(len(P125)) / FPS
ext = np.array([np.linalg.norm(S[a:b, :3].max(0) - S[a:b, :3].min(0)) for a, b in
                zip(np.r_[0, np.flatnonzero(np.diff(E)) + 1], np.r_[np.flatnonzero(np.diff(E)) + 1, len(E)])])
p999 = float(np.percentile(ext, 99.9))
d125 = np.linalg.norm(P125 - P125[0], axis=1)
chart125 = line_chart([(t125, d125, "s2")], 0, t125[-1], 0, 13, xlabel="seconds", ylabel="distance from start (m)",
                      yticks=[0, 3, 6, 9, 12], xticks=[0, 10, 20, 30], band=(0, p999, f"99.9% of episodes stay inside {p999:.1f} m"))

# ---------- chart 3: episode centres around the offset cluster ----------
starts = np.r_[0, np.flatnonzero(np.diff(E)) + 1]
ends = np.r_[starts[1:], len(E)]
cx = np.array([np.median(S[a:b, 0]) for a, b in zip(starts, ends)])
lo, hi = 340, 440
flag = set(summary["normalisation"]["episodes_flagged"])
w, h, L, R, T, B = 760, 240, 56, 14, 14, 34
dots = []
for e in range(lo, hi + 1):
    x = scale(e, lo, hi, L, w - R)
    y = scale(cx[e], -2.6, 0.8, h - B, T)
    dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 if e in flag else 3}" class="{"dotw" if e in flag else "dot"}"/>')
grid = "".join(f'<line x1="{L}" x2="{w - R}" y1="{scale(t, -2.6, 0.8, h - B, T):.1f}" y2="{scale(t, -2.6, 0.8, h - B, T):.1f}" class="grid"/>'
               f'<text x="{L - 6}" y="{scale(t, -2.6, 0.8, h - B, T) + 4:.1f}" class="tick" text-anchor="end">{t:g}</text>' for t in (-2.5, -1.5, -0.5, 0.5))
xt = "".join(f'<text x="{scale(t, lo, hi, L, w - R):.1f}" y="{h - B + 16}" class="tick" text-anchor="middle">{t}</text>' for t in range(340, 441, 20))
chart_centres = (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{grid}{xt}'
                 f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" class="axis"/>{"".join(dots)}'
                 f'<text x="{(L + w - R) / 2:.0f}" y="{h - 4}" class="lbl" text-anchor="middle">episode number</text>'
                 f'<text x="14" y="{(T + h - B) / 2:.0f}" class="lbl" text-anchor="middle" transform="rotate(-90 14 {(T + h - B) / 2:.0f})">episode centre, x (m)</text></svg>')

# ---------- chart 4: sync lag distribution ----------
lags = np.array(summary["sync"]["lags"])
trusted = lags[lags[:, 2] >= 0.5]
lag_ms = trusted[:, 1] * 1000 / FPS
bins = np.arange(-150, 151, 25)
hist, _ = np.histogram(np.clip(lag_ms, -149, 149), bins=bins)
w, h = 760, 200
bars = []
for i, c in enumerate(hist):
    x = scale(bins[i], -150, 150, L, w - R)
    x2 = scale(bins[i + 1], -150, 150, L, w - R)
    y = scale(c, 0, hist.max() * 1.1, h - B, T)
    bars.append(f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{x2 - x - 2:.1f}" height="{h - B - y:.1f}" class="bar"/>')
    if c:
        bars.append(f'<text x="{(x + x2) / 2:.1f}" y="{y - 4:.1f}" class="tick" text-anchor="middle">{c}</text>')
xt = "".join(f'<text x="{scale(t, -150, 150, L, w - R):.1f}" y="{h - B + 16}" class="tick" text-anchor="middle">{t:+d}</text>' for t in range(-150, 151, 50))
chart_sync = (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{"".join(bars)}{xt}'
              f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" class="axis"/>'
              f'<text x="{(L + w - R) / 2:.0f}" y="{h - 4}" class="lbl" text-anchor="middle">best alignment between camera motion and hand rotation (ms)</text></svg>')

# ---------- numbers ----------
norm = summary["normalisation"]
naive_wraps = 0
for a, b in zip(starts, ends):
    rv = S[a:b, 3:6]
    ang = geodesic_step(rotvec_to_mat(rv))
    naive = np.linalg.norm(np.diff(rv, axis=0), axis=1)
    naive_wraps += int(np.sum((naive > 1.0) & (ang < 0.3)))
true_spikes = 0
for a, b in zip(starts, ends):
    true_spikes += int(np.sum(geodesic_step(rotvec_to_mat(S[a:b, 3:6])) * FPS > 10))
range_all = norm["max_all"][0] - norm["min_all"][0]
range_wo = norm["max_without"][0] - norm["min_without"][0]
v = summary["verdicts"]
sy = val["sync"]
fault_rows = "".join(f"<tr><td>{html.escape(k)}</td><td class='num good'>{x['caught']} / {x['of']}</td></tr>" for k, x in val["faults"].items())
harmless_rows = "".join(f"<tr><td>{html.escape(k)}</td><td class='num good'>{x['flagged']} / {x['of']} flagged</td></tr>" for k, x in val["harmless"].items())
import re as _re
passport = json.dumps({k: p136[k] for k in ("episode", "verdict", "content_hash")} | {"metrics": {k: p136["metrics"][k] for k in ("frames", "max_speed_mps", "extent_m", "sync_lag_ms", "sync_peak")}, "findings": p136["findings"]}, indent=1)
passport = _re.sub(r"\[\s*([\d,\s]+?)\s*\]", lambda mm: "[" + ", ".join(x.strip() for x in mm.group(1).split(",")) + "]", passport)
e136 = evidence["136"]
e144 = evidence["144"]
e125 = evidence["125"]
q_eps = ", ".join(str(q["episode"]) for q in release["quarantined"])


# ---------- facts computed here so the copy never outruns the data ----------
from lockstep.checks import estimate_lag, rotation_speed
key = next(k for k, vv in info["features"].items() if vv.get("dtype") == "video")
meta_frames_ok = int(sum(int(round((float(r[f"videos/{key}/to_timestamp"]) - float(r[f"videos/{key}/from_timestamp"])) * FPS)) == int(r["length"]) for _, r in eps.iterrows()))
checked_files = summary["video_files_checked"]
checked_eps = eps[eps[f"videos/{key}/file_index"].isin(checked_files)]
corr_rot, corr_tr, flow_meds = [], [], []
for fidx in checked_files:
    z = np.load(f"{root}/videos/{key}/chunk-000/file-{fidx:03d}.mp4.signals.npz")
    for e, r in checked_eps[checked_eps[f"videos/{key}/file_index"] == fidx].iterrows():
        a, b = starts[e], ends[e]
        f0 = int(round(float(r[f"videos/{key}/from_timestamp"]) * FPS))
        fl = z["flow"][f0:f0 + (b - a)]
        if len(fl) != b - a or b - a <= 60:
            continue
        st = S[a:b]
        tr = np.r_[0.0, np.linalg.norm(np.diff(st[:, :3], axis=0), axis=1) * FPS]
        corr_rot.append(estimate_lag(fl[1:], rotation_speed(st, FPS)[1:], 10)[1])
        corr_tr.append(estimate_lag(fl[1:], tr[1:], 10)[1])
        flow_meds.append(float(np.median(fl)))
end_dist = float(np.linalg.norm(P125[-1] - P125[0]))
drop = float(P125[0, 2] - P125[-1, 2])
dc = np.median(np.array([np.median(S[a:b, :3], axis=0) for a, b in zip(starts, ends)]), axis=0)
off = [float(np.linalg.norm(np.median(S[starts[e]:ends[e], :3], axis=0) - dc)) for e in norm["episodes_flagged"] if e != 125]

ctx = dict(
    episodes=f"{summary['episodes']:,}", hours=summary["hours"], frames=f"{summary['frames']:,}",
    quarantined=v.get("QUARANTINE", 0), warned=v.get("PASS_WITH_WARNINGS", 0), passed=f"{v.get('PASS', 0):,}",
    j136=e136["pose_step_m"][3] * 100, flow136=e136["video_flow_px"][3], flowmed136=e136["flow_median_px"],
    j144=e144["pose_step_m"][3] * 100, flow144=e144["video_flow_px"][3],
    path125=round(float(np.linalg.norm(np.diff(P125, axis=0), axis=1).sum()), 1), ext125=round(float(np.linalg.norm(P125.max(0) - P125.min(0))), 1),
    speed125=e125["pose_speed_mps_median"], flow125=e125["video_flow_median"], p999=round(p999, 2),
    std_all=f'{norm["std_all"][0]:.2f}', std_wo=f'{norm["std_without"][0]:.2f}', range_all=round(range_all, 2), range_wo=round(range_wo, 2),
    frac=round(100 * norm["frames_flagged"] / norm["frames_total"], 2), squeeze=round(100 * range_wo / range_all),
    naive_wraps=naive_wraps, true_spikes=true_spikes,
    sync_n=len(trusted), sync_med=round(float(np.median(lag_ms)), 1), sync_p5=round(float(np.percentile(lag_ms, 5)), 1), sync_p95=round(float(np.percentile(lag_ms, 95)), 1),
    sync_trials=f"{sy['trials']:,}", sync_mae_ms=round(sy["mean_abs_error_frames"] * 1000 / FPS, 1), sync_half=round(100 * sy["within_half_frame"], 1),
    fv_c=val["frozen_video"]["caught"], fv_n=val["frozen_video"]["of"], release_hash=release["release_hash"], n_inc=f"{len(release['included']):,}",
    q_eps=q_eps, files=", ".join(str(x) for x in summary["video_files_checked"]),
    meta_ok=f"{meta_frames_ok:,}", n_video_eps=len(checked_eps), pct_video=round(100 * len(checked_eps) / len(eps)),
    corr_rot=round(float(np.median(corr_rot)), 2), corr_tr=round(float(np.median(corr_tr)), 2),
    flow_typ=round(float(np.median(flow_meds)), 2), end_dist=round(end_dist, 1), drop=round(drop, 1),
    off_lo=round(min(off), 1), off_hi=round(max(off), 1),
)

TEMPLATE = open("report_template.html", encoding="utf-8").read()
page = TEMPLATE
for k, val_ in ctx.items():
    page = page.replace("{{" + k + "}}", str(val_))
page = (page.replace("{{chart136}}", chart136).replace("{{chart125}}", chart125).replace("{{chart_centres}}", chart_centres)
            .replace("{{chart_sync}}", chart_sync).replace("{{fault_rows}}", fault_rows).replace("{{harmless_rows}}", harmless_rows)
            .replace("{{passport}}", html.escape(passport))
            .replace("{{img136}}", img_uri("out/evidence/ep136_strip.jpg")).replace("{{img125}}", img_uri("out/evidence/ep125_strip.jpg"))
            .replace("{{img_session}}", img_uri("out/evidence/session_strip.jpg", 1400)).replace("{{img_where}}", img_uri("out/evidence/patio_check.jpg", 1400)))
assert "{{" not in page, page[page.index("{{"):page.index("{{") + 40]
open("out/report.html", "w", encoding="utf-8").write(page)
print("wrote out/report.html", len(page) // 1024, "KB")
