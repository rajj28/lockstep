# Lockstep

QA for human demonstration data: catch the episodes that pass every structural check but are physically wrong, before a model trains on them.

I ran it over the public **UMI "cup in the wild"** dataset ([lerobot/umi_cup_in_the_wild](https://huggingface.co/datasets/lerobot/umi_cup_in_the_wild)): 1,447 episodes, 19.4 hours of people doing a task with a hand-held gripper and a wrist camera.

## What it found

| | |
|---|---|
| Structure (timestamps, frame indices, missing values, video index) | clean for all 1,447 episodes |
| **Quarantined** (kept, never deleted) | **7 episodes**: 4 where the pose alternates between two tracks ~37 cm apart while the camera barely moves, 2 teleports, 1 runaway that drifts 12 m away from a kitchen table |
| **Passed with a warning** | **15 episodes** whose motion is normal but whose origin moved by 1.6 to 2.9 m in the same room. They are 1.5% of frames, yet they inflate the std of x by 2.2× and its range from 1.73 m to 11.7 m, which is what mean-std (ACT) and min-max (Diffusion Policy) normalisation see |
| Not quarantined on purpose | 149 "rotation spikes" that a naive check would flag. They are axis-angle vectors wrapping at ±π; the true rotation between frames never exceeds 10 rad/s |
| Where | the 5 jump episodes I could view were all recorded outdoors (a sunny patio and a sidewalk café) |
| Camera–hand sync | optical flow aligned against hand **rotation** (r = 0.70) rather than translation (r = 0.23); every decoded episode lines up within 100 ms |

**Read the report (charts and the actual frames): https://rajj28.github.io/lockstep/**

## Checking the checker

Faults planted into 200 clean episodes, and harmless changes applied to the same 200:

| Planted fault | Caught |
|---|---|
| teleport (0.4 m for 5 frames) | 200 / 200 |
| slow drift (3 m over the episode) | 200 / 200 |
| frozen pose (8 frames) | 200 / 200 |
| dropped rows (3 frames) | 200 / 200 |
| NaN in one value | 200 / 200 |
| frozen video (one frame repeated 4 times) | 271 / 271 |

| Harmless change | Wrongly flagged |
|---|---|
| same rotation, other axis-angle form | 0 / 200 |
| sensor noise (1 mm, 0.2°) | 0 / 200 |
| whole episode shifted 0.3 m | 0 / 200 |

Sync: planted shifts of ±1, 2, 3 and 5 frames between video and pose, 2,168 trials, recovered with a mean error of 4.2 ms.

The first validation run caught a gap in my own checker: slow drift was found in only 1 of 200 episodes, because a standard deviation hides a smooth ramp. The runaway check now uses the episode's extent, with a threshold taken from the data (99.9% of episodes stay inside a 0.9 m box; the limit is 1.5 m).

## How it works

```
episode ─► structure ─► physics ─► video ─► camera–hand sync ─► passport ─► quarantine or release
```

- `lockstep/checks.py`: physics and integrity checks. Every finding has a reason code and a severity (`QUARANTINE`, `WARN`, `INFO`). Thresholds live in one `CONFIG` dict.
- `lockstep/video.py`: decodes LeRobot v3 AV1 video with ffmpeg and keeps five numbers per frame: brightness, frame difference, a 64-bit difference hash, sharpness and optical-flow magnitude.
- `scan.py`: runs everything, writes one passport per episode (`out/passports/`), `out/release_manifest.json` (included and quarantined episodes with content hashes, plus a release hash) and `out/summary.json`.
- `validate.py`: the planted-fault and harmless-change tests above.
- `build_report.py`: renders `out/report.html`. Every number on it is read from `out/` or recomputed from the dataset.

Design choices:
- **Quarantine, never delete.** The model team decides what to train on; the passport says why an episode was held back.
- **Rotation is compared as rotation.** Geodesic angle between consecutive rotation matrices, not differences of axis-angle numbers.
- **Sync against what the camera actually sees.** A wrist camera's image motion is dominated by the hand turning, so optical flow is aligned with rotation speed. On a glove, the gyroscope measures exactly that.
- **Whole-episode checks as well as per-frame ones.** A smooth runaway never trips a speed limit.

## Run it

Needs Python 3.11 and ffmpeg (with an AV1 decoder) on PATH.

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or .venv/bin/pip
python hfget.py lerobot/umi_cup_in_the_wild meta/info.json meta/stats.json meta/episodes/chunk-000/file-000.parquet data/chunk-000/file-000.parquet
python hfget.py lerobot/umi_cup_in_the_wild videos/observation.image/chunk-000/file-000.mp4   # optional: any of the 15 video files
python scan.py && python validate.py && python build_report.py
python -m pytest -q tests
```

`hfget.py` retries and resumes ranged downloads, because the network I built this on dropped most connections to Hugging Face.

The first scan decodes each video once (about 3 minutes per 200 MB file on a laptop CPU); after that a full re-scan of all 1,447 episodes takes about 4 seconds.

## Limits

- One public dataset at 10 fps. Video was decoded for 3 of its 15 files (323 episodes); the pose and metadata checks ran on all 1,447.
- The likely causes (tracking relocalisation outdoors, a moving map origin) are hypotheses read from the data and frames, not confirmed with the dataset's authors. The findings are about this LeRobot copy of the dataset.
- Thresholds suit a tabletop task with a hand-held gripper. A new task or device needs its own thresholds, taken from its own data the same way.
- There are no force or IMU channels in this dataset, so those checks are planned, not demonstrated.

## Credit

Data: UMI (Chi et al., *Universal Manipulation Interface*, RSS 2024), as published by LeRobot under Apache-2.0. Code: MIT.

Built by Ruturaj Sonkamble, Pune.
