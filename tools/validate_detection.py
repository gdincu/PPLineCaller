"""Reproduce before/after metrics on generated frames, not recorded footage.

Run: python tools/validate_detection.py [--baseline 66d7bbb]
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_detection import BASE, POLY, mapper, nv21, scene
import serve_caller as current


def measure(module, new):
    trajectories = [[220, 250, 280, 260], [220, 280, 280, 260],
                    [270, 272, 274, 273, 271]]
    hits = []
    for ys in trajectories:
        points = []
        first = None
        for i, y in enumerate(ys):
            points.append((420, y))
            if module.is_bounce(points) and first is None:
                first = i
        hits.append(first)
    calibration = module.AutoTuneCollector(POLY, BASE)
    empty = scene()
    for _ in range(45):
        calibration.add(empty)
    tuned, _ = calibration.suggest()
    settings = dict(BASE, **tuned)
    pipeline = []
    for ys in trajectories + [[220, 250, None, 280, 260]]:
        table = module.TableMapper()
        table.set_corners_quadrant_order(*POLY)
        caller, track = module.ServeCaller(table), module.BallTracker()
        caller.reset()
        previous, first = None, None
        for i, y in enumerate(ys):
            frame = empty.copy()
            if y is not None:
                cv2.circle(frame, (420, y), 7, (230, 230, 230), -1)
            det, previous = module.detect_ball_nv21(nv21(frame), 640, 480, track.pos,
                table_poly=POLY, prev_small=previous, **settings)
            pos = track.update(det)
            verdict = (caller.update(pos if track.observed else None, i / 30, track.track_id)
                       if new else caller.update(pos))
            if verdict == 'IN' and first is None:
                first = i
        pipeline.append(first)
    times = []
    previous = None
    for i in range(100):
        frame = empty.copy()
        cv2.circle(frame, (420, 250 + i % 10), 10, (230, 230, 230), -1)
        yuv = nv21(frame)
        start = time.perf_counter()
        _, previous = module.detect_ball_nv21(yuv, 640, 480, (420, 250),
            table_poly=POLY, prev_small=previous, **settings)
        elapsed = (time.perf_counter() - start) * 1000
        if i >= 10:
            times.append(elapsed)
    return dict(trajectory_first_bounce_frame=hits,
                generated_first_IN_frame=pipeline, auto_tune=tuned,
                detector_ms_p50=round(float(np.median(times)), 2),
                detector_ms_p95=round(float(np.percentile(times, 95)), 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='66d7bbb')
    args = parser.parse_args()
    source = subprocess.check_output(['git', 'show', f'{args.baseline}:app/serve_caller.py'],
                                     cwd=ROOT, encoding='utf-8')
    baseline = types.ModuleType('baseline_serve_caller')
    exec(compile(source, f'{args.baseline}:app/serve_caller.py', 'exec'), baseline.__dict__)
    print(json.dumps(dict(baseline_ref=args.baseline, opencv=cv2.__version__,
                          numpy=np.__version__, frame_size='640x480, 30 FPS timestamps',
                          case_order=['sharp', 'plateau', 'shallow', 'occlusion'],
                          baseline=measure(baseline, False),
                          improved=measure(current, True)), indent=2))


if __name__ == '__main__':
    main()
