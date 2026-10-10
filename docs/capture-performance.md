# Capture and preview performance changes

Implemented scope: camera throughput and preview/UI overhead (items 1 and 4
from the fast-ball investigation). HSV gates, streak shape limits, tracking
association and bounce algorithms in `app/serve_caller.py` are unchanged.

## Capture

- Probe advertised legacy Camera preview modes on rear cameras; apply size,
  NV21 format, recording hint and FPS range together before starting preview.
- Verify driver readback and try alternatives if a mode is rejected. Prefer a
  stable rate near 60 FPS, then 1280x720 within the same FPS tier. This may select
  another resolution/camera when that is necessary for higher FPS.
- Poll up to 60 times per second. Native callback sequence and arrival time
  distinguish new frames even when their pixels are identical. These are
  callback arrival timestamps, not sensor exposure timestamps.
- Drain the SurfaceTexture at 60Hz but skip Kivy's unused FBO drawing; decoded
  NV21 supplies the application preview. Sensor callback delivery is separate
  from the slower application preview uploads.
- Preserve automatic exposure/white balance and delayed neutral exposure
  compensation. No portable manual shutter setting is added to legacy Camera.
  A fixed higher frame rate can reduce brightness in dim halls.

60 FPS is conditional on legacy NV21 preview support and sufficient device
throughput. A phone may expose 60 FPS only through another capture API or its
native video recorder. Supported flexible ranges can still deliver fewer than
60 FPS; logging shows the configured range and observed callback rate separately.

## Detection and UI

- A worker owns its detection motion reference, tracker and caller. UI actions
  configure independent snapshots rather than mutating worker-owned objects.
- One frame can be running and one waiting. New submissions replace the waiting
  frame, avoiding an unbounded latency backlog. Replaced pending frames are
  counted. Capture callbacks skipped by UI polling can be assessed separately
  using callback FPS versus submitted FPS.
- A configuration generation discards in-flight results from an old serve,
  calibration, pause or parameter set. Predicted positions still cannot prove
  a bounce. The existing 200ms continuity reset is retained.
- Consume completed results every UI tick, including cached-camera ticks.
  Speak/show a verdict immediately and force its preview update.
- Live preview runs at 5 FPS using elapsed time, independent of capture rate.
  Setup/idle preview runs at 15 FPS. START SERVE disables ASSIST/MASK; an explicit
  debug toggle can enable them again.
- RGB upload uses a writable contiguous buffer view, and a newly created
  texture is flipped once through its coordinates. Each upload avoids the
  previous CPU image flip and `.tobytes()` copy. GL remains on Kivy's UI thread.

## Validation

Run `python -m unittest discover -s tests -v`. Checks include real worker
processing at 30/60 FPS, pending-frame replacement, stale-serve rejection,
configuration ownership, error recovery, camera-mode fallback/readback,
fresh identical callback frames, unused-FBO omission, texture orientation/reuse,
5 FPS preview at both capture rates, and verdict delivery on a cached frame.
Existing generated-frame and trajectory regressions also remain applicable.

A Windows host microbenchmark (Python 3.13, OpenCV 4.12, one OpenCV thread;
1280x720 uint8 frames; 20 warm-ups and 200 timed calls) measured:

| Preview preparation only | Median | p95 |
| --- | ---: | ---: |
| Image flip + RGB conversion + bytes copy | 1.891 ms | 2.257 ms |
| RGB conversion + buffer view | 0.093 ms | 0.119 ms |

This isolates preparation, excluding actual GPU upload, camera delivery,
detection and Kivy scheduling. It is not a whole-app or Android speedup result.
Live upload frequency is independently reduced from roughly 15 to 5 per second
(67% fewer uploads), and ordinary serves no longer run ASSIST's duplicate
analysis. Increasing capture from 30 to 60 also increases detection work per
second; compare latency, delivered FPS and thermal behavior rather than CPU
utilization alone.

## Device check

No Android device or newly built APK was available for host validation. Build
and run on the target phone, calibrate again if the selected camera/resolution
changes, then serve with ASSIST off. In logcat inspect:

- `selected camera`, `size`, `configured-fps`: driver configuration/readback.
- `callback FPS`: newly delivered callbacks per elapsed second.
- `submitted FPS` / `processed FPS`: handoff and worker throughput.
- `dropped`: pending submissions replaced while the worker was busy.
- `ui avg` / `detector`: UI callback cost and latest detector processing time.

Compare bright/dim halls and the same labelled serves. Confirm preview remains
upright, taps align with corners, IN speaks once, pause/resume does not reuse an
old trajectory, and the device sustains its rate after warming. If callback FPS
stays near 30, the implementation must not be described as delivering 60 FPS.

Primary API references: [Kivy Android camera provider](https://raw.githubusercontent.com/kivy/kivy/2.3.1/kivy/core/camera/camera_android.py),
[Kivy texture buffers and coordinate flips](https://kivy.org/doc/stable/api-kivy.graphics.texture.html),
[Android legacy camera parameters](https://developer.android.com/reference/android/hardware/Camera.Parameters).
