# Ping-Pong Serve Line Caller

An automated, real-time Ping-Pong doubles serve line-caller. Built using Python, Kivy, OpenCV, and NumPy, this project provides CPU-only, zero-training ball detection and homography-based court mapping to adjudicate doubles serves according to ITTF rules.

Runs only as a native Android app (APK) with real-time Text-to-Speech (TTS) announcements for instant calls (`IN` vs `FAULT`).

This project is directly inspired by and based on [clssmitty/PBLineCaller](https://github.com/clssmitty/PBLineCaller/), adapting court-mapping and trajectory bounce-detection principles specifically for table tennis doubles serve rules.

---

## Features

* **Android-only, CPU-Only Performance:** Optimized using classic computer vision (HSV thresholding, circularity filtering, and frame deltas). No heavy GPU requirements, PyTorch, or cloud AI (Roboflow) needed.
* **4-Point Calibration, positional taps:** Stand at your end on your right-hand edge and tap 1. near-right, 2. near-left, 3. far-left, 4. far-right (same order on both phones). Full-table mapping is kept (best homography conditioning) even though each phone only judges its own quarter.
* **One phone per quarter, nothing to configure:** Because both phones tap positionally, each phone's own quarter always lands on the same warp slot — no mode or quarter setting exists. Your quarter is drawn thick with a `MY` tag.
* **Bounce & Trajectory Analysis:** Tracks vertical velocity inversions ($\Delta y_1 > 0 \rightarrow \Delta y_2 < 0$) to accurately locate bounce points on the table.
* **ITTF Doubles Rule Verification:** Validates that doubles serve bounce 1 occurs in the Server's Right half and bounce 2 lands diagonally in the Receiver's Right half (ITTF 2.6.3). Centre line counts as IN.

---

## How It Works

### 1. Camera Setup

One phone per judged quarter, landscape, on a tripod at your end of the table on your right-hand edge, each framing the whole table. The two phones end up diagonal to each other (each judges its own near-right quarter).

```text
        [ RECEIVER SIDE ]
  +------------+------------+
  |      .     |     .      |
  +============+============+  <-- NET
  |      .     |     .      |
  +------------+------------+
        [ SERVER SIDE ]
  📷 server phone          📷 receiver phone
  (own end, own           (own end, own
   right edge)             right edge)
```

Both phones tap the corners in the same positional order (near-right, near-left, far-left, far-right from their own end), so each phone's quarter lands on the tap-1 warp slot.

### 2. Homography & Court Mapping (`TableMapper`)

Tapping the four corners maps camera pixel space to a fixed 2D top-down grid:
* **Net Line:** $y = H / 2$
* **Center Line:** $x = W / 2$
* **My Quarter (tap-1, near-right):** Bottom-Right
* Naming note: quadrant names in code (`server_right`, ...) are slot-based — `server_right` always means the tap-1 quadrant, i.e. MY quarter on every phone.

### 3. Ball Tracking & Bounce Detection (`detect_ball_hsv`, `BallTracker` & `is_bounce`)
* Color masks evaluate white and orange balls using HSV color ranges alongside spatial continuity checks.
* False positives are cut with three cheap gates: a **table-ROI mask** from the calibrated corners (kills crowd/lights/walls off the table), **inter-frame motion** (static white blobs like logos and tape are rejected), and stricter **shape** checks (circularity, hull solidity, circle fill-ratio, vertex count).
* A confirmed-track gate (`BallTracker`) only feeds the serve logic once a blob persists near its predicted position, so single-frame lookalikes can't inject phantom bounces.
* Sequential trajectory points stored in a `deque` evaluate vertical velocity flips ($\Delta y_1 > 0$ then $\Delta y_2 < 0$) to detect frame-accurate bounce events.

### 4. Serve Adjudication (`ServeCaller`, fixed quarter, no selection)
* There is nothing to configure: positional taps put MY quarter on the tap-1 warp slot on every phone, and it is drawn thick with a `MY` tag.
* The first bounce seen is judged: inside my quarter (+line pad, centre/net lines count IN) → **IN**, anything else (wrong half or `off_table`) → **FAULT**. Cross-half bounces never enter the quarter-cropped track, so no ignore-other-half logic is needed.
* Known gap: a fault landing far outside your crop produces no verdict on your phone (nothing to track) — the other phone or the players call those.

### 5. 30 FPS notes (quarter-ROI, streaks, exposure)
* Each phone processes only its own quarter (`TableMapper.quarter_poly` + bbox crop, ~1/5 pixels) but keeps the 4-corner homography so net/centre lines stay consistent. Bounce is refined to the between-frame midpoint in table space with a line pad (centre line = IN).
* Fast serves streak (aspect 2-4): detector has a `round` + `streak` path (`Streak circ >=`, `Streak aspect <=` in TUNING). Streak fill is judged against the enclosing rect, not the enclosing circle.
* Tracker defaults are 30 FPS-tuned (`Track jump` 180px, `Confirm hits` 1, 2-frame constant-velocity coast). Watch `logcat` `tick avg=..ms` — it must stay under ~33ms to actually process 30 FPS.
* Exposure/focus: the app best-effort locks AE/AWB and picks the fastest preview FPS range via pyjnius (see `_lock_exposure`); focus is locked once at startup (`fixed` → `infinity` → `edof`, falling back to continuous AF only if the device has no locked mode) so bad light can't make it hunt mid-serve. Short shutter darkens the image, so play under strong hall lighting; if the ball vanishes, lower `White V min` in TUNING.

---

## Acknowledgments & References

* **Original Concept:** [PBLineCaller](https://github.com/clssmitty/PBLineCaller/) by `@clssmitty` — Pickleball line calling system built on homography and trajectory tracking.
* **Rules Reference:** Official [ITTF Handbook (Rule 2.6.3)](https://www.ittf.com/) for doubles serve specifications.
