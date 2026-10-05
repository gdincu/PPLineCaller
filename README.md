# Ping-Pong Serve Line Caller (Android-only)

An automated, real-time Ping-Pong doubles serve line-caller. Built using Python, Kivy, OpenCV, and NumPy, this project provides CPU-only, zero-training ball detection and homography-based court mapping to adjudicate doubles serves according to ITTF rules.

Runs only as a native Android app (APK) with real-time Text-to-Speech (TTS) announcements for instant calls (`IN` vs `FAULT`). There is no desktop/webcam build. The APK is built via the GitHub Action (`build-apk`).

This project is directly inspired by and based on [clssmitty/PBLineCaller](https://github.com/clssmitty/PBLineCaller/), adapting court-mapping and trajectory bounce-detection principles specifically for table tennis doubles serve rules.

---

## Features

* **Android-only, CPU-Only Performance:** Optimized using classic computer vision (HSV thresholding, circularity filtering, and frame deltas). No heavy GPU requirements, PyTorch, or cloud AI (Roboflow) needed.
* **4-Point Calibration in fixed table order:** Tap the 4 corners as 1. server-right, 2. server-left, 3. receiver-left, 4. receiver-right (left/right from the server's perspective). Each tap is drawn numbered on the preview.
* **Side phone position, one quarter each:** Two phones on the side (one per half) each judge a single quarter — `SIDE: server half (B1)` or `SIDE: receiver half (B2)` via one combined setting.
* **Bounce & Trajectory Analysis:** Tracks vertical velocity inversions ($\Delta y_1 > 0 \rightarrow \Delta y_2 < 0$) to accurately locate bounce points on the table.
* **ITTF Doubles Rule Verification:** Validates that doubles serve bounce 1 occurs in the Server's Right half and bounce 2 lands diagonally in the Receiver's Right half (ITTF 2.6.3). Centre line counts as IN.

---

## How It Works

### 1. Camera Setup

Two phones on the side of the table, landscape, each framing the whole table (wide enough to see faults) but only calling its own quarter. Pick one setting per phone: `SIDE: server half (B1)` or `SIDE: receiver half (B2)`.

```text
       [ RECEIVER SIDE ]
  +------------+------------+
  |            |            |
  +============+============+  <-- NET
  |            |            |
  +------------+------------+
       [ SERVER SIDE ]
  📷 SIDE (server half, B1)   📷 SIDE (receiver half, B2)
```

### 2. Homography & Court Mapping (`TableMapper`)

Tapping the four corners in table order maps camera pixel space to a fixed 2D top-down grid:
* **Net Line:** $y = H / 2$
* **Center Line:** $x = W / 2$
* **Server Right Quadrant:** Bottom-Right
* **Receiver Right Quadrant:** Top-Left (Diagonal)

### 3. Ball Tracking & Bounce Detection (`detect_ball_hsv`, `BallTracker` & `is_bounce`)
* Color masks evaluate white and orange balls using HSV color ranges alongside spatial continuity checks.
* False positives are cut with three cheap gates: a **table-ROI mask** from the calibrated corners (kills crowd/lights/walls off the table), **inter-frame motion** (static white blobs like logos and tape are rejected), and stricter **shape** checks (circularity, hull solidity, circle fill-ratio, vertex count).
* A confirmed-track gate (`BallTracker`) only feeds the serve logic once a blob persists near its predicted position, so single-frame lookalikes can't inject phantom bounces.
* Sequential trajectory points stored in a `deque` evaluate vertical velocity flips ($\Delta y_1 > 0$ then $\Delta y_2 < 0$) to detect frame-accurate bounce events.

### 4. Serve Adjudication (`ServeCaller`, one combined side setting)
* **SIDE: server half (B1):** judges the first bounce only (must be `server_right`). Use on the server-half side phone.
* **SIDE: receiver half (B2):** ignores server-side bounces (bounce 1 belongs to the other phone) and judges the first receiver-side bounce (must be `receiver_right`). Use on the receiver-half side phone.
* Bounces outside the calibrated table (`off_table`) count as **FAULT** for the phone responsible for that half.

---

## Build the APK

Push to GitHub; the `build-apk` workflow builds with buildozer/python-for-android and uploads the APK artifact. Install on both phones, pick the side per phone, calibrate (4 taps in order), then START SERVE.

---

## Acknowledgments & References

* **Original Concept:** [PBLineCaller](https://github.com/clssmitty/PBLineCaller/) by `@clssmitty` — Pickleball line calling system built on homography and trajectory tracking.
* **Rules Reference:** Official [ITTF Handbook (Rule 2.6.3)](https://www.ittf.com/) for doubles serve specifications.
