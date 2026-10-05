# Ping-Pong Serve Line Caller

An automated, real-time Ping-Pong doubles serve line-caller. Built using Python, Kivy, OpenCV, and NumPy, this project provides CPU-only, zero-training ball detection and homography-based court mapping to adjudicate doubles serves according to ITTF rules.

This project is directly inspired by and based on [clssmitty/PBLineCaller](https://github.com/clssmitty/PBLineCaller/), adapting court-mapping and trajectory bounce-detection principles specifically for table tennis doubles serve rules.

---

## Features

* **CPU-Only Performance:** Optimized using classic computer vision (HSV thresholding, circularity filtering, and frame deltas). No heavy GPU requirements, PyTorch, or cloud AI (Roboflow) needed.
* **Simple 4-Point Calibration:** Interactive 1-step setup using a 4-point homography transform (`cv2.getPerspectiveTransform`) to warp the table into a standard top-down perspective.
* **Bounce & Trajectory Analysis:** Tracks vertical velocity inversions ($\Delta y_1 > 0 \rightarrow \Delta y_2 < 0$) to accurately locate bounce points on the table.
* **ITTF Doubles Rule Verification:** Validates that doubles serve bounce 1 occurs in the Server's Right half and bounce 2 lands diagonally in the Receiver's Right half (ITTF 2.6.3).
* **Cross-Platform:** Runs locally on Desktop (Webcam) or as a native Android App with real-time Text-to-Speech (TTS) announcements for instant calls (`IN` vs `FAULT`).

---

## How It Works

### 1. Camera Setup
Place your mobile device or camera on a tripod **behind the server**, elevated and centered on the center line looking toward the receiver in landscape orientation.

```text
       [ RECEIVER SIDE ]
  +------------+------------+  (Top-Left Quadrant = Receiver Right)
  |            |            |
  +============+============+  <-- NET
  |            |            |
  +------------+------------+  (Bottom-Right Quadrant = Server Right)
       [ SERVER SIDE ]
             📷 (Camera Position)
```

### 2. Homography & Court Mapping (`TableMapper`)
Tapping the four outer corners of the table maps camera pixel space to a fixed 2D top-down grid:
* **Net Line:** $y = H / 2$
* **Center Line:** $x = W / 2$
* **Server Right Quadrant:** Bottom-Right
* **Receiver Right Quadrant:** Top-Left (Diagonal)

### 3. Ball Tracking & Bounce Detection (`detect_ball_hsv` & `is_bounce`)
* Color masks evaluate white and orange balls using HSV color ranges alongside spatial continuity checks.
* Sequential trajectory points stored in a `deque` evaluate vertical velocity flips ($\Delta y_1 > 0$ then $\Delta y_2 < 0$) to detect frame-accurate bounce events.

### 4. Serve Adjudication (`ServeCaller`)
The state machine monitors the detected bounces and compares them against official rules:
- **Bounce 1:** Must land in `server_right`. (Otherwise $\rightarrow$ **FAULT**)
- **Bounce 2:** Must land in `receiver_right`. (Otherwise $\rightarrow$ **FAULT**)
- If both conditions pass $\rightarrow$ **IN**.

---

## Acknowledgments & References

* **Original Concept:** [PBLineCaller](https://github.com/clssmitty/PBLineCaller/) by `@clssmitty` — Pickleball line calling system built on homography and trajectory tracking.
* **Rules Reference:** Official [ITTF Handbook (Rule 2.6.3)](https://www.ittf.com/) for doubles serve specifications.
