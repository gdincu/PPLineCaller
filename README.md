# Ping-Pong Serve Line Caller

CPU-only doubles serve checker (Python, Kivy, OpenCV, NumPy). Android APK only. Zero-training ball detection + homography court mapping. Says **IN** via TTS when the bounce lands in your quadrant (+line pad, centre line = IN). Anything else stays silent.

Based on [clssmitty/PBLineCaller](https://github.com/clssmitty/PBLineCaller/).

## Setup

One phone per judged quarter, landscape on a tripod at your end, your right-hand edge. Frame **your quadrant only** (phones end up diagonal to each other).

Tap CALIBRATE, then your quadrant corners in order from your end: 1. near-right outer corner, 2. near-centre (end line + centre line), 3. far-centre (net + centre), 4. far-right (net + sideline). Then START SERVE. Your quarter is drawn with a thick outline.

## How it works

* **Mapping (`TableMapper`):** 4 quadrant taps → homography to a top-down rect. Your quarter always lands on the same warp slot, so there is no mode setting.
* **Detection:** white/orange HSV masks gated by quadrant-ROI, inter-frame motion, and shape (circularity, solidity, fill, vertices; relaxed `streak` path for fast-serve blur).
* **Speed:** Y-plane motion check first (static frames skip colour decode + HSV); otherwise only the ROI bbox is decoded from NV21. Preview uploads at ~15Hz during rallies. Keep `tick avg` under ~33ms (`logcat`).
* **Tracking (`BallTracker`, `is_bounce`):** confirmed track + dy-flip bounce detection, refined to the between-frame midpoint in table space.
* **Calling (`ServeCaller`):** IN-only. Inside quarter + 10px pad → `IN`; else keep watching.
* **Camera:** no AE/AWB lock (EC=0, flexible FPS range); focus locked once (`fixed` → `infinity` → `edof`, else continuous AF). Strong hall lighting helps.

## Tuning

* **TABLE PRESET:** tap once if you're on a blue table in a bright/concrete hall; otherwise skip it. It's the starting point, not a final tune.
* **AUTO TUNE:** table empty and phone still, tap it, wait for the "done" status (~1.5 s). Re-run whenever lighting changes (day/night, lights on/off).
* **ASSIST:** verify: on an empty table you should see zero yellow circles (or only tiny ones). If not, open TUNING and raise Min area / White V min, lower White S max until they disappear. During a serve the tracked ball shows as a red circle. Turn ASSIST off for match play.
* **MASK:** only needed if white false positives persist; the top-right inset shows what the detector sees. If it's mostly white (glare, floor, wall), tighten the white gates. It auto-enables ASSIST, so turn both off when done.
