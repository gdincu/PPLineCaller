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

* **ASSIST:** overlay of white-mask candidates (yellow, with area/circularity labels) plus a stats bar (`cands`, largest noise blob, expected ball range computed from your calibration, `min_area`). On an empty table every blob shown is a would-be false positive — raise `Min area` or tighten `White V min` / `White S max` until they disappear.
* **MASK:** small inset of the HSV white mask, to see if glare is leaking through.
* **TABLE PRESET:** one tap for a blue-table/concrete-hall view (`min_area 200, max_area 3200, roi_margin 30, motion_thresh 28, white_v_min 165, white_s_max 55`).
* **AUTO TUNE:** after calibrating, point at the empty table and tap it; hold still for the ~45-frame burst. Suggests `min_area` / `motion_thresh` / white gates from table noise and applies them (full report in `logcat`). Re-tap when lighting changes.
