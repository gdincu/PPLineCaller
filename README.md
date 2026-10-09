# Ping-Pong Serve Line Caller

CPU-only doubles serve checker (Python, Kivy, OpenCV, NumPy). Android APK only. Zero-training ball detection + homography court mapping. Says **IN** via TTS when the bounce lands in your quadrant (+line pad, centre line = IN). Anything else stays silent.

Based on [clssmitty/PBLineCaller](https://github.com/clssmitty/PBLineCaller/).

## Setup

One phone per judged quarter, landscape on a tripod at your end, your right-hand edge. Frame **your quadrant only** (phones end up diagonal to each other).

Tap CALIBRATE, then your quadrant corners in order from your end: 1. near-right outer corner, 2. near-centre (end line + centre line), 3. far-centre (net + centre), 4. far-right (net + sideline). Then START SERVE. Your quarter is drawn with a thick outline.

## How it works

* **Mapping (`TableMapper`):** 4 quadrant taps → homography to a top-down rect. Your quarter always lands on the same warp slot, so there is no mode setting.
* **Detection:** white/orange HSV masks gated by quadrant-ROI, inter-frame motion, and shape (circularity, solidity, fill, vertices; relaxed `streak` path for fast-serve blur).
* **Speed:** every detection uses the same NV21 Y-plane motion reference (static frames skip colour decode + HSV); otherwise only the ROI bbox is decoded. Preview uploads at ~15Hz during rallies, independently of detection. Cached preview buffers are skipped and ROI masks are reused. Keep `tick avg` under ~33ms (`logcat`).
* **Tracking (`BallTracker`, `bounce_vertex`):** confirmed real observations + accumulated descent/contact/ascent detection. Short plateaus and up to two missed observations are tolerated; coast predictions never prove a bounce. Calls use the observed contact vertex, with camera/track continuity checks.
* **Calling (`ServeCaller`):** IN-only. Inside quarter + 10px pad → `IN`; else keep watching.
* **Camera:** no AE/AWB lock (EC=0, flexible FPS range); focus locked once (`fixed` → `infinity` → `edof`, else continuous AF). Strong hall lighting helps.

## Tuning

* **TABLE PRESET:** tap once if you're on a blue table in a bright/concrete hall; otherwise skip it. It's the starting point, not a final tune.
* **AUTO TUNE:** table empty and phone still, tap it, wait for the "done" status (45 fresh frames, normally ~1.5–3 s; 6 s timeout). It pauses the current serve. Let exposure settle and retry if the scene is rejected as unstable. Re-run whenever lighting changes, then tap START SERVE. Neutral surfaces keep the existing white gates because an empty grey table cannot establish ball contrast.
* **ASSIST:** verify: on an empty table you should see zero yellow circles (or only tiny ones). If not, open TUNING and check whether they persist while moving the ball. Static glare can appear in ASSIST because that overlay omits motion gating. Adjust White V min / White S max carefully; raising Min area above a far-end ball can hide bounces. During a serve the tracked ball shows as a red circle. Turn ASSIST off for match play.
* **MASK:** only needed if white false positives persist; the top-right inset shows what the detector sees. If it's mostly white (glare, floor, wall), tighten the white gates. It auto-enables ASSIST, so turn both off when done.

## Development validation

No recorded footage is bundled. Regressions use deterministic trajectories and
generated OpenCV frame sequences, including NV21 encoding, lighting, glare,
occlusion and camera resolution changes. Run the same checks as CI:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate; Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python tools/validate_detection.py --baseline 66d7bbb
```

See [investigation and validation notes](docs/detection-validation.md) for root
causes, comparison results, and remaining device/footage checks.
