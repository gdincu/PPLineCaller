# Ping-Pong Serve Line Caller

CPU-only doubles serve checker (Python, Kivy, OpenCV, NumPy). Android APK only. Zero-training ball detection + homography court mapping. Says **IN** via TTS when the bounce lands in your quadrant (+line pad, centre line = IN). Anything else stays silent.

Based on [clssmitty/PBLineCaller](https://github.com/clssmitty/PBLineCaller/).

## Setup

One phone per judged quarter, landscape on a tripod at your end, your right-hand edge. Frame **your quadrant only** (phones end up diagonal to each other).

Tap CALIBRATE, then your quadrant corners in order from your end: 1. near-right outer corner, 2. near-centre (end line + centre line), 3. far-centre (net + centre), 4. far-right (net + sideline). Then START SERVE. Your quarter is drawn with a thick outline.

## How it works

* **Mapping (`TableMapper`):** 4 quadrant taps → homography to a top-down rect. Your quarter always lands on the same warp slot, so there is no mode setting.
* **Detection:** white/orange HSV masks gated by quadrant-ROI, inter-frame motion, and shape (circularity, solidity, fill, vertices; relaxed `streak` path for fast-serve blur).
* **Speed:** detection runs on a background worker using the NV21 Y-plane motion reference and ROI-only colour decode. The handoff holds at most one pending frame; a busy worker replaces it with the newest frame. Live preview uploads at 5Hz, setup at 15Hz, and a verdict updates immediately. Texture coordinates handle vertical orientation, avoiding an image flip and extra bytes copy. Camera callback IDs skip cached frames without comparing full buffers.
* **Tracking (`BallTracker`, `bounce_vertex`):** confirmed real observations + accumulated descent/contact/ascent detection. Short plateaus and up to two missed observations are tolerated; coast predictions never prove a bounce. Calls use the observed contact vertex, with camera/track continuity checks.
* **Calling (`ServeCaller`):** IN-only. Inside quarter + 10px pad → `IN`; else keep watching.
* **Camera:** probes supported rear-camera/resolution/FPS combinations, verifies driver readback and prefers a stable 60 FPS mode. Falls back to lower supported rates when necessary; 720p is preferred within the same FPS tier. Camera polling targets 60Hz independently of preview uploads. No AE/AWB lock (EC=0); focus locked once (`fixed` → `infinity` → `edof`, else continuous AF). Strong lighting helps: a higher capture rate can make dim scenes darker. This legacy preview path does not provide portable manual shutter control, and a phone's 60 FPS video-recording mode does not guarantee 60 FPS NV21 preview.

## Tuning

* **TABLE PRESET:** tap once if you're on a blue table in a bright/concrete hall; otherwise skip it. It's the starting point, not a final tune.
* **AUTO TUNE:** table empty and phone still, tap it, wait for the "done" status (45 fresh frames, normally ~1.5–3 s; 6 s timeout). It pauses the current serve. Let exposure settle and retry if the scene is rejected as unstable. Re-run whenever lighting changes, then tap START SERVE. Neutral surfaces keep the existing white gates because an empty grey table cannot establish ball contrast.
* **ASSIST:** verify: on an empty table you should see zero yellow circles (or only tiny ones). If not, open TUNING and check whether they persist while moving the ball. Static glare can appear in ASSIST because that overlay omits motion gating. Adjust White V min / White S max carefully; raising Min area above a far-end ball can hide bounces. During a serve the tracked ball shows as a red circle. START SERVE automatically disables ASSIST and MASK to save work; they can be explicitly re-enabled for debugging.
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

For capture/preview changes and device performance checks, see
[capture performance validation](docs/capture-performance.md). Logcat reports
the selected camera/size/FPS range, callback FPS, submitted/processed FPS,
worker queue drops and UI/detection time. Configured FPS is not a measurement
of delivered frames.

For crashes during native library loading, see the
[Android startup investigation](docs/android-startup.md).
