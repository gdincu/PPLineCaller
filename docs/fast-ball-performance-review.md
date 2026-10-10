# Fast-ball detection performance review

Date: 2026-10-09. Scope: investigation of the code before performance changes.
Implementation of items 1 and 4 is documented in [capture performance](capture-performance.md).

## Conclusion

There are both throughput risks and demonstrated detection failures. The largest measured CPU detector component on this host is colour-mask construction, but real Android camera delivery, exposure and graphics costs were not measured. The strongest reproduced explanation for speed-dependent misses is that motion blur changes white-ball colour and shape enough to fail the detector before tracking receives an observation.

At 20-30 km/h a ball travels 18.5-27.8 cm between 30 FPS captures, and 37.0-55.6 cm at 15 FPS. These are physical travel distances, not image displacement or vertical height. For a 40 mm ball they are about 4.6-6.9 diameters per 30 FPS interval. Image displacement depends on camera pose and distance. Merely reducing a few milliseconds of CPU time does not repair a ball absent from the colour mask.

## Ranked findings and proposed fixes

1. **Camera sampling and exposure** (`app/main.py:236`, `544`, `560-613`). The app requests 1280x720, polls at a nominal 30 Hz, and deliberately selects the lowest minimum FPS among ranges with the highest maximum. A range such as 15-30 permits reduced FPS in dim light; fixed-only high-FPS ranges are skipped. There is no explicit shutter-time control. Actual capture FPS/exposure are unknown. Add a sport capture mode choosing a supported stable rate, then investigate 60 FPS with a matching consumer rate. Higher FPS alone does not guarantee sufficiently short exposure. Camera2 can provide timestamps/exposure metadata and manual exposure on supported devices. A proposed 1/500-1/1000 s exposure target gives roughly 17-8 mm physical travel at 30 km/h; verify brightness/noise and support on the phone. Preserve the previous dark-startup fix by waiting for AE convergence and making this a deliberate mode, rather than blindly locking startup exposure.

2. **Blurred white balls fail colour segmentation** (`app/serve_caller.py:223-231`). The white mask requires S <= 60 and V >= 150 by default; TABLE PRESET tightens these to 55/165 (`app/main.py:87-93`). Averaging a white ball with a blue table makes its apparent colour saturated. A generated 20 px-diameter white ball passed with 20 px horizontal box blur, but failed with 40/60/80 px blur. At the 40 px-blurred centre, NV21-decoded HSV was [109,72,191]: V passes, S fails. Lowering V as far as 90, with min_area=40, did not recover this case. Raising S max to 90 recovered the 40 px case with default min_area=200; S max=150 recovered 60 px, but not 80 px. These are sensitivity probes, not safe global settings: wider colour gates can admit the table and glare. Prefer shorter exposure and a local motion/track-conditioned contrast candidate path, with strict temporal association, rather than globally broadening HSV.

3. **The existing blur path has hard shape limits** (`app/serve_caller.py:275-300`). Defaults require aspect <= 4 and streak circularity >= 0.35. Bright synthetic ellipses at axis ratios 1-4 passed; 5 and 6 failed (measured aspects 4.9/5.9; circularities 0.416/0.358). A larger aspect limit would recover this isolated shape case, but real streaks may already have failed colour/area gates. Use predicted direction, projected ball thickness and elapsed time to validate elongated candidates. The fixed preferred radius of 18 and previous-position distance penalty can favour a round distractor over a real displaced streak. Score multiple candidates against a timestamped motion prediction and calibrated expected scale. Do not rely on minEnclosingCircle radius as a streak's physical ball radius.

4. **Detection and expensive preview work share the UI callback** (`app/main.py:754-894`). Each rendered frame can perform ROI decode for detection, another full-frame decode, flip, BGR-to-RGB conversion, a bytes copy and texture upload. ASSIST reruns HSV/mask/contour diagnostics (`381-391`), and Auto Tune enables ASSIST automatically (`373-376`); START SERVE does not disable it. Keep ASSIST/MASK off during normal serves. Reuse diagnostics if enabled. Move detection to a worker with explicit frame ownership and a bounded latest-frame handoff, keeping GL work on the main thread. Throttle preview by elapsed time and optionally use a smaller display buffer while preserving coordinate mapping. Under load, record skipped frames rather than pretending every capture was processed.

5. **Tracking gates use frames and pixels rather than elapsed time** (`app/serve_caller.py:770-835`). A normal consecutive observation farther than 180 px starts a new track, clearing caller history. The gate expands for missing detector observations, but not for camera captures skipped between UI ticks. Timestamped prediction and uncertainty-based association should scale with actual elapsed capture time and camera geometry. Simply raising max_jump can join unrelated blobs and increase false calls. The caller clears history after more than two missing observations or a >200 ms observation gap (`901-918`); retain the principle that predictions cannot prove contact.

6. **Few samples and a camera-dependent bounce cue** (`app/serve_caller.py:727-754`). A bounce needs at least three real observations including descent and ascent in image y. Changing horizontal travel/perspective can hide a physical bounce even when detections are good. Add timestamped trajectory fitting and evaluate camera pose with labelled contacts. A single camera and table-plane homography cannot guarantee exact contact location for an airborne ball or reconstruct an unsampled impact reliably. Improving CPU throughput alone does not remove this geometry limitation.

7. **Default minimum area can exclude small distant balls** (`app/main.py:75-79`; `app/serve_caller.py:244-247`, `524-538`). min_area=200 corresponds to roughly a 16 px diameter before contour discretization. The fixture's far edge projects a 40 mm ball to about 13 px diameter. AUTO TUNE reduces area conservatively, but empty-table tuning cannot learn motion blur or a fast ball's colour. Fixed 5x5 median and 3x3 opening can additionally remove thin streaks/small balls before contour scoring. Use projected ball scale when deciding morphology and area; validate against real noise and glare.

## Host timing evidence

Windows Python 3.13.14, OpenCV 4.12.0, NumPy 2.2.6. Generated blue-table fixture based on tests/test_detection.py, one sharp moving white ball, static reference, warm cached ROI masks. 10 warm-up calls followed by 100 measured calls. Frame generation/NV21 encoding excluded. Timed wrappers add small overhead. Stage medians are separate distributions and should not be summed as a percentile of the total.

1280x720 fixture, OpenCV using the host's default 16 threads:

| Operation | Median ms | p95 ms |
| --- | ---: | ---: |
| Full ROI detector | 2.622 | 3.087 |
| Colour-mask creation within detector | 1.560 | not collected |
| Contour scoring within detector | 0.129 | not collected |
| Luma crop/resize within detector | 0.127 | not collected |
| ROI colour decode within detector | 0.117 | not collected |
| Motion mask within detector | 0.117 | not collected |
| Full-frame preview decode separately | 0.106 | 0.154 |
| Preview flip/RGB/bytes preparation separately | 1.266 | 1.654 |
| ASSIST diagnostics separately | 1.893 | 2.451 |

With one OpenCV thread, full detector median/p95 was 2.415/2.601 ms, preview preparation 1.208/1.404 ms, and ASSIST 2.105/2.238 ms. These host results do not select an Android thread count. An independent single-thread microbenchmark of the 640x396 detection crop measured HSV conversion 0.321 ms, white inRange 0.174 ms, medianBlur(5) 0.319 ms and opening(3) 0.053 ms. Colour-mask creation includes three inRange operations, mask combination and cleanup. The two orange hue ranges are adjacent and can be replaced by one [0,25] range with identical S/V bounds; this is a small exact optimization. Test morphology alternatives against small/thin balls before changing them.

**Unmeasured:** Android callback/Java-array copying, actual sensor cadence, exposure/readout, Kivy scheduling, provider FBO graphics, texture upload/GPU completion, thermal throttling, clutter-heavy scenes and real serve recall. This is not an Android FPS benchmark. No real footage or connected phone was used.

Kivy 2.3.1's Android provider independently schedules a 30 Hz SurfaceTexture/FBO update and keeps the latest preview buffer; grab_frame converts that buffer to bytes. The app's 15 Hz overlay throttle does not throttle that separate provider event. This is an upstream-source observation, not verification of the exact Kivy version shipped in the APK; buildozer.spec leaves Kivy unpinned.

The existing `tick avg` (`app/main.py:877-883`) measures only callback execution, excludes time between callbacks/provider graphics and early-return cached-frame ticks, and smooths costs with an EMA. A value below 33 ms does not establish 30 unique frames per second or low frame age.

## Implementation order

1. Add capture IDs/timestamps, unique capture and processed FPS, dropped-frame counts, capture-to-process age, stage p50/p95/max, render/non-render timing and detector rejection counts (colour, area, shape, association). Use a callback sequence for deduplication rather than whole-buffer equality when provider ownership permits it.
2. Record a short labelled set of slow and 20-30 km/h serves under actual hall lighting; log available FPS ranges and camera configuration. Compare ASSIST off/on and exposure/preview changes on the same clips/device.
3. Improve capture/exposure and separate detection from preview/UI work. At 60 FPS the per-frame budget is 16.7 ms and polling/provider changes must accompany the camera change.
4. Add blur/background-mixed synthetic regression cases plus real-clip replay. Improve local candidate extraction and time-aware association, preserving rejection of glare and predictions as bounce evidence.
5. Optimize colour-mask work, resolution and copies according to measured Android stage costs. Keep enough pixels to resolve the far-end ball.

All 29 existing unit tests passed (`.venv/Scripts/python.exe -m unittest discover -s tests -v`). Existing generated serves use sharp circles and 30 FPS timestamps; they are regression checks, not field accuracy or fast-blur coverage. TTS already runs on a background thread (`app/main.py:731-740`) and homography maps individual points rather than warping every frame, so neither is a leading per-frame CPU optimization target.

## Primary references

- [Kivy 2.3.1 Android provider source](https://raw.githubusercontent.com/kivy/kivy/2.3.1/kivy/core/camera/camera_android.py): latest buffer, byte conversion, provider update scheduling and FBO work.
- [Kivy Clock documentation](https://kivy.org/doc/stable/api-kivy.clock.html): scheduling is subject to clock/frame timing.
- [Android Camera2 capture requests](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest): AE FPS range and exposure time are separate controls; AE can override manual exposure requests, and capability checks are required.
