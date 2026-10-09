# Bounce detection and Auto Tune investigation

Baseline: `66d7bbb`. Work branch: `codex/improve-bounce-auto-tune`.

## Root causes and changes

| Root cause | Resulting change |
| --- | --- |
| A bounce required exactly two adjacent vertical deltas exceeding 3 px in opposite directions. A plateau or several shallow deltas never satisfied this test. | Accumulate displacement over at most eight real observations; require a consistent descent, a short contact/plateau and a consistent ascent. Decide as soon as the configured rise threshold is met. |
| Constant-velocity coast positions fed the caller as if measured. Correcting an overshooting prediction could create a phantom reversal; velocity was sometimes computed from the prediction too. | Explicit observed/predicted metadata; only real detections feed the caller. Measure velocity/displacement from the last real detection over the actual frame gap. Coast remains available for association/display. |
| Caller history survived long gaps and track teleports; confirmation candidates survived missing detections. | Change track identity on reacquisition, clear caller history across identities, more than two missed samples, or a processing/observation gap over 200 ms. Confirmation requires consecutive hits. |
| The crop included a 100 px airborne corridor, but the polygon mask excluded it. Margin dilation was also about half the requested radius. | Expand the detection mask upward through that corridor and apply the actual margin. Reuse four bounded, immutable ROI masks to avoid per-frame dilation cost. |
| Motion intersected individual colour pixels before contour shape checks. Slowly moving balls could become crescents and fail area/circularity. A crop-area-based early exit could skip small distant balls. | Preserve the whole colour contour when at least 2% (minimum two pixels) overlaps dilated motion; scale the early exit to minimum ball area. Static disconnected components remain gated. |
| Rendering alternated BGR gray with raw NV21 Y as the previous frame, changing both brightness offset and range. | Always detect from the raw NV21 path, independently of preview rendering. Full colour decode happens only for rendering. |
| Odd ROI starting rows misaligned NV21 chroma with luma. | Even-align both x and y; verify ROI decode against full decode. |
| Auto Tune could set min_area above the expected far-ball area, rounded to coarse 20 px steps; static glare drove that threshold upward. | Set a conservative lower threshold below the far-ball estimate with 5 px steps. Leave glare rejection to motion/shape. |
| Geometry used bbox width and a fixed 0.70 perspective ratio for all cameras. | Estimate projected ball scale from the actual calibrated near and far table edges, allowing conservative contour/optics tolerance. |
| Auto Tune pooled spatial p97 differences with pixel medians, diluting the noise tail. It also measured BGR gray while detection sometimes used limited-range Y. | Use temporal p90 of per-frame spatial p97 in actual camera luma units. BGR-only calibration converts gray into comparable limited-range units. |
| Bright lines/glare contaminated brightness percentiles; thresholds had dim-light floors and were partly driven by existing white thresholds. | Estimate dominant saturated table colour, excluding neutral highlights. Allow lower V in dim scenes and moderate colour casts. Repeated calibration no longer depends on previous white/area gates. |
| Unstable camera/exposure bursts were accepted; Kivy can return its cached preview more than once. | Skip identical buffers, ignore five settling samples, require at least 20 usable samples, and reject exposure/scene instability. Finish after 45 fresh frames or a six-second timeout. Rejected calibration preserves current settings. |
| Empty frames were used to shrink the ROI without any evidence about airborne ball paths. | Preserve ROI, tracking and bounce settings. Neutral surfaces retain white gates and report the ambiguity. |

NV21 Y is limited range [16, 235], whereas BGR grayscale spans [0, 255].
See [OpenCV colour conversions](https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html).
Kivy's [Android camera implementation](https://github.com/kivy/kivy/blob/master/kivy/core/camera/camera_android.py)
returns its cached buffer from `grab_frame`; repeated clock ticks therefore do not
necessarily represent new camera frames. Exact-buffer deduplication cannot
identify newly captured frames whose contents are identical; this is conservative
for static scenes and the timeout prevents an indefinitely running calibration.

## Validation performed

Environment: Windows host, Python 3.12, NumPy 2.2.6, OpenCV 4.12.0. Test
requirements are pinned; CI runs the unit suite on Python 3.11. The repository
contained neither tests nor recorded footage before this work.

All 26 tests passed locally. Run `python -m unittest discover -s tests -v` for deterministic regressions covering:

- Sharp, shallow and plateau bounces; first sufficient real rising sample.
- Jitter, straight motion, toss apexes and repeated separated peaks.
- Contact/line classification, including a rising sample across a boundary.
- Missed detections, overshooting coast correction, reacquisition, stalls and confirmation gaps.
- Dim/normal/bright illumination, moderate colour cast, glare, camera noise and exposure instability.
- Repeated calibration, neutral surfaces, camera-size changes and failed-calibration preservation.
- NV21 ROI decode, slow-ball shape, airborne entry and 640/1280/1920-wide camera frames.
- Production tick integration: preview throttling still uses one detection path and speaks once.
- Cached preview frames and a 120-frame noisy glare sequence with isolated glints (no false calls).

`python tools/validate_detection.py --baseline 66d7bbb` loads the immutable
baseline from Git and compares it with the working code. Generated frames are
640×480 with 30 FPS timestamps, a calibrated trapezoid, blue table, and a
7 px-radius white ball. These are regression fixtures, not a field accuracy dataset.

| Generated case | Old raw trajectory bounce | New raw trajectory bounce | Old tuned full pipeline | New tuned full pipeline |
| --- | --- | --- | --- | --- |
| Sharp reversal | Frame 3 | Frame 3 | No call | IN at frame 3 |
| Contact plateau | Missed | Frame 3 | No call | IN at frame 3 |
| Shallow reversal | Missed | Frame 4 | No call | IN at frame 4 |
| One occluded observation | — | — | No call | IN at frame 4 |

Frames are zero-based. Sharp and plateau cases call on the first rising
observation; the shallow case calls when accumulated rise reaches 3 px. There is
no added fixed confirmation delay. In this fixture old Auto Tune chose min_area
140; revised calibration chose 45, allowing the distant ball contour to survive.

A warmed 90-frame detector-only host sample measured median/p95 approximately
3.69/4.95 ms before and 4.03/5.00 ms after. Values vary with host load. This is
not an Android FPS result or a full tick benchmark: preview upload, camera,
Kivy scheduling and TTS are excluded. The wider approach mask and full-contour
motion check add work; mask reuse limits that cost.

## Limits and device follow-up

The tests establish repeatable regressions, not real-world recall or false-call
rates. No Android device, recorded serves, APK build or live camera was available
for this validation. Real footage must cover bright/dim/flickering halls, orange
and white balls, fast blur, near/net and line contacts, occlusion, and several
phone positions. Check `tick avg` under 33 ms at 30 FPS on target phones, with
ASSIST off, and measure contact-to-verdict latency and missed/false events against
manually labelled contacts.

One image-y reversal remains a camera-dependent bounce cue. Perspective and
ball travel direction can hide a physical impact; a single camera cannot prove
table-plane contact from that cue alone. The observed vertex is less biased than
the old midpoint on the rising leg, but missed contact frames and the ball centre
above the table still make close line calls uncertain. No claim of exact
between-frame contact reconstruction is made.

Empty-table calibration cannot measure ball colour, blur, speed, radius under
all camera poses, or contact jitter. It preserves dynamics settings and should
be checked in ASSIST with real serves. Low-saturation tables retain previous
white gates because background-only samples cannot reliably establish contrast.
Very small/subpixel or heavily blurred balls remain limited by the camera and
fixed mask morphology; improve framing/lighting in those cases.
