"""Kivy APK entry: ping-pong doubles serve line-caller. CPU-only (OpenCV + numpy).

Android-only: built as an APK via the GitHub Action (buildozer). There is
no desktop/webcam path.

Flow:
  1. Fix the phone at your end of the table on your right-hand edge
     (tripod, landscape), framing the whole table. Both phones tap the
     corners positionally from their own end, so each phone's own quarter
     always lands on the same warp slot — no mode or quarter setting.
  2. Tap CALIBRATE then tap the 4 table corners IN ORDER from YOUR end as
     you face the table:
        1. NEAR-RIGHT (your end, your right)
        2. NEAR-LEFT  (your end, your left)
        3. FAR-LEFT (far end, same long edge as 2)
        4. FAR-RIGHT (far end, same long edge as 1)
  3. Tap START SERVE, serve, app speaks/shows IN or FAULT for the first
     bounce seen: inside my quarter -> IN, else FAULT.
"""
import cv2
import numpy as np
import threading
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.metrics import dp

from serve_caller import (TableMapper, ServeCaller, detect_ball_hsv,
                            BallTracker)

from plyer import tts

# Ordered calibration prompts (positional: from your own end as you face
# the table — same order on both phones, so each phone's own quarter lands
# on the tap-1 warp slot).
CALIB_STEPS = (
    "NEAR-RIGHT (your end, your right)",
    "NEAR-LEFT (your end, your left)",
    "FAR-LEFT (far end, same edge as NEAR-LEFT)",
    "FAR-RIGHT (far end, same edge as NEAR-RIGHT)",
)
# Short tags drawn next to each tapped point on the preview.
CALIB_SHORT = ("1 NR", "2 NL", "3 FL", "4 FR")

# Live-tuning panel (collapsed by default). Defaults equal the hardcoded
# detector/tracker/bounce settings. Row format:
# (param key, label, slider min, slider max, step, value format, is-int).
TUNE_PARAMS = [
    ("min_area", "Min area", 20, 1000, 20, "{:.0f}", True),
    ("max_area", "Max area", 1000, 12000, 100, "{:.0f}", True),
    ("min_circ", "Circularity >=", 0.30, 0.95, 0.01, "{:.2f}", False),
    ("min_circ_streak", "Streak circ >=", 0.10, 0.60, 0.01, "{:.2f}", False),
    ("max_aspect", "Streak aspect <=", 1.5, 6.0, 0.1, "{:.1f}", False),
    ("min_solidity", "Solidity >=", 0.50, 1.00, 0.01, "{:.2f}", False),
    ("min_fill", "Fill ratio >=", 0.50, 1.00, 0.01, "{:.2f}", False),
    ("min_vertices", "Vertices >=", 0, 12, 1, "{:.0f}", True),
    ("roi_margin", "ROI margin", 0, 120, 5, "{:.0f}", True),
    ("motion_thresh", "Motion thr", 5, 100, 1, "{:.0f}", True),
    ("white_v_min", "White V min", 80, 255, 1, "{:.0f}", True),
    ("white_s_max", "White S max", 0, 150, 1, "{:.0f}", True),
    ("max_jump", "Track jump", 20, 300, 5, "{:.0f}", False),
    ("confirm_streak", "Confirm hits", 1, 5, 1, "{:.0f}", True),
    ("min_drop", "Bounce drop", 1, 15, 0.5, "{:.1f}", False),
    ("min_rise", "Bounce rise", 1, 15, 0.5, "{:.1f}", False),
]
TUNE_DEFAULTS = {
    "min_area": 160, "max_area": 4800, "min_circ": 0.65,
    "min_circ_streak": 0.35, "max_aspect": 4.0,
    "min_solidity": 0.85, "min_fill": 0.70, "min_vertices": 6,
    "roi_margin": 40, "motion_thresh": 25, "white_v_min": 150,
    "white_s_max": 60, "max_jump": 180.0, "confirm_streak": 1,
    "min_drop": 3.0, "min_rise": 3.0,
}


class PreviewImage(Image):
    """Camera preview: taps drive calibration, swipe-down collapses tuning.

    Tap vs swipe is decided on touch-up (movement < 12dp = tap). The swipe
    lives here — not on the tuning drawer, where it would fight the
    ScrollView — and taps are ignored mid-serve exactly as before.
    """

    def __init__(self, app, **kwargs):
        super().__init__(**kwargs)
        self._app = app
        self._down = None

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            self._down = (touch.x, touch.y, touch.uid)
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        d = self._down
        if d is not None and touch.uid == d[2]:
            self._down = None
            if self.collide_point(touch.x, touch.y):
                dx, dy = touch.x - d[0], touch.y - d[1]
                if (self._app.tuning_open and dy < -dp(60)
                        and abs(dx) < dp(40)):
                    self._app.toggle_tuning()
                    return True
                if abs(dx) < dp(12) and abs(dy) < dp(12):
                    return self._app.on_preview_tap(self, touch)
        return super().on_touch_up(touch)


class ServeApp(App):
    def build(self):
        self.table = TableMapper()
        # No mode/quarter selection: positional taps put MY quarter on the
        # tap-1 warp slot on every phone, so the caller always wants it.
        self.caller = ServeCaller(self.table)
        self.calib_pts = []
        self.calibrating = False
        # Android capture via Kivy's camera provider (pyjnius -> Camera API),
        # because cv2.VideoCapture has no working backend in the p4a opencv build.
        self.cam = None
        self.params = dict(TUNE_DEFAULTS)
        self.tracker = BallTracker(max_jump=self.params["max_jump"],
                                   confirm_streak=int(self.params["confirm_streak"]))
        self._prev_small = None
        self._tex = None
        self._said = False
        # perf: EMA of tick cost + render throttle (preview at ~15Hz during
        # SERVE_LIVE, full rate while calibrating/judging).
        self._tick_count = 0
        self._tick_ms_ema = 0.0
        self._render_every = 2
        self._exposure_locked = False
        self._tts_thread = None
        # Focus: always locked (fixed/infinity first) — no hunting mid-serve.
        # Continuous AF was tried but hunted in bad hall light; a frozen
        # focus plane tests better once the table is framed. Applied once
        # via pyjnius; falls back to continuous AF if the device has no
        # fixed/infinity/edof mode.
        self.tuning_open = False
        self._tune_sliders = {}
        self._tune_values = {}
        self._tune_spec = {k: (fmt, is_int) for k, _, _, _, _, fmt, is_int in TUNE_PARAMS}

        root = BoxLayout(orientation="vertical")
        # allow_stretch + keep_ratio: frame fills the widget (letterboxed),
        # so the preview is as large as possible; taps are mapped through
        # the displayed rect (norm_image_size) in on_preview_tap.
        # PreviewImage also turns a swipe-down into tuning collapse.
        self.view = PreviewImage(self, allow_stretch=True, keep_ratio=True)
        self.status = Label(text="Tap CALIBRATE, then 4 corners in order",
                            size_hint_y=0.12)

        bar = BoxLayout(size_hint_y=0.12)
        b_cal = Button(text="CALIBRATE")
        b_cal.bind(on_press=self.start_calib)
        b_go = Button(text="START SERVE")
        b_go.bind(on_press=self.start_serve)
        bar.add_widget(b_cal)
        bar.add_widget(b_go)
        self.action_row = bar

        root.add_widget(self.view)
        root.add_widget(self.status)
        root.add_widget(bar)
        tune_bar = BoxLayout(size_hint_y=0.10)
        self.tune_toggle = Button(text="TUNING +")
        self.tune_toggle.bind(on_press=self.toggle_tuning)
        b_defaults = Button(text="RESET DEFAULTS")
        b_defaults.bind(on_press=self.reset_tuning)
        tune_bar.add_widget(self.tune_toggle)
        tune_bar.add_widget(b_defaults)
        root.add_widget(tune_bar)
        self.tune_scroll = ScrollView(size_hint_y=None, height=0)
        self.tune_grid = GridLayout(cols=3, size_hint_y=None,
                                    spacing=dp(4), padding=dp(4))
        self.tune_grid.bind(minimum_height=self.tune_grid.setter("height"))
        for key, label, lo, hi, step, fmt, _is_int in TUNE_PARAMS:
            name = Label(text=label, size_hint_x=0.45,
                         size_hint_y=None, height=dp(40),
                         halign="left", valign="middle")
            name.bind(size=name.setter("text_size"))
            slider = Slider(min=lo, max=hi, step=step,
                            value=self.params[key],
                            size_hint_x=0.35,
                            size_hint_y=None, height=dp(40))
            slider.bind(value=lambda inst, val, k=key: self._on_tune(k, val))
            val_label = Label(text=fmt.format(self.params[key]),
                              size_hint_x=0.20,
                              size_hint_y=None, height=dp(40))
            self.tune_grid.add_widget(name)
            self.tune_grid.add_widget(slider)
            self.tune_grid.add_widget(val_label)
            self._tune_sliders[key] = slider
            self._tune_values[key] = val_label
        self.tune_scroll.add_widget(self.tune_grid)
        root.add_widget(self.tune_scroll)
        Clock.schedule_interval(self.tick, 1.0 / 30.0)
        return root

    # -- config ---------------------------------------------------------
    def toggle_tuning(self, *_):
        # Opening the drawer hides the action row: a fixed 320dp drawer
        # inside a proportional BoxLayout otherwise squeezes every button
        # row tiny. The drawer + preview keep full size, and the toggle
        # stays put as a big close handle (plus swipe-down on the preview
        # also collapses).
        self.tuning_open = not self.tuning_open
        if self.tuning_open:
            self.tune_toggle.text = "TUNING - CLOSE"
            w = self.action_row
            w.size_hint_y = None
            w.height = 0
            w.disabled = True
            w.opacity = 0
            self.tune_scroll.height = dp(320)
        else:
            self.tune_toggle.text = "TUNING +"
            w = self.action_row
            w.size_hint_y = 0.12
            w.disabled = False
            w.opacity = 1
            self.tune_scroll.height = 0

    def reset_tuning(self, *_):
        for key in self._tune_sliders:
            self._tune_sliders[key].value = TUNE_DEFAULTS[key]
        # re-apply explicitly (slider callbacks may not fire if a value
        # is already at its default)
        self.params = dict(TUNE_DEFAULTS)
        for key, (fmt, _is_int) in self._tune_spec.items():
            self._tune_values[key].text = fmt.format(self.params[key])
        self._apply_live_settings()
        self.status.text = "Tuning reset to defaults."

    def _on_tune(self, key, value):
        fmt, is_int = self._tune_spec[key]
        value = int(round(value)) if is_int else float(value)
        self.params[key] = value
        self._tune_values[key].text = fmt.format(value)
        self._apply_live_settings()

    def _apply_live_settings(self):
        p = self.params
        self.tracker.max_jump = float(p["max_jump"])
        self.tracker.confirm_streak = int(p["confirm_streak"])
        self.caller.min_drop = float(p["min_drop"])
        self.caller.min_rise = float(p["min_rise"])
    # -- calibration ----------------------------------------------------
    def start_calib(self, *_):
        self.calib_pts = []
        self.calibrating = True
        self._prev_small = None  # bbox changes after re-calibration
        self.status.text = f"Tap 1/4: {CALIB_STEPS[0]}"

    def _tap_to_frame(self, img, touch):
        """Window touch -> frame (x, y) pixels, or None if outside the image.

        The frame is letterboxed inside the widget (keep_ratio); only the
        centred norm_image_size rect shows the camera image.
        """
        h, w = self.frame.shape[:2]
        lx = touch.x - img.x
        ly = touch.y - img.y
        tw, th = img.norm_image_size
        if tw <= 0 or th <= 0:
            return None
        x0 = (img.width - tw) / 2.0
        y0 = (img.height - th) / 2.0
        if not (x0 <= lx <= x0 + tw and y0 <= ly <= y0 + th):
            return None
        u = (lx - x0) / tw   # 0 = left edge of frame
        v = (ly - y0) / th   # 1 = top edge of displayed image
        return u * w, (1.0 - v) * h  # frame row 0 is the top

    def on_preview_tap(self, img, touch):
        if getattr(self, "frame", None) is None:
            return False
        if not img.collide_point(touch.x, touch.y):
            return False
        pt = self._tap_to_frame(img, touch)
        if pt is None:
            if self.calibrating:
                self.status.text = (f"Tap inside the camera view "
                                    f"({len(self.calib_pts)}/4: {CALIB_STEPS[len(self.calib_pts)]})")
            return True
        x, y = pt
        if self.calibrating:
            # touch is in window coords; the Image widget sits above the
            # status label + buttons, so _tap_to_frame already subtracts the
            # widget position (origin bottom-left). NOTE: do NOT use
            # img.to_widget() here: with the default relative=False it
            # returns the point unchanged for plain widgets (no
            # subtraction), which put every marker one widget-height high.
            self.calib_pts.append((x, y))
            n = len(self.calib_pts)
            if n < 4:
                self.status.text = f"Tap {n + 1}/4: {CALIB_STEPS[n]}"
            else:
                # Positional taps from your own end: tap-1 (near-right) is
                # MY quarter on every phone, so no quarter selection needed.
                near_r, near_l, far_l, far_r = self.calib_pts
                self.table.set_corners_table_order(near_r, near_l,
                                                   far_l, far_r)
                self.calibrating = False
                self.caller = ServeCaller(self.table)
                self.tracker.reset()
                self._prev_small = None  # new crop -> fresh motion ref
                self.status.text = "Calibrated. Tap START SERVE."
            return True
        # Idle taps do nothing (quarter is fixed to the tap-1 slot).
        return False

    def start_serve(self, *_):
        if self.table.H is None:
            self.status.text = "Calibrate first (4 corners in order)."
            return
        self.caller = ServeCaller(self.table,
                                  min_drop=self.params["min_drop"],
                                  min_rise=self.params["min_rise"])
        self.caller.reset()
        self.tracker.reset()
        self._prev_small = None  # fresh motion reference per serve
        self._said = False
        self._tick_count = 0
        self._tick_ms_ema = 0.0
        self.status.text = "Watching serve..."

    # -- android camera -------------------------------------------------
    def on_start(self):
        try:
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA], self._camera_granted)
        except Exception:
            self._init_camera()

    # The permission callback runs on the Android UI thread; the camera
    # provider creates a GL texture, which must happen on Kivy's main thread.
    @mainthread
    def _camera_granted(self, permissions, grant_results):
        if grant_results and grant_results[0]:
            self._init_camera()
        else:
            self.status.text = "Camera permission denied"

    def _init_camera(self):
        if self.cam is not None:
            return
        try:
            from kivy.core.camera import Camera as CoreCamera
            self.cam = CoreCamera(index=0, resolution=(1280, 720))
            self.cam.start()
            self._lock_exposure()
            ok, msg = self._ensure_focus()
            Logger.info(f"PPLineCaller: initial focus: {msg}")
            # Warm the TTS engine now (background thread) so the first
            # verdict's "In"/"Fault" doesn't pay the 1-3s init cost.
            self.say("Ready")
        except Exception as e:
            self.status.text = f"Camera error: {e}"

    def _lock_exposure(self):
        """pyjnius experiment: shorten shutter to freeze the 30 FPS streak.

        At 30 FPS auto-exposure holds the shutter open ~1/30s, smearing a
        50 km/h serve into a faint streak. Forcing a short exposure keeps
        the frame rate at 30 FPS but sharpens the blob/ellipse. Old Camera
        API has no direct shutter-time call, so we: lock AE + white balance,
        nudge exposure-compensation down, and request the fastest preview
        FPS range the device reports. (Focus is handled separately in
        _ensure_focus: always continuous AF.) All best-effort: any failure
        only logs. Compensate darkness with strong hall lighting (preferred)
        or lower white_v_min via TUNING.
        """
        if self._exposure_locked:
            return
        try:
            cam = getattr(self.cam, "_android_camera", None)
            if cam is None:
                Logger.info("PPLineCaller: no _android_camera, skip exposure lock")
                return
            params = cam.getParameters()
            info = []
            try:
                if hasattr(params, "setAutoExposureLock"):
                    params.setAutoExposureLock(True)
                    info.append("ae-lock")
            except Exception as e:
                info.append(f"ae-lock-fail:{e}")
            try:
                if hasattr(params, "setAutoWhiteBalanceLock"):
                    params.setAutoWhiteBalanceLock(True)
                    info.append("awb-lock")
            except Exception as e:
                info.append(f"awb-lock-fail:{e}")
            try:
                lo, hi = params.getMinExposureCompensation(), params.getMaxExposureCompensation()
                # bias one step dark to shorten effective shutter on devices
                # where EC couples to exposure time; never below minimum.
                target = max(lo, min(hi, -1 if lo < 0 else lo))
                params.setExposureCompensation(target)
                info.append(f"ec={target}[{lo},{hi}]")
            except Exception as e:
                info.append(f"ec-fail:{e}")
            try:
                ranges = params.getSupportedPreviewFpsRange()
                if ranges:
                    # pick the range with the highest max (usually 30000 = 30fps)
                    best = max([(r[0], r[1]) for r in ranges], key=lambda r: r[1])
                    params.setPreviewFpsRange(best[0], best[1])
                    info.append(f"fps={best}")
            except Exception as e:
                info.append(f"fps-fail:{e}")
            try:
                cam.setParameters(params)
                self._exposure_locked = True
            except Exception as e:
                info.append(f"apply-fail:{e}")
            Logger.info("PPLineCaller: exposure experiment: " + " ".join(info))
        except Exception as e:
            Logger.info(f"PPLineCaller: exposure experiment skipped: {e}")

    def _ensure_focus(self):
        """Lock focus once at startup. Returns (ok, msg).

        Prefers fixed -> infinity -> edof: the lens never moves again, so
        bad hall light can't make it hunt mid-serve. Falls back to
        continuous-video -> continuous-picture -> auto (with an AF trigger
        for one-shot auto) on devices with no locked mode. Safe no-op when
        camera isn't ready.
        """
        if getattr(self, "cam", None) is None:
            return False, "camera not ready"
        cam = getattr(self.cam, "_android_camera", None)
        if cam is None:
            return False, "no _android_camera"
        want = ["fixed", "infinity", "edof"]
        fallback = ["continuous-video", "continuous-picture", "auto", "macro"]
        try:
            params = cam.getParameters()
        except Exception as e:
            return False, f"getParameters: {e}"
        try:
            supported = list(params.getSupportedFocusModes() or [])
        except Exception:
            supported = []
        pick = next((m for m in want if m in supported), None)
        if pick is None and not supported:
            # Some drivers hide the list but accept setFocusMode anyway.
            pick = want[0]
        if pick is None:
            # No locked mode on this device: fall back to continuous AF
            # rather than leaving focus wherever the driver put it.
            pick = next((m for m in fallback if m in supported), None)
        if pick is None:
            msg = f"none of {want + fallback} in {supported}"
            Logger.info(f"PPLineCaller: focus unsupported: {msg}")
            return False, msg
        try:
            params.setFocusMode(pick)
            cam.setParameters(params)
        except Exception as e:
            msg = f"setFocusMode({pick}): {e}"
            Logger.info(f"PPLineCaller: focus failed: {msg}")
            return False, msg
        # One-shot "auto" needs an AF trigger; continuous modes refocus
        # on their own. Null callback = fire and forget.
        if pick == "auto":
            try:
                cam.autoFocus(None)
            except Exception as e:
                Logger.info(f"PPLineCaller: autoFocus trigger skipped: {e}")
        msg = f"{pick} (supported={supported})"
        Logger.info(f"PPLineCaller: focus -> {msg}")
        return True, msg

    def _preview_size(self):
        # The device may not honour the requested resolution; use the preview
        # size the camera actually delivers so the NV21 buffer is decoded right.
        if getattr(self, "_actual_size", None) is None:
            try:
                size = self.cam._android_camera.getParameters().getPreviewSize()
                self._actual_size = (size.width, size.height)
            except Exception:
                self._actual_size = tuple(self.cam.resolution)
        return self._actual_size

    def _read_frame(self):
        if self.cam is None:
            return None
        try:
            buf = self.cam.grab_frame()
        except Exception:
            return None
        if buf is None:
            return None
        try:
            # NV21 -> BGR, same conversion Kivy's read_frame() does
            # (np.fromstring was removed in numpy 2.x, hence manual).
            w, h = self._preview_size()
            n = w * (h + h // 2)
            arr = np.frombuffer(buf, dtype=np.uint8)[:n].reshape((h + h // 2, w))
            frame = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV21)
            if not getattr(self, "_frame_logged", False):
                self._frame_logged = True
                Logger.info(f"PPLineCaller: buf={len(buf)} preview={w}x{h} "
                            f"Y mean={arr[:h].mean():.1f} BGR mean={frame.mean():.1f}")
            return frame
        except Exception as e:
            if not getattr(self, "_decode_err_logged", False):
                self._decode_err_logged = True
                Logger.exception(f"PPLineCaller: frame decode failed: {e}")
                self.status.text = f"Frame decode error: {e}"
            return None

    def say(self, text):
        """Speak without blocking the 30Hz tick: tts.speak() on Android can
        take 1-3s (engine init + utterance), which used to stall the whole
        detection loop and delay the verdict after it was already decided."""
        def _speak():
            try:
                tts.speak(text)
            except Exception:
                pass
        t = threading.Thread(target=_speak, daemon=True)
        self._tts_thread = t
        t.start()

    def _draw_calib_points(self, frame):
        """Draw each tapped corner (numbered) plus joining lines."""
        import numpy as np
        pts = [(int(x), int(y)) for x, y in self.calib_pts]
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False,
                          (0, 255, 255), 2)
        for i, (px, py) in enumerate(pts):
            cv2.circle(frame, (px, py), 12, (0, 255, 255), -1)
            cv2.circle(frame, (px, py), 12, (0, 0, 0), 2)
            cv2.putText(frame, CALIB_SHORT[i], (px + 16, py - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def tick(self, _dt):
        import time
        t0 = time.perf_counter()
        frame = self._read_frame()
        if frame is None:
            return
        self.frame = frame
        self._tick_count += 1
        if self.calibrating:
            self._draw_calib_points(frame)
            self._upload(frame)
            return
        if self.table.H is not None:
            p = self.params
            # Quarter-ROI: only MY quarter (tap-1 slot) is processed. Keeps
            # the 4-corner homography (stable lines) but crops HSV/contours
            # to ~1/4 pixels + top_extra for the incoming ball.
            qpoly = self.table.quarter_poly("server_right")
            det, curr_small = detect_ball_hsv(
                frame, self.tracker.pos, table_poly=qpoly,
                prev_small=self._prev_small, min_area=p["min_area"],
                max_area=p["max_area"], min_circ=p["min_circ"],
                min_circ_streak=p["min_circ_streak"],
                max_aspect=p["max_aspect"],
                min_solidity=p["min_solidity"], min_fill=p["min_fill"],
                min_vertices=int(p["min_vertices"]),
                roi_margin=int(p["roi_margin"]),
                motion_thresh=int(p["motion_thresh"]),
                white_v_min=int(p["white_v_min"]),
                white_s_max=int(p["white_s_max"]))
            if curr_small is not None:
                self._prev_small = curr_small
            pos = self.tracker.update((det[0], det[1]) if det else None)
            # coasted predictions reuse last radius for drawing
            r_draw = int(det[2]) if det else 6
            verdict = self.caller.update(pos)
            if verdict:
                self.status.text = f"{verdict}: {self.caller.reason}"
                if not self._said:
                    self.say("In" if verdict == "IN" else "Fault")
                    self._said = True
            else:
                self._said = False
                if self.caller.state == "SERVE_LIVE":
                    self.status.text = (f"Watching serve... "
                                        f"bounces={len(self.caller.bounces)}")
            # Render throttle: detect every frame, upload overlays at ~15Hz
            # during live rallies (saves flip/cvtColor/tobytes/blit per frame);
            # always render when decided/calibrated so calls are visible.
            live = self.caller.state == "SERVE_LIVE" and not verdict
            should_render = (not live) or (self._tick_count % self._render_every == 0)
            if should_render:
                if pos is not None:
                    x, y = (int(pos[0]), int(pos[1]))
                    cv2.circle(frame, (x, y), max(r_draw, 4), (0, 0, 255), 2)
                # draw MY quarter (thick) + full table (thin) + bounces
                if qpoly is not None:
                    cv2.polylines(frame, [qpoly.astype(int)], True, (0, 255, 0), 3)
                    cv2.putText(frame, "MY",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                cv2.polylines(frame, [self.table.corners.astype(int)], True, (0, 255, 0), 1)
                for i, (bx, by, q) in enumerate(self.caller.bounces):
                    cv2.circle(frame, (int(bx), int(by)), 8, (255, 0, 0), -1)
                    cv2.putText(frame, f"B{i+1}:{q}", (int(bx) + 10, int(by)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                self._upload(frame)
        else:
            self._upload(frame)
        # tick-time log: proves processed FPS holds 30 (33ms budget).
        ms = (time.perf_counter() - t0) * 1000.0
        self._tick_ms_ema = ms if self._tick_count <= 1 else 0.9 * self._tick_ms_ema + 0.1 * ms
        if self._tick_count % 60 == 0:
            Logger.info(f"PPLineCaller: tick avg={self._tick_ms_ema:.1f}ms "
                        f"last={ms:.1f}ms n={self._tick_count} "
                        f"bounces={len(getattr(self.caller, 'bounces', []))}")

    def _upload(self, frame):
        # Upload as RGB: GLES has no BGR texture format, and reuse one texture
        # instead of allocating a new one every frame.
        buf = cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_BGR2RGB).tobytes()
        size = (frame.shape[1], frame.shape[0])
        if self._tex is None or self._tex.size != size:
            self._tex = Texture.create(size=size, colorfmt="rgb")
        self._tex.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
        self.view.texture = self._tex
        self.view.canvas.ask_update()


if __name__ == "__main__":
    ServeApp().run()
