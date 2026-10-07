"""Kivy APK entry: ping-pong doubles serve line-caller. CPU-only (OpenCV + numpy).

Android-only: built as an APK via the GitHub Action (buildozer). There is
no desktop/webcam path.

Flow:
  1. Fix the phone at your end of the table on your right-hand edge
     (tripod, landscape), framing YOUR OWN quadrant only. Each phone's
     own quarter always lands on the same warp slot — no mode setting.
  2. Tap CALIBRATE then tap YOUR quadrant corners IN ORDER from YOUR end:
        1. NEAR-RIGHT outer corner (your end, your right)
        2. NEAR-CENTRE (your end, end line + centre line)
        3. FAR-CENTRE (net + centre line)
        4. FAR-RIGHT (net + sideline / net post)
  3. Tap START SERVE, serve, app speaks/shows IN when a bounce lands
     inside my quarter (+line pad). Anything else stays silent.
"""
import cv2
import numpy as np
import threading
import time
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
                            detect_ball_nv21, BallTracker,
                            debug_contours_for_frame, AutoTuneCollector,
                            expected_ball_area_range)

from plyer import tts

# Ordered calibration prompts: YOUR quadrant only (same order on both
# phones, so each phone's own quarter lands on the tap-1 warp slot).
CALIB_STEPS = (
    "NEAR-RIGHT outer corner (your end, your right)",
    "NEAR-CENTRE (your end, end line + centre line)",
    "FAR-CENTRE (net + centre line)",
    "FAR-RIGHT (net + sideline / net post)",
)
# Short tags drawn next to each tapped point on the preview.
CALIB_SHORT = ("1 NR", "2 NC", "3 FC", "4 FR")

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
    "min_area": 200, "max_area": 4800, "min_circ": 0.65,
    "min_circ_streak": 0.35, "max_aspect": 4.0,
    "min_solidity": 0.85, "min_fill": 0.70, "min_vertices": 6,
    "roi_margin": 40, "motion_thresh": 25, "white_v_min": 150,
    "white_s_max": 60, "max_jump": 180.0, "confirm_streak": 1,
    "min_drop": 3.0, "min_rise": 3.0,
}

# Recommended preset for the reference table view (user image: blue Sponeta
# 6-53i, indoor concrete hall, phone at right edge looking slightly down
# the table). Tighter white + area gates cut the concrete/glare false
# positives seen with defaults while still seeing the ball at the far end.
RECOMMENDED_TABLE_PRESET = {
    "min_area": 200, "max_area": 3200, "min_circ": 0.65,
    "min_circ_streak": 0.35, "max_aspect": 4.0,
    "min_solidity": 0.85, "min_fill": 0.70, "min_vertices": 6,
    "roi_margin": 30, "motion_thresh": 28, "white_v_min": 165,
    "white_s_max": 55, "max_jump": 180.0, "confirm_streak": 1,
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
        self.frame = None
        self._prev_small = None
        self._tex = None
        self._said = False
        # perf: EMA of tick cost + render throttle (preview at ~15Hz during
        # SERVE_LIVE, full rate while calibrating/judging).
        self._tick_count = 0
        self._tick_ms_ema = 0.0
        self._render_every = 2
        self._exposure_locked = False
        self.tuning_open = False
        self._tune_sliders = {}
        self._tune_values = {}
        self._tune_spec = {k: (fmt, is_int) for k, _, _, _, _, fmt, is_int in TUNE_PARAMS}
        # Tuning Assist / Auto-Tune state
        self.assist_mode = False
        self.assist_mask_preview = False
        self._auto_tune = None  # AutoTuneCollector while collecting
        self._auto_needed = 0

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
        # Assist + Auto-Tune row (always visible, compact)
        assist_bar = BoxLayout(size_hint_y=0.10)
        self.assist_btn = Button(text="ASSIST OFF")
        self.assist_btn.bind(on_press=self.toggle_assist)
        b_mask = Button(text="MASK OFF")
        b_mask.bind(on_press=self.toggle_mask_preview)
        self._mask_btn = b_mask
        b_auto = Button(text="AUTO TUNE")
        b_auto.bind(on_press=self.start_auto_tune)
        b_preset = Button(text="TABLE PRESET")
        b_preset.bind(on_press=self.apply_table_preset)
        assist_bar.add_widget(self.assist_btn)
        assist_bar.add_widget(b_mask)
        assist_bar.add_widget(b_auto)
        assist_bar.add_widget(b_preset)
        root.add_widget(assist_bar)
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

    # -- assist / auto-tune ---------------------------------------------
    def toggle_assist(self, *_):
        self.assist_mode = not self.assist_mode
        self.assist_btn.text = "ASSIST ON" if self.assist_mode else "ASSIST OFF"
        if self.assist_mode:
            self.status.text = "Assist ON: yellow=candidates, red=tracked. Tune sliders live."
        else:
            self.status.text = "Assist OFF."

    def toggle_mask_preview(self, *_):
        self.assist_mask_preview = not self.assist_mask_preview
        self._mask_btn.text = "MASK ON" if self.assist_mask_preview else "MASK OFF"
        # The inset is drawn by the assist overlay, so it needs ASSIST on.
        if self.assist_mask_preview and not self.assist_mode:
            self.assist_mode = True
            self.assist_btn.text = "ASSIST ON"
        self.status.text = "Mask preview ON (HSV white mask)" if self.assist_mask_preview else "Mask preview OFF"

    def apply_table_preset(self, *_):
        """One-tap preset tuned for the reference Sponeta hall view."""
        self.params.update(RECOMMENDED_TABLE_PRESET)
        for key, (fmt, _is_int) in self._tune_spec.items():
            if key in RECOMMENDED_TABLE_PRESET:
                self._tune_sliders[key].value = self.params[key]
                self._tune_values[key].text = fmt.format(self.params[key])
        self._apply_live_settings()
        far_a, near_a = expected_ball_area_range(
            self.table.quarter_poly("server_right"),
            self.frame.shape if self.frame is not None else (720, 1280, 3))
        self.status.text = (f"Table preset applied (min_area {RECOMMENDED_TABLE_PRESET['min_area']}, "
                            f"white V {RECOMMENDED_TABLE_PRESET['white_v_min']}). "
                            f"Exp ball {far_a:.0f}-{near_a:.0f}px. Fine-tune or AUTO TUNE next.")
        Logger.info(f"PPLineCaller: table preset applied {RECOMMENDED_TABLE_PRESET}")

    def start_auto_tune(self, *_):
        if self.table.H is None:
            self.status.text = "Calibrate first (4 corners), then AUTO TUNE on empty table."
            return
        if self._auto_tune is not None:
            self.status.text = "Auto-tune already running..."
            return
        # Caller must keep the table empty and the phone still while
        # ~45 frames (~1.5s) are collected.
        self._auto_tune = AutoTuneCollector(
            self.table.quarter_poly("server_right"),
            base_params=dict(self.params))
        self._auto_needed = 45
        self.status.text = "AUTO TUNE: keep table EMPTY & still for 2s... (collecting)"
        Logger.info("PPLineCaller: auto-tune start, collecting 45 frames")

    def _finish_auto_tune(self):
        collector = self._auto_tune
        self._auto_tune = None
        self._auto_needed = 0
        if collector is None or collector.frames < 5:
            self.status.text = "Auto-tune: not enough frames (camera not ready)."
            return
        try:
            suggested, report = collector.suggest()
            Logger.info(f"PPLineCaller: auto-tune report:\n{report}\nframes={collector.frames}")
            # Apply suggested keys live and move sliders so user sees them
            for k, v in suggested.items():
                self.params[k] = v
                if k in self._tune_sliders:
                    self._tune_sliders[k].value = v
                    fmt, _is = self._tune_spec[k]
                    self._tune_values[k].text = fmt.format(v)
            self._apply_live_settings()
            # Show concise status + keep full report in logcat
            changes = ", ".join(f"{k}={v}" for k, v in suggested.items())
            self.status.text = f"Auto-tune done ({collector.frames} fr): {changes}. Check logcat for details. Turn ASSIST ON to verify."
            # Auto-enable assist so user can immediately see the effect
            if not self.assist_mode:
                self.assist_mode = True
                self.assist_btn.text = "ASSIST ON"
        except Exception as e:
            Logger.exception(f"PPLineCaller: auto-tune failed: {e}")
            self.status.text = f"Auto-tune failed: {e}"

    def _draw_assist_overlay(self, frame, qpoly):
        """Yellow candidate contours + stats for live tuning. No-op if not assist."""
        if not self.assist_mode or qpoly is None:
            return
        try:
            p = self.params
            diags, mask = debug_contours_for_frame(
                frame, qpoly,
                roi_margin=int(p["roi_margin"]),
                white_v_min=int(p["white_v_min"]),
                white_s_max=int(p["white_s_max"]))
            # Draw candidates (up to 10 largest)
            for i, d in enumerate(diags[:10]):
                cx, cy = int(d["cx"]), int(d["cy"])
                r = int(max(d["r"], 3))
                is_pass = (d["area"] >= p["min_area"] and d["area"] <= p["max_area"]
                           and d["circ"] >= 0.30)  # rough
                col = (0, 255, 255) if not is_pass else (0, 165, 255)  # yellow vs orange
                cv2.circle(frame, (cx, cy), r, col, 1)
                # area label for the 4 largest
                if i < 4:
                    label = f"{d['area']:.0f} c{d['circ']:.2f}"
                    cv2.putText(frame, label, (cx + r + 2, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            # stats bar at bottom
            n = len(diags)
            far_a, near_a = expected_ball_area_range(qpoly, frame.shape)
            max_a = max((d["area"] for d in diags), default=0)
            motion_txt = ""
            if n:
                motion_txt = f"max noise {max_a:.0f}px"
            else:
                motion_txt = "no white blobs"
            stats = f"ASSIST cands={n} {motion_txt} | exp ball {far_a:.0f}-{near_a:.0f} | min_area {p['min_area']:.0f}"
            cv2.rectangle(frame, (0, frame.shape[0] - 22), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, stats, (6, frame.shape[0] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            if self.assist_mask_preview and mask is not None:
                try:
                    mh, mw = mask.shape[:2]
                    inset_w = min(320, frame.shape[1] // 3)
                    inset_h = int(inset_w * mh / max(mw, 1))
                    small_mask = cv2.resize(mask, (inset_w, inset_h))
                    inset = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
                    # top-right with border
                    x1 = frame.shape[1] - inset_w - 6
                    y1 = 6
                    x2, y2 = x1 + inset_w, y1 + inset_h
                    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)
                    frame[y1:y2, x1:x2] = inset
                    cv2.putText(frame, "HSV mask", (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                except Exception:
                    pass
        except Exception as e:
            Logger.info(f"PPLineCaller: assist draw failed: {e}")

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
        if self.frame is None:
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
                # Quadrant-only taps: MY quarter on every phone.
                nr, nc, fc, fr = self.calib_pts
                self.table.set_corners_quadrant_order(nr, nc, fc, fr)
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
            # Do NOT lock exposure here: AE has not converged yet, so
            # locking now freezes the dark boot frame (black room, only
            # lights visible). Let Android auto-expose first, then apply
            # a brightness-safe stabilization (no AE/AWB lock, EC=0,
            # flexible FPS range) once it has adapted.
            Clock.schedule_once(lambda dt: self._lock_exposure(), 3.0)
            ok, msg = self._ensure_focus()
            Logger.info(f"PPLineCaller: initial focus: {msg}")
            # Warm the TTS engine now (background thread) so the first
            # verdict doesn't pay the 1-3s init cost.
            self.say("Ready")
        except Exception as e:
            self.status.text = f"Camera error: {e}"

    def _lock_exposure(self):
        """Brightness-safe preview stabilization (delayed, best-effort).

        Never locks AE/AWB: locking before AE converges froze a dark boot
        frame. Resets exposure compensation to 0 and requests a FLEXIBLE
        FPS range: among ranges with the highest max (usually 30000 = 30
        fps) pick the one with the LOWEST min (e.g. [15000,30000] over
        [30000,30000]) so dim light may drop FPS to stay bright while
        bright light still gets 30 FPS. FPS is left untouched if no
        flexible range exists. Any failure only logs; focus is handled
        separately in _ensure_focus.
        """
        if self._exposure_locked:
            return
        try:
            cam = getattr(self.cam, "_android_camera", None)
            if cam is None:
                Logger.info("PPLineCaller: no _android_camera, skip exposure stabilize")
                return
            params = cam.getParameters()
            info = []
            try:
                lo, hi = params.getMinExposureCompensation(), params.getMaxExposureCompensation()
                target = max(lo, min(hi, 0))  # neutral exposure
                params.setExposureCompensation(target)
                info.append(f"ec={target}[{lo},{hi}]")
            except Exception as e:
                info.append(f"ec-fail:{e}")
            try:
                ranges = params.getSupportedPreviewFpsRange()
                if ranges:
                    parsed = [(r[0], r[1]) for r in ranges]
                    top_max = max(r[1] for r in parsed)
                    cands = [r for r in parsed if r[1] == top_max]
                    # lowest min among the fastest-max ranges: keeps 30fps
                    # available without forbidding low-light FPS drop.
                    best = min(cands, key=lambda r: r[0])
                    if best[0] == best[1]:
                        # only a fixed high-FPS range exists -> setting it
                        # would force darkness in dim rooms, so skip it.
                        info.append(f"fps-skip-fixed{best}")
                    else:
                        params.setPreviewFpsRange(best[0], best[1])
                        info.append(f"fps={best}")
                else:
                    info.append("fps-none")
            except Exception as e:
                info.append(f"fps-fail:{e}")
            try:
                cam.setParameters(params)
                self._exposure_locked = True
            except Exception as e:
                info.append(f"apply-fail:{e}")
            Logger.info("PPLineCaller: exposure stabilize: " + " ".join(info))
        except Exception as e:
            Logger.info(f"PPLineCaller: exposure stabilize skipped: {e}")

    def _ensure_focus(self):
        """Lock focus once at startup. Returns (ok, msg).

        Prefers fixed -> infinity -> edof: the lens never moves again, so
        bad hall light can't make it hunt mid-serve. Falls back to
        continuous-video -> continuous-picture -> auto (with an AF trigger
        for one-shot auto) on devices with no locked mode. Safe no-op when
        camera isn't ready.
        """
        if self.cam is None:
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

    def _grab_nv21(self):
        """Grab one preview buffer as (buf, w, h, nv21) without decoding.

        Grab and decode are split so tick() can run the ROI-first path:
        Y-plane motion check + ROI-only BGR decode, decoding the full
        frame only when an upload/render is due.
        """
        if self.cam is None:
            return None
        try:
            buf = self.cam.grab_frame()
        except Exception:
            return None
        if buf is None:
            return None
        try:
            w, h = self._preview_size()
            n = w * (h + h // 2)
            arr = np.frombuffer(buf, dtype=np.uint8)[:n].reshape((h + h // 2, w))
            return buf, w, h, arr
        except Exception as e:
            if not getattr(self, "_decode_err_logged", False):
                self._decode_err_logged = True
                Logger.exception(f"PPLineCaller: frame grab failed: {e}")
                self.status.text = f"Frame decode error: {e}"
            return None

    def _decode_full_nv21(self, nv21, w, h, buf=None):
        """Full-frame NV21 -> BGR (same conversion Kivy's read_frame() does)."""
        try:
            frame = cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
            if not getattr(self, "_frame_logged", False):
                self._frame_logged = True
                blen = len(buf) if buf is not None else -1
                Logger.info(f"PPLineCaller: buf={blen} preview={w}x{h} "
                            f"Y mean={nv21[:h].mean():.1f} BGR mean={frame.mean():.1f}")
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
        threading.Thread(target=_speak, daemon=True).start()

    def _draw_calib_points(self, frame):
        """Draw each tapped corner (numbered) plus joining lines."""
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
        t0 = time.perf_counter()
        raw = self._grab_nv21()
        if raw is None:
            return
        _buf, w, h, nv21 = raw
        self._tick_count += 1
        # -- auto-tune collection (empty-table burst) -------------------
        if self._auto_tune is not None:
            frame_at = self._decode_full_nv21(nv21, w, h, _buf)
            if frame_at is not None:
                self.frame = frame_at
                self._auto_tune.add(frame_at)
                n = self._auto_tune.frames
                self.status.text = f"AUTO TUNE: keep still... {n}/{self._auto_needed} frames"
                # small progress bar
                try:
                    bar_w = int(frame_at.shape[1] * n / max(self._auto_needed, 1))
                    cv2.rectangle(frame_at, (0, frame_at.shape[0] - 8),
                                  (bar_w, frame_at.shape[0]), (0, 255, 0), -1)
                except Exception:
                    pass
                # keep minimal overlay so user sees live view
                qp = self.table.quarter_poly("server_right")
                if qp is not None:
                    cv2.polylines(frame_at, [qp.astype(int)], True, (0, 255, 0), 3)
                self._upload(frame_at)
                if n >= self._auto_needed:
                    self._finish_auto_tune()
            return
        if self.calibrating:
            frame = self._decode_full_nv21(nv21, w, h, _buf)
            if frame is None:
                return
            self.frame = frame
            self._draw_calib_points(frame)
            self._upload(frame)
            return
        if self.table.H is None:
            frame = self._decode_full_nv21(nv21, w, h, _buf)
            if frame is None:
                return
            self.frame = frame
            self._upload(frame)
        else:
            p = self.params
            # ROI-first: Y-plane motion check + ROI-only BGR decode.
            # Full-frame BGR is decoded only when an upload/render is due;
            # static ROI-only frames skip colour decode + HSV entirely.
            # self.frame keeps the last full frame for tap mapping/overlays
            # (resolution is fixed, so stale shape still maps correctly).
            qpoly = self.table.quarter_poly("server_right")
            det_kwargs = dict(
                min_area=p["min_area"], max_area=p["max_area"],
                min_circ=p["min_circ"],
                min_circ_streak=p["min_circ_streak"],
                max_aspect=p["max_aspect"],
                min_solidity=p["min_solidity"], min_fill=p["min_fill"],
                min_vertices=int(p["min_vertices"]),
                roi_margin=int(p["roi_margin"]),
                motion_thresh=int(p["motion_thresh"]),
                white_v_min=int(p["white_v_min"]),
                white_s_max=int(p["white_s_max"]))
            frame = None
            # Anticipate the render throttle: non-live states and every
            # Nth live tick need a full frame; other live ticks run ROI-only.
            want_full = (self.caller.state != "SERVE_LIVE"
                         or (self._tick_count % self._render_every == 0))
            if qpoly is None:
                want_full = True
            if want_full:
                frame = self._decode_full_nv21(nv21, w, h, _buf)
                if frame is None:
                    return
                self.frame = frame
                det, curr_small = detect_ball_hsv(
                    frame, self.tracker.pos, table_poly=qpoly,
                    prev_small=self._prev_small, **det_kwargs)
            else:
                det, curr_small = detect_ball_nv21(
                    nv21, w, h, self.tracker.pos, table_poly=qpoly,
                    prev_small=self._prev_small, **det_kwargs)
            if curr_small is not None:
                self._prev_small = curr_small
            pos = self.tracker.update((det[0], det[1]) if det else None)
            # no detection (coasting): fall back to a small radius
            r_draw = int(det[2]) if det else 6
            verdict = self.caller.update(pos)
            if verdict == "IN":
                self.status.text = f"IN: {self.caller.reason}"
                if not self._said:
                    self.say("In")
                    self._said = True
            else:
                self._said = False
                if self.caller.state == "SERVE_LIVE":
                    self.status.text = (f"Watching serve... "
                                        f"bounces={len(self.caller.bounces)}")
            # Render throttle: detect every frame, upload overlays at ~15Hz
            # during live rallies (saves full decode + flip/cvtColor/blit);
            # always render when decided/calibrated so calls are visible.
            live = self.caller.state == "SERVE_LIVE" and not verdict
            should_render = (not live) or (self._tick_count % self._render_every == 0)
            if should_render:
                if frame is None:
                    # Verdict landed on an ROI-only tick: decode the full
                    # frame now from the same buffer for the overlay/upload.
                    frame = self._decode_full_nv21(nv21, w, h, _buf)
                    if frame is None:
                        return
                    self.frame = frame
                if pos is not None:
                    x, y = (int(pos[0]), int(pos[1]))
                    cv2.circle(frame, (x, y), max(r_draw, 4), (0, 0, 255), 2)
                # draw my quadrant (thick outline)
                if qpoly is not None:
                    cv2.polylines(frame, [qpoly.astype(int)], True, (0, 255, 0), 3)
                for i, (bx, by, q) in enumerate(self.caller.bounces):
                    cv2.circle(frame, (int(bx), int(by)), 8, (255, 0, 0), -1)
                    cv2.putText(frame, f"B{i+1}:{q}", (int(bx) + 10, int(by)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Tuning Assist overlay (candidates / expected ball / mask inset)
                try:
                    self._draw_assist_overlay(frame, qpoly)
                except Exception:
                    pass
                self._upload(frame)
        # tick-time log: proves processed FPS holds 30 (33ms budget).
        ms = (time.perf_counter() - t0) * 1000.0
        self._tick_ms_ema = ms if self._tick_count <= 1 else 0.9 * self._tick_ms_ema + 0.1 * ms
        if self._tick_count % 60 == 0:
            Logger.info(f"PPLineCaller: tick avg={self._tick_ms_ema:.1f}ms "
                        f"last={ms:.1f}ms n={self._tick_count} "
                        f"bounces={len(self.caller.bounces)}")

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
