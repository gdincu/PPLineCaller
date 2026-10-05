"""Kivy APK entry: ping-pong doubles serve line-caller. CPU-only (OpenCV + numpy).

Android-only: built as an APK via the GitHub Action (buildozer). There is
no desktop/webcam path.

Flow:
  1. Fix the phone on the side of the table by your half (tripod,
     landscape), framing the whole table.
  2. Pick this phone's side (one phone per side, one quarter each):
       SIDE: server half (B1)   -> judges bounce 1 in server-right only
       SIDE: receiver half (B2) -> judges bounce 2 in receiver-right only
  3. Tap CALIBRATE then tap the 4 table corners IN ORDER:
       1. SERVER-RIGHT (near end, server's right)
       2. SERVER-LEFT  (near end, server's left)
       3. RECEIVER-LEFT (far end, same long edge as 2)
       4. RECEIVER-RIGHT (far end, same long edge as 1)
     Left/right are fixed from the server's perspective facing the net, so
     the order works from the end or the side, either end of the table.
  4. Tap START SERVE, serve, app speaks/shows IN or FAULT.
"""
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.spinner import Spinner

from serve_caller import (TableMapper, ServeCaller, detect_ball_hsv,
                            BallTracker, small_gray)

from plyer import tts

# Ordered calibration prompts (table-centric, server perspective).
CALIB_STEPS = (
    "SERVER-RIGHT (near end, server's right)",
    "SERVER-LEFT (near end, server's left)",
    "RECEIVER-LEFT (far end, same edge as SERVER-LEFT)",
    "RECEIVER-RIGHT (far end, same edge as SERVER-RIGHT)",
)
# Short tags drawn next to each tapped point on the preview.
CALIB_SHORT = ("1 SR", "2 SL", "3 RL", "4 RR")

ROLE_OPTIONS = {
    "SIDE: server half (B1)": "server",
    "SIDE: receiver half (B2)": "receiver",
}


class ServeApp(App):
    def build(self):
        self.table = TableMapper()
        self.caller = ServeCaller(self.table, mode="server")
        self.calib_pts = []
        self.calibrating = False
        # Android capture via Kivy's camera provider (pyjnius -> Camera API),
        # because cv2.VideoCapture has no working backend in the p4a opencv build.
        self.cam = None
        self.tracker = BallTracker()
        self._prev_small = None
        self._tex = None
        self._said = False

        root = BoxLayout(orientation="vertical")
        # allow_stretch + keep_ratio: frame fills the widget (letterboxed),
        # so the preview is as large as possible; taps are mapped through
        # the displayed rect (norm_image_size) in on_tap.
        self.view = Image(allow_stretch=True, keep_ratio=True)
        self.view.bind(on_touch_down=self.on_tap)
        self.status = Label(text="Pick side, tap CALIBRATE, then 4 corners in order",
                            size_hint_y=0.12)

        cfg = BoxLayout(size_hint_y=0.12)
        self.role_spinner = Spinner(text="SIDE: server half (B1)",
                                    values=tuple(ROLE_OPTIONS.keys()))
        cfg.add_widget(self.role_spinner)

        bar = BoxLayout(size_hint_y=0.12)
        b_cal = Button(text="CALIBRATE")
        b_cal.bind(on_press=self.start_calib)
        b_go = Button(text="START SERVE")
        b_go.bind(on_press=self.start_serve)
        b_rst = Button(text="RESET")
        b_rst.bind(on_press=lambda *_: self.start_serve())
        bar.add_widget(b_cal)
        bar.add_widget(b_go)
        bar.add_widget(b_rst)

        root.add_widget(self.view)
        root.add_widget(self.status)
        root.add_widget(cfg)
        root.add_widget(bar)
        Clock.schedule_interval(self.tick, 1.0 / 30.0)
        return root

    # -- config ---------------------------------------------------------
    def _current_mode(self):
        return ROLE_OPTIONS.get(self.role_spinner.text, "server")

    # -- calibration ----------------------------------------------------
    def start_calib(self, *_):
        self.calib_pts = []
        self.calibrating = True
        self.status.text = f"Tap 1/4: {CALIB_STEPS[0]}"

    def on_tap(self, img, touch):
        if not self.calibrating or getattr(self, "frame", None) is None:
            return False
        if not img.collide_point(touch.x, touch.y):
            return False
        h, w = self.frame.shape[:2]
        # touch is in window coords; the Image widget sits above the status
        # label + buttons, so subtract its position to get widget-local
        # coords (origin bottom-left of the widget). NOTE: do NOT use
        # img.to_widget() here: with the default relative=False it returns
        # the point unchanged for plain widgets (no subtraction), which put
        # every marker one widget-height too high.
        lx = touch.x - img.x
        ly = touch.y - img.y
        # The frame is letterboxed inside the widget (keep_ratio); only the
        # centred norm_image_size rect shows the camera image.
        tw, th = img.norm_image_size
        if tw <= 0 or th <= 0:
            return False
        x0 = (img.width - tw) / 2.0
        y0 = (img.height - th) / 2.0
        if not (x0 <= lx <= x0 + tw and y0 <= ly <= y0 + th):
            self.status.text = (f"Tap inside the camera view "
                                f"({len(self.calib_pts)}/4: {CALIB_STEPS[len(self.calib_pts)]})")
            return True
        u = (lx - x0) / tw   # 0 = left edge of frame
        v = (ly - y0) / th   # 1 = top edge of displayed image
        x = u * w
        y = (1.0 - v) * h    # frame row 0 is the top of the camera image
        self.calib_pts.append((x, y))
        n = len(self.calib_pts)
        if n < 4:
            self.status.text = f"Tap {n + 1}/4: {CALIB_STEPS[n]}"
        else:
            srv_r, srv_l, recv_l, recv_r = self.calib_pts
            self.table.set_corners_table_order(srv_r, srv_l, recv_l, recv_r)
            self.caller = ServeCaller(self.table, mode=self._current_mode())
            self.calibrating = False
            self.status.text = "Calibrated. Tap START SERVE."
        return True

    def start_serve(self, *_):
        if self.table.H is None:
            self.status.text = "Calibrate first (4 corners in order)."
            return
        self.caller = ServeCaller(self.table, mode=self._current_mode())
        self.caller.reset()
        self.tracker.reset()
        self._said = False
        self.status.text = f"Watching serve [{self.caller.mode}]..."

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
        except Exception as e:
            self.status.text = f"Camera error: {e}"

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
        try:
            tts.speak(text)
        except Exception:
            pass

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
        frame = self._read_frame()
        if frame is None:
            return
        self.frame = frame
        if self.calibrating:
            self._draw_calib_points(frame)
        elif self.table.H is not None:
            det = detect_ball_hsv(frame, self.tracker.pos,
                                  table_poly=self.table.corners,
                                  prev_small=self._prev_small)
            self._prev_small, _ = small_gray(frame)
            pos = self.tracker.update(det)
            verdict = self.caller.update(pos)
            if pos is not None:
                x, y = (int(pos[0]), int(pos[1]))
                r = int(det[2]) if det else 4
                cv2.circle(frame, (x, y), max(r, 4), (0, 0, 255), 2)
            # draw table outline + bounces
            cv2.polylines(frame, [self.table.corners.astype(int)], True, (0, 255, 0), 2)
            for i, (bx, by, q) in enumerate(self.caller.bounces):
                cv2.circle(frame, (int(bx), int(by)), 8, (255, 0, 0), -1)
                cv2.putText(frame, f"B{i+1}:{q}", (int(bx) + 10, int(by)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if verdict:
                self.status.text = f"[{self.caller.mode}] {verdict}: {self.caller.reason}"
                if not self._said:
                    self.say("In" if verdict == "IN" else "Fault")
                    self._said = True
            else:
                self._said = False
                if self.caller.state == "SERVE_LIVE":
                    self.status.text = (f"Watching [{self.caller.mode}]... "
                                        f"bounces={len(self.caller.bounces)}")
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
