"""Kivy APK entry: ping-pong doubles serve line-caller. CPU-only (OpenCV + numpy).

Flow:
  1. Point phone at table (tripod behind server, elevated, centred).
  2. Tap CALIBRATE then tap the 4 table corners in the preview (TL,TR,BR,BL any order).
  3. Tap START SERVE, serve, app speaks/shows IN or FAULT.

Desktop test: `python app/main.py` (uses webcam). APK: build via GitHub Action.
"""
import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.utils import platform

from serve_caller import TableMapper, ServeCaller, detect_ball_hsv

IS_ANDROID = platform == "android"

try:
    from plyer import tts  # Android TTS; no-op on desktop if missing
except Exception:
    tts = None


class ServeApp(App):
    def build(self):
        self.table = TableMapper()
        self.caller = ServeCaller(self.table)
        self.calib_pts = []
        self.calibrating = False
        # Android: capture via Kivy's camera provider (pyjnius -> Camera API),
        # because cv2.VideoCapture has no working backend in the p4a opencv build.
        self.cam = None
        self.cap = None if IS_ANDROID else cv2.VideoCapture(0)
        self.last_ball = None

        root = BoxLayout(orientation="vertical")
        self.view = Image()
        self.view.bind(on_touch_down=self.on_tap)
        self.status = Label(text="Tap CALIBRATE, then tap 4 table corners", size_hint_y=0.12)
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
        root.add_widget(bar)
        Clock.schedule_interval(self.tick, 1.0 / 30.0)
        return root

    def start_calib(self, *_):
        self.calib_pts = []
        self.calibrating = True
        self.status.text = f"Tapping corners: {len(self.calib_pts)}/4"

    def on_tap(self, img, touch):
        if not self.calibrating or getattr(self, "frame", None) is None:
            return False
        h, w = self.frame.shape[:2]
        # Image widget stretches frame; map touch -> pixel
        x = touch.x / img.width * w
        y = (1.0 - touch.y / img.height) * h
        self.calib_pts.append((x, y))
        self.status.text = f"Tapping corners: {len(self.calib_pts)}/4"
        if len(self.calib_pts) == 4:
            self.table.set_corners(self.calib_pts)
            self.caller = ServeCaller(self.table)
            self.calibrating = False
            self.status.text = "Calibrated. Tap START SERVE."
        return True

    def start_serve(self, *_):
        if self.table.H is None:
            self.status.text = "Calibrate first (4 corners)."
            return
        self.caller.reset()
        self.status.text = "Watching serve..."

    def on_start(self):
        if not IS_ANDROID:
            return
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

    def _read_frame(self):
        if IS_ANDROID:
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
                w, h = self.cam.resolution
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((h + h // 2, w))
                return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV21)
            except Exception:
                return None
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def say(self, text):
        if tts is not None:
            try:
                tts.speak(text)
            except Exception:
                pass

    def tick(self, _dt):
        frame = self._read_frame()
        if frame is None:
            return
        self.frame = frame
        if self.table.H is not None and not self.calibrating:
            det = detect_ball_hsv(frame, self.last_ball)
            self.last_ball = (det[0], det[1]) if det else None
            verdict = self.caller.update(self.last_ball)
            if det:
                x, y, r = (int(det[0]), int(det[1]), int(det[2]))
                cv2.circle(frame, (x, y), max(r, 4), (0, 0, 255), 2)
            # draw table outline + bounces
            cv2.polylines(frame, [self.table.corners.astype(int)], True, (0, 255, 0), 2)
            for i, (bx, by, q) in enumerate(self.caller.bounces):
                cv2.circle(frame, (int(bx), int(by)), 8, (255, 0, 0), -1)
                cv2.putText(frame, f"B{i+1}:{q}", (int(bx) + 10, int(by)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if verdict:
                self.status.text = f"{verdict}: {self.caller.reason}"
                if not getattr(self, "_said", False):
                    self.say("In" if verdict == "IN" else "Fault")
                    self._said = True
            else:
                self._said = False
                if self.caller.state == "SERVE_LIVE":
                    self.status.text = f"Watching... bounces={len(self.caller.bounces)}"
        buf = cv2.flip(frame, 0).tobytes()
        tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        tex.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.view.texture = tex


if __name__ == "__main__":
    ServeApp().run()
