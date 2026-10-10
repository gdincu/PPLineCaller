"""Callback identity and timing for Kivy's Android NV21 provider."""
import threading
import time


def tracked_camera_class(base):
    class TrackedCamera(base):
        def __init__(self, **kwargs):
            self._capture_lock = threading.Lock()
            self.capture_count = 0
            self._capture_time = None
            try:
                super().__init__(**kwargs)
            except Exception:
                # Kivy opens the native camera before applying preview params
                # or creating GL resources. A failed constructor must release
                # that handle immediately so the next mode can reopen it.
                camera = getattr(self, '_android_camera', None)
                if camera is not None:
                    camera.release()
                    self._android_camera = None
                raise

        def _update(self, dt):
            # The app renders decoded NV21 itself. Consume the producer's
            # SurfaceTexture without drawing Kivy's unused FBO/texture preview.
            self._surface_texture.updateTexImage()

        def _on_preview_frame(self, data, camera):
            # Kivy registers this bound method during construction. Serialize
            # its buffer replacement with snapshots without changing ownership
            # or copying every frame at callback time.
            with self._capture_lock:
                super()._on_preview_frame(data, camera)
                self.capture_count += 1
                self._capture_time = time.perf_counter()

        def snapshot(self, last_sequence):
            with self._capture_lock:
                if self.capture_count == last_sequence:
                    return None
                buf = self.grab_frame()
                if buf is None:
                    return None
                return self.capture_count, self._capture_time, buf

    return TrackedCamera
