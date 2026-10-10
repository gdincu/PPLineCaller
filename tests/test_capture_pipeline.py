"""Camera fallbacks, real worker concurrency and preview integration checks."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from test_detection import BASE, app_method, mapper, nv21, scene
from capture_pipeline import (DetectionSession, DetectionWorker, TARGET_FPS,
                              SETUP_PREVIEW_FPS, preview_due, ranked_fps_ranges)
from camera_policy import discover_camera_modes
from android_capture import tracked_camera_class
from serve_caller import BallTracker, ServeCaller


PARAMS = dict(BASE, min_circ=.65, min_circ_streak=.35, max_aspect=4,
              min_solidity=.85, min_fill=.7, min_vertices=6, max_jump=180,
              confirm_streak=1, min_drop=3, min_rise=3)


def raw_frame(y=260):
    image = scene()
    cv2.circle(image, (420, y), 10, (230, 230, 230), -1)
    # Match the production ownership contract: immutable bytes + array view.
    buf = nv21(image).tobytes()
    return buf, 640, 480, np.frombuffer(buf, np.uint8).reshape(720, 640)


def await_result(worker):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        result = worker.take_result()
        if result is not None:
            return result
        time.sleep(.001)
    raise AssertionError('Detection worker did not return a result')


class CameraPolicyTests(unittest.TestCase):
    def test_prefers_fixed_60_then_falls_back_to_supported_30(self):
        ranges = [(15000, 30000), (30000, 30000), (30000, 60000),
                  (60000, 60000), (120000, 120000), (0, 30000)]
        self.assertEqual(ranked_fps_ranges(ranges), [
            (60000, 60000), (30000, 60000), (30000, 30000), (15000, 30000)])
        self.assertEqual(ranked_fps_ranges([(15000, 30000), (30000, 30000)])[0],
                         (30000, 30000))
        self.assertEqual(ranked_fps_ranges([]), [])

    def test_driver_rejection_falls_back_and_readback_reports_actual_range(self):
        cameras = []
        class Camera:
            def __init__(self, index):
                self.index = index
                self.current = (15000, 30000)
                self.size = (1280, 720)
                self.released = False
            def getParameters(self):
                params = SimpleNamespace(fps=self.current, size=self.size)
                params.setRecordingHint = lambda value: None
                params.setPreviewFormat = lambda value: None
                params.getPreviewFormat = lambda: 17
                params.setPreviewSize = lambda w, h: setattr(params, 'size', (w, h))
                params.getPreviewSize = lambda: SimpleNamespace(width=params.size[0], height=params.size[1])
                params.getSupportedPreviewFpsRange = lambda: [
                    (60000, 60000), (30000, 30000), (15000, 30000)]
                params.getSupportedPreviewSizes = lambda: [
                    SimpleNamespace(width=1280, height=720), SimpleNamespace(width=640, height=480)]
                params.setPreviewFpsRange = lambda lo, hi: setattr(params, 'fps', (lo, hi))
                params.getPreviewFpsRange = lambda values: values.__setitem__(slice(None), self.current)
                return params
            def setParameters(self, params):
                if params.fps == (60000, 60000) and params.size == (1280, 720):
                    raise RuntimeError('unsupported at this resolution')
                self.size = params.size
                # Camera zero silently clamps the request; camera one has a
                # genuine 60 FPS mode at a different resolution.
                self.current = (30000, 30000) if self.index == 0 else params.fps
            def release(self):
                self.released = True
        class Info:
            CAMERA_FACING_BACK = 0
        def open_camera(index):
            camera = Camera(index)
            cameras.append(camera)
            return camera
        api = SimpleNamespace(getNumberOfCameras=lambda: 2, open=open_camera,
            getCameraInfo=lambda index, info: setattr(info, 'facing', 0))
        modes = discover_camera_modes(api, Info, lambda msg: None)
        self.assertEqual((modes[0].index, modes[0].size, modes[0].configured),
                         (1, (640, 480), (60000, 60000)))
        self.assertTrue(all(camera.released for camera in cameras))
        self.assertTrue(all(mode.configured == (30000, 30000)
                            for mode in modes if mode.index == 0))

    def test_provider_drain_does_not_restart_sensor_capture(self):
        clock = SimpleNamespace(schedule_interval=Mock(return_value='new-event'))
        camera = SimpleNamespace(_update_ev=Mock(), _update=Mock(), start=Mock(), stop=Mock())
        event = camera._update_ev
        app_method('_schedule_camera_drain', {
            'Clock': clock, 'TARGET_FPS': TARGET_FPS})(SimpleNamespace(cam=camera))
        event.cancel.assert_called_once()
        clock.schedule_interval.assert_called_once_with(camera._update, 1 / TARGET_FPS)
        self.assertEqual(camera._update_ev, 'new-event')
        camera.start.assert_not_called()
        camera.stop.assert_not_called()

    def test_provider_counts_identical_fresh_frames_and_skips_unused_fbo(self):
        class Base:
            def __init__(self):
                self._surface_texture = SimpleNamespace(updateTexImage=Mock())
                self._fbo = SimpleNamespace(draw=Mock())
                self.grabs = 0
            def _on_preview_frame(self, data, camera):
                self.buffer = data
            def grab_frame(self):
                self.grabs += 1
                return self.buffer
        camera = tracked_camera_class(Base)()
        self.assertIsNone(camera.snapshot(0))
        camera._on_preview_frame(b'identical', None)
        one = camera.snapshot(0)
        self.assertIsNone(camera.snapshot(one[0]))
        camera._on_preview_frame(b'identical', None)
        two = camera.snapshot(one[0])
        self.assertEqual((one[0], two[0]), (1, 2))
        self.assertEqual(camera.grabs, 2)
        self.assertEqual(one[2], two[2])
        camera._update(1 / TARGET_FPS)
        camera._surface_texture.updateTexImage.assert_called_once()
        camera._fbo.draw.assert_not_called()


class WorkerTests(unittest.TestCase):
    def test_real_worker_preserves_bounce_at_30_and_60_fps(self):
        for fps in (30, 60):
            with self.subTest(fps=fps):
                worker = DetectionWorker()
                self.addCleanup(worker.close)
                worker.configure(DetectionSession(mapper(), PARAMS, True))
                for i, y in enumerate((240, 270, 250)):
                    worker.submit(raw_frame(y), i / fps)
                    result = await_result(worker)
                    self.assertIsNone(result.error)
                self.assertEqual(result.verdict, 'IN')
                self.assertEqual(len(result.bounces), 1)
                self.assertEqual(worker.stats(), (3, 3, 0))

    def test_busy_worker_replaces_pending_frame_and_discards_old_serve(self):
        entered, release = threading.Event(), threading.Event()
        seen = []
        class SlowSession:
            def process(self, raw, timestamp):
                seen.append(timestamp)
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('test failed to release worker')
                return SimpleNamespace(timestamp=timestamp, verdict='old')
        class NewSession:
            def process(self, raw, timestamp):
                return SimpleNamespace(timestamp=timestamp, verdict='new')
        worker = DetectionWorker()
        self.addCleanup(worker.close)
        self.addCleanup(release.set)
        worker.configure(SlowSession())
        raw = raw_frame()
        worker.submit(raw, 1)
        self.assertTrue(entered.wait(2))
        # These calls return while processing is blocked; only newest waits.
        worker.submit(raw, 2)
        worker.submit(raw, 3)
        self.assertEqual(worker.stats(), (3, 0, 1))
        worker.configure(NewSession())
        worker.submit(raw, 4)
        release.set()
        result = await_result(worker)
        self.assertEqual((result.timestamp, result.verdict), (4, 'new'))
        self.assertEqual(seen, [1])
        worker.close()
        worker.submit(raw, 5)
        self.assertIsNone(worker.take_result())

    def test_processing_errors_are_reported_without_killing_worker(self):
        worker = DetectionWorker()
        self.addCleanup(worker.close)
        worker.configure(SimpleNamespace(process=Mock(side_effect=ValueError('bad frame'))))
        worker.submit(raw_frame(), 0)
        self.assertEqual(await_result(worker).error, 'bad frame')
        worker.configure(DetectionSession(mapper(), PARAMS, True))
        worker.submit(raw_frame(), 1)
        self.assertIsNone(await_result(worker).error)

    def test_session_copies_ui_configuration(self):
        table, params = mapper(), dict(PARAMS)
        session = DetectionSession(table, params, True)
        table.H[:] = 0
        params['min_area'] = 9999
        result = session.process(raw_frame(), 0)
        self.assertIsNotNone(result.pos)
        self.assertEqual(session.params['min_area'], 200)


class PreviewTests(unittest.TestCase):
    def test_live_preview_is_5_fps_independent_of_capture_rate(self):
        for fps in (30, 60):
            last, uploads = None, 0
            for i in range(fps):
                now = i / fps
                if preview_due(now, last, live=True):
                    uploads += 1
                    last = now
            self.assertEqual(uploads, 5)

    def test_start_serve_disables_expensive_diagnostics(self):
        table = mapper()
        app = SimpleNamespace(_auto_tune=None, table=table, params=dict(PARAMS),
            calibrating=False,
            tracker=BallTracker(), caller=ServeCaller(table), assist_mode=True,
            assist_mask_preview=True, assist_btn=SimpleNamespace(text='ASSIST ON'),
            _mask_btn=SimpleNamespace(text='MASK ON'), _configure_detection=Mock(),
            status=SimpleNamespace(text=''))
        app_method('start_serve', {'ServeCaller': ServeCaller})(app)
        self.assertFalse(app.assist_mode)
        self.assertFalse(app.assist_mask_preview)
        self.assertEqual(app.assist_btn.text, 'ASSIST OFF')
        self.assertEqual(app._mask_btn.text, 'MASK OFF')
        app._configure_detection.assert_called_once()

    def test_start_serve_waits_for_recalibration_to_finish(self):
        table = mapper()
        caller = ServeCaller(table)
        app = SimpleNamespace(calibrating=True, _auto_tune=None, table=table,
            caller=caller, params=dict(PARAMS), tracker=BallTracker(),
            assist_btn=SimpleNamespace(text='ASSIST OFF'),
            _mask_btn=SimpleNamespace(text='MASK OFF'),
            _configure_detection=Mock(), status=SimpleNamespace(text=''))
        app_method('start_serve', {'ServeCaller': ServeCaller})(app)
        self.assertIs(app.caller, caller)
        self.assertEqual(caller.state, 'IDLE')
        app._configure_detection.assert_not_called()
        self.assertIn('Finish calibration', app.status.text)

    def test_short_auto_tune_restores_detection(self):
        app = SimpleNamespace(_auto_tune=SimpleNamespace(frames=2),
            _auto_needed=45, _auto_started=0, status=SimpleNamespace(text=''),
            _configure_detection=Mock())
        app_method('_finish_auto_tune')(app)
        self.assertIsNone(app._auto_tune)
        self.assertIsNone(app._auto_started)
        self.assertIn('not enough frames', app.status.text)
        app._configure_detection.assert_called_once()

    def test_tick_without_camera_keeps_permission_error_visible(self):
        table = mapper()
        detector = SimpleNamespace(take_result=lambda: None, stats=lambda: (0, 0, 0))
        app = SimpleNamespace(cam=None, _paused=False, _auto_tune=None,
            _detector=detector, caller=ServeCaller(table), table=table,
            calibrating=False, assist_mode=False, _last_render_time=None,
            _preview_raw=None, _last_result=None, _tick_count=0, _tick_ms_ema=0,
            _stats_time=-2, _stats_counts=(0, 0, 0), _stats_captures=0,
            _camera_index=None, _camera_fps_range=None,
            status=SimpleNamespace(text='Camera permission denied'))
        app._grab_nv21 = lambda: app_method('_grab_nv21')(app)
        app._preview_size = lambda: app_method('_preview_size')(app)
        app_method('tick')(app, 1 / TARGET_FPS)
        self.assertEqual(app.status.text, 'Camera permission denied')

    def test_texture_orientation_and_buffer_reuse(self):
        texture = SimpleNamespace(create=Mock(side_effect=lambda **kwargs: SimpleNamespace(
            size=kwargs['size'], flip_vertical=Mock(), blit_buffer=Mock())))
        app = SimpleNamespace(_tex=None, view=SimpleNamespace(canvas=SimpleNamespace(ask_update=Mock())))
        upload = app_method('_upload', {'Texture': texture})
        frame = np.array([[[0, 0, 255]], [[255, 0, 0]]], np.uint8)
        upload(app, frame)
        upload(app, frame)
        texture.create.assert_called_once()
        app._tex.flip_vertical.assert_called_once()
        buf = app._tex.blit_buffer.call_args.args[0]
        self.assertIsInstance(buf, memoryview)
        self.assertFalse(buf.readonly)
        self.assertEqual(buf.ndim, 1)
        self.assertEqual(list(buf), [255, 0, 0, 0, 0, 255])
        upload(app, np.zeros((4, 4, 3), np.uint8))
        self.assertEqual(texture.create.call_count, 2)
        app._tex.flip_vertical.assert_called_once()


if __name__ == '__main__':
    unittest.main()
