"""Driver simulations for camera, resolution, FPS and callback selection."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from camera_policy import (CameraMode, NV21, apply_preview_mode, discover_camera_modes,
                           ranked_fps_ranges, ranked_preview_sizes)
from android_capture import tracked_camera_class
from test_detection import app_method


class Parameters:
    def __init__(self, ranges, sizes):
        self.ranges, self.sizes = ranges, sizes
        self.fps, self.size, self.format = (15000, 30000), sizes[0], NV21
        self.hint = False

    def getSupportedPreviewFpsRange(self): return self.ranges
    def getSupportedPreviewSizes(self):
        return [SimpleNamespace(width=w, height=h) for w, h in self.sizes]
    def getPreviewSize(self):
        return SimpleNamespace(width=self.size[0], height=self.size[1])
    def getPreviewFormat(self): return self.format
    def getPreviewFpsRange(self, values): values[:] = self.fps
    def setPreviewSize(self, w, h): self.size = (w, h)
    def setPreviewFpsRange(self, lo, hi): self.fps = (lo, hi)
    def setPreviewFormat(self, value): self.format = value
    def setRecordingHint(self, value): self.hint = value


class Device:
    def __init__(self, ranges=((30000, 30000),), sizes=((1280, 720),), accept=None):
        self.params = Parameters(ranges, sizes)
        self.accept = accept or (lambda params: None)
        self.release = Mock()
        self.attempts = []

    def getParameters(self): return copy.deepcopy(self.params)
    def setParameters(self, params):
        self.attempts.append((params.size, params.fps))
        self.accept(params)
        self.params = params


class Info:
    CAMERA_FACING_BACK = 0


def discover(devices, facing=None):
    api = SimpleNamespace(getNumberOfCameras=lambda: len(devices),
        getCameraInfo=lambda i, info: setattr(info, 'facing', (facing or [0] * len(devices))[i]),
        open=Mock(side_effect=lambda i: devices[i]))
    return discover_camera_modes(api, Info, Mock()), api


class CameraSelectionTests(unittest.TestCase):
    def test_spanning_ranges_are_preserved_and_ranked_below_fixed_60(self):
        self.assertEqual(ranked_fps_ranges([
            (30000, 120000), (30000, 30000), (120000, 120000),
            (60000, 60000), (30000, 60000), (0, 30000), (60000, 60000)]),
            [(60000, 60000), (30000, 60000), (30000, 120000), (30000, 30000)])

    def test_selects_faster_rear_camera_instead_of_index_zero_or_front(self):
        slow = Device()
        fast = Device(ranges=((60000, 60000),))
        front = Device(ranges=((60000, 60000),))
        modes, api = discover([slow, front, fast], [0, 1, 0])
        self.assertEqual(modes[0].index, 2)
        self.assertEqual(modes[0].configured, (60000, 60000))
        self.assertEqual([call.args[0] for call in api.open.call_args_list], [0, 2])
        slow.release.assert_called_once()
        fast.release.assert_called_once()
        front.release.assert_not_called()

    def test_rejected_720p_60_uses_working_vga_60_before_720p_30(self):
        def reject(params):
            if params.size == (1280, 720) and params.fps[1] == 60000:
                raise RuntimeError('unsupported combination')
        device = Device(ranges=((30000, 30000), (60000, 60000)),
                        sizes=((1280, 720), (640, 480)), accept=reject)
        modes, _ = discover([device])
        self.assertEqual((modes[0].size, modes[0].configured),
                         ((640, 480), (60000, 60000)))
        self.assertTrue(any(mode.configured == (30000, 30000) for mode in modes))
        device.release.assert_called_once()

    def test_silent_driver_clamp_does_not_hide_another_working_camera(self):
        def clamp(params): params.fps = (30000, 30000)
        capped = Device(ranges=((60000, 60000),), accept=clamp)
        fast = Device(ranges=((30000, 60000),))
        modes, _ = discover([capped, fast])
        self.assertEqual(modes[0].index, 1)
        self.assertEqual(modes[-1].configured, (30000, 30000))

    def test_rejected_modes_release_camera_and_use_lower_fps(self):
        def reject(params):
            if params.fps[1] == 60000: raise RuntimeError('no 60 FPS')
        device = Device(ranges=((60000, 60000), (15000, 30000)), accept=reject)
        modes, _ = discover([device])
        self.assertEqual(modes[0].configured, (15000, 30000))
        device.release.assert_called_once()

    def test_total_failure_releases_camera_and_reports_error(self):
        device = Device(accept=Mock(side_effect=RuntimeError('busy')))
        with self.assertRaisesRegex(RuntimeError, 'No usable'):
            discover([device])
        device.release.assert_called_once()

    def test_apply_sets_recording_hint_nv21_and_checks_driver_size(self):
        device = Device(ranges=((60000, 60000),))
        self.assertEqual(apply_preview_mode(device, (1280, 720), (60000, 60000)),
                         (60000, 60000))
        self.assertTrue(device.params.hint)
        device.accept = lambda params: setattr(params, 'size', (640, 480))
        with self.assertRaisesRegex(RuntimeError, 'size or NV21'):
            apply_preview_mode(device, (1280, 720), (60000, 60000))

    def test_resolution_policy_keeps_ball_detail_and_prefers_720p(self):
        sizes = [SimpleNamespace(width=w, height=h) for w, h in
                 [(320, 240), (640, 480), (1920, 1080), (1280, 720), (641, 480)]]
        self.assertEqual(ranked_preview_sizes(sizes),
                         [(1280, 720), (640, 480), (1920, 1080)])


class CallbackTests(unittest.TestCase):
    def test_failed_provider_constructor_releases_native_camera(self):
        native = SimpleNamespace(release=Mock())
        class Base:
            def __init__(self):
                self._android_camera = native
                raise RuntimeError('preview parameters rejected')
        with self.assertRaisesRegex(RuntimeError, 'preview parameters rejected'):
            tracked_camera_class(Base)()
        native.release.assert_called_once()

    def make_camera(self):
        class Base:
            def __init__(self): self.buf = None
            def _on_preview_frame(self, data, camera): self.buf = data
            def grab_frame(self): return self.buf
        return tracked_camera_class(Base)()

    def test_identical_fresh_images_are_counted_cached_images_are_skipped(self):
        camera = self.make_camera()
        with patch('android_capture.time.perf_counter', side_effect=[1., 1. + 1 / 60]):
            camera._on_preview_frame(b'identical image', None)
            sequence, timestamp, buf = camera.snapshot(None)
            self.assertEqual((sequence, timestamp), (1, 1.))
            self.assertIsNone(camera.snapshot(sequence))
            camera._on_preview_frame(b'identical image', None)
            second = camera.snapshot(sequence)
        self.assertEqual(second[0], 2)
        self.assertAlmostEqual(second[1] - timestamp, 1 / 60)
        self.assertEqual(buf, second[2])

    def test_snapshot_before_first_frame_and_after_stop_is_empty(self):
        camera = self.make_camera()
        self.assertIsNone(camera.snapshot(None))
        camera._on_preview_frame(b'image', None)
        camera.buf = None
        self.assertIsNone(camera.snapshot(None))

    def test_surface_texture_is_drained_without_unused_fbo_rendering(self):
        camera = self.make_camera()
        camera._surface_texture = SimpleNamespace(updateTexImage=Mock())
        camera._refresh_fbo = Mock()
        camera._update(1 / 60)
        camera._surface_texture.updateTexImage.assert_called_once()
        camera._refresh_fbo.assert_not_called()

    def test_app_uses_callback_sequence_and_capture_timestamp(self):
        camera = self.make_camera()
        camera._on_preview_frame(bytes(640 * 480 * 3 // 2), None)
        app = SimpleNamespace(cam=camera, _last_capture_sequence=None,
                              _preview_size=lambda: (640, 480))
        grab = app_method('_grab_nv21')
        first = grab(app)
        self.assertEqual(first[3].shape, (720, 640))
        self.assertEqual(app._frame_timestamp, camera._capture_time)
        self.assertIsNone(grab(app))
        camera._on_preview_frame(bytes(640 * 480 * 3 // 2), None)
        self.assertIsNotNone(grab(app))


class StartupTests(unittest.TestCase):
    def start_app(self, fail_first=False):
        created, events = [], []
        class Provider:
            def __init__(self, **kwargs):
                self.options = kwargs
                self._android_camera = Device()
                self._release_camera = Mock()
                created.append(self)
                events.append('construct')
            def start(self):
                events.append('start')
                if fail_first and len(created) == 1:
                    raise RuntimeError('preview cannot start')
                self.started = True
        def apply(camera, size, fps):
            events.append('configure')
            return fps
        modes = [CameraMode(1, (1280, 720), (60000, 60000), (60000, 60000)),
                 CameraMode(1, (640, 480), (30000, 30000), (30000, 30000))]
        modules = {'jnius': SimpleNamespace(autoclass=Mock()),
                   'kivy.core.camera.camera_android': SimpleNamespace(CameraAndroid=Provider)}
        app = SimpleNamespace(cam=None, status=SimpleNamespace(text=''),
                              _schedule_camera_drain=Mock(),
                              _ensure_focus=Mock(return_value=(True, 'fixed')), say=Mock())
        init = app_method('_init_camera', {
            'discover_camera_modes': Mock(return_value=modes),
            'tracked_camera_class': lambda base: base, 'apply_preview_mode': apply,
            'Logger': SimpleNamespace(info=Mock(), warning=Mock()),
            'Clock': SimpleNamespace(schedule_once=Mock())})
        with patch.dict(sys.modules, modules):
            init(app)
        return app, created, events

    def test_configures_before_start_and_starts_selected_camera_once(self):
        app, created, events = self.start_app()
        self.assertEqual(events, ['construct', 'configure', 'start'])
        self.assertEqual(created[0].options,
                         {'index': 1, 'resolution': (1280, 720), 'stopped': True})
        self.assertEqual(app._camera_fps_range, (60000, 60000))
        app._schedule_camera_drain.assert_called_once()
        self.assertIs(app.cam, created[0])

    def test_rejected_start_releases_camera_and_tries_next_mode(self):
        app, created, events = self.start_app(fail_first=True)
        self.assertEqual(events, ['construct', 'configure', 'start'] * 2)
        created[0]._release_camera.assert_called_once()
        self.assertIs(app.cam, created[1])
        self.assertEqual(app._camera_fps_range, (30000, 30000))


if __name__ == '__main__':
    unittest.main()
