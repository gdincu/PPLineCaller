"""Deterministic trajectory and generated-frame regressions; no Kivy/device needed."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from serve_caller import (AutoTuneCollector, BallTracker, ServeCaller, TableMapper,
                          bounce_vertex, is_bounce, detect_ball_hsv,
                          detect_ball_nv21, decode_nv21_roi,
                          expected_ball_area_range)

POLY = np.array([[570, 440], [150, 440], [250, 180], [500, 180]], np.float32)
BASE = dict(min_area=200, max_area=4800, white_v_min=150, white_s_max=60,
            roi_margin=40, motion_thresh=25)


def scene(colour=(140, 65, 25), glare=False):
    frame = np.full((480, 640, 3), 35, np.uint8)
    cv2.fillPoly(frame, [POLY.astype(np.int32)], colour)
    if glare:
        cv2.rectangle(frame, (320, 280), (370, 310), (255, 255, 255), -1)
    return frame


def nv21(frame):
    h, w = frame.shape[:2]
    i420 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    n = h * w
    u = i420[n:n + n // 4]
    v = i420[n + n // 4:]
    vu = np.column_stack((v, u)).reshape(-1)
    return np.concatenate((i420[:n], vu)).reshape(h * 3 // 2, w)


def mapper():
    table = TableMapper()
    table.set_corners_quadrant_order(*POLY)
    return table


def tune(frame, base=None, count=45, raw=False):
    c = AutoTuneCollector(POLY, base or BASE)
    rng = np.random.default_rng(91)
    for _ in range(count):
        noisy = np.clip(frame.astype(np.int16) + rng.integers(-1, 2, frame.shape),
                        0, 255).astype(np.uint8)
        if raw:
            yuv = nv21(noisy)
            noisy = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
            c.add(noisy, yuv[:480])
        else:
            c.add(noisy)
    return c.suggest()


def app_method(name, globals_=None):
    """Execute the actual UI method without importing Android/Kivy on the host."""
    tree = ast.parse((ROOT / 'app/main.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ServeApp')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = dict(np=np, cv2=cv2, time=SimpleNamespace(perf_counter=lambda: 1.0),
              Logger=SimpleNamespace(info=lambda *a: None, exception=lambda *a: None),
              detect_ball_nv21=detect_ball_nv21)
    ns.update(globals_ or {})
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / 'app/main.py'), 'exec'), ns)
    return ns[name]


class BounceTests(unittest.TestCase):
    def test_sharp_shallow_and_plateau_on_first_sufficient_rise(self):
        for ys in ([100, 110, 105], [100, 102, 104, 103, 101],
                   [100, 110, 110, 105], [100, 110, 109.5, 106]):
            with self.subTest(ys=ys):
                points = [(350, y) for y in ys]
                self.assertTrue(is_bounce(points))
                self.assertFalse(is_bounce(points[:-1]))

    def test_no_jitter_toss_or_straight_motion(self):
        for ys in ([100, 101, 99, 100, 102, 100], [110, 100, 110],
                   [100, 110, 120, 130], [130, 120, 110, 100],
                   [100, 110, 105, 110, 106]):
            with self.subTest(ys=ys):
                self.assertFalse(is_bounce([(350, y) for y in ys]))

    def test_contact_uses_peak_not_rising_midpoint(self):
        self.assertEqual(bounce_vertex([(300, 280), (310, 310), (320, 290)]), (310, 310))
        self.assertEqual(bounce_vertex([(300, 280), (310, 310), (320, 310), (330, 290)]),
                         (315, 310))

    def test_line_call_uses_contact_not_rising_side(self):
        table = TableMapper()
        table.H = np.eye(3, dtype=np.float32)
        for contact_x, after_x, verdict in ((308, 340, 'IN'), (320, 290, None),
                                             (145, 145, 'IN'), (139, 139, None)):
            caller = ServeCaller(table)
            caller.reset()
            for point in ((contact_x - 5, 320), (contact_x, 350), (after_x, 330)):
                caller.update(point)
            self.assertEqual(caller.verdict, verdict)

    def test_gap_recovers_without_phantom_coast_bounce(self):
        track = BallTracker(max_jump=30)
        caller = ServeCaller(mapper())
        caller.reset()
        # Prediction reaches y=320, correction returns to y=310: still descending.
        for i, y in enumerate([280, 300, None, 310]):
            pos = track.update((350, y) if y is not None else None)
            caller.update(pos if track.observed else None, i / 30, track.track_id)
        self.assertIsNone(caller.verdict)
        self.assertEqual(caller.update(track.update((350, 300)), 4 / 30, track.track_id), 'IN')
        self.assertEqual(len(caller.bounces), 1)

    def test_coast_correction_velocity_uses_real_gap(self):
        track = BallTracker(max_jump=20)
        for point in [(0, 0), (10, 10), None, (20, 15)]:
            track.update(point)
        self.assertEqual(track._vel, (5, 2.5))
        self.assertTrue(track.observed)

    def test_confirmation_requires_consecutive_hits(self):
        track = BallTracker(confirm_streak=2)
        self.assertIsNone(track.update((0, 0)))
        track.update(None)
        self.assertIsNone(track.update((1, 1)))
        self.assertEqual(track.update((2, 2)), (2, 2))

    def test_reacquisition_long_gap_and_stall_do_not_join_tracks(self):
        for kind in ('track', 'gap', 'stall'):
            caller = ServeCaller(mapper())
            caller.reset()
            caller.update((350, 280), 0, 1)
            caller.update((350, 310), 1 / 30, 1)
            if kind == 'gap':
                for i in range(3):
                    caller.update(None, (i + 2) / 30, 1)
            caller.update((350, 290), .4 if kind == 'stall' else .17,
                          2 if kind == 'track' else 1)
            self.assertIsNone(caller.verdict)

    def test_outside_stays_silent_and_later_bounce_can_call(self):
        caller = ServeCaller(mapper())
        caller.reset()
        for y in (130, 150, 140):
            caller.update((100, y))
        self.assertIsNone(caller.verdict)
        for y in (250, 300, 280):
            caller.update((350, y), track_id=2)
        self.assertEqual(caller.verdict, 'IN')


class CalibrationTests(unittest.TestCase):
    def test_lighting_and_casts_keep_visible_ball(self):
        for colour, ball in [((65, 30, 12), (110, 110, 110)),
                             ((140, 65, 25), (220, 220, 220)),
                             ((210, 110, 55), (245, 245, 245)),
                             ((160, 90, 50), (215, 195, 175))]:
            with self.subTest(colour=colour):
                frame = scene(colour)
                params, report = tune(frame, raw=True)
                self.assertTrue(params, report)
                settings = dict(BASE, **params)
                previous = None
                for y in [250, 270]:
                    image = frame.copy()
                    cv2.circle(image, (400, y), 7, ball, -1)
                    det, previous = detect_ball_nv21(nv21(image), 640, 480,
                        table_poly=POLY, prev_small=previous, **settings)
                    self.assertIsNotNone(det, report)
                    self.assertLess(np.hypot(det[0] - 400, det[1] - y), 2)

    def test_glare_does_not_raise_far_ball_area_or_white_v(self):
        plain, _ = tune(scene())
        glare, _ = tune(scene(glare=True))
        far, _ = expected_ball_area_range(POLY, (480, 640, 3))
        self.assertLess(glare['min_area'], far)
        self.assertEqual(plain['white_v_min'], glare['white_v_min'])
        self.assertEqual(plain['min_area'], glare['min_area'])

    def test_geometry_uses_edges_not_bbox_and_scales_with_resolution(self):
        areas = expected_ball_area_range(POLY, (480, 640, 3))
        doubled = expected_ball_area_range(POLY * 2, (960, 1280, 3))
        np.testing.assert_allclose(np.array(areas) * 4, doubled)
        tilted = POLY.copy()
        tilted[:2, 0] -= 100
        self.assertEqual(areas, expected_ball_area_range(tilted, (480, 640, 3)))

    def test_repeated_tuning_is_independent_of_previous_white_area_gates(self):
        first, _ = tune(scene(), base=dict(BASE, min_area=800, white_v_min=250, white_s_max=10))
        second, _ = tune(scene(), base=dict(BASE, **first))
        self.assertEqual(first, second)

    def test_noise_percentile_is_not_diluted_by_pixel_median(self):
        c = AutoTuneCollector(POLY, BASE)
        frame = scene()
        rng = np.random.default_rng(3)
        for i in range(45):
            luma = np.full((480, 640), 80, np.uint8)
            luma[rng.random(luma.shape) < .10] = 92
            c.add(frame, luma)
        params, report = c.suggest()
        self.assertEqual(params['motion_thresh'], 15, report)

    def test_ambient_motion_and_exposure_instability_rejected(self):
        for kind in ('exposure', 'motion'):
            c = AutoTuneCollector(POLY, BASE)
            for i in range(45):
                frame = scene((80 + 100 * (i % 2), 40, 20)) if kind == 'exposure' else scene()
                luma = np.full((480, 640), 60 + (80 * (i % 2) if kind == 'motion' else 0), np.uint8)
                c.add(frame, luma)
            params, report = c.suggest()
            self.assertFalse(params)
            self.assertIn('unstable', report)

    def test_warmup_insufficient_shape_and_neutral_surface(self):
        self.assertFalse(tune(scene(), count=10)[0])
        c = AutoTuneCollector(POLY, BASE)
        for i in range(45):
            c.add(scene((250, 40, 10)) if i < c.WARMUP else scene())
        self.assertTrue(c.suggest()[0])
        c.add(np.zeros((240, 320, 3), np.uint8))
        self.assertFalse(c.suggest()[0])
        neutral, report = tune(scene((130, 130, 130)))
        self.assertNotIn('white_v_min', neutral)
        self.assertNotIn('white_s_max', neutral)
        self.assertNotIn('roi_margin', neutral)
        self.assertIn('unchanged', report)


class FramePipelineTests(unittest.TestCase):
    def test_airborne_approach_is_inside_mask(self):
        image = scene()
        cv2.circle(image, (350, 110), 8, (240, 240, 240), -1)
        for detect in ('bgr', 'nv21'):
            det, _ = (detect_ball_hsv(image, table_poly=POLY, min_area=40)
                      if detect == 'bgr' else detect_ball_nv21(nv21(image), 640, 480,
                                                table_poly=POLY, min_area=40))
            self.assertIsNotNone(det, detect)
            self.assertAlmostEqual(det[1], 110, delta=1)

    def test_slow_ball_keeps_shape_while_static_glare_is_excluded(self):
        previous = None
        last = None
        for y in (260, 261, 262, 264):
            image = scene(glare=True)
            cv2.circle(image, (430, y), 8, (240, 240, 240), -1)
            det, previous = detect_ball_nv21(nv21(image), 640, 480, last,
                table_poly=POLY, prev_small=previous, min_area=80)
            self.assertIsNotNone(det)
            self.assertAlmostEqual(det[0], 430, delta=1)
            last = det[:2]
        static = scene(glare=True)
        _, previous = detect_ball_nv21(nv21(static), 640, 480, table_poly=POLY, min_area=40)
        det, _ = detect_ball_nv21(nv21(static), 640, 480, table_poly=POLY,
                                 prev_small=previous, min_area=40)
        self.assertIsNone(det)

    def test_odd_roi_matches_full_nv21_decode(self):
        image = scene()
        yuv = nv21(image)
        full = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        crop, ox, oy = decode_nv21_roi(yuv, 640, 480, 101, 113, 413, 315)
        np.testing.assert_array_equal(crop, full[oy:oy + crop.shape[0], ox:ox + crop.shape[1]])

    def test_generated_sharp_plateau_shallow_and_occluded_serves(self):
        settings = dict(BASE, **tune(scene())[0])
        for ys in ([220, 250, 280, 260], [220, 280, 280, 260],
                   [270, 272, 274, 273, 271], [220, 250, None, 280, 260]):
            with self.subTest(ys=ys):
                caller, track = ServeCaller(mapper()), BallTracker()
                caller.reset()
                previous = None
                for i, y in enumerate(ys):
                    image = scene()
                    if y is not None:
                        cv2.circle(image, (420, y), 7, (230, 230, 230), -1)
                    det, previous = detect_ball_nv21(nv21(image), 640, 480, track.pos,
                        table_poly=POLY, prev_small=previous, **settings)
                    pos = track.update(det)
                    caller.update(pos if track.observed else None, i / 30, track.track_id)
                self.assertEqual(caller.verdict, 'IN')
                self.assertEqual(len(caller.bounces), 1)

    def test_render_throttle_never_changes_detection_path(self):
        detect = Mock(wraps=detect_ball_nv21)
        clock = iter([0, .001, 1/30, .034, 2/30, .068])
        tick = app_method('tick', {'detect_ball_nv21': detect,
            'time': SimpleNamespace(perf_counter=lambda: next(clock))})
        caller = ServeCaller(mapper())
        caller.reset()
        app = SimpleNamespace(_auto_tune=None, _tick_count=0, calibrating=False,
            table=caller.table, caller=caller, tracker=BallTracker(), params=dict(BASE,
            min_circ=.65, min_circ_streak=.35, max_aspect=4, min_solidity=.85,
            min_fill=.7, min_vertices=6), _last_detection_time=None, _prev_small=None,
            _render_every=2, _said=False, _tick_ms_ema=0, frame=None,
            status=SimpleNamespace(text=''), _decode_full_nv21=Mock(
                side_effect=lambda yuv, w, h, buf: cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)),
            _upload=Mock(), _draw_assist_overlay=Mock(), say=Mock())
        for y in (240, 270, 250):
            image = scene()
            cv2.circle(image, (420, y), 10, (230, 230, 230), -1)
            yuv = nv21(image)
            app._grab_nv21 = lambda: (yuv.tobytes(), 640, 480, yuv)
            tick(app, 1 / 30)
        self.assertEqual(detect.call_count, 3)
        self.assertEqual(app._decode_full_nv21.call_count, 2)
        self.assertEqual(caller.verdict, 'IN')
        app.say.assert_called_once_with('In')

    def test_multiple_camera_resolutions(self):
        for factor in (1, 2, 3):
            poly = POLY * factor
            frame = cv2.resize(scene(), (640 * factor, 480 * factor))
            c = AutoTuneCollector(poly, BASE)
            for _ in range(30):
                c.add(frame)
            params, report = c.suggest()
            self.assertTrue(params, report)
            cv2.circle(frame, (420 * factor, 260 * factor), 7 * factor,
                       (230, 230, 230), -1)
            det, _ = detect_ball_nv21(nv21(frame), 640 * factor, 480 * factor,
                table_poly=poly, **dict(BASE, **params))
            self.assertIsNotNone(det)
            self.assertAlmostEqual(det[0], 420 * factor, delta=2)

    def test_no_false_call_on_noisy_glare_scene(self):
        caller, track = ServeCaller(mapper()), BallTracker()
        caller.reset()
        settings = dict(BASE, **tune(scene(glare=True))[0])
        previous = None
        rng = np.random.default_rng(7)
        for i in range(120):
            image = np.clip(scene(glare=True).astype(np.int16) +
                            rng.integers(-2, 3, (480, 640, 3)), 0, 255).astype(np.uint8)
            # Short isolated glints must not assemble into a trajectory.
            if i % 20 == 0:
                cv2.circle(image, (420, 250), 6, (230, 230, 230), -1)
            det, previous = detect_ball_nv21(nv21(image), 640, 480, track.pos,
                table_poly=POLY, prev_small=previous, **settings)
            pos = track.update(det)
            caller.update(pos if track.observed else None, i / 30, track.track_id)
        self.assertIsNone(caller.verdict)
        self.assertFalse(caller.bounces)

    def test_auto_tune_clears_old_verdict_and_waits_before_new_serve(self):
        caller = ServeCaller(mapper())
        caller.verdict, caller.state = 'IN', 'DECIDED'
        app = SimpleNamespace(calibrating=False, table=caller.table, caller=caller,
            tracker=BallTracker(), params=dict(BASE), _auto_tune=None,
            _prev_small=np.ones((2, 2), np.uint8), status=SimpleNamespace(text=''))
        app_method('start_auto_tune', {'AutoTuneCollector': AutoTuneCollector})(app)
        self.assertIsNone(caller.verdict)
        self.assertEqual(caller.state, 'IDLE')
        self.assertIsNotNone(app._auto_tune)
        app_method('start_serve')(app)
        self.assertEqual(caller.state, 'IDLE')
        self.assertIn('Wait', app.status.text)

    def test_failed_auto_tune_does_not_mutate_live_settings(self):
        c = AutoTuneCollector(POLY, BASE)
        for _ in range(8):
            c.add(scene())
        app = SimpleNamespace(_auto_tune=c, _auto_needed=45, _auto_started=0,
            params=dict(BASE), status=SimpleNamespace(text=''),
            _apply_live_settings=Mock())
        app_method('_finish_auto_tune')(app)
        self.assertEqual(app.params, BASE)
        self.assertIn('rejected', app.status.text)
        app._apply_live_settings.assert_not_called()

    def test_cached_preview_is_not_reprocessed(self):
        yuv = nv21(scene())
        app = SimpleNamespace(cam=SimpleNamespace(grab_frame=lambda: yuv.tobytes()),
            _preview_size=lambda: (640, 480), _last_camera_buffer=None)
        grab = app_method('_grab_nv21')
        self.assertIsNotNone(grab(app))
        self.assertIsNone(grab(app))


if __name__ == '__main__':
    unittest.main()
