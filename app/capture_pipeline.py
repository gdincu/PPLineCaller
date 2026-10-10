"""Camera policy and bounded background detection, independent of Kivy/GL."""
from dataclasses import dataclass
import threading
import time

import numpy as np

from serve_caller import BallTracker, ServeCaller, TableMapper, detect_ball_nv21
from camera_policy import TARGET_FPS, ranked_fps_ranges


LIVE_PREVIEW_FPS = 5
SETUP_PREVIEW_FPS = 15


def preview_due(now, last_render, live=False):
    fps = LIVE_PREVIEW_FPS if live else SETUP_PREVIEW_FPS
    return last_render is None or now - last_render >= 1.0 / fps - 1e-9


@dataclass(frozen=True)
class DetectionResult:
    raw: tuple
    timestamp: float
    pos: object
    radius: int
    verdict: object
    state: str
    reason: str
    bounces: tuple
    processing_ms: float
    error: object = None


class DetectionSession:
    """One worker owns motion history, tracker and caller for one configuration.

    Detection thresholds and tracking/bounce algorithms are unchanged. UI data
    is copied so recalibration and slider edits cannot race a running frame.
    """
    def __init__(self, table, params, live):
        self.table = TableMapper()
        self.table.H = np.array(table.H, copy=True)
        self.table.corners = (None if table.corners is None
                              else np.array(table.corners, copy=True))
        self.poly = self.table.quarter_poly("server_right")
        self.params = dict(params)
        self.tracker = BallTracker(max_jump=params["max_jump"],
                                   confirm_streak=int(params["confirm_streak"]))
        self.caller = ServeCaller(self.table, min_drop=params["min_drop"],
                                  min_rise=params["min_rise"])
        if live:
            self.caller.reset()
        self.previous = None
        self.last_time = None

    def process(self, raw, timestamp):
        started = time.perf_counter()
        _, w, h, nv21 = raw
        if self.last_time is not None and timestamp - self.last_time > 0.20:
            self.tracker.reset()
            self.previous = None
        self.last_time = timestamp
        keys = ("min_area", "max_area", "min_circ", "min_circ_streak",
                "max_aspect", "min_solidity", "min_fill", "min_vertices",
                "roi_margin", "motion_thresh", "white_v_min", "white_s_max")
        det, current = detect_ball_nv21(
            nv21, w, h, self.tracker.pos, table_poly=self.poly,
            prev_small=self.previous, **{k: self.params[k] for k in keys})
        if current is not None:
            self.previous = current
        pos = self.tracker.update(det[:2] if det else None)
        verdict = self.caller.update(
            pos if self.tracker.observed else None,
            timestamp=timestamp, track_id=self.tracker.track_id)
        return DetectionResult(raw, timestamp, pos, int(det[2]) if det else 6,
                               verdict, self.caller.state, self.caller.reason,
                               tuple(self.caller.bounces),
                               (time.perf_counter() - started) * 1000)


class DetectionWorker:
    """One running frame + one replaceable pending frame, with no UI callbacks.

    Generations discard results from an old serve/calibration, even when a frame
    was already running. Results are immutable snapshots, never shared callers.
    """
    def __init__(self):
        self._condition = threading.Condition()
        self._generation = 0
        self._session = None
        self._pending = None
        self._result = None
        self._closed = False
        self.submitted = self.processed = self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ball-detection")
        self._thread.start()

    def configure(self, session):
        with self._condition:
            self._generation += 1
            self._session = session
            self._pending = self._result = None

    def submit(self, raw, timestamp):
        # The camera's bytes buffer is immutable; np.frombuffer is a view into
        # those owned bytes, not a view into a recycled Java callback array.
        with self._condition:
            if self._closed or self._session is None:
                return
            if self._pending is not None:
                self.dropped += 1
            self.submitted += 1
            self._pending = (self._generation, self._session, raw, timestamp)
            self._condition.notify()

    def take_result(self):
        with self._condition:
            result, self._result = self._result, None
            return result

    def stats(self):
        with self._condition:
            return self.submitted, self.processed, self.dropped

    def close(self):
        with self._condition:
            self._closed = True
            self._generation += 1
            self._pending = self._result = None
            self._condition.notify()
        self._thread.join(timeout=0.5)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                generation, session, raw, timestamp = self._pending
                self._pending = None
            try:
                result = session.process(raw, timestamp)
            except Exception as exc:
                result = DetectionResult(raw, timestamp, None, 6, None,
                                         "IDLE", "", (), 0, str(exc))
            with self._condition:
                self.processed += 1
                if generation == self._generation and not self._closed:
                    self._result = result
