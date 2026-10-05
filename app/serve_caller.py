"""CPU-only serve logic for ping-pong doubles. No GPU / torch / roboflow needed.

Doubles serve rule (ITTF 2.6.3, simplified):
  bounce 1 must be in SERVER's right half, bounce 2 must be in RECEIVER's
  right half (diagonal). Centre line counts as IN (part of right half).

Android-only: runs as an APK (Kivy + OpenCV). No desktop/webcam path.

Camera assumption: phone on tripod either behind the server (end view,
elevated, centred) or on the side of the table. Calibration taps are given
in fixed table-centric order so any viewpoint works:
  1. server-right, 2. server-left, 3. receiver-left, 4. receiver-right
(left/right as seen by the server facing the net - a fixed reference).
We warp the table to a top-down rectangle with a 4-point homography.
After warp:
  - table rect: (0,0) - (W,H), net at y=H/2, centre line at x=W/2
  - server at bottom (y>H/2), receiver at top (y<H/2)
  - server-right  = bottom-right quadrant (server-centric right)
  - receiver-right (diagonal, receiver's own right) = top-left quadrant
    (server-centric left, far end)

Device modes (see ServeCaller):
  - "full": one phone behind the server sees the whole table, judges B1+B2.
  - "server": phone on the side by the server half, judges bounce 1 only.
  - "receiver": phone on the side by the receiver half, ignores the
    server-side bounce and judges the first receiver-side bounce only.
    Use one phone per side so each phone only has to call its own quarter.
"""
from collections import deque
import cv2
import numpy as np

WARP_W, WARP_H = 300, 560  # warped table size (x=width, y=length)


def order_corners(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


class TableMapper:
    """4-click calibration -> homography to top-down view.

    Preferred: set_corners_table_order() with taps in fixed table-centric
    order (server-right, server-left, receiver-left, receiver-right,
    left/right from the server's perspective). Works from behind the
    server, from the opposite end, or from the side of the table.
    Legacy: set_corners() auto-orders 4 unordered taps; only reliable for
    the behind-server end view.
    """

    def __init__(self, corners_bgr_ordered=None):
        self.H = None
        self.corners = None
        if corners_bgr_ordered is not None:
            self.set_corners(corners_bgr_ordered)

    def set_corners_table_order(self, server_right, server_left,
                                receiver_left, receiver_right):
        """Set homography from 4 taps in table-centric order.

        Args are (x, y) camera pixels:
          server_right:   near end, server's right  -> warp (W, H)
          server_left:    near end, server's left   -> warp (0, H)
          receiver_left:  far end, server-side left -> warp (0, 0)
          receiver_right: far end, server-side right-> warp (W, 0)
        """
        src = np.array([server_right, server_left,
                        receiver_left, receiver_right], dtype=np.float32)
        # Clockwise in warp space would be BR, BL, TL, TR; cv2 needs
        # matching src->dst pairs, order among pairs does not matter.
        dst = np.array([[WARP_W, WARP_H], [0, WARP_H],
                        [0, 0], [WARP_W, 0]], dtype=np.float32)
        self.corners = src
        self.H = cv2.getPerspectiveTransform(src, dst)

    def set_corners(self, corners):
        self.corners = order_corners(corners)
        dst = np.array(
            [[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]], dtype=np.float32
        )
        self.H = cv2.getPerspectiveTransform(self.corners, dst)

    def to_table(self, x, y):
        """Camera pixel -> warped table coords."""
        p = np.array([[[float(x), float(y)]]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.H)[0, 0]
        return float(q[0]), float(q[1])

    def inside_table(self, x, y, margin=8):
        tx, ty = self.to_table(x, y)
        return -margin <= tx <= WARP_W + margin and -margin <= ty <= WARP_H + margin

    def quadrant(self, x, y, behind_server=True):
        """Return 'server_right' / 'server_left' / 'receiver_right' / 'receiver_left'.

        Quadrants are fixed in warp space (table-centric, left/right from
        the server's perspective): server side is y>H/2, receiver side
        y<H/2; right is x>W/2. The receiver's own right half is the
        top-left quadrant (diagonally opposite server-right).
        The camera viewpoint (end/side, either end) is encoded in the
        calibration taps, so no flip is applied here. `behind_server` is
        kept for backwards compatibility and ignored.
        """
        tx, ty = self.to_table(x, y)
        is_server_side = ty > WARP_H / 2
        is_table_right = tx > WARP_W / 2
        if is_server_side:
            return "server_right" if is_table_right else "server_left"
        else:
            # receiver faces the other way: their right is table-left
            return "receiver_right" if not is_table_right else "receiver_left"

    def side(self, x, y):
        """Return 'server' if warped y is on the server half, else 'receiver'."""
        _, ty = self.to_table(x, y)
        return "server" if ty > WARP_H / 2 else "receiver"


def detect_ball_hsv(frame_bgr, last_pos=None):
    """Zero-training white/orange ball detector. Returns (x, y, r) or None.

    Tune V thresholds for your hall lighting. Works best with dark background.
    ~3-5 ms on a phone CPU at 640px wide (no GPU needed).
    """
    if frame_bgr is None:
        return None
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    h, w = frame_bgr.shape[:2]
    scale = 640.0 / max(w, 1)
    small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))
    orange1 = cv2.inRange(hsv, np.array([5, 90, 90]), np.array([25, 255, 255]))
    orange2 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([5, 255, 255]))
    mask = cv2.bitwise_or(white, cv2.bitwise_or(orange1, orange2))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 15 or area > 4000:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.45:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        # prefer round + reasonably sized + near last position (continuity)
        score = circularity * 1000.0 / (1.0 + abs(r - 9.0))
        if last_pos is not None:
            dist = np.hypot(cx - last_pos[0] * scale, cy - last_pos[1] * scale)
            score /= 1.0 + dist / 80.0
        if score > best_score:
            best_score = score
            best = (cx / scale, cy / scale, r / scale)
    return best


def is_bounce(traj, min_drop=3.0, min_rise=3.0):
    """Same idea as the pickleball notebook: dy1>0 then dy2<0 (image y down).

    traj: list of (x, y) camera pixels, most recent last. Need >=3 points.
    Thresholds reject jitter.
    """
    if len(traj) < 3:
        return False
    dy1 = traj[-2][1] - traj[-3][1]
    dy2 = traj[-1][1] - traj[-2][1]
    return dy1 > min_drop and dy2 < -min_rise


class ServeCaller:
    """State machine: IDLE -> SERVE_LIVE -> DECIDED. Call reset() per serve.

    Modes:
      "full":     single phone sees the whole table; judges bounce 1 in
                 server_right then bounce 2 in receiver_right.
      "server":   side phone by the server half; judges the FIRST bounce
                 only (must be server_right). Ignores everything after.
      "receiver": side phone by the receiver half; ignores server-side
                 bounces (bounce 1) and judges the first receiver-side
                 bounce (must be receiver_right). Use with a second phone
                 in "server" mode so each phone only calls its own quarter.
    Off-table bounces (outside the calibrated quad) are FAULT in
    "full"/"server" modes, and FAULT in "receiver" mode once the bounce
    is on the receiver side.
    """

    MODES = ("full", "server", "receiver")

    def __init__(self, table: TableMapper, behind_server=True, mode="full"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.table = table
        self.behind_server = behind_server  # legacy, unused (orientation is in taps)
        self.mode = mode
        self.traj = deque(maxlen=12)
        self.bounces = []  # list of (x, y, quadrant)
        self.state = "IDLE"
        self.verdict = None  # "IN" | "FAULT" | None
        self.reason = ""

    def reset(self):
        self.traj.clear()
        self.bounces.clear()
        self.state = "SERVE_LIVE"
        self.verdict = None
        self.reason = "watching serve..."

    def update(self, ball_xy_or_none):
        """Feed one frame's ball centre (x, y) or None. Returns verdict or None."""
        if self.state != "SERVE_LIVE":
            return self.verdict
        if ball_xy_or_none is not None:
            self.traj.append(ball_xy_or_none)
            if is_bounce(list(self.traj)):
                x, y = ball_xy_or_none[0], ball_xy_or_none[1]
                q = self.table.quadrant(x, y)
                on_table = self.table.inside_table(x, y)
                if self.mode == "receiver" and self.table.side(x, y) == "server":
                    # bounce 1 happened on the other phone's half: ignore it
                    # (but keep trajectory continuity for the next bounce).
                    self.traj.clear()
                    self.traj.append(ball_xy_or_none)
                    return self.verdict
                # debounce: ignore same-quadrant double counts within 5 frames
                if not self.bounces or self.bounces[-1][2] != q or not on_table:
                    label = q if on_table else "off_table"
                    self.bounces.append((x, y, label))
                self.traj.clear()
                self.traj.append(ball_xy_or_none)
                self._judge()
        return self.verdict

    def _judge(self):
        if self.mode == "server":
            self._judge_single("server_right")
        elif self.mode == "receiver":
            # only receiver-side bounces reach here (server side ignored above)
            self._judge_single("receiver_right")
        else:
            self._judge_full()

    def _judge_single(self, want):
        q = self.bounces[-1][2]
        if q == want:
            self.verdict, self.state = "IN", "DECIDED"
            self.reason = f"bounce in {q}, OK for {self.mode} phone"
        else:
            self.verdict, self.state = "FAULT", "DECIDED"
            self.reason = f"bounce in {q}, need {want}"

    def _judge_full(self):
        if len(self.bounces) == 1:
            q = self.bounces[0][2]
            if q != "server_right":
                self.verdict, self.state = "FAULT", "DECIDED"
                self.reason = f"1st bounce in {q}, need server_right"
        elif len(self.bounces) >= 2:
            q1, q2 = self.bounces[0][2], self.bounces[1][2]
            if q1 == "server_right" and q2 == "receiver_right":
                self.verdict, self.state = "IN", "DECIDED"
                self.reason = "server_right -> receiver_right, diagonal OK"
            else:
                self.verdict, self.state = "FAULT", "DECIDED"
                self.reason = f"bounces {q1} -> {q2}, need server_right -> receiver_right"
