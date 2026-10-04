"""CPU-only serve logic for ping-pong doubles. No GPU / torch / roboflow needed.

Doubles serve rule (ITTF 2.6.3, simplified):
  bounce 1 must be in SERVER's right half, bounce 2 must be in RECEIVER's
  right half (diagonal). Centre line counts as IN (part of right half).

Camera assumption: phone on tripod behind server, elevated, centred on
centre-line, landscape. We warp the table to a top-down rectangle with a
4-point homography (one-time tap calibration). After warp:
  - table rect: (0,0) - (W,H), net at y=H/2, centre line at x=W/2
  - server at bottom (y>H/2), receiver at top (y<H/2)
  - server-right  = bottom-right quadrant
  - receiver-right (diagonal) = top-left quadrant
  If you film from the other end, pass behind_server=False.
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
    """4-click calibration -> homography to top-down view."""

    def __init__(self, corners_bgr_ordered=None):
        self.H = None
        self.corners = None
        if corners_bgr_ordered is not None:
            self.set_corners(corners_bgr_ordered)

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
        """Return 'server_right' / 'server_left' / 'receiver_right' / 'receiver_left'."""
        tx, ty = self.to_table(x, y)
        is_server_side = ty > WARP_H / 2
        # Camera behind server looking forward: image-right == table-right for server,
        # but receiver faces the other way so their right is image-left.
        is_right_from_camera = tx > WARP_W / 2
        if not behind_server:
            # viewing from opposite end: flip both axes interpretation
            is_server_side = not is_server_side
            is_right_from_camera = not is_right_from_camera
        # NOTE: from camera view, "right half" of whoever is serving/receiving:
        # server right = server side + camera-right, receiver right = receiver side + camera-left
        if is_server_side:
            return "server_right" if is_right_from_camera else "server_left"
        else:
            return "receiver_right" if not is_right_from_camera else "receiver_left"


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
    """State machine: IDLE -> SERVE_LIVE -> DECIDED. Call reset() per serve."""

    def __init__(self, table: TableMapper, behind_server=True):
        self.table = table
        self.behind_server = behind_server
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
                q = self.table.quadrant(*ball_xy_or_none, behind_server=self.behind_server)
                # debounce: ignore same-quadrant double counts within 5 frames
                if not self.bounces or self.bounces[-1][2] != q:
                    self.bounces.append((ball_xy_or_none[0], ball_xy_or_none[1], q))
                self.traj.clear()
                self.traj.append(ball_xy_or_none)
                self._judge()
        return self.verdict

    def _judge(self):
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
