"""CPU-only serve logic for ping-pong doubles. No GPU / torch / roboflow needed.

Doubles serve rule (ITTF 2.6.3, simplified):
  bounce 1 must be in SERVER's right half, bounce 2 must be in RECEIVER's
  right half (diagonal). Centre line counts as IN (part of right half).

Android-only: runs as an APK (Kivy + OpenCV). No desktop/webcam path.

Camera assumption: phone on tripod on the side of the table by your half,
framing the whole table (wide enough to see faults). Calibration taps are
given in fixed table-centric order so the side viewpoint works:
  1. server-right, 2. server-left, 3. receiver-left, 4. receiver-right
(left/right as seen by the server facing the net - a fixed reference).
We warp the table to a top-down rectangle with a 4-point homography.
After warp:
  - table rect: (0,0) - (W,H), net at y=H/2, centre line at x=W/2
  - server at bottom (y>H/2), receiver at top (y<H/2)
  - server-right  = bottom-right quadrant (server-centric right)
  - receiver-right (diagonal, receiver's own right) = top-left quadrant
    (server-centric left, far end)

Device modes (see ServeCaller, one phone per side):
  - "server": phone on the side by the server half, judges bounce 1 only
    (must be server-right).
  - "receiver": phone on the side by the receiver half, ignores the
    server-side bounce and judges the first receiver-side bounce only
    (must be receiver-right).
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
    left/right from the server's perspective). Works from the side of the
    table (or either end).
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

    def quarter_warp_quad(self, mode):
        """Warp-space rect of the judged quarter. Mode: 'server'|'receiver'."""
        hw, hh = WARP_W / 2.0, WARP_H / 2.0
        if mode == "receiver":
            # receiver-right = top-left (diagonal from server-right)
            return np.array([[0, 0], [hw, 0], [hw, hh], [0, hh]],
                            dtype=np.float32)
        # server-right = bottom-right
        return np.array([[hw, hh], [WARP_W, hh],
                         [WARP_W, WARP_H], [hw, WARP_H]], dtype=np.float32)

    def quarter_poly(self, mode):
        """Camera-pixel polygon of the judged quarter (4x2 float32) or None.

        Keeps the 4-corner calibration (needed for a stable homography) and
        derives the quarter from it via H^-1, so centre/net line tolerance
        stays consistent with quadrant().
        """
        if self.H is None:
            return None
        try:
            h_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            return None
        warp = self.quarter_warp_quad(mode).reshape(1, 4, 2)
        cam = cv2.perspectiveTransform(warp, h_inv.astype(np.float32))[0]
        return np.asarray(cam, dtype=np.float32)

    def quarter_table_rect(self, mode, pad=10.0):
        """Warp-space rect (x0, y0, x1, y1) of the judged quarter + pad.

        Used to judge line balls as IN: the pad expands the wanted quarter
        across the centre/net lines by ~half a line width in warp px.
        """
        q = self.quarter_warp_quad(mode)
        x0, y0 = float(q[:, 0].min()), float(q[:, 1].min())
        x1, y1 = float(q[:, 0].max()), float(q[:, 1].max())
        return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def crop_for_poly(frame_shape, poly, margin=40, top_extra=100):
    """Axis-aligned bbox of poly, expanded for airborne balls.

    margin: full-res px on all sides. top_extra: extra full-res px above the
    bbox (incoming ball is ~46 cm/frame above the table at 30 FPS, roughly
    80-120 image px), so the pre-bounce frame stays inside the crop and the
    track can acquire before the bounce.
    Returns (x0, y0, x1, y1, ox, oy) clipped to the frame.
    """
    h, w = frame_shape[:2]
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x0 = int(max(float(p[:, 0].min()) - margin, 0))
    y0 = int(max(float(p[:, 1].min()) - margin - top_extra, 0))
    x1 = int(min(float(p[:, 0].max()) + margin, w))
    y1 = int(min(float(p[:, 1].max()) + margin, h))
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return 0, 0, w, h, 0, 0
    return x0, y0, x1, y1, x0, y0


def _resize_small(frame_bgr, width=640):
    h, w = frame_bgr.shape[:2]
    # never upscale a small quarter crop: that would cost more px than the
    # full frame and inflate area thresholds. Downscale-only.
    scale = min(1.0, width / max(w, 1))
    if scale >= 1.0:
        return frame_bgr, 1.0
    return cv2.resize(frame_bgr, (max(int(w * scale), 1), max(int(h * scale), 1))), scale


def small_gray(frame_bgr, width=640):
    """Downscaled grayscale frame for inter-frame motion gating."""
    small, scale = _resize_small(frame_bgr, width)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small, scale


def detect_ball_hsv(frame_bgr, last_pos=None, table_poly=None, prev_small=None,
                    min_area=160.0, max_area=4800.0, min_circ=0.65,
                    min_solidity=0.85, min_fill=0.70, min_vertices=6,
                    min_circ_streak=0.35, max_aspect=4.0,
                    roi_margin=40, motion_thresh=25, white_v_min=150,
                    white_s_max=60):
    """Zero-training white/orange ball detector. Returns (det, curr_small).

    det is (x, y, r) in full-res coords or None; curr_small is the cropped
    small gray frame to store as prev_small next tick (avoids a 2nd resize).

    Quarter-ROI: pass the judged quarter polygon (TableMapper.quarter_poly)
    as table_poly. The frame is first cropped to the polygon bbox (+ margin
    + top_extra for the incoming ball), then downscaled to 640px wide, so
    HSV + contours run on ~1/4 the pixels. Areas are normalised to full-res
    px (area_full = area_small / scale^2) so sliders survive crop changes.

    Shape has two paths:
      round  - classic gates (min_circ/min_solidity/min_fill/min_vertices).
      streak - fast-serve motion blur: circ >= min_circ_streak and
               minAreaRect aspect <= max_aspect (2-4 typical at 30 FPS),
               with relaxed solidity/fill/vertices. Round scores higher so
               slow balls still prefer the strict path.
    Tune V thresholds for your hall lighting. Works best with dark background.
    """
    if frame_bgr is None:
        return None, None
    if frame_bgr.ndim == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)

    ox, oy = 0, 0
    crop = frame_bgr
    poly_local = None
    if table_poly is not None:
        p = np.asarray(table_poly, dtype=np.float32).reshape(-1, 2)
        if len(p) >= 3:
            x0, y0, x1, y1, ox, oy = crop_for_poly(frame_bgr.shape, p,
                                                   margin=int(roi_margin))
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                return None, prev_small
            poly_local = p - np.array([ox, oy], dtype=np.float32)

    small, scale = _resize_small(crop)
    h, w = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, np.array([0, 0, white_v_min]),
                        np.array([180, white_s_max, 255]))
    orange1 = cv2.inRange(hsv, np.array([5, 90, 90]), np.array([25, 255, 255]))
    orange2 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([5, 255, 255]))
    mask = cv2.bitwise_or(white, cv2.bitwise_or(orange1, orange2))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if poly_local is not None:
        roi = np.zeros((h, w), dtype=np.uint8)
        pts = (poly_local * scale).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(roi, [pts], 255)
            if roi_margin > 0:
                # dilate in small-px: roi_margin was full-res, convert
                k = max(int(float(roi_margin) * scale), 1)
                # cap kernel so a big margin can't swallow the whole crop
                k = min(k, min(h, w) // 2 or 1)
                roi = cv2.dilate(roi, np.ones((k, k), np.uint8))
            mask = cv2.bitwise_and(mask, roi)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if prev_small is not None and prev_small.shape == gray.shape:
        diff = cv2.absdiff(gray, prev_small)
        _, mot = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
        mot = cv2.dilate(mot, np.ones((5, 5), np.uint8))
        mask = cv2.bitwise_and(mask, mot)
    # else: first frame / bbox changed -> skip motion gate so track can acquire

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0
    inv_scale2 = 1.0 / max(scale * scale, 1e-6)
    for c in contours:
        area_small = cv2.contourArea(c)
        area_full = area_small * inv_scale2
        if area_full < min_area or area_full > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1:
            continue
        circularity = 4 * np.pi * area_small / (peri * peri)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area_small / hull_area if hull_area > 0 else 0.0
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        fill = area_small / (np.pi * r * r)
        nvert = len(cv2.approxPolyDP(c, 0.02 * peri, True))
        (_, _), (bw, bh), _ = cv2.minAreaRect(c)
        aspect = (max(bw, bh) / max(min(bw, bh), 1.0)
                  if bw > 0 and bh > 0 else 1.0)

        is_round = (circularity >= min_circ and solidity >= min_solidity
                    and fill >= min_fill and nvert >= min_vertices)
        # Streaks are elongated: the enclosing CIRCLE is mostly empty
        # (fill ~0.35-0.45), so judge fill against the enclosing RECT
        # (ellipse fills ~0.78 of its rect) instead.
        rect_fill = (area_small / max(bw * bh, 1.0)
                     if bw > 0 and bh > 0 else 0.0)
        is_streak = (circularity >= min_circ_streak and aspect <= max_aspect
                     and solidity >= min(0.70, min_solidity)
                     and (rect_fill >= 0.50 or fill >= 0.30)
                     and nvert >= 3)
        if not (is_round or is_streak):
            continue
        # prefer round + reasonably sized + near last position (continuity);
        # streaks score lower so slow balls keep the strict path.
        r_full = r / scale
        score = circularity * 1000.0 / (1.0 + abs(r_full - 18.0) / 18.0)
        if not is_round:
            score *= 0.6
        if last_pos is not None:
            dist = np.hypot((cx / scale + ox) - last_pos[0],
                            (cy / scale + oy) - last_pos[1])
            score /= 1.0 + dist / 80.0
        if score > best_score:
            best_score = score
            best = (cx / scale + ox, cy / scale + oy, r_full)
    return best, gray


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


class BallTracker:
    """Confirmed-track gate between raw detections and serve logic.

    A raw per-frame detection becomes the accepted track position only if
    it falls within max_jump (full-res px) of the last accepted position; a
    lost or teleporting track restarts acquisition and needs
    confirm_streak consecutive gated hits before it feeds the serve logic.
    The track drops after drop_after_missed frames with no detection. This
    stops single-frame false positives from injecting phantom points into
    bounce detection.

    30 FPS retune: a 50 km/h serve jumps ~46 cm/frame (~150 image px), so
    the default max_jump is 150 (not 80) and confirm_streak defaults to 1
    (a serve is only 3-5 frames; requiring 2 kills fast tracks when the
    quarter-ROI + motion gate already cut false positives). Up to
    coast_frames missed frames are bridged with constant-velocity
    prediction so one dropped detection doesn't split the trajectory and
    hide the bounce flip.
    """

    def __init__(self, max_jump=180.0, confirm_streak=1, drop_after_missed=7,
                 coast_frames=2):
        self.max_jump = float(max_jump)
        self.confirm_streak = int(confirm_streak)
        self.drop_after_missed = int(drop_after_missed)
        self.coast_frames = int(coast_frames)
        self.pos = None  # last accepted (x, y) or None
        self._cand = None
        self._cand_streak = 0
        self._missed = 0
        self._vel = None  # (vx, vy) full-res px/frame

    def reset(self):
        self.pos = None
        self._cand = None
        self._cand_streak = 0
        self._missed = 0
        self._vel = None

    def update(self, det_or_none):
        """Feed a raw (x, y[, r]) detection or None. Returns accepted (x, y) or None."""
        if det_or_none is None:
            self._missed += 1
            if (self.pos is not None and self._vel is not None
                    and self._missed <= self.coast_frames):
                # coast: predict through 1-2 missed frames so the
                # dy-flip bounce check still sees a continuous trajectory.
                vx, vy = self._vel
                if abs(vx) + abs(vy) <= self.max_jump * 2.0:
                    self.pos = (self.pos[0] + vx, self.pos[1] + vy)
                    return self.pos
            if self._missed > self.drop_after_missed:
                self.reset()
            return None
        self._missed = 0
        x, y = float(det_or_none[0]), float(det_or_none[1])
        if self.pos is not None:
            dx, dy = x - self.pos[0], y - self.pos[1]
            if np.hypot(dx, dy) <= self.max_jump:
                self._vel = (dx, dy)
                self.pos = (x, y)
                return self.pos
            self.pos = None  # teleport: restart acquisition below
            self._vel = None
        if (self._cand is not None
                and np.hypot(x - self._cand[0], y - self._cand[1]) <= self.max_jump):
            self._cand_streak += 1
        else:
            self._cand = (x, y)
            self._cand_streak = 1
        if self._cand_streak >= self.confirm_streak:
            if self._cand is not None and self._cand_streak > 1:
                self._vel = (x - self._cand[0], y - self._cand[1])
            self.pos = (x, y)
            self._cand = None
            self._cand_streak = 0
            return self.pos
        return None


class ServeCaller:
    """State machine: IDLE -> SERVE_LIVE -> DECIDED. Call reset() per serve.

    Modes (one phone per side, side of the table):
      "server":   side phone by the server half; judges the FIRST bounce
                 only (must be server_right). Ignores everything after.
      "receiver": side phone by the receiver half; ignores server-side
                 bounces (bounce 1) and judges the first receiver-side
                 bounce (must be receiver_right).
    Off-table bounces (outside the calibrated quad) are FAULT for the
    phone responsible for that half.
    """

    MODES = ("server", "receiver")

    def __init__(self, table: TableMapper, behind_server=True, mode="server",
                 min_drop=3.0, min_rise=3.0, line_pad=10.0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.table = table
        self.behind_server = behind_server  # legacy, unused (orientation is in taps)
        self.mode = mode
        self.min_drop = float(min_drop)
        self.min_rise = float(min_rise)
        self.line_pad = float(line_pad)  # warp-px across centre/net lines counting IN
        self.traj = deque(maxlen=12)
        self.bounces = []  # list of (x, y, quadrant)
        self.state = "IDLE"
        self.verdict = None  # "IN" | "FAULT" | None
        self.reason = ""

    def _bounce_point(self):
        """Refine the bounce to between-frame position in table space.

        At 30 FPS the bounce usually happens *between* frames. The dy-flip
        fires on (pre, post) samples straddling the bounce, so judging the
        post point alone biases the call by up to ~23 cm. We take the camera
        midpoint of traj[-2..-1], map it to table coords, and judge quarter
        membership there (perspective-correct). Returns
        (mx, my, tx, ty) or None if uncalibrated.
        """
        if len(self.traj) < 2 or self.table.H is None:
            return None
        ax, ay = self.traj[-2][:2]
        bx, by = self.traj[-1][:2]
        mx, my = (float(ax) + float(bx)) / 2.0, (float(ay) + float(by)) / 2.0
        try:
            tx, ty = self.table.to_table(mx, my)
        except Exception:
            return None
        return mx, my, tx, ty

    def _inside_wanted(self, tx, ty):
        """True if table point is inside this phone's judged quarter (+pad)."""
        try:
            x0, y0, x1, y1 = self.table.quarter_table_rect(
                self.mode, pad=self.line_pad)
        except Exception:
            return False
        return x0 <= tx <= x1 and y0 <= ty <= y1

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
            if is_bounce(list(self.traj), self.min_drop, self.min_rise):
                refined = self._bounce_point()
                if refined is None:
                    return self.verdict
                mx, my, tx, ty = refined
                q = self.table.quadrant(mx, my)
                on_table = self.table.inside_table(mx, my)
                if self.mode == "receiver" and self.table.side(mx, my) == "server":
                    # bounce 1 happened on the other phone's half: ignore it
                    # (but keep trajectory continuity for the next bounce).
                    self.traj.clear()
                    self.traj.append(ball_xy_or_none)
                    return self.verdict
                # debounce: ignore same-quadrant double counts within 5 frames
                if not self.bounces or self.bounces[-1][2] != q or not on_table:
                    label = q if on_table else "off_table"
                    self.bounces.append((mx, my, label))
                self.traj.clear()
                self.traj.append(ball_xy_or_none)
                self._judge()
        return self.verdict

    def _judge(self):
        if self.mode == "receiver":
            # only receiver-side bounces reach here (server side ignored above)
            self._judge_single("receiver_right")
        else:
            self._judge_single("server_right")

    def _judge_single(self, want):
        # Quarter check in table space (+line pad) is the verdict; the
        # quadrant string is kept for the on-screen label/debounce.
        mx, my, _q = self.bounces[-1]
        try:
            tx, ty = self.table.to_table(mx, my)
            inside = self._inside_wanted(tx, ty)
        except Exception:
            inside = False
        q = self.bounces[-1][2]
        if q == "off_table":
            self.verdict, self.state = "FAULT", "DECIDED"
            self.reason = f"bounce off table, need {want}"
            return
        if inside:
            self.verdict, self.state = "IN", "DECIDED"
            self.reason = f"bounce in {q}, OK for {self.mode} phone"
        else:
            self.verdict, self.state = "FAULT", "DECIDED"
            self.reason = f"bounce in {q}, need {want}"
