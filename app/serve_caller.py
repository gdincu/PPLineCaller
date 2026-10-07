"""CPU-only serve logic for ping-pong doubles. No GPU / torch / roboflow needed.

Doubles serve rule (ITTF 2.6.3, simplified):
  bounce 1 must be in SERVER's right half, bounce 2 must be in RECEIVER's
  right half (diagonal). Centre line counts as IN (part of right half).

Android-only: runs as an APK (Kivy + OpenCV). No desktop/webcam path.

Camera assumption: one phone per judged quarter, on a tripod at your
end of the table on your right-hand edge (phones end up diagonal to each
other), landscape, framing YOUR OWN quadrant only. Calibration taps are
your quadrant corners from YOUR end, same order on both phones:
  1. near-right outer, 2. near-centre, 3. far-centre, 4. far-right.
Each phone's own quarter always lands on the same warp slot
(bottom-right). IN-only: bounce inside my quad (+line pad,
quarter_table_rect(pad=10)) -> IN, else silence (keep watching).
We warp the table to a top-down rectangle with a 4-point homography.
After warp:
  - table rect: (0,0) - (W,H), net at y=H/2, centre line at x=W/2
  - own end at bottom (y>H/2), far end at top (y<H/2)
  - my quarter (tap-1, near-right) = bottom-right warp quadrant

Naming note: quadrant names below ("server_right", ...) are SLOT-based,
not physical — "server_right" always means the tap-1 (near-right) quadrant,
i.e. MY quarter on every phone. They are kept so geometry code stays
unchanged; user-facing strings say "my quarter" instead.
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

    Preferred: set_corners_quadrant_order() with 4 taps around YOUR OWN
    quadrant only (no need to frame the whole table):
      1. near-right outer corner, 2. near-centre (end line + centre line),
      3. far-centre (net + centre), 4. far-right (net + sideline).
    Each phone's own quarter always lands on the warp bottom-right slot
    ("server_right").
    Legacy: set_corners_table_order() (full-table 4 corners) and
    set_corners() (auto-ordered) are kept for backwards compatibility.
    """

    def __init__(self, corners_bgr_ordered=None):
        self.H = None
        self.corners = None
        if corners_bgr_ordered is not None:
            self.set_corners(corners_bgr_ordered)

    def set_corners_quadrant_order(self, near_right_outer, near_centre,
                                    far_centre, far_right):
        """Set homography from 4 taps around YOUR OWN quadrant only.

        Args are (x, y) camera pixels, tapped from your own end:
          near_right_outer: your end, your right table corner -> warp (W, H)
          near_centre: your end, end line + centre line -> warp (hw, H)
          far_centre: net + centre line -> warp (hw, hh)
          far_right: net + sideline (net post) -> warp (W, hh)
        Only this quadrant maps correctly; points far outside it
        extrapolate and are ignored (IN-only mode never faults them).
        """
        hw, hh = WARP_W / 2.0, WARP_H / 2.0
        src = np.array([near_right_outer, near_centre,
                        far_centre, far_right], dtype=np.float32)
        dst = np.array([[WARP_W, WARP_H], [hw, WARP_H],
                        [hw, hh], [WARP_W, hh]], dtype=np.float32)
        self.corners = src
        self.H = cv2.getPerspectiveTransform(src, dst)

    def set_corners_table_order(self, near_right, near_left,
                                 far_left, far_right):
        """Set homography from 4 full-table taps in positional order.

        Args are (x, y) camera pixels, tapped from your own end:
          near_right: your end, your right -> warp (W, H) (MY quarter)
          near_left:  your end, your left  -> warp (0, H)
          far_left:   far end, same edge   -> warp (0, 0)
          far_right:  far end, same edge   -> warp (W, 0)
        Kept for backwards compatibility; quadrant-only calibration above
        is preferred (narrow phones can't frame the whole table).
        """
        src = np.array([near_right, near_left,
                        far_left, far_right], dtype=np.float32)
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

        Names are SLOT-based (see module docstring): 'server_right' always
        means the tap-1 (near-right) quadrant = MY quarter on every phone.
        Geometry: own end is y>H/2, far end y<H/2; right is x>W/2.
        `behind_server` is kept for backwards compatibility and ignored.
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

    def quarter_warp_quad(self, quadrant):
        """Warp-space rect of one quadrant. Quadrant: 'server_right' |
        'server_left' | 'receiver_right' | 'receiver_left' (see quadrant()).
        """
        hw, hh = WARP_W / 2.0, WARP_H / 2.0
        quads = {
            "server_right": [[hw, hh], [WARP_W, hh],
                             [WARP_W, WARP_H], [hw, WARP_H]],
            "server_left": [[0, hh], [hw, hh], [hw, WARP_H], [0, WARP_H]],
            "receiver_right": [[0, 0], [hw, 0], [hw, hh], [0, hh]],
            "receiver_left": [[hw, 0], [WARP_W, 0],
                              [WARP_W, hh], [hw, hh]],
        }
        if quadrant not in quads:
            raise ValueError(f"quadrant must be one of {sorted(quads)}, "
                             f"got {quadrant!r}")
        return np.array(quads[quadrant], dtype=np.float32)

    def quarter_poly(self, quadrant):
        """Camera-pixel polygon of one quadrant (4x2 float32) or None.

        The judged quadrant is always "server_right" (the tap-1 slot = MY
        quarter on every phone). Derived from the 4-corner calibration via
        H^-1, so centre/net line tolerance stays consistent with quadrant().
        """
        if self.H is None:
            return None
        try:
            h_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            return None
        warp = self.quarter_warp_quad(quadrant).reshape(1, 4, 2)
        cam = cv2.perspectiveTransform(warp, h_inv.astype(np.float32))[0]
        return np.asarray(cam, dtype=np.float32)

    def quarter_table_rect(self, quadrant, pad=10.0):
        """Warp-space rect (x0, y0, x1, y1) of one quadrant + pad.

        Used to judge line balls as IN: the pad expands the wanted quadrant
        across the centre/net lines by ~half a line width in warp px.
        """
        q = self.quarter_warp_quad(quadrant)
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


def _build_roi_mask(h, w, poly_local, scale, roi_margin):
    """Small-res ROI mask for a crop, or None if no polygon."""
    if poly_local is None:
        return None
    roi = np.zeros((h, w), dtype=np.uint8)
    pts = (np.asarray(poly_local, dtype=np.float32) * scale).astype(np.int32)
    if len(pts) < 3:
        return None
    cv2.fillPoly(roi, [pts], 255)
    if roi_margin > 0:
        # dilate in small-px: roi_margin was full-res, convert
        k = max(int(float(roi_margin) * scale), 1)
        # cap kernel so a big margin can't swallow the whole crop
        k = min(k, min(h, w) // 2 or 1)
        roi = cv2.dilate(roi, np.ones((k, k), np.uint8))
    return roi


def _motion_mask(gray, prev_small, motion_thresh):
    """Dilated binary motion mask, or None if shapes mismatch (acquire)."""
    if prev_small is None or prev_small.shape != gray.shape:
        return None
    diff = cv2.absdiff(gray, prev_small)
    _, mot = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
    mot = cv2.dilate(mot, np.ones((5, 5), np.uint8))
    return mot


def _score_mask_contours(mask, scale, ox, oy, last_pos,
                         min_area, max_area, min_circ, min_solidity,
                         min_fill, min_vertices,
                         min_circ_streak, max_aspect):
    """Shared contour scoring for BGR and NV21-ROI paths. Returns best det."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
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
    return best


def _align_roi_for_nv21(x0, y0, x1, y1, w, h):
    """Even-align an ROI bbox for NV21 chroma subsampling (UV is half-res).

    x is byte-interleaved VU so both edges must be even; y UV rows are
    y//2 so height must be even. Returns (x0a, y0a, x1a, y1a) or None if
    the ROI is too small to decode.
    """
    x0a = max((int(x0) // 2) * 2, 0)
    x1a = min(((int(x1) + 1) // 2) * 2, w)
    y0a = max(int(y0), 0)
    y1a = min(int(y1), h)
    if y1a - y0a >= 2 and (y1a - y0a) % 2 == 1:
        # shrink by one row to keep height even (prefer keeping y0)
        if y1a < h:
            pass  # y1a stays, adjust below by -1
        y1a -= 1
    if x1a - x0a < 4 or y1a - y0a < 4:
        return None
    return x0a, y0a, x1a, y1a


def decode_nv21_roi(nv21, w, h, x0, y0, x1, y1):
    """Decode only an ROI of an NV21 buffer to BGR.

    nv21: uint8 array shaped (h + h//2, w). Returns (bgr_roi, ox, oy)
    with (ox, oy) = full-res origin, or (None, x0, y0) if too small.
    """
    a = _align_roi_for_nv21(x0, y0, x1, y1, w, h)
    if a is None:
        return None, int(x0), int(y0)
    x0a, y0a, x1a, y1a = a
    roi_h, roi_w = y1a - y0a, x1a - x0a
    y_plane = nv21[:h, :]
    vu_plane = nv21[h:, :]
    y_roi = y_plane[y0a:y1a, x0a:x1a]
    # UV rows cover 2 luma rows each
    vu_roi = vu_plane[(y0a // 2):(y1a // 2), x0a:x1a]
    expect_vu_rows = roi_h // 2
    if (y_roi.shape != (roi_h, roi_w)
            or vu_roi.shape != (expect_vu_rows, roi_w)):
        return None, int(x0), int(y0)
    chunk = np.vstack([y_roi, vu_roi])
    bgr = cv2.cvtColor(chunk, cv2.COLOR_YUV2BGR_NV21)
    return bgr, x0a, y0a


def y_gray_small_from_nv21(nv21, w, h, x0, y0, x1, y1, width=640):
    """Downscaled gray ROI straight from the NV21 Y (luma) plane.

    No BGR/HSV decode needed — this feeds the motion-first check (c).
    Uses the same even-aligned bbox as decode_nv21_roi so the scale and
    prev_small shape match the colour ROI path. Returns
    (gray_small, scale, ox, oy) or (None, 1.0, x0, y0) if too small.
    """
    a = _align_roi_for_nv21(x0, y0, x1, y1, w, h)
    if a is None:
        return None, 1.0, int(x0), int(y0)
    x0a, y0a, x1a, y1a = a
    y_roi = nv21[:h, :][y0a:y1a, x0a:x1a]
    if y_roi.size == 0:
        return None, 1.0, x0a, y0a
    rh, rw = y_roi.shape[:2]
    scale = min(1.0, width / max(rw, 1))
    if scale >= 1.0:
        return y_roi.copy(), 1.0, x0a, y0a
    small = cv2.resize(y_roi, (max(int(rw * scale), 1),
                               max(int(rh * scale), 1)))
    return small, scale, x0a, y0a


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
            x0, y0, x1, y1, _ox, _oy = crop_for_poly(frame_bgr.shape, p,
                                                      margin=int(roi_margin))
            # Even-align the bbox exactly like the NV21 ROI path, so both
            # paths share one scale/prev_small shape and the motion gate
            # stays valid when tick() alternates between them.
            fh, fw = frame_bgr.shape[:2]
            a = _align_roi_for_nv21(x0, y0, x1, y1, fw, fh)
            if a is None:
                return None, prev_small
            x0, y0, x1, y1 = a
            ox, oy = x0, y0
            crop = frame_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                return None, prev_small
            poly_local = p - np.array([ox, oy], dtype=np.float32)

    small, scale = _resize_small(crop)
    h, w = small.shape[:2]
    # Motion FIRST (c): gray + absdiff before any HSV work. Static frames
    # with no active track skip HSV/threshold/contours entirely.
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    roi = _build_roi_mask(h, w, poly_local, scale, int(roi_margin))
    mot = _motion_mask(gray, prev_small, int(motion_thresh))
    if (mot is not None and last_pos is None):
        gated = cv2.bitwise_and(mot, roi) if roi is not None else mot
        # ~0.15% of ROI pixels moving (~150 small-px): ball motion always
        # exceeds this; sensor noise / AE wobble does not.
        need = max(30, int(0.0015 * h * w))
        if int(cv2.countNonZero(gated)) < need:
            return None, gray
    # else: active track (last_pos set) -> always run full pipeline so a
    # slow/hanging ball near the track is never skipped; first frame /
    # bbox changed (mot None) -> run full pipeline to acquire.

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, np.array([0, 0, white_v_min]),
                        np.array([180, white_s_max, 255]))
    orange1 = cv2.inRange(hsv, np.array([5, 90, 90]), np.array([25, 255, 255]))
    orange2 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([5, 255, 255]))
    mask = cv2.bitwise_or(white, cv2.bitwise_or(orange1, orange2))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)

    if mot is not None:
        mask = cv2.bitwise_and(mask, mot)
    # else: first frame / bbox changed -> skip motion gate so track can acquire

    best = _score_mask_contours(mask, scale, ox, oy, last_pos,
                                min_area, max_area, min_circ, min_solidity,
                                min_fill, min_vertices,
                                min_circ_streak, max_aspect)
    return best, gray


def detect_ball_nv21(nv21, w, h, last_pos=None, table_poly=None,
                     prev_small=None,
                     min_area=160.0, max_area=4800.0, min_circ=0.65,
                     min_solidity=0.85, min_fill=0.70, min_vertices=6,
                     min_circ_streak=0.35, max_aspect=4.0,
                     roi_margin=40, motion_thresh=25, white_v_min=150,
                     white_s_max=60):
    """ROI-first NV21 detector: Y-plane motion check before any BGR decode.

    (b)+(c) combined: the ROI gray comes straight from the NV21 luma plane
    (no full-frame BGR decode). Static frames with no active track return
    (None, gray) without decoding colour at all; moving frames decode only
    the ROI bbox to BGR for HSV/contours. Detections are in the same
    full-res coords as detect_ball_hsv.

    nv21: uint8 array shaped (h + h//2, w). Falls back to (None, prev_small)
    if the ROI is degenerate.
    """
    if nv21 is None:
        return None, prev_small
    if table_poly is None:
        return None, prev_small
    p = np.asarray(table_poly, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3:
        return None, prev_small
    x0, y0, x1, y1, _ox, _oy = crop_for_poly((h, w), p,
                                             margin=int(roi_margin))
    gray, scale, ox, oy = y_gray_small_from_nv21(nv21, w, h, x0, y0, x1,
                                                 y1, width=640)
    if gray is None:
        return None, prev_small
    gh, gw = gray.shape[:2]
    poly_local = p - np.array([ox, oy], dtype=np.float32)
    roi = _build_roi_mask(gh, gw, poly_local, scale, int(roi_margin))
    mot = _motion_mask(gray, prev_small, int(motion_thresh))
    if mot is not None and last_pos is None:
        gated = cv2.bitwise_and(mot, roi) if roi is not None else mot
        need = max(30, int(0.0015 * gh * gw))
        if int(cv2.countNonZero(gated)) < need:
            return None, gray
    roi_bgr, ox2, oy2 = decode_nv21_roi(nv21, w, h, x0, y0, x1, y1)
    if roi_bgr is None:
        return None, gray
    # Resize with the SAME scale as the Y path so HSV geometry matches the
    # motion gate exactly (avoids a 2nd scale computation drifting by 1px).
    if scale >= 1.0:
        small = roi_bgr
    else:
        rh, rw = roi_bgr.shape[:2]
        small = cv2.resize(roi_bgr, (max(int(rw * scale), 1),
                                     max(int(rh * scale), 1)))
    sh, sw = small.shape[:2]
    if (sh, sw) != (gh, gw):
        # 1px rounding drift between Y and BGR resizes (odd ROI dims):
        # rebuild masks at colour size; motion gate is skipped this frame
        # rather than comparing mismatched shapes.
        roi = _build_roi_mask(sh, sw, poly_local, scale, int(roi_margin))
        mot = None
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, white_v_min]),
                        np.array([180, white_s_max, 255]))
    orange1 = cv2.inRange(hsv, np.array([5, 90, 90]), np.array([25, 255, 255]))
    orange2 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([5, 255, 255]))
    mask = cv2.bitwise_or(white, cv2.bitwise_or(orange1, orange2))
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if roi is not None:
        mask = cv2.bitwise_and(mask, roi)
    if mot is not None:
        mask = cv2.bitwise_and(mask, mot)
    best = _score_mask_contours(mask, scale, ox, oy, last_pos,
                                min_area, max_area, min_circ, min_solidity,
                                min_fill, min_vertices,
                                min_circ_streak, max_aspect)
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
        self._real = None  # last accepted REAL detection (never a coast guess)
        self._gap = 0  # frames elapsed since _real (0 when pos is real)

    def reset(self):
        self.pos = None
        self._cand = None
        self._cand_streak = 0
        self._missed = 0
        self._vel = None
        self._real = None
        self._gap = 0

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
                    self._gap += 1
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
                self._real = (x, y)
                self._gap = 0
                return self.pos
            if self._real is not None:
                # Post-coast correction: if the missed frame straddled the
                # bounce, the coast guess kept going DOWN while the real ball
                # came back UP, so the detection lands far from the guess.
                # Accept it against the last REAL point (wider window) and
                # recompute per-frame velocity over the gap, instead of
                # declaring a teleport and restarting the track (which hid
                # the bounce for 1-2 extra frames).
                dr = np.hypot(x - self._real[0], y - self._real[1])
                if dr <= self.max_jump * 1.25:
                    n = max(self._gap + 1, 1)
                    self._vel = ((x - self._real[0]) / n, (y - self._real[1]) / n)
                    self.pos = (x, y)
                    self._real = (x, y)
                    self._gap = 0
                    return self.pos
            self.pos = None  # teleport: restart acquisition below
            self._vel = None
            self._real = None
            self._gap = 0
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
            self._real = (x, y)
            self._gap = 0
            self._cand = None
            self._cand_streak = 0
            return self.pos
        return None


class ServeCaller:
    """State machine: IDLE -> SERVE_LIVE -> DECIDED. Call reset() per serve.

    IN-only: confirms IN when a bounce lands inside MY quarter
    (+line pad, centre/net lines count IN). Anything else stays silent
    (no FAULT verdict) — keep watching. `want` defaults to "server_right"
    (the tap-1 warp slot = MY quarter on every phone).
    """

    QUADRANTS = ("server_right", "server_left",
                 "receiver_right", "receiver_left")

    def __init__(self, table: TableMapper, want="server_right",
                 behind_server=True, min_drop=3.0, min_rise=3.0,
                 line_pad=10.0):
        if want not in self.QUADRANTS:
            raise ValueError(f"want must be one of {self.QUADRANTS}, got {want!r}")
        self.table = table
        self.behind_server = behind_server  # legacy, unused (orientation is in taps)
        self.want = want
        self.min_drop = float(min_drop)
        self.min_rise = float(min_rise)
        self.line_pad = float(line_pad)  # warp-px across centre/net lines counting IN
        self.traj = deque(maxlen=12)
        self.bounces = []  # list of (x, y, quadrant)
        self.state = "IDLE"
        self.verdict = None  # "IN" | None (IN-only: no FAULT verdict)
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
        """True if table point is inside this phone's quarter (+pad)."""
        try:
            x0, y0, x1, y1 = self.table.quarter_table_rect(
                self.want, pad=self.line_pad)
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
        """Feed one frame's ball centre (x, y) or None. Returns "IN" or None."""
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
                # debounce: ignore same-quadrant double counts within 5 frames
                if not self.bounces or self.bounces[-1][2] != q or not on_table:
                    label = q if on_table else "off_table"
                    self.bounces.append((mx, my, label))
                self.traj.clear()
                self.traj.append(ball_xy_or_none)
                self._judge()
        return self.verdict

    def _judge(self):
        self._judge_single(self.want)

    def _judge_single(self, want):
        # IN-only: inside my quad (+line pad) -> IN, else stay silent and
        # keep watching. Outside/off-table bounces never decide.
        mx, my, _q = self.bounces[-1]
        try:
            tx, ty = self.table.to_table(mx, my)
            inside = self._inside_wanted(tx, ty)
        except Exception:
            inside = False
        if inside:
            self.verdict, self.state = "IN", "DECIDED"
            self.reason = "bounce in my quarter: IN"
        else:
            self.verdict = None
            self.reason = "watching serve..."
