"""Legacy Android camera selection; imports no Android or Kivy modules."""
from dataclasses import dataclass

TARGET_FPS = 60
PREFERRED_SIZE = (1280, 720)
NV21 = 17


def fps_score(fps_range, target=TARGET_FPS):
    low, high = fps_range
    limit = target * 1000
    # Keep advertised ranges intact, including ranges spanning the target.
    return (min(high, limit), min(low, limit), -max(0, high - limit))


def ranked_fps_ranges(ranges, target=TARGET_FPS):
    valid = {(int(r[0]), int(r[1])) for r in ranges
             if 0 < int(r[0]) <= int(r[1]) and int(r[0]) <= target * 1000}
    return sorted(valid, key=lambda r: fps_score(r, target), reverse=True)


def ranked_preview_sizes(sizes):
    valid = {(int(s.width), int(s.height)) for s in sizes
             if s.width > 0 and s.height > 0 and not (s.width % 2 or s.height % 2)}
    # Preserve enough pixels for a distant ball, and prefer 720p within each
    # FPS tier. Only consider tiny previews when the camera has no other sizes.
    usable = {s for s in valid if s[0] >= 640 and s[1] >= 480}
    return sorted(usable or valid, key=size_score, reverse=True)


def size_score(size):
    w, h = size
    return (size == PREFERRED_SIZE, w * h <= 1280 * 720,
            -abs(w / h - 16 / 9), -abs(w * h - 1280 * 720))


def read_fps_range(params):
    actual = [0, 0]
    params.getPreviewFpsRange(actual)
    if not 0 < actual[0] <= actual[1]:
        raise RuntimeError(f"Invalid FPS readback: {actual}")
    return tuple(actual)


def apply_preview_mode(camera, size, fps_range):
    """Apply together before preview starts, then verify driver readback."""
    params = camera.getParameters()
    params.setRecordingHint(True)
    params.setPreviewFormat(NV21)
    params.setPreviewSize(*size)
    params.setPreviewFpsRange(*fps_range)
    camera.setParameters(params)
    actual = camera.getParameters()
    delivered = actual.getPreviewSize()
    if (delivered.width, delivered.height) != size or actual.getPreviewFormat() != NV21:
        raise RuntimeError("Driver changed preview size or NV21 format")
    return read_fps_range(actual)


@dataclass(frozen=True)
class CameraMode:
    index: int
    size: tuple
    requested: tuple
    configured: tuple

    def score(self):
        return (*fps_score(self.configured), *size_score(self.size), -self.index)


def discover_camera_modes(camera_api, camera_info, log):
    """Probe rear cameras one at a time; always release before Kivy opens one.

    Advertised FPS ranges and sizes need not work together. Test combinations
    and rank readback, so a rejected or silently clamped 60 FPS request cannot
    hide a working combination on another resolution or rear camera.
    """
    rear, other = [], []
    for index in range(camera_api.getNumberOfCameras()):
        info = camera_info()
        camera_api.getCameraInfo(index, info)
        (rear if info.facing == camera_info.CAMERA_FACING_BACK else other).append(index)
    modes = []
    for index in rear or other:
        camera = None
        try:
            camera = camera_api.open(index)
            params = camera.getParameters()
            ranges = ranked_fps_ranges(params.getSupportedPreviewFpsRange() or [])
            sizes = ranked_preview_sizes(params.getSupportedPreviewSizes() or [])
            log(f"camera={index} advertised-fps={ranges} preview-sizes={sizes}")
            for fps_range in ranges:
                for size in sizes:
                    try:
                        actual = apply_preview_mode(camera, size, fps_range)
                        mode = CameraMode(index, size, fps_range, actual)
                        modes.append(mode)
                        log(f"camera={index} size={size} requested={fps_range} configured={actual}")
                    except Exception as exc:
                        log(f"camera={index} size={size} fps={fps_range} rejected: {exc}")
        except Exception as exc:
            log(f"camera={index} unavailable: {exc}")
        finally:
            if camera is not None:
                camera.release()
    if not modes:
        raise RuntimeError("No usable camera preview configuration")
    return sorted(modes, key=lambda mode: mode.score(), reverse=True)
