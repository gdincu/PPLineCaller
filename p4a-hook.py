"""Preserve SDL's original error if native library initialization fails.

Native libraries and C++/OpenMP runtimes all come from the pinned NDK r28c.
Do not swap runtimes from an unrelated NDK or rewrite ELF files after linking.
"""

import re
from pathlib import Path

_FINISH_LOAD = "            mActivity.finishLoad();"
_GUARDED_FINISH_LOAD = _FINISH_LOAD + """
            // PPLineCaller: retain SDL's original native-library error dialog.
            if (SDLActivity.mBrokenLibraries) {
                Log.e(TAG, "Native library initialization failed; see the SDL error dialog.");
                return;
            }"""


def _guard_sdl_startup(dist_dir):
    """Stop before nativeSetenv if SDL failed to load the native libraries.

    finishLoad() catches loader failures and displays their original message,
    but p4a's UnpackFilesTask otherwise continues into unresolved JNI calls.
    Patch the generated Java before Gradle compiles it, including cached dists.
    """
    activity = Path(dist_dir) / "src/main/java/org/kivy/android/PythonActivity.java"
    source = activity.read_text(encoding="utf-8")
    # Check even a previously patched distribution: a new, unguarded call
    # must not be hidden by the old guard. Also catch differently spaced calls.
    calls = re.findall(r"\bfinishLoad\s*\(\s*\)\s*;", source)
    if len(calls) != 1 or source.count(_FINISH_LOAD) != 1:
        raise RuntimeError(
            "Cannot apply SDL startup guard: PythonActivity.finishLoad changed. "
            "Review the python-for-android bootstrap before building the APK."
        )
    if _GUARDED_FINISH_LOAD in source:
        return
    activity.write_text(
        source.replace(_FINISH_LOAD, _GUARDED_FINISH_LOAD), encoding="utf-8"
    )


def before_apk_build(self):
    dist = getattr(self, "_dist", None)
    if dist is None:
        return
    _guard_sdl_startup(dist.dist_dir)
