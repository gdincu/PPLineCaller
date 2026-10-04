"""python-for-android hook: align native libs to 16 KB page size.

Runs via `p4a.hook` in buildozer.spec, before gradle packages the APK
(`before_apk_build`). Android 15+ shows a compatibility dialog when the
APK's ELF load segments are not 16 KB aligned.

patchelf's `--page-size 16384` only re-aligns the ELF when the file is
actually rewritten, so we force a rewrite by adding then removing a
harmless DT_NEEDED entry (liblog.so, always present on Android).
"""

import glob
import os
import subprocess


def _align_libs(dist_dir, arch):
    for so in sorted(glob.glob(os.path.join(dist_dir, "libs", arch, "*.so"))):
        if not os.path.isfile(so):
            # symlink to a directory (e.g. libpybundle.so -> _python_bundle)
            continue
        with open(so, "rb") as f:
            if f.read(4) != b"\x7fELF":
                # not an ELF (libpybundle.so is the python bundle zip)
                continue
        subprocess.run(
            ["patchelf", "--page-size", "16384", "--add-needed", "liblog.so", so],
            check=True,
        )
        subprocess.run(
            ["patchelf", "--page-size", "16384", "--remove-needed", "liblog.so", so],
            check=True,
        )


def before_apk_build(self):
    dist = getattr(self, "_dist", None)
    if dist is None:
        return
    _align_libs(dist.dist_dir, "arm64-v8a")
    _align_libs(dist.dist_dir, "x86_64")