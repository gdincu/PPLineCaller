"""python-for-android hook: 16 KB page-size alignment of native libs.

Two parts (patchelf is NOT used - its ELF rewrite breaks JNI symbol
resolution in SDL2):

1. Everything p4a compiles (python, sdl2, kivy, numpy, opencv, thorvg, ...)
   is linked with `-Wl,-z,max-page-size=16384` via a patched archs.py and
   Application.mk in the p4a checkout, and CMake-built recipes (opencv,
   jpeg, libwebp) via patched NDK toolchain files - see
   .github/workflows/build-apk.yml.

2. Prebuilt libs shipped by NDK r25b (libc++_shared.so, libomp.so) are not
   16 KB aligned. GitHub runners preinstall a newer NDK (r27+, which ships
   16 KB-aligned prebuilts) under /usr/local/lib/android/sdk/ndk/, so we
   swap those files in before gradle packages the APK.
"""

import glob
import os
import shutil

_SWAP_LIBS = ("libc++_shared.so", "libomp.so")


def _find_newer_ndk_prebuilt_dir():
    for prebuilt in sorted(glob.glob(
            "/usr/local/lib/android/sdk/ndk/*/toolchains/llvm/prebuilt/linux-x86_64")):
        if _has_swap_libs(prebuilt):
            return prebuilt
    return None


def _has_swap_libs(prebuilt):
    sysroot = os.path.join(
        prebuilt, "sysroot", "usr", "lib", "aarch64-linux-android")
    if not os.path.isfile(os.path.join(sysroot, "libc++_shared.so")):
        return False
    return bool(glob.glob(
        os.path.join(prebuilt, "lib*/clang/*/lib/linux/aarch64/libomp.so")))


def _swap_prebuilt_libs(dist_dir, prebuilt):
    lib_dir = os.path.join(dist_dir, "libs", "arm64-v8a")
    sysroot = os.path.join(
        prebuilt, "sysroot", "usr", "lib", "aarch64-linux-android")
    libomp = glob.glob(os.path.join(
        prebuilt, "lib*/clang/*/lib/linux/aarch64/libomp.so"))
    sources = {
        "libc++_shared.so": os.path.join(sysroot, "libc++_shared.so"),
        "libomp.so": libomp[0] if libomp else None,
    }
    for lib in _SWAP_LIBS:
        src = sources[lib]
        dst = os.path.join(lib_dir, lib)
        if src and os.path.isfile(src) and os.path.isfile(dst):
            shutil.copy2(src, dst)


def before_apk_build(self):
    dist = getattr(self, "_dist", None)
    if dist is None:
        return
    prebuilt = _find_newer_ndk_prebuilt_dir()
    if prebuilt is None:
        return
    _swap_prebuilt_libs(dist.dist_dir, prebuilt)