# Android native startup failure

## Confirmed cause and fix

The subsequent SDL dialog on the Xiaomi 11T / Android 14 reports:

```
dlopen failed: can't enable GNU RELRO protection for ".../libpng16.so": Out of memory
```

This is a native ELF layout error, rather than a shortage of phone RAM. The
original APK's `libpng16.so` has the following relevant program headers:

| Segment | Virtual address | Memory size | End |
| --- | ---: | ---: | ---: |
| Final `PT_LOAD` | `0x4e5f0` | `0x708` | `0x4ecf8` |
| `PT_GNU_RELRO` | `0x4e5f0` | `0x1a10` | `0x50000` |

On a 4 KB device, Android reserves the LOAD image through `0x4f000`, but RELRO
asks `mprotect` to protect memory through `0x50000`. That extra page is outside
the library's reservation, causing `ENOMEM`. The old APK also has the same
problem in `libwebpdemux.so` and `libwebpmux.so`.

NDK r25b's older LLD rounds RELRO for `common-page-size=16384` without padding
the last LOAD segment. [LLVM PR #66042](https://github.com/llvm/llvm-project/pull/66042)
fixes this by adding `.relro_padding` and extending the associated LOAD.
The Android build now uses **NDK r28c**, which is also the recommendation in
the pinned python-for-android revision. All libraries and C++/OpenMP runtimes
come from this NDK; the hook no longer swaps in runtimes from an unrelated NDK.
Explicit 16 KB link flags remain enabled. This preserves RELRO protection and
16 KB compatibility while fixing loading on 4 KB devices.

Changing `buildozer.spec` and the NDK preparation script invalidates the exact
native/toolchain cache keys. Native outputs are rebuilt with the new linker.
The APK workflow now validates every native ELF, including Python extension
modules inside `libpybundle.so`, with:

```sh
python tools/validate_android_elf.py bin/*.apk
```

The check requires every LOAD segment to have compatible 16 KB alignment and
verifies RELRO stays inside Android's reserved LOAD image at both 4 KB and
16 KB page sizes. It deliberately accepts reserved gaps between LOAD segments,
matching [Android 14's linker](https://android.googlesource.com/platform/bionic/+/refs/tags/android-14.0.0_r1/linker/linker_phdr.cpp).
Only successfully validated APKs are uploaded as installable artifacts.

## Original secondary crash

The reported Xiaomi 11T / Android 14 stack ends in
`SDLActivity.nativeSetenv` from `PythonActivity.UnpackFilesTask.onPostExecute`.
This runs before `app/main.py`, so Python exception handling cannot fix it.

In the SDL2 bootstrap, `finishLoad()` loads the native libraries. SDL catches a
loader failure, sets `mBrokenLibraries`, and displays an **SDL Error** dialog
with the original loader message. The p4a unpack task then continues to call
`nativeSetenv` even when loading failed. If SDL was never loaded, this second
failure crashes the activity and hides the original error dialog.

The packaging hook now checks `mBrokenLibraries` immediately after
`finishLoad()`. On failure it returns with the original dialog still visible;
successful startup continues normally. The hook patches the generated Java
before Gradle compilation and also covers cached distributions. It fails the
build if the upstream patch location changes.

## Validation

The unmodified APK from [CI run 37925839819](https://github.com/gdincu/PPLineCaller/actions/runs/37925839819)
was inspected and launched on the local Android 15 emulator using ARM64
translation. All native libraries loaded, Python/Kivy initialized, and the app
reached its main loop. `libSDL2.so` exports both `JNI_OnLoad` and
`Java_org_libsdl_app_SDLActivity_nativeSetenv`. Every packaged ELF's `PT_LOAD`
segment was checked for 16 KB alignment. ARM translation did not reproduce the
phone's RELRO failure; alignment alone was an insufficient check.

The new ELF verifier rejects that original APK with precisely the three RELRO
overruns above. Regression cases use the failing libpng program headers and
cover the padded correction, both ELF classes/byte orders, all LOAD segments,
truncated headers, reserved gaps, and bundled Python extensions. A small ARM64
library linked locally with LLD 18 verifies the fixed linker emits
`.relro_padding`, with both LOAD and RELRO ending at the same page boundary.

The replacement APK from [CI run 37980870082](https://github.com/gdincu/PPLineCaller/actions/runs/37980870082),
built from fix commit `a867a1b`, passed all build checks and independent
verification of **152 native libraries**. In the shipped libpng, LOAD and RELRO
now both end at `0x50000`; WebP demux/mux end at `0x8000`/`0x10000` respectively.
On a real 4 KB Linux kernel, reserving each libpng's LOAD image and protecting
its RELRO range reproduces `ENOMEM` for the old APK and succeeds for the new one.
The rebuilt APK starts Python/Kivy and reaches the application main loop in
the Android 15 emulator. Installation on the Xiaomi remains the final device
check.

The regression tests compile and execute a small Java startup harness covering
both failed and successful loading, plus hook idempotence and upstream drift,
including additional loading calls in an already-patched distribution.
Run `python -m unittest discover -s tests -v`; the Java test requires a JDK.

CI also checks the real Java bootstrap from the `p4a.commit` pinned in
`buildozer.spec`. Set `P4A_SOURCE_DIR` to that checkout to run this check locally.
The APK workflow checks that the generated `PythonActivity.java` matches the
expected guarded source after Gradle successfully compiles it. Set `P4A_DIST_DIR`
to the generated distribution to enable that additional check; a configured but
missing checkout or distribution fails the tests.

Android builds install the Buildozer and Cython versions in
`requirements-android.txt` and fetch the exact p4a commit instead of following
the moving `develop` branch. Update these pins deliberately and run both CI
workflows before relying on a new toolchain.

SDK, NDK, and Ant have a separate cache from native outputs. Changing a hook or
local recipe invalidates native binaries without discarding the toolchain.
`tools/prepare-android-ndk.sh` is included in both cache keys and checks the
16 KB CMake linker patch on fresh and restored NDKs.

Install the APK rebuilt with NDK r28c on the phone. If native
loading still fails, record the message from the **SDL Error** dialog. It names
the original missing library, symbol, or linker error needed to fix the
underlying load failure. Alternatively, capture startup logs with USB debugging:

```sh
adb logcat -c
adb shell am force-stop org.example.pplinecaller
adb shell am start -W -n org.example.pplinecaller/org.kivy.android.PythonActivity
adb logcat -d -v threadtime > startup-logcat.txt
```

Relevant tags include `pythonutil`, `SDL`, `System.err`, and `AndroidRuntime`.
Keep the messages before the final exception, especially `Library loading error`.

Upstream control flow: [PythonActivity](https://github.com/kivy/python-for-android/blob/develop/pythonforandroid/bootstraps/sdl2/build/src/main/java/org/kivy/android/PythonActivity.java),
[PythonUtil](https://github.com/kivy/python-for-android/blob/develop/pythonforandroid/bootstraps/common/build/src/main/java/org/kivy/android/PythonUtil.java),
and [SDLActivity](https://github.com/libsdl-org/SDL/blob/release-2.30.11/android-project/app/src/main/java/org/libsdl/app/SDLActivity.java).
