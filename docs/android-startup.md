# Android native startup failure

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

## Validation and remaining diagnosis

The unmodified APK from [CI run 37925839819](https://github.com/gdincu/PPLineCaller/actions/runs/37925839819)
was inspected and launched on the local Android 15 emulator using ARM64
translation. All native libraries loaded, Python/Kivy initialized, and the app
reached its main loop. `libSDL2.so` exports both `JNI_OnLoad` and
`Java_org_libsdl_app_SDLActivity_nativeSetenv`. Every packaged ELF's `PT_LOAD`
segment was checked for 16 KB alignment. This does not reproduce or rule out a
device-specific loader failure on the Xiaomi phone.

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

Rebuild the APK with the updated hook and install it on the phone. If native
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
