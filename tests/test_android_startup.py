"""Packaging regressions for p4a's SDL native-library startup failure path."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "p4a_hook", Path(__file__).resolve().parents[1] / "p4a-hook.py"
)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)

# Exercise the generated Java's control flow: failed loading must keep the
# original SDL error visible and must never reach an unresolved JNI call.
JAVA = """
class SDLActivity {
    static boolean mBrokenLibraries;
    static int nativeCalls;
    static void nativeSetenv(String key, String value) {
        if (mBrokenLibraries) throw new UnsatisfiedLinkError("nativeSetenv");
        nativeCalls++;
    }
}
class Log {
    static int e(String tag, String message) { return 0; }
}
public class PythonActivity extends SDLActivity {
    static final String TAG = "PythonActivity";
    PythonActivity mActivity = this;
    boolean failedLoad;
    boolean errorDialogVisible;
    void finishLoad() {
        mBrokenLibraries = failedLoad;
        errorDialogVisible = failedLoad;
    }
    void onPostExecute() {
            mActivity.finishLoad();
            SDLActivity.nativeSetenv("ANDROID_ENTRYPOINT", "main.py");
    }
    public static void main(String[] args) {
        PythonActivity activity = new PythonActivity();
        activity.failedLoad = true;
        activity.onPostExecute();
        if (!activity.errorDialogVisible || nativeCalls != 0)
            throw new AssertionError("Failed load must retain its error and skip JNI");
        activity.failedLoad = false;
        activity.onPostExecute();
        if (activity.errorDialogVisible || nativeCalls != 1)
            throw new AssertionError("Successful load must continue startup");
    }
}
"""


class AndroidStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dist = Path(self.temp.name)
        self.activity = self.dist / "src/main/java/org/kivy/android/PythonActivity.java"
        self.activity.parent.mkdir(parents=True)
        self.activity.write_text(JAVA, encoding="utf-8")

    def test_guard_applies_and_is_idempotent(self):
        toolchain = SimpleNamespace(_dist=SimpleNamespace(dist_dir=str(self.dist)))
        hook.before_apk_build(toolchain)
        first = self.activity.read_text(encoding="utf-8")
        hook.before_apk_build(toolchain)
        self.assertEqual(first, self.activity.read_text(encoding="utf-8"))
        self.assertIn(hook._GUARDED_FINISH_LOAD, first)

    def test_changed_upstream_bootstrap_fails_build(self):
        for source in (JAVA.replace(hook._FINISH_LOAD, ""), JAVA + hook._FINISH_LOAD):
            with self.subTest(source=source):
                self.activity.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "finishLoad changed"):
                    hook._guard_sdl_startup(self.dist)
                self.assertEqual(source, self.activity.read_text(encoding="utf-8"))

    def test_cached_guard_does_not_hide_another_load_call(self):
        hook._guard_sdl_startup(self.dist)
        guarded = self.activity.read_text(encoding="utf-8")
        for extra_call in (
            hook._FINISH_LOAD,
            "mActivity.finishLoad();",
            "            mActivity . finishLoad ();",
            "            this.finishLoad();",
            "            finishLoad();",
        ):
            with self.subTest(extra_call=extra_call):
                source = guarded + "\n" + extra_call
                self.activity.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "finishLoad changed"):
                    hook._guard_sdl_startup(self.dist)
                self.assertEqual(source, self.activity.read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("javac") and shutil.which("java"), "JDK required")
    def test_failed_load_skips_jni_and_successful_load_continues(self):
        hook._guard_sdl_startup(self.dist)
        subprocess.run(
            ["javac", "-d", str(self.dist), str(self.activity)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["java", "-cp", str(self.dist), "PythonActivity"],
            check=True, capture_output=True, text=True,
        )


@unittest.skipUnless(os.environ.get("P4A_SOURCE_DIR"), "P4A_SOURCE_DIR required for upstream checks")
class UpstreamAndroidStartupTests(unittest.TestCase):
    """Check real Java in CI; the APK job also compiles it through Gradle."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dist = Path(self.temp.name)
        relative = Path("src/main/java/org/kivy/android/PythonActivity.java")
        upstream = (
            Path(os.environ["P4A_SOURCE_DIR"])
            / "pythonforandroid/bootstraps/sdl2/build" / relative
        )
        self.original = upstream.read_text(encoding="utf-8")
        self.activity = self.dist / relative
        self.activity.parent.mkdir(parents=True)
        self.activity.write_text(self.original, encoding="utf-8")
        hook._guard_sdl_startup(self.dist)
        self.guarded = self.activity.read_text(encoding="utf-8")

    def test_guard_applies_to_pinned_upstream_java(self):
        self.assertEqual(self.guarded.count(hook._GUARDED_FINISH_LOAD), 1)
        self.assertEqual(
            self.guarded.replace(hook._GUARDED_FINISH_LOAD, hook._FINISH_LOAD),
            self.original,
        )
        self.assertLess(
            self.guarded.index(hook._GUARDED_FINISH_LOAD) + len(hook._GUARDED_FINISH_LOAD),
            self.guarded.index("SDLActivity.nativeSetenv("),
        )
        hook._guard_sdl_startup(self.dist)
        self.assertEqual(self.guarded, self.activity.read_text(encoding="utf-8"))

    @unittest.skipUnless(os.environ.get("P4A_DIST_DIR"), "P4A_DIST_DIR required after APK build")
    def test_compiled_distribution_contains_expected_guard(self):
        generated = (
            Path(os.environ["P4A_DIST_DIR"])
            / "src/main/java/org/kivy/android/PythonActivity.java"
        )
        self.assertEqual(generated.read_text(encoding="utf-8"), self.guarded)


if __name__ == "__main__":
    unittest.main()
