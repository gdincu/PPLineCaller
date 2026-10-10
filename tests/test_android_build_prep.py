"""Native preparation must preserve alignment and reject upstream drift."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "android_build_prep",
    Path(__file__).resolve().parents[1] / "tools/prepare_python_for_android.py",
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


class AndroidBuildPrepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archs = self.root / "pythonforandroid/archs.py"
        self.application = self.root / "pythonforandroid/bootstraps/_sdl_common/build/jni/Application.mk"
        self.application.parent.mkdir(parents=True)
        self.archs.write_text("class Arch:\n" + prep.ORIGINAL_LDFLAGS + "\n", encoding="utf-8")
        self.application.write_text("APP_PLATFORM := android-24", encoding="utf-8")

    def test_alignment_patch_is_complete_and_idempotent(self):
        prep.patch_linker_flags(self.root)
        first = self.archs.read_bytes(), self.application.read_bytes()
        self.assertIn(prep.PATCHED_LDFLAGS, first[0].decode())
        self.assertEqual(first[1].decode().splitlines()[-1], prep.APP_LDFLAGS)
        prep.patch_linker_flags(self.root)
        self.assertEqual(first, (self.archs.read_bytes(), self.application.read_bytes()))

    def test_changed_or_duplicate_upstream_flags_fail_without_writing(self):
        for source in ("common_ldflags = []\n", (prep.ORIGINAL_LDFLAGS + "\n") * 2):
            with self.subTest(source=source):
                self.archs.write_text(source, encoding="utf-8")
                original = self.archs.read_bytes(), self.application.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "common_ldflags changed"):
                    prep.patch_linker_flags(self.root)
                self.assertEqual(original, (self.archs.read_bytes(), self.application.read_bytes()))

    def test_missing_makefile_does_not_partially_patch_archs(self):
        original = self.archs.read_bytes()
        self.application.unlink()
        with self.assertRaises(FileNotFoundError):
            prep.patch_linker_flags(self.root)
        self.assertEqual(original, self.archs.read_bytes())


if __name__ == "__main__":
    unittest.main()
