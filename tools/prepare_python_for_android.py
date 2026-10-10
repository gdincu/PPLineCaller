"""Fetch pinned p4a and apply the native linker patches hashed by the cache."""
import configparser
from pathlib import Path
import subprocess


ORIGINAL_LDFLAGS = "    common_ldflags = ['-L{ctx_libs_dir}']"
PATCHED_LDFLAGS = (
    "    common_ldflags = ['-L{ctx_libs_dir}', "
    "'-Wl,-z,max-page-size=16384', '-Wl,-z,common-page-size=16384']"
)
APP_LDFLAGS = "APP_LDFLAGS += -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384"


def patch_linker_flags(source_dir):
    """Fail on upstream drift and keep repeated preparation idempotent."""
    source_dir = Path(source_dir)
    archs = source_dir / "pythonforandroid/archs.py"
    application = source_dir / "pythonforandroid/bootstraps/_sdl_common/build/jni/Application.mk"
    source = archs.read_text(encoding="utf-8")
    makefile = application.read_text(encoding="utf-8")
    if source.count(ORIGINAL_LDFLAGS) == 1 and PATCHED_LDFLAGS not in source:
        source = source.replace(ORIGINAL_LDFLAGS, PATCHED_LDFLAGS)
    elif source.count(PATCHED_LDFLAGS) != 1 or ORIGINAL_LDFLAGS in source:
        raise RuntimeError("Cannot apply 16 KB linker flags: p4a common_ldflags changed")
    if APP_LDFLAGS not in makefile.splitlines():
        makefile = makefile.rstrip("\n") + "\n" + APP_LDFLAGS + "\n"
    archs.write_text(source, encoding="utf-8", newline="\n")
    application.write_text(makefile, encoding="utf-8", newline="\n")
    print("p4a linker flags: max-page-size=16384, common-page-size=16384")


def main():
    config = configparser.ConfigParser()
    config.read("buildozer.spec")
    commit = config["app"]["p4a.commit"]
    source_dir = Path(config["app"]["p4a.source_dir"])
    source_dir.mkdir(parents=True, exist_ok=True)
    for args in (
        ["init"],
        ["fetch", "--depth", "1", "https://github.com/kivy/python-for-android.git", commit],
        ["checkout", "--detach", "FETCH_HEAD"],
    ):
        subprocess.run(["git", "-C", str(source_dir), *args], check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(f"Unexpected p4a revision: {actual}")
    patch_linker_flags(source_dir)


if __name__ == "__main__":
    main()
