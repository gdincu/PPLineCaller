#!/usr/bin/env bash
# This file is hashed by both caches: changing linker patches invalidates them.
set -euo pipefail

ndk_revision="${1:?Usage: prepare-android-ndk.sh NDK_REVISION}"
ndk_dir="$HOME/.buildozer/android/platform/android-ndk-r${ndk_revision}"
if [ ! -f "$ndk_dir/source.properties" ]; then
  mkdir -p "$(dirname "$ndk_dir")"
  archive=$(mktemp --suffix=.zip)
  trap 'rm -f "$archive"' EXIT
  curl -sSfL -o "$archive" \
    "https://dl.google.com/android/repository/android-ndk-r${ndk_revision}-linux.zip"
  unzip -q "$archive" -d "$(dirname "$ndk_dir")"
fi

# Recheck restored toolchains too. source.properties only proves installation,
# not that the CMake linker flags have been patched.
python - "$ndk_dir" <<'PY'
from pathlib import Path
import sys

link_flags = "-Wl,--build-id=sha1 -Wl,-z,max-page-size=16384 -Wl,-z,common-page-size=16384"
patched = 0
for toolchain in (Path(sys.argv[1]) / "build/cmake").glob("*.cmake"):
    source = toolchain.read_text(encoding="utf-8")
    if link_flags not in source:
        updated = source.replace("-Wl,--build-id=sha1", link_flags)
        if updated != source:
            toolchain.write_text(updated, encoding="utf-8")
        source = updated
    if link_flags in source:
        patched += 1
if not patched:
    raise SystemExit("Cannot apply 16 KB linker flags: NDK CMake toolchain changed")
PY
grep -n "max-page-size" "$ndk_dir"/build/cmake/*.cmake
