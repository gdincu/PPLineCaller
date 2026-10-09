"""ELF regressions for Android's RELRO mprotect failure on 4 KB devices."""

import io
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
import zipfile

from tools.validate_android_elf import (
    PT_GNU_RELRO, PT_LOAD, Segment, layout_errors, native_libraries, read_segments,
)


def png_segments(padded=False):
    # Program headers from the failing libpng16.so. LOAD alignment is valid;
    # its final mapped page ends at 0x4f000, before RELRO ends at 0x50000.
    return [
        Segment(PT_LOAD, 0, 0, 0xD3DC, 0xD3DC, 0x4000),
        Segment(PT_LOAD, 0xD3DC, 0x113DC, 0x39214, 0x39214, 0x4000),
        Segment(PT_LOAD, 0x465F0, 0x4E5F0, 0x708, 0x1A10 if padded else 0x708, 0x4000),
        Segment(PT_GNU_RELRO, 0x465F0, 0x4E5F0, 0x708, 0x1A10, 1),
    ]


def elf_bytes(segments, bits=64, order="<"):
    header = struct.Struct(order + ("HHIQQQIHHHHHH" if bits == 64 else "HHIIIIIHHHHHH"))
    program = struct.Struct(order + ("IIQQQQQQ" if bits == 64 else "IIIIIIII"))
    offset = 16 + header.size
    size = max(offset + program.size * len(segments), *(s.offset + s.file_size for s in segments))
    data = bytearray(size)
    data[:7] = b"\x7fELF" + bytes((2 if bits == 64 else 1, 1 if order == "<" else 2, 1))
    header.pack_into(data, 16, 3, 183 if bits == 64 else 40, 1, 0, offset, 0, 0,
                     offset, program.size, len(segments), 0, 0, 0)
    for index, s in enumerate(segments):
        if bits == 64:
            fields = (s.kind, 6, s.offset, s.address, s.address, s.file_size, s.memory_size, s.alignment)
        else:
            fields = (s.kind, s.offset, s.address, s.address, s.file_size, s.memory_size, 6, s.alignment)
        program.pack_into(data, offset + index * program.size, *fields)
    return bytes(data)


class AndroidELFTests(unittest.TestCase):
    def test_original_png_relro_reaches_unmapped_page(self):
        errors = layout_errors(read_segments(elf_bytes(png_segments())))
        self.assertEqual(len(errors), 1)
        self.assertIn("at 0x4f000 on 4 KB pages", errors[0])

    def test_padded_load_covers_relro_on_both_page_sizes(self):
        self.assertEqual(layout_errors(png_segments(padded=True)), [])

    def test_every_load_must_be_aligned(self):
        segments = png_segments(padded=True)
        s = segments[1]
        segments[1] = Segment(s.kind, s.offset, s.address, s.file_size, s.memory_size, 4096)
        self.assertTrue(any("not 16 KB aligned" in e for e in layout_errors(segments)))

    def test_offset_congruence_is_required(self):
        segments = png_segments(padded=True)
        s = segments[1]
        segments[1] = Segment(s.kind, s.offset + 4096, s.address, s.file_size, s.memory_size, s.alignment)
        self.assertTrue(any("incompatible file/virtual offsets" in e for e in layout_errors(segments)))

    def test_relro_may_end_within_a_mapped_page(self):
        segments = [
            Segment(PT_LOAD, 0, 0, 0x700, 0x700, 16384),
            Segment(PT_GNU_RELRO, 0, 0, 0x700, 0x1000, 1),
        ]
        self.assertEqual(layout_errors(segments), [])

    def test_android_keeps_interior_load_gaps_reserved(self):
        segments = [
            Segment(PT_LOAD, 0, 0, 0x1000, 0x1000, 16384),
            Segment(PT_LOAD, 0x8000, 0x8000, 0x1000, 0x1000, 16384),
            Segment(PT_GNU_RELRO, 0, 0, 0, 0x9000, 1),
        ]
        self.assertEqual(layout_errors(segments), [])

    def test_missing_relro_is_rejected(self):
        self.assertIn("missing GNU_RELRO protection", layout_errors(png_segments(padded=True)[:-1]))

    def test_elf_classes_and_byte_orders(self):
        for bits in (32, 64):
            for order in ("<", ">"):
                with self.subTest(bits=bits, order=order):
                    self.assertEqual(read_segments(elf_bytes(png_segments(), bits, order)), png_segments())

    def test_truncated_or_non_elf_input_is_rejected(self):
        data = elf_bytes(png_segments())
        for broken in (b"not ELF", data[:32], data[:100], data[:-1]):
            with self.subTest(size=len(broken)):
                with self.assertRaises(ValueError):
                    read_segments(broken)

    def test_apk_includes_bundled_python_extensions(self):
        good = elf_bytes(png_segments(padded=True))
        bad = elf_bytes(png_segments())
        bundle_data = io.BytesIO()
        with tarfile.open(fileobj=bundle_data, mode="w:gz") as bundle:
            info = tarfile.TarInfo("_python_bundle/site-packages/module.so")
            info.size = len(bad)
            bundle.addfile(info, io.BytesIO(bad))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "app.apk"
            with zipfile.ZipFile(path, "w") as apk:
                apk.writestr("lib/arm64-v8a/libpng16.so", good)
                apk.writestr("lib/arm64-v8a/libpybundle.so", bundle_data.getvalue())
            libraries = list(native_libraries(path))
        self.assertEqual(len(libraries), 2)
        self.assertEqual(layout_errors(read_segments(libraries[0][1])), [])
        self.assertIn("module.so", libraries[1][0])
        self.assertTrue(layout_errors(read_segments(libraries[1][1])))


if __name__ == "__main__":
    unittest.main()
