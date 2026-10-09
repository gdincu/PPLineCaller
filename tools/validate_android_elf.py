"""Check APK native libraries for 16 KB alignment and loadable GNU RELRO.

Uses only the standard library, including for p4a's tarred Python extensions.
RELRO must stay inside the reserved LOAD image on both 4 KB and 16 KB devices.
"""

import argparse
from dataclasses import dataclass
import io
from pathlib import Path
import struct
import tarfile
import zipfile


PT_LOAD = 1
PT_GNU_RELRO = 0x6474E552
ALIGNMENT = 16384
PAGE_SIZES = (4096, 16384)


@dataclass(frozen=True)
class Segment:
    kind: int
    offset: int
    address: int
    file_size: int
    memory_size: int
    alignment: int


def read_segments(data):
    """Read ELF32/ELF64 program headers without depending on section headers."""
    if len(data) < 16 or data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    if data[4] not in (1, 2) or data[5] not in (1, 2):
        raise ValueError("unsupported ELF class or byte order")
    order = "<" if data[5] == 1 else ">"
    is_64 = data[4] == 2
    header = struct.Struct(order + ("HHIQQQIHHHHHH" if is_64 else "HHIIIIIHHHHHH"))
    program = struct.Struct(order + ("IIQQQQQQ" if is_64 else "IIIIIIII"))
    try:
        fields = header.unpack_from(data, 16)
        offset, entry_size, count = fields[4], fields[8], fields[9]
        if not count or count == 0xFFFF or entry_size < program.size:
            raise ValueError("invalid or unsupported program header table")
        if offset < 16 + header.size or offset + entry_size * count > len(data):
            raise ValueError("truncated or invalid program header table")
        segments = []
        for index in range(count):
            values = program.unpack_from(data, offset + index * entry_size)
            if is_64:
                kind, _, file_offset, address, _, size, memory, alignment = values
            else:
                kind, file_offset, address, _, size, memory, _, alignment = values
            if file_offset + size > len(data):
                raise ValueError("segment extends past the ELF file")
            segments.append(Segment(kind, file_offset, address, size, memory, alignment))
        return segments
    except struct.error as error:
        raise ValueError("truncated ELF header") from error


def _page_range(segment, page_size):
    start = segment.address // page_size * page_size
    end = (segment.address + segment.memory_size + page_size - 1) // page_size * page_size
    return start, end


def layout_errors(segments):
    loads = [s for s in segments if s.kind == PT_LOAD and s.memory_size]
    relros = [s for s in segments if s.kind == PT_GNU_RELRO and s.memory_size]
    errors = []
    if not loads:
        errors.append("no loadable segments")
    for segment in loads:
        if segment.file_size > segment.memory_size:
            errors.append("LOAD file size exceeds memory size")
        if segment.alignment < ALIGNMENT or segment.alignment & (segment.alignment - 1):
            errors.append(f"LOAD at {segment.address:#x} is not 16 KB aligned")
        if (segment.address - segment.offset) % ALIGNMENT:
            errors.append(f"LOAD at {segment.address:#x} has incompatible file/virtual offsets")
    if not relros:
        errors.append("missing GNU_RELRO protection")
    for page_size in PAGE_SIZES:
        # Android reserves the entire LOAD image with mmap(PROT_NONE), then
        # maps individual segments over it. Interior gaps stay reserved;
        # only a RELRO range outside that full image reaches unmapped memory.
        mapped = [_page_range(s, page_size) for s in loads]
        low = min((start for start, _ in mapped), default=0)
        high = max((end for _, end in mapped), default=0)
        for relro in relros:
            start, end = _page_range(relro, page_size)
            if start < low or end > high:
                cursor = start if start < low else high
                errors.append(
                    f"GNU_RELRO [{start:#x}, {end:#x}) reaches unmapped memory "
                    f"at {cursor:#x} on {page_size // 1024} KB pages"
                )
    return errors


def native_libraries(path):
    """Include Python extension modules inside p4a's libpybundle.so archive."""
    if path.suffix != ".apk":
        yield path.name, path.read_bytes()
        return
    with zipfile.ZipFile(path) as apk:
        for name in sorted(apk.namelist()):
            if not name.startswith("lib/") or not name.endswith(".so"):
                continue
            data = apk.read(name)
            if name.endswith("/libpybundle.so"):
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as bundle:
                    for member in bundle:
                        if member.isfile() and member.name.endswith(".so"):
                            with bundle.extractfile(member) as extension:
                                yield f"{name}!{member.name}", extension.read()
            else:
                yield name, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="APK or ELF library paths")
    args = parser.parse_args()
    failed = False
    count = 0
    for path in args.paths:
        found = 0
        try:
            for name, data in native_libraries(path):
                found += 1
                try:
                    errors = layout_errors(read_segments(data))
                except ValueError as error:
                    errors = [str(error)]
                for error in errors:
                    print(f"FAIL {path.name}:{name}: {error}")
                failed |= bool(errors)
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
            print(f"FAIL {path}: {error}")
            failed = True
        if not found:
            print(f"FAIL {path}: no native libraries found")
            failed = True
        count += found
    if not failed:
        print(f"PASS: {count} native libraries; 16 KB alignment and RELRO mapping on 4/16 KB pages")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
