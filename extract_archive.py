from __future__ import annotations

from pathlib import Path, PurePosixPath
from stat import S_ISLNK
from zipfile import ZipFile

ARCHIVE = Path("GOLD_AI_RESEARCH_CLEAN (1).zip")
OUTPUT = Path(".")
ALLOWED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".ini", ".csv", ".html", ".example", ".gitignore"}
MAX_TOTAL_BYTES = 5 * 1024 * 1024


def main() -> int:
    total = 0
    with ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if info.is_dir():
                continue
            if S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symbolic link is not allowed: {info.filename}")
            if Path(info.filename).suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Selected text files exceed the extraction size limit.")
            destination = OUTPUT / Path(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
            print(f"EXTRACTED: {destination} ({info.file_size} bytes)")
    print(f"TOTAL_EXTRACTED_BYTES: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
