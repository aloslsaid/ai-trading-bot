from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from stat import S_ISLNK
from zipfile import ZipFile

ARCHIVE = Path("GOLD_AI_RESEARCH_CLEAN (1).zip")
OUTPUT = Path("archive_extracted")
ALLOWED_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".ini", ".csv", ".html", ".example", ".gitignore"}
MAX_TOTAL_BYTES = 5 * 1024 * 1024


def main() -> int:
    total = 0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_root = OUTPUT.resolve()
    with ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            windows_path = PureWindowsPath(info.filename)
            if (path.is_absolute() or windows_path.is_absolute() or windows_path.drive
                    or "\\\\" in info.filename or ".." in path.parts):
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if S_ISLNK(mode):
                raise ValueError(f"Symbolic link is not allowed: {info.filename}")
            if Path(info.filename).suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            total += info.file_size
            if total > MAX_TOTAL_BYTES:
                raise ValueError("Selected text files exceed the extraction size limit.")
            destination = OUTPUT / Path(*path.parts)
            try:
                destination.resolve().relative_to(output_root)
            except ValueError:
                raise ValueError(f"Archive path escapes output directory: {info.filename}") from None
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
            print(f"EXTRACTED: {destination} ({info.file_size} bytes)")
    print(f"TOTAL_EXTRACTED_BYTES: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
