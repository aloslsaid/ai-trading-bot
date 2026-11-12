from __future__ import annotations

from pathlib import PurePosixPath
from zipfile import ZipFile, BadZipFile

ARCHIVE = "GOLD_AI_RESEARCH_CLEAN (1).zip"


def main() -> int:
    try:
        with ZipFile(ARCHIVE) as archive:
            bad_member = archive.testzip()
            if bad_member:
                print(f"CORRUPT_MEMBER: {bad_member}")
                return 2

            members = archive.infolist()
            print(f"ARCHIVE: {ARCHIVE}")
            print(f"FILES: {len(members)}")
            total = 0
            for member in members:
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    print(f"UNSAFE_PATH: {member.filename}")
                    return 3
                if member.is_dir():
                    continue
                total += member.file_size
                print(
                    f"{member.file_size:>10} bytes | "
                    f"{member.compress_size:>10} compressed | {member.filename}"
                )
            print(f"TOTAL_UNCOMPRESSED_BYTES: {total}")
            return 0
    except FileNotFoundError:
        print(f"Archive not found: {ARCHIVE}")
        return 1
    except BadZipFile:
        print("The uploaded archive is not a valid ZIP file.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
