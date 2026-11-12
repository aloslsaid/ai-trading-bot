"""Frame persistence. Parquet is the standard format (needs pyarrow).

If pyarrow is missing, falls back to pickle with a loud warning so the code still runs;
install requirements.txt to get real Parquet files.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

try:  # pragma: no cover - depends on environment
    import pyarrow  # noqa: F401

    HAVE_PARQUET = True
except ImportError:  # pragma: no cover
    HAVE_PARQUET = False


def _resolve(path: Path) -> Path:
    path = Path(path)
    if HAVE_PARQUET:
        return path.with_suffix(".parquet")
    log.warning("pyarrow not installed: using pickle instead of Parquet for %s", path.name)
    return path.with_suffix(".pkl")


def save_frame(df: pd.DataFrame, path: Path) -> Path:
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if HAVE_PARQUET:
        # Write via buffer then atomic replace — some filesystems report 0-byte
        # when pyarrow writes directly (observed on certain sandboxed mounts).
        import io
        buf = io.BytesIO()
        df.to_parquet(buf)
        data = buf.getvalue()
        if not data:
            raise RuntimeError(f"parquet serialization produced empty bytes for {out}")
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(out)
    else:
        df.to_pickle(out)
    return out


def load_frame(path: Path) -> pd.DataFrame:
    out = _resolve(path)
    if not out.exists():
        raise FileNotFoundError(f"{out} not found. Run the previous pipeline step first.")
    return pd.read_parquet(out) if HAVE_PARQUET else pd.read_pickle(out)


def frame_exists(path: Path) -> bool:
    return _resolve(path).exists()
