"""CSV importer for Dukascopy-style exports (separate Bid and Ask files).

Accepted timestamp columns: 'Gmt time' / 'Local time' / 'timestamp' / 'time' / 'date'.
Accepted formats include '01.06.2026 00:00:00.000' (Dukascopy web export) and ISO strings.
Timestamps are treated as UTC (Dukascopy: choose 'GMT' when exporting).
Unparseable rows are dropped and counted, never repaired.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_TS_NAMES = ("gmt time", "timestamp", "time", "date", "datetime", "local time")
_COLS = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume", "vol": "volume"}


def read_ohlc_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    ts_col = next((c for c in df.columns if c in _TS_NAMES), None)
    if ts_col is None:
        raise ValueError(f"{path}: no timestamp column found among {list(df.columns)}")
    sample = df[ts_col].astype(str).iloc[0]
    fmt = "%d.%m.%Y %H:%M:%S.%f" if "." in sample.split(" ")[0] else None
    ts = pd.to_datetime(df[ts_col], format=fmt, errors="coerce", utc=True)
    rename = {c: _COLS[c] for c in df.columns if c in _COLS}
    out = df.rename(columns=rename)
    for need in ("open", "high", "low", "close"):
        if need not in out.columns:
            raise ValueError(f"{path}: missing column '{need}'")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out = out[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(ts, name="timestamp")
    bad = int(out.index.isna().sum() + out[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if bad:
        log.warning("%s: dropping %d unparseable rows", Path(path).name, bad)
    out = out[~out.index.isna()].dropna(subset=["open", "high", "low", "close"])
    return out
