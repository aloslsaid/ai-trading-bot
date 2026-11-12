"""Data validation for M1 OHLC frames. Reports problems; drops only provably corrupt rows.

Missing candles are REPORTED, never filled or interpolated.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    name: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    unsorted_fixed: bool = False
    nonpositive_dropped: int = 0
    ohlc_inconsistent_dropped: int = 0
    suspicious_jumps: int = 0
    gaps_in_market_hours: int = 0
    daily_break_gaps: int = 0
    largest_gap_minutes: float = 0.0
    first: str = ""
    last: str = ""
    gap_samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def market_closed_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """Tolerant weekly close (UTC): Fri 21:00 -> Sun 23:00. Used only to avoid flagging weekends (covers DST shifts)."""
    wd, hr = idx.day_of_week, idx.hour
    return ((wd == 4) & (hr >= 21)) | (wd == 5) | ((wd == 6) & (hr < 23))


def validate_ohlc(df: pd.DataFrame, name: str, max_gap_minutes: int = 5, max_jump_pct: float = 1.5) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport(name=name, rows_in=len(df))
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("index must be a tz-aware (UTC) DatetimeIndex")
    df.index = df.index.tz_convert("UTC")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        rep.unsorted_fixed = True
    dup = df.index.duplicated(keep="first")
    rep.duplicates_dropped = int(dup.sum())
    df = df[~dup]

    px = ["open", "high", "low", "close"]
    nonfinite = ~np.isfinite(df[px].to_numpy(dtype=float)).all(axis=1)
    nonpos = (df[px] <= 0).any(axis=1) | nonfinite
    rep.nonpositive_dropped = int(nonpos.sum())
    df = df[~nonpos]

    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    rep.ohlc_inconsistent_dropped = int(bad.sum())
    df = df[~bad]

    if len(df) > 1:
        jumps = (df["close"].pct_change().abs() * 100) > max_jump_pct
        rep.suspicious_jumps = int(jumps.sum())  # flagged only; may be real news moves

        deltas = df.index.to_series().diff().dropna()
        start_times = df.index[:-1]
        mins = deltas.dt.total_seconds().to_numpy() / 60.0
        cand = np.flatnonzero(mins > max_gap_minutes)
        real_gaps = []
        for k in cand:
            t0, t1 = start_times[k], start_times[k] + pd.Timedelta(minutes=float(mins[k]))
            missing = pd.date_range(t0 + pd.Timedelta(minutes=1), t1 - pd.Timedelta(minutes=1), freq="1min")
            missing = missing[~market_closed_mask(missing)]  # minutes that should have had data
            if len(missing) <= max_gap_minutes:
                continue  # gap is just the weekend close
            # heuristic: a <=2h hole starting 20:00-22:59 UTC looks like the daily maintenance break
            if len(missing) <= 120 and 20 <= missing[0].hour <= 22:
                rep.daily_break_gaps += 1
                continue
            real_gaps.append((t0, len(missing)))
        rep.gaps_in_market_hours = len(real_gaps)
        if real_gaps:
            rep.largest_gap_minutes = float(max(m for _, m in real_gaps))
            rep.gap_samples = [f"{t:%Y-%m-%d %H:%M} (+{m}m missing)" for t, m in real_gaps[:5]]
        rep.first, rep.last = str(df.index[0]), str(df.index[-1])
    if len(df):
        rep.first, rep.last = str(df.index[0]), str(df.index[-1])
    rep.rows_out = len(df)
    log.info("validated %s: %s", name, {k: v for k, v in rep.to_dict().items() if k != "gap_samples"})
    return df, rep
