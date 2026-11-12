"""Yahoo Finance historical OHLC for XAU (GC=F futures) when Dukascopy is unavailable.

Produces Bid/Ask M1 by distributing each H1 bar across the hour with a fixed half-spread.
This is a practical research substitute; it is NOT true tick data and is labeled as
provider=Yahoo in the build summary. Spread is synthetic (configurable).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_SYMBOL = "GC=F"
DEFAULT_HALF_SPREAD = 0.35  # USD/oz half-spread approximation for research


def fetch_hourly(
    start: datetime | str,
    end: datetime | str | None = None,
    symbol: str = DEFAULT_SYMBOL,
) -> pd.DataFrame:
    """Download hourly OHLCV from Yahoo Finance (GC=F by default)."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for Yahoo downloads. Install with: pip install yfinance"
        ) from exc

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    log.info("Downloading %s hourly %s -> %s via Yahoo Finance", symbol, start_s, end_s)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_s, end=end_s, interval="1h", auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"Yahoo returned no data for {symbol} in [{start_s}, {end_s}]")
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    df = df[["open", "high", "low", "close", "volume"]].copy()
    # Normalize timezone to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    log.info("Yahoo %s: %d hourly bars %s -> %s", symbol, len(df), df.index[0], df.index[-1])
    return df


def hourly_to_bid_ask_m1(
    hourly: pd.DataFrame,
    half_spread: float = DEFAULT_HALF_SPREAD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand each H1 bar into 60 M1 bars (flat OHLC within the hour) and split bid/ask.

    Volume is distributed evenly. This is an approximation for research when ticks
    are unavailable; do not treat the intra-hour path as true market microstructure.
    """
    if half_spread < 0:
        raise ValueError("half_spread must be non-negative")
    if hourly.empty:
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        return empty, empty.copy()

    rows_bid = []
    rows_ask = []
    idx_list = []
    for ts, row in hourly.iterrows():
        base = pd.Timestamp(ts).tz_convert("UTC") if ts.tzinfo else pd.Timestamp(ts, tz="UTC")
        o, h, l, c, v = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0) or 0)
        vol_m1 = v / 60.0
        for m in range(60):
            t = base + pd.Timedelta(minutes=m)
            # Flat bar approximation inside the hour
            bid_o = o - half_spread
            bid_h = h - half_spread
            bid_l = l - half_spread
            bid_c = c - half_spread
            ask_o = o + half_spread
            ask_h = h + half_spread
            ask_l = l + half_spread
            ask_c = c + half_spread
            rows_bid.append([bid_o, bid_h, bid_l, bid_c, vol_m1])
            rows_ask.append([ask_o, ask_h, ask_l, ask_c, vol_m1])
            idx_list.append(t)

    index = pd.DatetimeIndex(idx_list, name="timestamp", tz="UTC")
    cols = ["open", "high", "low", "close", "volume"]
    bid = pd.DataFrame(rows_bid, columns=cols, index=index)
    ask = pd.DataFrame(rows_ask, columns=cols, index=index)
    return bid, ask


def download_and_build_m1(
    start: datetime | str,
    end: datetime | str | None = None,
    symbol: str = DEFAULT_SYMBOL,
    half_spread: float = DEFAULT_HALF_SPREAD,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full path: Yahoo H1 -> approximated Bid/Ask M1 + provenance metadata."""
    hourly = fetch_hourly(start, end, symbol)
    bid, ask = hourly_to_bid_ask_m1(hourly, half_spread)
    meta = {
        "provider": "Yahoo Finance",
        "symbol": symbol,
        "source_interval": "1h",
        "approximation": "flat H1 bars expanded to M1; synthetic fixed half-spread",
        "half_spread_usd": half_spread,
        "hourly_bars": len(hourly),
        "m1_bars": len(bid),
        "first": str(bid.index[0]) if len(bid) else None,
        "last": str(bid.index[-1]) if len(bid) else None,
        "note": "Not true Dukascopy tick data. Suitable for research training only.",
    }
    return bid, ask, meta
