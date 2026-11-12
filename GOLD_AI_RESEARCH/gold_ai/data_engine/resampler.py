"""Multi-timeframe dataset builder (M1 -> M5 ... W1) from merged Bid/Ask M1 data.

Rules: UTC only; bins that contain no M1 data are NOT created (no invented candles);
aggregation per side (bid/ask separately); spread = mean of M1 closing spreads in the bar.
`day_offset_hours` shifts H4/D1 bin boundaries (e.g. 22 or 21 to mimic the 17:00 New York
FX day roll). W1 bins start Sunday 00:00 UTC (+ offset).
"""
from __future__ import annotations

import pandas as pd

TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
ALL_TFS = [*TF_MINUTES, "W1"]


def resample_bid_ask(m1: pd.DataFrame, tf: str, day_offset_hours: int = 0) -> pd.DataFrame:
    if tf == "M1":
        out = m1.copy()
        out["spread_mean"] = out["spread_close"]
        out["n_m1"] = 1
        return out
    agg = {}
    for side in ("bid", "ask"):
        agg[f"{side}_open"] = "first"
        agg[f"{side}_high"] = "max"
        agg[f"{side}_low"] = "min"
        agg[f"{side}_close"] = "last"
    agg["volume"] = "sum"
    agg["spread_close"] = "mean"
    agg["spread_open"] = "first"
    shift = pd.Timedelta(hours=day_offset_hours)
    if tf == "W1":
        shifted = m1.index - shift
        key = (shifted.floor("D") - pd.to_timedelta((shifted.day_of_week + 1) % 7, unit="D")) + shift
        g = m1.groupby(key)
        out = g.agg(agg)
        out["n_m1"] = g.size()
    else:
        minutes = TF_MINUTES[tf]
        offset = f"{day_offset_hours}h" if minutes >= 240 else None
        r = m1.resample(f"{minutes}min", offset=offset, label="left", closed="left")
        out = r.agg(agg)
        out["n_m1"] = r.size()
    out = out[out["n_m1"] > 0].rename(columns={"spread_close": "spread_mean"})
    out["spread_close"] = out["spread_mean"]
    out.index.name = "timestamp"
    return out


def build_all(m1: pd.DataFrame, timeframes: list[str] | None = None, day_offset_hours: int = 0) -> dict[str, pd.DataFrame]:
    return {tf: resample_bid_ask(m1, tf, day_offset_hours) for tf in (timeframes or ALL_TFS)}
