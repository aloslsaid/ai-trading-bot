"""Merge Bid and Ask M1 frames into one frame. Only timestamps present in BOTH sides are kept."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

BA_COLS = ["bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close", "volume", "spread_open", "spread_close"]


def merge_bid_ask(bid: pd.DataFrame, ask: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    b = bid.add_prefix("bid_").rename(columns={"bid_volume": "bid_vol"})
    a = ask.add_prefix("ask_").rename(columns={"ask_volume": "ask_vol"})
    m = b.join(a, how="inner")
    stats = {
        "bid_rows": len(bid),
        "ask_rows": len(ask),
        "merged_rows": len(m),
        "bid_only_dropped": int(len(bid) - len(m)),
        "ask_only_dropped": int(len(ask) - len(m)),
    }
    m["volume"] = m[["bid_vol", "ask_vol"]].sum(axis=1)
    m = m.drop(columns=["bid_vol", "ask_vol"])
    m["spread_open"] = m["ask_open"] - m["bid_open"]
    m["spread_close"] = m["ask_close"] - m["bid_close"]
    crossed = int((m["spread_close"] < 0).sum() + (m["spread_open"] < 0).sum())
    stats["crossed_quotes_dropped"] = crossed
    if crossed:
        m = m[(m["spread_close"] >= 0) & (m["spread_open"] >= 0)]
        log.warning("dropped %d rows with ask < bid", crossed)
    m.index.name = "timestamp"
    log.info("bid/ask merge: %s", stats)
    return m[BA_COLS], stats
