"""Shared test fixtures. All market data here is SYNTHETIC or hand-built."""
import pandas as pd

from gold_ai.data_engine.merger import merge_bid_ask
from gold_ai.data_engine.synthetic import generate_synthetic_m1


def small_m1(days: int = 12, seed: int = 3) -> pd.DataFrame:
    bid, ask = generate_synthetic_m1(days=days, seed=seed, missing_frac=0.0)
    m1, _ = merge_bid_ask(bid, ask)
    return m1


def flat_frame(opens, highs, lows, closes, spread=0.4, start="2026-02-02 10:00", freq="15min") -> pd.DataFrame:
    """Hand-built merged Bid/Ask frame from mid OHLC lists."""
    idx = pd.date_range(start, periods=len(opens), freq=freq, tz="UTC")
    h = spread / 2
    d = {}
    for name, vals in (("open", opens), ("high", highs), ("low", lows), ("close", closes)):
        d[f"bid_{name}"] = [v - h for v in vals]
        d[f"ask_{name}"] = [v + h for v in vals]
    df = pd.DataFrame(d, index=idx)
    df["volume"] = 100.0
    df["spread_open"] = spread
    df["spread_close"] = spread
    df["spread_mean"] = spread
    return df
