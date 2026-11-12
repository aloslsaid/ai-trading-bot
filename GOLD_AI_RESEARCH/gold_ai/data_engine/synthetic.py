"""SYNTHETIC XAU/USD-like data generator. FOR TESTS AND DEMOS ONLY.

This is random data with stochastic volatility, closed weekends and a few missing minutes.
It is NOT market data. Anything computed from it says nothing about real gold, and
every output derived from it is labeled SYNTHETIC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .validator import market_closed_mask

SYNTHETIC_LABEL = "SYNTHETIC DATA - NOT REAL MARKET DATA"


def generate_synthetic_m1(start: str = "2026-01-05", days: int = 180, seed: int = 7, start_price: float = 2600.0, missing_frac: float = 0.0005) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min", tz="UTC")
    idx = idx[~market_closed_mask(idx)]
    n = len(idx)
    # stochastic volatility (AR(1) in log-vol) + slowly switching drift regimes
    logv = np.zeros(n)
    eps = rng.normal(0, 0.03, n)
    for i in range(1, n):
        logv[i] = 0.995 * logv[i - 1] + eps[i]
    sigma = 0.25 * np.exp(logv)  # USD per minute
    regime = np.repeat(rng.choice([-1.0, 0.0, 1.0], size=n // 3000 + 1), 3000)[:n]
    ret = rng.normal(0, 1, n) * sigma + regime * 0.004
    mid_close = start_price + np.cumsum(ret)
    mid_open = np.concatenate([[start_price], mid_close[:-1]])
    wick = np.abs(rng.normal(0, 1, (2, n))) * sigma * 0.6
    mid_high = np.maximum(mid_open, mid_close) + wick[0]
    mid_low = np.minimum(mid_open, mid_close) - wick[1]
    hour = idx.hour.to_numpy()
    base_spread = np.where((hour >= 21) | (hour < 1), 0.45, 0.28)
    spread = np.clip(base_spread + rng.normal(0, 0.03, n), 0.10, None)
    vol = rng.integers(20, 400, n).astype(float)
    keep = rng.random(n) > missing_frac
    frames = []
    for sign in (-1, 1):
        half = sign * spread / 2
        f = pd.DataFrame({"open": mid_open + half, "high": mid_high + half, "low": mid_low + half, "close": mid_close + half, "volume": vol}, index=idx)
        frames.append(f[keep].rename_axis("timestamp"))
    return frames[0], frames[1]  # bid, ask
