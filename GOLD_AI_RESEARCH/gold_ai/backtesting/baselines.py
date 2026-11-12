"""Baselines and a sample strategy used to sanity check the engine and the reports.

These are NOT trading recommendations. `sma_cross_atr` is only a plumbing example.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.classic import atr, mid_ohlc, sma
from .engine import BacktestConfig, Signals
from .metrics import compute_metrics


def sma_cross_atr(df: pd.DataFrame, fast: int = 20, slow: int = 50, atr_n: int = 14, sl_atr: float = 1.5, tp_atr: float = 3.0) -> Signals:
    m = mid_ohlc(df)
    f, s = sma(m["close"], fast), sma(m["close"], slow)
    cross_up = (f > s) & (f.shift(1) <= s.shift(1))
    cross_dn = (f < s) & (f.shift(1) >= s.shift(1))
    direction = pd.Series(0, index=df.index).mask(cross_up, 1).mask(cross_dn, -1)
    a = atr(m["high"], m["low"], m["close"], atr_n)
    return Signals(direction, a * sl_atr, a * tp_atr)


def random_entries(df: pd.DataFrame, prob: float = 0.01, atr_n: int = 14, sl_atr: float = 1.5, tp_atr: float = 3.0, seed: int = 0) -> Signals:
    """Baseline: random direction entries with the same stop/target structure."""
    rng = np.random.default_rng(seed)
    m = mid_ohlc(df)
    a = atr(m["high"], m["low"], m["close"], atr_n)
    fire = rng.random(len(df)) < prob
    dirs = rng.choice([-1, 1], len(df))
    return Signals(pd.Series(np.where(fire, dirs, 0), index=df.index), a * sl_atr, a * tp_atr)


def buy_and_hold(df: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """Buy at first ask open, sell at last bid close, with the same sizing cap (max leverage) and costs."""
    entry = df["ask_open"].iloc[0] + cfg.slippage_usd
    exit_ = df["bid_close"].iloc[-1] - cfg.slippage_usd
    lots = np.floor(cfg.initial_equity * cfg.max_leverage / entry / cfg.contract_size_oz / cfg.min_lot) * cfg.min_lot
    size = lots * cfg.contract_size_oz
    pnl = (exit_ - entry) * size - 2 * lots * cfg.commission_per_lot_per_side
    mark = (df["bid_close"] - entry) * size
    eq = cfg.initial_equity + mark
    trades = pd.DataFrame([{"pnl": pnl, "r_multiple": np.nan, "bars": len(df), "direction": "LONG"}])
    m = compute_metrics(trades, eq, cfg.initial_equity)
    m["net_profit"], m["final_equity"] = pnl, cfg.initial_equity + pnl
    m["return_pct"] = pnl / cfg.initial_equity * 100
    return m
