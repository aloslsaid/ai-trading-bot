"""Bar-based Bid/Ask backtester with realistic execution assumptions.

Execution model (no lookahead):
  * A signal is computed from data up to and including bar i (its close).
  * The order is filled at the OPEN of bar i+1: longs at ask_open + slippage, shorts at bid_open - slippage.
  * Longs exit on the BID side, shorts on the ASK side, so the spread is paid for real.
  * SL/TP are checked from the entry bar onward using that bar's high/low on the exit side.
  * If a bar could have touched both SL and TP, the STOP is assumed to hit first (conservative).
  * If a bar opens beyond a stop, the fill is at the (worse) open, plus slippage.
  * An opposite signal closes the position at the next open and reverses.
  * Commission is charged per lot per side. 1 lot = contract_size_oz ounces; minimum size 0.01 lot.
  * Sizing: risk_per_trade_pct of current equity divided by the stop distance, capped by max leverage.
  * A daily loss limit (UTC day) blocks new entries for the rest of that day.
Only ONE position at a time. Signal timeframe == execution timeframe in this milestone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_metrics


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.5
    max_leverage: float = 20.0
    slippage_usd: float = 0.05
    commission_per_lot_per_side: float = 3.5
    contract_size_oz: float = 100.0
    min_lot: float = 0.01
    max_daily_loss_pct: float = 3.0
    max_bars_in_trade: int | None = None

    @classmethod
    def from_settings(cls, settings) -> "BacktestConfig":
        c, r = settings["costs"], settings["risk"]
        return cls(
            initial_equity=r["initial_equity"],
            risk_per_trade_pct=r["risk_per_trade_pct"],
            max_leverage=r["max_leverage"],
            slippage_usd=c["slippage_usd"],
            commission_per_lot_per_side=c["commission_per_lot_per_side"],
            contract_size_oz=c["contract_size_oz"],
            max_daily_loss_pct=r["max_daily_loss_pct"],
        )


@dataclass
class Signals:
    """Strategy output aligned to the bar index. direction: +1 long, -1 short, 0 no action.
    sl_dist / tp_dist are price distances (USD/oz) from the FILL price; NaN tp_dist = no take profit.
    A valid sl_dist is required to open a trade (it defines position size)."""

    direction: pd.Series
    sl_dist: pd.Series
    tp_dist: pd.Series


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    metrics: dict
    skipped: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, sig: Signals, cfg: BacktestConfig, name: str = "strategy") -> BacktestResult:
    idx = df.index
    n = len(df)
    b_o, b_h, b_l, b_c = (df[f"bid_{k}"].to_numpy(float) for k in ("open", "high", "low", "close"))
    a_o, a_h, a_l, a_c = (df[f"ask_{k}"].to_numpy(float) for k in ("open", "high", "low", "close"))
    d = sig.direction.reindex(idx).fillna(0).to_numpy(int)
    sl_d = sig.sl_dist.reindex(idx).to_numpy(float)
    tp_d = sig.tp_dist.reindex(idx).to_numpy(float)
    days = idx.floor("D").to_numpy()
    slip = cfg.slippage_usd

    equity = cfg.initial_equity
    eq_curve = np.full(n, np.nan)
    trades: list[dict] = []
    skipped = {"no_stop": 0, "min_size": 0, "daily_loss_limit": 0}
    pos = 0  # +1/-1/0
    entry = sl = tp = size = risk_amt = 0.0
    entry_i = 0
    entry_comm = 0.0
    day_start_equity = equity
    cur_day = None
    day_blocked = False
    pending_close = False  # opposite signal seen at previous close

    def close_trade(i: int, px: float, reason: str) -> None:
        nonlocal equity, pos
        gross = pos * (px - entry) * size
        comm = entry_comm * 2
        pnl = gross - comm
        equity += pnl
        trades.append(
            {
                "strategy": name, "entry_time": idx[entry_i], "exit_time": idx[i], "direction": "LONG" if pos > 0 else "SHORT",
                "entry": entry, "exit": px, "size_oz": size, "lots": size / cfg.contract_size_oz, "gross_pnl": gross,
                "commission": comm, "pnl": pnl, "risk_amount": risk_amt, "r_multiple": pnl / risk_amt if risk_amt else np.nan,
                "bars": i - entry_i, "reason": reason,
            }
        )
        pos = 0

    def try_open(i: int, direction: int) -> bool:
        nonlocal pos, entry, sl, tp, size, risk_amt, entry_i, entry_comm
        s = sl_d[i - 1]
        if not np.isfinite(s) or s <= 0:
            skipped["no_stop"] += 1
            return False
        px = a_o[i] + slip if direction > 0 else b_o[i] - slip
        risk = equity * cfg.risk_per_trade_pct / 100
        oz = min(risk / s, equity * cfg.max_leverage / px)
        lots = math.floor(oz / cfg.contract_size_oz / cfg.min_lot + 1e-9) * cfg.min_lot
        if lots < cfg.min_lot - 1e-12:
            skipped["min_size"] += 1
            return False
        pos, entry, size, entry_i = direction, px, lots * cfg.contract_size_oz, i
        sl = px - direction * s
        t = tp_d[i - 1]
        tp = px + direction * t if np.isfinite(t) and t > 0 else np.nan
        risk_amt = size * s
        entry_comm = lots * cfg.commission_per_lot_per_side
        return True

    for i in range(n):
        if days[i] != cur_day:
            cur_day, day_start_equity, day_blocked = days[i], equity, False

        # 1) execute orders scheduled from the previous bar's close, at this bar's open
        if i > 0:
            prev_sig = d[i - 1]
            if pos != 0 and pending_close:
                px = b_o[i] - slip if pos > 0 else a_o[i] + slip
                close_trade(i, px, "reverse_signal")
                pending_close = False
            if pos == 0 and prev_sig != 0:
                if day_blocked:
                    skipped["daily_loss_limit"] += 1
                else:
                    try_open(i, prev_sig)
        # 2) manage open position inside this bar
        if pos != 0:
            if pos > 0:
                if b_o[i] <= sl:
                    close_trade(i, b_o[i] - slip, "stop_gap")
                elif np.isfinite(tp) and b_o[i] >= tp:
                    close_trade(i, tp, "tp_gap")
                elif b_l[i] <= sl:
                    close_trade(i, sl - slip, "stop")
                elif np.isfinite(tp) and b_h[i] >= tp:
                    close_trade(i, tp, "tp")
            else:
                if a_o[i] >= sl:
                    close_trade(i, a_o[i] + slip, "stop_gap")
                elif np.isfinite(tp) and a_o[i] <= tp:
                    close_trade(i, tp, "tp_gap")
                elif a_h[i] >= sl:
                    close_trade(i, sl + slip, "stop")
                elif np.isfinite(tp) and a_l[i] <= tp:
                    close_trade(i, tp, "tp")
            if pos != 0 and cfg.max_bars_in_trade and i - entry_i >= cfg.max_bars_in_trade:
                close_trade(i, (b_c[i] - slip) if pos > 0 else (a_c[i] + slip), "max_bars")
            if pos != 0 and d[i] == -pos:
                pending_close = True
        # 3) daily loss limit and mark-to-market equity at bar close
        unreal = 0.0
        if pos != 0:
            mark = b_c[i] if pos > 0 else a_c[i]
            unreal = pos * (mark - entry) * size - entry_comm * 2
        eq_curve[i] = equity + unreal
        if equity + unreal <= day_start_equity * (1 - cfg.max_daily_loss_pct / 100):
            day_blocked = True

    if pos != 0:  # close at the last available quote so the final equity is realized
        close_trade(n - 1, (b_c[-1] - slip) if pos > 0 else (a_c[-1] + slip), "end_of_data")
        eq_curve[-1] = equity

    tdf = pd.DataFrame(trades)
    eq = pd.Series(eq_curve, index=idx, name="equity")
    return BacktestResult(tdf, eq, compute_metrics(tdf, eq, cfg.initial_equity), skipped)
