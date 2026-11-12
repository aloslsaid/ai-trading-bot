"""Performance metrics from a trade list and an equity curve."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def max_drawdown(eq: pd.Series) -> tuple[float, float]:
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min()), float((dd / peak).min() * 100)


def _side(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "net_profit": 0.0, "win_rate_pct": float("nan")}
    return {"trades": int(len(t)), "net_profit": float(t["pnl"].sum()), "win_rate_pct": float((t["pnl"] > 0).mean() * 100)}


def compute_metrics(trades: pd.DataFrame, equity: pd.Series, initial_equity: float) -> dict:
    eq = equity.dropna()
    final = float(eq.iloc[-1]) if len(eq) else initial_equity
    dd_usd, dd_pct = max_drawdown(eq) if len(eq) else (0.0, 0.0)
    m: dict = {
        "initial_equity": initial_equity,
        "final_equity": final,
        "net_profit": final - initial_equity,
        "return_pct": (final / initial_equity - 1) * 100,
        "max_drawdown_usd": dd_usd,
        "max_drawdown_pct": dd_pct,
        "trades": int(len(trades)),
    }
    if len(eq) > 2:
        daily = eq.resample("1D").last().dropna().pct_change().dropna()
        sd = daily.std(ddof=1)
        dn = daily[daily < 0].std(ddof=1)
        m["sharpe"] = float(daily.mean() / sd * np.sqrt(252)) if sd and np.isfinite(sd) else float("nan")
        m["sortino"] = float(daily.mean() / dn * np.sqrt(252)) if dn and np.isfinite(dn) else float("nan")
    else:
        m["sharpe"] = m["sortino"] = float("nan")
    if trades.empty:
        m.update(win_rate_pct=float("nan"), profit_factor=float("nan"), expectancy_usd=float("nan"), expectancy_r=float("nan"),
                 avg_win=float("nan"), avg_loss=float("nan"), long=_side(trades), short=_side(trades), avg_bars=float("nan"))
        return m
    w, l = trades[trades["pnl"] > 0]["pnl"], trades[trades["pnl"] <= 0]["pnl"]
    m.update(
        win_rate_pct=float(len(w) / len(trades) * 100),
        profit_factor=_ratio(w.sum(), -l.sum()),
        expectancy_usd=float(trades["pnl"].mean()),
        expectancy_r=float(trades["r_multiple"].mean()),
        avg_win=float(w.mean()) if len(w) else 0.0,
        avg_loss=float(l.mean()) if len(l) else 0.0,
        avg_bars=float(trades["bars"].mean()),
        long=_side(trades[trades["direction"] == "LONG"]),
        short=_side(trades[trades["direction"] == "SHORT"]),
    )
    return m
