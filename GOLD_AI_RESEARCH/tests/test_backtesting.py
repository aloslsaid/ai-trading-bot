import unittest

import numpy as np
import pandas as pd

from gold_ai.backtesting.baselines import random_entries, sma_cross_atr
from gold_ai.backtesting.engine import BacktestConfig, Signals, run_backtest
from gold_ai.backtesting.splits import time_split
from tests.helpers import flat_frame, small_m1


def _sig(df, direction_at: dict, sl=2.0, tp=4.0):
    d = pd.Series(0, index=df.index)
    for k, v in direction_at.items():
        d.iloc[k] = v
    return Signals(d, pd.Series(sl, index=df.index), pd.Series(tp, index=df.index))


CFG = BacktestConfig(slippage_usd=0.0, commission_per_lot_per_side=0.0, max_daily_loss_pct=100.0)


class ExecutionTests(unittest.TestCase):
    def test_long_fills_at_ask_open_next_bar_and_tp_exits_at_bid(self):
        # mid prices; spread 0.4 -> ask = mid+0.2, bid = mid-0.2
        o = [100, 100, 101, 101, 101]
        h = [100, 101, 106, 101, 101]
        l = [100, 100, 100.5, 101, 101]
        c = [100, 100.5, 105, 101, 101]
        df = flat_frame(o, h, l, c)
        r = run_backtest(df, _sig(df, {0: 1}), CFG)
        t = r.trades.iloc[0]
        self.assertEqual(t["entry_time"], df.index[1])  # next bar, not signal bar
        self.assertAlmostEqual(t["entry"], 100.2)  # ask open of bar 1
        self.assertAlmostEqual(t["exit"], 100.2 + 4.0)  # tp level, hit on bar 2 (bid_high 105.8 >= 104.2)
        self.assertEqual(t["reason"], "tp")

    def test_short_fills_at_bid_and_exits_at_ask_stop(self):
        o = [100, 100, 100, 100]
        h = [100, 100, 103, 100]
        l = [100, 99.5, 99.8, 100]
        c = [100, 100, 102.5, 100]
        df = flat_frame(o, h, l, c)
        r = run_backtest(df, _sig(df, {0: -1}, sl=2.0, tp=10.0), CFG)
        t = r.trades.iloc[0]
        self.assertAlmostEqual(t["entry"], 99.8)  # bid open
        self.assertAlmostEqual(t["exit"], 99.8 + 2.0)  # stop at entry+sl, ask_high 103.2 >= 101.8
        self.assertEqual(t["reason"], "stop")
        self.assertLess(t["pnl"], 0)

    def test_stop_wins_when_both_touched_in_same_bar(self):
        o = [100, 100, 100]
        h = [100, 110, 100]
        l = [100, 90, 100]
        c = [100, 100, 100]
        df = flat_frame(o, h, l, c)
        r = run_backtest(df, _sig(df, {0: 1}, sl=2.0, tp=3.0), CFG)
        self.assertEqual(r.trades.iloc[0]["reason"], "stop")

    def test_gap_through_stop_fills_at_open_with_slippage(self):
        o = [100, 100, 90, 90]
        h = [100, 100.5, 90.5, 90]
        l = [100, 99.5, 89.5, 90]
        c = [100, 100, 90, 90]
        df = flat_frame(o, h, l, c)
        cfg = BacktestConfig(slippage_usd=0.1, commission_per_lot_per_side=0.0, max_daily_loss_pct=100.0)
        r = run_backtest(df, _sig(df, {0: 1}, sl=2.0, tp=50.0), cfg)
        t = r.trades.iloc[0]
        self.assertEqual(t["reason"], "stop_gap")
        self.assertAlmostEqual(t["exit"], 89.8 - 0.1)  # bid open 89.8 minus slippage

    def test_commission_and_sizing(self):
        o = [100, 100, 100, 100]
        h = [100, 100, 100, 100]
        l = [100, 100, 100, 100]
        c = [100, 100, 100, 100]
        df = flat_frame(o, h, l, c)
        cfg = BacktestConfig(initial_equity=10_000, risk_per_trade_pct=1.0, slippage_usd=0.0, commission_per_lot_per_side=5.0, max_daily_loss_pct=100.0)
        r = run_backtest(df, _sig(df, {0: 1}, sl=2.0, tp=None or np.nan), cfg)
        t = r.trades.iloc[0]
        # risk $100 / $2 stop = 50 oz = 0.5 lot; commission 0.5 * 5 * 2 sides = 5
        self.assertAlmostEqual(t["lots"], 0.5)
        self.assertAlmostEqual(t["commission"], 5.0)
        # flat mid: only spread cost: bought at ask 100.2, sold at bid 99.8 at end -> -0.4 * 50 - 5
        self.assertAlmostEqual(t["pnl"], -0.4 * 50 - 5.0)
        self.assertEqual(t["reason"], "end_of_data")

    def test_missing_stop_skips_trade(self):
        df = flat_frame([100] * 4, [100] * 4, [100] * 4, [100] * 4)
        sig = Signals(pd.Series([1, 0, 0, 0], index=df.index), pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index))
        r = run_backtest(df, sig, CFG)
        self.assertTrue(r.trades.empty)
        self.assertEqual(r.skipped["no_stop"], 1)

    def test_leverage_cap(self):
        df = flat_frame([2000] * 4, [2000] * 4, [2000] * 4, [2000] * 4)
        cfg = BacktestConfig(initial_equity=10_000, risk_per_trade_pct=50.0, max_leverage=5.0, slippage_usd=0.0, commission_per_lot_per_side=0.0, max_daily_loss_pct=100.0)
        r = run_backtest(df, _sig(df, {0: 1}, sl=0.5, tp=np.nan), cfg)
        self.assertLessEqual(r.trades.iloc[0]["size_oz"] * 2000.2, 10_000 * 5.0 + 1e-6)

    def test_reverse_signal_closes_and_reopens(self):
        o = [100, 100, 100, 100, 100]
        df = flat_frame(o, o, o, o)
        r = run_backtest(df, _sig(df, {0: 1, 2: -1}, sl=5.0, tp=np.nan), CFG)
        self.assertGreaterEqual(len(r.trades), 2)
        self.assertEqual(r.trades.iloc[0]["reason"], "reverse_signal")
        self.assertEqual(r.trades.iloc[1]["direction"], "SHORT")

    def test_daily_loss_limit_blocks_entries(self):
        o = [100, 100, 100, 100, 100, 100, 100, 100]
        h = [100, 100, 100, 100, 100, 100, 100, 100]
        l = [100, 99, 99, 99, 99, 99, 99, 99]
        df = flat_frame(o, h, l, o)
        cfg = BacktestConfig(slippage_usd=0.0, commission_per_lot_per_side=0.0, risk_per_trade_pct=2.0, max_daily_loss_pct=1.5)
        r = run_backtest(df, _sig(df, {0: 1, 2: 1, 4: 1, 6: 1}, sl=0.5, tp=np.nan), cfg)
        self.assertLess(len(r.trades), 4)
        self.assertGreater(r.skipped["daily_loss_limit"], 0)


class NoLookaheadTests(unittest.TestCase):
    def test_future_data_does_not_change_past_trades(self):
        m1 = small_m1(days=14)
        from gold_ai.data_engine.resampler import resample_bid_ask

        df = resample_bid_ask(m1, "M15")
        cut = len(df) - 200
        cfg = BacktestConfig()
        full = run_backtest(df, sma_cross_atr(df), cfg)
        part_df = df.iloc[:cut]
        part = run_backtest(part_df, sma_cross_atr(part_df), cfg)
        last_exit = part_df.index[-1]
        a = full.trades[full.trades["exit_time"] < last_exit].reset_index(drop=True)
        b = part.trades[part.trades["exit_time"] < last_exit].reset_index(drop=True)
        self.assertGreater(len(b), 3)
        pd.testing.assert_frame_equal(a[["entry_time", "exit_time", "entry", "exit", "pnl"]], b[["entry_time", "exit_time", "entry", "exit", "pnl"]])

    def test_equity_consistent_with_trades(self):
        from gold_ai.data_engine.resampler import resample_bid_ask

        df = resample_bid_ask(small_m1(days=14), "M15")
        r = run_backtest(df, sma_cross_atr(df), BacktestConfig())
        self.assertAlmostEqual(r.metrics["final_equity"], 10_000 + r.trades["pnl"].sum(), places=6)
        self.assertAlmostEqual(r.equity.dropna().iloc[-1], r.metrics["final_equity"], places=6)


class SplitAndBaselineTests(unittest.TestCase):
    def test_split_chronological_no_overlap(self):
        df = flat_frame(list(range(100, 200)), list(range(100, 200)), list(range(100, 200)), list(range(100, 200)))
        s = time_split(df, 0.6, 0.2)
        self.assertEqual(len(s["train"]) + len(s["validation"]) + len(s["test"]), 100)
        self.assertLess(s["train"].index[-1], s["validation"].index[0])
        self.assertLess(s["validation"].index[-1], s["test"].index[0])
        with self.assertRaises(ValueError):
            time_split(df, 0.8, 0.2)

    def test_random_baseline_is_seeded(self):
        df = flat_frame(list(range(100, 160)), list(range(101, 161)), list(range(99, 159)), list(range(100, 160)))
        a, b = random_entries(df, 0.2, seed=5), random_entries(df, 0.2, seed=5)
        self.assertTrue(a.direction.equals(b.direction))


if __name__ == "__main__":
    unittest.main()
