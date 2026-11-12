import unittest

import numpy as np
import pandas as pd

from gold_ai.indicators import classic as ind
from tests.helpers import small_m1


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        self.df = ind.mid_ohlc(small_m1())

    def test_sma_matches_manual(self):
        c = pd.Series(np.arange(1.0, 11.0))
        self.assertAlmostEqual(ind.sma(c, 5).iloc[-1], 8.0)
        self.assertTrue(np.isnan(ind.sma(c, 5).iloc[3]))

    def test_rsi_bounds_and_extremes(self):
        up = pd.Series(np.arange(1.0, 60.0))
        self.assertAlmostEqual(ind.rsi(up, 14).iloc[-1], 100.0)
        down = pd.Series(np.arange(60.0, 1.0, -1))
        self.assertAlmostEqual(ind.rsi(down, 14).iloc[-1], 0.0, places=6)
        r = ind.rsi(self.df["close"]).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())

    def test_ranges(self):
        h, l, c = self.df["high"], self.df["low"], self.df["close"]
        self.assertTrue((ind.atr(h, l, c).dropna() > 0).all())
        st = ind.stochastic(h, l, c)["stoch_k"].dropna()
        self.assertTrue(((st >= -1e-9) & (st <= 100 + 1e-9)).all())
        w = ind.williams_r(h, l, c).dropna()
        self.assertTrue(((w <= 1e-9) & (w >= -100 - 1e-9)).all())
        a = ind.adx(h, l, c)["adx"].dropna()
        self.assertTrue(((a >= 0) & (a <= 100)).all())

    def test_macd_hist_identity(self):
        m = ind.macd(self.df["close"]).dropna()
        self.assertTrue(np.allclose(m["macd_hist"], m["macd"] - m["macd_signal"]))

    def test_bollinger_order(self):
        b = ind.bollinger(self.df["close"]).dropna()
        self.assertTrue((b["bb_upper"] >= b["bb_mid"]).all() and (b["bb_mid"] >= b["bb_lower"]).all())

    def test_no_lookahead_all_indicators(self):
        full = ind.compute_classic(small_m1())
        cut = 3000
        part_src = small_m1().iloc[:cut]
        part = ind.compute_classic(part_src)
        a, b = full.iloc[:cut], part
        for col in a.columns:
            x, y = a[col].to_numpy(float), b[col].to_numpy(float)
            self.assertTrue(np.allclose(x, y, equal_nan=True, rtol=1e-9, atol=1e-9), f"{col} depends on future data")

    def test_vwap_resets_daily(self):
        v = ind.compute_classic(small_m1())["vwap"]
        first_of_day = v.groupby(v.index.floor("D")).head(1)
        mid = ind.mid_ohlc(small_m1())
        tp = (mid["high"] + mid["low"] + mid["close"]) / 3
        self.assertTrue(np.allclose(first_of_day.to_numpy(), tp.loc[first_of_day.index].to_numpy()))


if __name__ == "__main__":
    unittest.main()
