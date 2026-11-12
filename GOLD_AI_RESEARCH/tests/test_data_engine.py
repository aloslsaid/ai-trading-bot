import lzma
import struct
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gold_ai.data_engine.dukascopy import decode_bi5, hour_url, ticks_to_m1
from gold_ai.data_engine.importer import read_ohlc_csv
from gold_ai.data_engine.merger import merge_bid_ask
from gold_ai.data_engine.resampler import resample_bid_ask
from gold_ai.data_engine.validator import validate_ohlc
from tests.helpers import small_m1


def _ohlc(rows, start="2026-02-02 10:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0).rename_axis("timestamp")


class DukascopyTests(unittest.TestCase):
    def test_url_uses_zero_based_month(self):
        u = hour_url("XAUUSD", datetime(2026, 3, 9, 5, tzinfo=timezone.utc))
        self.assertTrue(u.endswith("/XAUUSD/2026/02/09/05h_ticks.bi5"))

    def test_decode_and_aggregate(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        recs = [(0, 2800500, 2800000, 1.0, 2.0), (30_000, 2801000, 2800400, 1.0, 1.0), (60_000, 2802000, 2801000, 1.0, 1.0)]
        payload = lzma.compress(b"".join(struct.pack(">IIIff", *r) for r in recs))
        t = decode_bi5(payload, hour, 1000.0)
        self.assertEqual(len(t), 3)
        self.assertAlmostEqual(t["bid"].iloc[0], 2800.0)
        self.assertAlmostEqual(t["ask"].iloc[0], 2800.5)
        bid, ask = ticks_to_m1(t)
        self.assertEqual(len(bid), 2)  # minute 10:00 and 10:01 only
        self.assertAlmostEqual(bid["high"].iloc[0], 2800.4)
        self.assertAlmostEqual(ask["close"].iloc[0], 2801.0)

    def test_empty_hour_and_corrupt(self):
        hour = datetime(2026, 2, 7, 10, tzinfo=timezone.utc)
        self.assertTrue(decode_bi5(b"", hour).empty)
        with self.assertRaises(ValueError):
            decode_bi5(lzma.compress(b"123"), hour)

    def test_no_invented_minutes(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        recs = [(0, 2800500, 2800000, 1.0, 1.0), (5 * 60_000, 2800500, 2800000, 1.0, 1.0)]
        t = decode_bi5(lzma.compress(b"".join(struct.pack(">IIIff", *r) for r in recs)), hour)
        bid, _ = ticks_to_m1(t)
        self.assertEqual(len(bid), 2)


class ImporterTests(unittest.TestCase):
    def test_dukascopy_csv_format(self):
        import tempfile, os
        text = "Gmt time,Open,High,Low,Close,Volume\n02.02.2026 10:00:00.000,2800.1,2800.9,2799.8,2800.5,12.5\n02.02.2026 10:01:00.000,2800.5,2801.0,2800.2,2800.7,8.0\nbad,1,2,3,4,5\n"
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bid.csv")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            df = read_ohlc_csv(p)
        self.assertEqual(len(df), 2)
        self.assertEqual(str(df.index[0]), "2026-02-02 10:00:00+00:00")


class ValidatorTests(unittest.TestCase):
    def test_duplicates_unsorted_and_bad_rows(self):
        df = _ohlc([(10, 11, 9, 10), (10, 11, 9, 10), (10, 8, 9, 10), (-1, 1, 1, 1), (10, 12, 9, 11)])
        df = pd.concat([df.iloc[[4]], df.iloc[:4]])  # unsorted
        df = pd.concat([df, df.iloc[[1]]])  # duplicate timestamp
        out, rep = validate_ohlc(df, "t")
        self.assertTrue(rep.unsorted_fixed)
        self.assertEqual(rep.duplicates_dropped, 1)
        self.assertEqual(rep.nonpositive_dropped, 1)
        self.assertEqual(rep.ohlc_inconsistent_dropped, 1)
        self.assertTrue(out.index.is_monotonic_increasing)

    def test_gap_detection_ignores_weekend_but_flags_midweek_gap(self):
        wk = pd.date_range("2026-02-02 08:00", "2026-02-02 12:00", freq="1min", tz="UTC")
        after_gap = pd.date_range("2026-02-02 14:00", "2026-02-02 14:30", freq="1min", tz="UTC")
        fri = pd.date_range("2026-02-06 20:00", "2026-02-06 20:59", freq="1min", tz="UTC")
        sun = pd.date_range("2026-02-08 23:00", "2026-02-08 23:30", freq="1min", tz="UTC")
        idx = wk.append(after_gap).append(fri).append(sun)
        df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1.0}, index=idx)
        _, rep = validate_ohlc(df, "g")
        # 2h midweek gap + the Tuesday->Friday hole are real gaps; the Friday->Sunday close is not
        self.assertGreaterEqual(rep.gaps_in_market_hours, 1)
        sample = " ".join(rep.gap_samples)
        self.assertIn("2026-02-02 12:00", sample)
        self.assertNotIn("2026-02-06 20:59", sample)

    def test_requires_tz_aware(self):
        df = _ohlc([(1, 2, 0.5, 1)])
        df.index = df.index.tz_localize(None)
        with self.assertRaises(ValueError):
            validate_ohlc(df, "x")


class MergerTests(unittest.TestCase):
    def test_inner_join_and_spread(self):
        bid = _ohlc([(10, 11, 9, 10), (10, 11, 9, 10), (10, 11, 9, 10)])
        ask = bid.iloc[1:] + 0.3
        ask["volume"] = 1.0
        m, st = merge_bid_ask(bid, ask)
        self.assertEqual(len(m), 2)
        self.assertEqual(st["bid_only_dropped"], 1)
        self.assertTrue(np.allclose(m["spread_close"], 0.3))

    def test_crossed_quotes_dropped(self):
        bid = _ohlc([(10, 11, 9, 10), (10, 11, 9, 10)])
        ask = bid - 0.5
        ask["volume"] = 1.0
        m, st = merge_bid_ask(bid, ask)
        self.assertEqual(len(m), 0)
        self.assertGreater(st["crossed_quotes_dropped"], 0)


class ResamplerTests(unittest.TestCase):
    def setUp(self):
        self.m1 = small_m1()

    def test_ohlc_aggregation_exact(self):
        h1 = resample_bid_ask(self.m1, "H1")
        t = h1.index[5]
        chunk = self.m1[(self.m1.index >= t) & (self.m1.index < t + pd.Timedelta(hours=1))]
        self.assertEqual(h1.loc[t, "bid_open"], chunk["bid_open"].iloc[0])
        self.assertEqual(h1.loc[t, "bid_close"], chunk["bid_close"].iloc[-1])
        self.assertEqual(h1.loc[t, "ask_high"], chunk["ask_high"].max())
        self.assertAlmostEqual(h1.loc[t, "spread_mean"], chunk["spread_close"].mean())

    def test_no_invented_bars_and_weekend_excluded(self):
        for tf in ("M5", "H4", "D1", "W1"):
            r = resample_bid_ask(self.m1, tf)
            self.assertTrue((r["n_m1"] > 0).all(), tf)
        d1 = resample_bid_ask(self.m1, "D1")
        self.assertFalse((d1.index.weekday == 5).any())

    def test_weekly_bins_start_sunday(self):
        w = resample_bid_ask(self.m1, "W1")
        self.assertTrue((w.index.weekday == 6).all())

    def test_monotonic_and_utc(self):
        for tf in ("M15", "H4", "W1"):
            r = resample_bid_ask(self.m1, tf)
            self.assertTrue(r.index.is_monotonic_increasing)
            self.assertEqual(str(r.index.tz), "UTC")


if __name__ == "__main__":
    unittest.main()
