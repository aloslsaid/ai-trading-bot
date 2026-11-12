import unittest
from unittest.mock import patch

from gold_ai.indicators.classic import compute_classic
from gold_ai.live_data import fetch_live_candles


class LiveDataTests(unittest.TestCase):
    @patch("gold_ai.live_data.requests.get")
    def test_provider_ohlc_is_not_mislabeled_as_broker_quotes(self, get):
        response = get.return_value
        response.status_code = 200
        response.json.return_value = {
            "values": [{
                "datetime": "2026-11-10 23:00:00",
                "open": "4000", "high": "4010", "low": "3990", "close": "4005",
                "volume": "12",
            }]
        }
        with patch.dict("os.environ", {"TWELVEDATA_API_KEY": "test-key"}, clear=False):
            frame = fetch_live_candles(outputsize=1)
        self.assertEqual(frame.attrs["provider"], "Twelve Data")
        self.assertFalse(frame.attrs["broker_bid_ask"])
        self.assertNotIn("bid_close", frame.columns)
        self.assertEqual(frame.iloc[-1]["close"], 4005)
        features = compute_classic(frame)
        self.assertEqual(len(features), 1)

    def test_missing_api_key_fails_clearly(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TWELVEDATA_API_KEY"):
                fetch_live_candles()


if __name__ == "__main__":
    unittest.main()
