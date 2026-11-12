import os
import unittest
from unittest.mock import patch

import requests

from gold_ai.live_data import fetch_live_candles
from gold_ai.notifications import telegram_send


class CredentialRedactionTests(unittest.TestCase):
    @patch.dict(os.environ, {"TWELVEDATA_API_KEY": "secret-twelve-key"})
    @patch("gold_ai.live_data.requests.get", side_effect=requests.Timeout(
        "request failed for https://api.twelvedata.com/time_series?apikey=secret-twelve-key"
    ))
    def test_twelve_data_request_error_does_not_expose_api_key(self, _request):
        with self.assertRaises(RuntimeError) as caught:
            fetch_live_candles(outputsize=1)
        self.assertNotIn("secret-twelve-key", str(caught.exception))

    @patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "secret-telegram-token",
        "TELEGRAM_CHAT_ID": "12345",
    })
    @patch("gold_ai.notifications.requests.post", side_effect=requests.ConnectionError(
        "https://api.telegram.org/botsecret-telegram-token/sendMessage"
    ))
    def test_telegram_request_error_does_not_expose_bot_token(self, _request):
        with self.assertRaises(RuntimeError) as caught:
            telegram_send("test")
        self.assertNotIn("secret-telegram-token", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
