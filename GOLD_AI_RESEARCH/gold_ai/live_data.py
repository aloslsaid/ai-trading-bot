"""Twelve Data provider candles; these are not broker Bid/Ask execution quotes."""
from __future__ import annotations

import os

import pandas as pd
import requests


LIVE_PROVIDER = "Twelve Data"


def fetch_live_candles(symbol: str | None = None, interval: str | None = None, outputsize: int = 500) -> pd.DataFrame:
    key = os.getenv("TWELVEDATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Set TWELVEDATA_API_KEY in .env (Twelve Data).")
    if outputsize < 1:
        raise ValueError("outputsize must be positive")
    symbol = symbol or os.getenv("LIVE_SYMBOL", "XAU/USD")
    interval = interval or os.getenv("LIVE_INTERVAL", "1min")
    try:
        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "apikey": key,
                "timezone": "UTC",
                "format": "JSON",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        # Requests exceptions can include the full URL, including the API key.
        raise RuntimeError(
            f"Twelve Data request failed ({type(exc).__name__}); check network and credentials."
        ) from None
    if response.status_code >= 400:
        raise RuntimeError(
            f"Twelve Data returned HTTP {response.status_code}; response details were redacted."
        )
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("Twelve Data returned an invalid JSON response.") from None
    if "values" not in payload:
        raise RuntimeError(f"Live feed error: {payload.get('message', payload)}")
    data = pd.DataFrame(payload["values"])
    if data.empty or "datetime" not in data:
        raise RuntimeError("Live provider returned no timestamped candles")
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True, errors="coerce")
    data = data.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    if "volume" not in data:
        data["volume"] = 0.0
    for column in ("open", "high", "low", "close", "volume"):
        if column not in data:
            if column == "volume":
                data[column] = 0.0
            else:
                raise RuntimeError(f"Live provider response is missing '{column}'")
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[["open", "high", "low", "close", "volume"]].dropna(
        subset=["open", "high", "low", "close"]
    )
    if data.empty:
        raise RuntimeError("Live provider returned no valid OHLC candles")
    data.attrs.update({
        "provider": LIVE_PROVIDER,
        "symbol": symbol,
        "interval": interval,
        "price_basis": "provider OHLC; not broker Bid/Ask",
        "broker_bid_ask": False,
    })
    return data
