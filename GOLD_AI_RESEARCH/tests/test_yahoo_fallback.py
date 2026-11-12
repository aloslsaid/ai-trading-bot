from argparse import Namespace
from unittest.mock import patch

import pandas as pd

from gold_ai.config import load_settings
from main import cmd_download_yahoo


def test_yahoo_approximation_never_overwrites_real_processed_data():
    settings = load_settings()
    bid = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    ask = bid.copy()
    meta = {"provider": "Yahoo Finance", "hourly_bars": 1, "m1_bars": 60}
    args = Namespace(
        start="2026-01-01",
        end="2026-01-02",
        half_spread=0.35,
        symbol="GC=F",
    )

    with patch("main.download_and_build_m1", return_value=(bid, ask, meta)):
        with patch("main.build_datasets", return_value={"coverage": {}}) as build:
            cmd_download_yahoo(args, settings)

    output_dir = build.call_args.args[3]
    assert output_dir == settings.path("yahoo_approx_dir")
    assert output_dir != settings.path("processed_dir")
    assert build.call_args.kwargs["label"] == "yahoo_approx"
