"""End-to-end data pipeline: validate Bid/Ask -> merge -> validate -> build timeframes -> save."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..config import Settings
from .merger import merge_bid_ask
from .resampler import ALL_TFS, build_all
from .storage import frame_exists, load_frame, save_frame
from .validator import validate_ohlc

log = logging.getLogger(__name__)


def build_datasets(
    bid: pd.DataFrame,
    ask: pd.DataFrame,
    settings: Settings,
    out_dir: Path,
    label: str = "real",
    raw_audit: dict | None = None,
) -> dict:
    d = settings["data"]
    bid_c, bid_rep = validate_ohlc(bid, "bid", d["max_gap_minutes"], d["max_jump_pct"])
    ask_c, ask_rep = validate_ohlc(ask, "ask", d["max_gap_minutes"], d["max_jump_pct"])
    m1, merge_stats = merge_bid_ask(bid_c, ask_c)
    if m1.empty:
        raise ValueError("No matching valid Bid/Ask candles; refusing to build empty datasets")
    frames = build_all(m1, settings.timeframes, d["day_offset_hours"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage = {}
    for tf, df in frames.items():
        if df.empty:
            raise ValueError(f"No {tf} bars were produced")
        save_frame(df, out_dir / tf)
        coverage[tf] = {
            "rows": len(df),
            "first": str(df.index[0]),
            "last": str(df.index[-1]),
            "missing_intervals": bid_rep.gaps_in_market_hours if tf == "M1" else None,
        }
    summary = {
        "label": label,
        "bid_validation": bid_rep.to_dict(),
        "ask_validation": ask_rep.to_dict(),
        "merge": merge_stats,
        "raw_hour_audit": raw_audit,
        "spread_close_stats": m1["spread_close"].describe().round(4).to_dict(),
        "coverage": coverage,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("datasets written to %s", out_dir)
    return summary


def load_timeframe(settings: Settings, tf: str, synthetic: bool = False) -> pd.DataFrame:
    folder = settings.path("synthetic_dir" if synthetic else "processed_dir")
    return load_frame(folder / tf)


def datasets_available(settings: Settings, synthetic: bool = False) -> bool:
    folder = settings.path("synthetic_dir" if synthetic else "processed_dir")
    return all(frame_exists(folder / tf) for tf in ALL_TFS)
