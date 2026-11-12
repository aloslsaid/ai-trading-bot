"""GOLD_AI_RESEARCH command line entry point (Milestone 1).

Research and simulation only. This project never places, modifies or closes trades.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gold_ai.backtesting.baselines import buy_and_hold, random_entries, sma_cross_atr  # noqa: E402
from gold_ai.backtesting.engine import BacktestConfig, run_backtest  # noqa: E402
from gold_ai.backtesting.splits import time_split  # noqa: E402
from gold_ai.config import load_settings  # noqa: E402
from gold_ai.data_engine import dukascopy  # noqa: E402
from gold_ai.data_engine.importer import read_ohlc_csv  # noqa: E402
from gold_ai.data_engine.pipeline import build_datasets, load_timeframe  # noqa: E402
from gold_ai.data_engine.synthetic import SYNTHETIC_LABEL, generate_synthetic_m1  # noqa: E402
from gold_ai.data_engine.yahoo import download_and_build_m1  # noqa: E402
from gold_ai.live_data import fetch_live_candles  # noqa: E402
from gold_ai.ml_engine import evaluate_model, model_signals, train_model  # noqa: E402
from gold_ai.reports.research_report import write_report  # noqa: E402

log = logging.getLogger("gold_ai")


def _range(args, s):
    def parse(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    start = parse(args.start) if args.start else parse(s["data"]["start_date"])
    end = parse(args.end) if args.end else datetime.now(timezone.utc)
    return start, end


def cmd_download(args, s):
    start, end = _range(args, s)
    raw_dir, pv = s.path("raw_dir"), s["data"]["dukascopy_point_value"]
    log.info("Downloading %s ticks %s -> %s UTC (cached valid hours are skipped)", s.symbol, start.date(), end.date())
    workers = args.workers if args.workers is not None else s["data"]["download_workers"]
    stats = dukascopy.download_range(s.symbol, start, end, raw_dir, workers)
    if stats["failed"]:
        log.warning("%d hours failed; re-run the same command to retry only those", stats["failed"])
    audit = dukascopy.audit_range(s.symbol, start, end, raw_dir)
    print("Raw BI5 coverage:", json.dumps(audit, indent=2))
    if audit["missing"] or audit["failed"] or audit["corrupt"]:
        log.warning(
            "Building from available real ticks only; coverage remains incomplete "
            "(missing=%d failed=%d corrupt=%d)",
            audit["missing"], audit["failed"], audit["corrupt"],
        )
    bid, ask = dukascopy.build_m1_from_raw(s.symbol, start, end, raw_dir, pv)
    build_datasets(bid, ask, s, s.path("processed_dir"), label="real", raw_audit=audit)
    log.info("Build complete. Next: python main.py coverage")


def cmd_import(args, s):
    bid, ask = read_ohlc_csv(args.bid), read_ohlc_csv(args.ask)
    summ = build_datasets(bid, ask, s, s.path("processed_dir"), label="real")
    print(json.dumps(summ["coverage"], indent=2))


def cmd_build(args, s):
    start, end = _range(args, s)
    audit = dukascopy.audit_range(s.symbol, start, end, s.path("raw_dir"))
    print("Raw BI5 coverage:", json.dumps(audit, indent=2))
    if audit["missing"] or audit["failed"] or audit["corrupt"]:
        log.warning(
            "Building from available real ticks only; coverage remains incomplete "
            "(missing=%d failed=%d corrupt=%d)",
            audit["missing"], audit["failed"], audit["corrupt"],
        )
    bid, ask = dukascopy.build_m1_from_raw(s.symbol, start, end, s.path("raw_dir"), s["data"]["dukascopy_point_value"])
    build_datasets(bid, ask, s, s.path("processed_dir"), label="real", raw_audit=audit)


def cmd_download_yahoo(args, s):
    """Fallback historical loader when Dukascopy is unreachable.

    Uses Yahoo Finance GC=F hourly bars (highly correlated to spot XAUUSD),
    expands to M1 with a fixed research spread, and writes processed datasets
    labeled 'real' with clear provenance in the build summary.
    """
    start, end = _range(args, s)
    half = args.half_spread if args.half_spread is not None else 0.35
    log.info(
        "Yahoo Finance fallback: %s %s -> %s (half_spread=%.3f)",
        args.symbol or "GC=F", start.date(), end.date(), half,
    )
    bid, ask, meta = download_and_build_m1(
        start, end, symbol=args.symbol or "GC=F", half_spread=half,
    )
    audit = {
        "source": "yahoo_finance",
        "meta": meta,
        "hours_expected": None,
        "valid": meta.get("hourly_bars"),
        "empty": 0,
        "missing": 0,
        "failed": 0,
        "corrupt": 0,
        "note": "Dukascopy unavailable; using Yahoo H1 expanded to M1 with synthetic spread.",
    }
    summary = build_datasets(bid, ask, s, s.path("yahoo_approx_dir"), label="yahoo_approx", raw_audit=audit)
    print("Yahoo build complete:", json.dumps({
        "m1_rows": len(bid),
        "coverage": summary["coverage"],
        "provider_meta": meta,
    }, indent=2, default=str))
    log.info("Yahoo approximation stored separately; it is not eligible for real-data training or backtesting.")


def _backtest(s, tf: str, synthetic: bool, out: Path, strategy: str = "baselines", model_path: str | None = None):
    df = load_timeframe(s, tf, synthetic)
    source_dir = s.path("synthetic_dir" if synthetic else "processed_dir")
    summary_file = source_dir / "build_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"No dataset summary at {summary_file}; run download/build first")
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if synthetic and summary.get("label") != SYNTHETIC_LABEL:
        raise ValueError("Synthetic backtest requires explicitly labeled synthetic data")
    if not synthetic and summary.get("label") != "real":
        raise ValueError("Refusing to evaluate non-real data as real market history")
    cfg = BacktestConfig.from_settings(s)
    split = time_split(df, s["backtest"]["train_frac"], s["backtest"]["val_frac"])
    test = split["test"]
    log.info("rows=%d train=%d val=%d test=%d (backtest uses held-out test period only)", len(df), *[len(v) for v in split.values()])
    if strategy == "ml":
        if synthetic:
            raise ValueError("ML model backtests require real market data")
        signals = model_signals(df, model_path or s["ml"]["model_path"])
        test_signals = type(signals)(
            signals.direction.loc[test.index], signals.sl_dist.loc[test.index], signals.tp_dist.loc[test.index]
        )
        results = {"ml_model": run_backtest(test, test_signals, cfg, "ml_model")}
    else:
        sma_signals = sma_cross_atr(df)
        random_signals = random_entries(df, prob=0.01, seed=1)
        test_sma = type(sma_signals)(
            sma_signals.direction.loc[test.index], sma_signals.sl_dist.loc[test.index], sma_signals.tp_dist.loc[test.index]
        )
        test_random = type(random_signals)(
            random_signals.direction.loc[test.index], random_signals.sl_dist.loc[test.index], random_signals.tp_dist.loc[test.index]
        )
        results = {
            "sma_cross_atr": run_backtest(test, test_sma, cfg, "sma_cross_atr"),
            "random_entries": run_backtest(test, test_random, cfg, "random_entries"),
        }
    base = {"buy_and_hold": buy_and_hold(test, cfg)}
    title = f"{s.symbol} out-of-sample research report"
    p = write_report(out, title, SYNTHETIC_LABEL, tf, results, base, summary, asdict(cfg), synthetic)
    for n, r in results.items():
        m = r.metrics
        print(f"{n:16s} trades={m['trades']:4d} net={m['net_profit']:9.2f} PF={m['profit_factor']:.2f} maxDD={m['max_drawdown_pct']:.1f}%")
    print(f"buy_and_hold     net={base['buy_and_hold']['net_profit']:9.2f}")
    print("Report:", p)


def cmd_backtest(args, s):
    out = ROOT / "reports_out" / f"{'synthetic' if args.synthetic else 'real'}_{args.tf}"
    _backtest(s, args.tf, args.synthetic, out, args.strategy, args.model)


def cmd_demo(args, s):
    print(f"*** {SYNTHETIC_LABEL} ***")
    bid, ask = generate_synthetic_m1(days=args.days)
    build_datasets(bid, ask, s, s.path("synthetic_dir"), label=SYNTHETIC_LABEL)
    _backtest(s, args.tf, True, ROOT / "reports_out" / f"synthetic_{args.tf}")


def _require_real_dataset(s, tf):
    summary_path = s.path("processed_dir") / "build_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No real-data build summary at {summary_path}; run download/build first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("label") != "real":
        raise ValueError("Training/evaluation requires a dataset explicitly labeled 'real'")
    return load_timeframe(s, tf)


def cmd_train(args, s):
    df = _require_real_dataset(s, args.tf)
    metrics = train_model(
        df,
        args.model or s["ml"]["model_path"],
        args.horizon if args.horizon is not None else s["ml"]["horizon_bars"],
        args.threshold if args.threshold is not None else s["ml"]["neutral_threshold"],
        s["backtest"]["train_frac"],
        s["backtest"]["val_frac"],
        data_label="real",
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


def cmd_evaluate(args, s):
    df = _require_real_dataset(s, args.tf)
    metrics = evaluate_model(df, args.model or s["ml"]["model_path"])
    out = Path(args.model or s["ml"]["model_path"]).with_suffix(".evaluation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print("Evaluation report:", out)


def cmd_live_test(args, s):
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    frame = fetch_live_candles(symbol=args.symbol, interval=args.interval, outputsize=args.outputsize)
    latest = frame.iloc[-1]
    print(json.dumps({
        "status": "connected",
        "provider": frame.attrs["provider"],
        "symbol": frame.attrs["symbol"],
        "interval": frame.attrs["interval"],
        "price_basis": frame.attrs["price_basis"],
        "broker_bid_ask": frame.attrs["broker_bid_ask"],
        "candles_received": len(frame),
        "last_candle_time_utc": frame.index[-1].isoformat(),
        "last_provider_ohlc": {
            key: float(latest[key]) for key in ("open", "high", "low", "close")
        },
        "warning": "Provider OHLC is not your broker's executable Bid/Ask quote.",
    }, indent=2))


def cmd_coverage(args, s):
    start, end = _range(args, s)
    print("Raw BI5 audit:", json.dumps(
        dukascopy.audit_range(s.symbol, start, end, s.path("raw_dir")), indent=2
    ))
    folder = s.path("synthetic_dir" if args.synthetic else "processed_dir")
    summary_path = folder / "build_summary.json"
    if not summary_path.exists():
        print(f"No processed build summary at {summary_path}; run download/build first.")
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Processed label: {summary['label']}")
    if summary.get("raw_hour_audit"):
        print("Build raw-hour audit:", json.dumps(summary["raw_hour_audit"], indent=2))
    for tf, coverage in summary["coverage"].items():
        print(f"{tf:4s} rows={coverage['rows']:8d}  {coverage['first']} -> {coverage['last']}")
    print("Data validation:", json.dumps({
        "bid": summary["bid_validation"],
        "ask": summary["ask_validation"],
        "merge": summary["merge"],
    }, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="GOLD_AI_RESEARCH (research only, never executes trades)")
    p.add_argument("--config", help="optional YAML override file")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("download", cmd_download), ("build", cmd_build)):
        sp = sub.add_parser(name)
        sp.add_argument("--start", help="YYYY-MM-DD (UTC); default = configured start date")
        sp.add_argument("--end", help="exclusive UTC date/hour; default = now")
        if name == "download":
            sp.add_argument("--workers", type=int, help="parallel downloads (default 1)")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser(
        "download-yahoo",
        help="Fallback: download 2026 gold H1 from Yahoo Finance (GC=F) and build datasets when Dukascopy is down",
    )
    sp.add_argument("--start", help="YYYY-MM-DD (UTC); default = configured start date")
    sp.add_argument("--end", help="exclusive UTC date/hour; default = now")
    sp.add_argument("--symbol", default="GC=F", help="Yahoo ticker (default GC=F gold futures)")
    sp.add_argument("--half-spread", type=float, dest="half_spread", default=0.35,
                    help="USD half-spread applied to mid price for bid/ask approximation")
    sp.set_defaults(fn=cmd_download_yahoo)
    sp = sub.add_parser("import-csv")
    sp.add_argument("--bid", required=True)
    sp.add_argument("--ask", required=True)
    sp.set_defaults(fn=cmd_import)
    sp = sub.add_parser("backtest")
    sp.add_argument("--tf", default="M15")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--strategy", choices=("baselines", "ml"), default="baselines")
    sp.add_argument("--model")
    sp.set_defaults(fn=cmd_backtest)
    sp = sub.add_parser("demo", help="end-to-end run on SYNTHETIC data")
    sp.add_argument("--tf", default="M15")
    sp.add_argument("--days", type=int, default=180)
    sp.set_defaults(fn=cmd_demo)
    sp = sub.add_parser("coverage")
    sp.add_argument("--synthetic", action="store_true")
    sp.add_argument("--start", help="first UTC date/hour to inspect")
    sp.add_argument("--end", help="exclusive UTC date/hour to inspect")
    sp.set_defaults(fn=cmd_coverage)
    sp = sub.add_parser("train")
    sp.add_argument("--tf", default="M15")
    sp.add_argument("--horizon", type=int)
    sp.add_argument("--threshold", type=float)
    sp.add_argument("--model")
    sp.set_defaults(fn=cmd_train)
    sp = sub.add_parser("evaluate")
    sp.add_argument("--tf", default="M15")
    sp.add_argument("--model")
    sp.set_defaults(fn=cmd_evaluate)
    sp = sub.add_parser("live-test", help="test Twelve Data independently; no trades or model required")
    sp.add_argument("--symbol")
    sp.add_argument("--interval", default="15min")
    sp.add_argument("--outputsize", type=int, default=10)
    sp.set_defaults(fn=cmd_live_test)
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.fn(args, load_settings(args.config))


if __name__ == "__main__":
    main()
