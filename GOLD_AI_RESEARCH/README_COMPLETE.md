# GOLD AI Research

Research and historical simulation only. This project has no broker connection and never places real trades. You execute any trades manually. Data gaps remain gaps: the pipeline never fabricates market candles.

## Windows PowerShell setup

Use Python 3.11 or newer and ensure `curl.exe` is on `PATH` (it is included with current Windows versions).

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project already declares its runtime dependencies in `requirements.txt` (pandas, NumPy, PyArrow, scikit-learn, joblib, and supporting packages). Dukascopy downloads use the Windows `curl.exe`; no separate downloader package or API key is required.

## Historical XAUUSD download and data quality

The end date is exclusive. This command requests all hours from January 1 through November 11, 2026; the downloader caps the request at the latest completed UTC hour, so it does not request future or incomplete hours.

```powershell
python main.py download --start 2026-01-01 --end 2026-11-12
python main.py coverage --start 2026-01-01 --end 2026-11-12
```

Downloads use `curl.exe`, retry transient failures including HTTP 503 with exponential backoff, and use one worker by default. Override with `--workers N` only if desired. Valid existing hourly files are validated and reused; only missing or corrupt hours are fetched again. Raw `.bi5` files are retained in `data\raw\dukascopy\XAUUSD\`. HTTP 404/no-tick hours are stored as zero-byte files and reported separately as empty, not converted into candles. A failed transfer is not cached, so re-running the same command resumes the download.

`coverage` audits every requested raw hour (valid, empty/no-tick, missing, failed, and corrupt), and reports processed timeframes, Bid/Ask validation, gaps, and suspicious jumps when a build summary exists. A complete per-hour CSV plus retry/HTTP details is saved under `data\raw\dukascopy\XAUUSD\coverage_*.csv`. Missing/corrupt raw hours and missing M1 candles are not filled. Failed hours are retried on the next identical download while successful and known-empty hours are not fetched again. `Retry-After` is honored for HTTP 429/503 responses, with bounded exponential backoff otherwise. The build uses available real ticks and labels incomplete coverage; verify the audit before training or interpreting results. Dukascopy's source availability and rate limits may differ by symbol and hour, so a full download cannot be guaranteed when its endpoint is unavailable or rate-limiting requests.

The decoder validates BI5 compression, record sizes, tick timestamps, quotes, and volumes. It aggregates actual ticks independently into Bid and Ask M1 OHLC candles, keeps only minutes with ticks, and builds M5/M15/M30/H1/H4/D1/W1 from those quotes. UTC is used throughout. No synthetic data is mixed into `data\processed\`.

## Build, train, evaluate, and backtest

`download` also builds datasets from available actual ticks, including when some hours remain unavailable; the raw-hour and candle-gap reports retain that incompleteness. Use `build` after correcting/refilling cached raw files or when you want to rebuild without downloading:

```powershell
python main.py build --start 2026-01-01 --end 2026-11-12
python main.py coverage --start 2026-01-01 --end 2026-11-12
python main.py train --tf M15
python main.py evaluate --tf M15
python main.py backtest --tf M15 --strategy ml --model models/xau_m15.joblib
```

The model trainer rejects datasets not explicitly labeled `real`, requires at least 500 valid labeled rows, and omits labels whose forward horizon crosses a missing bar interval. It engineers causal bar/indicator/return/time features, compares logistic regression, random forest, and histogram gradient boosting on a chronological validation period, refits the best validation model on train plus validation, then evaluates once on a held-out test period. The last prediction-horizon bars before each split are embargoed so their forward labels cannot cross into the next period. The fitted pipeline is saved to `models\xau_m15.joblib`; comparison/test metrics are written beside it. `evaluate` saves a separate `.evaluation.json` report using the frozen model. Classification metrics are not trading returns.

Backtests use the chronological held-out test period. Bid/Ask-side entries and exits, configured commission per lot per side, slippage, stop/target execution, leverage and position sizing limits, and a daily loss block are applied. Stops are assumed to hit first if both stop and target are crossed within a bar. The ML backtest requires a trained model; `--strategy baselines` (the default) runs plumbing/reference strategies instead:

```powershell
python main.py backtest --tf M15
```

Reports are written under `reports_out\`. Execution parameters are in `config\default.yaml`; verify they match your broker before interpreting a simulation. The defaults are examples, not claims about your broker's actual spread, commission, slippage, or fill behavior.

## Separate live-provider check

```powershell
python main.py live-test --symbol "XAU/USD" --interval 15min --outputsize 10
```

This independently tests Twelve Data using `TWELVEDATA_API_KEY` from `.env`. The report explicitly marks the returned provider OHLC as **not broker Bid/Ask**; it must not be used to claim broker spread, fills, or executable prices. This command makes no trades and does not require a trained model.

## Synthetic demo and tests

The demo is synthetic and isolated under `data\synthetic\`; it is clearly labeled, not accepted for ML training, and must never be treated as market history or used to infer trading performance.

```powershell
python main.py demo --days 180 --tf M15
python -m pytest -q
```

## Other commands

```powershell
python main.py --help
python main.py download --help
python main.py coverage --start 2026-01-01 --end 2026-11-12
python main.py import-csv --bid path\to\bid.csv --ask path\to\ask.csv
python main.py train --tf M15 --horizon 3 --threshold 0.0005
python main.py evaluate --tf M15
```

The default YAML config can be overridden with the global option `--config path\to\override.yaml`, placed before the command. Do not commit credentials, private data, or downloaded tick history.


## Yahoo research approximation (separate from real Bid/Ask data)

If Dukascopy is unavailable, the optional Yahoo command can retrieve GC=F hourly futures bars and expand each hour into flat M1 bars with a configurable fixed half-spread:

```powershell
python main.py download-yahoo --start 2026-01-01 --end 2026-11-12 --half-spread 0.35
```

This is **not** true XAUUSD spot tick data, not real M1 price action, and not a broker Bid/Ask feed. The generated path is explicitly approximate and is stored under `data\yahoo_approx\`, never in `data\processed\`. It is labeled `yahoo_approx`; the normal training and backtest commands accept only datasets labeled `real`, so this fallback cannot overwrite or masquerade as real market history.
