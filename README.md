# GOLD AI — Machine Learning Gold-Market Research

> An experimental gold-market analysis system combining historical price learning, technical-indicator features, Twelve Data live prices, and Telegram-delivered model analysis.

## Project vision
GOLD AI explores whether machine-learning models trained on historical gold-market behavior can generate useful short-horizon analytical signals for XAUUSD. The workflow connects historical data preparation and model training with live market-data retrieval and Telegram reporting.

## Research pipeline
- **Historical training window:** approximately one year of gold-market data, as specified for this experiment.
- **Data cleaning:** prepare and validate historical price records before training.
- **Feature engineering:** use approximately 15 technical/market indicators to represent recent market behavior.
- **Machine learning:** train a model to learn patterns from historical gold data.
- **Forecast horizon:** analyze the next three candles.
- **Live data:** retrieve current market data through the Twelve Data API.
- **AI analysis:** interpret model outputs and current market context.
- **Telegram delivery:** send live analysis and model results to a Telegram chat.

## Reported test observations
The project owner reports **25 successful outcomes and 5 unsuccessful outcomes across 30 experimental trades**. This is a small, owner-reported sample—not proof of profitability, predictive edge, or future reliability. It does not by itself account for transaction costs, spreads, slippage, drawdown, position sizing, market regime changes, or selection bias. A larger, timestamped, out-of-sample evaluation is necessary before making performance claims.

## Repository packaging
The repository currently contains:
- `GOLD_AI_RESEARCH_CLEAN (1).zip` — bundled project archive.
- `extract_archive.py` — controlled extraction helper.
- `README.md` — project guide.

Extract the supported source/text files:
```bash
python extract_archive.py
```
The helper writes selected extensions into `archive_extracted/`, rejects unsafe paths and symbolic links, and enforces a size limit. It is intentionally not a full general-purpose ZIP extractor. After extraction, read the bundled project documentation and dependency manifest before running any component.

The repository root does not expose a verified training or live-bot launch command, so no unverified command is presented here.

## Evaluation checklist
A robust trading evaluation should include chronological splits, leakage checks, a baseline strategy, unseen market periods, win rate, expectancy, profit factor, maximum drawdown, and transaction costs. Paper-trade before considering real capital. Model outputs should be logged with timestamps so predictions can be matched to outcomes without hindsight.

## Security
Keep Twelve Data, Telegram, OpenRouter, and all other credentials in local environment variables or an untracked `.env`. Revoke and replace any key that has appeared in a public commit or archive.

## Risk notice
GOLD AI is an experimental research tool, not financial advice and not a guarantee of profitable trades. Market predictions can be wrong; past performance does not predict future results. Do not enable live execution without independent validation and strict risk controls.

## Skills demonstrated
Financial time-series analysis · Data cleaning · Technical indicators · Machine learning · Live market-data APIs · Telegram automation · Model evaluation · Experimental research
