# GOLD AI — XAUUSD Research

GOLD AI is an experimental research project for historical gold-market (XAUUSD) data and model evaluation. The repository contains an archive of a larger research project, an extraction helper, and a small test area; the complete trading/research source is not currently present as directly runnable root modules.

## Current repository contents

- `GOLD_AI_RESEARCH_CLEAN (1).zip`: source archive.
- `GOLD_AI_RESEARCH/tests/test_yahoo_fallback.py`: a small existing test.
- `extract_archive.py`: safe extraction of selected text files from the archive.
- `README.md`: project documentation.

## Inspect the archive safely

Python 3.10+:

```bash
python extract_archive.py
```

The helper extracts only allow-listed text/configuration file types into `archive_extracted/`, rejects absolute paths, parent-directory traversal and symbolic links, and caps total extracted text size. Review the extracted files before running any project code. It does not extract binary datasets or model artifacts.

## Next steps to make the research reproducible

1. Review the extracted README and dependency manifest.
2. Move only reviewed source files into a clear project structure; retain the original archive.
3. Document market-data provider, instrument, timezone, tick/bar granularity and data license.
4. Define chronological train/validation/test periods and prevent future-data leakage.
5. Include transaction costs, spread, slippage, latency and risk controls in any backtest.
6. Add tests for data ingestion, feature calculations, target alignment and backtest accounting.
7. Report out-of-sample metrics and drawdown; do not report accuracy alone.

## Security and financial risk

Keep API keys in local environment variables or an untracked `.env` file. Never commit broker credentials or private market data. Historical model predictions do not guarantee future performance and are not financial advice. Do not connect an unreviewed model to a live brokerage account.

## Verification

No end-to-end training or backtest command is claimed for the repository root because the full source is not currently checked in outside the archive. Review and test the extracted source before publishing performance claims.

## License

No license is declared. All rights reserved unless the repository owner adds an explicit license.
