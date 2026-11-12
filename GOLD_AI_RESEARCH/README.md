# GOLD AI Research

Historical XAUUSD tick-data research, chronological ML evaluation, and cost-aware backtesting. It does not execute trades. Start with [README_COMPLETE.md](README_COMPLETE.md) for PowerShell setup and exact end-to-end commands.

Quick start:

```powershell
python main.py download --start 2026-01-01 --end 2026-11-12
python main.py coverage --start 2026-01-01 --end 2026-11-12
python main.py train --tf M15
python main.py evaluate --tf M15
python main.py backtest --tf M15 --strategy ml --model models/xau_m15.joblib
python main.py live-test --symbol "XAU/USD" --interval 15min --outputsize 10
python -m pytest -q
```

End dates are exclusive; downloads stop at the current completed UTC hour. Missing market data is reported, never synthesized. The live test uses Twelve Data provider OHLC, not broker Bid/Ask.
