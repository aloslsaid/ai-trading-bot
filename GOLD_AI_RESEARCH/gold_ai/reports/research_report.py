"""CSV + self-contained HTML research report for Milestone 1 backtests."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

KEYS = ["trades", "net_profit", "return_pct", "win_rate_pct", "profit_factor", "expectancy_usd", "expectancy_r", "avg_win", "avg_loss",
        "max_drawdown_usd", "max_drawdown_pct", "sharpe", "sortino", "avg_bars"]


def _fmt(v) -> str:
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:,.2f}"
    return str(v)


def _equity_svg(curves: dict[str, pd.Series], w: int = 900, h: int = 260) -> str:
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    series = {k: s.dropna().resample("1D").last().dropna() for k, s in curves.items() if len(s.dropna())}
    if not series:
        return ""
    lo = min(s.min() for s in series.values())
    hi = max(s.max() for s in series.values())
    t0 = min(s.index[0] for s in series.values())
    t1 = max(s.index[-1] for s in series.values())
    span_t, span_v = max((t1 - t0).total_seconds(), 1), max(hi - lo, 1e-9)
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" style="background:#fafafa;border:1px solid #ddd">']
    for i, (name, s) in enumerate(series.items()):
        pts = " ".join(f"{(t - t0).total_seconds() / span_t * (w - 20) + 10:.1f},{h - 20 - (v - lo) / span_v * (h - 40):.1f}" for t, v in s.items())
        parts.append(f'<polyline fill="none" stroke="{colors[i % 5]}" stroke-width="1.6" points="{pts}"/>')
        parts.append(f'<text x="14" y="{16 + 14 * i}" fill="{colors[i % 5]}" font-size="12">{html.escape(name)}</text>')
    parts.append(f'<text x="{w - 10}" y="{h - 4}" text-anchor="end" font-size="10" fill="#555">equity {lo:,.0f} to {hi:,.0f}</text></svg>')
    return "".join(parts)


def write_report(out_dir: Path, title: str, label: str, tf: str, results: dict, baselines: dict, build_summary: dict | None, cfg: dict, synthetic: bool) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"name": n, "kind": "strategy", **{k: r.metrics.get(k) for k in KEYS}} for n, r in results.items()]
    rows += [{"name": n, "kind": "baseline", **{k: m.get(k) for k in KEYS}} for n, m in baselines.items()]
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    for n, r in results.items():
        if not r.trades.empty:
            r.trades.to_csv(out_dir / f"trades_{n}.csv", index=False)
    banner = (f'<div style="background:#fef3c7;border:2px solid #d97706;padding:10px;font-weight:bold">{html.escape(label)}</div>' if synthetic
              else '<div style="background:#e0f2fe;border:1px solid #0284c7;padding:10px">HISTORICAL SIMULATION (backtest). Not a paper signal, not a live signal, not a forecast.</div>')
    th = "".join(f"<th>{k}</th>" for k in ["name", "kind", *KEYS])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(row[k]))}</td>" for k in ["name", "kind", *KEYS]) + "</tr>" for row in rows)
    curves = {n: r.equity for n, r in results.items()}
    dq = ""
    if build_summary:
        dq = f"<h2>Data quality</h2><pre>{html.escape(json.dumps({k: build_summary[k] for k in ('raw_hour_audit', 'bid_validation', 'ask_validation', 'merge', 'spread_close_stats')}, indent=2))}</pre>"
        cov = pd.DataFrame(build_summary["coverage"]).T.to_html()
        dq += f"<h3>Coverage by timeframe</h3>{cov}"
    skipped = "".join(f"<li>{html.escape(n)}: {html.escape(str(r.skipped))}</li>" for n, r in results.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1100px}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}th{{background:#f3f4f6}}td:first-child,th:first-child{{text-align:left}}pre{{background:#f5f5f5;padding:10px;overflow:auto;font-size:11px}}</style></head><body>
<h1>{html.escape(title)}</h1>{banner}
<p>Timeframe: <b>{tf}</b> &middot; Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<h2>Results</h2><div style="overflow-x:auto"><table><tr>{th}</tr>{body}</table></div>
<h2>Equity curves (daily)</h2>{_equity_svg(curves)}
<h2>Execution assumptions</h2><pre>{html.escape(json.dumps(cfg, indent=2, default=str))}</pre>
<p>Fills use real Bid/Ask: longs enter at ask and exit at bid, shorts the reverse. Signals use data up to bar close and fill at the next bar open. If SL and TP are both touched in one bar, the stop is assumed first.</p>
<h2>Skipped signals</h2><ul>{skipped}</ul>{dq}
<h2>Caveats</h2><ul><li>Strategies are reported on the chronological held-out test period only; results remain historical simulations, not forecasts.</li><li>Volume is tick volume, not traded volume.</li><li>Past simulated performance does not imply future results.</li></ul>
</body></html>"""
    p = out_dir / "report.html"
    p.write_text(doc, encoding="utf-8")
    return p
