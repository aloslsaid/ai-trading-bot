"""Dukascopy BI5 hourly ticks -> Bid/Ask M1 candles with resumable downloads."""
from __future__ import annotations

import logging
import lzma
import email.utils
import random
import struct
import subprocess
import tempfile
import time
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
_STATUS_LOCK = threading.Lock()
BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RECORD = struct.Struct(">IIIff")
_DTYPE = np.dtype([
    ("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
    ("ask_vol", ">f4"), ("bid_vol", ">f4"),
])


def hour_url(symbol: str, hour: datetime) -> str:
    return f"{BASE_URL}/{symbol}/{hour.year:04d}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"


def hour_path(raw_dir: Path, symbol: str, hour: datetime) -> Path:
    return raw_dir / "dukascopy" / symbol / f"{hour:%Y}" / f"{hour:%m}" / f"{hour:%d}" / f"{hour:%H}h_ticks.bi5"


def decode_bi5(payload: bytes, hour: datetime, point_value: float = 1000.0) -> pd.DataFrame:
    """Decode and validate one hourly BI5 payload into UTC Bid/Ask ticks."""
    if point_value <= 0:
        raise ValueError("point_value must be positive")
    if not payload:
        return pd.DataFrame(columns=["bid", "ask", "bid_vol", "ask_vol"], index=pd.DatetimeIndex([], tz="UTC"))
    raw = lzma.decompress(payload)
    if len(raw) % RECORD.size:
        raise ValueError(f"corrupt BI5 payload: {len(raw)} bytes is not a multiple of {RECORD.size}")
    arr = np.frombuffer(raw, dtype=_DTYPE)
    if (arr["ms"] >= 3_600_000).any():
        raise ValueError("corrupt BI5 payload: tick timestamp falls outside its hour")
    if (arr["bid"] == 0).any() or (arr["ask"] == 0).any():
        raise ValueError("corrupt BI5 payload: non-positive quote")
    if (arr["bid_vol"] < 0).any() or (arr["ask_vol"] < 0).any():
        raise ValueError("corrupt BI5 payload: negative tick volume")
    if not np.isfinite(arr["bid_vol"]).all() or not np.isfinite(arr["ask_vol"]).all():
        raise ValueError("corrupt BI5 payload: non-finite tick volume")
    base = pd.Timestamp(hour)
    base = base.tz_convert("UTC") if base.tzinfo else base.tz_localize("UTC")
    idx = base + pd.to_timedelta(arr["ms"].astype("int64"), unit="ms")
    ticks = pd.DataFrame(
        {
            "bid": arr["bid"].astype("float64") / point_value,
            "ask": arr["ask"].astype("float64") / point_value,
            "bid_vol": arr["bid_vol"].astype("float64"),
            "ask_vol": arr["ask_vol"].astype("float64"),
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )
    return ticks.sort_index(kind="stable")


def ticks_to_m1(ticks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate actual tick streams to Bid and Ask M1 OHLC; never fill empty minutes."""
    if ticks.empty:
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        return empty, empty.copy()
    ticks = ticks.sort_index(kind="stable")
    out = []
    for side in ("bid", "ask"):
        grouped = ticks[side].resample("1min")
        candles = grouped.ohlc()
        candles["volume"] = ticks[f"{side}_vol"].resample("1min").sum()
        candles["tick_count"] = grouped.count()
        candles = candles[candles["tick_count"] > 0].drop(columns="tick_count")
        candles.index.name = "timestamp"
        out.append(candles)
    return out[0], out[1]


def _curl_binary() -> str:
    """Prefer system curl; fall back to curl.exe on Windows only."""
    import platform
    import shutil
    if platform.system() == "Windows":
        return shutil.which("curl.exe") or shutil.which("curl") or "curl.exe"
    return shutil.which("curl") or "curl"


def _fetch(
    url: str,
    retries: int = 5,
    timeout: int = 30,
    expected_hour: datetime | None = None,
    status_info: dict | None = None,
    on_attempt_failure: Callable[[dict], None] | None = None,
) -> bytes | None:
    """Use system curl with Retry-After-aware backoff; b'' means HTTP 404."""
    info = status_info if status_info is not None else {}
    info.update({"attempts": 0, "http_status": None, "error": None})
    curl_bin = _curl_binary()
    for attempt in range(retries):
        temp_path = None
        header_path = None
        retry_after = None
        info.update({"attempts": attempt + 1, "http_status": None, "error": None})
        try:
            with tempfile.NamedTemporaryFile(suffix=".bi5", delete=False) as tmp:
                temp_path = Path(tmp.name)
            with tempfile.NamedTemporaryFile(suffix=".headers", delete=False) as tmp_headers:
                header_path = Path(tmp_headers.name)
            result = subprocess.run(
                [
                    curl_bin, "--location", "--fail", "--silent", "--show-error",
                    "--connect-timeout", str(timeout), "--max-time", str(max(timeout * 4, 120)),
                    "--user-agent", "Mozilla/5.0 GOLD_AI_RESEARCH",
                    "--output", str(temp_path), "--dump-header", str(header_path),
                    "--write-out", "%{http_code}", url,
                ],
                capture_output=True,
                text=True,
                timeout=max(timeout * 4 + 15, 135),
            )
            status = result.stdout.strip()
            info.update({"attempts": attempt + 1, "http_status": status or None})
            retry_after = _retry_after_seconds(header_path)
            info["retry_after_seconds"] = retry_after
            if status == "404":
                info["error"] = "HTTP 404: no BI5 object for this hour"
                return b""
            if result.returncode == 0 and status.startswith("2"):
                payload = temp_path.read_bytes()
                try:
                    decode_bi5(payload, expected_hour or datetime(2000, 1, 1, tzinfo=timezone.utc))
                except (lzma.LZMAError, ValueError, struct.error) as exc:
                    info["error"] = f"invalid_bi5: {exc}"
                    log.warning("invalid BI5 response for %s: %s", url, exc)
                else:
                    if payload:
                        return payload
                    info["error"] = "HTTP 200: empty response body; no ticks"
                    return b""
            retryable = status.startswith("5") or status in ("", "000", "408", "429")
            if not info.get("error"):
                info["error"] = f"HTTP {status or 'unknown'}: {result.stderr[-300:]}"
            log.warning(
                "curl failed: HTTP %s, exit %s, attempt %d/%d: %s",
                status, result.returncode, attempt + 1, retries, result.stderr[-500:],
            )
            if on_attempt_failure is not None:
                on_attempt_failure(dict(info))
            if status and not retryable and not status.startswith("2"):
                return None
        except (OSError, subprocess.SubprocessError) as exc:
            info.update({"attempts": attempt + 1, "error": f"{type(exc).__name__}: {exc}"})
            log.warning("%s for %s (attempt %d/%d): %s", type(exc).__name__, url, attempt + 1, retries, exc)
            if on_attempt_failure is not None:
                on_attempt_failure(dict(info))
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if header_path and header_path.exists():
                header_path.unlink(missing_ok=True)
        if attempt + 1 < retries:
            time.sleep(retry_after if retry_after is not None else _failure_delay(attempt, info))
    return None


def _failure_delay(attempt: int, status_info: dict) -> float:
    status = str(status_info.get("http_status") or "")
    if status.startswith("5") or status == "429":
        base = min(300.0, 5.0 * (2 ** attempt))
    else:
        base = min(60.0, 2.0 ** attempt)
    return base * random.uniform(0.8, 1.2)


def _retry_after_seconds(header_path: Path) -> float | None:
    if not header_path.exists():
        return None
    try:
        blocks = header_path.read_text(encoding="latin-1").split("\r\n\r\n")
    except OSError:
        return None
    for line in reversed("\r\n".join(blocks).splitlines()):
        if line.lower().startswith("retry-after:"):
            value = line.split(":", 1)[1].strip()
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    date = email.utils.parsedate_to_datetime(value)
                    return max(0.0, (date - datetime.now(date.tzinfo)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    return None
    return None


def _status_db_path(raw_dir: Path, symbol: str) -> Path:
    return Path(raw_dir) / "dukascopy" / symbol / "download_status.sqlite3"


def _record_status(raw_dir: Path, symbol: str, hour: datetime, status: str, details: dict | None = None) -> None:
    path = _status_db_path(raw_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    details = details or {}
    with _STATUS_LOCK:
        db = sqlite3.connect(path, timeout=30)
        try:
            with db:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS hour_status (
                        hour TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL,
                        http_status TEXT,
                        error TEXT,
                        retry_after_seconds REAL,
                        updated_at TEXT NOT NULL
                    )
                """)
                columns = {
                    row[1] for row in db.execute("PRAGMA table_info(hour_status)").fetchall()
                }
                if "retry_after_seconds" not in columns:
                    db.execute("ALTER TABLE hour_status ADD COLUMN retry_after_seconds REAL")
                db.execute(
                    """INSERT INTO hour_status(
                           hour,status,attempts,http_status,error,retry_after_seconds,updated_at
                       ) VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(hour) DO UPDATE SET status=excluded.status, attempts=excluded.attempts,
                       http_status=excluded.http_status,error=excluded.error,
                       retry_after_seconds=excluded.retry_after_seconds,updated_at=excluded.updated_at""",
                    (
                        _utc(hour).isoformat(),
                        status,
                        int(details.get("attempts", 0)),
                        details.get("http_status"),
                        details.get("error"),
                        details.get("retry_after_seconds"),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
        finally:
            db.close()


def _hour_statuses(raw_dir: Path, symbol: str) -> dict[str, dict]:
    path = _status_db_path(raw_dir, symbol)
    if not path.exists():
        return {}
    db = sqlite3.connect(path, timeout=30)
    try:
        try:
            columns = {
                row[1] for row in db.execute("PRAGMA table_info(hour_status)").fetchall()
            }
            if not columns:
                return {}
            retry_after_column = "retry_after_seconds" if "retry_after_seconds" in columns else "NULL"
            rows = db.execute(
                "SELECT hour,status,attempts,http_status,error," + retry_after_column + " FROM hour_status"
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
    finally:
        db.close()
    return {
        row[0]: {
            "status": row[1], "attempts": row[2], "http_status": row[3],
            "error": row[4], "retry_after_seconds": row[5],
        }
        for row in rows
    }


def _valid_cached_file(path: Path, hour: datetime) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size == 0:  # Dukascopy returns 404 for closed/no-data hours.
        return True
    try:
        decode_bi5(path.read_bytes(), hour)
    except (OSError, lzma.LZMAError, ValueError, struct.error):
        return False
    return True


def _cached_corruption_reason(path: Path, hour: datetime) -> str:
    try:
        decode_bi5(path.read_bytes(), hour)
    except (OSError, lzma.LZMAError, ValueError, struct.error) as exc:
        return f"cached BI5 invalid: {type(exc).__name__}: {exc}"
    return "cached BI5 validated"


def _download_hour(args: tuple[str, Path, datetime]) -> tuple[datetime, str]:
    symbol, raw_dir, hour = args
    path = hour_path(raw_dir, symbol, hour)
    if _valid_cached_file(path, hour):
        if path.stat().st_size == 0 and _utc(hour).isoformat() not in _hour_statuses(raw_dir, symbol):
            _record_status(raw_dir, symbol, hour, "empty")
        return hour, "cached"
    if path.exists():
        log.warning("corrupt cached BI5 file will be refreshed: %s", path)
    details = {}
    payload = _fetch(
        hour_url(symbol, hour),
        expected_hour=hour,
        status_info=details,
        on_attempt_failure=lambda failure: _record_status(raw_dir, symbol, hour, "failed", failure),
    )
    if payload is None:
        if str(details.get("error", "")).startswith("invalid_bi5:"):
            _record_status(raw_dir, symbol, hour, "corrupt", details)
            return hour, "corrupt"
        _record_status(raw_dir, symbol, hour, "failed", details)
        return hour, "rate_limited" if details.get("http_status") == "429" else "failed"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    try:
        temp.write_bytes(payload)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    status = "downloaded" if payload else "empty"
    _record_status(raw_dir, symbol, hour, status, details)
    return hour, status


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def iter_hours(start: datetime, end: datetime):
    """Yield whole hours in [start, end), capped at the current hour."""
    if _utc(start) >= _utc(end):
        raise ValueError("start must be earlier than end")
    cur = _utc(start).replace(minute=0, second=0, microsecond=0)
    last = min(_utc(end), datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    while cur < last:
        yield cur
        cur += timedelta(hours=1)


def download_range(symbol: str, start: datetime, end: datetime, raw_dir: Path, workers: int = 1) -> dict[str, int]:
    """Download raw hourly files; validated cached files are skipped and preserved."""
    if workers != 1:
        raise ValueError("Dukascopy downloads are serialized to one worker to respect provider rate limits")
    hours = list(iter_hours(start, end))
    stats = {"cached": 0, "downloaded": 0, "empty": 0, "failed": 0, "corrupt": 0, "rate_limited": 0}
    for i, hour in enumerate(hours, 1):
        _, status = _download_hour((symbol, Path(raw_dir), hour))
        if status in ("rate_limited", "corrupt"):
            if status == "corrupt":
                stats["corrupt"] += 1
            else:
                stats["failed"] += 1
                stats["rate_limited"] += 1
            log.error("Dukascopy returned repeated %s responses; pausing the range download at %s", status, hour)
            break
        stats[status] += 1
        if status == "failed":
            log.error(
                "Dukascopy request failed after retries; pausing at %s. "
                "Re-run to retry this and later missing hours without refetching cached files.",
                hour,
            )
            break
        if i % 200 == 0:
            log.info("progress %d/%d %s", i, len(hours), stats)
    log.info("download finished: %s", stats)
    return stats


def audit_range(symbol: str, start: datetime, end: datetime, raw_dir: Path) -> dict:
    """Audit every requested hour and persist a complete per-hour CSV report."""
    hours = list(iter_hours(start, end))
    history = _hour_statuses(raw_dir, symbol)
    result = {
        "hours_expected": len(hours), "valid": 0, "empty": 0, "missing": 0,
        "failed": 0, "corrupt": 0, "missing_samples": [], "failed_samples": [],
        "corrupt_samples": [],
    }
    rows = []
    for hour in hours:
        path = hour_path(raw_dir, symbol, hour)
        recorded = history.get(_utc(hour).isoformat(), {})
        if not path.exists():
            if recorded.get("status") in ("failed", "corrupt"):
                state = recorded["status"]
                reason = recorded.get("error") or f"last recorded status: {state}"
                result[state] += 1
                if state == "failed" and len(result["failed_samples"]) < 10:
                    result["failed_samples"].append({
                        "hour": hour.isoformat(), "http_status": recorded.get("http_status"),
                        "retry_after_seconds": recorded.get("retry_after_seconds"),
                        "reason": reason,
                        "error": recorded.get("error"),
                    })
                elif state == "corrupt" and len(result["corrupt_samples"]) < 10:
                    result["corrupt_samples"].append({"hour": hour.isoformat(), "error": reason})
            else:
                state = "missing"
                reason = "hour not downloaded"
                result["missing"] += 1
                if len(result["missing_samples"]) < 10:
                    result["missing_samples"].append(hour.isoformat())
        elif path.stat().st_size == 0:
            state = "empty"
            reason = recorded.get("error") or "zero-byte cached BI5; source response is unknown"
            result["empty"] += 1
        elif _valid_cached_file(path, hour):
            state = "valid"
            reason = "BI5 decompressed and validated"
            result["valid"] += 1
        else:
            state = "corrupt"
            reason = _cached_corruption_reason(path, hour)
            if recorded.get("error"):
                reason += f"; last download attempt: {recorded['error']}"
            result["corrupt"] += 1
            if len(result["corrupt_samples"]) < 10:
                result["corrupt_samples"].append({"hour": hour.isoformat(), "error": reason})
        rows.append({
            "hour_utc": hour.isoformat(),
            "status": state,
            "bytes": path.stat().st_size if path.exists() else None,
            "attempts": recorded.get("attempts", 0),
            "http_status": recorded.get("http_status"),
            "retry_after_seconds": recorded.get("retry_after_seconds"),
            "reason": reason,
            "error": recorded.get("error"),
        })
    report_path = _status_db_path(raw_dir, symbol).with_name(
        f"coverage_{_utc(start):%Y%m%d}_{_utc(end):%Y%m%d}.csv"
    )
    pd.DataFrame(rows).to_csv(report_path, index=False)
    result["hourly_report"] = str(report_path)
    return result


def build_m1_from_raw(
    symbol: str,
    start: datetime,
    end: datetime,
    raw_dir: Path,
    point_value: float = 1000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode cached raw hours; gaps are reported but never synthesized."""
    bids, asks, bad, missing = [], [], 0, 0
    for hour in iter_hours(start, end):
        path = hour_path(raw_dir, symbol, hour)
        if not path.exists():
            missing += 1
            continue
        if path.stat().st_size == 0:
            continue
        try:
            bid, ask = ticks_to_m1(decode_bi5(path.read_bytes(), hour, point_value))
        except (OSError, lzma.LZMAError, ValueError, struct.error) as exc:
            bad += 1
            log.error("corrupt raw file %s: %s", path, exc)
            continue
        bids.append(bid)
        asks.append(ask)
    if bad:
        log.warning("%d corrupt raw files skipped (they will be retried by download)", bad)
    if missing:
        log.warning("%d raw hours are missing in the requested range", missing)
    if not bids:
        raise RuntimeError("No raw data decoded. Run the download step first (needs internet).")
    return pd.concat(bids).sort_index(), pd.concat(asks).sort_index()
