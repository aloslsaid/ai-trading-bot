import lzma
import struct
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from gold_ai.data_engine import dukascopy


def _payload():
    record = struct.pack(">IIIff", 1_000, 2_800_500, 2_800_000, 1.0, 2.0)
    return lzma.compress(record)


def _write_response_files(command, status="200", payload=b"", retry_after=None):
    headers = Path(command[command.index("--dump-header") + 1])
    output = Path(command[command.index("--output") + 1])
    header = f"HTTP/1.1 {status} response\r\n"
    if retry_after is not None:
        header += f"Retry-After: {retry_after}\r\n"
    headers.write_text(header + "\r\n", encoding="ascii")
    output.write_bytes(payload)


class DownloadTests(unittest.TestCase):
    def test_retry_after_is_read_from_response_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.headers"
            path.write_text(
                "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 17\r\n\r\n",
                encoding="ascii",
            )
            self.assertEqual(dukascopy._retry_after_seconds(path), 17)

    def test_server_backoff_uses_exponential_jitter(self):
        with patch("gold_ai.data_engine.dukascopy.random.uniform", return_value=1.2):
            self.assertEqual(dukascopy._failure_delay(1, {"http_status": "503"}), 12.0)
        with patch("gold_ai.data_engine.dukascopy.random.uniform", return_value=0.8):
            self.assertEqual(dukascopy._failure_delay(0, {"http_status": "502"}), 4.0)

    def test_http_503_retries_with_exponential_backoff_then_succeeds(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        attempts = []

        def fake_curl(command, **kwargs):
            attempts.append(command)
            if len(attempts) == 1:
                _write_response_files(command, "503")
                return subprocess.CompletedProcess(command, 22, "503", "Service Unavailable")
            _write_response_files(command, payload=_payload())
            return subprocess.CompletedProcess(command, 0, "200", "")

        with patch("gold_ai.data_engine.dukascopy.subprocess.run", side_effect=fake_curl):
            with patch("gold_ai.data_engine.dukascopy.time.sleep") as sleep:
                with patch("gold_ai.data_engine.dukascopy.random.uniform", return_value=1.0):
                    result = dukascopy._fetch(
                        "https://example.invalid/hour.bi5", retries=2, expected_hour=hour
                    )

        self.assertEqual(result, _payload())
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(5.0)

    def test_http_429_honors_retry_after_before_retrying(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        attempts = []

        def fake_curl(command, **kwargs):
            attempts.append(command)
            if len(attempts) == 1:
                _write_response_files(command, "429", retry_after="7")
                return subprocess.CompletedProcess(command, 22, "429", "rate limited")
            _write_response_files(command, payload=_payload())
            return subprocess.CompletedProcess(command, 0, "200", "")

        with patch("gold_ai.data_engine.dukascopy.subprocess.run", side_effect=fake_curl):
            with patch("gold_ai.data_engine.dukascopy.time.sleep") as sleep:
                result = dukascopy._fetch(
                    "https://example.invalid/hour.bi5", retries=2, expected_hour=hour
                )
        self.assertEqual(result, _payload())
        self.assertEqual(len(attempts), 2)
        sleep.assert_called_once_with(7.0)

    def test_http_200_empty_is_an_empty_hour_not_a_failed_transfer(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        calls = []

        def fake_curl(command, **kwargs):
            calls.append(command)
            _write_response_files(command, payload=b"")
            return subprocess.CompletedProcess(command, 0, "200", "")

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            with patch("gold_ai.data_engine.dukascopy.subprocess.run", side_effect=fake_curl):
                with patch("gold_ai.data_engine.dukascopy.time.sleep") as sleep:
                    self.assertEqual(
                        dukascopy._download_hour(("XAUUSD", raw, hour))[1], "empty"
                    )
            audit = dukascopy.audit_range("XAUUSD", hour, hour.replace(hour=11), raw)
            report = Path(audit["hourly_report"]).read_text(encoding="utf-8")
        self.assertEqual(len(calls), 1)
        self.assertEqual(audit["empty"], 1)
        self.assertEqual(audit["failed"], 0)
        self.assertIn("HTTP 200: empty response body; no ticks", report)
        sleep.assert_not_called()

    def test_corrupt_http_200_is_not_persisted_and_is_reported_as_corrupt(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        corrupt_payload = lzma.compress(b"not-a-bi5-record")

        def fake_curl(command, **kwargs):
            _write_response_files(command, payload=corrupt_payload)
            return subprocess.CompletedProcess(command, 0, "200", "")

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            with patch("gold_ai.data_engine.dukascopy.subprocess.run", side_effect=fake_curl):
                with patch("gold_ai.data_engine.dukascopy.time.sleep"):
                    result = dukascopy._download_hour(("XAUUSD", raw, hour))
            path = dukascopy.hour_path(raw, "XAUUSD", hour)
            audit = dukascopy.audit_range("XAUUSD", hour, hour.replace(hour=11), raw)
        self.assertEqual(result[1], "corrupt")
        self.assertFalse(path.exists())
        self.assertEqual(audit["corrupt"], 1)
        self.assertEqual(audit["failed"], 0)
        self.assertIn("invalid_bi5", audit["corrupt_samples"][0]["error"])

    def test_timeout_retries_and_records_connection_error(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        attempts = []
        failures = []

        def fake_curl(command, **kwargs):
            attempts.append(command)
            if len(attempts) == 1:
                raise subprocess.TimeoutExpired(command, timeout=1)
            _write_response_files(command, payload=_payload())
            return subprocess.CompletedProcess(command, 0, "200", "")

        with patch("gold_ai.data_engine.dukascopy.subprocess.run", side_effect=fake_curl):
            with patch("gold_ai.data_engine.dukascopy.time.sleep"):
                result = dukascopy._fetch(
                    "https://example.invalid/hour.bi5",
                    retries=2,
                    expected_hour=hour,
                    on_attempt_failure=failures.append,
                )
        self.assertEqual(result, _payload())
        self.assertEqual(len(attempts), 2)
        self.assertIn("TimeoutExpired", failures[0]["error"])

    def test_resume_retries_failed_hour_then_skips_valid_cached_file(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            details = {"attempts": 5, "http_status": "503", "error": "HTTP 503"}
            dukascopy._record_status(raw, "XAUUSD", hour, "failed", details)
            with patch("gold_ai.data_engine.dukascopy._fetch", return_value=_payload()) as fetch:
                first = dukascopy._download_hour(("XAUUSD", raw, hour))
                second = dukascopy._download_hour(("XAUUSD", raw, hour))
            audit = dukascopy.audit_range("XAUUSD", hour, hour.replace(hour=11), raw)
        self.assertEqual(first[1], "downloaded")
        self.assertEqual(second[1], "cached")
        fetch.assert_called_once()
        self.assertEqual(audit["valid"], 1)
        self.assertEqual(audit["failed"], 0)

    def test_cached_corrupt_file_is_refetched_and_valid_file_resumes(self):
        hour = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            path = dukascopy.hour_path(Path(directory), "XAUUSD", hour)
            path.parent.mkdir(parents=True)
            path.write_bytes(b"corrupt")
            with patch("gold_ai.data_engine.dukascopy._fetch", return_value=_payload()) as fetch:
                self.assertEqual(
                    dukascopy._download_hour(("XAUUSD", Path(directory), hour))[1],
                    "downloaded",
                )
                self.assertEqual(
                    dukascopy._download_hour(("XAUUSD", Path(directory), hour))[1],
                    "cached",
                )
            fetch.assert_called_once()
            self.assertEqual(path.read_bytes(), _payload())

    def test_audit_distinguishes_valid_empty_corrupt_and_missing_hours(self):
        start = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            dukascopy.hour_path(raw, "XAUUSD", start).parent.mkdir(parents=True)
            dukascopy.hour_path(raw, "XAUUSD", start).write_bytes(_payload())
            dukascopy.hour_path(raw, "XAUUSD", start.replace(hour=11)).write_bytes(b"corrupt")
            dukascopy.hour_path(raw, "XAUUSD", start.replace(hour=12)).write_bytes(b"")
            audit = dukascopy.audit_range("XAUUSD", start, start.replace(hour=14), raw)
            report = Path(audit["hourly_report"]).read_text(encoding="utf-8")
        self.assertEqual(audit["hours_expected"], 4)
        self.assertEqual(audit["valid"], 1)
        self.assertEqual(audit["corrupt"], 1)
        self.assertEqual(audit["empty"], 1)
        self.assertEqual(audit["missing"], 1)
        self.assertIn("cached BI5 invalid", report)

    def test_audit_reports_failed_hours_separately_and_writes_per_hour_report(self):
        start = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            details = {"attempts": 5, "http_status": "503", "error": "Service Unavailable"}
            dukascopy._record_status(raw, "XAUUSD", start, "failed", details)
            audit = dukascopy.audit_range("XAUUSD", start, start.replace(hour=12), raw)
            report = Path(audit["hourly_report"]).read_text(encoding="utf-8")
        self.assertEqual(audit["failed"], 1)
        self.assertEqual(audit["missing"], 1)
        self.assertEqual(audit["failed_samples"][0]["http_status"], "503")
        self.assertIn("failed", report)
        self.assertIn("missing", report)
        self.assertIn("Service Unavailable", report)

    def test_single_worker_pauses_after_retry_exhaustion(self):
        start = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        with patch(
            "gold_ai.data_engine.dukascopy._download_hour",
            return_value=(start, "failed"),
        ) as download:
            stats = dukascopy.download_range(
                "XAUUSD", start, start.replace(hour=12), Path("unused"), workers=1
            )
        self.assertEqual(stats["failed"], 1)
        download.assert_called_once()

    def test_download_rejects_more_than_one_worker(self):
        start = datetime(2026, 2, 2, 10, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "serialized"):
            dukascopy.download_range(
                "XAUUSD", start, start.replace(hour=11), Path("unused"), workers=2
            )


if __name__ == "__main__":
    unittest.main()
