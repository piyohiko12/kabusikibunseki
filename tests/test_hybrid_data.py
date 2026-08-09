import pathlib
import unittest
from unittest import mock

import pandas as pd

from lib import data_fetcher, moomoo_client, moomoo_fetcher, settings_store


class MoomooFetcherTests(unittest.TestCase):
    def test_normalize_code(self):
        self.assertEqual(moomoo_fetcher.normalize_code("AAPL"), "US.AAPL")
        self.assertEqual(moomoo_fetcher.normalize_code("US.NVDA"), "US.NVDA")
        self.assertEqual(moomoo_fetcher.normalize_code("700.HK"), "HK.00700")
        self.assertEqual(moomoo_fetcher.normalize_code("7203.T"), "JP.7203")

    def test_yahoo_only_code_is_rejected_for_fallback(self):
        with self.assertRaises(moomoo_fetcher.MoomooError):
            moomoo_fetcher.normalize_code("^GSPC")

    def test_period_start(self):
        now = pd.Timestamp("2026-08-09")
        self.assertEqual(moomoo_fetcher._period_start("1y", now), "2025-08-09")
        self.assertEqual(moomoo_fetcher._period_start("5d", now), "2026-08-01")

    def test_normalise_history(self):
        source = pd.DataFrame({
            "time_key": ["2026-08-07 09:30:00", "2026-08-07 09:31:00"],
            "open": [100, 101], "high": [102, 103], "low": [99, 100],
            "close": [101, 102], "volume": [1000, 1200],
        })
        result = moomoo_fetcher._normalise_history(source)
        self.assertEqual(list(result.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual(float(result.iloc[-1]["Close"]), 102.0)
        self.assertTrue(result.index.is_monotonic_increasing)


class HybridDataTests(unittest.TestCase):
    def test_corporate_actions_are_merged_by_date(self):
        primary = pd.DataFrame(
            {"Open": [100], "High": [102], "Low": [99], "Close": [101],
             "Volume": [1000]},
            index=pd.to_datetime(["2026-08-07"]),
        )
        yahoo = pd.DataFrame(
            {"Dividends": [0.25], "Stock Splits": [0.0]},
            index=pd.DatetimeIndex(["2026-08-07"], tz="America/New_York"),
        )
        result = data_fetcher._merge_corporate_actions(primary, yahoo, "1d")
        self.assertEqual(float(result.iloc[0]["Dividends"]), 0.25)
        self.assertEqual(float(result.iloc[0]["Stock Splits"]), 0.0)


if __name__ == "__main__":
    unittest.main()


class ConnectionIsolationTests(unittest.TestCase):
    """OpenDが応答しないときにUIを固まらせないための約束事。"""

    def test_module_never_creates_its_own_quote_context(self):
        # OpenQuoteContextはOpenD未起動だと例外を返さず無限に再接続する。
        # 接続はmoomoo_client(TCP事前確認+タイムアウト)経由に限定する。
        source = pathlib.Path(moomoo_fetcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("OpenQuoteContext(", source)

    def test_open_context_delegates_to_moomoo_client(self):
        with mock.patch.object(moomoo_client, "_ctx", return_value="CTX") as ctx:
            self.assertEqual(moomoo_fetcher._open_context(), "CTX")
        ctx.assert_called_once_with()

    def test_open_context_raises_instead_of_blocking(self):
        with mock.patch.object(moomoo_client, "_ctx", return_value=None), \
                mock.patch.object(moomoo_client, "status",
                                  return_value={"state": "no_opend",
                                                "message": "OpenDに接続できません"}):
            with self.assertRaises(moomoo_fetcher.MoomooError):
                moomoo_fetcher._open_context()

    def test_shared_context_is_not_closed(self):
        source = pathlib.Path(moomoo_fetcher.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ctx.close()", source)


class HistoryQuotaOptInTests(unittest.TestCase):
    """歴史的K線クォータは30日戻らないので、既定では消費しない。"""

    def test_history_is_disabled_by_default(self):
        for settings in ({}, {"moomoo_enabled": True},
                         {"moomoo_chart_history": True}):
            with self.subTest(settings=settings):
                with mock.patch.object(settings_store, "load",
                                       return_value=settings):
                    self.assertFalse(moomoo_fetcher.history_enabled())

    def test_history_needs_both_switches(self):
        with mock.patch.object(settings_store, "load",
                               return_value={"moomoo_enabled": True,
                                             "moomoo_chart_history": True}):
            self.assertTrue(moomoo_fetcher.history_enabled())

    def test_fetch_history_refuses_before_touching_opend(self):
        with mock.patch.object(moomoo_fetcher, "history_enabled",
                               return_value=False), \
                mock.patch.object(moomoo_fetcher, "_open_context") as ctx:
            with self.assertRaises(moomoo_fetcher.MoomooError):
                moomoo_fetcher.fetch_history("AAPL", "1y", "1d")
        ctx.assert_not_called()

    def test_chart_history_skips_moomoo_when_opted_out(self):
        frame = pd.DataFrame({"Close": [1.0]},
                             index=pd.DatetimeIndex(["2026-08-07"]))
        with mock.patch.object(moomoo_fetcher, "history_enabled",
                               return_value=False), \
                mock.patch.object(moomoo_fetcher, "fetch_history") as moomoo, \
                mock.patch.object(data_fetcher, "fetch_history",
                                  return_value=frame):
            result, meta = data_fetcher.fetch_chart_history.__wrapped__(
                "AAPL", "1y", "1d")
        moomoo.assert_not_called()
        self.assertEqual(meta["source"], "Yahoo Finance")
        self.assertIsNone(meta["fallback_reason"])
        self.assertFalse(result.empty)
