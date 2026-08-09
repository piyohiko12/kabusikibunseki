import unittest

import pandas as pd

from lib import data_fetcher, moomoo_fetcher


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
