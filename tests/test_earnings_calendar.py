import datetime as dt
import unittest
from unittest.mock import patch

import pandas as pd

from lib import data_fetcher


class _CalendarOnlyTicker:
    def __init__(self, calendar):
        self.calendar = calendar

    @property
    def analyst_price_targets(self):
        raise AssertionError("軽量取得で目標株価を参照してはいけません")

    @property
    def recommendations(self):
        raise AssertionError("軽量取得でレーティングを参照してはいけません")

    @property
    def upgrades_downgrades(self):
        raise AssertionError("軽量取得で格付け変更を参照してはいけません")


class EarningsCalendarTests(unittest.TestCase):
    def setUp(self):
        data_fetcher.fetch_earnings_calendar.clear()

    def tearDown(self):
        data_fetcher.fetch_earnings_calendar.clear()

    def test_fetches_only_calendar_and_keeps_public_shape(self):
        ticker = _CalendarOnlyTicker({
            "Earnings Date": [dt.date(2026, 10, 29)],
            "Earnings Average": "1.57",
        })
        with patch.object(data_fetcher.yf, "Ticker", return_value=ticker) as factory:
            result = data_fetcher.fetch_earnings_calendar(" aapl ")

        factory.assert_called_once_with("AAPL")
        self.assertEqual(result, {
            "earnings_date": "2026-10-29",
            "eps_estimate": 1.57,
        })

    def test_supports_legacy_dataframe_calendar(self):
        calendar = pd.DataFrame({"Value": [
            [pd.Timestamp("2026-11-05"), pd.Timestamp("2026-11-06")],
            2.25,
        ]}, index=["Earnings Date", "Earnings Average"])
        with patch.object(
                data_fetcher.yf, "Ticker",
                return_value=_CalendarOnlyTicker(calendar)):
            result = data_fetcher.fetch_earnings_calendar("MSFT")

        self.assertEqual(result["earnings_date"], "2026-11-05 00:00:00")
        self.assertEqual(result["eps_estimate"], 2.25)

    def test_malformed_calendar_and_values_fail_safe(self):
        malformed = [
            None,
            "unexpected",
            ["not", "a", "mapping"],
            {"Earnings Date": "not-a-date", "Earnings Average": "not-a-number"},
            {"Earnings Date": [[[[None]]]], "Earnings Average": float("inf")},
            {"Earnings Date": {"unexpected": "mapping"}, "Earnings Average": True},
        ]
        for index, calendar in enumerate(malformed):
            with self.subTest(index=index), patch.object(
                    data_fetcher.yf, "Ticker",
                    return_value=_CalendarOnlyTicker(calendar)):
                result = data_fetcher.fetch_earnings_calendar.__wrapped__("AAPL")
            self.assertEqual(result, {
                "earnings_date": None,
                "eps_estimate": None,
            })

    def test_constructor_and_calendar_property_exceptions_fail_safe(self):
        expected = {"earnings_date": None, "eps_estimate": None}
        with patch.object(data_fetcher.yf, "Ticker", side_effect=RuntimeError("offline")):
            self.assertEqual(
                data_fetcher.fetch_earnings_calendar.__wrapped__("AAPL"), expected)

        class BrokenTicker:
            @property
            def calendar(self):
                raise AttributeError("schema changed")

        with patch.object(data_fetcher.yf, "Ticker", return_value=BrokenTicker()):
            self.assertEqual(
                data_fetcher.fetch_earnings_calendar.__wrapped__("AAPL"), expected)

        with patch.object(
                data_fetcher.yf, "Ticker",
                return_value=_CalendarOnlyTicker({"Earnings Date": []})), \
                patch.object(data_fetcher, "_calendar_scalar",
                             side_effect=RuntimeError("malformed value")):
            self.assertEqual(
                data_fetcher.fetch_earnings_calendar.__wrapped__("AAPL"), expected)

    def test_empty_symbol_skips_yahoo(self):
        with patch.object(data_fetcher.yf, "Ticker") as factory:
            result = data_fetcher.fetch_earnings_calendar("  ")
        factory.assert_not_called()
        self.assertEqual(result, {
            "earnings_date": None,
            "eps_estimate": None,
        })


if __name__ == "__main__":
    unittest.main()
