import unittest
from datetime import date, datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from lib import event_intelligence as ei


def history_frame(periods=180, end="2026-08-07"):
    index = pd.bdate_range(end=end, periods=periods)
    close = pd.Series(100 + np.arange(periods) * 0.1, index=index, dtype=float)
    frame = pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": 1_000_000,
        "Dividends": 0.0,
        "Stock Splits": 0.0,
    }, index=index)
    return frame


class MacroCalendarTests(unittest.TestCase):
    def test_official_2026_dates_sessions_and_urls_are_exposed(self):
        events = ei.default_macro_events()
        by_key = {(row["kind"], row["event_date"]): row for row in events}
        self.assertEqual(by_key[("cpi", "2026-08-12")]["session"], "PRE_MARKET")
        self.assertEqual(by_key[("employment", "2026-09-04")]["event_time_et"], "08:30")
        self.assertEqual(by_key[("fomc", "2026-09-16")]["session"], "REGULAR")
        self.assertTrue(by_key[("cpi", "2026-08-12")]["url"].startswith("https://www.bls.gov/"))
        self.assertTrue(by_key[("fomc", "2026-09-16")]["url"].startswith(
            "https://www.federalreserve.gov/"))
        self.assertEqual(by_key[("employment", "2026-09-04")]["verified_at"],
                         ei.CALENDAR_VERIFIED_AT)

    def test_macro_calendar_can_be_replaced_by_caller(self):
        custom = [{
            "kind": "cpi", "name": "差替CPI", "event_date": "2026-08-13",
            "event_time_et": "08:30", "session": "PRE_MARKET",
            "source": "Official", "url": "https://example.test/cpi",
            "verified_at": "2026-08-11",
        }]
        report = ei.build_event_intelligence(
            "AAPL", history_frame(), macro_events=custom,
            as_of=date(2026, 8, 10), fetched_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        )
        macro = next(event for event in report["events"] if event["kind"] == "cpi")
        self.assertEqual(macro["event_date"], "2026-08-13")
        self.assertEqual(macro["name"], "差替CPI")
        self.assertEqual(macro["freshness"]["source_as_of"], "2026-08-11")


class EventStudyTests(unittest.TestCase):
    def test_event_study_reports_magnitude_direction_volatility_and_ratio(self):
        frame = history_frame(periods=90, end="2026-08-07")
        event_dates = [frame.index[i].date() for i in (20, 35, 50, 65, 75)]
        for i in (20, 35, 50, 65, 75):
            frame.iloc[i, frame.columns.get_loc("Close")] *= 1.05
        study = ei.event_study(frame, event_dates, as_of=date(2026, 8, 10))
        self.assertEqual(study["sample_size"], 5)
        self.assertGreater(study["median_abs_move_pct"], 4.0)
        self.assertGreater(study["sensitivity_ratio"], 2.0)
        self.assertIsNotNone(study["up_rate_pct"])
        self.assertIsNotNone(study["volatility_20d_annualized_pct"])
        self.assertIn("因果", study["method_note"])

    def test_malformed_or_short_history_is_safe(self):
        study = ei.event_study(pd.DataFrame({"x": [1, 2]}), ["2026-01-01"])
        self.assertEqual(study["sample_size"], 0)
        self.assertIsNone(study["median_abs_move_pct"])

    def test_current_incomplete_daily_bar_is_excluded(self):
        frame = history_frame(periods=30, end="2026-08-10")
        frame.loc[pd.Timestamp("2026-08-10"), "Close"] *= 2
        study = ei.event_study(frame, [], as_of=date(2026, 8, 10))
        self.assertEqual(study["price_through"], "2026-08-07")


class BuildReportTests(unittest.TestCase):
    def test_report_contains_explainable_fields_and_never_changes_trade_score(self):
        frame = history_frame()
        earnings_dates = [str(frame.index[i].date()) for i in (20, 50, 80, 110, 140)]
        report = ei.build_event_intelligence(
            "US.AAPL", frame,
            info={"name": "Apple Inc.", "sector": "Technology"},
            analyst={"earnings_date": "2026-10-29", "eps_estimate": 1.72},
            earnings_dates=earnings_dates,
            news=[{
                "title": "Apple raises guidance after hours",
                "summary": "Demand exceeded prior assumptions.",
                "pub_date": "2026-08-09 20:05", "provider": "Example News",
                "url": "https://example.test/news",
            }],
            as_of=date(2026, 8, 10),
            fetched_at=datetime(2026, 8, 10, 3, tzinfo=timezone.utc),
        )
        self.assertEqual(report["ticker"], "AAPL")
        self.assertEqual(report["score_effect"], 0)
        self.assertFalse(report["automatic_trade_score"])
        self.assertFalse(report["meta"]["moomoo_history_quota_used"])
        self.assertGreaterEqual(report["event_count"], 5)
        for event in report["events"]:
            self.assertEqual(set(event), set(ei.EVENT_FIELDS))
            self.assertEqual(set(event["scenarios"]), {"up", "down", "two_sided"})
            self.assertEqual(event["score_effect"], 0)
            self.assertFalse(event["automatic_trade_score"])
            self.assertIn(event["impact_level"], {"HIGH", "MEDIUM", "LOW", "UNKNOWN"})
            self.assertIn(event["confidence_label"], {"HIGH", "MEDIUM", "LOW"})
            self.assertTrue(event["evidence"])
            self.assertIn("price_through", event["freshness"])

        earnings = next(event for event in report["events"] if event["kind"] == "earnings"
                        and event["status"] == "UPCOMING")
        self.assertEqual(earnings["event_date"], "2026-10-29")
        news = next(event for event in report["events"] if event["source"] == "Example News")
        self.assertEqual(news["session"], "AFTER_MARKET")

    def test_earnings_calendar_session_is_used_when_available(self):
        report = ei.build_event_intelligence(
            "AAPL", history_frame(),
            analyst={"earnings_date": "2026-08-20", "pub_type": "AFTER_MARKET"},
            macro_events=[], as_of=date(2026, 8, 10))
        self.assertEqual(report["events"][0]["session"], "AFTER_MARKET")

    def test_recent_dividend_and_split_are_preserved_as_context(self):
        frame = history_frame()
        frame.loc[frame.index[-10], "Dividends"] = 0.25
        frame.loc[frame.index[-5], "Stock Splits"] = 4.0
        report = ei.build_event_intelligence(
            "AAPL", frame, as_of=date(2026, 8, 10), macro_events=[])
        kinds = {event["kind"] for event in report["events"]}
        self.assertEqual(kinds, {"dividend", "split"})
        for event in report["events"]:
            self.assertEqual(event["session"], "REGULAR_OPEN")
            self.assertEqual(event["status"], "RECENT")

    def test_only_material_recent_news_is_included_and_deduplicated(self):
        news = [
            None,
            {"title": "Routine analyst opinion", "pub_date": "2026-08-09"},
            {"title": "Company announces stock split", "pub_date": "2026-08-09",
             "url": "https://example.test/duplicate-source"},
            {"title": "Company announces stock split", "pub_date": "2026-08-09",
             "url": "https://example.test/split"},
            {"title": "Old merger report", "pub_date": "2026-01-01"},
        ]
        report = ei.build_event_intelligence(
            "AAPL", history_frame(), news=news, macro_events=[],
            as_of=date(2026, 8, 10))
        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(report["events"][0]["kind"], "split")

        no_news = ei.build_event_intelligence(
            "AAPL", history_frame(), news=news, macro_events=[],
            as_of=date(2026, 8, 10), max_news_events=0)
        self.assertEqual(no_news["events"], [])

    def test_short_keyword_does_not_match_inside_unrelated_word(self):
        self.assertIsNone(ei.classify_news_event({
            "title": "Company takes steps toward a new office",
            "summary": "Routine update",
        }))

    def test_missing_history_keeps_upcoming_events_with_low_confidence(self):
        report = ei.build_event_intelligence(
            "AAPL", pd.DataFrame(), analyst={"earnings_date": "2026-08-20"},
            macro_events=[], as_of=date(2026, 8, 10))
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["events"][0]["confidence_label"], "LOW")
        self.assertTrue(report["warnings"])

    def test_invalid_ticker_is_rejected(self):
        with self.assertRaises(ValueError):
            ei.build_event_intelligence("../../bad", history_frame())


class FetchBoundaryTests(unittest.TestCase):
    def test_partial_network_failure_keeps_available_events_and_reports_sources(self):
        frame = history_frame()
        with (
            patch.object(ei.data_fetcher, "fetch_info", return_value={"name": "Apple"}),
            patch.object(ei.data_fetcher, "fetch_analyst",
                         return_value={"earnings_date": "2026-08-20"}),
            patch.object(ei.data_fetcher, "fetch_earnings_history",
                         side_effect=RuntimeError("earnings offline")),
            patch.object(ei.data_fetcher, "fetch_history", return_value=frame) as history,
            patch.object(ei.news_fetcher, "fetch_news", side_effect=RuntimeError("news offline")),
            patch.object(ei.news_fetcher, "fetch_sec_filings", return_value=[]),
        ):
            report = ei._fetch_event_intelligence_uncached(
                "US.AAPL", as_of_date="2026-08-10")

        history.assert_called_once_with("AAPL", "2y", "1d")
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["source_status"]["earnings_history"], "failed")
        self.assertEqual(report["source_status"]["yahoo_news"], "failed")
        self.assertIn("earnings_history", report["errors"])
        self.assertTrue(any(event["kind"] == "earnings" for event in report["events"]))
        self.assertFalse(report["meta"]["moomoo_history_quota_used"])

    def test_include_news_false_skips_both_news_sources(self):
        with (
            patch.object(ei.data_fetcher, "fetch_info", return_value={}),
            patch.object(ei.data_fetcher, "fetch_analyst", return_value={}),
            patch.object(ei.data_fetcher, "fetch_earnings_history", return_value=[]),
            patch.object(ei.data_fetcher, "fetch_history", return_value=history_frame()),
            patch.object(ei.news_fetcher, "fetch_news") as yahoo_news,
            patch.object(ei.news_fetcher, "fetch_sec_filings") as sec,
        ):
            report = ei._fetch_event_intelligence_uncached(
                "AAPL", as_of_date="2026-08-10", include_news=False)
        yahoo_news.assert_not_called()
        sec.assert_not_called()
        self.assertNotIn("yahoo_news", report["source_status"])
        self.assertNotIn("sec_filings", report["source_status"])

    def test_cache_clear_is_public_and_bad_as_of_is_rejected(self):
        self.assertTrue(callable(ei.fetch_event_intelligence.clear))
        with self.assertRaises(ValueError):
            ei._fetch_event_intelligence_uncached("AAPL", as_of_date="not-a-date")


if __name__ == "__main__":
    unittest.main()
