import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from lib import market_intelligence as mi


def history_frame(*, rows=260, daily_return=0.001, start=100.0,
                  volume=2_000_000.0, end="2026-08-07") -> pd.DataFrame:
    index = pd.bdate_range(end=end, periods=rows)
    close = start * np.power(1.0 + daily_return, np.arange(rows))
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.002,
        "Low": close * 0.998,
        "Close": close,
        "Volume": np.full(rows, volume),
    }, index=index)


def catalog_histories(*, missing=()) -> dict[str, pd.DataFrame]:
    frames = {mi.MARKET_BENCHMARK: history_frame(daily_return=0.0005)}
    for offset, spec in enumerate(mi.SECTOR_CATALOG):
        frames[spec["etf"]] = history_frame(
            daily_return=0.0002 + offset * 0.00005)
    for symbol in missing:
        frames.pop(symbol, None)
    return frames


def sector_histories(sector="TECHNOLOGY") -> dict[str, pd.DataFrame]:
    spec = next(item for item in mi.SECTOR_CATALOG if item["key"] == sector)
    frames = {
        mi.MARKET_BENCHMARK: history_frame(daily_return=0.0004),
        spec["etf"]: history_frame(daily_return=0.0007),
    }
    returns = (0.0020, 0.0012, 0.0002, -0.0010)
    for symbol, daily_return in zip(spec["candidates"], returns):
        frames[symbol] = history_frame(daily_return=daily_return)
    return frames


class CatalogAndTrendTests(unittest.TestCase):
    def test_catalog_has_all_11_unique_sector_etfs_and_four_candidates(self):
        self.assertEqual(len(mi.SECTOR_CATALOG), 11)
        self.assertEqual(len({item["key"] for item in mi.SECTOR_CATALOG}), 11)
        self.assertEqual(len({item["etf"] for item in mi.SECTOR_CATALOG}), 11)
        self.assertTrue(all(len(item["candidates"]) == 4
                            for item in mi.SECTOR_CATALOG))

    def test_rising_and_falling_histories_have_explainable_extreme_scores(self):
        benchmark = history_frame(daily_return=0.0003)
        rising = mi.score_trend(
            history_frame(daily_return=0.0015), benchmark)
        falling = mi.score_trend(
            history_frame(daily_return=-0.0015), benchmark)

        self.assertEqual(rising["trend"], "up")
        self.assertEqual(rising["score"], 100.0)
        self.assertEqual(falling["trend"], "down")
        self.assertEqual(falling["score"], 0.0)
        self.assertEqual(sum(item["weight"] for item in rising["components"]), 100)
        self.assertEqual(sum(item["points"] for item in rising["components"]), 100)
        self.assertTrue(all(item["detail"] for item in rising["components"]))

    def test_market_score_does_not_fake_relative_strength_against_itself(self):
        result = mi.score_trend(history_frame(daily_return=0.001))
        keys = {item["key"] for item in result["components"]}
        self.assertNotIn("relative_20d_positive", keys)
        self.assertEqual(result["expected_points"], 80.0)
        self.assertEqual(result["score"], 100.0)

    def test_missing_benchmark_is_partial_but_does_not_discard_asset_trend(self):
        result = mi.score_trend(
            history_frame(daily_return=0.001), pd.DataFrame())
        self.assertEqual(result["trend"], "up")
        self.assertEqual(result["data_status"], "partial")
        relative = next(item for item in result["components"]
                        if item["key"] == "relative_20d_positive")
        self.assertFalse(relative["available"])
        self.assertLess(result["coverage_pct"], 100)

    def test_short_invalid_and_infinite_history_is_safe(self):
        short = history_frame(rows=20)
        short.loc[short.index[-1], "Close"] = np.inf
        result = mi.score_trend(short)
        self.assertEqual(result["trend"], "unavailable")
        self.assertEqual(result["data_status"], "unavailable")
        self.assertIsNone(result["score"])
        self.assertTrue(result["warnings"])

        missing_close = mi.score_trend(pd.DataFrame({"Volume": [1, 2, 3]}))
        self.assertEqual(missing_close["trend"], "unavailable")
        self.assertIsNone(missing_close["metrics"]["close"])


class MarketOverviewTests(unittest.TestCase):
    def test_overview_returns_market_all_sectors_breadth_rank_and_safety_meta(self):
        result = mi.analyze_market_frames(catalog_histories())
        self.assertEqual(result["meta"]["status"], "ok")
        self.assertEqual(len(result["sectors"]), 11)
        self.assertEqual(sum(result["breadth"][key]
                             for key in ("up", "down", "neutral", "unavailable")),
                         11)
        self.assertEqual(sorted(item["rank"] for item in result["sectors"]),
                         list(range(1, 12)))
        self.assertFalse(result["meta"]["history_quota_consumed"])
        self.assertFalse(result["meta"]["allow_new_history_quota"])
        self.assertTrue(result["meta"]["read_only"])
        self.assertIn("保証", result["disclaimer"])

    def test_partial_sector_failure_is_explicit_and_other_results_survive(self):
        missing = mi.SECTOR_CATALOG[3]["etf"]
        result = mi.analyze_market_frames(
            catalog_histories(missing=(missing,)),
            errors={missing: "mock timeout"},
        )
        self.assertEqual(result["meta"]["status"], "partial")
        self.assertIn(missing, result["meta"]["failed"])
        self.assertEqual(result["meta"]["errors"][missing], "mock timeout")
        failed_sector = next(item for item in result["sectors"]
                             if item["etf"] == missing)
        self.assertEqual(failed_sector["trend"], "unavailable")
        self.assertEqual(len(result["sectors"]), 11)

    def test_unavailable_short_history_never_receives_sector_rank(self):
        frames = {symbol: frame.tail(21)
                  for symbol, frame in catalog_histories().items()}
        result = mi.analyze_market_frames(frames)
        self.assertEqual(result["meta"]["status"], "unavailable")
        self.assertTrue(all(item["rank"] is None for item in result["sectors"]))
        self.assertEqual(len(result["meta"]["analysis_unavailable"]), 12)

    def test_usable_but_incomplete_history_marks_overview_partial(self):
        frames = {symbol: frame.tail(70)
                  for symbol, frame in catalog_histories().items()}
        result = mi.analyze_market_frames(frames)
        self.assertEqual(result["meta"]["status"], "partial")
        self.assertTrue(all(item["rank"] is not None for item in result["sectors"]))
        self.assertEqual(len(result["meta"]["analysis_partial"]), 12)

    def test_snapshot_only_enriches_freshness_and_does_not_change_daily_score(self):
        frames = catalog_histories()
        without = mi.analyze_market_frames(frames)
        with_snapshot = mi.analyze_market_frames(frames, snapshots={
            "SPY": {"price": 999.0, "change_percent": 1.2,
                    "update_time": "2026-08-07 15:59:00"},
        })
        self.assertEqual(without["market"]["score"], with_snapshot["market"]["score"])
        self.assertEqual(with_snapshot["market"]["realtime_snapshot"]["price"], 999.0)
        self.assertEqual(
            with_snapshot["market"]["realtime_snapshot"]["update_time"],
            "2026-08-07T19:59:00+00:00")
        self.assertEqual(
            with_snapshot["market"]["freshness"]["snapshot_status"], "stale")
        self.assertEqual(
            with_snapshot["market"]["realtime_snapshot"]["timezone_assumption"],
            "America/New_York")
        self.assertIn("moomoo OpenAPI snapshot", with_snapshot["market"]["source"])
        self.assertFalse(with_snapshot["meta"]["history_quota_consumed"])


class CandidateRankingTests(unittest.TestCase):
    def test_strongest_representative_stock_ranks_first_with_full_reasons(self):
        result = mi.rank_sector_candidates(
            "XLK", sector_histories(), top_n=2)
        self.assertEqual(result["sector"]["key"], "TECHNOLOGY")
        self.assertEqual(result["ranking"][0]["symbol"], "AAPL")
        self.assertEqual(len(result["candidates"]), 2)
        first = result["candidates"][0]
        self.assertTrue(first["eligible"])
        self.assertEqual(first["candidate_status"], "候補")
        self.assertEqual(sum(item["weight"] for item in first["components"]), 100)
        self.assertTrue(all(item["detail"] for item in first["components"]))
        self.assertIn("代表4銘柄", result["disclaimer"])
        self.assertIn("保証", result["disclaimer"])

    def test_missing_sector_context_never_promotes_stock_to_candidate(self):
        frames = sector_histories()
        frames.pop("XLK")
        result = mi.rank_sector_candidates("テクノロジー", frames)
        self.assertEqual(result["meta"]["status"], "partial")
        self.assertEqual(result["candidates"], [])
        self.assertTrue(all(not item["eligible"] for item in result["ranking"]))
        self.assertTrue(all(item["candidate_status"] == "データ不足"
                            for item in result["ranking"]))

    def test_missing_market_context_never_promotes_stock_to_candidate(self):
        frames = sector_histories()
        frames.pop(mi.MARKET_BENCHMARK)
        result = mi.rank_sector_candidates("TECHNOLOGY", frames)
        self.assertEqual(result["meta"]["status"], "partial")
        self.assertEqual(result["candidates"], [])
        self.assertTrue(all(not item["eligible"] for item in result["ranking"]))

    def test_short_history_score_is_visible_but_low_confidence_is_not_promoted(self):
        frames = sector_histories()
        for symbol, frame in list(frames.items()):
            frames[symbol] = frame.tail(70)
        result = mi.rank_sector_candidates("TECHNOLOGY", frames)
        self.assertEqual(result["meta"]["status"], "partial")
        self.assertEqual(result["candidates"], [])
        self.assertTrue(all(item["recommendation_score"] is not None
                            for item in result["ranking"]))
        self.assertTrue(all(item["confidence_pct"] < 60
                            for item in result["ranking"]))

    def test_partial_120_day_trend_coverage_propagates_to_candidate(self):
        frames = {symbol: frame.tail(120)
                  for symbol, frame in sector_histories().items()}
        result = mi.rank_sector_candidates("TECHNOLOGY", frames)
        self.assertEqual(result["meta"]["status"], "partial")
        self.assertEqual(result["candidates"], [])
        self.assertTrue(all(item["trend"]["data_status"] == "partial"
                            for item in result["ranking"]))
        self.assertTrue(all(item["coverage_pct"] < 100
                            for item in result["ranking"]))
        self.assertTrue(all(not item["data_complete"]
                            for item in result["ranking"]))

    def test_no_strong_candidate_is_not_misreported_as_data_failure(self):
        frames = sector_histories()
        for symbol in next(item for item in mi.SECTOR_CATALOG
                           if item["key"] == "TECHNOLOGY")["candidates"]:
            frames[symbol] = history_frame(daily_return=-0.001)
        result = mi.rank_sector_candidates("TECHNOLOGY", frames)
        self.assertEqual(result["meta"]["status"], "ok")
        self.assertEqual(result["meta"]["selection_status"], "none")
        self.assertEqual(result["candidates"], [])
        self.assertTrue(all(item["candidate_status"] == "優先度低"
                            for item in result["ranking"]))

    def test_liquidity_is_a_gate_and_not_confused_with_missing_data(self):
        frames = sector_histories()
        for symbol in next(item for item in mi.SECTOR_CATALOG
                           if item["key"] == "TECHNOLOGY")["candidates"]:
            frames[symbol] = history_frame(
                daily_return=0.002, volume=1_000.0)
        result = mi.rank_sector_candidates("TECHNOLOGY", frames)
        self.assertEqual(result["meta"]["status"], "ok")
        self.assertEqual(result["meta"]["data_complete_count"], 4)
        self.assertEqual(result["meta"]["eligible_count"], 0)
        self.assertEqual(result["candidates"], [])
        self.assertTrue(all(item["candidate_status"] == "流動性基準外"
                            for item in result["ranking"]))

    def test_unknown_sector_and_invalid_top_n_fail_before_network(self):
        with self.assertRaises(ValueError):
            mi.rank_sector_candidates("NOT_A_SECTOR", {})
        with self.assertRaises(ValueError):
            mi.rank_sector_candidates("XLK", {}, top_n=0)


class DataAccessSafetyTests(unittest.TestCase):
    def tearDown(self):
        mi.clear_market_intelligence_cache()

    def test_history_bundle_uses_existing_yahoo_layer_and_keeps_partial_success(self):
        good = history_frame()

        def side_effect(symbol, period, interval):
            self.assertEqual(period, "1y")
            self.assertEqual(interval, "1d")
            if symbol == "XLF":
                raise RuntimeError("mock failure")
            return good

        with patch.object(mi.data_fetcher, "fetch_history",
                          side_effect=side_effect) as fetch_history:
            histories, meta = mi._load_history_bundle_uncached(
                ("SPY", "XLK", "XLF"))

        self.assertEqual(fetch_history.call_count, 3)
        self.assertEqual(set(histories), {"SPY", "XLK"})
        self.assertEqual(meta["status"], "partial")
        self.assertEqual(meta["failed"], ("XLF",))
        self.assertFalse(meta["history_quota_consumed"])

    def test_public_overview_batches_optional_moomoo_snapshot_once(self):
        symbols = (mi.MARKET_BENCHMARK,) + tuple(
            item["etf"] for item in mi.SECTOR_CATALOG)
        load_meta = {
            "source": "Yahoo Finance", "fetched_at": "2026-08-08T00:00:00+00:00",
            "errors": {},
        }
        with patch.object(mi, "_load_history_bundle",
                          return_value=(catalog_histories(), load_meta)) as loader, \
                patch.object(mi.moomoo_client, "snapshot",
                             return_value={"SPY": {"price": 650.0}}) as snapshot:
            result = mi.get_us_market_overview(include_realtime=True)

        loader.assert_called_once_with(symbols, "1y")
        snapshot.assert_called_once_with(symbols)
        self.assertEqual(result["meta"]["realtime_received"], 1)
        self.assertEqual(result["meta"]["snapshot_status"], "partial")
        self.assertEqual(result["meta"]["snapshot_received_symbols"], ("SPY",))
        self.assertEqual(len(result["meta"]["snapshot_missing_symbols"]), 11)
        self.assertTrue(result["meta"]["realtime_error"])
        self.assertFalse(result["meta"]["history_quota_consumed"])

    def test_realtime_false_never_calls_moomoo(self):
        load_meta = {
            "source": "Yahoo Finance", "fetched_at": "2026-08-08T00:00:00+00:00",
            "errors": {},
        }
        with patch.object(mi, "_load_history_bundle",
                          return_value=(catalog_histories(), load_meta)), \
                patch.object(mi.moomoo_client, "snapshot") as snapshot:
            result = mi.get_us_market_overview(include_realtime=False)
        snapshot.assert_not_called()
        self.assertFalse(result["meta"]["realtime_requested"])
        self.assertEqual(result["meta"]["snapshot_status"], "not_requested")
        self.assertEqual(result["meta"]["snapshot_missing_symbols"], ())

    def test_all_requested_snapshots_are_marked_available(self):
        symbols = (mi.MARKET_BENCHMARK,) + tuple(
            item["etf"] for item in mi.SECTOR_CATALOG)
        snapshots = {symbol: {"price": 100.0} for symbol in symbols}
        meta = mi._snapshot_delivery_meta(
            symbols, snapshots, requested=True, load_error=None)
        self.assertEqual(meta["snapshot_status"], "available")
        self.assertEqual(meta["realtime_received"], len(symbols))
        self.assertEqual(meta["snapshot_missing_symbols"], ())
        self.assertIsNone(meta["realtime_error"])

    def test_empty_snapshot_is_reported_as_optional_source_unavailable(self):
        load_meta = {
            "source": "Yahoo Finance", "fetched_at": "2026-08-08T00:00:00+00:00",
            "errors": {},
        }
        with patch.object(mi, "_load_history_bundle",
                          return_value=(catalog_histories(), load_meta)), \
                patch.object(mi.moomoo_client, "snapshot", return_value={}):
            result = mi.get_us_market_overview(include_realtime=True)
        self.assertEqual(result["meta"]["snapshot_status"], "unavailable")
        self.assertEqual(result["meta"]["realtime_received"], 0)
        self.assertEqual(len(result["meta"]["snapshot_missing_symbols"]), 12)
        self.assertTrue(result["meta"]["realtime_error"])
        self.assertEqual(result["meta"]["history_source"], "Yahoo Finance")

    def test_module_contains_no_history_quota_or_trading_api_call(self):
        source = inspect.getsource(mi)
        self.assertNotIn("fetch_chart_history(", source)
        self.assertNotIn("request_history_kline(", source)
        self.assertNotIn("place_order(", source)
        self.assertNotIn("OpenSecTradeContext", source)

    def test_combined_api_only_loads_explicit_candidate_sectors(self):
        overview = {"market": {}, "sectors": [], "breadth": {}, "meta": {}}
        candidate = {"sector": {"key": "TECHNOLOGY"}}
        with patch.object(mi, "get_us_market_overview", return_value=overview), \
                patch.object(mi, "get_sector_candidates",
                             return_value=candidate) as get_candidates:
            result = mi.get_market_intelligence(
                candidate_sectors=("XLK", "TECHNOLOGY"), include_realtime=False)
        get_candidates.assert_called_once_with(
            "TECHNOLOGY", top_n=3, include_realtime=False, period="1y")
        self.assertEqual(set(result["candidate_sectors"]), {"TECHNOLOGY"})

    def test_combined_api_accepts_one_sector_string_without_character_iteration(self):
        overview = {"market": {}, "sectors": [], "breadth": {}, "meta": {}}
        with patch.object(mi, "get_us_market_overview", return_value=overview), \
                patch.object(mi, "get_sector_candidates",
                             return_value={"sector": {}}) as get_candidates:
            mi.get_market_intelligence(candidate_sectors="XLK")
        get_candidates.assert_called_once()

    def test_invalid_combined_request_fails_before_any_loader(self):
        with patch.object(mi, "get_us_market_overview") as overview:
            with self.assertRaises(ValueError):
                mi.get_market_intelligence(candidate_sectors=("UNKNOWN",))
            with self.assertRaises(ValueError):
                mi.get_market_intelligence(candidate_sectors=("XLK",), top_n=0)
        overview.assert_not_called()


if __name__ == "__main__":
    unittest.main()
