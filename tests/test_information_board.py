import copy
import inspect
import json
import unittest

from lib import information_board


def complete_input():
    return {
        "ticker": " aapl ",
        "as_of": "2026-08-10T12:00:00+00:00",
        "snapshot": {
            "price": 200.0,
            "change_pct": 3.25,
            "bid": 199.90,
            "ask": 200.10,
            "source": "moomoo OpenAPI",
            "update_time": "2026-08-10T11:59:00+00:00",
        },
        "entry_evaluation": {
            "verdict": "BUY",
            "summary": "必須条件と買いスコアが成立しています",
            "buy": {"score": 75, "threshold": 70},
            "evaluated_at": "2026-08-10T11:58:00+00:00",
        },
        "holding_evaluation": {
            "verdict": "RISK_EXIT",
            "summary": "リスク退出条件が成立しています",
            "risk_exit": {"score": 65, "threshold": 60},
            "take_profit": {"score": 20, "threshold": 55},
            "evaluated_at": "2026-08-10T11:58:00+00:00",
        },
        "levels": [
            {
                "type": "サポート", "price": 190.0,
                "zone_low": 188.0, "zone_high": 192.0,
                "strength": 4, "basis": "スイング安値",
            },
            {
                "type": "resistance", "price": 215.0,
                "zone_low": 214.0, "zone_high": 216.0,
                "strength": 3, "basis": "出来高集中帯",
            },
        ],
        "alert_checked_at": "2026-08-10T11:59:30+00:00",
        "checked_alerts": [
            (
                {"id": "p1", "ticker": "AAPL", "kind": "price_above",
                 "value": 199.0, "enabled": True},
                {"triggered": True, "actual": "$200.00", "reason": ""},
            ),
            {
                "alert": {"id": "r1", "ticker": "AAPL", "kind": "rsi_above",
                          "value": 70, "enabled": True},
                "result": {"triggered": False, "actual": "取得できず",
                           "reason": "RSIを計算できませんでした"},
            },
            {
                "id": "p2", "ticker": "AAPL", "kind": "price_below",
                "value": 180, "triggered": False, "actual": "$200.00",
            },
        ],
        "analyst": {
            "targets": {"mean": 230, "median": 225, "high": 260, "low": 190},
            "ratings": {"strongBuy": 8, "buy": 14, "hold": 7,
                        "sell": 1, "strongSell": 0},
            "earnings_date": "2026-10-30",
            "eps_estimate": 1.75,
            "changes": [{
                "date": "2026-08-09", "firm": "Example Research",
                "grade": "Buy", "action": "up", "target": 240,
            }],
        },
        "event_report": {
            "status": "ok",
            "events": [{
                "event_id": "cpi-2026-08",
                "kind": "cpi",
                "display_name_ja": "米国消費者物価指数（CPI）",
                "event_date": "2026-08-12",
                "status": "UPCOMING",
                "status_label_ja": "今後の予定",
                "session_label_ja": "プレマーケット",
                "impact_stars": 4,
                "impact_label_ja": "影響大",
                "directional_bias_label_ja": "過去は上下両方向",
                "source": "米国労働統計局",
                "url": "https://www.bls.gov/cpi/",
            }],
        },
        "news": [
            {
                "title": "Apple announces product update",
                "summary": "Company announcement summary.",
                "pub_date": "2026-08-10 11:30",
                "provider": "Example News",
                "url": "https://news.example.com/apple-update",
            },
            # 同じURLは重複として1件へまとめる。
            {
                "title": "Apple announces product update (duplicate feed title)",
                "summary": "Longer duplicate summary from a second feed.",
                "pub_date": "2026-08-10 11:31",
                "provider": "Example News 2",
                "url": "https://news.example.com/apple-update",
            },
            {
                "title": "Unsafe link still has a readable headline",
                "pub_date": "2026-08-10 10:00",
                "provider": "Unknown",
                "url": "javascript:alert(1)",
            },
        ],
    }


class NormalizationTests(unittest.TestCase):
    def test_normalized_item_has_fixed_schema_and_japanese_labels(self):
        item = information_board.normalize_item({
            "ticker": "mu",
            "category": "news",
            "kind": "news",
            "importance": "high",
            "title_ja": "半導体ニュース",
            "summary_ja": "供給動向",
            "occurred_at": "2026-08-10T10:00:00Z",
            "source": "Example",
            "url": "https://example.com/news/1",
            "tags": ["半導体", "半導体", "市況"],
            "data": {"value": 1},
        })
        self.assertEqual(tuple(item), information_board.ITEM_FIELDS)
        self.assertEqual(item["ticker"], "MU")
        self.assertEqual(item["category_label_ja"], "ニュース")
        self.assertEqual(item["importance_label_ja"], "重要")
        self.assertEqual(item["importance_rank"], 4)
        self.assertEqual(item["tags"], ["半導体", "市況"])
        self.assertEqual(len(item["id"]), 20)

    def test_id_is_deterministic_and_explicit_identity_controls_dedupe(self):
        base = {
            "category": "news", "kind": "news", "importance": "info",
            "title_ja": "見出しA", "identity_key": "provider:123",
        }
        changed = {**base, "title_ja": "見出しB", "summary_ja": "より詳しい本文"}
        first = information_board.normalize_item(base)
        second = information_board.normalize_item(changed)
        self.assertEqual(first["id"], second["id"])

    def test_missing_title_or_unknown_category_is_rejected(self):
        self.assertIsNone(information_board.normalize_item({
            "category": "news", "kind": "news",
        }))
        self.assertIsNone(information_board.normalize_item({
            "category": "posting", "kind": "post", "title_ja": "投稿",
        }))

    def test_safe_url_accepts_public_http_and_rejects_unsafe_targets(self):
        self.assertEqual(
            information_board.safe_url("https://example.com/a?q=1"),
            "https://example.com/a?q=1",
        )
        for value in (
            "javascript:alert(1)", "data:text/html,x", "file:///tmp/a",
            "//example.com/path", "https://user:pass@example.com/",
            "http://127.0.0.1/a", "http://2130706433/a",
            "http://0x7f000001/a", "http://localhost/a", "http://169.254.1.1/a",
            "https://example.com/a b",
        ):
            with self.subTest(value=value):
                self.assertIsNone(information_board.safe_url(value))


class BoardBuildTests(unittest.TestCase):
    def test_complete_input_aggregates_every_requested_category(self):
        result = information_board.build_information_board(complete_input())
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(
            {item["category"] for item in result["items"]},
            set(information_board.CATEGORY_LABELS),
        )
        self.assertGreaterEqual(result["counts"]["total"], 13)
        self.assertEqual(result["counts"]["duplicates_removed"], 1)
        self.assertEqual(len({item["id"] for item in result["items"]}),
                         len(result["items"]))

    def test_duplicate_selection_and_ids_do_not_depend_on_input_order(self):
        payload = complete_input()
        forward = information_board.build_information_board(payload)
        reversed_payload = copy.deepcopy(payload)
        reversed_payload["news"].reverse()
        reversed_result = information_board.build_information_board(reversed_payload)
        self.assertEqual(forward["items"], reversed_result["items"])

    def test_metadata_explicitly_disables_mutation_network_orders_and_quota(self):
        metadata = information_board.build_information_board({})["metadata"]
        self.assertTrue(metadata["read_only"])
        self.assertFalse(metadata["posting_enabled"])
        self.assertFalse(metadata["stores_data"])
        self.assertFalse(metadata["uses_network"])
        self.assertFalse(metadata["places_orders"])
        self.assertFalse(metadata["uses_moomoo_history"])
        self.assertFalse(metadata["uses_moomoo_history_quota"])
        self.assertFalse(metadata["moomoo_history_quota_used"])
        self.assertEqual(metadata["moomoo_history_requests"], 0)
        self.assertEqual(metadata["score_effect"], 0)

    def test_rule_results_are_preserved_and_not_recalculated(self):
        payload = complete_input()
        payload["entry_evaluation"]["buy"] = {"score": 0, "threshold": 70}
        result = information_board.build_information_board(payload)
        entry = next(item for item in result["items"]
                     if item["kind"] == "entry_verdict")
        holding = next(item for item in result["items"]
                       if item["kind"] == "holding_verdict")
        self.assertEqual(entry["data"]["verdict"], "BUY")
        self.assertEqual(entry["data"]["buy_score"], 0.0)
        self.assertEqual(holding["data"]["verdict"], "RISK_EXIT")
        self.assertEqual(holding["importance"], "critical")

    def test_invalid_buy_risk_plan_is_marked_for_visual_execution_hold(self):
        payload = complete_input()
        payload["entry_evaluation"]["risk_plan"] = {"valid": False}
        result = information_board.build_information_board(payload)
        entry = next(item for item in result["items"]
                     if item["kind"] == "entry_verdict")
        self.assertEqual(entry["data"]["verdict"], "BUY")
        self.assertTrue(entry["data"]["blocked"])

    def test_explicit_visual_hold_is_preserved_without_changing_verdict(self):
        payload = complete_input()
        payload["entry_evaluation"]["visual_blocked"] = True
        result = information_board.build_information_board(payload)
        entry = next(item for item in result["items"]
                     if item["kind"] == "entry_verdict")
        self.assertEqual(entry["data"]["verdict"], "BUY")
        self.assertTrue(entry["data"]["blocked"])

    def test_mode_verdict_mismatch_is_omitted_with_warning(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "entry_evaluation": {"verdict": "RISK_EXIT"},
            "holding_evaluation": {"verdict": "BUY"},
        })
        self.assertEqual(result["items"], [])
        self.assertEqual(result["counts"]["warnings"], 2)
        self.assertTrue(all(
            "組み合わせが不正" in warning for warning in result["warnings"]
        ))

    def test_unavailable_alert_is_not_mislabeled_as_not_triggered(self):
        result = information_board.build_information_board(complete_input())
        alerts = [item for item in result["items"] if item["category"] == "alert"]
        unavailable = next(item for item in alerts if item["data"]["alert_id"] == "r1")
        not_triggered = next(item for item in alerts if item["data"]["alert_id"] == "p2")
        self.assertIn("判定不能", unavailable["title_ja"])
        self.assertIsNone(unavailable["data"]["triggered"])
        self.assertIn("未成立", not_triggered["title_ja"])
        self.assertFalse(not_triggered["data"]["triggered"])

    def test_description_only_checked_alert_remains_displayable_without_guessing_kind(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "checked_alerts": [{
                "description": "AAPL: 価格が上回る $200.00",
                "triggered": True,
                "actual": "$201.00",
                "reason": "",
            }],
        })
        item = result["items"][0]
        self.assertEqual(item["category"], "alert")
        self.assertEqual(item["data"]["alert_kind"], "checked_alert")
        self.assertIn("AAPL: 価格が上回る", item["title_ja"])

    def test_rule_alert_actual_uses_beginner_label_not_internal_code(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "checked_alerts": [{
                "alert": {
                    "id": "risk", "ticker": "AAPL",
                    "kind": "rule_risk_exit", "enabled": True,
                },
                "result": {
                    "triggered": True, "actual": "RISK_EXIT", "reason": "",
                },
            }],
        })
        item = result["items"][0]
        self.assertIn("保有株を売る候補（損失を抑える）", item["summary_ja"])
        self.assertNotIn("RISK_EXIT", item["summary_ja"])
        self.assertEqual(
            item["data"]["actual"], "保有株を売る候補（損失を抑える）")

    def test_blocked_buy_alert_keeps_wait_message(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "checked_alerts": [{
                "alert": {
                    "id": "buy", "ticker": "AAPL",
                    "kind": "rule_buy", "enabled": True,
                },
                "result": {
                    "triggered": True, "actual": "BUY_BLOCKED", "reason": "",
                },
            }],
        })
        item = result["items"][0]
        self.assertIn("買い条件あり・今は待つ", item["summary_ja"])
        self.assertNotIn("BUY_BLOCKED", item["summary_ja"])

    def test_caller_warnings_are_preserved_and_deduplicated(self):
        result = information_board.build_information_board({
            "warnings": ["ニュース取得失敗", "ニュース取得失敗"],
        })
        self.assertEqual(result["warnings"], ["ニュース取得失敗"])
        self.assertEqual(result["counts"]["warnings"], 1)

    def test_support_resistance_distance_is_derived_only_from_supplied_price(self):
        result = information_board.build_information_board(complete_input())
        support = next(item for item in result["items"]
                       if item["category"] == "level" and item["kind"] == "support")
        self.assertAlmostEqual(support["data"]["distance_from_current_pct"], -5.0)

        payload = complete_input()
        payload.pop("snapshot")
        without_price = information_board.build_information_board(payload)
        support = next(item for item in without_price["items"] if item["kind"] == "support")
        self.assertIsNone(support["data"]["distance_from_current_pct"])
        self.assertNotIn("現在値から", support["summary_ja"])

    def test_snapshot_change_can_be_calculated_from_supplied_previous_close(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "snapshot": {"price": 102, "previous_close": 100},
        })
        snapshot = result["items"][0]
        self.assertAlmostEqual(snapshot["data"]["change_pct"], 2.0)
        self.assertTrue(snapshot["data"]["change_pct_calculated_from_previous_close"])
        self.assertIn("+2.00%", snapshot["summary_ja"])

    def test_snapshot_names_the_selected_extended_session(self):
        result = information_board.build_information_board({
            "ticker": "SKHY",
            "snapshot": {
                "price": 163.77, "previous_close": 163.09,
                "price_session": "afterhours", "price_field": "after_price",
                "price_timestamp_verified": False,
                "source": "moomoo OpenAPI snapshot",
            },
        })
        snapshot = result["items"][0]
        self.assertEqual(snapshot["title_ja"], "アフター価格のスナップショット")
        self.assertIn("アフター価格 $163.77", snapshot["summary_ja"])
        self.assertEqual(snapshot["data"]["price_session"], "afterhours")
        self.assertFalse(snapshot["data"]["price_timestamp_verified"])

    def test_malformed_zone_is_omitted_not_silently_reversed(self):
        result = information_board.build_information_board({
            "ticker": "MU",
            "levels": [{
                "type": "support", "price": 100,
                "zone_low": 105, "zone_high": 95, "strength": 9,
            }],
        })
        item = result["items"][0]
        self.assertIsNone(item["data"]["zone_low"])
        self.assertIsNone(item["data"]["zone_high"])
        self.assertIsNone(item["data"]["strength"])
        self.assertGreaterEqual(result["counts"]["warnings"], 2)

    def test_unsafe_news_url_is_removed_without_dropping_valid_headline(self):
        result = information_board.build_information_board(complete_input())
        item = next(item for item in result["items"]
                    if item["title_ja"].startswith("Unsafe link"))
        self.assertIsNone(item["url"])
        self.assertTrue(any("安全でないURL" in warning for warning in result["warnings"]))

    def test_event_and_news_with_same_source_url_are_not_double_counted(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "event_report": {"events": [{
                "event_id": "same-story",
                "display_name_ja": "業績見通しの更新",
                "event_date": "2026-08-10",
                "status": "RECENT",
                "impact_stars": 4,
                "url": "https://example.com/story/1",
            }]},
            "news": [{
                "title": "Guidance update",
                "pub_date": "2026-08-10",
                "url": "https://example.com/story/1",
            }],
        })
        self.assertEqual(result["counts"]["total"], 1)
        self.assertEqual(result["counts"]["duplicates_removed"], 1)
        self.assertEqual(result["items"][0]["category"], "event")
        self.assertIn("Guidance update", result["items"][0]["summary_ja"])

    def test_analyst_and_event_earnings_are_merged_without_losing_details(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "analyst": {
                "earnings_date": "2026-10-30",
                "eps_estimate": 1.75,
            },
            "event_report": {"events": [{
                "event_id": "earnings-2026-10-30",
                "kind": "earnings",
                "display_name_ja": "決算発表",
                "event_date": "2026-10-30",
                "impact_stars": 5,
            }]},
        })
        earnings = [item for item in result["items"]
                    if item["kind"] == "earnings"]
        self.assertEqual(len(earnings), 1)
        self.assertEqual(earnings[0]["data"]["eps_estimate"], 1.75)
        self.assertEqual(earnings[0]["data"]["impact_stars"], 5.0)

    def test_zero_star_event_is_distinct_from_low_impact(self):
        result = information_board.build_information_board({
            "ticker": "AAPL",
            "event_report": {"status": "ok", "events": [{
                "event_id": "unknown-impact",
                "kind": "company_news",
                "display_name_ja": "影響を判定できないイベント",
                "event_date": "2026-08-12",
                "impact_stars": 0,
            }]},
        })
        event = result["items"][0]
        self.assertEqual(event["importance"], "info")
        self.assertIn("判定材料なし", event["summary_ja"])

    def test_partial_or_malformed_inputs_do_not_fabricate_items(self):
        result = information_board.build_information_board({
            "ticker": "bad ticker!",
            "snapshot": {"price": "not-a-number"},
            "entry_evaluation": {"verdict": "STRONG_BUY"},
            "holding_evaluation": ["HOLD"],
            "levels": [None, {"type": "support"}],
            "checked_alerts": [{"kind": "price_above"}],
            "analyst": {"earnings_date": "someday", "targets": {"mean": "?"}},
            "event_report": {"status": "unavailable", "events": [None]},
            "news": [{"summary": "見出しなし"}],
        })
        self.assertIsNone(result["ticker"])
        self.assertEqual(result["items"], [])
        self.assertGreaterEqual(result["counts"]["warnings"], 8)

    def test_non_mapping_input_returns_safe_empty_board(self):
        result = information_board.build_information_board(["not", "a", "mapping"])
        self.assertEqual(result["items"], [])
        self.assertTrue(result["metadata"]["read_only"])
        self.assertEqual(result["counts"]["warnings"], 1)

    def test_result_is_json_serializable(self):
        json.dumps(information_board.build_information_board(complete_input()),
                   ensure_ascii=False)

    def test_module_has_no_streamlit_network_storage_or_order_dependency(self):
        source = inspect.getsource(information_board)
        for forbidden in (
            "import streamlit", "import requests", "urllib.request", "sqlite3",
            "Path(", "open(", "place_order", "fetch_chart_history",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


class FilterAndSortTests(unittest.TestCase):
    def setUp(self):
        self.items = information_board.build_information_board(complete_input())["items"]

    def test_filter_by_category_importance_query_and_minimum(self):
        news = information_board.filter_board_items(self.items, categories="news")
        self.assertTrue(news)
        self.assertTrue(all(item["category"] == "news" for item in news))

        urgent = information_board.filter_items(
            self.items, importance={"critical", "high"})
        self.assertTrue(urgent)
        self.assertTrue(all(item["importance"] in {"critical", "high"}
                            for item in urgent))

        minimum = information_board.filter_board_items(
            self.items, minimum_importance="high")
        self.assertTrue(all(item["importance_rank"] >= 4 for item in minimum))

        cpi = information_board.filter_board_items(self.items, query="消費者物価")
        self.assertEqual(len(cpi), 1)
        self.assertEqual(cpi[0]["category"], "event")

    def test_triggered_only_excludes_unavailable_and_false_alerts(self):
        triggered = information_board.filter_board_items(self.items, triggered_only=True)
        self.assertEqual(len(triggered), 1)
        self.assertTrue(triggered[0]["data"]["triggered"])

    def test_sort_modes_are_deterministic_and_do_not_mutate_input(self):
        original = copy.deepcopy(self.items)
        important = information_board.sort_board_items(self.items, sort_by="importance")
        newest = information_board.sort_items(self.items, sort_by="newest")
        oldest = information_board.sort_board_items(self.items, sort_by="oldest")
        category = information_board.sort_board_items(self.items, sort_by="category")

        self.assertEqual(important[0]["importance"], "critical")
        dated_newest = [item for item in newest if item["occurred_at"]]
        dated_oldest = [item for item in oldest if item["occurred_at"]]
        self.assertGreaterEqual(dated_newest[0]["occurred_at"], dated_newest[-1]["occurred_at"])
        self.assertLessEqual(dated_oldest[0]["occurred_at"], dated_oldest[-1]["occurred_at"])
        categories = [list(information_board.CATEGORY_LABELS).index(item["category"])
                      for item in category]
        self.assertEqual(categories, sorted(categories))
        self.assertEqual(self.items, original)
        self.assertEqual(
            [item["id"] for item in important],
            [item["id"] for item in information_board.sort_items(self.items, sort_by="priority")],
        )

    def test_filter_returns_copies(self):
        filtered = information_board.filter_board_items(self.items, categories="news")
        filtered[0]["title_ja"] = "変更"
        self.assertNotEqual(filtered[0]["title_ja"],
                            next(item["title_ja"] for item in self.items
                                 if item["id"] == filtered[0]["id"]))


if __name__ == "__main__":
    unittest.main()
