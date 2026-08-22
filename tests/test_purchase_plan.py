import copy
from datetime import datetime, timezone
from pathlib import Path
import unittest

from lib import trade_summary


NOW = datetime(2026, 8, 10, 14, 0, 0, tzinfo=timezone.utc)


def purchase_input(verdict="BUY"):
    return {
        "current_price": 100.0,
        "current_price_source": "moomoo OpenAPI",
        "snapshot": {
            "price": 100.0,
            "bid": 99.95,
            "ask": 100.0,
            "update_time": "2026-08-10T13:59:40+00:00",
            "suspension": False,
            "source": "moomoo OpenAPI",
        },
        "rule_evaluation": {
            "verdict": verdict,
            "summary": "既存の日足ルールによる判定",
            "position_mode": "entry",
            "regime": "UPTREND",
            "risk_plan": {
                "entry": 100.0,
                "stop": 95.0,
                "target": 110.0,
                "rr": 2.0,
                "valid": True,
                "stop_source": "既存の支持帯",
                "target_source": "既存の抵抗帯",
            },
            "buy": {
                "enabled": True,
                "valid": True,
                "available": True,
                "passed": verdict != "NEUTRAL",
                "checks": [{
                    "metric": "risk_reward",
                    "op": ">=",
                    "threshold": 1.5,
                    "required": True,
                    "status": "passed",
                }],
            },
            "gates": [
                {"key": "history", "label": "履歴", "passed": True,
                 "required": True, "reason": "確認済み"},
                {"key": "spread", "label": "価格差", "passed": True,
                 "required": True, "reason": "上限以内"},
            ],
        },
        "levels": [
            {"type": "サポート", "price": 96.0, "zone_low": 95.0,
             "zone_high": 97.0, "strength": 4},
            {"type": "抵抗線", "price": 110.0, "zone_low": 109.0,
             "zone_high": 111.0, "strength": 4},
        ],
        "trend": {
            "direction": "up", "label": "上昇", "strength": 75,
            "data_quality": "realtime", "source": "moomoo OpenAPI",
        },
        "session": {
            "current_session": {
                "session": "regular", "tradable": True,
                "reason": "現在は取引可能です",
                "as_of": NOW.isoformat(),
            },
        },
        "event": {
            "as_of": "2026-08-10",
            "events": [{
                "status": "UPCOMING", "name": "雇用統計",
                "event_date": "2026-08-15", "impact_level": "MEDIUM",
            }],
        },
    }


def reason_codes(plan):
    return [row["code"] for row in plan["wait_reasons"]]


class PositionSizeTests(unittest.TestCase):
    def test_integer_shares_respect_loss_and_budget(self):
        result = trade_summary.calculate_position_size(101, 95, 120, 1_010)
        self.assertTrue(result["available"])
        self.assertEqual(result["shares_by_loss"], 20)
        self.assertEqual(result["shares_by_budget"], 10)
        self.assertEqual(result["shares"], 10)
        self.assertEqual(result["estimated_cost"], 1_010)
        self.assertEqual(result["estimated_planned_loss"], 60)

    def test_one_share_over_loss_limit_returns_zero(self):
        result = trade_summary.calculate_position_size(101, 95, 5.99)
        self.assertTrue(result["available"])
        self.assertEqual(result["shares"], 0)
        self.assertFalse(result["can_buy_one_share"])

    def test_invalid_and_boolean_values_are_not_guessed(self):
        for values in ((100, 100, 10), (True, 95, 10), (100, 95, None)):
            with self.subTest(values=values):
                result = trade_summary.calculate_position_size(*values)
                self.assertFalse(result["available"])
                self.assertEqual(result["shares"], 0)


class PurchaseReadinessTests(unittest.TestCase):
    def test_ready_separates_daily_buy_from_execution_readiness(self):
        result = trade_summary.build_purchase_plan(
            purchase_input(), max_loss=120, max_investment=1_010, now=NOW,
        )
        self.assertEqual(result["authoritative_verdict"], "BUY")
        self.assertTrue(result["daily_signal"]["is_buy_candidate"])
        self.assertTrue(result["daily_signal"]["unchanged_by_execution_checks"])
        self.assertEqual(result["status"], "READY")
        self.assertTrue(result["actionable"])
        self.assertEqual(result["execution_readiness"]["status"], "READY")
        self.assertIsNone(result["buy_zone"]["low"])
        self.assertAlmostEqual(result["buy_zone"]["reference_price"], 100.0)
        self.assertAlmostEqual(result["buy_zone"]["high"], 101.0)
        self.assertAlmostEqual(result["chase_warning"]["current_rr"], 2.0)
        self.assertEqual(result["position_size"]["planned_purchase_price"], 101.0)
        self.assertEqual(result["position_size"]["target_price"], 110.0)
        self.assertEqual(result["position_size"]["estimated_target_profit"], 90.0)
        self.assertIn("通常時の想定損失を超える", result["disclaimer"])
        self.assertIn("gap_slippage_risk",
                      [row["code"] for row in result["cautions"]])

    def test_neutral_and_wait_never_become_ready(self):
        neutral = trade_summary.build_purchase_plan(
            purchase_input("NEUTRAL"), now=NOW)
        waiting = trade_summary.build_purchase_plan(
            purchase_input("WAIT"), now=NOW)
        self.assertEqual(neutral["status"], "NOT_CANDIDATE")
        self.assertEqual(neutral["authoritative_verdict"], "NEUTRAL")
        self.assertFalse(neutral["actionable"])
        self.assertEqual(waiting["status"], "WAIT")
        self.assertEqual(waiting["authoritative_verdict"], "WAIT")
        self.assertFalse(waiting["actionable"])

    def test_existing_required_gate_blocks_even_if_verdict_says_buy(self):
        payload = purchase_input()
        payload["rule_evaluation"]["gates"][0].update({
            "passed": False, "reason": "履歴不足",
        })
        result = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertEqual(result["authoritative_verdict"], "BUY")
        self.assertEqual(result["status"], "WAIT")
        self.assertIn("gate_history", reason_codes(result))

    def test_daily_buy_condition_is_separate_from_external_gate_wait(self):
        payload = purchase_input("WAIT")
        payload["rule_evaluation"]["gates"][0].update({
            "passed": False, "reason": "現在の履歴鮮度を再確認",
        })
        result = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertTrue(result["daily_signal"]["is_buy_candidate"])
        self.assertEqual(result["daily_signal"]["verdict"], "BUY")
        self.assertEqual(result["authoritative_verdict"], "WAIT")
        self.assertFalse(result["actionable"])

    def test_premarket_is_actionable_when_explicitly_tradable(self):
        payload = purchase_input()
        payload["session"]["current_session"]["session"] = "premarket"
        ready = trade_summary.build_purchase_plan(payload, now=NOW)
        payload["session"]["current_session"]["tradable"] = False
        blocked = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertEqual(ready["status"], "READY")
        self.assertEqual(blocked["status"], "WAIT")
        self.assertIn("session_not_tradable", reason_codes(blocked))

    def test_price_above_rr_limit_is_a_chase_wait(self):
        payload = purchase_input()
        payload["snapshot"].update({"price": 102.0, "bid": 101.95, "ask": 102.0})
        result = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertEqual(result["status"], "WAIT")
        self.assertEqual(result["authoritative_verdict"], "BUY")
        self.assertEqual(result["chase_warning"]["status"], "above_limit")
        self.assertIn("chasing_price", reason_codes(result))

    def test_price_below_reference_is_not_treated_as_below_a_buy_zone(self):
        payload = purchase_input()
        payload["snapshot"].update({"price": 99.0, "bid": 98.95, "ask": 99.0})
        result = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertEqual(result["status"], "READY")
        self.assertIsNone(result["buy_zone"]["low"])
        self.assertEqual(result["buy_zone"]["reference_price"], 100.0)
        caution = next(item for item in result["cautions"]
                       if item["code"] == "below_reference")
        self.assertIn("下落", caution["detail_ja"])

    def test_stop_or_target_prices_invalidate_current_execution(self):
        for ask, expected in ((95.0, "stop_reached"), (110.0, "target_reached")):
            payload = purchase_input()
            payload["snapshot"].update({"price": ask, "bid": ask - 0.05,
                                         "ask": ask})
            with self.subTest(ask=ask):
                result = trade_summary.build_purchase_plan(payload, now=NOW)
                self.assertEqual(result["status"], "WAIT")
                self.assertIn(expected, reason_codes(result))


class QuoteAndEventSafetyTests(unittest.TestCase):
    def test_missing_ask_bid_or_update_time_is_not_actionable(self):
        cases = (("ask", "ask_unavailable"),
                 ("bid", "spread_unavailable"),
                 ("update_time", "quote_time_unavailable"))
        for key, expected in cases:
            payload = purchase_input()
            payload["snapshot"].pop(key)
            with self.subTest(key=key):
                result = trade_summary.build_purchase_plan(payload, now=NOW)
                self.assertFalse(result["actionable"])
                self.assertIn(expected, reason_codes(result))

    def test_stale_quote_and_suspension_block_current_purchase(self):
        stale_payload = purchase_input()
        stale_payload["snapshot"]["update_time"] = "2026-08-10T13:58:00+00:00"
        stale = trade_summary.build_purchase_plan(stale_payload, now=NOW)
        suspended_payload = purchase_input()
        suspended_payload["snapshot"]["suspension"] = "SUSPENDED"
        suspended = trade_summary.build_purchase_plan(suspended_payload, now=NOW)
        self.assertEqual(stale["status"], "WAIT")
        self.assertIn("quote_stale", reason_codes(stale))
        self.assertEqual(suspended["status"], "WAIT")
        self.assertIn("trading_suspended", reason_codes(suspended))

        unknown_payload = purchase_input()
        unknown_payload["snapshot"].pop("suspension")
        unknown = trade_summary.build_purchase_plan(unknown_payload, now=NOW)
        self.assertFalse(unknown["actionable"])
        self.assertIn("suspension_unavailable", reason_codes(unknown))

    def test_spread_gate_or_explicit_limit_is_required(self):
        payload = purchase_input()
        payload["rule_evaluation"]["gates"] = [
            gate for gate in payload["rule_evaluation"]["gates"]
            if gate["key"] != "spread"
        ]
        unknown = trade_summary.build_purchase_plan(payload, now=NOW)
        allowed = trade_summary.build_purchase_plan(
            payload, max_spread_pct=0.10, now=NOW)
        payload["snapshot"].update({"bid": 99.0, "ask": 100.0})
        wide = trade_summary.build_purchase_plan(
            payload, max_spread_pct=0.10, now=NOW)
        self.assertFalse(unknown["actionable"])
        self.assertIn("spread_limit_unavailable", reason_codes(unknown))
        self.assertEqual(allowed["status"], "READY")
        self.assertEqual(wide["status"], "WAIT")
        self.assertIn("spread_too_wide", reason_codes(wide))

    def test_missing_or_imminent_high_event_is_not_treated_as_no_event(self):
        missing_payload = purchase_input()
        missing_payload.pop("event")
        missing = trade_summary.build_purchase_plan(missing_payload, now=NOW)
        high_payload = purchase_input()
        high_payload["event"]["events"][0].update({
            "event_date": "2026-08-12", "impact_level": "HIGH",
        })
        high = trade_summary.build_purchase_plan(high_payload, now=NOW)
        self.assertFalse(missing["actionable"])
        self.assertIn("event_unavailable", reason_codes(missing))
        self.assertFalse(high["actionable"])
        self.assertIn("high_impact_event", reason_codes(high))
        self.assertEqual(high["authoritative_verdict"], "BUY")

    def test_all_upcoming_events_are_scanned_for_imminent_high_impact(self):
        payload = purchase_input()
        payload["event"]["events"] = [
            {"status": "UPCOMING", "name": "中影響イベント",
             "event_date": "2026-08-11", "impact_level": "MEDIUM"},
            {"status": "UPCOMING", "name": "高影響イベント",
             "event_date": "2026-08-12", "impact_level": "HIGH"},
        ]
        result = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertFalse(result["actionable"])
        self.assertIn("high_impact_event", reason_codes(result))

    def test_partial_event_report_is_not_treated_as_fully_checked(self):
        payload = purchase_input()
        payload["event"].update({
            "status": "partial", "warnings": ["ニュース取得失敗"],
        })
        result = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertFalse(result["actionable"])
        self.assertIn("event_partial", reason_codes(result))

    def test_rr_threshold_must_be_a_required_existing_check(self):
        payload = purchase_input()
        payload["rule_evaluation"]["buy"]["checks"][0]["required"] = False
        result = trade_summary.build_purchase_plan(payload, now=NOW)
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertFalse(result["buy_zone"]["available"])
        self.assertIn("price_zone_unavailable", reason_codes(result))


class PurchasePlanSafetyTests(unittest.TestCase):
    def test_zero_share_plan_blocks_purchase(self):
        result = trade_summary.build_purchase_plan(
            purchase_input(), max_loss=5.99, now=NOW)
        self.assertEqual(result["position_size"]["shares"], 0)
        self.assertEqual(result["status"], "WAIT")
        self.assertIn("position_size_zero", reason_codes(result))

    def test_input_is_unchanged_and_result_is_read_only(self):
        payload = purchase_input()
        original = copy.deepcopy(payload)
        result = trade_summary.build_purchase_plan(payload, max_loss=120, now=NOW)
        self.assertEqual(payload, original)
        self.assertTrue(result["read_only"])
        self.assertFalse(result["places_orders"])
        self.assertFalse(result["uses_network"])
        self.assertFalse(result["uses_moomoo_history_quota"])
        self.assertEqual(result["score_effect"], 0)

    def test_stock_page_uses_existing_context_and_never_places_an_order(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "views" / "stock_analysis.py").read_text(encoding="utf-8")
        purchase_block = source.split(
            "# 購入プランは取得済みの確定日足", 1)[1].split(
                "# ---------------------------------------------------------------- 今日の判断", 1)[0]
        self.assertIn("trade_summary.build_purchase_plan", purchase_block)
        self.assertIn("**summary_common", purchase_block)
        self.assertIn('"rule_evaluation": entry_evaluation', purchase_block)
        self.assertNotIn("fetch_chart_history", purchase_block)
        self.assertNotIn("place_order", purchase_block)
        self.assertIn("上限株数（推奨ではない）", purchase_block)


if __name__ == "__main__":
    unittest.main()
