import copy
import unittest

import numpy as np
import pandas as pd

from lib import trade_summary


def complete_input(verdict="BUY"):
    return {
        "current_price": 100.0,
        "current_price_source": "moomoo OpenAPI",
        "rule_evaluation": {
            "verdict": verdict,
            "summary": "既存ルールの判定理由",
            "position_mode": "holding" if verdict in {
                "RISK_EXIT", "TAKE_PROFIT", "HOLD",
            } else "entry",
            "regime": "UPTREND",
            "risk_plan": {
                "entry": 100.0, "stop": 95.0, "target": 110.0,
                "rr": 2.0, "stop_source": "支持帯（強度4）",
                "target_source": "抵抗帯（強度4）",
            },
        },
        "levels": [
            {"type": "サポート", "price": 96.0, "zone_low": 95.0,
             "zone_high": 97.0, "strength": 4, "basis": "スイング"},
            {"type": "抵抗線", "price": 110.0, "zone_low": 109.0,
             "zone_high": 111.0, "strength": 3, "basis": "出来高集中帯"},
        ],
        "trend": {
            "direction": "up", "label": "上昇", "strength": 75,
            "data_quality": "realtime", "source": "moomoo OpenAPI",
        },
        "session": {
            "current_session": {
                "session": "regular", "tradable": True,
                "reason": "現在はregularセッションです",
                "source": "market_state", "data_quality": {"score": 1.0},
            },
            "session_changes": {"sessions": {"regular": {
                "price": 100.0, "change_vs_previous_close_pct": 1.25,
            }}},
        },
        "event": {
            "as_of": "2026-08-10",
            "events": [{
                "status": "UPCOMING", "name": "米国CPI発表",
                "event_date": "2026-08-15", "session": "PRE_MARKET",
                "impact_level": "MEDIUM", "impact_score": 55,
                "directional_bias": "TWO_SIDED", "source": "BLS",
            }],
        },
    }


class VerdictProjectionTests(unittest.TestCase):
    def test_all_six_rule_verdicts_are_kept_distinct(self):
        expected_labels = {
            "BUY": "買い条件成立",
            "WAIT": "待機",
            "NEUTRAL": "中立",
            "RISK_EXIT": "リスク退出条件成立",
            "TAKE_PROFIT": "利確条件成立",
            "HOLD": "保有継続",
        }
        for code, label in expected_labels.items():
            with self.subTest(code=code):
                result = trade_summary.build_trade_summary(complete_input(code))
                self.assertEqual(result["verdict"]["code"], code)
                self.assertEqual(result["verdict"]["label_ja"], label)
                self.assertTrue(result["verdict"]["available"])

    def test_missing_or_unknown_verdict_waits_instead_of_guessing(self):
        missing = trade_summary.build_trade_summary({})
        unknown = trade_summary.build_trade_summary({
            "rule_evaluation": {"verdict": "STRONG_BUY"},
        })
        self.assertEqual(missing["verdict"]["code"], "WAIT")
        self.assertFalse(missing["verdict"]["available"])
        self.assertIn("データがない", missing["verdict"]["reason"])
        self.assertEqual(unknown["verdict"]["code"], "WAIT")
        self.assertIn("未対応", unknown["verdict"]["reason"])

    def test_nested_entry_and_holding_evaluations_are_selected_explicitly(self):
        payload = {
            "position_mode": "holding",
            "rule_evaluation": {
                "entry": {"verdict": "BUY", "position_mode": "entry"},
                "holding": {"verdict": "HOLD", "position_mode": "holding"},
            },
        }
        result = trade_summary.build_trade_summary(payload)
        self.assertEqual(result["verdict"]["code"], "HOLD")
        self.assertEqual(result["verdict"]["position_mode"], "holding")


class PricePlanAndLevelTests(unittest.TestCase):
    def test_valid_plan_is_normalized_and_rr_is_recomputed(self):
        payload = complete_input()
        payload["rule_evaluation"]["risk_plan"]["rr"] = 99.0
        result = trade_summary.build_trade_summary(payload)
        self.assertTrue(result["risk_plan_valid"])
        self.assertEqual(result["entry"]["price"], 100.0)
        self.assertEqual(result["stop"]["price"], 95.0)
        self.assertEqual(result["target"]["price"], 110.0)
        self.assertAlmostEqual(result["rr"]["value"], 2.0)
        self.assertEqual(result["rr"]["label_ja"], "1 : 2.00")
        self.assertIn("算出", result["rr"]["source"])

    def test_invalid_stop_and_target_are_not_exposed_as_usable(self):
        payload = complete_input()
        payload["rule_evaluation"]["risk_plan"].update({
            "stop": 101.0, "target": 99.0,
        })
        result = trade_summary.build_trade_summary(payload)
        self.assertFalse(result["risk_plan_valid"])
        self.assertFalse(result["stop"]["available"])
        self.assertFalse(result["target"]["available"])
        self.assertFalse(result["rr"]["available"])
        self.assertEqual(result["action_priorities"][0]["code"], "complete_risk_plan")
        self.assertTrue(result["action_priorities"][0]["blocking"])

    def test_existing_invalid_plan_flag_is_honored(self):
        payload = complete_input()
        payload["rule_evaluation"]["risk_plan"]["valid"] = False
        result = trade_summary.build_trade_summary(payload)
        self.assertFalse(result["risk_plan_valid"])
        self.assertTrue(any("無効と判定" in warning
                            for warning in result["data_quality"]["warnings"]))

    def test_nearest_support_and_resistance_use_zone_edges(self):
        result = trade_summary.build_trade_summary(complete_input())
        self.assertEqual(result["support"]["price"], 96.0)
        self.assertEqual(result["support"]["edge_price"], 97.0)
        self.assertAlmostEqual(result["support"]["distance_pct"], -3.0)
        self.assertEqual(result["support"]["stars"], "★★★★☆")
        self.assertEqual(result["resistance"]["edge_price"], 109.0)
        self.assertAlmostEqual(result["resistance"]["distance_pct"], 9.0)

    def test_stale_levels_on_wrong_side_are_ignored(self):
        payload = complete_input()
        payload["levels"] = [
            {"type": "サポート", "price": 120, "zone_low": 119, "zone_high": 121},
            {"type": "抵抗線", "price": 80, "zone_low": 79, "zone_high": 81},
        ]
        result = trade_summary.build_trade_summary(payload)
        self.assertFalse(result["support"]["available"])
        self.assertFalse(result["resistance"]["available"])

    def test_missing_current_price_does_not_choose_a_nearest_level(self):
        result = trade_summary.build_trade_summary({
            "levels": complete_input()["levels"],
            "rule_evaluation": {"verdict": "NEUTRAL"},
        })
        self.assertFalse(result["current_price"]["available"])
        self.assertFalse(result["support"]["available"])
        self.assertFalse(result["resistance"]["available"])

    def test_snapshot_precedes_history_and_history_is_close_only_fallback(self):
        history = pd.DataFrame({"Close": [90.0, 91.0]},
                               index=pd.date_range("2026-08-07", periods=2))
        live = trade_summary.build_trade_summary({
            "snapshot": {"price": 101.0, "source": "moomoo OpenAPI",
                         "update_time": "2026-08-10 10:00:00"},
            "history": history,
        })
        fallback = trade_summary.build_trade_summary({"history": history})
        self.assertEqual(live["current_price"]["value"], 101.0)
        self.assertEqual(live["current_price"]["quality"], "realtime")
        self.assertEqual(fallback["current_price"]["value"], 91.0)
        self.assertEqual(fallback["current_price"]["quality"], "close_only")
        self.assertTrue(any("リアルタイムではなく" in warning
                            for warning in fallback["data_quality"]["warnings"]))


class ContextProjectionTests(unittest.TestCase):
    def test_intraday_trend_is_preferred_and_regime_is_safe_fallback(self):
        direct = trade_summary.build_trade_summary(complete_input())
        self.assertEqual(direct["trend"]["code"], "up")
        self.assertEqual(direct["trend"]["strength"], 75.0)
        self.assertEqual(direct["trend"]["basis"], "当日方向の既存診断")

        payload = complete_input()
        payload.pop("trend")
        payload["rule_evaluation"]["regime"] = "RANGE"
        fallback = trade_summary.build_trade_summary(payload)
        self.assertEqual(fallback["trend"]["code"], "sideways")
        self.assertIn("当日の方向ではありません", fallback["trend"]["basis"])

    def test_nested_session_report_is_summarized_without_assuming_tradability(self):
        result = trade_summary.build_trade_summary(complete_input())
        self.assertEqual(result["session"]["label_ja"], "立会時間")
        self.assertTrue(result["session"]["tradable"])
        self.assertAlmostEqual(result["session"]["change_vs_previous_close_pct"], 1.25)

        payload = complete_input()
        payload["session"]["current_session"].update({
            "session": "unknown", "tradable": None,
        })
        unknown = trade_summary.build_trade_summary(payload)
        self.assertIsNone(unknown["session"]["tradable"])
        self.assertIn("confirm_session_tradability",
                      [row["code"] for row in unknown["action_priorities"]])

    def test_next_upcoming_event_is_used_and_high_imminent_event_is_flagged(self):
        payload = complete_input()
        payload["event"]["events"] = [
            {"status": "RECENT", "name": "公表済みニュース",
             "event_date": "2026-08-10", "impact_level": "HIGH"},
            {"status": "UPCOMING", "display_name_ja": "米国CPI発表",
             "event_date": "2026-08-12", "impact_level": "HIGH",
             "impact_score": 68, "session_label_ja": "プレマーケット"},
            {"status": "UPCOMING", "name": "決算発表",
             "event_date": "2026-10-01", "impact_level": "HIGH"},
        ]
        result = trade_summary.build_trade_summary(payload)
        event = result["event_risk"]
        self.assertEqual(event["event_name"], "米国CPI発表")
        self.assertEqual(event["days_until"], 2)
        self.assertTrue(event["imminent"])
        self.assertEqual(event["impact_stars"], 4)
        self.assertIn("review_event_volatility",
                      [row["code"] for row in result["action_priorities"]])

    def test_empty_event_report_is_different_from_missing_event_data(self):
        empty = trade_summary.build_trade_summary({"event": {"events": []}})
        missing = trade_summary.build_trade_summary({})
        self.assertTrue(empty["event_risk"]["available"])
        self.assertEqual(empty["event_risk"]["level"], "none")
        self.assertFalse(empty["event_risk"]["imminent"])
        self.assertFalse(missing["event_risk"]["available"])
        self.assertEqual(missing["event_risk"]["level"], "unknown")

    def test_unavailable_event_report_never_claims_no_upcoming_events(self):
        result = trade_summary.build_trade_summary({
            "event": {
                "status": "unavailable", "events": [],
                "warnings": ["カレンダー取得失敗"],
            },
        })
        event = result["event_risk"]
        self.assertFalse(event["available"])
        self.assertEqual(event["level"], "unknown")
        self.assertIn("取得できません", event["reason"])
        self.assertEqual(event["warnings"], ("カレンダー取得失敗",))

    def test_partial_report_keeps_a_valid_upcoming_event_with_warning(self):
        payload = complete_input()
        payload["event"].update({
            "status": "partial", "warnings": ["ニュース取得失敗"],
        })
        result = trade_summary.build_trade_summary(payload)
        self.assertTrue(result["event_risk"]["available"])
        self.assertEqual(result["event_risk"]["report_status"], "partial")
        self.assertEqual(result["event_risk"]["warnings"], ("ニュース取得失敗",))


class SafetyAndQualityTests(unittest.TestCase):
    def test_event_and_context_never_change_rule_verdict_or_score(self):
        payload = complete_input("BUY")
        payload["event"]["events"][0].update({
            "event_date": "2026-08-10", "impact_level": "HIGH",
            "impact_score": 100,
        })
        result = trade_summary.build_trade_summary(payload)
        self.assertEqual(result["verdict"]["code"], "BUY")
        self.assertEqual(result["score_effect"], 0)
        self.assertFalse(result["automatic_trade_score"])
        self.assertTrue(result["read_only"])
        self.assertFalse(result["places_orders"])
        self.assertFalse(result["uses_network"])

    def test_complete_input_reports_full_coverage(self):
        result = trade_summary.build_trade_summary(complete_input())
        self.assertEqual(result["data_quality"]["status"], "sufficient")
        self.assertEqual(result["data_quality"]["coverage_pct"], 100)
        self.assertEqual(result["data_quality"]["missing"], ())

    def test_missing_data_stays_missing_and_generates_refresh_priority(self):
        result = trade_summary.build_trade_summary(None)
        self.assertEqual(result["data_quality"]["status"], "insufficient")
        self.assertGreater(len(result["data_quality"]["missing"]), 0)
        self.assertIn("refresh_missing_data",
                      [row["code"] for row in result["action_priorities"]])
        self.assertIsNone(result["entry"]["price"])
        self.assertIsNone(result["stop"]["price"])
        self.assertIsNone(result["target"]["price"])

    def test_risk_exit_is_always_the_first_priority(self):
        payload = complete_input("RISK_EXIT")
        payload["session"]["current_session"].update({
            "session": "closed", "tradable": False,
        })
        result = trade_summary.build_trade_summary(payload)
        self.assertEqual(result["action_priorities"][0]["code"], "confirm_risk_exit")
        self.assertEqual(
            [row["priority"] for row in result["action_priorities"]],
            list(range(1, len(result["action_priorities"]) + 1)),
        )

    def test_input_is_not_mutated_and_non_finite_values_are_rejected(self):
        payload = complete_input()
        payload["rule_evaluation"]["risk_plan"]["stop"] = np.nan
        original = copy.deepcopy(payload)
        result = trade_summary.build_trade_summary(payload)
        self.assertEqual(payload["levels"], original["levels"])
        self.assertTrue(np.isnan(payload["rule_evaluation"]["risk_plan"]["stop"]))
        self.assertFalse(result["stop"]["available"])
        self.assertFalse(result["risk_plan_valid"])

    def test_non_mapping_input_is_rejected(self):
        with self.assertRaises(TypeError):
            trade_summary.build_trade_summary("AAPL")


if __name__ == "__main__":
    unittest.main()
