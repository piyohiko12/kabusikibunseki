import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import pandas as pd

from lib import alerts


class RuleAlertTests(unittest.TestCase):
    def test_profit_and_risk_exit_are_distinct(self):
        profit = alerts.new_alert("AAPL", "rule_take_profit")
        risk = alerts.new_alert("AAPL", "rule_risk_exit")
        self.assertTrue(alerts.check(profit, {"rule_verdict": "TAKE_PROFIT"})["triggered"])
        self.assertFalse(alerts.check(risk, {"rule_verdict": "TAKE_PROFIT"})["triggered"])

    def test_legacy_sell_matches_either_long_exit(self):
        legacy = alerts.new_alert("AAPL", "rule_sell")
        self.assertTrue(alerts.check(legacy, {"rule_verdict": "RISK_EXIT"})["triggered"])
        self.assertTrue(alerts.check(legacy, {"rule_verdict": "TAKE_PROFIT"})["triggered"])
        self.assertFalse(alerts.check(legacy, {"rule_verdict": "BUY"})["triggered"])

    def test_entry_and_holding_verdicts_are_selected_separately(self):
        ctx = {"entry_verdict": "BUY", "holding_verdict": "HOLD"}
        self.assertTrue(alerts.check(
            alerts.new_alert("AAPL", "rule_buy"), ctx)["triggered"])
        self.assertFalse(alerts.check(
            alerts.new_alert("AAPL", "rule_risk_exit"), ctx)["triggered"])

    def test_blocked_buy_alert_still_triggers_but_explains_to_wait(self):
        alert = alerts.new_alert("AAPL", "rule_buy")
        checked = alerts.check(alert, {
            "entry_verdict": "BUY", "entry_blocked": True,
        })
        self.assertTrue(checked["triggered"])
        self.assertEqual(checked["actual"], "BUY_BLOCKED")
        self.assertEqual(
            alerts.format_actual(alert, checked["actual"]),
            "買い条件あり・今は待つ",
        )

    def test_price_alert_prefers_realtime_snapshot(self):
        frame = pd.DataFrame({"Close": [100.0, 101.0]})
        alert = alerts.new_alert("AAPL", "price_above", 105)
        result = alerts.check(alert, {"df": frame, "price": 106.0,
                                      "previous_close": 100.0})
        self.assertTrue(result["triggered"])
        self.assertEqual(result["actual"], "$106.00")

    def test_explicit_missing_session_price_does_not_fall_back_to_daily_close(self):
        frame = pd.DataFrame({"Close": [100.0, 110.0]})
        price_result = alerts.check(
            alerts.new_alert("AAPL", "price_above", 105),
            {"df": frame, "price": None},
        )
        change_result = alerts.check(
            alerts.new_alert("AAPL", "change_above", 0.1),
            {"df": frame, "price": None, "previous_close": 100.0},
        )
        level_result = alerts.check(
            alerts.new_alert("AAPL", "near_resistance", 3),
            {"df": frame, "price": None,
             "levels": [{"type": "抵抗線", "price": 111.0}]},
        )
        for result in (price_result, change_result, level_result):
            self.assertFalse(result["triggered"])
            self.assertEqual(result["actual"], "取得できず")

    def test_rule_actuals_are_japanese_and_legacy_sell_is_not_short_sale(self):
        cases = {
            "BUY": "買い候補",
            "BUY_BLOCKED": "買い条件あり・今は待つ",
            "RISK_EXIT": "保有株を売る候補（損失を抑える）",
            "TAKE_PROFIT": "保有株を売る候補（利益を確定する）",
            "HOLD": "そのまま保有",
            "SELL": "保有株を売る候補（以前の設定）",
        }
        for actual, expected in cases.items():
            with self.subTest(actual=actual):
                shown = alerts.format_actual(
                    alerts.new_alert("AAPL", "rule_sell"), actual)
                self.assertIn(expected, shown)
                self.assertNotIn("空売り", shown)
                self.assertNotIn(f"（{actual}）", shown)

    def test_non_rule_actual_is_not_relabelled(self):
        alert = alerts.new_alert("AAPL", "price_above", 100)
        self.assertEqual(alerts.format_actual(alert, "$101.00"), "$101.00")
        self.assertEqual(alerts.format_actual(alert, 0), "0")

    def test_resistance_alert_ignores_already_crossed_level(self):
        result = alerts.check(alerts.new_alert(
            "AAPL", "near_resistance", 5), {
                "price": 101,
                "levels": [
                    {"type": "抵抗線", "price": 100},
                    {"type": "抵抗線", "price": 110},
                ],
            })
        self.assertFalse(result["triggered"])
        self.assertEqual(result["actual"], "8.91%")

    def test_support_alert_ignores_level_above_current_price(self):
        result = alerts.check(alerts.new_alert(
            "AAPL", "near_support", 5), {
                "price": 101,
                "levels": [
                    {"type": "サポート", "price": 100},
                    {"type": "サポート", "price": 110},
                ],
            })
        self.assertTrue(result["triggered"])
        self.assertEqual(result["actual"], "0.99%")

    def test_missing_threshold_fails_safe_in_check_and_describe(self):
        broken = {"ticker": "AAPL", "kind": "price_above", "value": None}
        result = alerts.check(broken, {"price": 120})
        self.assertFalse(result["triggered"])
        self.assertEqual(result["actual"], "設定不正")
        self.assertIn("基準値が不正", alerts.describe(broken))

    def test_load_skips_missing_values_and_disables_malformed_enabled_flag(self):
        payload = {"alerts": [
            {"ticker": "AAPL", "kind": "price_above", "value": None},
            {"ticker": "MSFT", "kind": "price_below", "value": "101.5",
             "enabled": "false", "note": 123},
            {"ticker": "NVDA", "kind": "rule_buy"},
        ]}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(alerts, "DATA_FILE", path):
                loaded = alerts.load()
        self.assertEqual([row["ticker"] for row in loaded], ["MSFT", "NVDA"])
        self.assertEqual(loaded[0]["value"], 101.5)
        self.assertFalse(loaded[0]["enabled"])
        self.assertEqual(loaded[0]["note"], "")


if __name__ == "__main__":
    unittest.main()
