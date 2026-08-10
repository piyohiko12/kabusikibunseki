import unittest

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

    def test_price_alert_prefers_realtime_snapshot(self):
        frame = pd.DataFrame({"Close": [100.0, 101.0]})
        alert = alerts.new_alert("AAPL", "price_above", 105)
        result = alerts.check(alert, {"df": frame, "price": 106.0,
                                      "previous_close": 100.0})
        self.assertTrue(result["triggered"])
        self.assertEqual(result["actual"], "$106.00")

    def test_rule_actuals_are_japanese_and_legacy_sell_is_not_short_sale(self):
        cases = {
            "BUY": "新規買い候補",
            "RISK_EXIT": "保有株の売却候補・リスク退出",
            "TAKE_PROFIT": "保有株の売却候補・利益確定",
            "HOLD": "保有継続",
            "SELL": "保有株の手仕舞い・旧形式",
        }
        for actual, expected in cases.items():
            with self.subTest(actual=actual):
                shown = alerts.format_actual(
                    alerts.new_alert("AAPL", "rule_sell"), actual)
                self.assertIn(expected, shown)
                self.assertNotIn("空売り", shown)

    def test_non_rule_actual_is_not_relabelled(self):
        alert = alerts.new_alert("AAPL", "price_above", 100)
        self.assertEqual(alerts.format_actual(alert, "$101.00"), "$101.00")
        self.assertEqual(alerts.format_actual(alert, 0), "0")


if __name__ == "__main__":
    unittest.main()
