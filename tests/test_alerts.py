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


if __name__ == "__main__":
    unittest.main()
