import copy
import unittest

import numpy as np
import pandas as pd

from lib import rules


def sample_frame(rows: int = 260, direction: str = "up") -> pd.DataFrame:
    index = pd.bdate_range("2025-01-02", periods=rows)
    if direction == "up":
        close = np.linspace(100.0, 130.0, rows)
        sma200 = np.linspace(92.0, 120.0, rows)
        sma50 = np.linspace(96.0, 125.0, rows)
    elif direction == "down":
        close = np.linspace(130.0, 100.0, rows)
        sma200 = np.linspace(122.0, 110.0, rows)
        sma50 = np.linspace(118.0, 105.0, rows)
    else:
        close = np.full(rows, 100.0)
        sma200 = np.full(rows, 100.0)
        sma50 = np.full(rows, 100.0)

    frame = pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 0.8,
        "Low": close - 0.8,
        "Close": close,
        "Volume": np.full(rows, 1_000.0),
        "VOL_MA20": np.full(rows, 1_000.0),
        "SMA20": pd.Series(close, index=index).rolling(20).mean().to_numpy(),
        "SMA50": sma50,
        "SMA200": sma200,
        "RSI": np.linspace(42.0, 45.0, rows),
        "MACD_hist": np.linspace(-0.2, 0.3, rows),
        "STOCH_K": np.linspace(30.0, 50.0, rows),
        "BB_low": close - 4.0,
        "BB_up": close + 4.0,
    }, index=index)
    return frame


def sample_levels(price: float = 130.0) -> list[dict]:
    return [
        {"type": "サポート", "price": price - 2.0,
         "zone_low": price - 2.5, "zone_high": price - 1.5, "strength": 4},
        {"type": "抵抗線", "price": price + 8.0,
         "zone_low": price + 7.5, "zone_high": price + 8.5, "strength": 4},
    ]


def side(threshold: int, conditions: list[dict], caps: dict | None = None) -> dict:
    result = {"threshold": threshold, "conditions": conditions}
    if caps is not None:
        result["group_caps"] = caps
    return result


def condition(metric: str, op: str, value: float, points: int = 10,
              required: bool = False) -> dict:
    return {"metric": metric, "op": op, "value": value,
            "points": points, "required": required}


class MetricTests(unittest.TestCase):
    def test_every_metric_declares_a_supported_group(self):
        self.assertTrue(rules.METRICS)
        for metric in rules.METRICS.values():
            self.assertIn(metric["group"], rules.GROUPS)

    def test_derived_momentum_and_trend_metrics(self):
        frame = sample_frame()
        ctx = {"df": frame, "levels": sample_levels(),
               "risk_plan": rules.build_risk_plan(frame, sample_levels())}

        self.assertGreater(rules.METRICS["sma200_slope"]["fn"](ctx), 0)
        self.assertGreater(rules.METRICS["sma50_above_200"]["fn"](ctx), 0)
        self.assertGreater(rules.METRICS["rsi_change"]["fn"](ctx), 0)
        self.assertGreater(rules.METRICS["macd_hist_change"]["fn"](ctx), 0)
        self.assertEqual(rules.METRICS["macd_hist_improving_2"]["fn"](ctx), 1)
        self.assertEqual(rules.METRICS["reversal_confirm"]["fn"](ctx), 1)
        self.assertGreaterEqual(rules.METRICS["support_strength"]["fn"](ctx), 3)
        self.assertGreater(rules.METRICS["risk_reward"]["fn"](ctx), 1)

    def test_breakout_and_atr_normalised_return(self):
        frame = sample_frame()
        frame.loc[frame.index[-1], ["Open", "High", "Low", "Close"]] = [135, 136, 134, 136]
        ctx = {"df": frame, "levels": []}
        self.assertGreaterEqual(rules.METRICS["breakout_20d"]["fn"](ctx), 0)
        self.assertGreater(rules.METRICS["ret_1d_atr"]["fn"](ctx), 0)


class SideEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = sample_frame()
        self.ctx = {"df": self.frame, "levels": sample_levels(),
                    "risk_plan": rules.build_risk_plan(self.frame, sample_levels())}

    def test_required_missing_is_unavailable_and_cannot_pass(self):
        result = rules.evaluate_side(self.ctx, side(10, [
            condition("support_strength", ">=", 3, 10, True),
        ]))
        self.assertTrue(result["passed"])

        missing = rules.evaluate_side({"df": self.frame, "levels": []}, side(10, [
            condition("support_strength", ">=", 3, 10, True),
        ]))
        self.assertFalse(missing["passed"])
        self.assertFalse(missing["available"])
        self.assertEqual(missing["checks"][0]["status"], "unavailable")

    def test_required_failure_blocks_even_when_score_reaches_threshold(self):
        result = rules.evaluate_side(self.ctx, side(10, [
            condition("rsi", ">=", 90, 10, True),
            condition("sma200_gap", ">=", 0, 20),
        ]))
        self.assertGreaterEqual(result["score"], result["threshold"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["required_failed"])

    def test_group_cap_limits_correlated_points_and_total(self):
        result = rules.evaluate_side(self.ctx, side(20, [
            condition("sma20_gap", ">=", -100, 20),
            condition("sma50_gap", ">=", -100, 20),
            condition("sma200_gap", ">=", -100, 20),
        ], {"trend": 25}))
        self.assertEqual(result["raw_score"], 60)
        self.assertEqual(result["score"], 25)
        self.assertEqual(result["raw_total"], 60)
        self.assertEqual(result["total"], 25)

    def test_non_positive_threshold_is_invalid_and_never_passes(self):
        result = rules.evaluate_side(self.ctx, side(0, [
            condition("rsi", ">=", 0, 10),
        ]))
        self.assertFalse(result["valid"])
        self.assertFalse(result["passed"])
        self.assertTrue(result["invalid_reasons"])


class RegimeAndVerdictTests(unittest.TestCase):
    def simple_rule(self) -> dict:
        return {
            "schema_version": 2,
            "allowed_regimes": list(rules.REGIMES),
            "buy": side(10, [condition("sma200_gap", ">=", 0, 10)]),
            "take_profit": side(10, [condition("rsi", ">=", 40, 10)]),
            "risk_exit": side(10, [condition("ret_1d_atr", "<=", -0.1, 10)]),
        }

    def test_market_regimes(self):
        self.assertEqual(rules.market_regime(sample_frame(direction="up")), "UPTREND")
        self.assertEqual(rules.market_regime(sample_frame(direction="down")), "DOWNTREND")
        self.assertEqual(rules.market_regime(sample_frame(direction="range")), "RANGE")

        high_vol = sample_frame(direction="range")
        jumps = np.resize(np.array([80.0, 120.0]), len(high_vol))
        high_vol["Close"] = jumps
        high_vol["High"] = jumps + 5
        high_vol["Low"] = jumps - 5
        self.assertEqual(rules.market_regime(high_vol), "HIGH_VOL")

    def test_regime_and_required_external_gate_yield_wait(self):
        frame = sample_frame()
        disallowed = self.simple_rule()
        disallowed["allowed_regimes"] = ["RANGE"]
        self.assertEqual(rules.evaluate(frame, disallowed)["verdict"], "WAIT")

        external = [{"key": "freshness", "label": "データ鮮度", "passed": False,
                     "required": True, "reason": "古い"}]
        result = rules.evaluate(frame, self.simple_rule(), external_gates=external)
        self.assertEqual(result["verdict"], "WAIT")
        self.assertEqual(result["gates"][1]["key"], "freshness")

    def test_empty_allowed_regimes_blocks_entry(self):
        rule = self.simple_rule()
        rule["allowed_regimes"] = []
        result = rules.evaluate(sample_frame(), rule)
        self.assertEqual(result["verdict"], "WAIT")
        self.assertFalse(result["gates"][0]["passed"])

    def test_unknown_external_gate_is_warning_only(self):
        external = [{"key": "freshness", "passed": None, "required": True}]
        result = rules.evaluate(sample_frame(), self.simple_rule(),
                                external_gates=external)
        self.assertEqual(result["verdict"], "BUY")

    def test_holding_risk_exit_has_priority_over_profit_and_external_gate(self):
        frame = sample_frame()
        frame.loc[frame.index[-1], "Close"] = frame["Close"].iloc[-2] - 3
        rule = self.simple_rule()
        external = [{"key": "freshness", "passed": False, "required": True}]
        result = rules.evaluate(frame, rule, position_mode="holding",
                                external_gates=external)
        self.assertEqual(result["verdict"], "RISK_EXIT")
        self.assertIs(result["sell"], result["take_profit"])

    def test_holding_take_profit_and_hold(self):
        rule = self.simple_rule()
        profit = rules.evaluate(sample_frame(), rule, position_mode="holding")
        self.assertEqual(profit["verdict"], "TAKE_PROFIT")

        hold_rule = self.simple_rule()
        hold_rule["take_profit"] = side(10, [condition("rsi", ">=", 99, 10)])
        hold = rules.evaluate(sample_frame(), hold_rule, position_mode="holding")
        self.assertEqual(hold["verdict"], "HOLD")

    def test_take_profit_is_not_blocked_by_entry_safety_gate(self):
        external = [{"key": "earnings", "passed": False, "required": True}]
        result = rules.evaluate(
            sample_frame(), self.simple_rule(), position_mode="holding",
            external_gates=external)
        self.assertEqual(result["verdict"], "TAKE_PROFIT")

    def test_holding_waits_when_risk_exit_is_not_configured(self):
        rule = self.simple_rule()
        rule["risk_exit"] = side(0, [])
        result = rules.evaluate(sample_frame(), rule, position_mode="holding")
        self.assertEqual(result["verdict"], "WAIT")
        self.assertIn("未設定", result["summary"])

    def test_bad_position_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            rules.evaluate(sample_frame(), self.simple_rule(), position_mode="short")


class RiskPlanTests(unittest.TestCase):
    def test_levels_are_used_for_stop_target_and_rr(self):
        frame = sample_frame()
        plan = rules.build_risk_plan(frame, sample_levels())
        self.assertTrue(plan["valid"])
        self.assertLess(plan["stop"], plan["entry"])
        self.assertGreater(plan["target"], plan["entry"])
        self.assertGreater(plan["rr"], 1)
        self.assertIn("支持帯", plan["stop_source"])
        self.assertIn("抵抗帯", plan["target_source"])

    def test_atr_fallback_works_without_levels(self):
        plan = rules.build_risk_plan(sample_frame(), [])
        self.assertTrue(plan["valid"])
        self.assertAlmostEqual(plan["rr"], 2.0 / 1.5, places=6)
        self.assertIn("ATR", plan["stop_source"])


class MigrationAndValidationTests(unittest.TestCase):
    def test_old_sell_is_preserved_as_take_profit(self):
        old = {
            "buy": side(10, [condition("rsi", "<=", 40, 10)]),
            "sell": side(15, [condition("rsi", ">=", 70, 15)]),
        }
        upgraded = rules.upgrade_rule(old)
        self.assertEqual(upgraded["schema_version"], 2)
        self.assertEqual(upgraded["take_profit"], upgraded["sell"])
        self.assertEqual(upgraded["take_profit"]["conditions"][0]["metric"], "rsi")
        self.assertFalse(upgraded["buy"]["conditions"][0]["required"])
        self.assertNotIn("take_profit", old)

    def test_take_profit_is_kept_even_if_schema_marker_is_missing(self):
        v2_like = {
            "buy": side(10, [condition("rsi", "<=", 40, 10)]),
            "take_profit": side(17, [condition("rsi", ">=", 70, 17)]),
        }
        upgraded = rules.upgrade_rule(v2_like)
        self.assertEqual(upgraded["take_profit"]["threshold"], 17)
        self.assertEqual(upgraded["sell"], upgraded["take_profit"])

    def test_untouched_legacy_defaults_are_replaced_by_v2_defaults(self):
        legacy = rules._legacy_default_rules()
        migrated = rules.upgrade_rules(legacy)

        self.assertEqual(migrated["押し目買い"]["buy"]["threshold"], 70)
        self.assertTrue(migrated["押し目買い"]["risk_exit"]["conditions"])
        reversal = [item for item in migrated["押し目買い"]["buy"]["conditions"]
                    if item["metric"] == "reversal_confirm"]
        self.assertTrue(reversal[0]["required"])

    def test_modified_legacy_default_is_preserved_as_custom(self):
        legacy = rules._legacy_default_rules()
        custom = copy.deepcopy(legacy)
        custom["押し目買い"]["buy"]["threshold"] = 61
        original_conditions = copy.deepcopy(custom["押し目買い"]["buy"]["conditions"])

        migrated = rules.upgrade_rules(custom)["押し目買い"]
        self.assertEqual(migrated["buy"]["threshold"], 61)
        self.assertEqual(
            [{key: value for key, value in item.items() if key != "required"}
             for item in migrated["buy"]["conditions"]],
            original_conditions,
        )
        self.assertEqual(migrated["risk_exit"]["conditions"], [])
        self.assertEqual(migrated["take_profit"], migrated["sell"])

    def test_conditions_from_table_reads_required_column(self):
        table = pd.DataFrame([{
            "指標": rules.METRICS["rsi"]["label"], "条件": "<=",
            "しきい値": 40.0, "配点": 20, "必須": True,
        }])
        converted = rules.conditions_from_table(table)
        self.assertEqual(converted, [{"metric": "rsi", "op": "<=", "value": 40.0,
                                      "points": 20, "required": True}])

    def test_validate_detects_zero_threshold_cap_correlation_and_conflict(self):
        bad = {
            "schema_version": 2,
            "allowed_regimes": list(rules.REGIMES),
            "group_caps": {"trend": 0},
            "buy": side(0, [
                condition("sma20_gap", ">=", 5, 10, True),
                condition("sma20_gap", "<=", -5, 10, True),
                condition("sma50_gap", ">=", 0, 10),
                condition("sma200_gap", ">=", 0, 10),
            ]),
            "take_profit": side(10, [condition("rsi", ">=", 70, 10)]),
            "risk_exit": side(10, [condition("ret_1d_atr", "<=", -1, 10)]),
        }
        problems = rules.validate(bad)
        joined = "\n".join(problems)
        self.assertIn("合格点は1点以上", joined)
        self.assertIn("グループ上限は1点以上", joined)
        self.assertIn("相関重複", joined)
        self.assertIn("同時成立できない", joined)

    def test_default_v2_rules_validate_and_require_reversal(self):
        for rule in rules.default_rules().values():
            self.assertEqual(rules.validate(rule), [])
            reversal = [item for item in rule["buy"]["conditions"]
                        if item["metric"] == "reversal_confirm"]
            self.assertEqual(len(reversal), 1)
            self.assertTrue(reversal[0]["required"])
            self.assertTrue(rule["group_caps"])


if __name__ == "__main__":
    unittest.main()
