import inspect
from pathlib import Path
import unittest

from lib import trade_visuals


EXPECTED_KEYS = {
    "code",
    "icon",
    "action_label_ja",
    "context_label_ja",
    "title_ja",
    "description_ja",
    "severity",
    "color",
    "is_buy",
    "is_sell",
    "is_actionable",
    "short_sale",
    "available",
}


class VerdictVisualTests(unittest.TestCase):
    def test_all_supported_codes_have_distinct_visual_actions(self):
        expected = {
            "BUY": ("➕", "【買い】新規買い候補", "success", "green"),
            "NEUTRAL": ("—", "【見送り】今は買わない", "info", "gray"),
            "WAIT": ("⏳", "【保留】売買せず待機", "warning", "orange"),
            "RISK_EXIT": (
                "⚠️", "【売却】保有株の売却候補（リスク退出）", "error", "red"),
            "TAKE_PROFIT": (
                "✅", "【売却】保有株の売却候補（利益確定）", "info", "blue"),
            "HOLD": ("●", "【継続】保有継続", "info", "gray"),
        }
        for code, visual in expected.items():
            with self.subTest(code=code):
                result = trade_visuals.verdict_visual(code)
                self.assertEqual(set(result), EXPECTED_KEYS)
                self.assertEqual(
                    (result["icon"], result["action_label_ja"],
                     result["severity"], result["color"]),
                    visual,
                )
                self.assertEqual(result["code"], code)
                self.assertFalse(result["short_sale"])
                self.assertTrue(result["available"])

    def test_action_flags_separate_buy_sell_and_no_action(self):
        buy = trade_visuals.verdict_visual("BUY")
        self.assertTrue(buy["is_buy"])
        self.assertFalse(buy["is_sell"])
        self.assertTrue(buy["is_actionable"])

        for code in ("RISK_EXIT", "TAKE_PROFIT"):
            with self.subTest(code=code):
                sell = trade_visuals.verdict_visual(code)
                self.assertFalse(sell["is_buy"])
                self.assertTrue(sell["is_sell"])
                self.assertTrue(sell["is_actionable"])
                self.assertIn("保有株", sell["description_ja"])
                self.assertIn("新規の空売り", sell["description_ja"])

        for code in ("NEUTRAL", "WAIT", "HOLD"):
            with self.subTest(code=code):
                no_action = trade_visuals.verdict_visual(code)
                self.assertFalse(no_action["is_buy"])
                self.assertFalse(no_action["is_sell"])
                self.assertFalse(no_action["is_actionable"])

    def test_blocked_buy_is_visible_but_not_actionable(self):
        result = trade_visuals.verdict_visual("BUY", blocked=True)
        self.assertEqual(result["code"], "BUY")
        self.assertEqual(result["icon"], "⏸️")
        self.assertEqual(
            result["action_label_ja"], "【買い・実行保留】新規買い候補")
        self.assertEqual(result["severity"], "warning")
        self.assertEqual(result["color"], "orange")
        self.assertTrue(result["is_buy"])
        self.assertFalse(result["is_sell"])
        self.assertFalse(result["is_actionable"])
        self.assertFalse(result["short_sale"])
        self.assertTrue(result["available"])

    def test_blocked_does_not_change_non_buy_verdicts(self):
        for code in ("NEUTRAL", "WAIT", "RISK_EXIT", "TAKE_PROFIT", "HOLD"):
            with self.subTest(code=code):
                self.assertEqual(
                    trade_visuals.verdict_visual(code, blocked=True),
                    trade_visuals.verdict_visual(code),
                )

    def test_unknown_or_missing_codes_fall_back_to_non_actionable_wait(self):
        for code in (None, "", "SELL", "STRONG_BUY", object()):
            with self.subTest(code=code):
                result = trade_visuals.verdict_visual(code)
                self.assertEqual(result["code"], "WAIT")
                self.assertEqual(result["icon"], "？")
                self.assertEqual(
                    result["action_label_ja"], "【判定不能】売買せず待機")
                self.assertFalse(result["is_buy"])
                self.assertFalse(result["is_sell"])
                self.assertFalse(result["is_actionable"])
                self.assertFalse(result["short_sale"])
                self.assertFalse(result["available"])

    def test_code_matching_is_case_and_whitespace_tolerant(self):
        result = trade_visuals.verdict_visual(
            "  risk_exit  ", position_mode=" HOLDING ",
        )
        self.assertEqual(result["code"], "RISK_EXIT")
        self.assertTrue(result["is_sell"])

    def test_entry_mode_accepts_only_entry_verdicts(self):
        for code in ("BUY", "NEUTRAL", "WAIT"):
            with self.subTest(code=code):
                result = trade_visuals.verdict_visual(
                    code, position_mode="entry",
                )
                self.assertEqual(result["code"], code)
                self.assertTrue(result["available"])

        for code in ("RISK_EXIT", "TAKE_PROFIT", "HOLD"):
            with self.subTest(code=code):
                result = trade_visuals.verdict_visual(
                    code, position_mode="entry",
                )
                self._assert_unavailable_wait(result)

    def test_holding_mode_accepts_only_holding_verdicts(self):
        for code in ("RISK_EXIT", "TAKE_PROFIT", "HOLD", "WAIT"):
            with self.subTest(code=code):
                result = trade_visuals.verdict_visual(
                    code, position_mode="holding",
                )
                self.assertEqual(result["code"], code)
                self.assertTrue(result["available"])

        for code in ("BUY", "NEUTRAL"):
            with self.subTest(code=code):
                result = trade_visuals.verdict_visual(
                    code, position_mode="holding",
                )
                self._assert_unavailable_wait(result)

    def test_unknown_position_mode_is_unavailable(self):
        result = trade_visuals.verdict_visual(
            "BUY", position_mode="portfolio",
        )
        self._assert_unavailable_wait(result)

    def test_holding_wait_uses_holding_context(self):
        entry = trade_visuals.verdict_visual("WAIT", position_mode="entry")
        holding = trade_visuals.verdict_visual("WAIT", position_mode="holding")
        self.assertEqual(entry["context_label_ja"], "新規エントリー")
        self.assertEqual(holding["context_label_ja"], "保有中")

    def test_evaluation_visual_blocks_buy_when_risk_plan_is_invalid(self):
        result = trade_visuals.evaluation_visual({
            "verdict": "BUY", "risk_plan": {"valid": False},
        }, position_mode="entry")
        self.assertEqual(result["code"], "BUY")
        self.assertIn("実行保留", result["action_label_ja"])
        self.assertFalse(result["is_actionable"])

    def test_evaluation_visual_preserves_safe_mode_validation(self):
        result = trade_visuals.evaluation_visual({
            "verdict": "RISK_EXIT", "risk_plan": {"valid": True},
        }, position_mode="entry")
        self._assert_unavailable_wait(result)

    def _assert_unavailable_wait(self, result):
        self.assertEqual(result["code"], "WAIT")
        self.assertEqual(result["icon"], "？")
        self.assertEqual(
            result["action_label_ja"], "【判定不能】売買せず待機")
        self.assertFalse(result["available"])
        self.assertFalse(result["is_buy"])
        self.assertFalse(result["is_sell"])
        self.assertFalse(result["is_actionable"])
        self.assertFalse(result["short_sale"])

    def test_module_is_pure_and_has_no_ui_network_or_io_dependencies(self):
        source = inspect.getsource(trade_visuals)
        for forbidden in (
            "streamlit", "requests", "urllib", "socket", "open(",
            "sqlite", "moomoo", "yfinance",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source.lower())

    def test_all_three_views_use_the_shared_visual_projection(self):
        root = Path(__file__).resolve().parents[1]
        signals = (root / "views" / "signals.py").read_text(encoding="utf-8")
        stock = (root / "views" / "stock_analysis.py").read_text(encoding="utf-8")
        board = (root / "lib" / "board_ui.py").read_text(encoding="utf-8")
        self.assertIn("trade_visuals.evaluation_visual", signals)
        self.assertIn("trade_visuals.evaluation_visual", stock)
        self.assertIn("trade_visuals.evaluation_visual", board)
        self.assertIn("_verdict_label(evaluated)", signals)
        self.assertNotIn('_verdict_label(evaluated["verdict"])', signals)
        self.assertNotIn("VERDICT_STYLE =", signals)
        self.assertNotIn("TRADE_VERDICT_VIEW =", stock)


if __name__ == "__main__":
    unittest.main()
