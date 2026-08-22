"""リアルタイム売買タイミングUIの安全契約を確認する。"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


SOURCE_PATH = Path(__file__).parents[1] / "views" / "stock_analysis.py"


class RealtimeTimingUiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.card_node = next(
            node for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "render_realtime_timing_card"
        )
        cls.fragment_node = next(
            node for node in ast.walk(cls.card_node)
            if isinstance(node, ast.FunctionDef)
            and node.name == "render_realtime_timing_panel"
        )
        cls.card = ast.get_source_segment(cls.source, cls.card_node) or ""
        cls.fragment = ast.get_source_segment(cls.source, cls.fragment_node) or ""

    def test_card_is_called_once_before_daily_history_fetch(self):
        call_text = "render_realtime_timing_card(ticker)"
        call_position = self.source.index(call_text, self.source.index("st.title("))
        history_position = self.source.index(
            "data_fetcher.fetch_chart_history(", call_position)
        self.assertLess(call_position, history_position)
        ticker_guard = self.source.index("if not ticker:")
        self.assertGreater(call_position, self.source.index("st.stop()", ticker_guard))
        self.assertEqual(self.source.count(call_text), 1)
        self.assertEqual(
            self.source.count("### ⚡ リアルタイム売買タイミング"), 1)
        self.assertNotIn("realtime_box =", self.source)

    def test_card_is_a_top_level_function_independent_of_daily_page_state(self):
        self.assertLess(self.card_node.lineno, self.source[:self.source.index(
            'st.title("📈 銘柄分析")')].count("\n") + 1)
        self.assertEqual([arg.arg for arg in self.card_node.args.args], ["ticker"])
        self.assertIn("with st.container(border=True):", self.card)
        loaded_names = {
            node.id for node in ast.walk(self.card_node)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        forbidden = {
            "hist", "_base_meta", "snapshot", "price_now", "prev",
            "market_state", "decision_context", "entry_evaluation",
            "holding_evaluation", "purchase_plan", "active_rule", "tab_today",
            "realtime_box", "summary_box", "info", "period_label",
            "fetch_period", "display_days",
        }
        self.assertFalse(loaded_names & forbidden)
        self.assertNotIn("fetch_chart_history", self.card)
        self.assertNotIn("allow_new_quota", self.card)
        self.assertIn(
            "上のリアルタイム売買タイミングは、moomooの現在データがあれば利用できます",
            self.source,
        )

    def test_monitor_is_explicit_opt_in_and_uses_fragment_intervals(self):
        self.assertIn("st.session_state[realtime_monitor_key] = False", self.source)
        self.assertIn("REALTIME_REFRESH_SECONDS = (5, 10, 30)", self.source)
        self.assertIn("@st.fragment(run_every=", self.source)
        self.assertIn("リアルタイム監視（ONで開始 / OFFで停止）", self.source)
        self.assertIn('st.caption("⏸ 監視中" if realtime_monitoring else "▶ 停止中")', self.source)

    def test_live_adapter_does_not_fall_back_to_delayed_quote(self):
        self.assertIn('getattr(data_fetcher, "fetch_current_klines", None)', self.source)
        self.assertIn("fetch_realtime_snapshot(ticker)", self.source)
        self.assertIn("Yahoo Financeの代替値はリアルタイム判定に使用しません", self.source)
        self.assertIn('"uses_history_quota": False', self.source)
        self.assertIn('if "moomoo" not in live_source.lower()', self.source)

    def test_ui_session_names_are_adapted_for_current_kline_api(self):
        self.assertIn('"premarket": "pre"', self.source)
        self.assertIn('"afterhours": "after"', self.source)
        self.assertIn("session=api_session", self.source)

    def test_memory_is_separated_by_symbol_mode_and_session(self):
        self.assertIn(
            "identity = (ticker, realtime_mode, live_session_code)", self.source)
        self.assertIn("previous_memory=realtime_memories.get(identity)", self.source)
        self.assertIn("同じ足を何度更新しても確認回数には加えません", self.source)

    def test_holding_copy_never_implies_short_sale(self):
        self.assertIn("すでに保有している株の売却だけを意味します", self.source)
        self.assertIn("新しい空売りではありません", self.source)
        self.assertNotIn("OpenSecTradeContext", self.source)
        self.assertNotIn(".place_order(", self.source)
        self.assertIn("監視する損切り価格", self.source)
        self.assertIn("監視する利益確定価格", self.source)
        self.assertNotIn('"daily_verdict"', self.fragment)
        self.assertNotIn('"stop": live_risk.get', self.source)

    def test_three_beginner_steps_and_quota_disclosure_are_present(self):
        for label in ("① 確定1分足の短期条件", "② 今のタイミング", "③ 売買前の安全確認"):
            self.assertIn(label, self.source)
        self.assertIn("moomoo過去K線枠の追加使用は0です", self.source)
        self.assertIn("日足・チャートは別機能です", self.source)
        self.assertIn("そちらが別途", self.source)

    def test_realtime_judgement_is_independent_of_daily_and_purchase_plan(self):
        for forbidden in (
            "daily_signal", "daily_verdict", "holding_evaluation",
            "build_purchase_plan", "purchase_plan=", "確定日足",
        ):
            self.assertNotIn(forbidden, self.fragment)
        self.assertIn("形成途中の足、日足、購入プランの判定は混ぜません", self.fragment)

    def test_live_session_state_is_refetched_inside_fragment(self):
        self.assertIn("live_market_state = data_fetcher.fetch_market_state(ticker)", self.fragment)
        self.assertIn("market_state=live_market_state.get", self.fragment)
        self.assertIn("calendar_session = session_intelligence.detect_current_session", self.fragment)
        self.assertIn('"calendar_session": calendar_session.get("session")', self.fragment)

    def test_beginner_labels_do_not_expose_risk_exit_jargon(self):
        self.assertIn("保有株を売る候補（損失を抑える）", self.source)
        helper = self.source.split("def realtime_action_visual", 1)[1].split(
            "def realtime_gate_summary", 1)[0]
        self.assertNotIn('"label_ja": action.get("label_ja")', helper)

    def test_all_four_trading_sessions_are_actionable(self):
        self.assertIn('"regular", "premarket", "afterhours", "overnight"', self.fragment)
        self.assertIn("プレ・アフター・夜間も取引対象です", self.fragment)
        self.assertNotIn("観測だけ", self.fragment)
        self.assertIn("1分足や気配値は短時間で反転します", self.source)

    def test_missing_session_specific_quality_fails_closed(self):
        self.assertIn('meta.get("available") is not True', self.source)
        self.assertIn('meta.get("partial") is not False', self.source)
        self.assertIn("received_session != expected_session", self.source)
        self.assertIn("bars.empty or benchmark.empty", self.source)
        self.assertIn("そのセッションの1分足・気配値・データ品質を確認できない場合は判断を保留", self.fragment)
        self.assertIn("data_fetcher.build_session_snapshot(", self.fragment)
        self.assertIn('live_decision_snapshot.get("decision_ready") is not True', self.fragment)
        self.assertIn("現在の取引時間に対応した価格・気配・", self.fragment)
        self.assertIn("snapshot=live_decision_snapshot", self.fragment)

    def test_nonactionable_exit_is_not_presented_as_immediately_tradable(self):
        helper = self.source.split("def realtime_action_visual", 1)[1].split(
            "def realtime_gate_summary", 1)[0]
        self.assertIn("価格目安に到達（今は取引できません）", helper)
        self.assertIn('result.get("actionable") is not True', helper)
        self.assertIn("if not nonactionable_exit:", helper)

    def test_entry_and_holding_both_use_current_session_minute_bars(self):
        self.assertIn(
            "買い・保有中のどちらも、現在セッションの確定1分足とSPYを同じ条件で使う",
            self.fragment,
        )
        self.assertIn(
            "live_klines = fetch_realtime_klines_for_ui(ticker, live_session_code)",
            self.fragment,
        )
        self.assertNotIn('if realtime_mode == "entry":\n            live_klines =', self.fragment)

    def test_overnight_monitoring_is_not_stopped_at_midnight(self):
        self.assertNotIn("realtime_page_market_date", self.source)
        self.assertNotIn("日付が変わりました", self.fragment)
        self.assertIn(
            "identity = (ticker, realtime_mode, live_session_code)", self.fragment)

    def test_subscription_retention_is_disclosed(self):
        self.assertIn("再利用のため購読枠が保持される場合があります", self.source)


if __name__ == "__main__":
    unittest.main()
