"""銘柄分析の価格経路・遅延読込・外部表示に関する統合契約。"""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
SOURCE_PATH = ROOT / "views" / "stock_analysis.py"


def _call_name(node: ast.Call) -> str:
    parts = []
    value = node.func
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _if_names(ancestors) -> set[str]:
    names = set()
    for ancestor in ancestors:
        if isinstance(ancestor, ast.If):
            names.update(
                node.id for node in ast.walk(ancestor.test)
                if isinstance(node, ast.Name)
            )
    return names


def _has_explicit_button_gate(ancestors) -> bool:
    for ancestor in ancestors:
        if not isinstance(ancestor, ast.If):
            continue
        if any(
                isinstance(node, ast.Call) and _call_name(node) == "st.button"
                for node in ast.walk(ancestor.test)):
            return True
    return False


class _CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.calls = []

    def generic_visit(self, node):
        self.stack.append(node)
        if isinstance(node, ast.Call):
            self.calls.append(
                (_call_name(node), node, tuple(self.stack[:-1])))
        super().generic_visit(node)
        self.stack.pop()


class StockAnalysisRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        collector = _CallCollector()
        collector.visit(tree)
        cls.calls = collector.calls
        cls.chart_start_line = cls.source[:cls.source.index(
            "# ---------------------------------------------------------------- チャート・指標")].count("\n") + 1

    def test_session_price_is_shared_by_top_purchase_alert_chart_and_board(self):
        required = (
            "session_price = data_fetcher.select_session_price(",
            '"price": price_for_alerts,',
            '"snapshot": session_snapshot,',
            '"current_price": price_now,',
            "stock_board_snapshot = dict(session_snapshot or {})",
        )
        for snippet in required:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, self.source)
        self.assertIn('"current_price_as_of": session_price.get("as_of")',
                      self.source)
        self.assertNotIn(
            '"current_price_as_of": (session_price.get("as_of") '
            'or session_price.get("observed_at"))', self.source)

    def test_hidden_news_social_and_financial_calls_require_explicit_action(self):
        expected_gate = {
            "data_fetcher.fetch_analyst": "load_analyst",
            "news_fetcher.fetch_news": "load_news",
            "news_fetcher.fetch_google_news": "load_news",
            "news_fetcher.fetch_sec_filings": "load_news",
            "news_fetcher.fetch_social": "load_social",
            "data_fetcher.fetch_earnings_history": "load_financial",
            "data_fetcher.fetch_annual_financials": "load_financial",
        }
        for call_name, gate in expected_gate.items():
            matched = [ancestors for name, _, ancestors in self.calls
                       if name == call_name]
            self.assertTrue(matched, call_name)
            for ancestors in matched:
                self.assertIn(gate, _if_names(ancestors), call_name)
        self.assertNotIn("chart_financial_load", self.source)
        self.assertIn('load_financial = st.button(', self.source)
        self.assertIn('st.session_state["chart_financial_results"]', self.source)
        self.assertIn("while len(financial_store) > 8", self.source)
        self.assertIn("data_fetcher.fetch_earnings_calendar(ticker)", self.source)
        self.assertIn('st.session_state.setdefault("stock_analyst_results", {})',
                      self.source)

    def test_chart_remote_calls_are_one_shot_and_results_are_bounded(self):
        remote_calls = {
            "data_fetcher.fetch_chart_history",
            "data_fetcher.fetch_order_book",
            "data_fetcher.fetch_history",
            "data_fetcher.moomoo_status",
        }
        matched = [
            (name, node, ancestors)
            for name, node, ancestors in self.calls
            if name in remote_calls and node.lineno >= self.chart_start_line
        ]
        self.assertTrue(matched)
        for name, node, ancestors in matched:
            gated = (
                "chart_refresh_clicked" in _if_names(ancestors)
                or _has_explicit_button_gate(ancestors)
            )
            self.assertTrue(gated, f"{name} at line {node.lineno}")

        for snippet in (
                'st.session_state["chart_detail_results"]',
                '"support-resistance-htf"',
                "moomoo_settings_fingerprint",
                "chart_feature_key",
                "benchmark_specs",
                "chart_features_loaded",
                "while len(chart_result_store) > 2"):
            self.assertIn(snippet, self.source)
        self.assertNotIn("chart_detail_request", self.source)
        self.assertNotIn("moomoo_client.snapshot.clear()", self.source)

    def test_external_urls_are_validated_before_rendering(self):
        for snippet in (
            "information_board.safe_url(event.get(\"url\"))",
            "information_board.safe_url(n.get(\"url\"))",
            "information_board.safe_url(p.get(\"url\"))",
            "html.escape(post_url, quote=True)",
            'rel="noopener noreferrer"',
        ):
            self.assertIn(snippet, self.source)
        self.assertNotIn('href="{p[\"url\"]}"', self.source)

    def test_external_event_news_and_social_text_use_plain_markdown(self):
        for snippet in (
            "ui.plain_markdown(event['display_name_ja'])",
            "ui.plain_markdown(item)",
            "ui.plain_markdown(source_text)",
            "ui.plain_markdown(title)",
            "ui.plain_markdown(error)",
        ):
            self.assertIn(snippet, self.source)
        self.assertGreaterEqual(self.source.count("ui.plain_markdown("), 10)
        self.assertNotIn("md_escape", self.source)


if __name__ == "__main__":
    unittest.main()
