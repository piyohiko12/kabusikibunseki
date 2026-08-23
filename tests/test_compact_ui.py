"""情報密度を保ちながら初期表示を簡潔にするUI契約。"""

from pathlib import Path
import unittest

from lib import ui


ROOT = Path(__file__).parents[1]


class CompactUiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (ROOT / "app.py").read_text(encoding="utf-8")
        cls.stock = (ROOT / "views" / "stock_analysis.py").read_text(
            encoding="utf-8")
        cls.board = (ROOT / "lib" / "board_ui.py").read_text(encoding="utf-8")

    def test_shared_css_reduces_headings_metrics_and_mobile_padding(self):
        for selector in (
                '[data-testid="stAppViewContainer"] h1',
                '[data-testid="stMetricValue"]',
                '[data-testid="stAlert"]',
                '@media (max-width: 700px)'):
            self.assertIn(selector, self.app)
        self.assertIn("font-size: clamp(1.08rem", self.app)
        self.assertIn("padding: 4rem .78rem 1.5rem", self.app)
        self.assertIn('[data-testid="stMainBlockContainer"]', self.app)
        self.assertIn('[data-testid="stTab"]', self.app)

    def test_secondary_purchase_controls_start_collapsed(self):
        for label in (
                "寄付き・イベント診断を更新",
                "購入株数を計算",
                "購入を中止する条件",
                "すでに株を持っている場合",
                "判定の理由・相場の状態・支持抵抗"):
            self.assertIn(f'st.expander("{label}"', self.stock)
        self.assertIn('with st.expander(f"{trend_scope}・支持抵抗を見る")', self.stock)
        self.assertIn('"立会の方向" if current in', self.stock)
        self.assertNotIn('st.markdown("#### 3. 購入株数の上限を計算")', self.stock)
        self.assertNotIn('st.markdown("#### 当日の方向と重要価格帯")', self.stock)

    def test_wait_reasons_are_deduplicated_before_rendering(self):
        self.assertIn("seen_wait_reasons = set()", self.stock)
        self.assertIn("if semantic_key in seen_wait_reasons", self.stock)
        self.assertIn("main_wait_reasons = [", self.stock)
        self.assertIn('purchase_plan.get("description_ja")', self.stock)
        self.assertIn("主な理由:", self.stock)
        self.assertNotIn("for reason in wait_reasons[:2]", self.stock)

    def test_primary_alerts_do_not_embed_large_markdown_headings(self):
        self.assertNotIn('f"### {purchase_icon}', self.stock)
        self.assertNotIn('f"### {visual[', self.stock)
        self.assertNotIn('f"### {holding_visual[', self.stock)
        self.assertNotIn('f"### {visual[', self.board)

    def test_information_board_does_not_repeat_large_card_headings(self):
        self.assertIn('st.markdown("**🎯 売買の目安**")', self.board)
        self.assertNotIn('st.markdown("### 🎯 売買の目安")', self.board)
        self.assertNotIn('st.markdown(f"#### {_plain_markdown', self.board)
        self.assertIn('if label in {"最重要", "重要"}', self.board)

    def test_initial_controls_and_optional_monitor_are_collapsed(self):
        self.assertIn('with st.popover("⚙️ 表示設定")', self.stock)
        self.assertIn('realtime_panel_label = (', self.stock)
        self.assertIn('expanded=bool(st.session_state[realtime_monitor_key])', self.stock)
        self.assertNotIn('st.markdown("**⚡ リアルタイム売買タイミング**")', self.stock)

    def test_stock_tabs_are_short_and_related_detail_is_grouped(self):
        self.assertIn(
            '"今日", "チャート", "ニュース", "情報一覧", "板・需給", "関連市場",',
            self.stock)
        self.assertIn("tab_tape = tab_flow = tab_orderflow", self.stock)
        self.assertIn('"表示する情報", ["板・歩み値", "需給・IV"]', self.stock)
        self.assertIn('"選んだ情報を読み込む"', self.stock)
        self.assertIn("if (orderflow_load and", self.stock)
        self.assertIn('orderflow_section == "板・歩み値"', self.stock)
        self.assertIn('orderflow_section == "需給・IV"', self.stock)
        self.assertNotIn('"🔬 板・歩み値", "🏦 需給・IV"', self.stock)

    def test_html_chips_escape_dynamic_labels(self):
        rendered = ui.chip('</span><script>alert("x")</script>', "blue")
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)

    def test_plain_markdown_disables_external_links_images_and_html(self):
        rendered = ui.plain_markdown(
            '[公式](https://phishing.invalid) ![追跡](https://tracker.invalid/p.gif) '
            '<script>alert("x")</script>')
        self.assertNotIn("[公式](", rendered)
        self.assertNotIn("![追跡](", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("https://", rendered)
        self.assertIn(r"\[公式\]\(", rendered)
        self.assertIn(r"\!\[追跡\]\(", rendered)
        self.assertIn(r"\<script\>", rendered)

    def test_declared_streamlit_minimum_uses_compatible_width_arguments(self):
        self.assertNotIn('width="stretch"', self.stock)
        self.assertIn('use_container_width=True', self.stock)

    def test_compact_grid_escapes_external_text_and_is_accessible(self):
        rendered = ui.compact_kpi_grid([
            ("<銘柄>", "<script>alert(1)</script>", "A&B"),
        ])
        self.assertIn('role="list"', rendered)
        self.assertIn('role="listitem"', rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("A&amp;B", rendered)
        self.assertNotIn("<script>", rendered)

    def test_main_price_groups_use_compact_grid_not_four_metric_cards(self):
        self.assertGreaterEqual(self.stock.count("ui.compact_kpi_grid("), 3)
        self.assertIn("grid-template-columns: repeat(auto-fit", self.app)
        self.assertIn("grid-template-columns: repeat(3", self.app)

    def test_chart_keeps_only_preset_and_two_compact_actions_visible(self):
        self.assertIn("c_preset, c_cfg, c_refresh = st.columns", self.stock)
        self.assertIn('with st.popover("⚙️ 設定"', self.stock)
        self.assertNotIn("c_preset, c_type, c_interval, c_action", self.stock)


if __name__ == "__main__":
    unittest.main()
