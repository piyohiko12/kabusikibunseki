"""情報密度を保ちながら初期表示を簡潔にするUI契約。"""

from pathlib import Path
import unittest


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
        self.assertIn("padding: .72rem .78rem 1.5rem", self.app)

    def test_secondary_purchase_controls_start_collapsed(self):
        for label in (
                "寄付き・イベント診断を更新",
                "購入株数を計算",
                "購入を中止する条件",
                "すでに株を持っている場合",
                "判定の理由・相場の状態・支持抵抗",
                "当日の方向・支持抵抗を見る"):
            self.assertIn(f'st.expander("{label}"', self.stock)
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


if __name__ == "__main__":
    unittest.main()
