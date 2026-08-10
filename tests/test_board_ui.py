import inspect
from pathlib import Path
import unittest

from lib import board_ui


class InformationBoardUiTests(unittest.TestCase):
    def test_external_text_is_escaped(self):
        escaped = board_ui._plain_markdown(
            "<script>alert(1)</script>\n![画像](https://example.com/a.png)")
        self.assertNotIn("<script>", escaped)
        self.assertNotIn("![画像]", escaped)
        self.assertIn(r"\<script\>", escaped)

    def test_time_is_shown_in_japan_time(self):
        self.assertIn(
            "2026/08/10 09:00 JST",
            board_ui._format_time("2026-08-10T00:00:00+00:00"))

    def test_date_only_does_not_invent_a_time(self):
        self.assertEqual(
            board_ui._format_time("2026-10-30"),
            "2026/10/30（時刻未定）",
        )

    def test_filter_and_sort_are_deterministic(self):
        items = [
            {"id": "b", "category_label_ja": "ニュース", "importance": "low",
             "importance_label_ja": "低", "importance_rank": 1,
             "title_ja": "通常", "occurred_at": "2026-08-10T01:00:00+00:00"},
            {"id": "a", "category_label_ja": "アラート", "importance": "high",
             "importance_label_ja": "高", "importance_rank": 3,
             "title_ja": "成立", "occurred_at": "2026-08-10T00:00:00+00:00"},
        ]
        filtered = board_ui._filter_items(items, ["アラート"], ["高"], "成立")
        self.assertEqual([item["id"] for item in filtered], ["a"])
        items.sort(key=lambda item: board_ui._item_sort_key(item, "importance"))
        self.assertEqual([item["id"] for item in items], ["a", "b"])

    def test_trade_overview_keeps_entry_and_holding_actions_visible(self):
        items = [
            {
                "category": "trade", "kind": "holding_verdict",
                "summary_ja": "損失管理条件が成立",
                "data": {"position_mode": "holding", "verdict": "RISK_EXIT"},
            },
            {
                "category": "trade", "kind": "entry_verdict",
                "summary_ja": "買い条件が成立",
                "data": {"position_mode": "entry", "verdict": "BUY"},
            },
        ]
        rows = board_ui._trade_overview_rows(items)
        self.assertEqual([row["mode"] for row in rows], ["entry", "holding"])
        self.assertEqual(rows[0]["heading_ja"], "未保有なら")
        self.assertIn("買い", rows[0]["visual"]["action_label_ja"])
        self.assertEqual(rows[1]["heading_ja"], "ロング保有中なら")
        self.assertIn("売却", rows[1]["visual"]["action_label_ja"])
        self.assertIn("新規の空売り", rows[1]["visual"]["description_ja"])

    def test_trade_overview_ignores_trade_words_in_news_or_analyst_items(self):
        items = [
            {
                "category": "news", "kind": "entry_verdict",
                "data": {"position_mode": "entry", "verdict": "BUY"},
            },
            {
                "category": "analyst", "kind": "holding_verdict",
                "data": {"position_mode": "holding", "verdict": "RISK_EXIT"},
            },
        ]
        self.assertEqual(board_ui._trade_overview_rows(items), [])

    def test_trade_overview_safely_rejects_mode_verdict_mismatch(self):
        rows = board_ui._trade_overview_rows([{
            "category": "trade", "kind": "entry_verdict",
            "data": {"position_mode": "entry", "verdict": "RISK_EXIT"},
        }])
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["visual"]["available"])
        self.assertEqual(
            rows[0]["visual"]["action_label_ja"],
            "【判定不能】売買せず待機",
        )

    def test_ui_has_no_posting_storage_or_market_api(self):
        source = inspect.getsource(board_ui)
        for forbidden in (
            "create_post", "create_reply", "sqlite3", "data_fetcher",
            "fetch_chart_history", "place_order",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_standalone_page_uses_yahoo_history_without_mutations(self):
        page = (Path(__file__).resolve().parents[1] / "views" / "board.py").read_text(
            encoding="utf-8")
        self.assertIn('fetch_history(ticker, "2y", "1d")', page)
        for forbidden in (
            "fetch_chart_history", "alerts_lib.save", "alerts_lib.new_alert",
            "settings_store.save", "place_order", "sqlite3",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, page)


if __name__ == "__main__":
    unittest.main()
