import unittest
from unittest import mock

from lib import news_fetcher


def _item(title, symbols=None):
    content = {
        "title": title,
        "summary": "summary",
        "pubDate": "2026-08-22T12:00:00Z",
        "provider": {"displayName": "Test"},
        "canonicalUrl": {"url": "https://example.com/article"},
    }
    if symbols is not None:
        content["finance"] = {"stockTickers": symbols}
    return {"content": content}


class YahooNewsRelevanceTests(unittest.TestCase):
    def setUp(self):
        news_fetcher.fetch_news.clear()

    def tearDown(self):
        news_fetcher.fetch_news.clear()

    def test_explicitly_unrelated_recommendations_are_removed(self):
        rows = [
            _item("Apple update", ["AAPL"]),
            _item("Unrelated sports tax story", ["BABA", "DIS"]),
            _item("Untagged provider story", None),
        ]
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = rows
            result = news_fetcher.fetch_news("AAPL", "Apple Inc.")
        self.assertEqual(
            [row["title"] for row in result],
            ["Apple update"],
        )

    def test_symbol_normalization_accepts_us_prefix_and_share_class(self):
        item = {"relatedTickers": [{"symbol": "US.BRK-B"}],
                "content": _item("Berkshire update")["content"]}
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = [item]
            result = news_fetcher.fetch_news("BRK.B")
        self.assertEqual(len(result), 1)

    def test_untagged_recommendations_need_company_or_ticker_mention(self):
        rows = [
            _item("Apple launches a new product", None),
            _item("Anthropic plans a market debut", None),
            _item("AAPL valuation update", None),
        ]
        with mock.patch.object(news_fetcher.yf, "Ticker") as ticker_class:
            ticker_class.return_value.news = rows
            result = news_fetcher.fetch_news("AAPL", "Apple Inc.")
        self.assertEqual(
            [row["title"] for row in result],
            ["Apple launches a new product", "AAPL valuation update"],
        )


if __name__ == "__main__":
    unittest.main()
