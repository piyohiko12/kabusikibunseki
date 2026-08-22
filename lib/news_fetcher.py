"""ニュース(Yahoo Finance)とネットの反応(Stocktwits / Hacker News / Mastodon)の取得。

投稿は共通形式のdictに正規化する:
  {source, user, body, created_at("YYYY-MM-DD HH:MM" UTC), sentiment, url, likes}
センチメントラベル(Bullish/Bearish)を持つのはStocktwitsのみ。
"""

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from email.utils import parsedate_to_datetime

import streamlit as st
import yfinance as yf

from lib.data_fetcher import FetchError

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) stock-analyzer/1.0"}

_CORPORATE_WORDS = {
    "inc", "incorporated", "corp", "corporation", "company", "co", "ltd",
    "limited", "plc", "holdings", "holding", "group", "sa", "ag", "nv",
}
_KNOWN_NEWS_ALIASES = {
    "GOOG": ("google",), "GOOGL": ("google",),
    "META": ("facebook",), "BRK-B": ("berkshire",),
}


def _sec_user_agent() -> dict:
    """SEC EDGAR用のUser-Agent。

    SECは公正アクセスポリシーでUser-Agentへの連絡先明記を求めている。
    メールアドレスは公開リポジトリに含めないため、環境変数
    SEC_CONTACT か data/settings.json の "sec_contact" から読み込む
    (未設定ならプロジェクトURLを連絡先として送る)。
    """
    contact = os.environ.get("SEC_CONTACT")
    if not contact:
        try:
            from lib import settings_store
            contact = settings_store.load().get("sec_contact")
        except Exception:
            contact = None
    contact = contact or "https://github.com/piyohiko12/kabusikibunseki"
    return {"User-Agent": f"stock-analyzer/1.0 (personal use; contact: {contact})"}

SOCIAL_SOURCES = ["Stocktwits", "Hacker News", "Mastodon"]

SEC_FORM_LABELS = {
    "10-K": "年次報告書",
    "10-Q": "四半期報告書",
    "8-K": "臨時報告書(重要イベント)",
    "DEF 14A": "委任状説明書(株主総会)",
    "20-F": "年次報告書(外国企業)",
    "6-K": "臨時報告書(外国企業)",
    "S-1": "証券届出書(新規発行)",
    "SC 13D": "大量保有報告",
    "SC 13G": "大量保有報告(簡易)",
    "SD": "特定開示報告",
}


def _get_json(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", text or "")).strip()


def _normalized_news_symbol(value) -> str:
    if isinstance(value, dict):
        value = value.get("symbol") or value.get("ticker") or value.get("code")
    text = str(value or "").strip().upper()
    if text.startswith("US."):
        text = text[3:]
    return text.replace(".", "-")


def _tagged_news_symbols(item: dict, content: dict) -> set[str]:
    """Yahooが記事へ明示した関連銘柄だけを正規化して返す。"""
    tagged = []

    def add(values) -> None:
        if isinstance(values, (str, dict)):
            tagged.append(values)
        elif isinstance(values, (list, tuple, set)):
            tagged.extend(values)

    finance = content.get("finance")
    if isinstance(finance, dict):
        add(finance.get("stockTickers"))
        add(finance.get("tickers"))
    for holder in (item, content):
        add(holder.get("relatedTickers"))
        add(holder.get("stockTickers"))
    return {symbol for symbol in map(_normalized_news_symbol, tagged) if symbol}


def _news_aliases(ticker: str, company_name: str | None) -> set[str]:
    symbol = _normalized_news_symbol(ticker)
    aliases = {symbol, symbol.replace("-", "."), symbol.replace("-", "")}
    aliases.update(_KNOWN_NEWS_ALIASES.get(symbol, ()))
    words = re.findall(r"[a-z0-9]+", str(company_name or "").casefold())
    meaningful = [word for word in words if word not in _CORPORATE_WORDS]
    if meaningful:
        phrase = " ".join(meaningful)
        aliases.add(phrase)
        aliases.update(word for word in meaningful if len(word) >= 4)
    return {alias.casefold() for alias in aliases if len(alias) >= 2}


def _mentions_company(text: str, ticker: str, company_name: str | None) -> bool:
    searchable = str(text or "").casefold()
    for alias in _news_aliases(ticker, company_name):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", searchable):
            return True
    return False


@st.cache_data(ttl=900, show_spinner="ニュースを取得中...")
def fetch_news(ticker: str, company_name: str | None = None) -> list[dict]:
    """最新ニュースを最大10件取得する。取得できない場合は空リスト。"""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        raise FetchError(str(e)) from e

    news = []
    for item in items[:10]:
        c = item.get("content") or {}
        title = c.get("title")
        if not title:
            continue
        tagged_symbols = _tagged_news_symbols(item, c)
        if (tagged_symbols
                and _normalized_news_symbol(ticker) not in tagged_symbols):
            # Yahooのおすすめ記事が銘柄ニュースへ混ざる場合があるため、
            # 明示タグが別銘柄だけの記事は表示・イベント化しない。
            continue
        article_text = " ".join((title, c.get("summary") or "",
                                 c.get("description") or ""))
        if (not tagged_symbols and company_name
                and not _mentions_company(article_text, ticker, company_name)):
            continue
        news.append({
            "title": title,
            "summary": c.get("summary") or c.get("description") or "",
            "pub_date": (c.get("pubDate") or "")[:16].replace("T", " "),
            "provider": (c.get("provider") or {}).get("displayName") or "",
            "url": ((c.get("canonicalUrl") or {}).get("url")
                    or (c.get("clickThroughUrl") or {}).get("url") or ""),
        })
    return news


@st.cache_data(ttl=900, show_spinner="ニュースを取得中...")
def fetch_google_news(ticker: str, lang: str = "en") -> list[dict]:
    """Google News RSSからニュースを取得する(lang: "en" または "ja")。"""
    if lang == "ja":
        query = urllib.parse.quote(f"{ticker} 株")
        params = "hl=ja&gl=JP&ceid=JP:ja"
    else:
        query = urllib.parse.quote(f"{ticker} stock")
        params = "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={query}&{params}"
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            root = ET.fromstring(r.read())
    except Exception as e:
        raise FetchError(str(e)) from e

    news = []
    for item in root.findall(".//item")[:12]:
        title = item.findtext("title") or ""
        source = item.findtext("source") or ""
        # タイトル末尾の「 - 媒体名」を除去
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        try:
            dt = parsedate_to_datetime(item.findtext("pubDate") or "")
            pub = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pub = ""
        news.append({
            "title": title,
            "summary": "",
            "pub_date": pub,
            "provider": source,
            "url": item.findtext("link") or "",
        })
    return news


@st.cache_data(ttl=604800, show_spinner=False)
def _sec_cik_map() -> dict:
    """ティッカー → CIK(SEC企業番号)の対応表。週1回更新で十分。"""
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json",
                                 headers=_sec_user_agent())
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    return {v["ticker"]: v["cik_str"] for v in data.values()}


@st.cache_data(ttl=21600, show_spinner="SEC開示情報を取得中...")
def fetch_sec_filings(ticker: str) -> list[dict]:
    """SEC EDGARから主要な開示書類(直近10件)を取得する。米国上場企業のみ。"""
    try:
        cik = _sec_cik_map().get(ticker)
    except Exception as e:
        raise FetchError(str(e)) from e
    if cik is None:
        return []
    try:
        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        req = urllib.request.Request(url, headers=_sec_user_agent())
        with urllib.request.urlopen(req, timeout=30) as r:
            recent = json.loads(r.read().decode("utf-8"))["filings"]["recent"]
    except Exception as e:
        raise FetchError(str(e)) from e

    filings = []
    for form, date, acc, doc in zip(recent["form"], recent["filingDate"],
                                    recent["accessionNumber"],
                                    recent["primaryDocument"]):
        if form not in SEC_FORM_LABELS:
            continue
        filings.append({
            "title": f"{form}: {SEC_FORM_LABELS[form]}",
            "summary": "",
            "pub_date": f"{date} 00:00",
            "provider": "SEC EDGAR",
            "url": (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{acc.replace('-', '')}/{doc}"),
        })
        if len(filings) >= 10:
            break
    return filings


def trending_terms(posts: list[dict], exclude: str) -> list[tuple[str, int]]:
    """投稿中のカシュタグ($XXX)・ハッシュタグを集計する(1投稿1カウント)。"""
    pat = re.compile(r"[$#]([A-Za-z]{1,10})\b")
    counter: Counter = Counter()
    for p in posts:
        for term in {m.upper() for m in pat.findall(p["body"])}:
            if term != exclude.upper():
                counter[term] += 1
    return counter.most_common(8)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stocktwits(ticker: str) -> list[dict]:
    """Stocktwits(投資家SNS)のシンボルストリーム。無効シンボルは空リスト。"""
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise FetchError(f"Stocktwits HTTP {e.code}") from e
    except Exception as e:
        raise FetchError(str(e)) from e

    posts = []
    for m in data.get("messages", []):
        user = (m.get("user") or {}).get("username") or ""
        posts.append({
            "source": "Stocktwits",
            "user": user,
            "body": m.get("body") or "",
            "created_at": (m.get("created_at") or "")[:16].replace("T", " "),
            "sentiment": ((m.get("entities") or {}).get("sentiment") or {}).get("basic"),
            "url": f"https://stocktwits.com/{user}/message/{m.get('id')}" if user else "",
            "likes": (m.get("likes") or {}).get("total", 0),
        })
    return posts


@st.cache_data(ttl=900, show_spinner=False)
def fetch_hackernews(ticker: str) -> list[dict]:
    """Hacker News(Algolia検索API)でティッカーに言及した投稿・コメントを取得。

    Algoliaはタイポ許容で無関係な結果を返すため(例: AAPL→apples)、
    ティッカーの完全一致(大文字・単語境界)でフィルタする。
    """
    url = (f"https://hn.algolia.com/api/v1/search_by_date?query=%22{ticker}%22"
           f"&tags=(story,comment)&hitsPerPage=30")
    try:
        data = _get_json(url)
    except Exception as e:
        raise FetchError(str(e)) from e

    exact = re.compile(rf"\b{re.escape(ticker)}\b")
    posts = []
    for h in data.get("hits", []):
        raw = f"{h.get('title') or ''} {h.get('comment_text') or ''} {h.get('story_title') or ''}"
        if not exact.search(raw):
            continue
        body = h.get("title") or _strip_html(h.get("comment_text") or "")
        if not body:
            continue
        if h.get("comment_text") and h.get("story_title"):
            body = f"{body[:400]}\n\n(記事: {h['story_title']})"
        posts.append({
            "source": "Hacker News",
            "user": h.get("author") or "",
            "body": body,
            "created_at": (h.get("created_at") or "")[:16].replace("T", " "),
            "sentiment": None,
            "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
            "likes": h.get("points") or 0,
        })
    return posts


@st.cache_data(ttl=900, show_spinner=False)
def fetch_mastodon(ticker: str) -> list[dict]:
    """Mastodon(mastodon.social)のハッシュタグタイムラインを取得。"""
    url = f"https://mastodon.social/api/v1/timelines/tag/{ticker}?limit=15"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise FetchError(f"Mastodon HTTP {e.code}") from e
    except Exception as e:
        raise FetchError(str(e)) from e

    posts = []
    for s in data:
        body = _strip_html(s.get("content") or "")
        if not body:
            continue
        posts.append({
            "source": "Mastodon",
            "user": (s.get("account") or {}).get("acct") or "",
            "body": body[:500],
            "created_at": (s.get("created_at") or "")[:16].replace("T", " "),
            "sentiment": None,
            "url": s.get("url") or "",
            "likes": s.get("favourites_count") or 0,
        })
    return posts


def fetch_social(ticker: str) -> dict:
    """全ソースの投稿を集約する。個別ソースの失敗は errors に記録して続行。

    戻り値: {posts, bullish, bearish, total, errors}
    (bullish/bearishはセンチメントラベルを持つStocktwits投稿の集計)
    """
    posts: list[dict] = []
    errors: list[str] = []
    for name, fetcher in (("Stocktwits", fetch_stocktwits),
                          ("Hacker News", fetch_hackernews),
                          ("Mastodon", fetch_mastodon)):
        try:
            posts.extend(fetcher(ticker))
        except FetchError:
            errors.append(name)
    posts.sort(key=lambda p: p["created_at"], reverse=True)
    bullish = sum(1 for p in posts if p["sentiment"] == "Bullish")
    bearish = sum(1 for p in posts if p["sentiment"] == "Bearish")
    return {"posts": posts, "bullish": bullish, "bearish": bearish,
            "total": len(posts), "errors": errors}
