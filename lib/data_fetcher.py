"""yfinanceとmoomoo OpenAPIからのデータ取得。

すべての取得関数を st.cache_data でキャッシュし、同じ銘柄への
重複リクエストを避ける。チャート・最新価格・板情報はmoomooを優先し、
OpenDや権限に問題がある場合はyfinanceへ自動フォールバックする。
"""

import pandas as pd
import streamlit as st
import yfinance as yf

from lib import moomoo_fetcher


class FetchError(Exception):
    """ネットワークエラーやレート制限などによる取得失敗。"""


@st.cache_data(ttl=900, show_spinner="株価データを取得中...")
def fetch_history(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """株価履歴を取得する(日足/週足/月足)。無効なティッカーの場合は空のDataFrame。"""
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    except Exception as e:
        raise FetchError(str(e)) from e
    if df is None or df.empty:
        return pd.DataFrame()
    return df


@st.cache_data(ttl=300, show_spinner="67銘柄の5分足を取得中...")
def fetch_intraday_batch(tickers: tuple[str, ...], period: str = "1d",
                         interval: str = "5m") -> dict[str, pd.DataFrame]:
    """複数銘柄の分足をまとめて取得する(1銘柄ずつだとレート制限に当たるため)。

    取得できなかった銘柄はキーごと含めない。呼び出し側で欠損を検査する。
    """
    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period=period, interval=interval,
                          group_by="ticker", auto_adjust=False, progress=False,
                          threads=True)
    except Exception as e:
        raise FetchError(str(e)) from e
    if raw is None or raw.empty:
        return {}

    cols = ["Open", "High", "Low", "Close", "Volume"]
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        if not isinstance(sub, pd.DataFrame) or sub.empty:
            continue
        if not all(c in sub.columns for c in cols):
            continue
        df = sub[cols].dropna(subset=["Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if not df.empty:
            out[t] = df
    return out


def _date_keys(index: pd.Index) -> pd.Index:
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    return pd.Index(dates.date)


def _merge_corporate_actions(primary: pd.DataFrame, yahoo: pd.DataFrame,
                             interval: str) -> pd.DataFrame:
    """moomooのOHLCVにYahooの配当・分割イベントを付加する。"""
    out = primary.copy()
    for column in ("Dividends", "Stock Splits"):
        out[column] = 0.0
    if yahoo.empty or interval not in {"1d", "1wk", "1mo"}:
        return out

    yahoo_keys = _date_keys(yahoo.index)
    primary_keys = _date_keys(out.index)
    for column in ("Dividends", "Stock Splits"):
        if column not in yahoo.columns:
            continue
        values = pd.Series(
            pd.to_numeric(yahoo[column], errors="coerce").fillna(0).to_numpy(),
            index=yahoo_keys,
        ).groupby(level=0).sum()
        out[column] = [float(values.get(key, 0.0)) for key in primary_keys]
    return out


@st.cache_data(ttl=30, show_spinner=False)
def moomoo_status() -> dict:
    """moomoo SDK/OpenDの接続状態。"""
    return moomoo_fetcher.connection_status()


@st.cache_data(ttl=30, show_spinner=False)
def fetch_realtime_snapshot(ticker: str) -> dict:
    """moomooの最新価格。利用できなければフォールバック理由のみ返す。"""
    try:
        snapshot = moomoo_fetcher.fetch_snapshot(ticker)
    except moomoo_fetcher.MoomooError as exc:
        return {"source": "Yahoo Finance", "fallback_reason": str(exc)}
    if not snapshot or not snapshot.get("price"):
        return {"source": "Yahoo Finance",
                "fallback_reason": "moomooから最新価格を取得できませんでした"}
    return snapshot


@st.cache_data(ttl=120, show_spinner="チャートデータを取得中...")
def fetch_chart_history(ticker: str, period: str,
                        interval: str = "1d") -> tuple[pd.DataFrame, dict]:
    """チャート用OHLCVをmoomoo優先で取得し、取得元メタデータも返す。"""
    fallback_reason = None
    try:
        moomoo = moomoo_fetcher.fetch_history(ticker, period, interval)
    except moomoo_fetcher.MoomooError as exc:
        moomoo = pd.DataFrame()
        fallback_reason = str(exc)

    if not moomoo.empty:
        try:
            yahoo = fetch_history(ticker, period, interval)
        except FetchError:
            yahoo = pd.DataFrame()
        enriched = _merge_corporate_actions(moomoo, yahoo, interval)
        return enriched, {
            "source": "moomoo OpenAPI",
            "code": moomoo_fetcher.normalize_code(ticker),
            "fallback_reason": None,
        }

    yahoo = fetch_history(ticker, period, interval)
    return yahoo, {
        "source": "Yahoo Finance",
        "code": ticker,
        "fallback_reason": fallback_reason or "moomooからデータを取得できませんでした",
    }


@st.cache_data(ttl=10, show_spinner="moomooの板情報を取得中...")
def fetch_order_book(ticker: str, num: int = 10) -> pd.DataFrame:
    """moomooの板情報を読み取り専用で取得する。"""
    try:
        return moomoo_fetcher.fetch_order_book(ticker, num)
    except moomoo_fetcher.MoomooError as exc:
        raise FetchError(str(exc)) from exc


@st.cache_data(ttl=10, show_spinner=False)
def fetch_recent_ticks(ticker: str, num: int = 60) -> pd.DataFrame:
    """支持線・抵抗線の再評価用に直近の歩み値を取得する。"""
    try:
        return moomoo_fetcher.fetch_recent_ticks(ticker, num)
    except moomoo_fetcher.MoomooError as exc:
        raise FetchError(str(exc)) from exc


@st.cache_data(ttl=30, show_spinner=False)
def fetch_capital_distribution(ticker: str) -> dict | None:
    """支持線・抵抗線の再評価用に当日の資金フローを取得する。"""
    try:
        return moomoo_fetcher.fetch_capital_distribution(ticker)
    except moomoo_fetcher.MoomooError as exc:
        raise FetchError(str(exc)) from exc


@st.cache_data(ttl=900, show_spinner="銘柄情報を取得中...")
def fetch_info(ticker: str) -> dict:
    """企業情報・ファンダメンタル指標を取得する。

    無効なティッカーの場合は空のdictを返す。
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        raise FetchError(str(e)) from e

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    name = info.get("longName") or info.get("shortName")
    if price is None and name is None:
        return {}

    return {
        "name": name or ticker,
        "price": price,
        "previous_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "per": info.get("trailingPE"),
        "pbr": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "dividend_rate": info.get("dividendRate"),
        "dividend_yield": info.get("dividendYield"),
        "market_cap": info.get("marketCap"),
    }


@st.cache_data(ttl=86400, show_spinner="財務データを取得中...")
def fetch_annual_financials(ticker: str) -> pd.DataFrame:
    """年次の売上高とEPS(直近4年分)を取得する。

    戻り値: index=会計年度(西暦)、columns=["revenue", "eps"](古い年→新しい年の順)。
    取得できない場合は空のDataFrame。
    """
    try:
        stmt = yf.Ticker(ticker).income_stmt
    except Exception as e:
        raise FetchError(str(e)) from e

    if stmt is None or stmt.empty:
        return pd.DataFrame()

    def row(*names):
        for n in names:
            if n in stmt.index:
                return stmt.loc[n]
        return None

    revenue = row("Total Revenue")
    eps = row("Basic EPS", "Diluted EPS")
    if revenue is None and eps is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "revenue": revenue if revenue is not None else pd.NA,
        "eps": eps if eps is not None else pd.NA,
    })
    out.index = pd.to_datetime(out.index).year
    out = out[~out.index.duplicated()].sort_index()
    return out.tail(4)


@st.cache_data(ttl=21600, show_spinner="アナリスト情報を取得中...")
def fetch_analyst(ticker: str) -> dict:
    """アナリストの目標株価・レーティング・格付け変更・次回決算日を取得する。

    各項目は取得できなければNone(全滅でも例外にしない)。
    """
    t = yf.Ticker(ticker)
    out = {"targets": None, "ratings": None, "changes": [],
           "earnings_date": None, "eps_estimate": None}

    try:
        pt = t.analyst_price_targets
        if pt and pt.get("mean"):
            out["targets"] = {k: pt.get(k) for k in ("mean", "median", "high", "low")}
    except Exception:
        pass

    try:
        rec = t.recommendations
        if rec is not None and not rec.empty:
            row = rec.iloc[0]
            out["ratings"] = {k: int(row[k]) for k in
                              ("strongBuy", "buy", "hold", "sell", "strongSell")}
    except Exception:
        pass

    try:
        ud = t.upgrades_downgrades
        if ud is not None and not ud.empty:
            for idx, row in ud.head(6).iterrows():
                out["changes"].append({
                    "date": str(idx)[:10],
                    "firm": row.get("Firm") or "",
                    "grade": row.get("ToGrade") or "",
                    "action": row.get("Action") or "",
                    "target": float(row["currentPriceTarget"])
                              if pd.notna(row.get("currentPriceTarget")) else None,
                })
    except Exception:
        pass

    try:
        cal = t.calendar or {}
        dates = cal.get("Earnings Date") or []
        if dates:
            out["earnings_date"] = str(dates[0])
        if cal.get("Earnings Average") is not None:
            out["eps_estimate"] = float(cal["Earnings Average"])
    except Exception:
        pass

    return out


@st.cache_data(ttl=900, show_spinner="ランキングを取得中...")
def fetch_screener(kind: str, count: int = 10) -> pd.DataFrame:
    """Yahooスクリーナー(day_gainers / day_losers / most_actives)を取得する。"""
    try:
        r = yf.screen(kind, count=count)
    except Exception as e:
        raise FetchError(str(e)) from e
    quotes = (r or {}).get("quotes", [])
    return pd.DataFrame([{
        "ティッカー": q.get("symbol"),
        "銘柄名": q.get("shortName") or q.get("longName") or "",
        "株価": q.get("regularMarketPrice"),
        "前日比": q.get("regularMarketChangePercent"),
        "出来高": q.get("regularMarketVolume"),
    } for q in quotes])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_earnings_history(ticker: str) -> list[str]:
    """過去の決算発表日(YYYY-MM-DD)の一覧。取得できなければ空リスト。"""
    try:
        ed = yf.Ticker(ticker).earnings_dates
    except Exception:
        return []
    if ed is None or ed.empty:
        return []
    today = pd.Timestamp.now(tz=ed.index.tz) if ed.index.tz else pd.Timestamp.now()
    past = [str(d)[:10] for d in ed.index if d < today]
    return past[:12]


def dividend_yield_percent(info: dict) -> float | None:
    """配当利回りを%値で返す。

    値の単位が曖昧にならないよう、配当額/株価から計算できる場合はそちらを優先。
    dividendYieldへのフォールバックは%値として扱う(yfinance 0.2.55以降の仕様。
    実測: AAPL = 0.34 → 0.34%)。
    """
    rate = info.get("dividend_rate")
    price = info.get("price")
    if rate and price:
        return rate / price * 100
    return info.get("dividend_yield")
