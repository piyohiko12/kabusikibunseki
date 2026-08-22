"""yfinanceとmoomoo OpenAPIからのデータ取得。

すべての取得関数を st.cache_data でキャッシュし、同じ銘柄への
重複リクエストを避ける。チャート・最新価格・板情報はmoomooを優先し、
OpenDや権限に問題がある場合はyfinanceへ自動フォールバックする。
"""

import math

import pandas as pd
import streamlit as st
import yfinance as yf

from lib import moomoo_client, moomoo_fetcher, session_intelligence


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


def _last_history_time(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame is None or frame.empty:
        return None
    try:
        stamp = pd.Timestamp(frame.index[-1])
    except (TypeError, ValueError):
        return None
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


@st.cache_data(ttl=30, show_spinner=False)
def moomoo_status() -> dict:
    """moomoo SDK/OpenDの接続状態。"""
    return moomoo_fetcher.connection_status()


def fetch_realtime_snapshot(ticker: str) -> dict:
    """moomooの最新価格。利用できなければフォールバック理由のみ返す。

    取得はmoomoo_client.snapshotに一本化している(そちらが5秒キャッシュを持つ)。
    """
    state = moomoo_client.status()
    if state["state"] != moomoo_client._State.OK:
        return {"source": "Yahoo Finance", "session_quotes": {},
                "fallback_reason": state["message"]}
    snap = moomoo_client.snapshot((ticker,)).get(ticker)
    session_quotes = (snap.get("session_quotes")
                      if isinstance(snap, dict) else None)
    has_session_price = bool(
        isinstance(session_quotes, dict)
        and any(isinstance(row, dict) and row.get("available")
                for row in session_quotes.values()))
    if not snap or (not snap.get("price") and not has_session_price):
        return {"source": "Yahoo Finance", "session_quotes": {},
                "fallback_reason": "moomooから最新価格を取得できませんでした"}
    return {**snap, "code": moomoo_client.to_code(ticker) or ticker,
            "source": "moomoo OpenAPI"}


def _decision_number(value, *, positive: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _decision_timestamp(value) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    try:
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(
                "America/New_York", ambiguous="NaT", nonexistent="NaT")
        else:
            stamp = stamp.tz_convert("America/New_York")
    except (TypeError, ValueError):
        return None
    return None if pd.isna(stamp) else stamp


def _decision_session_name(value) -> str:
    return {
        "pre": "premarket", "premarket": "premarket",
        "regular": "regular",
        "after": "afterhours", "afterhours": "afterhours",
        "overnight": "overnight",
    }.get(str(value or "").strip().lower(), "unknown")


def _timestamp_session(stamp: pd.Timestamp) -> str:
    """NYSE定例日・短縮日を考慮した時刻セッション。"""
    try:
        detected = session_intelligence.detect_current_session(
            stamp, overnight_eligible=True)
    except (TypeError, ValueError, OverflowError):
        return "closed"
    return str(detected.get("session") or "closed")


def build_session_snapshot(snapshot: dict | None, decision_quote: dict | None,
                           session: str, *, max_alignment_seconds: float = 90.0,
                           max_price_deviation_pct: float = 1.0,
                           max_bbo_deviation_pct: float = 0.20) -> dict:
    """engine向け気配を対象セッションへ安全に正規化する純粋関数。

    時間外では、snapshot専用価格、K_1Mのセッション付きbar時刻、同一snapshot
    recordのBid/Ask・更新時刻が整合した場合だけtop-level価格を公開する。不明点を
    genericな立会価格で補完せず、失敗時は価格欄をNoneにして判定を止める。
    """
    raw = dict(snapshot or {})
    quote = dict(decision_quote or {})
    session_name = _decision_session_name(session)
    errors: list[str] = []

    result = {
        **raw,
        "raw_snapshot_price": raw.get("price"),
        "raw_snapshot_bid": raw.get("bid"),
        "raw_snapshot_ask": raw.get("ask"),
        "raw_snapshot_update_time": raw.get("update_time"),
        "decision_session": session_name,
        "decision_ready": False,
        "decision_errors": errors,
        "decision_price_source": None,
        "decision_bar_time": quote.get("bar_time") or quote.get("updated_at"),
        "uses_daily_bars": False,
    }

    if str(raw.get("source") or "").strip().lower() != "moomoo openapi":
        errors.append("moomoo OpenAPIのsnapshotではありません")
    if session_name == "unknown":
        errors.append("判定セッションを確認できません")

    bid = _decision_number(raw.get("bid"), positive=True)
    ask = _decision_number(raw.get("ask"), positive=True)
    if bid is None or ask is None or ask < bid:
        errors.append("対象セッションのBid/Askを確認できません")

    session_quotes = (raw.get("session_quotes")
                      if isinstance(raw.get("session_quotes"), dict) else {})
    session_row = (session_quotes.get(session_name)
                   if isinstance(session_quotes.get(session_name), dict) else {})
    snapshot_stamp = _decision_timestamp(
        raw.get("update_time_iso") or session_row.get("snapshot_updated_at")
        or raw.get("update_time"))
    price = None
    price_source = None

    if session_name == "regular":
        price = _decision_number(raw.get("price"), positive=True)
        price_source = "snapshot.last_price"
        if price is None:
            errors.append("立会の現在値を確認できません")
        if snapshot_stamp is None or _timestamp_session(snapshot_stamp) != "regular":
            errors.append("立会のsnapshot更新時刻を確認できません")
    elif session_name in {"premarket", "afterhours", "overnight"}:
        expected_price_field = {
            "premarket": "pre_price", "afterhours": "after_price",
            "overnight": "overnight_price",
        }[session_name]
        price = _decision_number(session_row.get("price"), positive=True)
        price_source = f"snapshot.{session_row.get('price_field') or session_name}"
        if (session_row.get("available") is not True or price is None
                or session_row.get("price_field") != expected_price_field
                or str(session_row.get("source") or "").strip().lower()
                != "moomoo openapi snapshot"
                or session_row.get("uses_daily_bars") is True):
            errors.append("対象セッションの専用価格を確認できません")

        bar_price = _decision_number(quote.get("price"), positive=True)
        bar_stamp = _decision_timestamp(
            quote.get("bar_time") or quote.get("updated_at"))
        quote_session = _decision_session_name(quote.get("session"))
        snapshot_symbol = str(raw.get("code") or "").strip().upper().split(".", 1)[-1]
        quote_symbol = str(quote.get("symbol") or "").strip().upper().split(".", 1)[-1]
        if (quote.get("available") is not True
                or quote.get("timestamp_verified") is not True
                or quote.get("timeframe") != "K_1M"
                or "moomoo openapi" not in str(
                    quote.get("source") or "").strip().lower()
                or quote.get("uses_daily_bars") is True or bar_price is None
                or bar_stamp is None or quote_session != session_name
                or not snapshot_symbol or quote_symbol != snapshot_symbol
                or _timestamp_session(bar_stamp) != session_name):
            errors.append("対象セッションのK_1M価格・時刻を確認できません")
        if snapshot_stamp is None or _timestamp_session(snapshot_stamp) != session_name:
            errors.append("対象セッションのsnapshot更新時刻を確認できません")
        if snapshot_stamp is not None and bar_stamp is not None:
            alignment = abs(float((snapshot_stamp - bar_stamp).total_seconds()))
            if alignment > max(0.0, float(max_alignment_seconds)):
                errors.append("snapshotとK_1Mの更新時刻が離れています")
        if price is not None and bar_price is not None:
            deviation = abs(price / bar_price - 1.0) * 100.0
            if deviation > max(0.0, float(max_price_deviation_pct)):
                errors.append("snapshot専用価格とK_1M価格が一致しません")

    if price is not None and bid is not None and ask is not None:
        # K_1M終値との時間差許容とは分け、執行可能な最良気配(BBO)との乖離は
        # 狭く制限する。古い時間外価格を1%幅で通さないためのfail-closed条件。
        tolerance = max(0.0, float(max_bbo_deviation_pct)) / 100.0
        if price < bid * (1.0 - tolerance) or price > ask * (1.0 + tolerance):
            errors.append("対象セッション価格とBid/Askが整合しません")

    if errors:
        return {
            **result, "price": None, "bid": None, "ask": None,
            "update_time": None, "decision_price_source": price_source,
        }
    return {
        **result, "price": price, "bid": bid, "ask": ask,
        "update_time": snapshot_stamp.isoformat(),
        "decision_ready": True, "decision_price_source": price_source,
        "decision_errors": [],
    }


@st.cache_data(ttl=3, show_spinner=False)
def _fetch_current_klines_cached(symbols: tuple[str, ...], num: int,
                                 session: str) -> dict:
    """同じリアルタイム1分足の重複取得を短時間だけまとめる。"""
    return moomoo_client.current_klines(symbols, num=num, session=session)


def fetch_current_klines(symbols, num: int = 120,
                         session: str = "regular") -> dict:
    """銘柄とSPYの現在1分足を安全な共有購読から取得する。

    過去K線APIは呼ばず、失敗時も ``frames`` と ``meta.errors`` を持つ構造を
    返す。入力シンボルはmoomoo_client側で正規化される。
    """
    values = (symbols,) if isinstance(symbols, str) else tuple(symbols or ())
    try:
        return _fetch_current_klines_cached(
            tuple(str(value) for value in values), int(num), str(session))
    except Exception as exc:
        return {
            "frames": {},
            "meta": {
                "available": False,
                "partial": False,
                "source": "Unavailable",
                "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "session": str(session or "regular").strip().lower(),
                "requested_symbols": tuple(
                    str(value or "").strip().upper() for value in values),
                "subscribed_symbols": (),
                "errors": {
                    "adapter": f"リアルタイム1分足を取得できませんでした: {exc}"
                },
                "quota": {},
                "decision_quotes": {},
                "subscription_reused": False,
                "retry_after_seconds": None,
                "timeframe": "K_1M",
                "uses_daily_bars": False,
                "uses_history_quota": False,
                "history_requests": 0,
            },
        }


# 他の公開fetcherと同じようにテスト・UIから短期キャッシュを消去できるようにする。
fetch_current_klines.clear = _fetch_current_klines_cached.clear


@st.cache_data(ttl=30, show_spinner="チャートデータを取得中...")
def _fetch_chart_history_cached(ticker: str, period: str, interval: str,
                                allow_new_quota: bool,
                                settings_token: tuple) -> tuple[pd.DataFrame, dict]:
    """チャート用OHLCVをmoomoo優先で取得し、取得元メタデータも返す。

    ``allow_new_quota`` は新規銘柄の過去K線枠を使う明示指定。明示時も予約枠は
    温存し、足種別の永続キャッシュを最優先する。
    """
    del settings_token  # Streamlitのキャッシュキーに設定値を含めるための引数。
    fallback_reason = None
    moomoo_meta = {}
    if not moomoo_fetcher.history_enabled():
        # クォータを消費しないよう、moomooには問い合わせない。
        return fetch_history(ticker, period, interval), {
            "source": "Yahoo Finance",
            "code": ticker,
            "fetched_at": None,
            "cache_status": "fallback",
            "quota": None,
            "remain": None,
            "fallback_reason": None,
        }
    try:
        moomoo = moomoo_fetcher.fetch_history(
            ticker, period, interval, allow_new_quota=allow_new_quota)
        moomoo_meta = dict(moomoo.attrs.get("moomoo_meta", {}))
    except moomoo_fetcher.MoomooError as exc:
        moomoo = pd.DataFrame()
        fallback_reason = str(exc)

    if not moomoo.empty:
        try:
            yahoo = fetch_history(ticker, period, interval)
        except FetchError:
            yahoo = pd.DataFrame()
        if moomoo_meta.get("cache_status") == "stale" and not yahoo.empty:
            moomoo_last = _last_history_time(moomoo)
            yahoo_last = _last_history_time(yahoo)
            if (moomoo_last is None or yahoo_last is None
                    or yahoo_last >= moomoo_last):
                return yahoo, {
                    "source": "Yahoo Finance",
                    "code": ticker,
                    "fetched_at": moomoo_meta.get("fetched_at"),
                    "cache_status": "fallback",
                    "quota": moomoo_meta.get("quota"),
                    "remain": moomoo_meta.get("remain"),
                    "fallback_reason": (
                        "期限切れmoomooキャッシュより新しいYahoo系列を使用しました。 "
                        + str(moomoo_meta.get("fallback_reason") or "")).rstrip(),
                }
        enriched = _merge_corporate_actions(moomoo, yahoo, interval)
        metadata = {
            "source": "moomoo OpenAPI",
            "code": moomoo_fetcher.normalize_code(ticker),
            "fetched_at": None,
            "cache_status": "refreshed",
            "quota": None,
            "remain": None,
            "fallback_reason": fallback_reason,
        }
        metadata.update({key: moomoo_meta.get(key, metadata[key]) for key in metadata})
        return enriched, metadata

    yahoo = fetch_history(ticker, period, interval)
    return yahoo, {
        "source": "Yahoo Finance",
        "code": ticker,
        "fetched_at": moomoo_meta.get("fetched_at"),
        "cache_status": moomoo_meta.get("cache_status", "fallback"),
        "quota": moomoo_meta.get("quota"),
        "remain": moomoo_meta.get("remain"),
        "fallback_reason": (fallback_reason or moomoo_meta.get("fallback_reason")
                            or "moomooからデータを取得できませんでした"),
    }


def fetch_chart_history(ticker: str, period: str, interval: str = "1d",
                        allow_new_quota: bool = False) -> tuple[pd.DataFrame, dict]:
    """設定変更を即時反映しつつ、同じ設定内では30秒キャッシュする。"""
    settings = moomoo_fetcher._integration_settings()
    token = (
        bool(settings.get("enabled")), bool(moomoo_fetcher.history_enabled()),
        str(settings.get("host")),
        int(settings.get("port", 11111)), settings.get("history_reserve"),
    )
    return _fetch_chart_history_cached(
        ticker, period, interval, allow_new_quota, token)


# 既存のStreamlitキャッシュ関数と同じクリア操作を維持する。
fetch_chart_history.clear = _fetch_chart_history_cached.clear


@st.cache_data(ttl=10, show_spinner=False)
def fetch_market_state(ticker: str) -> dict:
    """moomooの市場状態。取得不能時はWAIT判定に使える失敗情報を返す。"""
    try:
        state = moomoo_fetcher.fetch_market_state(ticker)
    except moomoo_fetcher.MoomooError as exc:
        return {
            "code": ticker,
            "stock_name": "",
            "market_state": None,
            "source": "Unavailable",
            "fallback_reason": str(exc),
        }
    if state:
        return state
    return {
        "code": ticker,
        "stock_name": "",
        "market_state": None,
        "source": "Unavailable",
        "fallback_reason": "moomooから市場状態を取得できませんでした",
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
