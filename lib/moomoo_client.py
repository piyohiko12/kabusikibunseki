"""moomoo OpenAPI(読み取り専用)からのリアルタイムデータ取得。

yfinanceの置き換えではなく上乗せとして使う。moomooが使えないときは
呼び出し側が従来どおりyfinanceにフォールバックできるよう、
この層は失敗しても例外を投げずにNone/空を返す。

安全のための方針:

- **取引系のAPIには一切触れない。** import するのは相場用の OpenQuoteContext だけで、
  発注に使う OpenSecTradeContext はこのファイルからも他のファイルからも参照しない。
  したがってバグや誤操作で注文が出ることは原理的に起こらない。
- **認証情報をこのリポジトリでは扱わない。** ログインはOpenD側で完結するため、
  パスワード等をコードや設定ファイルに置く必要がない(このリポジトリは公開設定)。
- **接続先は既定でローカルのOpenDのみ**(127.0.0.1)。

ハング対策(重要):
OpenQuoteContext のコンストラクタは、OpenDが起動していないと約26秒間隔で
無限に再接続を試み、決して例外を返さない。そのまま呼ぶとStreamlitが固まるため、
(1)TCPの事前確認 (2)デーモンスレッド+タイムアウト (3)失敗結果の一時キャッシュ
の3段構えで隔離する。
"""

import socket
import threading
import time

import pandas as pd
import streamlit as st

# OpenDへの接続の既定値。設定で上書きできる。
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11111

# 事前のTCP確認と、接続確立を待つ上限(秒)。
_PORT_CHECK_TIMEOUT = 0.4
_CONNECT_TIMEOUT = 8.0

# 接続に失敗した直後は、しばらく再試行しない(毎回の再実行で待たされないため)。
_RETRY_INTERVAL = 60.0

_last_failure: dict = {"at": 0.0, "reason": ""}
_lock = threading.Lock()


class _State:
    OFF = "off"                  # 設定で無効
    NOT_INSTALLED = "not_installed"  # moomoo-api 未インストール
    NO_OPEND = "no_opend"        # OpenDが起動していない
    TIMEOUT = "timeout"          # OpenDは居るが応答しない
    ERROR = "error"
    OK = "ok"


def to_code(ticker: str) -> str | None:
    """"AAPL" → "US.AAPL" のようにmoomooの銘柄コードへ変換する。

    指数(^GSPC など)はティッカー体系が異なり無料権限では扱えないことが多いため
    Noneを返し、呼び出し側でyfinanceのままにする。
    """
    t = (ticker or "").strip().upper()
    if not t or t.startswith("^") or "=" in t:
        return None
    if "." in t:  # すでに "US.AAPL" 形式
        return t
    return f"US.{t}"


def _settings() -> dict:
    from lib import settings_store
    s = settings_store.load()
    return {
        "enabled": bool(s.get("moomoo_enabled", False)),
        "host": s.get("moomoo_host") or DEFAULT_HOST,
        "port": int(s.get("moomoo_port") or DEFAULT_PORT),
    }


def _port_open(host: str, port: int) -> bool:
    """OpenDが待ち受けているかを即座に確認する(未起動なら数ミリ秒で判定)。"""
    try:
        with socket.create_connection((host, port), timeout=_PORT_CHECK_TIMEOUT):
            return True
    except OSError:
        return False


@st.cache_resource(show_spinner=False)
def _context(host: str, port: int):
    """OpenQuoteContextを1つだけ作って使い回す。失敗時はNone。

    Streamlitの再実行のたびに接続し直さないよう cache_resource で保持する。
    """
    try:
        from moomoo import OpenQuoteContext
        from moomoo.common.sys_config import SysConfig
    except Exception:
        return None

    # SDKのコンソールログを止め、SDKのスレッドをデーモン化して終了を妨げないようにする。
    try:
        SysConfig.enable_console_log(False)
        SysConfig.set_all_thread_daemon(True)
    except Exception:
        pass

    box: dict = {}

    def _connect():
        try:
            box["ctx"] = OpenQuoteContext(host=host, port=port)
        except Exception as e:  # 到達しない想定だが念のため
            box["err"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=_connect, daemon=True)
    th.start()
    th.join(timeout=_CONNECT_TIMEOUT)
    if th.is_alive():
        # スレッドは残るがデーモンなのでアプリ終了を妨げない。
        return None
    return box.get("ctx")


@st.cache_data(ttl=10, show_spinner=False)
def status() -> dict:
    """接続状態を返す。UIの表示用。

    戻り値: {"state": _State, "message": 表示用の文言}
    毎回TCP確認をしないよう短時間キャッシュする。
    """
    cfg = _settings()
    if not cfg["enabled"]:
        return {"state": _State.OFF, "message": "moomoo連携はオフ"}

    try:
        import moomoo  # noqa: F401
    except Exception:
        return {"state": _State.NOT_INSTALLED,
                "message": "moomoo-api が未インストール(pip install moomoo-api)"}

    if not _port_open(cfg["host"], cfg["port"]):
        return {"state": _State.NO_OPEND,
                "message": f"OpenDに接続できません({cfg['host']}:{cfg['port']})。"
                           "OpenDを起動してログインしてください"}

    with _lock:
        if time.time() - _last_failure["at"] < _RETRY_INTERVAL:
            return {"state": _State.TIMEOUT,
                    "message": f"OpenDが応答しません({_last_failure['reason']})"}

    ctx = _context(cfg["host"], cfg["port"])
    if ctx is None:
        with _lock:
            _last_failure.update(at=time.time(), reason="接続タイムアウト")
        return {"state": _State.TIMEOUT, "message": "OpenDが応答しません(接続タイムアウト)"}

    return {"state": _State.OK, "message": "moomoo接続中"}


def is_ready() -> bool:
    return status()["state"] == _State.OK


def _ctx():
    """使える状態のcontextを返す。使えなければNone(例外は投げない)。"""
    if not is_ready():
        return None
    cfg = _settings()
    return _context(cfg["host"], cfg["port"])


def _ok(ret) -> bool:
    """SDKの戻り値 ret が成功かどうか。RET_OK は 0。"""
    return ret == 0


# --------------------------------------------------------------- 購読の管理
# 板・歩み値は事前の subscribe が必要で、購読数には口座ごとの上限がある。
# 表示中の銘柄だけを購読し、切り替えたら前の銘柄を解除して上限を使い切らないようにする。
_subscribed: dict = {}


def _ensure_subscribed(ctx, code: str, subtypes: list) -> bool:
    key = tuple(sorted(str(s) for s in subtypes))
    with _lock:
        prev = _subscribed.get(key)
        if prev == code:
            return True
    try:
        if prev:
            ctx.unsubscribe([prev], subtypes)
        ret, msg = ctx.subscribe([code], subtypes)
    except Exception:
        return False
    if not _ok(ret):
        return False
    with _lock:
        _subscribed[key] = code
    return True


# ------------------------------------------------------------------- 取得API

@st.cache_data(ttl=5, show_spinner=False)
def snapshot(tickers: tuple[str, ...]) -> dict[str, dict]:
    """リアルタイムのスナップショット(現在値・前日終値・PER等)を返す。

    subscribe不要で、指定銘柄をまとめて1回で取得できる。
    取得できない銘柄は結果に含めない(呼び出し側はyfinanceにフォールバックする)。
    """
    ctx = _ctx()
    if ctx is None:
        return {}
    codes, back = [], {}
    for t in tickers:
        c = to_code(t)
        if c:
            codes.append(c)
            back[c] = t.strip().upper()
    if not codes:
        return {}
    try:
        ret, data = ctx.get_market_snapshot(codes)
    except Exception:
        return {}
    if not _ok(ret) or not isinstance(data, pd.DataFrame) or data.empty:
        return {}

    out = {}
    for row in data.to_dict("records"):
        t = back.get(row.get("code"))
        if not t:
            continue
        last = row.get("last_price")
        prev = row.get("prev_close_price")
        if last is None or pd.isna(last):
            # 価格が取れない銘柄は結果に含めない。
            # 呼び出し側が「moomooに無い」と判断してyfinanceに戻せるようにする。
            continue
        change_pct = None
        if last is not None and prev:
            try:
                change_pct = (float(last) / float(prev) - 1) * 100
            except (TypeError, ZeroDivisionError):
                change_pct = None
        out[t] = {
            "price": last,
            "previous_close": prev,
            "change_percent": change_pct,
            "open": row.get("open_price"),
            "high": row.get("high_price"),
            "low": row.get("low_price"),
            "volume": row.get("volume"),
            "turnover": row.get("turnover"),
            "per": row.get("pe_ttm_ratio") or row.get("pe_ratio"),
            "pbr": row.get("pb_ratio"),
            "dividend_yield": row.get("dividend_ratio_ttm"),
            "market_cap": row.get("total_market_val"),
            "eps": row.get("earning_per_share"),
            "update_time": row.get("update_time"),
            "suspension": row.get("suspension"),
        }
    return out


@st.cache_data(ttl=3, show_spinner=False)
def order_book(ticker: str, num: int = 10) -> dict | None:
    """板情報(買い/売りの気配)を返す。

    戻り値: {"bids": [(価格, 数量, 注文数), ...], "asks": [...], "code": str}
    LV2権限が無い場合は1〜数階層しか返らないことがある。
    """
    ctx = _ctx()
    code = to_code(ticker)
    if ctx is None or not code:
        return None
    try:
        from moomoo import SubType
    except Exception:
        return None
    if not _ensure_subscribed(ctx, code, [SubType.ORDER_BOOK]):
        return None
    try:
        ret, data = ctx.get_order_book(code, num=num)
    except Exception:
        return None
    if not _ok(ret) or not isinstance(data, dict):
        return None

    def rows(key):
        out = []
        for item in data.get(key, []) or []:
            if len(item) >= 3:
                out.append((float(item[0]), int(item[1]), int(item[2])))
        return out

    bids, asks = rows("Bid"), rows("Ask")
    if not bids and not asks:
        return None
    return {"code": code, "bids": bids, "asks": asks}


@st.cache_data(ttl=3, show_spinner=False)
def recent_ticks(ticker: str, num: int = 60) -> pd.DataFrame:
    """歩み値(直近の約定)を返す。取得できなければ空のDataFrame。"""
    ctx = _ctx()
    code = to_code(ticker)
    if ctx is None or not code:
        return pd.DataFrame()
    try:
        from moomoo import SubType
    except Exception:
        return pd.DataFrame()
    if not _ensure_subscribed(ctx, code, [SubType.TICKER]):
        return pd.DataFrame()
    try:
        ret, data = ctx.get_rt_ticker(code, num=num)
    except Exception:
        return pd.DataFrame()
    if not _ok(ret) or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    cols = [c for c in ["time", "price", "volume", "turnover", "ticker_direction", "type"]
            if c in data.columns]
    return data[cols].iloc[::-1].reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def capital_distribution(ticker: str) -> dict | None:
    """当日の資金流入(大口/中口/小口の売買代金)を返す。yfinanceにはない情報。"""
    ctx = _ctx()
    code = to_code(ticker)
    if ctx is None or not code:
        return None
    try:
        ret, data = ctx.get_capital_distribution(code)
    except Exception:
        return None
    if not _ok(ret) or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    row = data.to_dict("records")[0]

    def num(key):
        v = row.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    tiers = [
        ("大口", num("capital_in_big") - num("capital_out_big"),
         num("capital_in_big"), num("capital_out_big")),
        ("中口", num("capital_in_mid") - num("capital_out_mid"),
         num("capital_in_mid"), num("capital_out_mid")),
        ("小口", num("capital_in_small") - num("capital_out_small"),
         num("capital_in_small"), num("capital_out_small")),
    ]
    if not any(t[2] or t[3] for t in tiers):
        return None
    return {"tiers": tiers, "net": sum(t[1] for t in tiers),
            "update_time": row.get("update_time")}


SESSION_RANKS = {
    "pre": ("get_us_pre_market_rank", "pre_market"),
    "after": ("get_us_after_hours_rank", "after_hours"),
    "overnight": ("get_us_overnight_rank", "overnight"),
}


@st.cache_data(ttl=60, show_spinner=False)
def session_rank(session: str, count: int = 10, losers: bool = False) -> pd.DataFrame:
    """米国株のプレ/アフター/夜間セッションのランキング。

    session: "pre" | "after" | "overnight"、losers=Trueで値下がり順。
    Yahoo Financeでは取得できない時間外の値動きを一覧できる。
    """
    ctx = _ctx()
    spec = SESSION_RANKS.get(session)
    if ctx is None or not spec:
        return pd.DataFrame()
    method, prefix = spec
    try:
        from moomoo import RankSortDir
        sort_dir = RankSortDir.ASCENDING if losers else RankSortDir.DESCENDING
        ret, data = getattr(ctx, method)(sort_dir=sort_dir, count=count)
    except Exception:
        return pd.DataFrame()
    if not _ok(ret):
        return pd.DataFrame()
    # このAPI群は (件数, DataFrame) のタプルを返す
    if isinstance(data, tuple):
        data = data[1] if len(data) > 1 else None
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "ティッカー": data.get("security", pd.Series(dtype=str)).map(_strip_market),
        "銘柄名": data.get("name"),
        "時間外価格": data.get(f"{prefix}_price"),
        "時間外変化率": data.get(f"{prefix}_change_ratio"),
        "出来高": data.get(f"{prefix}_volume"),
        "終値": data.get("close_price"),
    })
    return out.dropna(subset=["ティッカー"]).reset_index(drop=True)


def _strip_market(code) -> str | None:
    """"US.AAPL" や {"code": "US.AAPL"} から "AAPL" を取り出す。"""
    if isinstance(code, dict):
        code = code.get("code") or code.get("security")
    if not isinstance(code, str) or not code:
        return None
    return code.split(".", 1)[1] if "." in code else code


@st.cache_data(ttl=3600, show_spinner=False)
def fed_watch() -> pd.DataFrame:
    """FedWatchの政策金利織り込み確率(会合日 × 金利レンジ × 確率)。"""
    ctx = _ctx()
    if ctx is None:
        return pd.DataFrame()
    try:
        ret, data = ctx.get_fed_watch_target_rate()
    except Exception:
        return pd.DataFrame()
    if not _ok(ret) or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    cols = [c for c in ["meeting_date", "target_range", "probability"] if c in data.columns]
    if len(cols) < 3:
        return pd.DataFrame()
    out = data[cols].copy()
    out["probability"] = pd.to_numeric(out["probability"], errors="coerce")
    return out.dropna(subset=["probability"])


@st.cache_data(ttl=3600, show_spinner=False)
def short_interest(ticker: str, num: int = 24) -> pd.DataFrame:
    """空売り残高の推移(米国株)。列: 日付・空売り株数・浮動株比率・日数(days to cover)。"""
    ctx = _ctx()
    code = to_code(ticker)
    if ctx is None or not code:
        return pd.DataFrame()
    try:
        res = ctx.get_short_interest(code, num=num)
    except Exception:
        return pd.DataFrame()
    # このAPIは (ret, 米国用DataFrame, 香港用DataFrame) を返す
    if not isinstance(res, tuple) or len(res) < 2 or not _ok(res[0]):
        return pd.DataFrame()
    us = res[1]
    if not isinstance(us, pd.DataFrame) or us.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "日付": us.get("timestamp_str"),
        "空売り株数": pd.to_numeric(us.get("shares_short"), errors="coerce"),
        "浮動株比率": pd.to_numeric(us.get("short_percent"), errors="coerce"),
        "買い戻し日数": pd.to_numeric(us.get("days_to_cover"), errors="coerce"),
        "終値": pd.to_numeric(us.get("close_price"), errors="coerce"),
    })
    return out.dropna(subset=["空売り株数"]).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def institutional_holding(ticker: str, num: int = 12) -> pd.DataFrame:
    """機関投資家の保有推移。列: 報告期・機関数・保有株数・保有比率と各変化。"""
    ctx = _ctx()
    code = to_code(ticker)
    if ctx is None or not code:
        return pd.DataFrame()
    try:
        ret, data = ctx.get_shareholders_institutional(code, num=num)
    except Exception:
        return pd.DataFrame()
    if not _ok(ret) or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "報告期": data.get("period_text"),
        "機関数": pd.to_numeric(data.get("institution_quantity"), errors="coerce"),
        "機関数の増減": pd.to_numeric(data.get("institution_quantity_change"), errors="coerce"),
        "保有株数": pd.to_numeric(data.get("holder_quantity"), errors="coerce"),
        "保有株数の増減": pd.to_numeric(data.get("holder_quantity_change"), errors="coerce"),
        "保有比率": pd.to_numeric(data.get("holder_pct"), errors="coerce"),
        "保有比率の増減": pd.to_numeric(data.get("holder_pct_change"), errors="coerce"),
    })
    out = out.dropna(subset=["報告期"])
    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def option_volatility(ticker: str) -> dict | None:
    """オプションのIV(予想変動率)とHV(実績変動率)の推移および現在の評価。"""
    ctx = _ctx()
    code = to_code(ticker)
    if ctx is None or not code:
        return None
    try:
        ret, data = ctx.get_option_volatility(code)
    except Exception:
        return None
    if not _ok(ret) or not isinstance(data, pd.DataFrame) or data.empty:
        return None
    series = pd.DataFrame({
        "日付": data.get("timestamp_str"),
        "IV": pd.to_numeric(data.get("implied_volatility"), errors="coerce"),
        "HV": pd.to_numeric(data.get("history_volatility"), errors="coerce"),
        "IVプレミアム": pd.to_numeric(data.get("volatility_premium"), errors="coerce"),
    }).dropna(subset=["IV"], how="all")
    if series.empty:
        return None
    head = data.iloc[0]
    return {
        "series": series.reset_index(drop=True),
        "average_iv": pd.to_numeric(pd.Series([head.get("average_impvol")]),
                                    errors="coerce").iloc[0],
        "status": head.get("impvol_status"),
        "analysis": head.get("analysis") or "",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def put_call_ratio(days: int = 180) -> pd.DataFrame:
    """米国株オプション市場全体のPut/Callレシオ(出来高ベース)の推移。"""
    ctx = _ctx()
    if ctx is None:
        return pd.DataFrame()
    try:
        from moomoo import OptionMarket, OptionStatisticDataType
        end = pd.Timestamp.today().normalize()
        begin = end - pd.Timedelta(days=days)
        res = ctx.get_option_market_statistic(
            OptionMarket.US_SECURITY, OptionStatisticDataType.VOLUME,
            begin_time=begin.strftime("%Y-%m-%d"), end_time=end.strftime("%Y-%m-%d"))
    except Exception:
        return pd.DataFrame()
    if not isinstance(res, tuple) or not _ok(res[0]):
        return pd.DataFrame()
    data = res[1]
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "日付": data.get("time"),
        "Put/Call": pd.to_numeric(data.get("ratio"), errors="coerce"),
        "Call": pd.to_numeric(data.get("call_value"), errors="coerce"),
        "Put": pd.to_numeric(data.get("put_value"), errors="coerce"),
    }).dropna(subset=["Put/Call"])
    return out.reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def history_quota() -> dict | None:
    """履歴K線の残りクォータ。無料枠の消費状況を画面で確認できるようにする。"""
    ctx = _ctx()
    if ctx is None:
        return None
    try:
        ret, data = ctx.get_history_kl_quota(get_detail=False)
    except Exception:
        return None
    if not _ok(ret):
        return None
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return None
        row = data.to_dict("records")[0]
    elif isinstance(data, dict):
        row = data
    else:
        return None
    return {"used": row.get("used_quota"), "remain": row.get("remain_quota")}
