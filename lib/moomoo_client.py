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

from lib import session_intelligence

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

# moomooの正式なmarket prefixだけを「既に正規化済み」とみなす。
# 米国のclass share（BRK.Bなど）もドットを含むため、単に"."の有無で
# 判定するとBRK.BをそのままSDKへ渡してしまう。
_MARKET_PREFIXES = frozenset({"US", "HK", "SH", "SZ", "SG", "MY", "JP", "CC"})


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
    prefix, separator, _ = t.partition(".")
    if separator and prefix in _MARKET_PREFIXES:  # すでに "US.AAPL" 形式
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

# 1分足はリアルタイム判定専用に、板・歩み値とは別のlease poolで管理する。
# moomooは購読後60秒未満の解除を認めない。複数の画面・銘柄集合を相互に
# 追い出さず、十分に古く未使用になった自管理購読だけを掃除する。
_CURRENT_KLINE_MIN_LEASE_SECONDS = 60.0
_CURRENT_KLINE_IDLE_SECONDS = 120.0
_current_kline_lock = threading.Lock()
_current_kline_leases: dict = {}
# 初版のprivate名を参照するテスト・開発用コードとの互換alias。
_current_kline_lease = _current_kline_leases


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


def _current_kline_result(*, frames=None, available: bool = False,
                          partial: bool = False, session: str,
                          requested_symbols=(), subscribed_symbols=(),
                          errors=None, quota=None, decision_quotes=None,
                          reused: bool = False,
                          retry_after_seconds: float | None = None) -> dict:
    """current_klinesの成功・失敗を同じpickle可能な形へそろえる。"""
    fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
    return {
        "frames": dict(frames or {}),
        "meta": {
            "available": bool(available),
            "partial": bool(partial),
            "source": "moomoo OpenAPI current K_1M" if frames else "Unavailable",
            "fetched_at": fetched_at,
            "session": session,
            "requested_symbols": tuple(requested_symbols),
            "subscribed_symbols": tuple(subscribed_symbols),
            "errors": dict(errors or {}),
            "quota": dict(quota or {}),
            "decision_quotes": dict(decision_quotes or {}),
            "subscription_reused": bool(reused),
            "retry_after_seconds": retry_after_seconds,
            "timeframe": "K_1M",
            "uses_daily_bars": False,
            "uses_history_quota": False,
            "history_requests": 0,
        },
    }


def _quota_number(value) -> int | None:
    """boolや小数を受け入れず、購読枠の非負整数だけを返す。"""
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != number:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= 0 else None


def _normalise_subscription_list(value) -> tuple[dict | None, str | None]:
    """query_subscriptionのsub_listを ``種別 -> code tuple`` にそろえる。"""
    if not isinstance(value, dict):
        return None, "現在の購読一覧を解析できませんでした"
    result: dict[str, tuple[str, ...]] = {}
    collected: dict[str, set[str]] = {}
    for raw_subtype, raw_codes in value.items():
        subtype_value = getattr(raw_subtype, "value", raw_subtype)
        if not isinstance(subtype_value, str):
            return None, "現在の購読種別を解析できませんでした"
        subtype = subtype_value.strip().upper()
        if "." in subtype:
            subtype = subtype.rsplit(".", 1)[-1]
        if not subtype:
            return None, "現在の購読種別を解析できませんでした"
        if isinstance(raw_codes, (str, bytes)) or not isinstance(
                raw_codes, (list, tuple, set, frozenset)):
            return None, "現在の購読銘柄を解析できませんでした"
        bucket = collected.setdefault(subtype, set())
        for raw_code in raw_codes:
            if not isinstance(raw_code, str):
                return None, "現在の購読銘柄を解析できませんでした"
            code = raw_code.strip().upper()
            if not code:
                return None, "現在の購読銘柄を解析できませんでした"
            if "." not in code:
                code = to_code(code) or code
            bucket.add(code)
    for subtype, codes in collected.items():
        result[subtype] = tuple(sorted(codes))
    return result, None


def _query_subscription_quota(ctx) -> tuple[dict | None, str | None]:
    """全接続の購読枠を解析する。曖昧な応答は安全側で失敗にする。"""
    try:
        ret, data = ctx.query_subscription(is_all_conn=True)
    except Exception as exc:
        return None, f"購読残枠を確認できませんでした: {type(exc).__name__}: {exc}"
    if not _ok(ret):
        return None, f"購読残枠を確認できませんでした: {data}"
    if not isinstance(data, dict):
        return None, "購読残枠の応答形式を解析できませんでした"

    parsed = {
        key: _quota_number(data.get(key))
        for key in ("total_used", "own_used", "remain")
    }
    if any(value is None for value in parsed.values()):
        return None, "購読残枠の数値を解析できませんでした"
    if "sub_list" not in data:
        return None, "現在の購読一覧を解析できませんでした"
    sub_list, sub_error = _normalise_subscription_list(data["sub_list"])
    if sub_error:
        return None, sub_error
    parsed["sub_list"] = sub_list
    return parsed, None


def _session_spec(name: str):
    """要求セッションを検証し、共通の全時間帯K線購読設定を返す。

    K_1Mの購読条件を画面セッションごとに変えると、SDK上は同じcode/subtypeが
    既購読に見えてもRTHのまま残り得る。米国株の判定用購読は常にALLへ統一し、
    取得後の1分足を ``_decision_quote`` で要求セッションだけに絞る。
    """
    try:
        from moomoo import Session
    except Exception as exc:
        return None, False, f"moomoo Sessionを読み込めませんでした: {exc}"

    normalised = str(name or "regular").strip().lower()
    allowed = {
        "regular", "pre", "premarket", "after", "afterhours",
        "extended", "all", "overnight",
    }
    if normalised not in allowed:
        return None, False, (
            "sessionはregular/pre/premarket/after/afterhours/extended/"
            "all/overnightから選んでください"
        )
    enum_name = "ALL"
    extended_time = True
    enum_value = getattr(Session, enum_name, None)
    if enum_value is None:
        return None, extended_time, (
            f"現在のmoomoo-apiはSession.{enum_name}に対応していません"
        )
    return enum_value, extended_time, None


def _normalise_current_kline(data: pd.DataFrame, num: int) -> pd.DataFrame:
    """get_cur_klineの応答を既存チャートと同じOHLCV形式へ変換する。"""
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    required = {"time_key", "open", "high", "low", "close", "volume"}
    if not required.issubset(data.columns):
        return pd.DataFrame()

    index = pd.to_datetime(data["time_key"], errors="coerce")
    frame = pd.DataFrame({
        "Open": pd.to_numeric(data["open"], errors="coerce").to_numpy(),
        "High": pd.to_numeric(data["high"], errors="coerce").to_numpy(),
        "Low": pd.to_numeric(data["low"], errors="coerce").to_numpy(),
        "Close": pd.to_numeric(data["close"], errors="coerce").to_numpy(),
        "Volume": pd.to_numeric(data["volume"], errors="coerce").to_numpy(),
    }, index=pd.DatetimeIndex(index, name="Datetime"))
    if "turnover" in data.columns:
        frame["Turnover"] = pd.to_numeric(
            data["turnover"], errors="coerce").to_numpy()
    frame = frame.loc[~frame.index.isna()].dropna(subset=["Close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.tail(num)


def _canonical_kline_session(name: str) -> str:
    return {
        "pre": "premarket", "premarket": "premarket",
        "regular": "regular",
        "after": "afterhours", "afterhours": "afterhours",
        "overnight": "overnight", "extended": "extended", "all": "all",
    }.get(str(name or "").strip().lower(), "unknown")


def _market_timestamp(value) -> pd.Timestamp | None:
    """moomooの米国市場ローカル時刻をtimezone-awareにする。"""
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


def _session_at_market_time(stamp: pd.Timestamp) -> str:
    """NYSE定例日・短縮日を考慮してbar時刻のセッションを返す。"""
    try:
        detected = session_intelligence.detect_current_session(
            stamp, overnight_eligible=True)
    except (TypeError, ValueError, OverflowError):
        return "closed"
    return str(detected.get("session") or "closed")


def _decision_quote(frame: pd.DataFrame, session_name: str, symbol: str) -> dict:
    """指定セッション内の最新K_1M終値と検証可能なbar時刻を返す。"""
    requested = _canonical_kline_session(session_name)
    allowed = {
        "premarket": {"premarket"}, "regular": {"regular"},
        "afterhours": {"afterhours"}, "overnight": {"overnight"},
        "extended": {"premarket", "afterhours"},
        "all": {"premarket", "regular", "afterhours", "overnight"},
    }.get(requested, set())
    base = {
        "available": False, "session": requested, "requested_session": requested,
        "symbol": str(symbol or "").strip().upper(),
        "price": None, "updated_at": None, "bar_time": None,
        "timestamp_verified": False, "timestamp_semantics": "bar_start",
        "timeframe": "K_1M", "source": "moomoo OpenAPI current K_1M",
        "uses_daily_bars": False,
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame:
        return {**base, "error": "利用できる1分足がありません"}

    selected = None
    for position, raw_stamp in enumerate(frame.index):
        stamp = _market_timestamp(raw_stamp)
        if stamp is None:
            continue
        actual = _session_at_market_time(stamp)
        if actual in allowed:
            selected = position, stamp, actual
    if selected is None:
        return {**base, "error": "指定セッションの1分足がありません"}

    position, stamp, actual = selected
    try:
        price = float(frame.iloc[position]["Close"])
    except (TypeError, ValueError, OverflowError):
        price = None
    if price is None or not pd.notna(price) or price <= 0:
        return {**base, "error": "指定セッションの価格を確認できません"}
    stamp_iso = stamp.isoformat()
    return {
        **base, "available": True, "session": actual,
        "price": price, "updated_at": stamp_iso, "bar_time": stamp_iso,
        "timestamp_verified": True,
    }


def _current_kline_symbols(
        symbols) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...], dict]:
    """入力をUSコードへ正規化し、比較対象SPYを必ず同じ購読へ含める。"""
    values = (symbols,) if isinstance(symbols, str) else tuple(symbols or ())
    codes: list[str] = []
    code_to_symbol: dict[str, str] = {}
    requested_symbols: list[str] = []
    errors: dict[str, str] = {}
    for original in values:
        label = str(original or "").strip().upper() or "(empty)"
        code = to_code(label)
        if not code or not code.startswith("US."):
            errors[label] = "リアルタイム1分足は米国銘柄だけに対応しています"
            continue
        symbol = code.split(".", 1)[1]
        if code not in code_to_symbol:
            codes.append(code)
            code_to_symbol[code] = symbol
            requested_symbols.append(symbol)
    if codes and "US.SPY" not in code_to_symbol:
        codes.append("US.SPY")
        code_to_symbol["US.SPY"] = "SPY"
    return tuple(sorted(codes)), code_to_symbol, tuple(requested_symbols), errors


def _current_kline_key(ctx, codes: tuple[str, ...], session_name: str) -> tuple:
    """全セッション共通leaseのkeyを作る。

    ``session_name`` は旧private呼び出しとの互換用。購読自体は常にALLなので、
    regular→afterhoursの画面切替でも同じleaseを再利用する。
    """
    try:
        hash(ctx)
        context_key = ctx
    except TypeError:
        context_key = ("context_id", id(ctx))
    return context_key, codes, "all"


def _active_all_session_codes(ctx) -> set[str]:
    """このprocessがALL購読を確認済みのcode集合を返す。"""
    covered: set[str] = set()
    for lease in _current_kline_leases.values():
        if lease.get("ctx") is ctx and lease.get("state") == "active":
            covered.update(lease.get("codes") or ())
    return covered


def _elapsed(now: float, value) -> float | None:
    try:
        return max(0.0, now - float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _quota_for_codes(quota: dict, codes: tuple[str, ...]) -> tuple[dict, tuple[str, ...]]:
    existing = set((quota.get("sub_list") or {}).get("K_1M", ()))
    missing = tuple(code for code in codes if code not in existing)
    result = {
        **quota,
        "required": len(missing),
        "already_subscribed": tuple(code for code in codes if code in existing),
        "missing": missing,
    }
    return result, missing


def _cleanup_current_kline_leases(now: float, target_ctx) -> str | None:
    """新規取得時だけ、120秒使われていない自管理購読を安全に掃除する。"""
    blocked_error = None
    candidate_keys = []
    for key, lease in list(_current_kline_leases.items()):
        age = _elapsed(now, lease.get("started_at"))
        idle = _elapsed(now, lease.get("last_used"))
        if (age is not None and idle is not None
                and age >= _CURRENT_KLINE_MIN_LEASE_SECONDS
                and idle >= _CURRENT_KLINE_IDLE_SECONDS):
            candidate_keys.append(key)

    for key in candidate_keys:
        lease = _current_kline_leases.get(key)
        if lease is None:
            continue
        lease_ctx = lease.get("ctx")
        managed = set(lease.get("managed_codes") or ())
        if lease.get("state") == "uncertain":
            reconciled, reconcile_error = _query_subscription_quota(lease_ctx)
            if reconcile_error:
                if lease_ctx is target_ctx:
                    blocked_error = reconcile_error
                continue
            present = set(
                (reconciled.get("sub_list") or {}).get("K_1M", ()))
            # 応答が返らなかった購読のうち、サーバーで確認できた分だけが
            # 解除対象。存在しないcodeへunsubscribeして新規取得を塞がない。
            managed &= present
            lease["managed_codes"] = tuple(sorted(managed))
        protectors: dict[str, tuple] = {}
        for code in managed:
            for other_key, other in _current_kline_leases.items():
                if other_key == key or other.get("ctx") is not lease_ctx:
                    continue
                if code in set(other.get("codes") or ()):
                    protectors[code] = other_key
                    break
        removable = tuple(sorted(managed - set(protectors)))
        if removable:
            try:
                ret, msg = lease_ctx.unsubscribe(
                    list(removable), [lease.get("subtype")])
            except Exception as exc:
                message = ("古い1分足購読を解除できませんでした: "
                           f"{type(exc).__name__}: {exc}")
                if lease_ctx is target_ctx:
                    blocked_error = message
                continue
            if not _ok(ret):
                # SDKが失敗時に内部sub_recordを変更するためleaseは保持する。
                message = f"古い1分足購読を解除できませんでした: {msg}"
                if lease_ctx is target_ctx:
                    blocked_error = message
                continue

        # 共有コードの所有権を残るleaseへ渡し、後で孤児購読にならないようにする。
        for code, protector_key in protectors.items():
            protector = _current_kline_leases.get(protector_key)
            if protector is None:
                continue
            owned = set(protector.get("managed_codes") or ())
            owned.add(code)
            protector["managed_codes"] = tuple(sorted(owned))
        _current_kline_leases.pop(key, None)
    return blocked_error


def _acquire_current_kline_lease(ctx, codes: tuple[str, ...], *,
                                 session_name: str, session_value,
                                 extended_time: bool, subtype) -> tuple[dict | None, dict]:
    """全セッションK_1M購読をquota-aware poolで再利用し、安全に追加する。"""
    now = time.monotonic()
    wanted_key = _current_kline_key(ctx, codes, session_name)
    with _current_kline_lock:
        lease = _current_kline_leases.get(wanted_key)
        recovered_managed: set[str] = set()
        if lease and lease.get("state") == "active":
            lease["last_used"] = now
            age = _elapsed(now, lease.get("started_at")) or 0.0
            return dict(lease.get("quota") or {}), {
                "ok": True, "reused": True, "age": age,
                "subscribed_codes": codes,
            }

        if lease and lease.get("state") == "uncertain":
            age = _elapsed(now, lease.get("started_at"))
            lease["last_used"] = now
            if age is None or age < _CURRENT_KLINE_MIN_LEASE_SECONDS:
                retry_after = (_CURRENT_KLINE_MIN_LEASE_SECONDS
                               if age is None else
                               _CURRENT_KLINE_MIN_LEASE_SECONDS - age)
                return None, {
                    "ok": False,
                    "error": "前回の購読結果を確認できないため、新しい購読を保留します",
                    "retry_after": retry_after,
                    "subscribed_codes": tuple(lease.get("observed_codes") or ()),
                }

            reconciled, reconcile_error = _query_subscription_quota(ctx)
            if reconcile_error:
                return None, {
                    "ok": False, "error": reconcile_error,
                    "subscribed_codes": tuple(lease.get("observed_codes") or ()),
                }
            reconciled, missing = _quota_for_codes(reconciled, codes)
            existing = set(reconciled.get("already_subscribed") or ())
            recovered_managed = (
                set(lease.get("managed_codes") or ()) & existing)
            mode_upgrade = set(lease.get("mode_upgrade_codes") or ())
            if not missing and not mode_upgrade:
                lease.update({
                    "state": "active", "last_used": now,
                    "managed_codes": tuple(sorted(recovered_managed)),
                    "quota": reconciled,
                })
                return reconciled, {
                    "ok": True, "reused": True, "age": age,
                    "subscribed_codes": codes,
                }
            # 既存codeのALL化を試みた応答が不明な場合、query_subscriptionでは
            # session条件まで証明できない。60秒経過後に安全に再試行する。
            # 新規codeが無い場合も同様に、このuncertain leaseを一旦捨てる。
            _current_kline_leases.pop(wanted_key, None)

        cleanup_error = _cleanup_current_kline_leases(now, ctx)
        quota, quota_error = _query_subscription_quota(ctx)
        if quota_error:
            return None, {"ok": False, "error": quota_error,
                          "subscribed_codes": ()}
        quota, missing = _quota_for_codes(quota, codes)
        already = tuple(quota.get("already_subscribed") or ())
        # query_subscriptionのsub_listはsession条件を含まない。別画面・外部接続の
        # 既購読codeでも、このprocessでALL購読を確認できないものは一度だけ
        # 同じcodeをALLで再subscribeして購読条件を明示する。既存codeなので
        # 新規購読枠のrequiredには数えず、自管理unsubscribe対象にもしない。
        compatible = _active_all_session_codes(ctx)
        mode_upgrade = tuple(
            code for code in already if code not in compatible)
        subscribe_codes = tuple(sorted(set(missing) | set(mode_upgrade)))
        if subscribe_codes and cleanup_error:
            return None, {
                "ok": False, "error": cleanup_error, "quota": quota,
                "subscribed_codes": already,
            }
        if quota["remain"] < len(missing):
            return None, {
                "ok": False,
                "error": (f"リアルタイム1分足の購読枠が不足しています"
                          f"（必要 {len(missing)} / 残り {quota['remain']}）"),
                "quota": quota,
                "subscribed_codes": already,
            }

        if not subscribe_codes:
            _current_kline_leases[wanted_key] = {
                "ctx": ctx, "codes": codes, "subtype": subtype,
                "session": session_name, "started_at": now, "last_used": now,
                "state": "active", "quota": quota,
                "managed_codes": tuple(sorted(recovered_managed)),
            }
            return quota, {"ok": True, "reused": False, "age": 0.0,
                           "subscribed_codes": codes}

        subscribe_started = time.monotonic()
        managed_codes = tuple(sorted(recovered_managed | set(missing)))
        uncertain = {
            "ctx": ctx, "codes": codes, "subtype": subtype,
            "session": session_name, "started_at": subscribe_started,
            "last_used": now, "state": "uncertain", "quota": quota,
            "managed_codes": managed_codes, "attempted_codes": subscribe_codes,
            "mode_upgrade_codes": mode_upgrade,
            "observed_codes": already,
        }
        try:
            ret, msg = ctx.subscribe(
                list(subscribe_codes), [subtype], is_first_push=False,
                subscribe_push=False, extended_time=extended_time,
                session=session_value,
            )
        except Exception as exc:
            _current_kline_leases[wanted_key] = uncertain
            return None, {
                "ok": False,
                "error": f"1分足を購読できませんでした: {type(exc).__name__}: {exc}",
                "quota": quota,
                "retry_after": _CURRENT_KLINE_MIN_LEASE_SECONDS,
                "subscribed_codes": already,
            }
        if not _ok(ret):
            _current_kline_leases[wanted_key] = uncertain
            return None, {
                "ok": False,
                "error": f"1分足を購読できませんでした: {msg}",
                "quota": quota,
                "retry_after": _CURRENT_KLINE_MIN_LEASE_SECONDS,
                "subscribed_codes": already,
            }

        _current_kline_leases[wanted_key] = {
            **uncertain, "state": "active", "observed_codes": codes,
        }
        return quota, {"ok": True, "reused": False, "age": 0.0,
                       "subscribed_codes": codes}


def current_klines(symbols, num: int = 120, session: str = "regular") -> dict:
    """銘柄とSPYの現在1分足を、履歴K線枠を使わず読み取り専用で返す。

    購読はcontext・銘柄集合ごとに全セッション共通で再利用する。利用中の別集合は
    維持し、古い自管理購読だけを60秒制約と残り枠を確認して掃除する。
    """
    session_name = str(session or "regular").strip().lower()
    try:
        count = int(num)
    except (TypeError, ValueError, OverflowError):
        count = 0
    if count < 1 or count > 1000:
        return _current_kline_result(
            session=session_name, errors={"num": "numは1〜1000で指定してください"})

    session_value, extended_time, session_error = _session_spec(session_name)
    if session_error:
        return _current_kline_result(
            session=session_name, errors={"session": session_error})

    codes, code_to_symbol, requested_symbols, errors = _current_kline_symbols(symbols)
    if not codes:
        if not errors:
            errors["symbols"] = "銘柄を1つ以上指定してください"
        return _current_kline_result(
            session=session_name, requested_symbols=requested_symbols,
            errors=errors)

    ctx = _ctx()
    if ctx is None:
        errors["connection"] = status()["message"]
        return _current_kline_result(
            session=session_name, requested_symbols=requested_symbols,
            errors=errors)
    try:
        from moomoo import AuType, KLType, SubType
    except Exception as exc:
        errors["sdk"] = f"moomooの1分足APIを読み込めませんでした: {exc}"
        return _current_kline_result(
            session=session_name, requested_symbols=requested_symbols,
            errors=errors)

    quota, lease = _acquire_current_kline_lease(
        ctx, codes, session_name="all", session_value=session_value,
        extended_time=extended_time, subtype=SubType.K_1M,
    )
    if not lease.get("ok"):
        errors["subscription"] = str(lease.get("error") or "1分足を購読できませんでした")
        active_symbols = tuple(
            str(code).split(".", 1)[-1]
            for code in lease.get("subscribed_codes") or ()
        )
        return _current_kline_result(
            session=session_name, requested_symbols=requested_symbols,
            subscribed_symbols=active_symbols,
            errors=errors, quota=lease.get("quota") or quota,
            retry_after_seconds=lease.get("retry_after"))

    frames: dict[str, pd.DataFrame] = {}
    for code in codes:
        symbol = code_to_symbol[code]
        try:
            ret, data = ctx.get_cur_kline(
                code, count, ktype=KLType.K_1M, autype=AuType.NONE)
        except Exception as exc:
            errors[symbol] = f"1分足を取得できませんでした: {type(exc).__name__}: {exc}"
            continue
        if not _ok(ret):
            errors[symbol] = f"1分足を取得できませんでした: {data}"
            continue
        frame = _normalise_current_kline(data, count)
        if frame.empty:
            errors[symbol] = "利用できる1分足がありません"
            continue
        frames[symbol] = frame

    decision_quotes: dict[str, dict] = {}
    for code in codes:
        symbol = code_to_symbol[code]
        quote = _decision_quote(
            frames.get(symbol, pd.DataFrame()), session_name, symbol)
        decision_quotes[symbol] = quote
        if not quote.get("available"):
            errors.setdefault(
                symbol, str(quote.get("error") or "指定セッションの価格を確認できません"))

    complete = (bool(frames) and not errors and len(frames) == len(codes)
                and all(row.get("available") for row in decision_quotes.values()))
    partial = bool(frames) and not complete
    return _current_kline_result(
        frames=frames, available=complete, partial=partial,
        session=session_name, requested_symbols=requested_symbols,
        subscribed_symbols=tuple(code_to_symbol[code] for code in codes),
        errors=errors, quota=quota, decision_quotes=decision_quotes,
        reused=bool(lease.get("reused")),
    )


# ------------------------------------------------------------------- 取得API

def _positive_snapshot_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if pd.notna(number) and number > 0 else None


def _snapshot_session_quotes(row: dict, fetched_at: str) -> dict[str, dict]:
    """snapshotの各価格を、時刻の証明範囲を誇張せず公開する。"""
    snapshot_stamp = _market_timestamp(row.get("update_time"))
    snapshot_updated_at = (
        snapshot_stamp.isoformat() if snapshot_stamp is not None else None)
    snapshot_session = (
        _session_at_market_time(snapshot_stamp)
        if snapshot_stamp is not None else "closed")
    fields = {
        "premarket": ("pre_price", "pre_volume"),
        "regular": ("last_price", "volume"),
        "afterhours": ("after_price", "after_volume"),
        "overnight": ("overnight_price", "overnight_volume"),
    }
    result: dict[str, dict] = {}
    for session_name, (price_field, volume_field) in fields.items():
        price = _positive_snapshot_number(row.get(price_field))
        regular_timestamp = (
            session_name == "regular" and snapshot_session == "regular")
        result[session_name] = {
            "available": price is not None,
            "session": session_name,
            "price": price,
            "volume": _positive_snapshot_number(row.get(volume_field)),
            # SDKは時間外価格ごとの時刻を返さない。汎用update_timeを時間外の
            # 鮮度時刻に流用せず、K_1Mのdecision_quotesでbar時刻を確認する。
            "updated_at": snapshot_updated_at if regular_timestamp else None,
            "snapshot_updated_at": snapshot_updated_at,
            "fetched_at": fetched_at,
            "timestamp_verified": bool(regular_timestamp),
            "actionable_from_snapshot": bool(price is not None and regular_timestamp),
            "price_field": price_field,
            "source": "moomoo OpenAPI snapshot",
            "uses_daily_bars": False,
        }
    return result

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

    fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
    out = {}
    for row in data.to_dict("records"):
        t = back.get(row.get("code"))
        if not t:
            continue
        last = row.get("last_price")
        prev = row.get("prev_close_price")
        session_quotes = _snapshot_session_quotes(row, fetched_at)
        if not any(item["available"] for item in session_quotes.values()):
            # どのセッションにも価格が無ければ、呼び出し側がYahooへ戻せるよう除外。
            continue
        change_pct = None
        if last is not None and prev:
            try:
                change_pct = (float(last) / float(prev) - 1) * 100
            except (TypeError, ZeroDivisionError):
                change_pct = None
        out[t] = {
            "name": row.get("name") or t,
            "price": last,
            "previous_close": prev,
            "change_percent": change_pct,
            "bid": row.get("bid_price"),
            "ask": row.get("ask_price"),
            "open": row.get("open_price"),
            "high": row.get("high_price"),
            "low": row.get("low_price"),
            "average_price": row.get("avg_price"),
            "volume": row.get("volume"),
            "turnover": row.get("turnover"),
            "turnover_rate": row.get("turnover_rate"),
            "amplitude": row.get("amplitude"),
            "per": row.get("pe_ttm_ratio") or row.get("pe_ratio"),
            "pbr": row.get("pb_ratio"),
            "dividend_yield": row.get("dividend_ratio_ttm"),
            "market_cap": row.get("total_market_val"),
            "eps": row.get("earning_per_share"),
            # セッション別データもsnapshot 1回から返す。購読も履歴K線枠も不要。
            "pre_price": row.get("pre_price"),
            "pre_high": row.get("pre_high_price"),
            "pre_low": row.get("pre_low_price"),
            "pre_volume": row.get("pre_volume"),
            "pre_change_percent": row.get("pre_change_rate"),
            "after_price": row.get("after_price"),
            "after_high": row.get("after_high_price"),
            "after_low": row.get("after_low_price"),
            "after_volume": row.get("after_volume"),
            "after_change_percent": row.get("after_change_rate"),
            "overnight_price": row.get("overnight_price"),
            "overnight_high": row.get("overnight_high_price"),
            "overnight_low": row.get("overnight_low_price"),
            "overnight_volume": row.get("overnight_volume"),
            "overnight_change_percent": row.get("overnight_change_rate"),
            "volume_ratio": row.get("volume_ratio"),
            "update_time": row.get("update_time"),
            "update_time_iso": session_quotes["regular"]["snapshot_updated_at"],
            "snapshot_fetched_at": fetched_at,
            "session_quotes": session_quotes,
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


@st.cache_data(ttl=900, show_spinner=False)
def economic_calendar(days: int = 7) -> pd.DataFrame:
    """米国の重要経済イベントを読み取り専用で取得する。

    履歴K線APIではないため月間K線枠を消費しない。SDK/OpenD/権限が未対応なら
    空DataFrameを返し、他の市場分析を止めない。
    """
    ctx = _ctx()
    if ctx is None or not hasattr(ctx, "get_economic_calendar"):
        return pd.DataFrame()
    try:
        from moomoo import EconomicImportance, Market
        start = pd.Timestamp.now(tz="America/New_York").date()
        end = start + pd.Timedelta(days=max(0, min(int(days), 30)))
        result = ctx.get_economic_calendar(
            begin_date=start.isoformat(), end_date=end.isoformat(),
            market_list=[Market.US], importance=EconomicImportance.HIGH, count=100)
    except Exception:
        return pd.DataFrame()
    if not isinstance(result, tuple) or len(result) < 2 or not _ok(result[0]):
        return pd.DataFrame()
    data = result[1]
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    columns = [c for c in ["timestamp", "country", "title", "star",
                           "previous", "consensus", "actual"] if c in data.columns]
    return data[columns].copy().reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def earnings_calendar(days: int = 7, count: int = 50) -> pd.DataFrame:
    """直近の米国株決算予定を時価総額順で取得する(読み取り専用)。"""
    ctx = _ctx()
    if ctx is None or not hasattr(ctx, "get_earnings_calendar"):
        return pd.DataFrame()
    try:
        from moomoo import EarningsCalendarSortType, Market
        start = pd.Timestamp.now(tz="America/New_York").date()
        # OpenAPIは1回につき最大7日幅。
        end = start + pd.Timedelta(days=max(0, min(int(days), 7)))
        ret, data = ctx.get_earnings_calendar(
            market=Market.US, sort_type=EarningsCalendarSortType.MARKET_CAP,
            begin_date=start.isoformat(), end_date=end.isoformat())
    except Exception:
        return pd.DataFrame()
    if not _ok(ret) or not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
    columns = [c for c in ["security", "name", "earnings_date", "pub_type",
                           "eps_predict", "revenue_predict", "iv", "iv_rank",
                           "market_cap", "price"] if c in data.columns]
    out = data[columns].copy().head(max(1, min(int(count), 100)))
    if "security" in out:
        out["ticker"] = out["security"].map(_strip_market)
    return out.reset_index(drop=True)


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
    """原資産のIV(予想変動率)とHV(実績変動率)を取得する。

    ``get_option_volatility`` はオプション*契約*コード専用であり、AAPL等の
    原資産コードを渡してはいけない。ここでは原資産専用のoverview/history
    APIをfeature-detectして使う。古いSDKで両APIが未提供なら、契約を推測せず
    ``None`` を返す（共有Quote contextは呼び出し側で使い回すためcloseしない）。
    """
    ctx = _ctx()
    code = to_code(ticker)
    if ctx is None or not code:
        return None

    overview_method = getattr(ctx, "get_option_underlying_overview", None)
    history_method = getattr(ctx, "get_option_underlying_his_volatility", None)
    if not callable(overview_method) and not callable(history_method):
        return None

    overview = pd.DataFrame()
    if callable(overview_method):
        try:
            ret, data = overview_method([code])
            if _ok(ret) and isinstance(data, pd.DataFrame) and not data.empty:
                overview = data.copy()
        except Exception:
            # overviewが未許可でも履歴APIだけ使える場合がある。
            overview = pd.DataFrame()

    history_parts: list[pd.DataFrame] = []
    if callable(history_method):
        end = pd.Timestamp.now(tz="America/New_York").normalize()
        begin = end - pd.Timedelta(days=180)
        page_req_key = None
        seen_page_keys = set()
        # APIの異常な循環paginationでUIを止めないため上限を設ける。
        for _ in range(20):
            try:
                result = history_method(
                    code,
                    begin_time=begin.strftime("%Y-%m-%d"),
                    end_time=end.strftime("%Y-%m-%d"),
                    page_req_key=page_req_key,
                )
            except Exception:
                break
            if not isinstance(result, tuple) or len(result) < 2 or not _ok(result[0]):
                break
            data = result[1]
            if isinstance(data, pd.DataFrame) and not data.empty:
                history_parts.append(data.copy())
            next_key = result[2] if len(result) >= 3 else None
            if next_key is None or next_key in seen_page_keys:
                break
            seen_page_keys.add(next_key)
            page_req_key = next_key

    history = (pd.concat(history_parts, ignore_index=True)
               if history_parts else pd.DataFrame())
    if history.empty and overview.empty:
        return None

    if history.empty:
        latest = overview.iloc[0]
        series = pd.DataFrame({
            "日付": [pd.Timestamp.now(tz="America/New_York").date().isoformat()],
            "IV": [pd.to_numeric(pd.Series([latest.get("iv")]), errors="coerce").iloc[0]],
            "HV": [pd.to_numeric(pd.Series([latest.get("hv_30d")]), errors="coerce").iloc[0]],
        })
    else:
        date_values = history.get("time")
        if date_values is None:
            date_values = pd.to_datetime(history.get("timestamp"), unit="s", errors="coerce")
        series = pd.DataFrame({
            "日付": date_values,
            "IV": pd.to_numeric(history.get("iv"), errors="coerce"),
            "HV": pd.to_numeric(history.get("hv"), errors="coerce"),
        })
    series["IVプレミアム"] = series["IV"] - series["HV"]
    series = series.dropna(subset=["IV", "HV"], how="all")
    if series.empty:
        return None
    sort_dates = pd.to_datetime(series["日付"], utc=True, errors="coerce")
    if sort_dates.notna().any():
        series = (series.assign(_sort_date=sort_dates)
                  .sort_values("_sort_date", kind="stable", na_position="last")
                  .drop(columns="_sort_date"))

    # overviewのcurrent snapshotが履歴末尾より新しい場合でも、同日重複を
    # 推測で追加せず、カード用の現在値としてだけ保持する。
    overview_row = overview.iloc[0] if not overview.empty else None
    average_iv = pd.to_numeric(series["IV"], errors="coerce").mean()
    return {
        "series": series.reset_index(drop=True),
        "average_iv": average_iv,
        "status": "available",
        "analysis": "",
        "source": "moomoo 原資産オプション統計",
        "code": code,
        "overview": overview_row.to_dict() if overview_row is not None else {},
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
    # 実機SDKの [used, remain, [detail...]] 形式も、履歴取得側と同じ規則で解釈する。
    from lib.moomoo_fetcher import _quota_parts
    used, remain, _ = _quota_parts(data)
    if used is None or remain is None:
        return None
    return {"used": used, "remain": remain}
