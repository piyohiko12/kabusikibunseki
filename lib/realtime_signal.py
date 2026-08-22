"""米国株のリアルタイム売買タイミングを整理する純粋関数。

このモジュールは、呼び出し側が取得済みの1分足、気配値、セッション情報だけを
使う。ネットワーク、moomoo OpenD、履歴K線、口座、注文APIへアクセスしない。
日足や日足から作った購入プランは、リアルタイム判定の条件・得点に使わない。

テクニカル得点には確定済み1分足だけを使う。形成中の足と現在気配は混ぜず、
現在気配は鮮度、スプレッド、1分足VWAPからの乖離、保有株の退出確認にだけ使う。
"""

from __future__ import annotations

from datetime import time
import math
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
MARKET_TZ = ZoneInfo("America/New_York")
REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
TRADING_SESSIONS = ("premarket", "regular", "afterhours", "overnight")

DEFAULT_CONFIG = {
    "interval_minutes": 1,
    "timestamp_semantics": "start",
    "min_completed_bars": 30,
    "min_benchmark_bars": 6,
    "max_bar_age_seconds": 90.0,
    "quote_max_age_seconds": 15.0,
    "max_future_skew_seconds": 5.0,
    "max_spread_pct": 0.20,
    "setup_score": 60.0,
    "entry_score": 75.0,
    "release_score": 65.0,
    "confirmation_bars": 2,
    "release_bars": 2,
    "alert_cooldown_bars": 5,
    "rsi_min": 50.0,
    "rsi_max": 72.0,
    "volume_ratio_min": 1.15,
    "relative_strength_min_pct": 0.0,
    "rms_max_pct": 0.25,
    "max_vwap_extension_atr": 1.0,
    "max_gap_intervals": 2.0,
    "actionable_sessions": TRADING_SESSIONS,
    "require_moomoo": True,
}

STATE_DISPLAY = {
    "DATA_WAIT": {
        "label_ja": "リアルタイム情報を待つ",
        "description_ja": "価格や確定1分足を安全に確認できるまで待ちます。",
        "tone": "neutral",
    },
    "WAIT": {
        "label_ja": "今は待つ",
        "description_ja": "取引時間、現在価格または安全条件の確認が残っています。",
        "tone": "warning",
    },
    "NEUTRAL": {
        "label_ja": "短期の買い条件は未成立",
        "description_ja": "確定1分足の上向き条件がまだ足りません。",
        "tone": "neutral",
    },
    "BUY_SETUP": {
        "label_ja": "買い条件を確認中",
        "description_ja": "条件が整い始めています。確定足での継続を確認します。",
        "tone": "info",
    },
    "BUY_READY": {
        "label_ja": "買いを検討できる",
        "description_ja": "短期条件が、異なる確定1分足で続けて成立しています。",
        "tone": "positive",
    },
    "HOLD": {
        "label_ja": "保有を監視",
        "description_ja": "現在の気配では損切り・利益確定価格に達していません。",
        "tone": "info",
    },
    "TAKE_PROFIT": {
        "label_ja": "保有株の利益確定を確認",
        "description_ja": "新しい空売りではなく、保有株の利益確定条件です。",
        "tone": "info",
    },
    "RISK_EXIT": {
        "label_ja": "保有株のリスク退出を確認",
        "description_ja": "損失抑制条件を最優先で確認してください。",
        "tone": "danger",
    },
}


def _number(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number >= minimum else None


def _mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _timestamp(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(stamp):
        return None
    try:
        if stamp.tzinfo is None:
            return stamp.tz_localize(MARKET_TZ)
        return stamp.tz_convert(MARKET_TZ)
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str | None:
    stamp = _timestamp(value)
    return stamp.isoformat() if stamp is not None else None


def realtime_config(overrides: Mapping[str, Any] | None = None) -> dict:
    """設定を正規化し、安全性を崩す矛盾した値を拒否する。"""
    cfg = {**DEFAULT_CONFIG, **_mapping(overrides)}

    integer_keys = {
        "interval_minutes": 1,
        "min_completed_bars": 1,
        "min_benchmark_bars": 1,
        "confirmation_bars": 1,
        "release_bars": 1,
        "alert_cooldown_bars": 0,
    }
    for key, minimum in integer_keys.items():
        value = _integer(cfg.get(key), minimum=minimum)
        if value is None:
            raise ValueError(f"{key} は {minimum} 以上の整数で指定してください")
        cfg[key] = value

    positive_keys = (
        "max_bar_age_seconds", "quote_max_age_seconds",
        "max_spread_pct", "max_gap_intervals",
    )
    for key in positive_keys:
        value = _number(cfg.get(key), positive=True)
        if value is None:
            raise ValueError(f"{key} は0より大きい数値で指定してください")
        cfg[key] = value

    nonnegative_keys = (
        "max_future_skew_seconds", "setup_score", "entry_score",
        "release_score", "rsi_min", "rsi_max", "volume_ratio_min",
        "rms_max_pct", "max_vwap_extension_atr",
    )
    for key in nonnegative_keys:
        value = _number(cfg.get(key))
        if value is None or value < 0:
            raise ValueError(f"{key} は0以上の数値で指定してください")
        cfg[key] = value
    relative = _number(cfg.get("relative_strength_min_pct"))
    if relative is None:
        raise ValueError("relative_strength_min_pct は有限な数値で指定してください")
    cfg["relative_strength_min_pct"] = relative

    if not cfg["setup_score"] <= cfg["release_score"] < cfg["entry_score"]:
        raise ValueError("setup_score <= release_score < entry_score が必要です")
    if cfg["rsi_min"] >= cfg["rsi_max"]:
        raise ValueError("rsi_min は rsi_max 未満で指定してください")
    semantics = str(cfg.get("timestamp_semantics") or "").strip().lower()
    if semantics not in {"start", "end"}:
        raise ValueError("timestamp_semantics は start または end を指定してください")
    cfg["timestamp_semantics"] = semantics

    sessions = cfg.get("actionable_sessions")
    if isinstance(sessions, str):
        sessions = (sessions,)
    if not isinstance(sessions, (list, tuple, set, frozenset)):
        raise ValueError("actionable_sessions はセッション名の配列で指定してください")
    normalised_sessions = tuple(dict.fromkeys(
        str(item).strip().lower() for item in sessions if str(item).strip()))
    if not normalised_sessions:
        raise ValueError("actionable_sessions は1つ以上指定してください")
    unknown_sessions = tuple(
        item for item in normalised_sessions if item not in TRADING_SESSIONS)
    if unknown_sessions:
        raise ValueError(
            "未対応の取引セッションです: " + ", ".join(unknown_sessions))
    cfg["actionable_sessions"] = normalised_sessions
    cfg["require_moomoo"] = bool(cfg.get("require_moomoo", True))
    return cfg


def _session_name(session: Mapping[str, Any] | None,
                  now_et: pd.Timestamp) -> str:
    name = str(_mapping(session).get("session") or "").strip().lower()
    if name:
        return name
    wall = now_et.time().replace(tzinfo=None)
    if wall < time(4, 0) or wall >= time(20, 0):
        return "overnight"
    if wall < time(9, 30):
        return "premarket"
    if wall < time(16, 0):
        return "regular"
    return "afterhours"


def _wall_time(value: Any, fallback: time) -> time:
    if isinstance(value, time):
        return value.replace(tzinfo=None)
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError):
        stamp = None
    if stamp is not None and not pd.isna(stamp):
        try:
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert(MARKET_TZ)
            return stamp.time().replace(tzinfo=None)
        except (TypeError, ValueError):
            pass
    text = str(value or "").strip()
    if text:
        try:
            return time.fromisoformat(text).replace(tzinfo=None)
        except ValueError:
            pass
    return fallback


def _session_bounds(name: str, now_et: pd.Timestamp,
                    session: Mapping[str, Any] | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = now_et.normalize()
    source = _mapping(session)
    if name == "regular":
        close = _wall_time(source.get("regular_close"), time(16, 0))
        return day + pd.Timedelta(hours=9, minutes=30), day + pd.Timedelta(
            hours=close.hour, minutes=close.minute)
    if name == "premarket":
        return day + pd.Timedelta(hours=4), day + pd.Timedelta(hours=9, minutes=30)
    if name == "afterhours":
        regular_close = _wall_time(source.get("regular_close"), time(16, 0))
        close = _wall_time(source.get("afterhours_close"), time(20, 0))
        return day + pd.Timedelta(
            hours=regular_close.hour, minutes=regular_close.minute), day + pd.Timedelta(
            hours=close.hour, minutes=close.minute)
    if name == "overnight":
        if now_et.time().replace(tzinfo=None) >= time(20, 0):
            return day + pd.Timedelta(hours=20), day + pd.Timedelta(days=1, hours=4)
        return day - pd.Timedelta(hours=4), day + pd.Timedelta(hours=4)
    return day, day + pd.Timedelta(days=1)


def completed_intraday_bars(
    bars: pd.DataFrame | None,
    *,
    now: Any,
    session: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """現在形成中の足を除き、現在セッションの確定足と品質情報を返す。"""
    cfg = realtime_config(config)
    now_et = _timestamp(now)
    issues: list[dict] = []

    def issue(code: str, detail: str) -> None:
        if code not in {item["code"] for item in issues}:
            issues.append({"code": code, "detail_ja": detail})

    empty = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    if now_et is None:
        issue("now_invalid", "現在時刻を解釈できません")
        return empty, {
            "valid": False, "issues": issues, "session": "unknown",
            "session_key": None, "completed_count": 0,
            "dropped_incomplete": 0, "latest_bar_time": None,
            "latest_bar_end": None, "age_seconds": None,
        }
    if bars is None or not isinstance(bars, pd.DataFrame) or bars.empty:
        issue("bars_missing", "1分足を取得できません")
        return empty, {
            "valid": False, "issues": issues,
            "session": _session_name(session, now_et),
            "session_key": None, "completed_count": 0,
            "dropped_incomplete": 0, "latest_bar_time": None,
            "latest_bar_end": None, "age_seconds": None,
        }

    missing = [column for column in REQUIRED_COLUMNS if column not in bars.columns]
    if missing:
        issue("columns_missing", "必要な列がありません: " + ", ".join(missing))
        return empty, {
            "valid": False, "issues": issues,
            "session": _session_name(session, now_et),
            "session_key": None, "completed_count": 0,
            "dropped_incomplete": 0, "latest_bar_time": None,
            "latest_bar_end": None, "age_seconds": None,
        }

    try:
        index = pd.DatetimeIndex(pd.to_datetime(bars.index, errors="coerce"))
    except (TypeError, ValueError):
        index = pd.DatetimeIndex([])
    if len(index) != len(bars) or index.isna().any():
        issue("timestamps_invalid", "1分足の時刻を解釈できません")
        return empty, {
            "valid": False, "issues": issues,
            "session": _session_name(session, now_et),
            "session_key": None, "completed_count": 0,
            "dropped_incomplete": 0, "latest_bar_time": None,
            "latest_bar_end": None, "age_seconds": None,
        }
    try:
        if index.tz is None:
            index = index.tz_localize(MARKET_TZ)
        else:
            index = index.tz_convert(MARKET_TZ)
    except (TypeError, ValueError):
        issue("timestamps_invalid", "1分足のtimezoneを解釈できません")
        return empty, {
            "valid": False, "issues": issues,
            "session": _session_name(session, now_et),
            "session_key": None, "completed_count": 0,
            "dropped_incomplete": 0, "latest_bar_time": None,
            "latest_bar_end": None, "age_seconds": None,
        }

    if not index.is_monotonic_increasing:
        issue("timestamps_not_monotonic", "1分足の時刻が昇順ではありません")
    if index.has_duplicates:
        issue("timestamps_duplicated", "同じ時刻の1分足が重複しています")

    frame = bars.loc[:, list(REQUIRED_COLUMNS)].copy()
    frame.index = index
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    name = _session_name(session, now_et)
    start, end = _session_bounds(name, now_et, session)
    interval = pd.Timedelta(minutes=cfg["interval_minutes"])
    future = frame.index > now_et + pd.Timedelta(seconds=cfg["max_future_skew_seconds"])
    if bool(future.any()):
        issue("future_bar", "現在時刻より未来の1分足が含まれています")

    in_session = (frame.index >= start) & (frame.index < end)
    if cfg["timestamp_semantics"] == "start":
        completed = frame.index + interval <= now_et
        bar_ends = frame.index + interval
    else:
        completed = frame.index <= now_et
        bar_ends = frame.index
    dropped_incomplete = int((in_session & ~completed).sum())
    frame = frame.loc[in_session & completed]
    completed_ends = bar_ends[in_session & completed]

    # 形成中の足は価格が変化中で欠損を含む場合もあるため、品質判定にも混ぜない。
    # 以降の検証対象を、実際に特徴量へ渡す確定足だけに限定する。
    values = frame.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        issue("ohlcv_non_finite", "確定足の価格または出来高に欠損・無限値があります")
    if not frame.empty:
        ohlc = frame.loc[:, ("Open", "High", "Low", "Close")]
        if (ohlc <= 0).any().any():
            issue("price_non_positive", "確定足に0以下の価格が含まれています")
        if (frame["Volume"] < 0).any():
            issue("volume_negative", "確定足にマイナスの出来高が含まれています")
        invalid_high = frame["High"] < frame[["Open", "Close", "Low"]].max(axis=1)
        invalid_low = frame["Low"] > frame[["Open", "Close", "High"]].min(axis=1)
        if bool(invalid_high.any() or invalid_low.any()):
            issue("ohlc_inconsistent", "確定足の高値・安値と始値・終値の関係が不正です")

    latest_time = frame.index[-1] if not frame.empty else None
    latest_end = completed_ends[-1] if len(completed_ends) else None
    age = None if latest_end is None else float((now_et - latest_end).total_seconds())
    if age is not None and age < -cfg["max_future_skew_seconds"]:
        issue("bar_time_inconsistent", "最新確定足の時刻が現在時刻より未来です")
    if age is not None and age > cfg["max_bar_age_seconds"]:
        issue("bar_stale", f"最新確定足から{age:.0f}秒経過しています")

    if len(frame) >= 2:
        gaps = frame.index.to_series().diff().dropna().dt.total_seconds()
        max_allowed = cfg["interval_minutes"] * 60 * cfg["max_gap_intervals"]
        if not gaps.empty and float(gaps.max()) > max_allowed:
            issue("bar_gap", "確定1分足の間隔に大きな欠損があります")

    session_day = start.date().isoformat()
    return frame, {
        "valid": not issues,
        "issues": issues,
        "session": name,
        "session_key": f"{session_day}:{name}",
        "completed_count": int(len(frame)),
        "dropped_incomplete": dropped_incomplete,
        "latest_bar_time": _iso(latest_time),
        "latest_bar_end": _iso(latest_end),
        "age_seconds": age,
    }


def _rsi(prices: pd.Series, period: int = 14) -> float | None:
    if len(prices) < period + 1:
        return None
    delta = prices.astype(float).diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_gain = _number(gain.iloc[-1])
    avg_loss = _number(loss.iloc[-1])
    if avg_gain is None or avg_loss is None:
        return None
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _atr(frame: pd.DataFrame, period: int = 14) -> float | None:
    if len(frame) < period + 1:
        return None
    previous = frame["Close"].shift()
    true_range = pd.concat([
        frame["High"] - frame["Low"],
        (frame["High"] - previous).abs(),
        (frame["Low"] - previous).abs(),
    ], axis=1).max(axis=1)
    return _number(true_range.tail(period).mean(), positive=True)


def _relative_strength(target: pd.DataFrame, benchmark: pd.DataFrame,
                       periods: int = 5) -> float | None:
    if target.empty or benchmark.empty:
        return None
    common = target.index.intersection(benchmark.index)
    needed = periods + 1
    if len(common) < needed or common[-1] != target.index[-1]:
        return None
    use = common[-needed:]
    target_base = _number(target.loc[use[0], "Close"], positive=True)
    target_last = _number(target.loc[use[-1], "Close"], positive=True)
    bench_base = _number(benchmark.loc[use[0], "Close"], positive=True)
    bench_last = _number(benchmark.loc[use[-1], "Close"], positive=True)
    if None in (target_base, target_last, bench_base, bench_last):
        return None
    return ((target_last / target_base - 1.0)
            - (bench_last / bench_base - 1.0)) * 100.0


def calculate_intraday_features(
    bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
) -> dict:
    """確定済み・同一セッションの足だけから因果的な特徴量を返す。"""
    if bars is None or bars.empty:
        return {
            "last_close": None, "ema5": None, "ema13": None,
            "ema5_slope": None, "vwap": None, "momentum_5m_pct": None,
            "rsi14": None, "volume_ratio": None,
            "relative_strength_5m_pct": None, "rms_12_pct": None,
            "atr14": None,
        }
    close = bars["Close"].astype(float)
    volume = bars["Volume"].astype(float)
    ema5_series = close.ewm(span=5, adjust=False).mean()
    ema13_series = close.ewm(span=13, adjust=False).mean()
    typical = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    positive_volume = volume.clip(lower=0)
    total_volume = float(positive_volume.sum())
    vwap = (None if total_volume <= 0 else
            _number((typical * positive_volume).sum() / total_volume, positive=True))

    momentum = None
    if len(close) >= 6 and float(close.iloc[-6]) > 0:
        momentum = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1.0) * 100.0

    volume_ratio = None
    if len(volume) >= 21:
        prior = volume.iloc[-21:-1]
        prior = prior[prior > 0]
        if not prior.empty:
            median = float(prior.median())
            if median > 0:
                volume_ratio = float(volume.iloc[-1]) / median

    returns = close.pct_change().dropna().tail(12)
    rms = (None if returns.empty else
           _number(float(np.sqrt(np.mean(np.square(returns.to_numpy(float))))) * 100.0))

    return {
        "last_close": _number(close.iloc[-1], positive=True),
        "ema5": _number(ema5_series.iloc[-1], positive=True),
        "ema13": _number(ema13_series.iloc[-1], positive=True),
        "ema5_slope": (_number(ema5_series.iloc[-1] - ema5_series.iloc[-2])
                       if len(ema5_series) >= 2 else None),
        "vwap": vwap,
        "momentum_5m_pct": _number(momentum),
        "rsi14": _rsi(close),
        "volume_ratio": _number(volume_ratio),
        "relative_strength_5m_pct": _relative_strength(bars, benchmark_bars),
        "rms_12_pct": rms,
        "atr14": _atr(bars),
    }


def _technical_checks(features: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict]:
    checks: list[dict] = []

    def add(key: str, label: str, points: int, actual: float | None,
            passed: bool | None, requirement: str, *, required: bool = False) -> None:
        checks.append({
            "key": key, "label_ja": label, "points": int(points),
            "actual": _number(actual), "passed": passed,
            "required": bool(required), "requirement_ja": requirement,
        })

    ema5 = _number(features.get("ema5"), positive=True)
    ema13 = _number(features.get("ema13"), positive=True)
    slope = _number(features.get("ema5_slope"))
    ema_available = None not in (ema5, ema13, slope)
    add("ema_alignment", "EMA5がEMA13より上で上向き", 25, ema5,
        None if not ema_available else bool(ema5 > ema13 and slope > 0),
        "EMA5 > EMA13、かつEMA5の傾きが正", required=True)

    close = _number(features.get("last_close"), positive=True)
    vwap = _number(features.get("vwap"), positive=True)
    add("above_vwap", "確定終値がセッションVWAPより上", 15, close,
        None if close is None or vwap is None else bool(close > vwap),
        "確定終値 > セッションVWAP", required=True)

    momentum = _number(features.get("momentum_5m_pct"))
    add("momentum_5m", "5分モメンタムが正", 15, momentum,
        None if momentum is None else bool(momentum > 0), "0%より大きい")

    rsi = _number(features.get("rsi14"))
    add("rsi14", "RSI(14)が過熱しすぎていない", 15, rsi,
        None if rsi is None else bool(cfg["rsi_min"] <= rsi <= cfg["rsi_max"]),
        f"{cfg['rsi_min']:.0f}〜{cfg['rsi_max']:.0f}")

    ratio = _number(features.get("volume_ratio"))
    add("volume_ratio", "相対出来高が増えている", 15, ratio,
        None if ratio is None else bool(ratio >= cfg["volume_ratio_min"]),
        f"{cfg['volume_ratio_min']:.2f}倍以上")

    relative = _number(features.get("relative_strength_5m_pct"))
    add("relative_strength", "5分でSPYより強い", 10, relative,
        None if relative is None else bool(relative > cfg["relative_strength_min_pct"]),
        f"{cfg['relative_strength_min_pct']:+.2f}%より大きい", required=True)

    rms = _number(features.get("rms_12_pct"))
    add("rms", "短期変動が大きすぎない", 5, rms,
        None if rms is None else bool(rms <= cfg["rms_max_pct"]),
        f"{cfg['rms_max_pct']:.2f}%以下")
    return checks


def _gate(key: str, label: str, passed: bool | None, reason: str,
          *, category: str, blocking: bool = True) -> dict:
    return {
        "key": key, "label_ja": label, "passed": passed,
        "blocking": bool(blocking), "category": category,
        "reason_ja": reason,
    }


def _suspension(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().upper()
    if text in {"1", "TRUE", "YES", "Y", "SUSPENDED", "SUSPEND"}:
        return True
    if text in {"0", "FALSE", "NO", "N", "NORMAL", "NONE",
                "NOT_SUSPENDED", "UNSUSPENDED"}:
        return False
    return None


def _quote(snapshot: Mapping[str, Any] | None, now_et: pd.Timestamp,
           cfg: Mapping[str, Any]) -> tuple[dict, list[dict]]:
    source = _mapping(snapshot)
    bid = _number(source.get("bid"), positive=True)
    ask = _number(source.get("ask"), positive=True)
    last = _number(source.get("price"), positive=True)
    updated = _timestamp(source.get("update_time"))
    age = None if updated is None else float((now_et - updated).total_seconds())
    spread = None
    if bid is not None and ask is not None and ask >= bid:
        midpoint = (ask + bid) / 2.0
        if midpoint > 0:
            spread = (ask - bid) / midpoint * 100.0
    moomoo = str(source.get("source") or "").strip().lower() == "moomoo openapi"
    time_valid = (age is not None
                  and age >= -cfg["max_future_skew_seconds"]
                  and age <= cfg["quote_max_age_seconds"])
    suspended = _suspension(source.get("suspension"))
    gates = [
        _gate("quote_source", "リアルタイム価格の取得元",
              moomoo if cfg["require_moomoo"] else bool(last or bid or ask),
              ("moomoo OpenAPIの気配値です" if moomoo else
               "moomoo OpenAPIの気配値を確認できません"), category="DATA"),
        _gate("quote_freshness", "気配値の鮮度", time_valid,
              ("更新時刻を確認できません" if age is None else
               f"更新から{age:.0f}秒 / 上限{cfg['quote_max_age_seconds']:.0f}秒"),
              category="DATA"),
        _gate("bid_ask", "売買気配", bid is not None and ask is not None and ask >= bid,
              ("Bid/Askを確認できません" if spread is None else
               f"Bid {bid:.4f} / Ask {ask:.4f}"), category="DATA"),
    ]
    return {
        "bid": bid, "ask": ask, "last": last,
        "update_time": _iso(updated), "age_seconds": age,
        "spread_pct": spread, "source": str(source.get("source") or ""),
        "suspended": suspended,
        # 保有中の退出判定はBidだけで行えるが、取引停止状態が不明またはTrueなら
        # そのBidを実行可能な価格として扱わない。
        "fresh_bid": bool(
            moomoo and time_valid and bid is not None and suspended is False),
    }, gates


def _position_prices(position: Mapping[str, Any] | None) -> dict:
    """利用者が明示した保有価格だけを正規化する。"""
    holding = _mapping(position)
    entry = _number(
        holding.get("entry_price", holding.get("entry")), positive=True)
    stop = _number(holding.get("stop_price", holding.get("stop")), positive=True)
    target = _number(holding.get("target_price", holding.get("target")), positive=True)
    return {"entry": entry, "stop": stop, "target": target,
            "maximum_buy_price": None, "minimum_rr": None}


def _holding_prices_valid(entry: float | None, stop: float | None,
                          target: float | None) -> tuple[bool, str]:
    if stop is not None and target is not None and stop >= target:
        return False, "損切り価格は利益確定価格より低く設定してください"
    if entry is not None and stop is not None and stop >= entry:
        return False, "損切り価格は取得価格より低く設定してください"
    if entry is not None and target is not None and entry >= target:
        return False, "利益確定価格は取得価格より高く設定してください"
    return True, "入力した保有株の価格関係を確認しました"


def _position_is_open(position: Mapping[str, Any] | None) -> bool:
    source = _mapping(position)
    for key in ("held", "is_open"):
        explicit = source.get(key)
        if isinstance(explicit, bool):
            return explicit
    return _number(source.get("entry_price"), positive=True) is not None


def _memory(previous: Mapping[str, Any] | None, session_key: str | None) -> dict:
    source = _mapping(previous)
    if source.get("session_key") != session_key:
        source = {}
    stable = str(source.get("stable_state") or "NEUTRAL").upper()
    if stable not in STATE_DISPLAY:
        stable = "NEUTRAL"
    latched = str(source.get("latched_state") or "").upper()
    if latched != "BUY_READY":
        latched = None
    return {
        "session_key": session_key,
        "last_processed_bar_time": (
            str(source.get("last_processed_bar_time"))
            if source.get("last_processed_bar_time") else None),
        "stable_state": stable,
        "latched_state": latched,
        "pending_state": (
            "BUY_READY" if str(source.get("pending_state") or "").upper()
            == "BUY_READY" else None),
        "pending_count": _integer(source.get("pending_count"), minimum=0) or 0,
        "release_count": _integer(source.get("release_count"), minimum=0) or 0,
        "cooldown_remaining": (
            _integer(source.get("cooldown_remaining"), minimum=0) or 0),
        "last_alert_bar_time": (
            str(source.get("last_alert_bar_time"))
            if source.get("last_alert_bar_time") else None),
        "peak_bid": _number(source.get("peak_bid"), positive=True),
    }


def _consecutive(previous_time: str | None, current_time: str | None,
                 interval_minutes: int) -> bool:
    previous = _timestamp(previous_time)
    current = _timestamp(current_time)
    if previous is None or current is None:
        return False
    return current - previous == pd.Timedelta(minutes=interval_minutes)


def _entry_state(score: float, required_passed: bool, signal_time: str | None,
                 session_key: str | None, previous: Mapping[str, Any] | None,
                 cfg: Mapping[str, Any], *, process_bar: bool) -> tuple[str, dict, dict]:
    memory = _memory(previous, session_key)
    previous_latched = memory["latched_state"]
    previous_stable = memory["stable_state"]
    new_bar = bool(process_bar and signal_time
                   and signal_time != memory["last_processed_bar_time"])
    if new_bar and memory["cooldown_remaining"] > 0:
        memory["cooldown_remaining"] -= 1

    raw_ready = required_passed and score >= cfg["entry_score"]
    raw_setup = score >= cfg["setup_score"]
    state = "BUY_SETUP" if raw_setup else "NEUTRAL"

    if new_bar:
        if not required_passed:
            memory.update({
                "latched_state": None, "pending_state": None,
                "pending_count": 0, "release_count": 0,
            })
        elif raw_ready:
            memory["release_count"] = 0
            if memory["latched_state"] == "BUY_READY":
                state = "BUY_READY"
                memory["pending_state"] = None
                memory["pending_count"] = 0
            else:
                consecutive = (
                    memory["pending_state"] == "BUY_READY"
                    and _consecutive(memory["last_processed_bar_time"], signal_time,
                                     cfg["interval_minutes"])
                )
                memory["pending_state"] = "BUY_READY"
                memory["pending_count"] = (
                    memory["pending_count"] + 1 if consecutive else 1)
                if memory["pending_count"] >= cfg["confirmation_bars"]:
                    memory["latched_state"] = "BUY_READY"
                    memory["pending_state"] = None
                    memory["pending_count"] = 0
                    state = "BUY_READY"
                else:
                    state = "BUY_SETUP"
        elif memory["latched_state"] == "BUY_READY":
            memory["pending_state"] = None
            memory["pending_count"] = 0
            if score >= cfg["release_score"]:
                memory["release_count"] = 0
                state = "BUY_READY"
            else:
                memory["release_count"] += 1
                if memory["release_count"] >= cfg["release_bars"]:
                    memory["latched_state"] = None
                    memory["release_count"] = 0
                    state = "BUY_SETUP" if raw_setup else "NEUTRAL"
                else:
                    # 解除条件も確定足で連続確認し、1本の揺れでは表示を反転させない。
                    state = "BUY_READY"
        else:
            memory.update({
                "pending_state": None, "pending_count": 0,
                "release_count": 0,
            })
        memory["last_processed_bar_time"] = signal_time
    else:
        if not required_passed:
            state = "BUY_SETUP" if raw_setup else "NEUTRAL"
        elif memory["latched_state"] == "BUY_READY":
            state = "BUY_READY"
        elif raw_ready or raw_setup:
            state = "BUY_SETUP"

    latch_activated = previous_latched != "BUY_READY" and memory["latched_state"] == "BUY_READY"
    emit = bool(latch_activated and memory["cooldown_remaining"] == 0
                and memory["last_alert_bar_time"] != signal_time)
    if emit:
        memory["cooldown_remaining"] = cfg["alert_cooldown_bars"]
        memory["last_alert_bar_time"] = signal_time
    memory["stable_state"] = state
    confirmation = {
        "phase": ("confirmed" if state == "BUY_READY" else
                  "confirming" if raw_ready else
                  "release_pending" if previous_latched == "BUY_READY" else "idle"),
        "streak": int(memory["pending_count"]),
        "required": int(cfg["confirmation_bars"]),
        "remaining": max(int(cfg["confirmation_bars"] - memory["pending_count"]), 0),
    }
    alert = {
        "emit": emit, "kind": "BUY_READY" if emit else None,
        "cooldown_remaining": int(memory["cooldown_remaining"]),
    }
    return state, memory, {"confirmation": confirmation, "alert": alert,
                           "previous_state": previous_stable}


def _display(state: str, reason: str | None = None) -> tuple[str, str, str]:
    source = STATE_DISPLAY[state]
    return source["label_ja"], str(reason or source["description_ja"]), source["tone"]


def evaluate_realtime_signal(
    symbol: str,
    bars: pd.DataFrame,
    *,
    benchmark_bars: pd.DataFrame | None = None,
    snapshot: Mapping[str, Any] | None = None,
    session: Mapping[str, Any] | None = None,
    purchase_plan: Mapping[str, Any] | None = None,
    now: Any,
    position: Mapping[str, Any] | None = None,
    previous_memory: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict:
    """確定1分足と現在気配から、表示専用の安定化済み状態を返す。

    同じ確定足を繰り返し評価しても確認回数は増えない。返却された ``memory`` を
    次回の ``previous_memory`` へそのまま渡すと、2本確認と通知cooldownが働く。
    ``purchase_plan`` は後方互換性のため残しているが、判定には使用しない。
    """
    cfg = realtime_config(config)
    now_et = _timestamp(now)
    if now_et is None:
        raise ValueError("nowは有効な日時で指定してください")

    completed, bar_meta = completed_intraday_bars(
        bars, now=now_et, session=session, config=cfg)
    benchmark, benchmark_meta = completed_intraday_bars(
        benchmark_bars, now=now_et, session=session, config=cfg)
    signal_time = bar_meta.get("latest_bar_time")
    session_key = bar_meta.get("session_key")
    memory = _memory(previous_memory, session_key)

    features = calculate_intraday_features(completed, benchmark)
    checks = _technical_checks(features, cfg)
    score = float(sum(item["points"] for item in checks if item["passed"] is True))
    required_unavailable = any(
        item["required"] and item["passed"] is None for item in checks)
    required_passed = all(
        item["passed"] is True for item in checks if item["required"])
    raw_state = ("BUY_READY" if required_passed and score >= cfg["entry_score"] else
                 "BUY_SETUP" if score >= cfg["setup_score"] else "NEUTRAL")

    quote, quote_gates = _quote(snapshot, now_et, cfg)
    position_open = _position_is_open(position)
    position_prices = _position_prices(position)
    session_info = _mapping(session)
    session_name = str(session_info.get("session") or bar_meta.get("session") or "unknown").lower()
    # OpenDのmarket_stateは短時間キャッシュされるため、呼出側が同時刻の
    # カレンダー判定を添えた場合は、同じセッションを示すときだけ実行可能にする。
    calendar_session_name = str(
        session_info.get("calendar_session") or session_name).lower()
    supplied_snapshot = _mapping(snapshot)
    decision_session = str(
        supplied_snapshot.get("decision_session") or "").strip().lower()
    session_quote_ready = (
        supplied_snapshot.get("decision_ready") is True
        and supplied_snapshot.get("uses_daily_bars") is False
        and decision_session == session_name
    )
    session_quote_gate = _gate(
        "session_quote", "現在セッションの気配",
        session_quote_ready,
        (f"{session_name}の1分足と気配時刻を照合済みです"
         if session_quote_ready else
         "現在セッションの1分足と気配時刻を照合できません"),
        category="DATA",
    )
    holding_entry = position_prices["entry"]
    holding_prices_valid, holding_prices_reason = _holding_prices_valid(
        holding_entry, position_prices["stop"], position_prices["target"])

    data_gates = [
        _gate("bars_valid", "確定1分足の品質", bool(bar_meta.get("valid")),
              ("確定1分足を利用できます" if bar_meta.get("valid") else
               "; ".join(item["detail_ja"] for item in bar_meta.get("issues") or [])),
              category="DATA"),
        _gate("bar_count", "確定1分足の本数",
              len(completed) >= cfg["min_completed_bars"],
              f"{len(completed)}本 / 必要{cfg['min_completed_bars']}本", category="DATA"),
        _gate("benchmark_valid", "SPYの確定1分足",
              bool(benchmark_meta.get("valid"))
              and len(benchmark) >= cfg["min_benchmark_bars"],
              (f"{len(benchmark)}本 / 必要{cfg['min_benchmark_bars']}本" if benchmark is not None
               else "SPYの1分足を取得できません"), category="DATA"),
        *quote_gates,
        session_quote_gate,
    ]
    if required_unavailable:
        data_gates.append(_gate(
            "required_feature", "必須の短期指標", False,
            "EMA、VWAPまたはSPY比を計算できません", category="DATA"))

    suspended = quote.get("suspended")
    spread = _number(quote.get("spread_pct"))
    ask = _number(quote.get("ask"), positive=True)
    bid = _number(quote.get("bid"), positive=True)
    stop = position_prices["stop"]
    target = position_prices["target"]
    session_aligned = (
        session_name == calendar_session_name
        and session_name in cfg["actionable_sessions"]
    )

    safety_gates = [
        _gate("session", "現在の取引セッション",
              session_info.get("tradable") is True,
              str(session_info.get("reason") or "取引可能な状態を確認できません"),
              category="SESSION"),
        _gate("actionable_session", "リアルタイム判定の対象時間",
              session_aligned,
              (f"{session_name}は判定対象です"
               if session_aligned
               else (f"市場状態は{session_name}、取引カレンダーは"
                     f"{calendar_session_name}のため判定を保留します")),
              category="SESSION"),
        _gate("suspension", "取引停止状態", suspended is False,
              ("取引停止ではありません" if suspended is False else
               "取引停止状態を確認できません" if suspended is None else "取引停止中です"),
              category="SESSION"),
        _gate("spread", "Bid/Askの価格差",
              spread is not None and spread <= cfg["max_spread_pct"],
              ("スプレッドを計算できません" if spread is None else
               f"{spread:.3f}% / 上限{cfg['max_spread_pct']:.3f}%"),
              category="LIQUIDITY"),
    ]
    vwap = _number(features.get("vwap"), positive=True)
    atr = _number(features.get("atr14"), positive=True)
    chase_limit = (None if vwap is None or atr is None else
                   vwap + cfg["max_vwap_extension_atr"] * atr)
    safety_gates.append(_gate(
        "vwap_extension", "VWAPからの追い掛け幅",
        ask is not None and chase_limit is not None and ask <= chase_limit,
        ("追い掛け幅を計算できません" if ask is None or chase_limit is None else
         f"Ask {ask:.4f} / 上限 {chase_limit:.4f}"), category="PRICE"))

    # 保有時の価格目安は新規買いとは別に見せる。未設定を推測で補わず、
    # HOLDのままどちらを監視できていないかをUIが明示できるようにする。
    holding_gates = [
        _gate("holding_price_relation", "保有株の価格設定",
              holding_prices_valid, holding_prices_reason,
              category="RISK"),
        _gate("holding_monitor_level", "保有株の監視価格",
              stop is not None or target is not None,
              ("損切りまたは利益確定価格を確認しました"
               if stop is not None or target is not None else
               "監視する損切り価格または利益確定価格を入力してください"),
              category="RISK"),
        _gate("holding_bid", "保有株の売却気配",
              bid is not None,
              (f"Bid {bid:.4f}" if bid is not None else
               "保有株のBidを確認できません"),
              category="DATA"),
        _gate("holding_stop", "保有株の損切り目安", stop is not None,
              (f"損切り目安 {stop:.4f}" if stop is not None else
               "損切り価格を確認できません"),
              category="RISK", blocking=False),
        _gate("holding_target", "保有株の利益確定目安", target is not None,
              (f"利益確定目安 {target:.4f}" if target is not None else
               "利益確定価格を確認できません"),
              category="RISK", blocking=False),
    ] if position_open else []

    if position_open:
        # 保有中の判断に、新規買い条件・SPY比較などを混ぜない。
        # 気配値、取引可否、停止状態、保有株の価格目安だけを利用者へ示す。
        holding_gate_keys = {"session", "actionable_session", "suspension"}
        holding_quote_gates = [
            item for item in quote_gates
            if item["key"] in {"quote_source", "quote_freshness"}
        ]
        holding_quote_gates.append(session_quote_gate)
        gates = [
            *holding_quote_gates,
            *(item for item in safety_gates if item["key"] in holding_gate_keys),
            *holding_gates,
        ]
    else:
        gates = [*data_gates, *safety_gates]
    data_blockers = [item for item in data_gates
                     if item["blocking"] and item["passed"] is not True]
    safety_blockers = [item for item in safety_gates
                       if item["blocking"] and item["passed"] is not True]
    holding_blockers = [item for item in gates
                        if item["blocking"] and item["passed"] is not True]

    # 保有中は、利用者が指定した価格とfresh bidだけで退出目安を確認する。
    exit_reason = None
    if position_open and not holding_prices_valid:
        state = "DATA_WAIT"
        exit_reason = holding_prices_reason
    elif position_open and stop is None and target is None:
        state = "DATA_WAIT"
        exit_reason = "監視する損切り価格または利益確定価格を入力してください"
    elif position_open and quote["fresh_bid"] and bid is not None:
        previous_peak = _number(memory.get("peak_bid"), positive=True)
        supplied_peak = _number(_mapping(position).get("peak_price"), positive=True)
        memory["peak_bid"] = max(
            value for value in (bid, previous_peak, supplied_peak) if value is not None)
        if stop is not None and bid <= stop:
            state = "RISK_EXIT"
            exit_reason = f"Bid {bid:.4f} が損切り目安 {stop:.4f} 以下です"
        elif target is not None and bid >= target:
            state = "TAKE_PROFIT"
            exit_reason = f"Bid {bid:.4f} が利益確定目安 {target:.4f} 以上です"
        else:
            state = "HOLD"
            if stop is None and target is None:
                exit_reason = (
                    "損切り・利益確定価格が未設定のため、価格到達判定はできません")
            elif stop is None:
                exit_reason = (
                    "損切り価格を確認できません。利益確定価格のみ監視しています")
            elif target is None:
                exit_reason = (
                    "利益確定価格を確認できません。損切り価格のみ監視しています")
    elif position_open:
        state = "DATA_WAIT"
        exit_reason = "新鮮なBidを確認できないため、保有株の退出条件を断定しません"
    elif data_blockers:
        state = "DATA_WAIT"
    elif safety_blockers:
        state = "WAIT"
    else:
        state, memory, transition = _entry_state(
            score, required_passed, signal_time, session_key, previous_memory,
            cfg, process_bar=True)

    if position_open or data_blockers or safety_blockers:
        transition = {
            "confirmation": {
                "phase": "holding" if position_open else "blocked",
                "streak": int(memory.get("pending_count", 0)),
                "required": int(cfg["confirmation_bars"]),
                "remaining": max(
                    int(cfg["confirmation_bars"] - memory.get("pending_count", 0)), 0),
            },
            "alert": {"emit": False, "kind": None,
                      "cooldown_remaining": int(memory.get("cooldown_remaining", 0))},
            "previous_state": memory.get("stable_state"),
        }
        previous_state = str(memory.get("stable_state") or "")
        if position_open and state in {"RISK_EXIT", "TAKE_PROFIT"} and previous_state != state:
            transition["alert"] = {
                "emit": True, "kind": state,
                "cooldown_remaining": int(memory.get("cooldown_remaining", 0)),
            }
        if signal_time and signal_time != memory.get("last_processed_bar_time"):
            memory["last_processed_bar_time"] = signal_time
        if not position_open and signal_time and signal_time != _mapping(previous_memory).get(
                "last_processed_bar_time"):
            memory.update({
                "latched_state": None, "pending_state": None,
                "pending_count": 0, "release_count": 0,
            })
        memory["stable_state"] = state

    reason = exit_reason
    if reason is None and state == "DATA_WAIT" and data_blockers:
        reason = data_blockers[0]["reason_ja"]
    if reason is None and state == "WAIT" and safety_blockers:
        reason = safety_blockers[0]["reason_ja"]
    label, description, tone = _display(state, reason)
    holding_execution_ready = not holding_blockers
    actionable = state == "BUY_READY" or (
        state in {"RISK_EXIT", "TAKE_PROFIT"}
        and holding_execution_ready
        and quote.get("fresh_bid") is True
    )
    quality_blockers = holding_blockers if position_open else data_blockers
    technical_available = not quality_blockers
    quality_status = "GOOD" if technical_available else "INVALID"

    prices = {
        "last_closed": _number(features.get("last_close"), positive=True),
        "bid": bid, "ask": ask, "spread_pct": spread,
        "vwap": vwap, "atr14": atr, "vwap_chase_limit": chase_limit,
        "maximum_buy_price": None, "holding_entry": holding_entry,
        "stop": stop, "target": target,
        "current_rr": None, "minimum_rr": None,
    }
    components = {
        "technical": {
            "score": score, "raw_state": raw_state,
            "required_passed": required_passed, "checks": checks,
        },
        "execution": {
            "ready": (holding_execution_ready if position_open else
                      not data_blockers and not safety_blockers),
            "blocking_gate_keys": [item["key"] for item in (
                holding_blockers if position_open else
                [*data_blockers, *safety_blockers])],
        },
        "holding": {
            "is_open": position_open, "fresh_bid": bool(quote["fresh_bid"]),
            "stop": stop, "target": target, "peak_bid": memory.get("peak_bid"),
        },
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "symbol": str(symbol or "").strip().upper(),
        "state": state,
        "status": state,
        "position_mode": "holding" if position_open else "entry",
        "actionable": actionable,
        "label_ja": label,
        "reason_ja": description,
        "signal_bar_time": signal_time,
        "score": score,
        "thresholds": {
            "setup_score": cfg["setup_score"],
            "entry_score": cfg["entry_score"],
            "release_score": cfg["release_score"],
            "confirmation_bars": cfg["confirmation_bars"],
        },
        "checks": checks,
        "gates": gates,
        "features": features,
        "prices": prices,
        "data_quality": {
            "status": quality_status,
            "issues": [item["reason_ja"] for item in quality_blockers],
            "bars": bar_meta,
            "benchmark": benchmark_meta,
            "quote": quote,
        },
        "confirmation": transition["confirmation"],
        "alert": transition["alert"],
        "action": {
            "code": state, "label_ja": label,
            "description_ja": description, "actionable": actionable,
            "tone": tone,
        },
        "components": components,
        "memory": memory,
        "read_only": True,
        "places_orders": False,
        "uses_network": False,
        "uses_daily_data": False,
        "uses_purchase_plan": False,
        "uses_moomoo_history_quota": False,
        "disclaimer": (
            "確定済みデータから売買タイミングを整理する参考情報です。"
            "注文は実行せず、約定価格や損失額を保証しません。"
        ),
    }
    return result


__all__ = [
    "DEFAULT_CONFIG", "SCHEMA_VERSION", "STATE_DISPLAY", "TRADING_SESSIONS",
    "calculate_intraday_features", "completed_intraday_bars",
    "evaluate_realtime_signal", "realtime_config",
]
