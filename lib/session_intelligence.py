"""米国株の取引セッションと次回寄付き方向を診断する純粋関数群。

このモジュールは、呼び出し側から渡された時刻・価格・特徴量だけを扱う。
ネットワーク、moomoo OpenD、履歴K線、口座、注文APIには一切アクセスしない。

寄付き方向の数値は学習済み・校正済み確率ではない。入力を有限範囲へ正規化し、
対象セッション別の固定重みで足し合わせた透明なヒューリスティックである。
特徴量不足や取引可否不明時は確率を返さず ``unknown`` とする。
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from math import exp, log, tanh
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


MARKET_TZ = ZoneInfo("America/New_York")
SESSION_NAMES = ("premarket", "regular", "afterhours", "overnight")

PREMARKET_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)
AFTERHOURS_CLOSE = time(20, 0)
EARLY_AFTERHOURS_CLOSE = time(17, 0)
OVERNIGHT_OPEN = time(20, 0)
OVERNIGHT_CLOSE = time(4, 0)


MARKET_STATE_TO_SESSION = {
    "PRE_MARKET_BEGIN": "premarket",
    "PRE_MARKET": "premarket",
    "PRE_MARKET_END": "premarket",
    "MORNING": "regular",
    "AFTERNOON": "regular",
    "AFTER_HOURS_BEGIN": "afterhours",
    "AFTER_HOURS": "afterhours",
    "OVERNIGHT": "overnight",
    "NIGHT_OPEN": "overnight",
    "NIGHT_OPENING": "overnight",
}

CLOSED_MARKET_STATES = {
    "CLOSED", "REST", "AFTER_HOURS_END", "NIGHT_END", "HOLIDAY",
}


FEATURE_ALIASES = {
    "gap_change_pct": "gap_pct",
    "futures_change_pct": "futures_pct",
    "spy_change_pct": "spy_pct",
    "qqq_change_pct": "qqq_pct",
    "stock_momentum_pct": "momentum_pct",
    "volume_ratio": "relative_volume",
    "relative_volume_ratio": "relative_volume",
    "event_direction": "event_score",
}

FEATURE_LABELS = {
    "gap_pct": "直近セッションの対終値ギャップ",
    "futures_pct": "関連先物",
    "spy_pct": "SPY",
    "qqq_pct": "QQQ",
    "momentum_pct": "銘柄モメンタム",
    "relative_volume": "相対出来高",
    "event_score": "イベント方向",
    "event_risk": "イベント不確実性",
}

FEATURE_SCALES = {
    "gap_pct": 1.5,
    "futures_pct": 1.0,
    "spy_pct": 0.8,
    "qqq_pct": 0.8,
    "momentum_pct": 2.0,
}

# 値はlogitへの最大寄与。統計学習した係数ではなく、透明性を優先した固定値。
TARGET_WEIGHTS = {
    "regular": {
        "gap_pct": 0.85, "futures_pct": 0.65, "spy_pct": 0.35,
        "qqq_pct": 0.35, "momentum_pct": 0.30,
        "relative_volume": 0.20, "event_score": 0.55,
    },
    "premarket": {
        "gap_pct": 0.50, "futures_pct": 0.70, "spy_pct": 0.35,
        "qqq_pct": 0.35, "momentum_pct": 0.25,
        "relative_volume": 0.15, "event_score": 0.40,
    },
    "afterhours": {
        "gap_pct": 0.35, "futures_pct": 0.25, "spy_pct": 0.35,
        "qqq_pct": 0.35, "momentum_pct": 0.55,
        "relative_volume": 0.25, "event_score": 0.80,
    },
    "overnight": {
        "gap_pct": 0.30, "futures_pct": 0.55, "spy_pct": 0.25,
        "qqq_pct": 0.25, "momentum_pct": 0.35,
        "relative_volume": 0.15, "event_score": 0.55,
    },
}

FEATURE_MAX_AGES = {
    "gap_pct": pd.Timedelta(hours=2),
    "futures_pct": pd.Timedelta(hours=1),
    "spy_pct": pd.Timedelta(hours=1),
    "qqq_pct": pd.Timedelta(hours=1),
    "momentum_pct": pd.Timedelta(days=7),
    "relative_volume": pd.Timedelta(hours=2),
    "event_score": pd.Timedelta(days=7),
    "event_risk": pd.Timedelta(days=7),
}


def _normalise_market_state(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.rsplit(".", 1)[-1] if "." in text else text


def _as_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) and number not in (float("inf"), float("-inf")) else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quality_label(score: float, insufficient: bool = False) -> str:
    if insufficient:
        return "insufficient"
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _to_market_time(value: object | None) -> tuple[pd.Timestamp, list[str]]:
    warnings: list[str] = []
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("nowは有効な日時である必要があります")
    if stamp.tzinfo is None:
        # テストやUIからのローカル壁時計を扱える一方、暗黙解釈は品質情報へ残す。
        warnings.append("timezoneなしの時刻をAmerica/New_Yorkとして解釈しました")
        stamp = stamp.tz_localize(MARKET_TZ)
    else:
        stamp = stamp.tz_convert(MARKET_TZ)
    return stamp, warnings


def _to_utc(value: object, *, naive_market_time: bool = True) -> tuple[pd.Timestamp | None, str | None]:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None, "timestampを解釈できません"
    if pd.isna(stamp):
        return None, "timestampが欠損しています"
    warning = None
    if stamp.tzinfo is None:
        zone = MARKET_TZ if naive_market_time else timezone.utc
        stamp = stamp.tz_localize(zone)
        warning = "timezoneなしのtimestampを米国東部時間として解釈しました"
    return stamp.tz_convert("UTC"), warning


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (occurrence - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    current = next_month - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _easter_sunday(year: int) -> date:
    """Gregorian calendarの復活祭日（Meeus/Jones/Butcher法）。"""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed_fixed(day: date, *, saturday_previous: bool = True) -> date:
    if day.weekday() == 5 and saturday_previous:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def nyse_holidays(year: int) -> frozenset[date]:
    """NYSEの定例終日休場日を返す（臨時休場はextra_holidaysで補う）。"""
    new_year = date(year, 1, 1)
    holidays = {
        # 元日が土曜の場合、NYSEは通常その前日を振替休場にしない。
        _observed_fixed(new_year, saturday_previous=False),
        _nth_weekday(year, 1, 0, 3),   # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),   # Washington's Birthday
        _easter_sunday(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),     # Memorial Day
        _observed_fixed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed_fixed(date(year, 12, 25)),
    }
    if year >= 2022:
        holidays.add(_observed_fixed(date(year, 6, 19)))
    return frozenset(holidays)


def _holiday_set(extra_holidays: Iterable[object]) -> set[date]:
    out: set[date] = set()
    for value in extra_holidays or ():
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            continue
        if not pd.isna(stamp):
            out.add(stamp.date())
    return out


def is_nyse_trading_day(day: date, extra_holidays: Iterable[object] = ()) -> bool:
    """週末・NYSE定例休場・呼び出し側指定の臨時休場を除外する。"""
    return (day.weekday() < 5 and day not in nyse_holidays(day.year)
            and day not in _holiday_set(extra_holidays))


def regular_close_time(day: date, extra_holidays: Iterable[object] = ()) -> time | None:
    """通常は16:00、代表的な短縮取引日は13:00。休場日はNone。"""
    if not is_nyse_trading_day(day, extra_holidays):
        return None
    thanksgiving = _nth_weekday(day.year, 11, 3, 4)
    day_after_thanksgiving = thanksgiving + timedelta(days=1)
    july_third = date(day.year, 7, 3)
    christmas_eve = date(day.year, 12, 24)
    early = (
        day == day_after_thanksgiving
        or (day == july_third and day.weekday() in (0, 1, 2, 3))
        or (day == christmas_eve and day.weekday() in (0, 1, 2, 3))
    )
    return EARLY_CLOSE if early else REGULAR_CLOSE


def afterhours_close_time(day: date, extra_holidays: Iterable[object] = ()) -> time | None:
    """通常は20:00、13:00短縮取引日は17:00。休場日はNone。

    NYSE系会場の公表時間に合わせる。個別銘柄・ブローカーの取引可否は、この
    カレンダー値だけで断定せず、利用可能なら ``market_state`` を優先する。
    """
    regular_close = regular_close_time(day, extra_holidays)
    if regular_close is None:
        return None
    return EARLY_AFTERHOURS_CLOSE if regular_close == EARLY_CLOSE else AFTERHOURS_CLOSE


def _calendar_session(now_et: pd.Timestamp, extra_holidays: Iterable[object]) -> tuple[str, bool]:
    day = now_et.date()
    wall = now_et.time().replace(tzinfo=None)
    trading_day = is_nyse_trading_day(day, extra_holidays)
    close = regular_close_time(day, extra_holidays)
    after_close = afterhours_close_time(day, extra_holidays)

    # 00:00-04:00は、その暦日が取引日なら前夜から続くovernight枠。
    if wall < OVERNIGHT_CLOSE and trading_day:
        return "overnight", trading_day
    if trading_day and PREMARKET_OPEN <= wall < REGULAR_OPEN:
        return "premarket", trading_day
    if trading_day and close is not None and REGULAR_OPEN <= wall < close:
        return "regular", trading_day
    if trading_day and close is not None and after_close is not None and close <= wall < after_close:
        return "afterhours", trading_day

    # 20:00以降は、直後の暦日が取引日である場合だけ翌取引日のovernight枠。
    tomorrow = day + timedelta(days=1)
    if wall >= OVERNIGHT_OPEN and is_nyse_trading_day(tomorrow, extra_holidays):
        return "overnight", trading_day
    return "closed", trading_day


def detect_current_session(
    now: object | None = None,
    market_state: object = None,
    overnight_eligible: bool | None = None,
    extra_holidays: Iterable[object] = (),
) -> dict:
    """現在の米国株セッションを、市場状態優先・NY時間補完で判定する。

    ``overnight_eligible=None`` のままovernight時間帯に入った場合、銘柄ごとの
    24時間取引可否を推測せず ``session='unknown'`` とする。返却するカレンダーは
    定例休場を内蔵するが、臨時休場は ``extra_holidays`` で明示する。
    """
    now_et, warnings = _to_market_time(now)
    state = _normalise_market_state(market_state)
    calendar_session, trading_day = _calendar_session(now_et, extra_holidays)
    mapped = MARKET_STATE_TO_SESSION.get(state)
    source = "calendar"
    tradable: bool | None

    if mapped:
        session = mapped
        tradable = True
        source = "market_state"
        if mapped != calendar_session:
            warnings.append(
                f"市場状態({mapped})と時刻帯({calendar_session})が一致しません。市場状態を優先しました")
    elif state in CLOSED_MARKET_STATES:
        session = "closed"
        tradable = False
        source = "market_state"
    else:
        session = calendar_session
        tradable = session != "closed"
        if state:
            warnings.append(f"未対応の市場状態 {state} のため時刻帯で補完しました")

    if session == "overnight" and source != "market_state":
        if overnight_eligible is False:
            session, tradable = "closed", False
            warnings.append("この銘柄はovernight取引対象外として指定されています")
        elif overnight_eligible is None:
            session, tradable = "unknown", None
            warnings.append("銘柄のovernight取引可否を確認できません")

    score = 1.0 if source == "market_state" else 0.75
    if state and not mapped and state not in CLOSED_MARKET_STATES:
        score -= 0.10
    if any("timezoneなし" in item for item in warnings):
        score -= 0.10
    if tradable is None:
        score = min(score, 0.35)
    score = _clamp(score, 0.0, 1.0)

    close = regular_close_time(now_et.date(), extra_holidays)
    after_close = afterhours_close_time(now_et.date(), extra_holidays)
    if session == "unknown":
        reason = "現在はovernight時間帯ですが、銘柄の24時間取引可否が不明です"
    elif session == "closed" and calendar_session == "overnight" and overnight_eligible is False:
        reason = "overnight時間帯ですが、この銘柄は対象外です"
    elif session == "closed":
        reason = "現在は定例セッション外または休場日です"
    else:
        reason = f"現在は{session}セッションです"

    return {
        "session": session,
        "calendar_session": calendar_session,
        "tradable": tradable,
        "as_of": now_et,
        "timezone": "America/New_York",
        "market_state": state or None,
        "source": source,
        "overnight_eligible": overnight_eligible,
        "is_trading_day": trading_day,
        "regular_close": close,
        "afterhours_close": after_close,
        "calendar_scope": "NYSE定例休場・代表的短縮日（臨時休場は呼び出し側指定）",
        "data_quality": {"score": score, "label": _quality_label(score)},
        "warnings": warnings,
        "reason": reason,
        "read_only": True,
    }


def _local_timestamp(day: date, wall: time) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, wall), tz=MARKET_TZ)


def next_session_open(
    target_session: str,
    now: object | None = None,
    overnight_eligible: bool | None = None,
    extra_holidays: Iterable[object] = (),
) -> dict:
    """指定セッションの「現在より後に来る」次回開始時刻を返す。"""
    target = str(target_session or "").strip().lower()
    if target not in SESSION_NAMES:
        raise ValueError(f"target_sessionは{SESSION_NAMES}から選んでください")
    now_et, warnings = _to_market_time(now)

    scheduled: pd.Timestamp | None = None
    trading_date: date | None = None
    for offset in range(0, 370):
        candidate_day = now_et.date() + timedelta(days=offset)
        if not is_nyse_trading_day(candidate_day, extra_holidays):
            continue
        if target == "premarket":
            candidate = _local_timestamp(candidate_day, PREMARKET_OPEN)
        elif target == "regular":
            candidate = _local_timestamp(candidate_day, REGULAR_OPEN)
        elif target == "afterhours":
            candidate = _local_timestamp(
                candidate_day, regular_close_time(candidate_day, extra_holidays) or REGULAR_CLOSE)
        else:
            candidate = _local_timestamp(candidate_day - timedelta(days=1), OVERNIGHT_OPEN)
        if candidate > now_et:
            scheduled, trading_date = candidate, candidate_day
            break

    if scheduled is None:
        status, available = "unknown", None
        warnings.append("370日以内に次回セッションを特定できませんでした")
    elif target == "overnight" and overnight_eligible is False:
        status, available = "unavailable", False
        warnings.append("この銘柄はovernight取引対象外として指定されています")
    elif target == "overnight" and overnight_eligible is None:
        status, available = "unknown", None
        warnings.append("予定時刻は算出しましたが、銘柄のovernight取引可否が不明です")
    else:
        status, available = "scheduled", True

    score = 0.75
    if any("timezoneなし" in item for item in warnings):
        score -= 0.10
    if available is None:
        score = min(score, 0.40)
    return {
        "target_session": target,
        "open_time": scheduled,
        "trading_date": trading_date,
        "available": available,
        "status": status,
        "timezone": "America/New_York",
        "data_quality": {"score": score, "label": _quality_label(score)},
        "warnings": warnings,
        "calendar_scope": "NYSE定例休場・代表的短縮日（臨時休場は呼び出し側指定）",
        "read_only": True,
    }


def _session_observation(value: object) -> dict:
    if isinstance(value, Mapping):
        price = next((_as_float(value.get(key)) for key in
                      ("price", "last", "current", "close")
                      if _as_float(value.get(key)) is not None), None)
        return {
            "price": price,
            "open": _as_float(value.get("open")),
            "reference_price": _as_float(
                value.get("reference_price", value.get("prior_price"))),
            "volume": _as_float(value.get("volume")),
            "timestamp": value.get("timestamp", value.get("as_of")),
            "source": value.get("source"),
            "quality": _as_float(value.get("quality")),
        }
    return {"price": _as_float(value), "open": None, "reference_price": None,
            "volume": None, "timestamp": None, "source": None, "quality": None}


def _price_change(current: float | None, reference: float | None) -> tuple[float | None, float | None]:
    if current is None or reference is None or reference <= 0:
        return None, None
    return current - reference, (current / reference - 1) * 100


def compute_session_changes(
    session_prices: Mapping[str, object] | None,
    previous_close: object = None,
) -> dict:
    """立会・プレ・アフター・overnightの価格変化を同じ基準で整形する。

    各セッション値は数値、または ``price/open/reference_price/volume/timestamp/
    source/quality`` を持つdictで渡せる。欠損価格を他セッションから推定しない。
    """
    supplied = session_prices or {}
    observations = {
        name: _session_observation(supplied.get(name)) for name in SESSION_NAMES
    }
    previous = _as_float(previous_close)
    if previous is not None and previous <= 0:
        previous = None

    prior_candidates = {
        "afterhours": ("regular",),
        "overnight": ("afterhours", "regular"),
        "premarket": ("overnight", "afterhours", "regular"),
        "regular": ("premarket", "overnight"),
    }
    sessions: dict[str, dict] = {}
    quality_scores = []
    for name in SESSION_NAMES:
        obs = observations[name]
        price = obs["price"]
        if price is None or price <= 0:
            sessions[name] = {
                "session": name, "status": "missing", "price": None,
                "previous_close": previous,
                "change_vs_previous_close": None,
                "change_vs_previous_close_pct": None,
                "session_open": obs["open"], "change_vs_open": None,
                "change_vs_open_pct": None, "prior_session": None,
                "prior_price": None, "change_vs_prior_session": None,
                "change_vs_prior_session_pct": None, "volume": obs["volume"],
                "timestamp": obs["timestamp"], "source": obs["source"],
                "data_quality": {"score": 0.0, "label": "insufficient"},
            }
            continue

        prior_price = obs["reference_price"]
        prior_session = "explicit" if prior_price is not None else None
        if prior_price is None:
            for candidate in prior_candidates[name]:
                candidate_price = observations[candidate]["price"]
                if candidate_price is not None and candidate_price > 0:
                    prior_price, prior_session = candidate_price, candidate
                    break
        if prior_price is None:
            prior_price, prior_session = previous, "previous_close" if previous is not None else None

        prev_change, prev_pct = _price_change(price, previous)
        open_change, open_pct = _price_change(price, obs["open"])
        prior_change, prior_pct = _price_change(price, prior_price)
        quality = 0.45
        quality += 0.20 if previous is not None else 0.0
        quality += 0.10 if obs["timestamp"] is not None else 0.0
        quality += 0.10 if obs["source"] else 0.0
        quality += 0.10 if obs["open"] is not None or prior_price is not None else 0.0
        if obs["quality"] is not None:
            quality *= _clamp(obs["quality"], 0.0, 1.0)
        quality = _clamp(quality, 0.0, 1.0)
        quality_scores.append(quality)
        sessions[name] = {
            "session": name, "status": "ok", "price": price,
            "previous_close": previous,
            "change_vs_previous_close": prev_change,
            "change_vs_previous_close_pct": prev_pct,
            "session_open": obs["open"], "change_vs_open": open_change,
            "change_vs_open_pct": open_pct, "prior_session": prior_session,
            "prior_price": prior_price, "change_vs_prior_session": prior_change,
            "change_vs_prior_session_pct": prior_pct, "volume": obs["volume"],
            "timestamp": obs["timestamp"], "source": obs["source"],
            "data_quality": {"score": quality, "label": _quality_label(quality)},
        }

    available = tuple(name for name in SESSION_NAMES if sessions[name]["status"] == "ok")
    aggregate = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return {
        "previous_close": previous,
        "sessions": sessions,
        "available": available,
        "missing": tuple(name for name in SESSION_NAMES if name not in available),
        "data_quality": {
            "score": aggregate,
            "label": _quality_label(aggregate, insufficient=not available),
        },
        "read_only": True,
    }


def _canonical_features(features: Mapping[str, object] | None) -> dict[str, object]:
    source = dict(features or {})
    out: dict[str, object] = {}
    for key, value in source.items():
        canonical = FEATURE_ALIASES.get(str(key), str(key))
        if canonical not in out or canonical == key:
            out[canonical] = value
    return out


def _feature_observation(key: str, value: object, now_utc: pd.Timestamp) -> dict:
    if isinstance(value, Mapping):
        raw = value.get("value")
        timestamp = value.get("timestamp", value.get("as_of"))
        source = value.get("source")
        explicit_quality = _as_float(value.get("quality"))
    else:
        raw, timestamp, source, explicit_quality = value, None, None, None

    if key == "event_risk":
        if isinstance(raw, str):
            numeric = 1.0 if raw.strip().lower() in {"true", "yes", "1", "high"} else 0.0
        else:
            numeric = 1.0 if bool(raw) else 0.0
    else:
        numeric = _as_float(raw)
    if numeric is None:
        return {"value": None, "quality": 0.0, "status": "missing",
                "timestamp": None, "source": source, "warnings": []}

    quality = 0.70 if timestamp is None else 1.0
    warnings: list[str] = []
    parsed_time = None
    if timestamp is not None:
        parsed_time, warning = _to_utc(timestamp)
        if warning:
            warnings.append(warning)
            quality *= 0.85
        if parsed_time is None:
            quality *= 0.50
        else:
            age = now_utc - parsed_time
            if age < -pd.Timedelta(minutes=5):
                return {
                    "value": numeric,
                    "quality": 0.0,
                    "status": "invalid_time",
                    "timestamp": parsed_time,
                    "source": source,
                    "warnings": [*warnings, "timestampが現在より未来のため利用しません"],
                }
            else:
                age = max(age, pd.Timedelta(0))
                max_age = FEATURE_MAX_AGES[key]
                if age > max_age * 3:
                    return {"value": numeric, "quality": 0.0, "status": "stale",
                            "timestamp": parsed_time, "source": source,
                            "warnings": [*warnings, f"{age}経過し鮮度上限を超えています"]}
                if age > max_age:
                    quality *= max(0.20, 1 - float(age / (max_age * 3)))
                    warnings.append(f"鮮度目安({max_age})を超えています")
    if explicit_quality is not None:
        quality *= _clamp(explicit_quality, 0.0, 1.0)
    return {
        "value": numeric,
        "quality": _clamp(quality, 0.0, 1.0),
        "status": "ok",
        "timestamp": parsed_time,
        "source": source,
        "warnings": warnings,
    }


def _standardize_feature(key: str, value: float, anchor: float | None = None) -> tuple[float, str]:
    if key in FEATURE_SCALES:
        return tanh(value / FEATURE_SCALES[key]), f"tanh({value:.4g}/{FEATURE_SCALES[key]:g})"
    if key == "event_score":
        return _clamp(value, -1.0, 1.0), "[-1,+1]へ制限"
    if key == "relative_volume":
        if value <= 0:
            return 0.0, "相対出来高が0以下のため寄与なし"
        if anchor is None or abs(anchor) < 1e-9:
            return 0.0, "方向を持つgap/momentumがないため信頼度にだけ使用"
        amplification = _clamp(log(max(value, 1.0)) / log(4.0), 0.0, 1.0)
        signed = amplification if anchor > 0 else -amplification
        return signed, "出来高増をgap/momentum方向の確認材料として使用"
    return 0.0, "方向寄与なし"


def diagnose_next_open(
    features: Mapping[str, object] | None,
    target_session: str = "regular",
    now: object | None = None,
    current_session: str | Mapping[str, object] | None = None,
    *,
    overnight_eligible: bool | None = None,
    extra_holidays: Iterable[object] = (),
) -> dict:
    """選択した次回セッション開始方向を透明な固定重みで診断する。

    返す ``probability_up/down`` は校正済み予測ではない。最低2つの方向特徴がなく、
    またはovernight取引可否が不明な場合は ``None`` として推測を拒否する。
    """
    target = str(target_session or "").strip().lower()
    if target not in SESSION_NAMES:
        raise ValueError(f"target_sessionは{SESSION_NAMES}から選んでください")
    now_et, time_warnings = _to_market_time(now)
    now_utc = now_et.tz_convert("UTC")
    target_open = next_session_open(
        target, now_et, overnight_eligible=overnight_eligible,
        extra_holidays=extra_holidays)
    canonical = _canonical_features(features)
    weights = TARGET_WEIGHTS[target]

    observed = {
        key: _feature_observation(key, canonical.get(key), now_utc)
        for key in (*weights, "event_risk")
    }
    anchor_values = []
    for key in ("gap_pct", "momentum_pct"):
        item = observed[key]
        if item["status"] == "ok":
            standard, _ = _standardize_feature(key, item["value"])
            anchor_values.append(standard)
    anchor = sum(anchor_values) / len(anchor_values) if anchor_values else None

    feature_rows = []
    directional_count = 0
    logit = 0.0
    quality_numerator = 0.0
    total_weight = sum(abs(weight) for weight in weights.values())
    for key, weight in weights.items():
        item = observed[key]
        if item["status"] == "ok":
            standardized, transform = _standardize_feature(
                key, item["value"], anchor=anchor)
            contribution = standardized * weight * item["quality"]
            if key != "relative_volume" and abs(standardized) > 1e-12:
                directional_count += 1
            logit += contribution
            quality_numerator += abs(weight) * item["quality"]
        else:
            standardized, contribution, transform = None, 0.0, "利用しません"
        feature_rows.append({
            "key": key,
            "label": FEATURE_LABELS[key],
            "value": item["value"],
            "status": item["status"],
            "standardized": standardized,
            "weight": weight,
            "quality": item["quality"],
            "contribution": contribution,
            "transform": transform,
            "timestamp": item["timestamp"],
            "source": item["source"],
            "warnings": item["warnings"],
        })

    event_risk = observed["event_risk"]
    event_risk_active = event_risk["status"] == "ok" and bool(event_risk["value"])
    quality_score = quality_numerator / total_weight if total_weight else 0.0
    if event_risk_active and observed["event_score"]["status"] != "ok":
        quality_score *= 0.80
    quality_score = _clamp(quality_score, 0.0, 1.0)

    unavailable_target = target_open["available"] is not True
    insufficient = directional_count < 2 or quality_score < 0.25 or unavailable_target
    if insufficient:
        probability_up = probability_down = None
        direction = "unknown"
        effective_logit = None
    else:
        # 校正済みに見せないため、入力充足度で0方向へ縮小し80%を上限とする。
        effective_logit = _clamp(logit * (0.35 + 0.65 * quality_score), -1.386294, 1.386294)
        probability_up = 1.0 / (1.0 + exp(-effective_logit))
        probability_down = 1.0 - probability_up
        direction = ("up" if probability_up >= 0.55 else
                     "down" if probability_up <= 0.45 else "neutral")

    strength = 0.0 if probability_up is None else abs(probability_up - 0.5) * 2
    confidence_score = strength * quality_score * min(1.0, directional_count / 4)
    confidence = "medium" if (
        not insufficient and confidence_score >= 0.20
        and quality_score >= 0.55 and directional_count >= 3) else "low"

    missing = tuple(row["key"] for row in feature_rows if row["status"] != "ok")
    positive = sorted((row for row in feature_rows if row["contribution"] > 0),
                      key=lambda row: row["contribution"], reverse=True)
    negative = sorted((row for row in feature_rows if row["contribution"] < 0),
                      key=lambda row: row["contribution"])
    if unavailable_target:
        reason = ("対象セッションの取引可否を確認できないため方向を診断しません"
                  if target_open["available"] is None else
                  "対象銘柄は指定セッションで取引できないため方向を診断しません")
    elif directional_count < 2:
        reason = "方向を持つ独立特徴が2つ未満のため推測しません"
    elif quality_score < 0.25:
        reason = "データ品質が不足しているため推測しません"
    elif direction == "neutral":
        reason = "上向き・下向きの材料が拮抗しています"
    else:
        reason = f"固定重みの合成は{direction}寄りですが、校正済み予測ではありません"

    if isinstance(current_session, Mapping):
        current_name = current_session.get("session")
    else:
        current_name = current_session
    if current_name is None:
        current_name = detect_current_session(
            now_et, overnight_eligible=overnight_eligible,
            extra_holidays=extra_holidays)["session"]

    return {
        "target_session": target,
        "target_open": target_open,
        "current_session": current_name,
        "direction": direction,
        "probability_up": probability_up,
        "probability_down": probability_down,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "data_quality": {
            "score": quality_score,
            "label": _quality_label(quality_score, insufficient=insufficient),
            "directional_feature_count": directional_count,
            "missing_features": missing,
        },
        "raw_logit": logit,
        "effective_logit": effective_logit,
        "features": feature_rows,
        "top_positive": [row["key"] for row in positive[:3]],
        "top_negative": [row["key"] for row in negative[:3]],
        "event_risk": event_risk_active,
        "reason": reason,
        "warnings": [*time_warnings, *target_open["warnings"]],
        "method": "transparent_heuristic_logistic_v1",
        "calibrated": False,
        "read_only": True,
        "uses_opend_history_quota": False,
        "disclaimer": (
            "統計的に校正された勝率ではなく、入力特徴量を固定重みで合成した参考診断です。"
            "売買判断や注文には直接使用しないでください。"),
    }


def analyze_session_intelligence(
    *,
    now: object | None = None,
    market_state: object = None,
    overnight_eligible: bool | None = None,
    extra_holidays: Iterable[object] = (),
    session_prices: Mapping[str, object] | None = None,
    previous_close: object = None,
    target_session: str = "regular",
    features: Mapping[str, object] | None = None,
) -> dict:
    """現在セッション・セッション別変化・次回寄付き診断をまとめて返す。"""
    current = detect_current_session(
        now, market_state=market_state, overnight_eligible=overnight_eligible,
        extra_holidays=extra_holidays)
    changes = compute_session_changes(session_prices, previous_close)
    merged_features = dict(features or {})

    # gapが明示されていない場合だけ、現在セッションの実測対終値変化から補う。
    has_gap = "gap_pct" in merged_features or "gap_change_pct" in merged_features
    current_change = changes["sessions"].get(current["session"], {})
    derived_gap = current_change.get("change_vs_previous_close_pct")
    if not has_gap and derived_gap is not None:
        merged_features["gap_pct"] = {
            "value": derived_gap,
            "timestamp": current_change.get("timestamp"),
            "source": current_change.get("source") or "session_prices",
            "quality": current_change.get("data_quality", {}).get("score"),
        }

    diagnosis = diagnose_next_open(
        merged_features, target_session=target_session, now=current["as_of"],
        current_session=current, overnight_eligible=overnight_eligible,
        extra_holidays=extra_holidays)
    return {
        "current_session": current,
        "session_changes": changes,
        "next_open_diagnosis": diagnosis,
        "calibrated": False,
        "read_only": True,
        "uses_opend_history_quota": False,
    }


__all__ = [
    "SESSION_NAMES", "MARKET_STATE_TO_SESSION", "FEATURE_ALIASES",
    "TARGET_WEIGHTS", "nyse_holidays", "is_nyse_trading_day",
    "regular_close_time", "afterhours_close_time", "detect_current_session",
    "next_session_open",
    "compute_session_changes", "diagnose_next_open",
    "analyze_session_intelligence",
]
