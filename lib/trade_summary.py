"""既取得データだけから、銘柄別の売買情報を一画面向けに整理する。

このモジュールは表示用の投影層であり、売買判定を新しく生成しない。とくに、
イベント・セッション・トレンド・支持抵抗を既存ルールのスコアへ加点せず、
注文APIも呼び出さない。入力不足時は値を推測せず ``None`` と理由を返す。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1

VERDICT_CODES = (
    "BUY", "WAIT", "NEUTRAL", "RISK_EXIT", "TAKE_PROFIT", "HOLD",
)

_VERDICT_DISPLAY = {
    "BUY": {
        "label_ja": "買い条件成立",
        "summary_ja": "設定した必須条件と買い判定が成立しています。",
        "tone": "positive",
    },
    "WAIT": {
        "label_ja": "待機",
        "summary_ja": "条件またはデータを確認できるまで待機します。",
        "tone": "warning",
    },
    "NEUTRAL": {
        "label_ja": "中立",
        "summary_ja": "買い条件は成立していません。",
        "tone": "neutral",
    },
    "RISK_EXIT": {
        "label_ja": "リスク退出条件成立",
        "summary_ja": "保有中のリスク退出条件が成立しています。",
        "tone": "danger",
    },
    "TAKE_PROFIT": {
        "label_ja": "利確条件成立",
        "summary_ja": "保有中の利確条件が成立しています。",
        "tone": "info",
    },
    "HOLD": {
        "label_ja": "保有継続",
        "summary_ja": "保有中の退出条件は成立していません。",
        "tone": "neutral",
    },
}

_SESSION_LABELS = {
    "premarket": "プレマーケット",
    "regular": "立会時間",
    "afterhours": "アフターマーケット",
    "overnight": "夜間・24時間帯",
    "closed": "セッション外・休場",
    "unknown": "取引可否不明",
}

_TREND_LABELS = {
    "up": "上昇",
    "down": "下降",
    "sideways": "もみ合い",
    "high_vol": "高ボラティリティ",
    "unknown": "判定不能",
}

_REGIME_TO_TREND = {
    "UPTREND": "up",
    "DOWNTREND": "down",
    "RANGE": "sideways",
    "HIGH_VOL": "high_vol",
}

_EVENT_LEVELS = {
    "HIGH": ("high", "影響大"),
    "MEDIUM": ("medium", "影響中"),
    "LOW": ("low", "影響小"),
    "UNKNOWN": ("unknown", "判定材料不足"),
}


def _number(value: Any, *, positive: bool = False) -> float | None:
    """boolを数値として扱わず、有限なfloatだけを返す。"""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _last_close(history: Any) -> tuple[float | None, Any]:
    """DataFrame風または辞書形式の履歴から、確定終値を安全に取り出す。"""
    if history is None:
        return None, None
    try:
        if hasattr(history, "columns") and "Close" in history.columns and len(history):
            series = history["Close"]
            return _number(series.iloc[-1], positive=True), history.index[-1]
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass

    source = _mapping(history)
    closes = source.get("Close", source.get("close"))
    if isinstance(closes, Sequence) and not isinstance(closes, (str, bytes)) and closes:
        timestamps = source.get("index", source.get("timestamps"))
        stamp = None
        if isinstance(timestamps, Sequence) and not isinstance(timestamps, (str, bytes)):
            stamp = timestamps[-1] if timestamps else None
        return _number(closes[-1], positive=True), stamp
    return _number(closes, positive=True), source.get("as_of")


def _current_price(payload: Mapping[str, Any]) -> dict:
    snapshot = _mapping(payload.get("snapshot"))
    explicit = _number(payload.get("current_price"), positive=True)
    snap_price = _number(snapshot.get("price"), positive=True)
    if explicit is not None:
        same_as_snapshot = (
            snap_price is not None
            and math.isclose(explicit, snap_price, rel_tol=1e-9, abs_tol=1e-9)
        )
        source = payload.get("current_price_source")
        if not source and same_as_snapshot:
            source = snapshot.get("source")
        explicit_quality = payload.get("current_price_quality")
        return {
            "value": explicit,
            "available": True,
            "source": str(source or "呼び出し側の現在値"),
            "as_of": payload.get("current_price_as_of", snapshot.get("update_time")),
            "quality": str(explicit_quality) if explicit_quality else (
                "realtime"
                if same_as_snapshot and snapshot.get("source") == "moomoo OpenAPI"
                else "supplied"
            ),
        }
    if snap_price is not None:
        return {
            "value": snap_price,
            "available": True,
            "source": str(snapshot.get("source") or "snapshot"),
            "as_of": snapshot.get("update_time"),
            "quality": "realtime" if snapshot.get("source") == "moomoo OpenAPI" else "snapshot",
        }
    close, stamp = _last_close(payload.get("history"))
    if close is not None:
        return {
            "value": close,
            "available": True,
            "source": "直近確定日足",
            "as_of": stamp,
            "quality": "close_only",
        }
    return {
        "value": None, "available": False, "source": None,
        "as_of": None, "quality": "unavailable",
    }


def _select_rule_evaluation(payload: Mapping[str, Any]) -> dict:
    direct = payload.get("rule_evaluation")
    if isinstance(direct, Mapping) and "verdict" in direct:
        return deepcopy(dict(direct))

    mode = str(payload.get("position_mode") or "entry").strip().lower()
    if isinstance(direct, Mapping):
        nested = direct.get("holding" if mode == "holding" else "entry")
        if isinstance(nested, Mapping):
            return deepcopy(dict(nested))
    key = "holding_evaluation" if mode == "holding" else "entry_evaluation"
    selected = payload.get(key)
    return deepcopy(dict(selected)) if isinstance(selected, Mapping) else {}


def _verdict(evaluation: Mapping[str, Any], requested_mode: Any) -> dict:
    raw = str(evaluation.get("verdict") or "").strip().upper()
    available = raw in VERDICT_CODES
    code = raw if available else "WAIT"
    display = _VERDICT_DISPLAY[code]
    mode = str(evaluation.get("position_mode") or requested_mode or "entry").lower()
    if mode not in {"entry", "holding"}:
        mode = "entry"
    if available:
        reason = str(evaluation.get("summary") or display["summary_ja"])
    elif raw:
        reason = f"未対応の判定コード（{raw}）のため待機します。"
    else:
        reason = "売買判定データがないため待機します。"
    return {
        "code": code,
        "label_ja": display["label_ja"],
        "summary_ja": display["summary_ja"],
        "reason": reason,
        "tone": display["tone"],
        "position_mode": mode,
        "available": available,
        "source": "rule_evaluation" if available else None,
    }


def _price_item(price: float | None, *, source: Any = None,
                current: float | None = None, reason: str | None = None) -> dict:
    distance = None
    if price is not None and current is not None and current > 0:
        distance = (price / current - 1) * 100
    return {
        "price": price,
        "available": price is not None,
        "source": str(source) if source else None,
        "distance_from_current_pct": distance,
        "reason": reason,
    }


def _risk_plan(payload: Mapping[str, Any], evaluation: Mapping[str, Any],
               current: float | None) -> tuple[dict, list[str]]:
    supplied = payload.get("risk_plan")
    source = _mapping(supplied) if isinstance(supplied, Mapping) else _mapping(
        evaluation.get("risk_plan"))
    warnings: list[str] = []

    entry_value = _number(source.get("entry"), positive=True)
    stop_value = _number(source.get("stop"), positive=True)
    target_value = _number(source.get("target"), positive=True)

    entry_reason = None if entry_value is not None else "エントリー価格を取得できません"
    stop_reason = None if stop_value is not None else "損切り目安を取得できません"
    target_reason = None if target_value is not None else "目標価格を取得できません"

    if entry_value is not None and stop_value is not None and stop_value >= entry_value:
        warnings.append("損切り目安がエントリー価格未満ではないため利用しません")
        stop_value, stop_reason = None, "価格の上下関係が不正です"
    if entry_value is not None and target_value is not None and target_value <= entry_value:
        warnings.append("目標価格がエントリー価格を上回らないため利用しません")
        target_value, target_reason = None, "価格の上下関係が不正です"

    rr_value = None
    rr_source = None
    if None not in (entry_value, stop_value, target_value):
        risk = entry_value - stop_value
        reward = target_value - entry_value
        if risk > 0 and reward > 0:
            rr_value = reward / risk
            rr_source = "エントリー・損切り・目標価格から算出"
    if rr_value is None:
        supplied_rr = _number(source.get("rr"), positive=True)
        if supplied_rr is not None and stop_value is not None and target_value is not None:
            rr_value, rr_source = supplied_rr, "既存リスク計画"

    valid = all(value is not None for value in (
        entry_value, stop_value, target_value, rr_value))
    if source.get("valid") is False:
        valid = False
        warnings.append("既存リスク計画が無効と判定されているため利用しません")
    if source and not valid and not warnings:
        warnings.append("エントリー・損切り・目標・R:Rの一部が不足しています")

    return {
        "entry": _price_item(
            entry_value, source=source.get("entry_source", "既存リスク計画") if source else None,
            current=current, reason=entry_reason),
        "stop": _price_item(
            stop_value, source=source.get("stop_source"), current=current,
            reason=stop_reason),
        "target": _price_item(
            target_value, source=source.get("target_source"), current=current,
            reason=target_reason),
        "rr": {
            "value": rr_value,
            "available": rr_value is not None,
            "label_ja": "—" if rr_value is None else f"1 : {rr_value:.2f}",
            "source": rr_source,
            "reason": None if rr_value is not None else "有効な価格計画が不足しています",
        },
        "valid": valid,
        "warnings": tuple(warnings),
    }, warnings


def _normalise_level(row: Mapping[str, Any], current: float,
                     kind: str) -> dict | None:
    center = _number(row.get("price"), positive=True)
    if center is None:
        return None
    low = _number(row.get("zone_low"), positive=True)
    high = _number(row.get("zone_high"), positive=True)
    if low is None:
        low = center
    if high is None:
        high = center
    if low > high:
        low, high = high, low
    inside = low <= current <= high
    if kind == "support" and not inside and center > current:
        return None
    if kind == "resistance" and not inside and center < current:
        return None
    edge = current if inside else high if kind == "support" else low
    distance = 0.0 if inside else (edge / current - 1) * 100
    strength_number = _number(row.get("strength"))
    strength = None if strength_number is None else max(1, min(5, int(strength_number)))
    return {
        "available": True,
        "type": "サポート" if kind == "support" else "抵抗線",
        "price": center,
        "zone_low": low,
        "zone_high": high,
        "edge_price": edge,
        "distance_pct": distance,
        "strength": strength,
        "stars": None if strength is None else "★" * strength + "☆" * (5 - strength),
        "state": "ゾーン内" if inside else "下側" if kind == "support" else "上側",
        "basis": row.get("basis"),
        "source": "既存の支持抵抗分析",
    }


def _nearest_levels(levels: Any, current: float | None) -> tuple[dict, dict]:
    unavailable = {
        "available": False, "type": None, "price": None,
        "zone_low": None, "zone_high": None, "edge_price": None,
        "distance_pct": None, "strength": None, "stars": None,
        "state": "判定不能", "basis": None,
        "source": None,
    }
    if current is None or not isinstance(levels, Sequence) or isinstance(levels, (str, bytes)):
        return deepcopy(unavailable), deepcopy(unavailable)

    supports, resistances = [], []
    for original in levels:
        if not isinstance(original, Mapping):
            continue
        kind_text = str(original.get("type") or "")
        if "サポート" in kind_text:
            item = _normalise_level(original, current, "support")
            if item:
                supports.append(item)
        elif "抵抗" in kind_text:
            item = _normalise_level(original, current, "resistance")
            if item:
                resistances.append(item)
    support = min(supports, key=lambda row: abs(row["distance_pct"])) if supports else deepcopy(unavailable)
    resistance = (
        min(resistances, key=lambda row: abs(row["distance_pct"]))
        if resistances else deepcopy(unavailable)
    )
    return support, resistance


def _trend(payload: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict:
    supplied = _mapping(payload.get("trend"))
    direction = str(supplied.get("direction") or "").strip().lower()
    if direction in _TREND_LABELS:
        return {
            "code": direction,
            "label_ja": str(supplied.get("label") or _TREND_LABELS[direction]),
            "strength": _number(supplied.get("strength")),
            "data_quality": supplied.get("data_quality", "unknown"),
            "source": supplied.get("source"),
            "basis": "当日方向の既存診断",
            "available": direction != "unknown",
        }

    regime = str(evaluation.get("regime") or "").strip().upper()
    projected = _REGIME_TO_TREND.get(regime)
    if projected:
        return {
            "code": projected,
            "label_ja": _TREND_LABELS[projected],
            "strength": None,
            "data_quality": "rule_regime",
            "source": "rule_evaluation.regime",
            "basis": "日足レジーム（当日の方向ではありません）",
            "available": True,
        }
    return {
        "code": "unknown", "label_ja": _TREND_LABELS["unknown"],
        "strength": None, "data_quality": "unavailable", "source": None,
        "basis": "方向データなし", "available": False,
    }


def _session(payload: Mapping[str, Any]) -> dict:
    report = _mapping(payload.get("session"))
    current = _mapping(report.get("current_session"))
    if not current and "session" in report:
        current = report
    code = str(current.get("session") or "unknown").strip().lower()
    if code not in _SESSION_LABELS:
        code = "unknown"
    changes = _mapping(report.get("session_changes"))
    sessions = _mapping(changes.get("sessions"))
    row = _mapping(sessions.get(code))
    available = bool(current) and code != "unknown"
    return {
        "code": code,
        "label_ja": _SESSION_LABELS[code],
        "tradable": _as_bool(current.get("tradable")),
        "reason": current.get("reason") or (
            "セッション情報を取得できません" if not available else None
        ),
        "as_of": current.get("as_of"),
        "price": _number(row.get("price"), positive=True),
        "change_vs_previous_close_pct": _number(row.get("change_vs_previous_close_pct")),
        "data_quality": deepcopy(current.get("data_quality")) if current else "unavailable",
        "source": current.get("source"),
        "available": available,
    }


def _impact_stars(event: Mapping[str, Any]) -> int:
    explicit = _number(event.get("impact_stars"))
    if explicit is not None:
        return max(0, min(5, int(explicit)))
    score = _number(event.get("impact_score"))
    if score is None or score <= 0:
        return 0
    return min(5, int(min(score, 100.0) // 20) + 1)


def _event_risk(payload: Mapping[str, Any]) -> dict:
    report = _mapping(payload.get("event", payload.get("events")))
    raw_events = report.get("events")
    report_status = str(report.get("status") or "").strip().lower()
    raw_warnings = report.get("warnings") or ()
    if isinstance(raw_warnings, (str, bytes)):
        raw_warnings = (raw_warnings,)
    report_warnings = tuple(str(item) for item in raw_warnings if item)
    has_event_list = (isinstance(raw_events, Sequence)
                      and not isinstance(raw_events, (str, bytes)))
    report_available = has_event_list and report_status != "unavailable"
    as_of = _date(payload.get("as_of", report.get("as_of")))
    candidates = []
    for index, original in enumerate(raw_events or () if report_available else ()):
        if not isinstance(original, Mapping):
            continue
        status = str(original.get("status") or "").strip().upper()
        if status != "UPCOMING":
            continue
        event_day = _date(original.get("event_date"))
        days = _number(original.get("days_until"))
        if days is None and as_of is not None and event_day is not None:
            days = float((event_day - as_of).days)
        if days is not None and days < 0:
            continue
        sort_day = event_day.toordinal() if event_day else date.max.toordinal()
        candidates.append((sort_day, index, dict(original), days))

    if not candidates:
        incomplete = report_status in {"partial", "unavailable"}
        available = report_available and not incomplete
        reason = ("イベント情報の一部または全部を取得できません" if incomplete
                  else "今後のイベントは登録されていません" if report_available
                  else "イベント情報を取得できません")
        return {
            "level": "none" if available else "unknown",
            "label_ja": "直近予定なし" if available else "判定材料不足",
            "available": available,
            "event_name": None, "event_date": None, "session": None,
            "days_until": None, "imminent": False if available else None,
            "impact_stars": 0, "impact_stars_text": "☆☆☆☆☆",
            "directional_bias": None, "reason": reason,
            "source": None, "report_status": report_status or None,
            "warnings": report_warnings,
        }

    _, _, event, days = min(candidates, key=lambda item: (item[0], item[1]))
    level_code = str(event.get("impact_level") or "UNKNOWN").strip().upper()
    level, label = _EVENT_LEVELS.get(level_code, _EVENT_LEVELS["UNKNOWN"])
    stars = _impact_stars(event)
    imminent = None if days is None else days <= 2
    name = event.get("display_name_ja", event.get("name"))
    return {
        "level": level,
        "label_ja": label,
        "available": True,
        "event_name": str(name) if name else "名称未取得のイベント",
        "event_date": event.get("event_date"),
        "session": event.get("session_label_ja", event.get("session_label", event.get("session"))),
        "days_until": None if days is None else int(days),
        "imminent": imminent,
        "impact_stars": stars,
        "impact_stars_text": "★" * stars + "☆" * (5 - stars),
        "directional_bias": event.get(
            "directional_bias_label_ja", event.get("directional_bias")),
        "reason": (
            "2日以内の予定イベントです" if imminent is True
            else "今後の予定イベントです" if imminent is False
            else "日付差を確認できない予定イベントです"
        ),
        "source": event.get("source"),
        "report_status": report_status or None,
        "warnings": report_warnings,
    }


def _data_quality(*, price: Mapping[str, Any], verdict: Mapping[str, Any],
                  plan: Mapping[str, Any], support: Mapping[str, Any],
                  resistance: Mapping[str, Any], trend: Mapping[str, Any],
                  session: Mapping[str, Any], event_risk: Mapping[str, Any],
                  warnings: Sequence[str]) -> dict:
    components = {
        "現在値": bool(price.get("available")),
        "売買判定": bool(verdict.get("available")),
        "価格計画": bool(plan.get("valid")),
        "支持抵抗": bool(support.get("available") or resistance.get("available")),
        "トレンド": bool(trend.get("available")),
        "セッション": bool(session.get("available")),
        "イベント": bool(event_risk.get("available")),
    }
    available_count = sum(components.values())
    total = len(components)
    coverage = available_count / total if total else 0.0
    status = "sufficient" if coverage >= 0.85 else "partial" if coverage >= 0.50 else "insufficient"
    label = {"sufficient": "十分", "partial": "一部不足", "insufficient": "不足"}[status]
    quality_warnings = list(dict.fromkeys(str(item) for item in warnings if item))
    if price.get("quality") == "close_only":
        quality_warnings.append("現在値はリアルタイムではなく直近確定日足です")
    elif price.get("quality") == "session_price_time_unverified":
        quality_warnings.append(
            "時間外価格の個別更新時刻を確認できないため参考値として表示しています")
    return {
        "status": status,
        "label_ja": label,
        "coverage": coverage,
        "coverage_pct": round(coverage * 100),
        "available_count": available_count,
        "total_count": total,
        "components": components,
        "missing": tuple(label for label, available in components.items() if not available),
        "warnings": tuple(dict.fromkeys(quality_warnings)),
        "current_price_timestamp_verified": (
            price.get("quality") != "session_price_time_unverified"
        ),
        "score_effect": 0,
    }


def _action_priorities(*, verdict: Mapping[str, Any], plan: Mapping[str, Any],
                       session: Mapping[str, Any], event_risk: Mapping[str, Any],
                       quality: Mapping[str, Any]) -> list[dict]:
    actions: list[dict] = []

    def add(code: str, label: str, detail: str, severity: str,
            *, blocking: bool = False) -> None:
        actions.append({
            "code": code, "label_ja": label, "detail_ja": detail,
            "severity": severity, "blocking": blocking,
        })

    code = verdict["code"]
    if code == "RISK_EXIT":
        add("confirm_risk_exit", "リスク退出条件を最優先で確認",
            "保有中のリスク退出条件と、その成立根拠を確認してください。", "danger")
    elif code == "TAKE_PROFIT":
        add("confirm_take_profit", "利確条件を確認",
            "利確条件の成立根拠と保有状況を確認してください。", "info")
    elif code == "BUY":
        if plan.get("valid"):
            add("review_entry_plan", "買い条件と価格計画を確認",
                "エントリー・損切り・目標・R:Rをまとめて確認してください。", "positive")
        else:
            add("complete_risk_plan", "価格計画が揃うまで新規実行を保留",
                "買い条件は成立していますが、有効な損切り・目標・R:Rが不足しています。",
                "warning", blocking=True)
    elif code == "WAIT":
        add("resolve_wait_reason", "待機理由を確認",
            verdict["reason"], "warning", blocking=True)
    elif code == "NEUTRAL":
        add("wait_for_entry_conditions", "買い条件の成立を待つ",
            "未成立の必須条件とスコア内訳を確認してください。", "neutral")
    else:  # HOLD
        add("monitor_exit_plan", "保有中の退出条件を監視",
            "損切り目安・目標価格と、利確／リスク退出条件を確認してください。", "neutral")

    if event_risk.get("imminent") is True and event_risk.get("level") == "high":
        add("review_event_volatility", "直近イベントの値幅拡大に注意",
            f"{event_risk.get('event_name') or '高影響イベント'}が2日以内に予定されています。"
            "方向を売買スコアへ加えず、値幅リスクだけを確認してください。", "warning")

    if session.get("tradable") is not True:
        detail = ("現在は取引可能時間外です。" if session.get("tradable") is False
                  else "この銘柄の現在セッションでの取引可否を確認できません。")
        add("confirm_session_tradability", "取引可能なセッションを確認",
            detail, "warning", blocking=session.get("tradable") is None)

    if quality.get("status") == "insufficient":
        missing = "、".join(quality.get("missing") or ()) or "必要データ"
        add("refresh_missing_data", "不足データを取得",
            f"不足項目: {missing}。推測で補わず、取得後に再確認してください。", "warning",
            blocking=True)

    for priority, action in enumerate(actions, start=1):
        action["priority"] = priority
    return actions


def build_trade_summary(inputs: Mapping[str, Any] | None = None) -> dict:
    """既存分析結果を日本語UI向けの売買情報サマリーへ投影する純粋関数。

    主な入力キーは ``current_price`` / ``snapshot`` / ``history`` / ``levels`` /
    ``rule_evaluation`` / ``trend`` / ``session`` / ``event``。ルール評価は単一結果の
    ほか、``{"entry": ..., "holding": ...}`` と ``position_mode`` の組合せも受ける。
    返す判定は常に既存ルール評価をそのまま採用し、補助情報による加点・上書きを
    行わない。
    """
    if inputs is None:
        payload: dict[str, Any] = {}
    elif isinstance(inputs, Mapping):
        payload = dict(inputs)
    else:
        raise TypeError("inputsは辞書形式で指定してください")

    evaluation = _select_rule_evaluation(payload)
    verdict = _verdict(evaluation, payload.get("position_mode"))
    current_price = _current_price(payload)
    plan, plan_warnings = _risk_plan(payload, evaluation, current_price["value"])
    support, resistance = _nearest_levels(payload.get("levels"), current_price["value"])
    trend = _trend(payload, evaluation)
    session = _session(payload)
    event_risk = _event_risk(payload)
    quality = _data_quality(
        price=current_price, verdict=verdict, plan=plan,
        support=support, resistance=resistance, trend=trend,
        session=session, event_risk=event_risk, warnings=plan_warnings,
    )
    priorities = _action_priorities(
        verdict=verdict, plan=plan, session=session,
        event_risk=event_risk, quality=quality,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "current_price": current_price,
        "entry": plan["entry"],
        "stop": plan["stop"],
        "target": plan["target"],
        "rr": plan["rr"],
        "risk_plan_valid": plan["valid"],
        "support": support,
        "resistance": resistance,
        "trend": trend,
        "session": session,
        "event_risk": event_risk,
        "data_quality": quality,
        "action_priorities": priorities,
        "score_effect": 0,
        "automatic_trade_score": False,
        "read_only": True,
        "places_orders": False,
        "uses_network": False,
        "disclaimer": (
            "既存の分析結果を整理した参考情報です。売買判断や利益を保証せず、"
            "注文は実行しません。"
        ),
    }


def calculate_position_size(
    purchase_price: Any,
    stop_price: Any,
    max_loss: Any,
    max_investment: Any = None,
) -> dict:
    """許容損失と購入予算から、買える整数株数の上限を返す。

    株価や損切り価格は呼び出し側が確定した値だけを受け取り、この関数では
    価格を補完しない。1株当たり損失が許容損失を超える場合は0株を返す。
    """
    price = _number(purchase_price, positive=True)
    stop = _number(stop_price, positive=True)
    loss_limit = _number(max_loss, positive=True)
    budget = (None if max_investment is None
              else _number(max_investment, positive=True))
    errors: list[str] = []
    if price is None:
        errors.append("購入価格を確認できません")
    if stop is None:
        errors.append("損切り価格を確認できません")
    if loss_limit is None:
        errors.append("許容損失は0より大きい金額で入力してください")
    if max_investment is not None and budget is None:
        errors.append("購入予算は0より大きい金額で入力してください")
    if price is not None and stop is not None and stop >= price:
        errors.append("損切り価格は購入価格より低く設定してください")

    if errors:
        return {
            "available": False,
            "shares": 0,
            "shares_by_loss": None,
            "shares_by_budget": None,
            "purchase_price": price,
            "stop_price": stop,
            "risk_per_share": None,
            "max_loss": loss_limit,
            "max_investment": budget,
            "estimated_cost": None,
            "estimated_max_loss": None,
            "estimated_planned_loss": None,
            "can_buy_one_share": False,
            "limiting_factor": None,
            "reason_ja": " ／ ".join(errors),
        }

    risk_per_share = price - stop
    shares_by_loss = max(int(math.floor(loss_limit / risk_per_share)), 0)
    shares_by_budget = (
        None if budget is None else max(int(math.floor(budget / price)), 0)
    )
    shares = (shares_by_loss if shares_by_budget is None
              else min(shares_by_loss, shares_by_budget))
    if shares_by_budget is None or shares_by_loss < shares_by_budget:
        limiting = "許容損失"
    elif shares_by_budget < shares_by_loss:
        limiting = "購入予算"
    else:
        limiting = "許容損失と購入予算"
    reason = (
        "1株でも許容損失または購入予算を超えるため、上限は0株です"
        if shares == 0 else
        f"{limiting}を超えない整数株数です"
    )
    return {
        "available": True,
        "shares": shares,
        "shares_by_loss": shares_by_loss,
        "shares_by_budget": shares_by_budget,
        "purchase_price": price,
        "stop_price": stop,
        "risk_per_share": risk_per_share,
        "max_loss": loss_limit,
        "max_investment": budget,
        "estimated_cost": shares * price,
        "estimated_max_loss": shares * risk_per_share,
        "estimated_planned_loss": shares * risk_per_share,
        "can_buy_one_share": shares >= 1,
        "limiting_factor": limiting,
        "reason_ja": reason,
    }


def _required_entry_rr(evaluation: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """既存の買い条件から、実行価格にも必要な最低R:Rを取り出す。"""
    buy = _mapping(evaluation.get("buy"))
    checks = buy.get("checks")
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        return None, None
    thresholds = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        if str(check.get("metric") or "") != "risk_reward":
            continue
        if str(check.get("op") or "") != ">=":
            continue
        if check.get("required") is not True:
            continue
        threshold = _number(check.get("threshold"), positive=True)
        if threshold is not None:
            thresholds.append(threshold)
    if not thresholds:
        return None, None
    return max(thresholds), "既存の買い条件（損失と利益の比率）"


def _daily_buy_signal(evaluation: Mapping[str, Any]) -> dict:
    """現在気配や外部ゲートと分けて、確定日足側の条件だけを投影する。"""
    buy = _mapping(evaluation.get("buy"))
    regime_gate = next((
        gate for gate in evaluation.get("gates") or ()
        if isinstance(gate, Mapping)
        and str(gate.get("key") or "") == "regime"
        and str(gate.get("source") or "") == "engine"
    ), None)
    if not buy or not buy.get("enabled") or not buy.get("valid") or not buy.get("available"):
        return {
            "verdict": "WAIT", "is_buy_candidate": False, "available": False,
            "label_ja": "日足の買い条件を確認できません",
            "reason_ja": "確定日足の設定または指標が不足しています。",
        }
    if regime_gate is not None and regime_gate.get("passed") is not True:
        return {
            "verdict": "WAIT", "is_buy_candidate": False, "available": True,
            "label_ja": "日足の相場条件を待ちます",
            "reason_ja": str(regime_gate.get("reason") or "利用中ルールの相場条件外です。"),
        }
    if buy.get("passed") is True:
        return {
            "verdict": "BUY", "is_buy_candidate": True, "available": True,
            "label_ja": "日足の買い条件は成立",
            "reason_ja": "確定日足の条件と点数は成立しています。",
        }
    return {
        "verdict": "NEUTRAL", "is_buy_candidate": False, "available": True,
        "label_ja": "日足の買い条件は未成立",
        "reason_ja": "確定日足の条件または点数が足りません。",
    }


def _purchase_price_zone(
    summary: Mapping[str, Any],
    minimum_rr: float | None,
) -> dict:
    """既存risk planと最低R:Rだけから、判定価格と購入上限を返す。"""
    entry = _number(_mapping(summary.get("entry")).get("price"), positive=True)
    stop = _number(_mapping(summary.get("stop")).get("price"), positive=True)
    target = _number(_mapping(summary.get("target")).get("price"), positive=True)
    if not summary.get("risk_plan_valid") or None in (entry, stop, target):
        return {
            "available": False, "low": None, "high": None,
            "reference_price": entry, "minimum_rr": minimum_rr,
            "reason_ja": "損切り・利益確定の価格計画が揃っていません",
            "method_ja": None,
        }
    if minimum_rr is None:
        return {
            "available": False, "low": None, "high": None,
            "reference_price": entry, "minimum_rr": None,
            "reason_ja": "買い条件に必要な損失と利益の比率を確認できません",
            "method_ja": None,
        }

    # (target - price) / (price - stop) >= minimum_rr を満たす価格上限。
    high = (target + minimum_rr * stop) / (1.0 + minimum_rr)
    if not math.isfinite(high) or high < entry or high >= target:
        return {
            "available": False, "low": None, "high": None,
            "reference_price": entry, "minimum_rr": minimum_rr,
            "reason_ja": "現在の価格計画では、必要な損失と利益の比率を保てる買い上限を作れません",
            "method_ja": None,
        }
    return {
        "available": True,
        # entryは買い下限ではなく、確定日足で価格計画を作った基準値。
        "low": None,
        "high": high,
        "reference_price": entry,
        "maximum_price": high,
        "minimum_rr": minimum_rr,
        "reason_ja": None,
        "method_ja": (
            "判定価格を基準に、既存ルールの損失と利益の比率を保てる上限を逆算"
        ),
    }


def _live_risk_reward(price: float | None, stop: float | None,
                      target: float | None) -> float | None:
    if None in (price, stop, target) or not stop < price < target:
        return None
    return (target - price) / (price - stop)


def _datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    converter = getattr(value, "to_pydatetime", None)
    if callable(converter):
        try:
            converted = converter()
            return converted if isinstance(converted, datetime) else None
        except (TypeError, ValueError, OverflowError):
            return None
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _quote_age_seconds(value: Any, now: Any,
                       assumed_timezone: Any = None) -> float | None:
    quote_time = _datetime_value(value)
    current = _datetime_value(now) if now is not None else datetime.now(timezone.utc)
    if quote_time is None or current is None:
        return None
    if quote_time.tzinfo is None and current.tzinfo is not None:
        if assumed_timezone is None:
            return None
        quote_time = quote_time.replace(tzinfo=assumed_timezone)
    elif quote_time.tzinfo is not None and current.tzinfo is None:
        current = current.replace(tzinfo=quote_time.tzinfo)
    try:
        return (current - quote_time).total_seconds()
    except TypeError:
        return None


def _suspension_state(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().upper()
    if text in {"1", "TRUE", "YES", "Y", "SUSPENDED", "SUSPEND"}:
        return True
    if text in {"0", "FALSE", "NO", "N", "NORMAL", "NONE",
                "NOT_SUSPENDED", "UNSUSPENDED"}:
        return False
    return None


def _imminent_high_event(report: Any, now: Any = None) -> dict | None:
    """表示用の直近1件とは別に、2日以内の高影響イベントを全件確認する。"""
    source = _mapping(report)
    raw_events = source.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        return None
    base_day = _date(source.get("as_of"))
    if base_day is None:
        current = _datetime_value(now) if now is not None else datetime.now(timezone.utc)
        base_day = current.date() if current is not None else None
    candidates = []
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            continue
        if str(event.get("status") or "").strip().upper() != "UPCOMING":
            continue
        if str(event.get("impact_level") or "").strip().upper() != "HIGH":
            continue
        days = _number(event.get("days_until"))
        event_day = _date(event.get("event_date"))
        if days is None and base_day is not None and event_day is not None:
            days = float((event_day - base_day).days)
        if days is None or not 0 <= days <= 2:
            continue
        candidates.append((int(days), index, dict(event)))
    if not candidates:
        return None
    days, _, event = min(candidates, key=lambda row: (row[0], row[1]))
    name = event.get("display_name_ja", event.get("name"))
    return {
        "event_name": str(name) if name else "高影響イベント",
        "event_date": event.get("event_date"),
        "days_until": days,
    }


def build_purchase_plan(
    inputs: Mapping[str, Any] | None = None,
    *,
    max_loss: Any = None,
    max_investment: Any = None,
    max_spread_pct: Any = None,
    quote_max_age_seconds: Any = 60,
    now: Any = None,
) -> dict:
    """既存の買い判定を、購入直前に確認する実行計画へ投影する。

    判定や価格は新しく推測しない。``BUY`` 以外を購入可能へ変更せず、既存の
    必須ゲート・blocking、明示された取引可否、risk planを優先する。現在の
    購入準備は日足の買い判定と分離し、イベントや気配値で既存判定を変更しない。
    """
    if inputs is None:
        payload: dict[str, Any] = {}
    elif isinstance(inputs, Mapping):
        payload = dict(inputs)
    else:
        raise TypeError("inputsは辞書形式で指定してください")

    entry_payload = dict(payload)
    entry_payload["position_mode"] = "entry"
    summary = build_trade_summary(entry_payload)
    evaluation = _select_rule_evaluation(entry_payload)
    verdict = _mapping(summary.get("verdict"))
    verdict_code = str(verdict.get("code") or "WAIT")
    daily_signal = _daily_buy_signal(evaluation)
    minimum_rr, rr_source = _required_entry_rr(evaluation)
    zone = _purchase_price_zone(summary, minimum_rr)

    snapshot = _mapping(payload.get("snapshot"))
    bid = _number(snapshot.get("bid"), positive=True)
    ask = _number(snapshot.get("ask"), positive=True)
    current = _number(_mapping(summary.get("current_price")).get("value"), positive=True)
    execution_price = ask
    execution_source = "現在の売気配" if ask is not None else None
    entry = _number(_mapping(summary.get("entry")).get("price"), positive=True)
    stop = _number(_mapping(summary.get("stop")).get("price"), positive=True)
    target = _number(_mapping(summary.get("target")).get("price"), positive=True)
    live_rr = _live_risk_reward(execution_price, stop, target)
    spread_pct = None
    if bid is not None and ask is not None and ask >= bid:
        midpoint = (ask + bid) / 2.0
        spread_pct = (ask - bid) / midpoint * 100.0 if midpoint > 0 else None
    session_time = _datetime_value(_mapping(summary.get("session")).get("as_of"))
    assumed_timezone = None if session_time is None else session_time.tzinfo
    quote_age = _quote_age_seconds(snapshot.get("update_time"), now, assumed_timezone)
    quote_age_limit = _number(quote_max_age_seconds, positive=True)
    configured_spread_limit = (
        None if max_spread_pct is None else _number(max_spread_pct, positive=True)
    )
    suspended = _suspension_state(snapshot.get("suspension"))

    wait_reasons: list[dict] = []
    cautions: list[dict] = []
    reason_codes: set[str] = set()

    def add_reason(code: str, label: str, detail: str, source: str,
                   *, category: str = "DATA", unavailable: bool = False) -> None:
        if code in reason_codes:
            return
        reason_codes.add(code)
        wait_reasons.append({
            "code": code, "label_ja": label, "detail_ja": detail,
            "source": source, "category": category,
            "unavailable": unavailable,
        })

    def add_caution(code: str, label: str, detail: str) -> None:
        cautions.append({"code": code, "label_ja": label, "detail_ja": detail})

    # 既存entry判定とblockingを最優先に並べる。
    if not verdict.get("available"):
        add_reason("entry_unavailable", "買い判定を確認できません",
                   str(verdict.get("reason") or "判定データがありません"),
                   "既存の買い判定", category="SIGNAL", unavailable=True)
    elif verdict_code == "WAIT":
        add_reason("entry_wait", "買い判定が待機です",
                   str(verdict.get("reason") or "確認事項が残っています"),
                   "既存の買い判定", category="SIGNAL")
    elif verdict_code == "NEUTRAL":
        add_reason("entry_not_ready", "買い条件がまだ揃っていません",
                   str(verdict.get("reason") or "買い条件の成立を待ちます"),
                   "既存の買い判定", category="SIGNAL")
    elif verdict_code != "BUY":
        add_reason("wrong_position_mode", "買い判定ではありません",
                   "保有中の判定結果は、新しく買う判断には使いません。",
                   "既存の買い判定", category="SIGNAL", unavailable=True)

    for action in summary.get("action_priorities") or ():
        if isinstance(action, Mapping) and action.get("blocking") is True:
            add_reason(
                f"existing_{action.get('code') or 'blocking'}",
                str(action.get("label_ja") or "既存の確認事項があります"),
                str(action.get("detail_ja") or "確認後にもう一度判定してください"),
                "既存のblocking", category="SIGNAL",
            )
    for gate in evaluation.get("gates") or ():
        if not isinstance(gate, Mapping):
            continue
        failed = gate.get("passed") is False and (
            gate.get("required") is True or str(gate.get("key") or "") == "spread"
        )
        if failed:
            add_reason(
                f"gate_{gate.get('key') or 'unknown'}",
                str(gate.get("label") or "買う前の確認項目が未成立です"),
                str(gate.get("reason") or "条件を確認してください"),
                "既存の必須ゲート",
                category=("LIQUIDITY" if str(gate.get("key")) == "spread"
                          else "DATA"),
            )

    tradable = _mapping(summary.get("session")).get("tradable")
    if tradable is False:
        add_reason("session_not_tradable", "現在は購入できる取引時間ではありません",
                   str(_mapping(summary.get("session")).get("reason")
                       or "取引可能な時間に再確認してください"),
                   "取得済みセッション情報", category="SESSION")
    elif tradable is not True:
        add_reason("session_unknown", "現在購入できるか確認できません",
                   str(_mapping(summary.get("session")).get("reason")
                       or "取引可否を確認してから進んでください"),
                   "取得済みセッション情報", category="SESSION")

    if not zone.get("available"):
        add_reason("price_zone_unavailable", "買う価格の上限を作れません",
                   str(zone.get("reason_ja")), "既存の価格計画",
                   category="PRICE", unavailable=True)

    # 日足BUYとは別に、現在の購入準備をquoteから確認する。
    if ask is None:
        add_reason("ask_unavailable", "現在の売気配を確認できません",
                   "実際に買える価格が分からないため、売気配の取得後に再確認してください。",
                   "取得済みsnapshot", category="DATA", unavailable=True)
    if bid is None or spread_pct is None:
        add_reason("spread_unavailable", "売買価格の差を確認できません",
                   "買値と売値の差が分からないため、価格差の取得後に再確認してください。",
                   "取得済みsnapshot", category="LIQUIDITY", unavailable=True)
    if quote_age_limit is None:
        add_reason("quote_age_setting_invalid", "価格の鮮度基準を確認できません",
                   "価格の鮮度基準は0秒より大きく設定してください。",
                   "購入計画の設定", category="DATA", unavailable=True)
    elif quote_age is None:
        add_reason("quote_time_unavailable", "価格の更新時刻を確認できません",
                   "古い価格で買わないよう、更新時刻を取得してから再確認してください。",
                   "取得済みsnapshot", category="DATA", unavailable=True)
    elif quote_age < -5:
        add_reason("quote_time_inconsistent", "価格の更新時刻を確認してください",
                   "価格の更新時刻が現在時刻より未来になっています。",
                   "取得済みsnapshot", category="DATA", unavailable=True)
    elif quote_age > quote_age_limit:
        add_reason("quote_stale", "表示価格が古くなっています",
                   f"更新から{quote_age:.0f}秒経過しています。最新の気配値を取得してください。",
                   "取得済みsnapshot", category="DATA")
    if suspended is True:
        add_reason("trading_suspended", "この銘柄は取引停止中です",
                   "取引再開と最新の価格計画を確認するまで購入しません。",
                   "取得済みsnapshot", category="SESSION")
    elif suspended is None:
        add_reason("suspension_unavailable", "取引停止の状態を確認できません",
                   "取引可能な状態か確認できるまで購入判断を保留します。",
                   "取得済みsnapshot", category="DATA", unavailable=True)

    spread_gate = next((
        gate for gate in evaluation.get("gates") or ()
        if isinstance(gate, Mapping) and str(gate.get("key") or "") == "spread"
    ), None)
    if max_spread_pct is not None and configured_spread_limit is None:
        add_reason("spread_limit_invalid", "価格差の上限を確認できません",
                   "価格差の上限は0より大きい割合で設定してください。",
                   "購入計画の設定", category="LIQUIDITY", unavailable=True)
    elif spread_pct is not None and configured_spread_limit is not None:
        if spread_pct > configured_spread_limit:
            add_reason("spread_too_wide", "売買価格の差が大きすぎます",
                       f"現在 {spread_pct:.3f}% / 上限 {configured_spread_limit:.3f}% です。",
                       "取得済みsnapshot", category="LIQUIDITY")
    elif spread_pct is not None:
        if spread_gate is None or spread_gate.get("passed") is not True:
            add_reason("spread_limit_unavailable", "価格差が許容範囲か確認できません",
                       "既存の価格差ゲートまたは上限設定を確認してください。",
                       "既存の必須ゲート", category="LIQUIDITY", unavailable=True)

    chase_status = "unavailable"
    chase_warning: bool | None = None
    if execution_price is not None and stop is not None and target is not None:
        if execution_price <= stop:
            chase_status, chase_warning = "plan_broken", True
            add_reason("stop_reached", "価格計画を作り直してください",
                       "購入参考価格が損切りの目安以下です。以前の買い計画をそのまま使いません。",
                       "既存の価格計画", category="PRICE")
        elif execution_price >= target:
            chase_status, chase_warning = "target_reached", True
            add_reason("target_reached", "利益確定の目安まで上昇しています",
                       "購入参考価格が利益確定の目安以上のため、追いかけて買いません。",
                       "既存の価格計画", category="PRICE")
        elif zone.get("available") and execution_price > zone["high"]:
            chase_status, chase_warning = "above_limit", True
            add_reason("chasing_price", "追いかけ買いになる価格です",
                       "現在の購入参考価格では、ルールに必要な損失と利益の比率を保てません。",
                       "既存の価格計画", category="PRICE")
        elif zone.get("available"):
            chase_status, chase_warning = "within_limit", False
            if entry is not None and execution_price < entry:
                add_caution(
                    "below_reference",
                    "判定に使った価格より安くなっています",
                    "損失と利益の比率は改善しますが、下落が続いていないかチャートを再確認してください。",
                )

    event = _mapping(summary.get("event_risk"))
    imminent_high = _imminent_high_event(
        payload.get("event", payload.get("events")), now=now)
    if event.get("available") is not True:
        add_reason("event_unavailable", "今後のイベントを確認できません",
                   str(event.get("reason") or "予定なしとは判断せず、取得後に再確認してください。"),
                   "取得済みイベント情報", category="EVENT", unavailable=True)
    elif str(event.get("report_status") or "").lower() == "partial":
        add_reason("event_partial", "イベント情報が一部不足しています",
                   "未取得の情報に重要イベントが含まれる可能性があるため、更新後に再確認してください。",
                   "取得済みイベント情報", category="EVENT", unavailable=True)
    elif imminent_high is not None:
        add_reason(
            "high_impact_event",
            "2日以内に大きなイベントがあります",
            f"{imminent_high['event_name']}の前後は値動きが大きくなるため、通過後に再確認してください。",
            "取得済みイベント情報", category="EVENT",
        )

    sizing_price = zone.get("high") if zone.get("available") else None
    position_size = calculate_position_size(
        sizing_price, stop, max_loss, max_investment,
    )
    position_size.update({
        "planned_purchase_price": sizing_price,
        "target_price": target,
        "estimated_target_profit": (
            None if not position_size.get("available") or target is None
            else position_size["shares"] * (target - position_size["purchase_price"])
        ),
    })
    size_requested = max_loss is not None or max_investment is not None
    if size_requested and not position_size.get("available"):
        add_reason("position_size_unavailable", "購入株数を計算できません",
                   str(position_size.get("reason_ja")), "通常時の許容損失と購入予算",
                   category="RISK", unavailable=True)
    elif size_requested and position_size["shares"] == 0:
        add_reason("position_size_zero", "購入上限は0株です",
                   str(position_size.get("reason_ja")), "通常時の許容損失と購入予算",
                   category="RISK")
    add_caution(
        "gap_slippage_risk",
        "実際の損失は想定を超えることがあります",
        "価格の飛びや注文価格との差により、損切り価格で売れず通常時の想定損失を超える場合があります。",
    )

    unavailable = any(item.get("unavailable") for item in wait_reasons)
    if not verdict.get("available") or verdict_code not in {"BUY", "WAIT", "NEUTRAL"}:
        status = "UNAVAILABLE"
    elif verdict_code == "NEUTRAL":
        status = "NOT_CANDIDATE"
    elif verdict_code == "WAIT":
        status = "WAIT"
    elif unavailable:
        status = "UNAVAILABLE"
    elif wait_reasons:
        status = "WAIT"
    else:
        status = "READY"

    display = {
        "READY": (
            "購入を検討できる状態",
            "買い条件と価格・損失管理の確認が揃っています。上限価格と株数を守って検討してください。",
        ),
        "WAIT": (
            "今は待つ",
            wait_reasons[0]["detail_ja"] if wait_reasons else "確認事項が残っています。",
        ),
        "NOT_CANDIDATE": (
            "今は買い候補ではありません",
            "既存ルールの買い条件が揃うまで待ちます。",
        ),
        "UNAVAILABLE": (
            "購入判断ができません",
            wait_reasons[0]["detail_ja"] if wait_reasons else "必要な情報を確認できません。",
        ),
    }
    label, description = display[status]
    wait_status = None if status == "READY" else (
        "WAIT_" + str(wait_reasons[0].get("category") or "DATA")
        if wait_reasons else "WAIT_DATA"
    )
    return {
        "schema_version": 1,
        "status": status,
        "label_ja": label,
        "description_ja": description,
        "can_consider_purchase": status == "READY",
        "actionable": status == "READY",
        "authoritative_verdict": verdict_code,
        "daily_signal": {
            **daily_signal,
            "authoritative_entry_verdict": verdict_code,
            "unchanged_by_execution_checks": True,
        },
        "execution_readiness": {
            "status": wait_status or "READY",
            "actionable": status == "READY",
            "label_ja": label,
            "description_ja": description,
        },
        "wait_reasons": wait_reasons,
        "cautions": cautions,
        "buy_zone": zone,
        "execution_price": {
            "value": execution_price,
            "source": execution_source,
            "uses_ask": ask is not None,
        },
        "quote": {
            "bid": bid,
            "ask": ask,
            "spread_pct": spread_pct,
            "max_spread_pct": configured_spread_limit,
            "update_time": snapshot.get("update_time"),
            "age_seconds": quote_age,
            "max_age_seconds": quote_age_limit,
            "suspended": suspended,
        },
        "chase_warning": {
            "status": chase_status,
            "warned": chase_warning,
            "current_rr": live_rr,
            "minimum_rr": minimum_rr,
            "minimum_rr_source": rr_source,
            "maximum_price": zone.get("high"),
            "excess_pct": (
                None if execution_price is None or not zone.get("available")
                else (execution_price / zone["high"] - 1.0) * 100
            ),
        },
        "risk": {
            "entry": entry,
            "stop": stop,
            "target": target,
            "planned_rr": _mapping(summary.get("rr")).get("value"),
            "current_rr": live_rr,
            "minimum_rr": minimum_rr,
            "risk_plan_valid": bool(summary.get("risk_plan_valid")),
        },
        "position_size": position_size,
        "position_size_basis_ja": (
            "買う価格の上限で約定しても許容損失を超えない株数"
            if position_size.get("available") else None
        ),
        "support": deepcopy(summary.get("support")),
        "resistance": deepcopy(summary.get("resistance")),
        "session": deepcopy(summary.get("session")),
        "event_risk": deepcopy(summary.get("event_risk")),
        "data_quality": deepcopy(summary.get("data_quality")),
        "score_effect": 0,
        "automatic_trade_score": False,
        "read_only": True,
        "places_orders": False,
        "uses_network": False,
        "uses_moomoo_history_quota": False,
        "disclaimer": (
            "既存の分析結果から購入前の確認事項を整理した参考情報です。"
            "注文は実行せず、売買や利益を保証しません。価格の飛びや注文価格との差で、"
            "実際の損失が通常時の想定損失を超える場合があります。"
        ),
    }


__all__ = [
    "SCHEMA_VERSION", "VERDICT_CODES", "build_trade_summary",
    "build_purchase_plan", "calculate_position_size",
]
