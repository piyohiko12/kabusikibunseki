"""既取得データだけから、銘柄別の売買情報を一画面向けに整理する。

このモジュールは表示用の投影層であり、売買判定を新しく生成しない。とくに、
イベント・セッション・トレンド・支持抵抗を既存ルールのスコアへ加点せず、
注文APIも呼び出さない。入力不足時は値を推測せず ``None`` と理由を返す。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
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
        return {
            "value": explicit,
            "available": True,
            "source": str(source or "呼び出し側の現在値"),
            "as_of": payload.get("current_price_as_of", snapshot.get("update_time")),
            "quality": (
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


__all__ = ["SCHEMA_VERSION", "VERDICT_CODES", "build_trade_summary"]
