"""既に取得済みの銘柄情報を、読み取り専用の掲示板項目へ整理する。

このモジュールは純粋な投影層である。ネットワーク取得、投稿、保存、注文、
moomoo 過去K線の追加取得は一切行わない。入力にない事実を推測で補わず、
不正または不足した行は警告付きで読み飛ばす。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import ipaddress
import json
import math
import re
import socket
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = 1

CATEGORY_LABELS = {
    "price": "株価・市場",
    "trade": "売買判定",
    "level": "支持・抵抗",
    "alert": "株価アラーム",
    "analyst": "アナリスト・決算",
    "event": "イベント",
    "news": "ニュース",
}

IMPORTANCE_LABELS = {
    "critical": "最重要",
    "high": "重要",
    "medium": "要確認",
    "low": "参考",
    "info": "情報",
}

IMPORTANCE_RANKS = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "info": 1,
}

ITEM_FIELDS = (
    "id",
    "ticker",
    "category",
    "category_label_ja",
    "kind",
    "importance",
    "importance_label_ja",
    "importance_rank",
    "title_ja",
    "summary_ja",
    "occurred_at",
    "source",
    "url",
    "tags",
    "data",
)

_CATEGORY_ORDER = {code: index for index, code in enumerate(CATEGORY_LABELS)}

_VERDICT_LABELS = {
    "BUY": "買い条件成立",
    "WAIT": "待機",
    "NEUTRAL": "中立",
    "RISK_EXIT": "リスク退出条件成立",
    "TAKE_PROFIT": "利確条件成立",
    "HOLD": "保有継続",
}

_ALERT_LABELS = {
    "price_above": "価格が上回る",
    "price_below": "価格が下回る",
    "change_above": "前日比が上回る",
    "change_below": "前日比が下回る",
    "rsi_above": "RSI(14)が上回る",
    "rsi_below": "RSI(14)が下回る",
    "near_support": "サポートに近づく",
    "near_resistance": "抵抗線に近づく",
    "rule_buy": "買い判定になる",
    "rule_take_profit": "利益確定判定になる",
    "rule_risk_exit": "リスク退出判定になる",
    "rule_sell": "手仕舞い判定になる（旧形式）",
}

_EVENT_STATUS_LABELS = {
    "UPCOMING": "今後の予定",
    "RECENT": "最近のイベント",
    "HISTORICAL": "過去のイベント",
    "UNKNOWN": "時期不明",
}

_RATING_LABELS = {
    "strongBuy": "強い買い",
    "buy": "買い",
    "hold": "中立",
    "sell": "売り",
    "strongSell": "強い売り",
}

_ACTION_LABELS = {
    "up": "格上げ",
    "upgrade": "格上げ",
    "upgraded": "格上げ",
    "down": "格下げ",
    "downgrade": "格下げ",
    "downgraded": "格下げ",
    "init": "新規評価",
    "initiated": "新規評価",
    "reit": "評価継続",
    "reiterated": "評価継続",
    "maintain": "評価維持",
    "maintained": "評価維持",
}

_DROP = object()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _number(value: Any, *, positive: bool = False, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    if positive and result <= 0:
        return None
    if nonnegative and result < 0:
        return None
    return result


def _one_line(value: Any, maximum: int = 300) -> str | None:
    if value is None:
        return None
    if isinstance(value, (Mapping, list, tuple, set)):
        return None
    try:
        text = str(value)
    except Exception:
        return None
    if any(ord(character) == 0 for character in text):
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maximum] if text else None


def _ticker(value: Any) -> str | None:
    text = _one_line(value, 32)
    if not text:
        return None
    text = text.upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=_:/-]{0,31}", text):
        return None
    return text


def _timestamp(value: Any) -> str | None:
    """比較可能な日時だけを ISO 形式で返す。解釈不能な文字列は捨てる。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _one_line(value, 100)
    if not text:
        return None
    # 日付だけの入力へ存在しない時刻（00:00）を付けない。
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(candidate).isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _timestamp_number(value: Any) -> float | None:
    normalized = _timestamp(value)
    if normalized is None:
        return None
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def safe_url(value: Any) -> str | None:
    """画面上の外部リンクとして扱える http/https URLだけを返す。

    認証情報入りURL、localhost、ローカル・予約済みIP、制御文字や空白を含む
    URLはリンク化しない。これは取得関数ではなく、文字列検査だけを行う。
    """
    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url or len(url) > 2_048:
        return None
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127
           for character in url):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"}:
            return None
        if not parsed.netloc or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        # 不正なポートは property 参照時に ValueError になる。
        _ = parsed.port
    except (TypeError, ValueError):
        return None

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
        return None
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        address = None
    if address is None:
        # URLで許される旧式IPv4表記（2130706433、0177.0.0.1、0x7f000001等）も
        # ローカルアドレスへ解釈され得るため、DNSを使わないinet_atonで検査する。
        try:
            address = ipaddress.ip_address(socket.inet_aton(hostname))
        except (OSError, ValueError):
            address = None
    if address is not None and (
        address.is_private or address.is_loopback or address.is_link_local
        or address.is_multicast or address.is_reserved or address.is_unspecified
    ):
        return None
    return url


def _json_value(value: Any, *, depth: int = 0) -> Any:
    """表示用dataをJSON互換の事実値だけへ制限する。"""
    if depth > 6:
        return _DROP
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _DROP
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        result = {}
        for key in sorted(value, key=lambda item: str(item)):
            key_text = _one_line(key, 100)
            if not key_text:
                continue
            cleaned = _json_value(value[key], depth=depth + 1)
            if cleaned is not _DROP:
                result[key_text] = cleaned
        return result
    if _is_sequence(value):
        result = []
        for item in value[:100]:
            cleaned = _json_value(item, depth=depth + 1)
            if cleaned is not _DROP:
                result.append(cleaned)
        return result
    return _DROP


def _canonical(value: Any) -> str:
    cleaned = _json_value(value)
    if cleaned is _DROP:
        cleaned = None
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_item(item: Mapping[str, Any] | None, *, default_ticker: Any = None) -> dict | None:
    """1件を固定スキーマへ正規化する。必須事実がなければ ``None`` を返す。"""
    if not isinstance(item, Mapping):
        return None
    source = dict(item)
    category = _one_line(source.get("category"), 32)
    if category not in CATEGORY_LABELS:
        return None
    kind = _one_line(source.get("kind"), 64)
    title = _one_line(source.get("title_ja", source.get("title")), 300)
    if not kind or not title:
        return None

    importance = _one_line(source.get("importance"), 32) or "info"
    if importance not in IMPORTANCE_LABELS:
        importance = "info"
    summary = _one_line(source.get("summary_ja", source.get("summary")), 2_000)
    ticker = _ticker(source.get("ticker")) or _ticker(default_ticker)
    occurred_at = _timestamp(source.get("occurred_at"))
    provider = _one_line(source.get("source"), 200)
    url = safe_url(source.get("url"))

    raw_tags = source.get("tags")
    tags: list[str] = []
    if _is_sequence(raw_tags):
        for raw in raw_tags:
            tag = _one_line(raw, 50)
            if tag and tag not in tags:
                tags.append(tag)

    raw_data = source.get("data")
    data = _json_value(raw_data if isinstance(raw_data, Mapping) else {})
    if not isinstance(data, dict):
        data = {}

    explicit_identity = _one_line(source.get("identity_key"), 1_000)
    identity = explicit_identity or _canonical({
        "ticker": ticker,
        "category": category,
        "kind": kind,
        "title": title,
        "occurred_at": occurred_at,
        "source": provider,
        "url": url,
    })
    item_id = sha256(identity.encode("utf-8")).hexdigest()[:20]

    normalized = {
        "id": item_id,
        "ticker": ticker,
        "category": category,
        "category_label_ja": CATEGORY_LABELS[category],
        "kind": kind,
        "importance": importance,
        "importance_label_ja": IMPORTANCE_LABELS[importance],
        "importance_rank": IMPORTANCE_RANKS[importance],
        "title_ja": title,
        "summary_ja": summary,
        "occurred_at": occurred_at,
        "source": provider,
        "url": url,
        "tags": tags,
        "data": data,
    }
    return normalized


def _completeness(item: Mapping[str, Any]) -> tuple[int, int, str]:
    present = sum(bool(item.get(key)) for key in (
        "summary_ja", "occurred_at", "source", "url", "tags", "data",
    ))
    return (int(item.get("importance_rank") or 0), present, _canonical(item))


def _merge_duplicate_items(left: dict, right: dict) -> dict:
    """同じ情報源の2項目を、入力順に依存せず表示情報を失わず統合する。"""
    preferred, other = (
        (right, left) if _completeness(right) > _completeness(left)
        else (left, right)
    )
    merged = deepcopy(preferred)
    parts: list[str] = []
    preferred_summary = _one_line(preferred.get("summary_ja"), 2_000)
    if preferred_summary:
        parts.append(preferred_summary)
    other_title = _one_line(other.get("title_ja"), 300)
    preferred_title = _one_line(preferred.get("title_ja"), 300)
    if (other_title and other_title != preferred_title
            and not any(other_title in part for part in parts)):
        label = "原文見出し" if other.get("category") == "news" else "関連情報"
        parts.append(f"{label}: {other_title}")
    other_summary = _one_line(other.get("summary_ja"), 2_000)
    if other_summary and other_summary not in parts:
        parts.append(other_summary)
    merged["summary_ja"] = _one_line(" ／ ".join(parts), 2_000)

    merged["tags"] = list(dict.fromkeys([
        *list(preferred.get("tags") or []),
        *list(other.get("tags") or []),
    ]))
    merged["data"] = {
        **_mapping(other.get("data")),
        **_mapping(preferred.get("data")),
    }
    merged["source"] = preferred.get("source") or other.get("source")
    merged["url"] = preferred.get("url") or other.get("url")
    merged["occurred_at"] = preferred.get("occurred_at") or other.get("occurred_at")
    return merged


def _dedupe(items: Iterable[dict]) -> tuple[list[dict], int]:
    by_id: dict[str, dict] = {}
    total = 0
    for item in items:
        total += 1
        current = by_id.get(item["id"])
        if current is None:
            by_id[item["id"]] = item
        else:
            by_id[item["id"]] = _merge_duplicate_items(current, item)

    # 同じ原文URLから作られたニュースとイベントは、カテゴリが異なっても
    # 二重表示しない。重要度と情報充足度が高い方を決定論的に残す。
    by_source: dict[str, dict] = {}
    for item in by_id.values():
        url = item.get("url")
        data = _mapping(item.get("data"))
        earnings_date = data.get("earnings_date") or data.get("event_date")
        is_earnings = (
            item.get("kind") == "earnings"
            and item.get("category") in {"analyst", "event"}
            and earnings_date
        )
        if is_earnings:
            source_key = (
                f"earnings|{item.get('ticker') or ''}|"
                f"{str(earnings_date)[:10]}"
            )
        elif url and item.get("category") in {"event", "news"}:
            source_key = f"external-url|{url}"
        else:
            source_key = f"item-id|{item['id']}"
        current = by_source.get(source_key)
        if current is None:
            by_source[source_key] = item
        else:
            by_source[source_key] = _merge_duplicate_items(current, item)
    return list(by_source.values()), total - len(by_source)


def _importance_for_verdict(verdict: str, mode: str) -> str:
    if mode == "holding":
        return {
            "RISK_EXIT": "critical", "TAKE_PROFIT": "high",
            "WAIT": "medium", "HOLD": "low",
        }.get(verdict, "info")
    return {"BUY": "high", "WAIT": "medium", "NEUTRAL": "low"}.get(
        verdict, "info")


def _score_data(evaluation: Mapping[str, Any], mode: str) -> tuple[dict, list[str]]:
    result: dict[str, Any] = {}
    summary_parts: list[str] = []
    if mode == "entry":
        side = _mapping(evaluation.get("buy"))
        score = _number(side.get("score"), nonnegative=True)
        threshold = _number(side.get("threshold"), nonnegative=True)
        if score is not None:
            result["buy_score"] = score
        if threshold is not None:
            result["buy_threshold"] = threshold
        if score is not None and threshold is not None:
            summary_parts.append(f"買いスコア {score:g}/{threshold:g}")
    else:
        for key, label in (("risk_exit", "リスク退出"), ("take_profit", "利益確定")):
            side = _mapping(evaluation.get(key))
            score = _number(side.get("score"), nonnegative=True)
            threshold = _number(side.get("threshold"), nonnegative=True)
            if score is not None:
                result[f"{key}_score"] = score
            if threshold is not None:
                result[f"{key}_threshold"] = threshold
            if score is not None and threshold is not None:
                summary_parts.append(f"{label} {score:g}/{threshold:g}")
    return result, summary_parts


def _build_rule_item(
    evaluation: Any,
    mode: str,
    ticker: str | None,
    warnings: list[str],
) -> dict | None:
    label = "新規買い" if mode == "entry" else "保有中"
    if evaluation is None:
        return None
    if not isinstance(evaluation, Mapping):
        warnings.append(f"{label}判定は辞書形式でないため表示しませんでした。")
        return None
    raw_verdict = evaluation.get("verdict")
    if isinstance(raw_verdict, Mapping):
        raw_verdict = raw_verdict.get("code")
    verdict = _one_line(raw_verdict, 40)
    verdict = verdict.upper() if verdict else None
    if verdict not in _VERDICT_LABELS:
        warnings.append(f"{label}判定コードを確認できないため表示しませんでした。")
        return None
    valid_verdicts = (
        {"BUY", "NEUTRAL", "WAIT"} if mode == "entry"
        else {"RISK_EXIT", "TAKE_PROFIT", "HOLD", "WAIT"}
    )
    if verdict not in valid_verdicts:
        warnings.append(
            f"{label}判定と判定コードの組み合わせが不正なため表示しませんでした。")
        return None
    score_data, score_parts = _score_data(evaluation, mode)
    risk_plan = _mapping(evaluation.get("risk_plan"))
    blocked = bool(evaluation.get("visual_blocked") is True)
    if verdict == "BUY" and risk_plan.get("valid") is False:
        blocked = True
    supplied_summary = _one_line(evaluation.get("summary"), 1_000)
    parts = ([supplied_summary] if supplied_summary else []) + score_parts
    occurred_at = (
        evaluation.get("evaluated_at") or evaluation.get("as_of")
        or evaluation.get("decision_time")
    )
    return normalize_item({
        "ticker": ticker,
        "category": "trade",
        "kind": f"{mode}_verdict",
        "importance": _importance_for_verdict(verdict, mode),
        "title_ja": f"{label}判定: {_VERDICT_LABELS[verdict]}",
        "summary_ja": " ／ ".join(parts) if parts else None,
        "occurred_at": occurred_at,
        "source": evaluation.get("source") or "既存ルール評価",
        "tags": [label, _VERDICT_LABELS[verdict]],
        "data": {
            "position_mode": mode,
            "verdict": verdict,
            "verdict_label_ja": _VERDICT_LABELS[verdict],
            "blocked": blocked,
            **score_data,
        },
        "identity_key": f"trade|{ticker or ''}|{mode}|{_timestamp(occurred_at) or ''}",
    })


def _build_price_item(snapshot: Any, ticker: str | None, warnings: list[str]) -> dict | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        warnings.append("株価スナップショットは辞書形式でないため表示しませんでした。")
        return None
    row = dict(snapshot)
    price = _number(
        row.get("price", row.get("last_price", row.get("current_price"))), positive=True)
    bid = _number(row.get("bid"), positive=True)
    ask = _number(row.get("ask"), positive=True)
    previous_close = _number(row.get("previous_close", row.get("prev_close")), positive=True)
    change_pct = _number(row.get(
        "change_pct", row.get("change_percent", row.get("change_rate"))))
    change_pct_calculated = False
    if change_pct is None and price is not None and previous_close is not None:
        change_pct = (price / previous_close - 1) * 100
        change_pct_calculated = True
    if price is None and bid is None and ask is None:
        warnings.append("株価スナップショットに表示可能な価格がありません。")
        return None
    summary_parts = []
    if price is not None:
        summary_parts.append(f"現在値 ${price:,.2f}")
    if change_pct is not None:
        summary_parts.append(f"前日比 {change_pct:+.2f}%")
    if bid is not None:
        summary_parts.append(f"買気配 ${bid:,.2f}")
    if ask is not None:
        summary_parts.append(f"売気配 ${ask:,.2f}")
    as_of = row.get("update_time", row.get("as_of"))
    importance = "medium" if change_pct is not None and abs(change_pct) >= 3 else "info"
    return normalize_item({
        "ticker": ticker,
        "category": "price",
        "kind": "snapshot",
        "importance": importance,
        "title_ja": "株価スナップショット",
        "summary_ja": " ／ ".join(summary_parts),
        "occurred_at": as_of,
        "source": row.get("source"),
        "tags": ["株価", "気配値" if bid is not None or ask is not None else "現在値"],
        "data": {
            "price": price, "previous_close": previous_close,
            "change_pct": change_pct,
            "change_pct_calculated_from_previous_close": change_pct_calculated,
            "bid": bid, "ask": ask,
            "market_state": _one_line(row.get("market_state"), 100),
        },
        "identity_key": (
            f"price|{ticker or ''}|{_timestamp(as_of) or ''}|"
            f"{_one_line(row.get('source'), 100) or ''}"
        ),
    })


def _level_kind(value: Any) -> tuple[str, str] | None:
    text = (_one_line(value, 50) or "").lower()
    if text == "support" or "サポート" in text or "支持" in text:
        return "support", "サポート"
    if text == "resistance" or "抵抗" in text or "レジスタンス" in text:
        return "resistance", "抵抗線"
    return None


def _build_level_items(
    levels: Any,
    ticker: str | None,
    current_price: float | None,
    warnings: list[str],
) -> list[dict]:
    if levels is None:
        return []
    if not _is_sequence(levels):
        warnings.append("支持・抵抗データは一覧形式でないため表示しませんでした。")
        return []
    items = []
    for index, original in enumerate(levels):
        if not isinstance(original, Mapping):
            warnings.append(f"支持・抵抗データ{index + 1}件目は形式不正のため除外しました。")
            continue
        row = dict(original)
        kind = _level_kind(row.get("type", row.get("kind")))
        price = _number(row.get("price"), positive=True)
        if kind is None or price is None:
            warnings.append(f"支持・抵抗データ{index + 1}件目は種類または価格不明のため除外しました。")
            continue
        code, label = kind
        zone_low = _number(row.get("zone_low"), positive=True)
        zone_high = _number(row.get("zone_high"), positive=True)
        if zone_low is not None and zone_high is not None and zone_low > zone_high:
            warnings.append(f"支持・抵抗データ{index + 1}件目の帯域が不正なため帯域だけ省略しました。")
            zone_low = zone_high = None
        strength = _number(row.get("strength"), nonnegative=True)
        if strength is not None and (not strength.is_integer() or not 1 <= strength <= 5):
            warnings.append(f"支持・抵抗データ{index + 1}件目の強度が範囲外のため省略しました。")
            strength = None
        distance_pct = (
            (price / current_price - 1) * 100
            if current_price is not None and current_price > 0 else None
        )
        parts = [f"中心価格 ${price:,.2f}"]
        if zone_low is not None and zone_high is not None:
            parts.append(f"帯 ${zone_low:,.2f}〜${zone_high:,.2f}")
        if strength is not None:
            stars = "★" * int(strength) + "☆" * (5 - int(strength))
            parts.append(f"強度 {stars}")
        if distance_pct is not None:
            parts.append(f"現在値から {distance_pct:+.2f}%")
        importance = "high" if strength is not None and strength >= 4 else (
            "medium" if strength is not None and strength >= 3 else "low")
        item = normalize_item({
            "ticker": ticker,
            "category": "level",
            "kind": code,
            "importance": importance,
            "title_ja": f"{label} ${price:,.2f}",
            "summary_ja": " ／ ".join(parts),
            "occurred_at": row.get("as_of", row.get("detected_at")),
            "source": row.get("source") or row.get("basis") or "既存の支持抵抗分析",
            "tags": [label],
            "data": {
                "type": code, "type_label_ja": label, "price": price,
                "zone_low": zone_low, "zone_high": zone_high,
                "strength": strength, "distance_from_current_pct": distance_pct,
                "basis": _one_line(row.get("basis"), 500),
            },
            "identity_key": (
                f"level|{ticker or ''}|{code}|{price:.8f}|"
                f"{zone_low if zone_low is not None else ''}|"
                f"{zone_high if zone_high is not None else ''}"
            ),
        })
        if item:
            items.append(item)
    return items


def _alert_parts(original: Any) -> tuple[dict, dict] | None:
    if _is_sequence(original) and len(original) == 2:
        if isinstance(original[0], Mapping) and isinstance(original[1], Mapping):
            return dict(original[0]), dict(original[1])
        return None
    if not isinstance(original, Mapping):
        return None
    row = dict(original)
    alert = _mapping(row.get("alert")) or row
    result = (
        _mapping(row.get("result")) or _mapping(row.get("check"))
        or {key: row.get(key) for key in ("triggered", "actual", "reason", "checked_at")
            if key in row}
    )
    return alert, result


def _alert_threshold(kind: str, value: float | None) -> str | None:
    if value is None:
        return None
    if kind in {"price_above", "price_below"}:
        return f"${value:,.2f}"
    if kind in {"change_above", "change_below", "near_support", "near_resistance"}:
        return f"{value:,.2f}%"
    return f"{value:g}"


def _build_alert_items(
    checked_alerts: Any,
    ticker: str | None,
    default_checked_at: Any,
    warnings: list[str],
) -> list[dict]:
    if checked_alerts is None:
        return []
    if not _is_sequence(checked_alerts):
        warnings.append("アラーム照合結果は一覧形式でないため表示しませんでした。")
        return []
    items = []
    for index, original in enumerate(checked_alerts):
        parts = _alert_parts(original)
        if parts is None:
            warnings.append(f"アラーム{index + 1}件目は形式不正のため除外しました。")
            continue
        alert, result = parts
        if alert.get("enabled") is False:
            continue
        alert_ticker = _ticker(alert.get("ticker"))
        if ticker and alert_ticker and alert_ticker != ticker:
            continue
        kind = _one_line(alert.get("kind"), 80)
        description = _one_line(alert.get("description"), 300)
        if not kind and description:
            # 簡易呼び出し側は判定済みの説明文だけを渡すことがある。この場合も
            # 種類や設定値を推測せず、説明文をそのまま表示する。
            kind = "checked_alert"
        if not kind:
            warnings.append(f"アラーム{index + 1}件目は種類不明のため除外しました。")
            continue
        label = _ALERT_LABELS.get(
            kind, _one_line(alert.get("label"), 150) or description)
        if not label:
            warnings.append(f"アラーム{index + 1}件目は未知の種類のため除外しました。")
            continue
        reason = _one_line(result.get("reason"), 1_000)
        triggered = result.get("triggered") if isinstance(result.get("triggered"), bool) else None
        if reason:
            state, state_label, importance = "unavailable", "判定不能", "medium"
            triggered_value = None
        elif triggered is True:
            state, state_label = "triggered", "成立"
            importance = "critical" if kind == "rule_risk_exit" else "high"
            triggered_value = True
        elif triggered is False:
            state, state_label, importance = "not_triggered", "未成立", "info"
            triggered_value = False
        else:
            warnings.append(f"アラーム{index + 1}件目は判定状態不明のため除外しました。")
            continue
        value = _number(alert.get("value"))
        threshold = _alert_threshold(kind, value)
        actual = _one_line(result.get("actual"), 300)
        note = _one_line(alert.get("note"), 500)
        summary_parts = []
        if threshold:
            summary_parts.append(f"設定値 {threshold}")
        if actual:
            summary_parts.append(f"確認値 {actual}")
        if reason:
            summary_parts.append(f"理由: {reason}")
        if note:
            summary_parts.append(f"メモ: {note}")
        source_id = _one_line(alert.get("id"), 200)
        fallback_identity = f"{kind}|{value}|{note or ''}"
        checked_at = result.get("checked_at") or default_checked_at
        item = normalize_item({
            "ticker": alert_ticker or ticker,
            "category": "alert",
            "kind": state,
            "importance": importance,
            "title_ja": f"アラーム{state_label}: {label}",
            "summary_ja": " ／ ".join(summary_parts) if summary_parts else None,
            "occurred_at": checked_at,
            "source": "現在画面でのアラーム照合",
            "tags": ["アラーム", state_label, label],
            "data": {
                "alert_id": source_id,
                "alert_kind": kind,
                "alert_label_ja": label,
                "state": state,
                "state_label_ja": state_label,
                "triggered": triggered_value,
                "value": value,
                "actual": actual,
                "reason": reason,
                "note": note,
            },
            "identity_key": (
                f"alert|{alert_ticker or ticker or ''}|"
                f"{source_id or fallback_identity}"
            ),
        })
        if item:
            items.append(item)
    return items


def _build_analyst_items(analyst: Any, ticker: str | None, warnings: list[str]) -> list[dict]:
    if analyst is None:
        return []
    if not isinstance(analyst, Mapping):
        warnings.append("アナリスト情報は辞書形式でないため表示しませんでした。")
        return []
    row = dict(analyst)
    as_of = row.get("as_of", row.get("updated_at"))
    items: list[dict] = []

    targets = _mapping(row.get("targets"))
    target_values = {
        key: _number(targets.get(key), positive=True)
        for key in ("mean", "median", "high", "low")
    }
    target_values = {key: value for key, value in target_values.items() if value is not None}
    if target_values:
        labels = {"mean": "平均", "median": "中央値", "high": "高値", "low": "安値"}
        summary = " ／ ".join(
            f"{labels[key]} ${target_values[key]:,.2f}"
            for key in ("mean", "median", "high", "low") if key in target_values
        )
        items.append(normalize_item({
            "ticker": ticker, "category": "analyst", "kind": "price_targets",
            "importance": "medium", "title_ja": "アナリスト目標株価",
            "summary_ja": summary, "occurred_at": as_of,
            "source": row.get("source") or "アナリスト集計",
            "tags": ["目標株価"], "data": target_values,
            "identity_key": f"analyst-targets|{ticker or ''}|{_timestamp(as_of) or ''}",
        }))

    ratings = _mapping(row.get("ratings"))
    rating_values: dict[str, int] = {}
    for key in _RATING_LABELS:
        value = _number(ratings.get(key), nonnegative=True)
        if value is not None and value.is_integer():
            rating_values[key] = int(value)
        elif ratings.get(key) is not None:
            warnings.append(f"アナリスト評価「{_RATING_LABELS[key]}」の件数が不正なため省略しました。")
    if rating_values:
        summary = " ／ ".join(
            f"{_RATING_LABELS[key]} {rating_values[key]}件"
            for key in _RATING_LABELS if key in rating_values
        )
        items.append(normalize_item({
            "ticker": ticker, "category": "analyst", "kind": "ratings",
            "importance": "low", "title_ja": "アナリスト評価の内訳",
            "summary_ja": summary, "occurred_at": as_of,
            "source": row.get("source") or "アナリスト集計",
            "tags": ["レーティング"], "data": rating_values,
            "identity_key": f"analyst-ratings|{ticker or ''}|{_timestamp(as_of) or ''}",
        }))

    earnings_date = _timestamp(row.get("earnings_date"))
    if row.get("earnings_date") is not None and earnings_date is None:
        warnings.append("次回決算日は解釈できない形式のため表示しませんでした。")
    if earnings_date:
        eps = _number(row.get("eps_estimate"))
        parts = [f"予定 {earnings_date}"]
        if eps is not None:
            parts.append(f"予想EPS {eps:g}")
        items.append(normalize_item({
            "ticker": ticker, "category": "analyst", "kind": "earnings",
            "importance": "high", "title_ja": "次回決算予定",
            "summary_ja": " ／ ".join(parts), "occurred_at": earnings_date,
            "source": row.get("source") or "決算カレンダー",
            "tags": ["決算", "予定"],
            "data": {"earnings_date": earnings_date, "eps_estimate": eps},
            "identity_key": f"earnings|{ticker or ''}|{earnings_date}",
        }))

    changes = row.get("changes")
    if changes is not None and not _is_sequence(changes):
        warnings.append("格付け変更は一覧形式でないため表示しませんでした。")
    for index, original in enumerate(changes if _is_sequence(changes) else ()):
        if not isinstance(original, Mapping):
            warnings.append(f"格付け変更{index + 1}件目は形式不正のため除外しました。")
            continue
        change = dict(original)
        firm = _one_line(change.get("firm"), 150)
        grade = _one_line(change.get("grade", change.get("to_grade")), 150)
        raw_action = _one_line(change.get("action"), 100)
        action = _ACTION_LABELS.get((raw_action or "").lower(), raw_action)
        if not any((firm, grade, action)):
            warnings.append(f"格付け変更{index + 1}件目は内容不明のため除外しました。")
            continue
        changed_at = _timestamp(change.get("date", change.get("occurred_at")))
        target = _number(change.get("target"), positive=True)
        parts = []
        if action:
            parts.append(action)
        if grade:
            parts.append(f"評価 {grade}")
        if target is not None:
            parts.append(f"目標 ${target:,.2f}")
        title = f"格付け情報: {firm}" if firm else "格付け情報"
        items.append(normalize_item({
            "ticker": ticker, "category": "analyst", "kind": "rating_change",
            "importance": "medium", "title_ja": title,
            "summary_ja": " ／ ".join(parts) if parts else None,
            "occurred_at": changed_at,
            "source": firm or row.get("source") or "アナリスト情報",
            "tags": ["格付け変更"] + ([action] if action else []),
            "data": {"firm": firm, "grade": grade, "action": action, "target": target},
            "identity_key": (
                f"rating-change|{ticker or ''}|{firm or ''}|{changed_at or ''}|"
                f"{grade or ''}|{action or ''}"
            ),
        }))
    return [item for item in items if item is not None]


def _event_importance(event: Mapping[str, Any]) -> str:
    stars = _number(event.get("impact_stars"), nonnegative=True)
    if stars is not None and 0 <= stars <= 5:
        if stars == 0:
            return "info"
        if stars >= 5:
            return "critical"
        if stars >= 4:
            return "high"
        if stars >= 2:
            return "medium"
        return "low"
    level = (_one_line(event.get("impact_level"), 30) or "").upper()
    return {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(level, "info")


def _build_event_items(report: Any, ticker: str | None, warnings: list[str]) -> list[dict]:
    if report is None:
        return []
    if _is_sequence(report):
        rows = report
        report_row = {}
    elif isinstance(report, Mapping):
        report_row = dict(report)
        rows = report_row.get("events")
        if rows is None:
            rows = []
        status = (_one_line(report_row.get("status"), 50) or "").lower()
        if status in {"partial", "unavailable"}:
            warnings.append("イベント情報は一部または全部を取得できていません。")
        raw_warnings = report_row.get("warnings")
        if isinstance(raw_warnings, str):
            raw_warnings = [raw_warnings]
        if _is_sequence(raw_warnings):
            for raw in raw_warnings:
                warning = _one_line(raw, 500)
                if warning:
                    warnings.append(f"イベント情報: {warning}")
    else:
        warnings.append("イベント情報は辞書または一覧形式でないため表示しませんでした。")
        return []
    if not _is_sequence(rows):
        warnings.append("イベント明細は一覧形式でないため表示しませんでした。")
        return []

    items = []
    for index, original in enumerate(rows):
        if not isinstance(original, Mapping):
            warnings.append(f"イベント{index + 1}件目は形式不正のため除外しました。")
            continue
        event = dict(original)
        name = _one_line(event.get("display_name_ja", event.get("name")), 300)
        if not name:
            warnings.append(f"イベント{index + 1}件目は名称不明のため除外しました。")
            continue
        event_date = _timestamp(event.get("event_date", event.get("occurred_at")))
        if event.get("event_date") is not None and event_date is None:
            warnings.append(f"イベント{index + 1}件目の日付を解釈できませんでした。")
        status = (_one_line(event.get("status"), 50) or "UNKNOWN").upper()
        status_label = _one_line(event.get("status_label_ja"), 100) or _EVENT_STATUS_LABELS.get(status)
        session_label = _one_line(
            event.get("session_label_ja", event.get("session_label", event.get("session"))), 150)
        direction = _one_line(
            event.get("directional_bias_label_ja", event.get("directional_bias")), 200)
        original_name = _one_line(event.get("original_name"), 500)
        raw_evidence = event.get("evidence")
        evidence = []
        if _is_sequence(raw_evidence):
            evidence = [value for value in (
                _one_line(raw, 500) for raw in raw_evidence[:2]
            ) if value]
        stars = _number(event.get("impact_stars"), nonnegative=True)
        if stars is not None and (not stars.is_integer() or not 0 <= stars <= 5):
            warnings.append(f"イベント{index + 1}件目の影響度が範囲外のため省略しました。")
            stars = None
        parts = []
        if status_label:
            parts.append(status_label)
        if event_date:
            parts.append(f"日時 {event_date}")
        if stars == 0:
            parts.append("影響度 判定材料なし（☆☆☆☆☆）")
        elif stars is not None:
            star_count = int(stars)
            parts.append("影響度 " + "★" * star_count + "☆" * (5 - star_count))
        if session_label:
            parts.append(f"対象 {session_label}")
        if direction:
            parts.append(f"過去の傾向 {direction}")
        if original_name and original_name != name:
            parts.append(f"原文見出し: {original_name}")
        if evidence:
            parts.append("根拠: " + " / ".join(evidence))
        raw_url = event.get("url")
        url = safe_url(raw_url)
        if raw_url and url is None:
            warnings.append(f"イベント{index + 1}件目の安全でないURLを除外しました。")
        event_id = _one_line(event.get("event_id", event.get("id")), 300)
        fallback_identity = f"{name}|{event_date or ''}"
        item = normalize_item({
            "ticker": ticker, "category": "event", "kind": _one_line(event.get("kind"), 80) or "event",
            "importance": _event_importance({**event, "impact_stars": stars}),
            "title_ja": f"イベント: {name}",
            "summary_ja": " ／ ".join(parts) if parts else None,
            "occurred_at": event_date,
            "source": event.get("source") or report_row.get("source"),
            "url": url,
            "tags": ["イベント"] + ([status_label] if status_label else []),
            "data": {
                "event_id": event_id, "name": name, "status": status,
                "original_name": original_name, "evidence": evidence,
                "status_label_ja": status_label, "event_date": event_date,
                "event_time_et": _one_line(event.get("event_time_et"), 100),
                "session_label_ja": session_label, "impact_stars": stars,
                "impact_label_ja": _one_line(event.get("impact_label_ja"), 100),
                "directional_bias_label_ja": direction,
                "confidence_label_ja": _one_line(event.get("confidence_label_ja"), 100),
            },
            "identity_key": (
                f"event|{ticker or ''}|"
                f"{event_id or url or fallback_identity}"
            ),
        })
        if item:
            items.append(item)
    return items


def _build_news_items(news: Any, ticker: str | None, warnings: list[str]) -> list[dict]:
    if news is None:
        return []
    if isinstance(news, Mapping):
        rows = news.get("items", news.get("news"))
    else:
        rows = news
    if not _is_sequence(rows):
        warnings.append("ニュースは一覧形式でないため表示しませんでした。")
        return []
    items = []
    for index, original in enumerate(rows):
        if not isinstance(original, Mapping):
            warnings.append(f"ニュース{index + 1}件目は形式不正のため除外しました。")
            continue
        row = dict(original)
        title = _one_line(row.get("title", row.get("headline")), 300)
        if not title:
            warnings.append(f"ニュース{index + 1}件目は見出し不明のため除外しました。")
            continue
        summary = _one_line(row.get("summary", row.get("description")), 2_000)
        published = _timestamp(
            row.get("pub_date", row.get("published_at", row.get("date"))))
        provider = _one_line(row.get("provider", row.get("source")), 200)
        raw_url = row.get("url", row.get("link"))
        url = safe_url(raw_url)
        if raw_url and url is None:
            warnings.append(f"ニュース{index + 1}件目の安全でないURLを除外しました。")
        source_id = _one_line(row.get("news_id", row.get("id")), 300)
        supplied_importance = _one_line(row.get("importance"), 30)
        importance = supplied_importance if supplied_importance in IMPORTANCE_LABELS else "info"
        fallback_identity = f"{title}|{published or ''}|{provider or ''}"
        item = normalize_item({
            "ticker": ticker, "category": "news", "kind": "news",
            "importance": importance, "title_ja": title,
            "summary_ja": summary, "occurred_at": published,
            "source": provider, "url": url,
            "tags": ["ニュース"] + list(row.get("tags") or [])
            if _is_sequence(row.get("tags")) else ["ニュース"],
            "data": {
                "news_id": source_id,
                "provider": provider,
                "published_at": published,
                "sentiment": _one_line(row.get("sentiment"), 100),
            },
            "identity_key": (
                f"news|{ticker or ''}|"
                f"{source_id or url or fallback_identity}"
            ),
        })
        if item:
            items.append(item)
    return items


def _as_codes(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    if isinstance(value, Iterable):
        return {str(item) for item in value}
    return {str(value)}


def filter_board_items(
    items: Any,
    *,
    categories: Any = None,
    importance: Any = None,
    minimum_importance: str | None = None,
    query: Any = None,
    triggered_only: bool = False,
) -> list[dict]:
    """カテゴリ・重要度・検索語で絞り込む。元の一覧は変更しない。"""
    if not _is_sequence(items):
        return []
    category_codes = _as_codes(categories)
    importance_codes = _as_codes(importance)
    minimum_rank = IMPORTANCE_RANKS.get(str(minimum_importance)) if minimum_importance else None
    query_text = (_one_line(query, 300) or "").casefold()
    filtered = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if category_codes is not None and item.get("category") not in category_codes:
            continue
        if importance_codes is not None and item.get("importance") not in importance_codes:
            continue
        if minimum_rank is not None and int(item.get("importance_rank") or 0) < minimum_rank:
            continue
        if triggered_only and _mapping(item.get("data")).get("triggered") is not True:
            continue
        if query_text:
            haystack = " ".join(str(value) for value in (
                item.get("ticker"), item.get("category_label_ja"), item.get("title_ja"),
                item.get("summary_ja"), item.get("source"), " ".join(item.get("tags") or []),
            ) if value is not None).casefold()
            if query_text not in haystack:
                continue
        filtered.append(deepcopy(dict(item)))
    return filtered


def sort_board_items(items: Any, *, sort_by: str = "importance") -> list[dict]:
    """重要度、新着、古い順、カテゴリ順で安定的に並べる。"""
    if not _is_sequence(items):
        return []
    rows = [deepcopy(dict(item)) for item in items if isinstance(item, Mapping)]

    def timestamp_key(item: Mapping[str, Any]) -> float | None:
        return _timestamp_number(item.get("occurred_at"))

    def newest_key(item: Mapping[str, Any]) -> tuple:
        stamp = timestamp_key(item)
        return (
            stamp is None,
            -(stamp or 0),
            -int(item.get("importance_rank") or 0),
            _CATEGORY_ORDER.get(str(item.get("category")), 999),
            str(item.get("id") or ""),
        )

    def oldest_key(item: Mapping[str, Any]) -> tuple:
        stamp = timestamp_key(item)
        return (
            stamp is None,
            stamp or 0,
            -int(item.get("importance_rank") or 0),
            _CATEGORY_ORDER.get(str(item.get("category")), 999),
            str(item.get("id") or ""),
        )

    def importance_key(item: Mapping[str, Any]) -> tuple:
        stamp = timestamp_key(item)
        return (
            -int(item.get("importance_rank") or 0),
            stamp is None,
            -(stamp or 0),
            _CATEGORY_ORDER.get(str(item.get("category")), 999),
            str(item.get("id") or ""),
        )

    def category_key(item: Mapping[str, Any]) -> tuple:
        stamp = timestamp_key(item)
        return (
            _CATEGORY_ORDER.get(str(item.get("category")), 999),
            -int(item.get("importance_rank") or 0),
            stamp is None,
            -(stamp or 0),
            str(item.get("id") or ""),
        )

    key = {
        "importance": importance_key,
        "priority": importance_key,
        "newest": newest_key,
        "oldest": oldest_key,
        "category": category_key,
    }.get(str(sort_by).lower(), importance_key)
    return sorted(rows, key=key)


def build_information_board(
    inputs: Mapping[str, Any] | None = None,
    *,
    sort_by: str = "importance",
) -> dict:
    """既取得データを読み取り専用の銘柄情報掲示板へ集約する。

    対応する主な入力キーは ``ticker``, ``snapshot``, ``entry_evaluation``,
    ``holding_evaluation``, ``levels``, ``checked_alerts``, ``analyst``,
    ``event_report``（``events`` も可）, ``news``。I/Oは行わない。
    """
    warnings: list[str] = []
    if inputs is None:
        payload: dict[str, Any] = {}
    elif isinstance(inputs, Mapping):
        payload = dict(inputs)
    else:
        payload = {}
        warnings.append("入力は辞書形式でないため、掲示板を空の状態で返しました。")

    raw_input_warnings = payload.get("warnings")
    if isinstance(raw_input_warnings, str):
        raw_input_warnings = [raw_input_warnings]
    if _is_sequence(raw_input_warnings):
        for raw_warning in raw_input_warnings:
            warning = _one_line(raw_warning, 1_000)
            if warning:
                warnings.append(warning)

    ticker = _ticker(payload.get("ticker"))
    if payload.get("ticker") is not None and ticker is None:
        warnings.append("ティッカーを解釈できなかったため、銘柄名なしで表示します。")

    snapshot = payload.get("snapshot")
    price_item = _build_price_item(snapshot, ticker, warnings)
    current_price = None
    if price_item:
        current_price = _number(price_item["data"].get("price"), positive=True)

    entry = payload.get("entry_evaluation")
    holding = payload.get("holding_evaluation")
    nested = payload.get("rule_evaluation")
    if isinstance(nested, Mapping):
        if "verdict" in nested:
            mode = _one_line(nested.get("position_mode"), 30)
            if mode == "holding" and holding is None:
                holding = nested
            elif mode != "holding" and entry is None:
                entry = nested
        else:
            entry = entry if entry is not None else nested.get("entry")
            holding = holding if holding is not None else nested.get("holding")

    raw_items: list[dict] = []
    if price_item:
        raw_items.append(price_item)
    for item in (
        _build_rule_item(entry, "entry", ticker, warnings),
        _build_rule_item(holding, "holding", ticker, warnings),
    ):
        if item:
            raw_items.append(item)
    raw_items.extend(_build_level_items(payload.get("levels"), ticker, current_price, warnings))
    raw_items.extend(_build_alert_items(
        payload.get("checked_alerts"), ticker, payload.get("alert_checked_at"), warnings))
    raw_items.extend(_build_analyst_items(payload.get("analyst"), ticker, warnings))

    event_report = payload.get("event_report")
    if event_report is None:
        event_report = payload.get("events", payload.get("event"))
    raw_items.extend(_build_event_items(event_report, ticker, warnings))
    raw_items.extend(_build_news_items(payload.get("news"), ticker, warnings))

    items, duplicate_count = _dedupe(raw_items)
    items = sort_board_items(items, sort_by=sort_by)
    warnings = list(dict.fromkeys(warnings))
    counts_by_category = {
        code: sum(item["category"] == code for item in items)
        for code in CATEGORY_LABELS
    }
    counts_by_importance = {
        code: sum(item["importance"] == code for item in items)
        for code in IMPORTANCE_LABELS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": ticker,
        "as_of": _timestamp(payload.get("as_of")),
        "items": items,
        "counts": {
            "total": len(items),
            "by_category": counts_by_category,
            "by_importance": counts_by_importance,
            "duplicates_removed": duplicate_count,
            "warnings": len(warnings),
        },
        "warnings": warnings,
        "metadata": {
            "read_only": True,
            "posting_enabled": False,
            "stores_data": False,
            "uses_network": False,
            "places_orders": False,
            "uses_moomoo_history": False,
            "uses_moomoo_history_quota": False,
            "moomoo_history_quota_used": False,
            "moomoo_history_requests": 0,
            "automatic_trade_score": False,
            "score_effect": 0,
            "generated_from_supplied_data": True,
            "disclaimer_ja": (
                "既に取得済みの情報を一覧化した参考表示です。投稿・保存・注文・"
                "外部取得は行わず、売買判断や利益を保証しません。"
            ),
        },
    }


# 短い呼び名も公開し、表示層から自然に利用できるようにする。
build_board = build_information_board
filter_items = filter_board_items
sort_items = sort_board_items


__all__ = [
    "SCHEMA_VERSION",
    "CATEGORY_LABELS",
    "IMPORTANCE_LABELS",
    "IMPORTANCE_RANKS",
    "ITEM_FIELDS",
    "safe_url",
    "normalize_item",
    "build_information_board",
    "build_board",
    "filter_board_items",
    "filter_items",
    "sort_board_items",
    "sort_items",
]
