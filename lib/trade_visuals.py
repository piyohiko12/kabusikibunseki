"""売買判定コードを、初心者にも読める一貫した表示へ変換する。

このモジュールは既存の判定結果を表示用に投影するだけで、判定、通信、保存、
注文は行わない。``売る候補`` は保有株の売却だけを意味し、新規の空売りを
表さない。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_VISUALS: dict[str, dict[str, Any]] = {
    "BUY": {
        "icon": "➕",
        "action_label_ja": "買い候補",
        "context_label_ja": "株を持っていない場合",
        "title_ja": "買いの条件がそろっています",
        "description_ja": "買い条件がそろっています。買う価格と損切りの目安を確認してください。",
        "severity": "success",
        "color": "green",
        "is_buy": True,
        "is_sell": False,
        "is_actionable": True,
    },
    "NEUTRAL": {
        "icon": "—",
        "action_label_ja": "今は買わない",
        "context_label_ja": "株を持っていない場合",
        "title_ja": "今回は見送ります",
        "description_ja": "買い条件が足りないため、今は新しく買いません。保有株を売る合図ではありません。",
        "severity": "info",
        "color": "gray",
        "is_buy": False,
        "is_sell": False,
        "is_actionable": False,
    },
    "WAIT": {
        "icon": "⏳",
        "action_label_ja": "判断を待つ",
        "context_label_ja": "株を持っていない場合",
        "title_ja": "まだ売買しません",
        "description_ja": "必要な情報がそろうまで、売買の判断を保留します。",
        "severity": "warning",
        "color": "orange",
        "is_buy": False,
        "is_sell": False,
        "is_actionable": False,
    },
    "RISK_EXIT": {
        "icon": "⚠️",
        "action_label_ja": "保有株を売る候補（損失を抑える）",
        "context_label_ja": "株を持っている場合",
        "title_ja": "損失を抑えるため売却を検討",
        "description_ja": "損失が広がるのを抑えるため、保有株の売却や縮小を検討します。新しい空売りではありません。",
        "severity": "error",
        "color": "red",
        "is_buy": False,
        "is_sell": True,
        "is_actionable": True,
    },
    "TAKE_PROFIT": {
        "icon": "✅",
        "action_label_ja": "保有株を売る候補（利益を確定）",
        "context_label_ja": "株を持っている場合",
        "title_ja": "利益を確定するため売却を検討",
        "description_ja": "利益を確定するため、保有株の一部または全部の売却を検討します。新しい空売りではありません。",
        "severity": "info",
        "color": "blue",
        "is_buy": False,
        "is_sell": True,
        "is_actionable": True,
    },
    "HOLD": {
        "icon": "●",
        "action_label_ja": "そのまま保有",
        "context_label_ja": "株を持っている場合",
        "title_ja": "保有を続けます",
        "description_ja": "売る条件は成立していないため、そのまま保有する判定です。追加で買う合図ではありません。",
        "severity": "info",
        "color": "gray",
        "is_buy": False,
        "is_sell": False,
        "is_actionable": False,
    },
}

_UNKNOWN = {
    "icon": "？",
    "action_label_ja": "判断できません",
    "context_label_ja": "判定不明",
    "title_ja": "判定できません",
    "description_ja": "判定を確認できないため、売買せずに待ちます。",
    "severity": "warning",
    "color": "orange",
    "is_buy": False,
    "is_sell": False,
    "is_actionable": False,
}

_BLOCKED_BUY = {
    "icon": "⏸️",
    "action_label_ja": "買い条件あり・今は待つ",
    "context_label_ja": "株を持っていない場合",
    "title_ja": "買いの条件は成立・実行は保留",
    "description_ja": "買いの条件はそろっていますが、確認が必要です。今は買わずに待ちます。",
    "severity": "warning",
    "color": "orange",
    "is_buy": True,
    "is_sell": False,
    "is_actionable": False,
}

_FIXED_KEYS = (
    "code",
    "icon",
    "action_label_ja",
    "context_label_ja",
    "title_ja",
    "description_ja",
    "severity",
    "color",
    "is_buy",
    "is_sell",
    "is_actionable",
    "short_sale",
    "available",
)

_VALID_CODES_BY_MODE = {
    "entry": {"BUY", "NEUTRAL", "WAIT"},
    "holding": {"RISK_EXIT", "TAKE_PROFIT", "HOLD", "WAIT"},
}


def verdict_visual(
    code: Any,
    *,
    blocked: bool = False,
    position_mode: str | None = None,
) -> dict:
    """既存の判定コードを、安全で視認しやすい表示情報へ変換する。

    未知のコードは推測せず、売買しない待機表示へフォールバックする。
    ``blocked`` は ``BUY`` にだけ適用され、買い候補であることを保ったまま
    実行可能フラグを無効にする。
    """
    normalized = str(code or "").strip().upper()
    mode = None if position_mode is None else str(position_mode).strip().lower()
    known = normalized in _VISUALS
    valid_mode = mode is None or mode in _VALID_CODES_BY_MODE
    compatible = (
        known
        and valid_mode
        and (mode is None or normalized in _VALID_CODES_BY_MODE[mode])
    )
    safe_code = normalized if compatible else "WAIT"

    if not compatible:
        source = _UNKNOWN
    elif safe_code == "BUY" and bool(blocked):
        source = _BLOCKED_BUY
    else:
        source = _VISUALS[safe_code]

    context_label = source["context_label_ja"]
    if compatible and safe_code == "WAIT" and mode == "holding":
        context_label = "株を持っている場合"
    result = {
        "code": safe_code,
        **source,
        "context_label_ja": context_label,
        "short_sale": False,
        "available": compatible,
    }
    # 公開契約のキーが定義追加の影響を受けないよう、固定キーだけを返す。
    return {key: result[key] for key in _FIXED_KEYS}


def evaluation_visual(evaluation: Any, *, position_mode: str) -> dict:
    """ルール評価全体から、価格計画を含めた安全な表示を返す。"""
    if not isinstance(evaluation, Mapping):
        return verdict_visual(None, position_mode=position_mode)
    verdict = evaluation.get("verdict")
    if isinstance(verdict, Mapping):
        verdict = verdict.get("code")
    risk_plan = evaluation.get("risk_plan")
    risk_plan = risk_plan if isinstance(risk_plan, Mapping) else {}
    blocked = bool(
        evaluation.get("visual_blocked") is True
        or evaluation.get("blocked") is True
        or (str(verdict or "").strip().upper() == "BUY"
            and risk_plan.get("valid") is False)
    )
    return verdict_visual(
        verdict, blocked=blocked, position_mode=position_mode)
