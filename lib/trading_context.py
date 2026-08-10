"""売買判定画面で共有する、取得済みデータの前処理。

このモジュールはAPIへ接続しない。各画面がすでに取得した日足・市場状態・
スナップショットを受け取り、形成途中足を除いた同一条件の判定コンテキストを作る。
"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from lib import indicators, levels, signal_context


SAFETY_DEFAULTS = {
    "min_history": 220,
    "max_stale_business_days": 5,
    "min_median_dollar_volume": 5_000_000.0,
    "max_spread_pct": 0.50,
    "earnings_blackout_days": 2,
}


def safety_config(rule: Mapping | None) -> dict:
    """保存ルールの安全設定を既定値へ重ね、独立した辞書で返す。"""
    saved = rule.get("safety") if isinstance(rule, Mapping) else None
    return {
        **SAFETY_DEFAULTS,
        **(dict(saved) if isinstance(saved, Mapping) else {}),
    }


def _canonical_history(history: pd.DataFrame, years: int | None) -> pd.DataFrame:
    """画面の表示期間に左右されない判定期間へそろえる。"""
    if history is None or history.empty:
        return pd.DataFrame()
    out = history.copy().sort_index()
    if years is None:
        return out
    years = max(int(years), 1)
    last = pd.Timestamp(out.index[-1])
    cutoff = last - pd.DateOffset(years=years)
    return out.loc[out.index >= cutoff]


def prepare_from_history(
    history: pd.DataFrame,
    *,
    source_meta: Mapping | None = None,
    market_meta: Mapping | None = None,
    snapshot: Mapping | None = None,
    earnings_date=None,
    canonical_years: int | None = 2,
) -> dict | None:
    """取得済み日足から、詳細画面と共通の判定コンテキストを作る。

    確定足だけを使い、指標の既定値と直近182日の支持抵抗を適用する。
    追加の履歴取得は行わないため、moomoo過去K線枠を消費しない。
    """
    source = dict(source_meta or {})
    market = dict(market_meta or {})
    live = dict(snapshot or {})
    code = source.get("code") or "US.UNKNOWN"
    canonical = _canonical_history(history, canonical_years)
    completed, bar_meta = signal_context.completed_daily_bars(
        canonical, code, market.get("market_state"))
    if completed.empty:
        return None

    frame = indicators.add_indicators(completed)
    display = indicators.slice_display(frame, 182)
    found_levels = levels.find_levels(display)

    price = live.get("price")
    previous_close = live.get("previous_close")
    if price is None:
        price = float(frame["Close"].iloc[-1])
    if previous_close is None:
        previous_close = (float(frame["Close"].iloc[-2])
                          if len(frame) > 1 else float(price))
    return {
        "df": frame,
        "levels": found_levels,
        "source_meta": source,
        "market_meta": market,
        "bar_meta": bar_meta,
        "snapshot": live,
        "earnings_date": earnings_date,
        "price": float(price),
        "previous_close": float(previous_close),
    }


def external_gates(context: Mapping, rule: Mapping | None) -> list[dict]:
    """共通の安全設定を使って、売買判定前の必須ゲートを作る。"""
    cfg = safety_config(rule)
    source = context.get("source_meta") or {}
    return signal_context.build_external_gates(
        context["df"],
        code=source.get("code") or "US.UNKNOWN",
        bar_meta=context.get("bar_meta"),
        snapshot=context.get("snapshot"),
        earnings_date=context.get("earnings_date"),
        min_history=int(cfg["min_history"]),
        max_stale_business_days=int(cfg["max_stale_business_days"]),
        min_median_dollar_volume=float(cfg["min_median_dollar_volume"]),
        max_spread_pct=float(cfg["max_spread_pct"]),
        earnings_blackout_days=int(cfg["earnings_blackout_days"]),
    )
