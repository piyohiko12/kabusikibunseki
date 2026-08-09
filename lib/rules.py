"""説明可能な売買判定ルールの評価（v2）。

このモジュールは判定材料を表示するだけで、注文処理は一切持たない。

v2 の原則:
- 必須条件（required）と加点条件を分ける
- 相関しやすい指標はグループ上限（group_caps）で多重加点を抑える
- エントリーと、保有中の利確・リスク退出を分ける
- 相場レジーム、外部データ品質ゲート、ATRベースのリスク計画を明示する
- 旧 buy/sell 形式は失わず v2 へ移行する
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "rules.json"

SCHEMA_VERSION = 2
OPS = {">=": "以上", "<=": "以下"}
GROUPS = ("trend", "momentum", "level", "participation", "risk")
GROUP_LABELS = {
    "trend": "トレンド",
    "momentum": "モメンタム",
    "level": "価格帯",
    "participation": "出来高・参加",
    "risk": "リスク",
}
REGIMES = ("UPTREND", "DOWNTREND", "RANGE", "HIGH_VOL")


# ------------------------------------------------------------- 数値ヘルパー

def _last(series) -> float | None:
    if series is None or len(series) == 0:
        return None
    value = series.iloc[-1] if hasattr(series, "iloc") else series[-1]
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def _lag(series, bars: int = 1) -> float | None:
    if series is None or len(series) <= bars:
        return None
    value = series.iloc[-(bars + 1)] if hasattr(series, "iloc") else series[-(bars + 1)]
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def _gap_pct(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or "Close" not in df.columns:
        return None
    average, price = _last(df[column]), _last(df["Close"])
    if average in (None, 0) or price is None:
        return None
    return (price / average - 1) * 100


def _return_pct(df: pd.DataFrame, days: int) -> float | None:
    if "Close" not in df.columns or len(df) <= days:
        return None
    base, price = _lag(df["Close"], days), _last(df["Close"])
    if base in (None, 0) or price is None:
        return None
    return (price / base - 1) * 100


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    required = {"High", "Low", "Close"}
    if df is None or not required.issubset(df.columns):
        return pd.Series(dtype=float)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _atr_value(df: pd.DataFrame, period: int = 14) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    return _last(_atr_series(df, period))


def _atr_pct(df: pd.DataFrame) -> float | None:
    atr, price = _atr_value(df), _last(df.get("Close"))
    if atr is None or price in (None, 0):
        return None
    return atr / price * 100


def _bb_percent_b(df: pd.DataFrame) -> float | None:
    if "BB_up" not in df.columns or "BB_low" not in df.columns:
        return None
    upper, lower, price = _last(df["BB_up"]), _last(df["BB_low"]), _last(df["Close"])
    if None in (upper, lower, price) or upper == lower:
        return None
    return (price - lower) / (upper - lower) * 100


def _drawdown_52w(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or not {"High", "Close"}.issubset(df.columns):
        return None
    index = pd.DatetimeIndex(df.index)
    year = df.loc[index >= index.max() - pd.Timedelta(days=365)]
    if year.empty:
        return None
    high, price = float(year["High"].max()), _last(df["Close"])
    if high == 0 or price is None:
        return None
    return (price / high - 1) * 100


def _volume_ratio(df: pd.DataFrame) -> float | None:
    column = next((name for name in ("VOL_MA20", "VOL_MA")
                   if name in df.columns), None)
    if column is None or "Volume" not in df.columns:
        return None
    average, volume = _last(df[column]), _last(df["Volume"])
    if average in (None, 0) or volume is None:
        return None
    return volume / average


def _annual_vol(df: pd.DataFrame) -> float | None:
    if df is None or "Close" not in df.columns or len(df) < 30:
        return None
    returns = df["Close"].pct_change().dropna().tail(60)
    if len(returns) < 2:
        return None
    value = float(returns.std() * np.sqrt(252) * 100)
    return value if np.isfinite(value) else None


def _series_change(df: pd.DataFrame, column: str, bars: int = 1) -> float | None:
    if column not in df.columns:
        return None
    current, previous = _last(df[column]), _lag(df[column], bars)
    if current is None or previous is None:
        return None
    return current - previous


def _sma200_slope(df: pd.DataFrame, bars: int = 20) -> float | None:
    if "SMA200" not in df.columns:
        return None
    current, previous = _last(df["SMA200"]), _lag(df["SMA200"], bars)
    if current is None or previous in (None, 0):
        return None
    return (current / previous - 1) * 100


def _sma50_above_200(df: pd.DataFrame) -> float | None:
    if "SMA50" not in df.columns or "SMA200" not in df.columns:
        return None
    short, long = _last(df["SMA50"]), _last(df["SMA200"])
    if short is None or long in (None, 0):
        return None
    return (short / long - 1) * 100


def _macd_improving_two(df: pd.DataFrame) -> float | None:
    if "MACD_hist" not in df.columns or len(df) < 3:
        return None
    values = [_lag(df["MACD_hist"], 2), _lag(df["MACD_hist"], 1),
              _last(df["MACD_hist"])]
    if any(value is None for value in values):
        return None
    return 1.0 if values[0] < values[1] < values[2] else 0.0


def _reversal_confirmation(df: pd.DataFrame, bullish: bool = True) -> float | None:
    """終値・RSI・MACDヒストグラムが同方向へ反転したら1を返す。"""
    if len(df) < 2 or "Close" not in df.columns:
        return None
    price_change = _series_change(df, "Close")
    rsi_change = _series_change(df, "RSI")
    macd_change = _series_change(df, "MACD_hist")
    if None in (price_change, rsi_change, macd_change):
        return None
    if bullish:
        return 1.0 if price_change > 0 and rsi_change > 0 and macd_change > 0 else 0.0
    return 1.0 if price_change < 0 and rsi_change < 0 and macd_change < 0 else 0.0


def _breakout_20d(df: pd.DataFrame) -> float | None:
    """終値の直前20本高値に対する位置（%）。0以上でブレイク。"""
    if len(df) < 21 or not {"High", "Close"}.issubset(df.columns):
        return None
    prior_high = float(df["High"].iloc[-21:-1].max())
    price = _last(df["Close"])
    if prior_high == 0 or price is None:
        return None
    return (price / prior_high - 1) * 100


def _return_atr(df: pd.DataFrame) -> float | None:
    atr = _atr_value(df)
    price, previous = _last(df.get("Close")), _lag(df.get("Close"), 1)
    if atr in (None, 0) or price is None or previous is None:
        return None
    return (price - previous) / atr


def _nearest_level(ctx: dict, kind: str) -> dict | None:
    price = _last(ctx.get("df", pd.DataFrame()).get("Close"))
    if price is None:
        return None
    candidates = []
    for level in ctx.get("levels") or []:
        if level.get("type") != kind:
            continue
        try:
            level_price = float(level["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if (kind == "抵抗線" and level_price > price) or (
                kind == "サポート" and level_price < price):
            candidates.append(level)
    if not candidates:
        return None
    return min(candidates, key=lambda level: abs(float(level["price"]) - price))


def _level_distance(ctx: dict, kind: str) -> float | None:
    level = _nearest_level(ctx, kind)
    price = _last(ctx.get("df", pd.DataFrame()).get("Close"))
    if level is None or price in (None, 0):
        return None
    level_price = float(level["price"])
    return ((level_price / price - 1) * 100 if kind == "抵抗線"
            else (1 - level_price / price) * 100)


def _level_strength(ctx: dict, kind: str) -> float | None:
    level = _nearest_level(ctx, kind)
    if level is None:
        return None
    value = level.get("strength", level.get("base_strength"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------- リスク計画・レジーム

def build_risk_plan(df: pd.DataFrame, levels: list | None = None,
                    support_buffer_atr: float = 0.5,
                    fallback_stop_atr: float = 1.5,
                    fallback_target_atr: float = 2.0,
                    min_level_strength: int = 2) -> dict:
    """現在値、ATR、支持・抵抗帯から機械的な entry/stop/target/R:R を返す。"""
    empty = {
        "entry": None, "stop": None, "target": None, "atr": None,
        "risk": None, "reward": None, "rr": None, "valid": False,
        "stop_source": "取得できず", "target_source": "取得できず",
    }
    if df is None or df.empty:
        return empty
    entry, atr = _last(df.get("Close")), _atr_value(df)
    if entry is None or atr in (None, 0):
        return {**empty, "entry": entry, "atr": atr}

    usable = []
    for level in levels or []:
        try:
            level_price = float(level["price"])
            strength = int(level.get("strength", level.get("base_strength", 1)))
        except (KeyError, TypeError, ValueError):
            continue
        if strength >= int(min_level_strength):
            usable.append((level, level_price))

    supports = [(level, price) for level, price in usable
                if level.get("type") == "サポート" and price < entry]
    resistance = [(level, price) for level, price in usable
                  if level.get("type") == "抵抗線" and price > entry]

    if supports:
        support, _ = max(supports, key=lambda item: item[1])
        anchor = float(support.get("zone_low", support["price"]))
        stop = anchor - max(float(support_buffer_atr), 0.0) * atr
        stop_source = f"支持帯（強度{int(support.get('strength', support.get('base_strength', 1)))})"
    else:
        stop = entry - max(float(fallback_stop_atr), 0.1) * atr
        stop_source = f"ATR×{max(float(fallback_stop_atr), 0.1):g}"
    if stop >= entry:
        stop = entry - max(float(fallback_stop_atr), 0.1) * atr
        stop_source = f"ATR×{max(float(fallback_stop_atr), 0.1):g}"

    if resistance:
        target_level, _ = min(resistance, key=lambda item: item[1])
        target = float(target_level.get("zone_low", target_level["price"]))
        if target <= entry:
            target = float(target_level["price"])
        target_source = (
            f"抵抗帯（強度{int(target_level.get('strength', target_level.get('base_strength', 1)))})")
    else:
        target = entry + max(float(fallback_target_atr), 0.1) * atr
        target_source = f"ATR×{max(float(fallback_target_atr), 0.1):g}"
    if target <= entry:
        target = entry + max(float(fallback_target_atr), 0.1) * atr
        target_source = f"ATR×{max(float(fallback_target_atr), 0.1):g}"

    risk, reward = entry - stop, target - entry
    rr = reward / risk if risk > 0 and reward > 0 else None
    return {
        "entry": entry, "stop": stop, "target": target, "atr": atr,
        "risk": risk, "reward": reward, "rr": rr,
        "valid": rr is not None and np.isfinite(rr),
        "stop_source": stop_source, "target_source": target_source,
    }


def _risk_reward(ctx: dict) -> float | None:
    plan = ctx.get("risk_plan")
    if not isinstance(plan, dict):
        plan = build_risk_plan(ctx.get("df"), ctx.get("levels"))
    value = plan.get("rr")
    return float(value) if value is not None and np.isfinite(float(value)) else None


def market_regime(df: pd.DataFrame) -> str:
    """日足の長期構造と変動率から4つの相場レジームを返す。"""
    if df is None or df.empty:
        return "RANGE"
    atr_pct, annual_vol = _atr_pct(df), _annual_vol(df)
    if ((atr_pct is not None and atr_pct >= 4.0)
            or (annual_vol is not None and annual_vol >= 55.0)):
        return "HIGH_VOL"

    price = _last(df.get("Close"))
    sma50 = _last(df.get("SMA50"))
    sma200 = _last(df.get("SMA200"))
    slope = _sma200_slope(df)
    if None in (price, sma50, sma200, slope):
        return "RANGE"
    if price > sma200 and sma50 > sma200 and slope > 0:
        return "UPTREND"
    if price < sma200 and sma50 < sma200 and slope < 0:
        return "DOWNTREND"
    return "RANGE"


# ------------------------------------------------------------- 指標の定義

METRICS: dict[str, dict] = {
    "rsi": {"label": "RSI(14)", "unit": "", "digits": 1, "group": "momentum",
            "help": "70超で買われすぎ、30未満で売られすぎとされる",
            "fn": lambda c: _last(c["df"].get("RSI"))},
    "rsi_change": {"label": "RSIの1本変化", "unit": "", "digits": 1,
                   "group": "momentum", "help": "プラスはRSIが前の足から改善",
                   "fn": lambda c: _series_change(c["df"], "RSI")},
    "sma20_gap": {"label": "SMA20との乖離率", "unit": "%", "digits": 2,
                  "group": "trend", "help": "プラスは20日移動平均より上",
                  "fn": lambda c: _gap_pct(c["df"], "SMA20")},
    "sma50_gap": {"label": "SMA50との乖離率", "unit": "%", "digits": 2,
                  "group": "trend", "help": "中期トレンドに対する位置",
                  "fn": lambda c: _gap_pct(c["df"], "SMA50")},
    "sma200_gap": {"label": "SMA200との乖離率", "unit": "%", "digits": 2,
                   "group": "trend", "help": "長期トレンドに対する位置",
                   "fn": lambda c: _gap_pct(c["df"], "SMA200")},
    "sma200_slope": {"label": "SMA200の20本傾き", "unit": "%", "digits": 2,
                     "group": "trend", "help": "20本前のSMA200からの変化率",
                     "fn": lambda c: _sma200_slope(c["df"])},
    "sma50_above_200": {"label": "SMA50−SMA200", "unit": "%", "digits": 2,
                        "group": "trend", "help": "0以上でSMA50がSMA200より上",
                        "fn": lambda c: _sma50_above_200(c["df"])},
    "macd_hist": {"label": "MACDヒストグラム", "unit": "", "digits": 3,
                  "group": "momentum", "help": "プラスは短期が長期を上回る",
                  "fn": lambda c: _last(c["df"].get("MACD_hist"))},
    "macd_hist_change": {"label": "MACDヒストグラムの1本変化", "unit": "",
                         "digits": 3, "group": "momentum",
                         "help": "プラスは前の足から改善",
                         "fn": lambda c: _series_change(c["df"], "MACD_hist")},
    "macd_hist_improving_2": {"label": "MACDヒストグラム2本連続改善", "unit": "",
                              "digits": 0, "group": "momentum",
                              "help": "成立=1、不成立=0",
                              "fn": lambda c: _macd_improving_two(c["df"])},
    "reversal_confirm": {"label": "上向き反転確認", "unit": "", "digits": 0,
                         "group": "momentum",
                         "help": "終値・RSI・MACDヒストグラムがそろって改善すると1",
                         "fn": lambda c: _reversal_confirmation(c["df"], True)},
    "bearish_reversal_confirm": {"label": "下向き反転確認", "unit": "", "digits": 0,
                                 "group": "momentum",
                                 "help": "終値・RSI・MACDヒストグラムがそろって悪化すると1",
                                 "fn": lambda c: _reversal_confirmation(c["df"], False)},
    "stoch_k": {"label": "ストキャスティクス%K", "unit": "", "digits": 1,
                "group": "momentum", "help": "80超で買われすぎ、20未満で売られすぎ",
                "fn": lambda c: _last(c["df"].get("STOCH_K"))},
    "bb_b": {"label": "ボリンジャー%B", "unit": "", "digits": 1,
             "group": "momentum", "help": "0=下限、100=上限",
             "fn": lambda c: _bb_percent_b(c["df"])},
    "vol_ratio": {"label": "出来高(20日平均比)", "unit": "倍", "digits": 2,
                  "group": "participation", "help": "1.0で平均並み",
                  "fn": lambda c: _volume_ratio(c["df"])},
    "ret_1d": {"label": "前日比", "unit": "%", "digits": 2, "group": "momentum",
               "help": "", "fn": lambda c: _return_pct(c["df"], 1)},
    "ret_5d": {"label": "5日リターン", "unit": "%", "digits": 2,
               "group": "momentum", "help": "",
               "fn": lambda c: _return_pct(c["df"], 5)},
    "ret_20d": {"label": "20日リターン", "unit": "%", "digits": 2,
                "group": "momentum", "help": "",
                "fn": lambda c: _return_pct(c["df"], 20)},
    "breakout_20d": {"label": "20日高値ブレイク", "unit": "%", "digits": 2,
                     "group": "momentum", "help": "0以上で直前20本の高値を突破",
                     "fn": lambda c: _breakout_20d(c["df"])},
    "ret_1d_atr": {"label": "1日変化÷ATR", "unit": "ATR", "digits": 2,
                   "group": "risk", "help": "1日変化をATRで標準化",
                   "fn": lambda c: _return_atr(c["df"])},
    "dd_52w": {"label": "52週高値からの下落率", "unit": "%", "digits": 2,
               "group": "risk", "help": "マイナスが大きいほど高値から離れている",
               "fn": lambda c: _drawdown_52w(c["df"])},
    "atr_pct": {"label": "ATR(価格比)", "unit": "%", "digits": 2,
                "group": "risk", "help": "1日の平均的な値動きの大きさ",
                "fn": lambda c: _atr_pct(c["df"])},
    "ann_vol": {"label": "年率ボラティリティ", "unit": "%", "digits": 1,
                "group": "risk", "help": "直近60日から算出",
                "fn": lambda c: _annual_vol(c["df"])},
    "dist_support": {"label": "直近サポートまでの距離", "unit": "%", "digits": 2,
                     "group": "level", "help": "小さいほどサポートに近い",
                     "fn": lambda c: _level_distance(c, "サポート")},
    "dist_resistance": {"label": "直近レジスタンスまでの距離", "unit": "%",
                        "digits": 2, "group": "level",
                        "help": "小さいほど上値が近い",
                        "fn": lambda c: _level_distance(c, "抵抗線")},
    "support_strength": {"label": "直近サポートの強度", "unit": "★", "digits": 0,
                         "group": "level", "help": "最寄り支持帯の強度(1〜5)",
                         "fn": lambda c: _level_strength(c, "サポート")},
    "resistance_strength": {"label": "直近レジスタンスの強度", "unit": "★",
                            "digits": 0, "group": "level",
                            "help": "最寄り抵抗帯の強度(1〜5)",
                            "fn": lambda c: _level_strength(c, "抵抗線")},
    "risk_reward": {"label": "ATRベースR:R", "unit": "倍", "digits": 2,
                    "group": "risk", "help": "参考目標までの利益幅÷参考損切り幅",
                    "fn": _risk_reward},
}


# ------------------------------------------------------------- v2 既定ルール

def _cond(metric: str, op: str, value: float, points: int,
          required: bool = False) -> dict:
    return {"metric": metric, "op": op, "value": float(value),
            "points": int(points), "required": bool(required)}


def _common_risk_exit() -> dict:
    return {
        "threshold": 35,
        "group_caps": {"trend": 20, "momentum": 20, "risk": 25},
        "conditions": [
            _cond("ret_1d_atr", "<=", -1.0, 25),
            _cond("sma50_above_200", "<=", 0.0, 20),
            _cond("macd_hist_change", "<=", 0.0, 15),
            _cond("bearish_reversal_confirm", ">=", 1.0, 15),
        ],
    }


def _legacy_default_rules() -> dict:
    """v1で配布した3既定ルール。完全一致する保存値の識別だけに使う。"""
    return {
        "押し目買い": {
            "buy": {"threshold": 60, "conditions": [
                {"metric": "sma200_gap", "op": ">=", "value": 0.0, "points": 25},
                {"metric": "rsi", "op": "<=", "value": 40.0, "points": 25},
                {"metric": "dist_support", "op": "<=", "value": 3.0, "points": 25},
                {"metric": "vol_ratio", "op": ">=", "value": 1.0, "points": 15},
                {"metric": "dd_52w", "op": "<=", "value": -5.0, "points": 10},
            ]},
            "sell": {"threshold": 55, "conditions": [
                {"metric": "rsi", "op": ">=", "value": 70.0, "points": 30},
                {"metric": "dist_resistance", "op": "<=", "value": 1.5, "points": 30},
                {"metric": "sma200_gap", "op": "<=", "value": 0.0, "points": 25},
                {"metric": "macd_hist", "op": "<=", "value": 0.0, "points": 15},
            ]},
        },
        "順張り(トレンドフォロー)": {
            "buy": {"threshold": 70, "conditions": [
                {"metric": "sma50_gap", "op": ">=", "value": 0.0, "points": 25},
                {"metric": "sma200_gap", "op": ">=", "value": 0.0, "points": 25},
                {"metric": "macd_hist", "op": ">=", "value": 0.0, "points": 20},
                {"metric": "rsi", "op": ">=", "value": 55.0, "points": 15},
                {"metric": "vol_ratio", "op": ">=", "value": 1.2, "points": 15},
            ]},
            "sell": {"threshold": 60, "conditions": [
                {"metric": "sma50_gap", "op": "<=", "value": 0.0, "points": 35},
                {"metric": "macd_hist", "op": "<=", "value": 0.0, "points": 30},
                {"metric": "rsi", "op": "<=", "value": 45.0, "points": 20},
                {"metric": "ret_5d", "op": "<=", "value": -3.0, "points": 15},
            ]},
        },
        "逆張り(売られすぎ)": {
            "buy": {"threshold": 65, "conditions": [
                {"metric": "rsi", "op": "<=", "value": 30.0, "points": 35},
                {"metric": "bb_b", "op": "<=", "value": 10.0, "points": 25},
                {"metric": "stoch_k", "op": "<=", "value": 20.0, "points": 20},
                {"metric": "dd_52w", "op": "<=", "value": -15.0, "points": 20},
            ]},
            "sell": {"threshold": 60, "conditions": [
                {"metric": "rsi", "op": ">=", "value": 65.0, "points": 40},
                {"metric": "bb_b", "op": ">=", "value": 90.0, "points": 35},
                {"metric": "stoch_k", "op": ">=", "value": 80.0, "points": 25},
            ]},
        },
    }


def default_rules() -> dict:
    """反転確認・独立グループ上限を備えた暫定v2ルールを返す。"""
    caps = {"trend": 30, "momentum": 30, "level": 25,
            "participation": 10, "risk": 15}
    rules = {
        "押し目買い": {
            "schema_version": SCHEMA_VERSION,
            "allowed_regimes": ["UPTREND", "RANGE"],
            "group_caps": caps,
            "risk": {"support_buffer_atr": 0.5, "fallback_stop_atr": 1.5,
                     "fallback_target_atr": 2.0, "min_level_strength": 3},
            "buy": {"threshold": 70, "conditions": [
                _cond("sma200_gap", ">=", 0.0, 10, True),
                _cond("sma200_slope", ">=", 0.0, 15),
                _cond("rsi", "<=", 45.0, 15),
                _cond("rsi_change", ">=", 0.0, 10),
                _cond("reversal_confirm", ">=", 1.0, 15, True),
                _cond("dist_support", "<=", 3.0, 10, True),
                _cond("support_strength", ">=", 3.0, 10, True),
                _cond("vol_ratio", ">=", 0.8, 10),
                _cond("risk_reward", ">=", 1.5, 10, True),
            ]},
            "take_profit": {"threshold": 55, "conditions": [
                _cond("dist_resistance", "<=", 2.0, 15, True),
                _cond("resistance_strength", ">=", 3.0, 15, True),
                _cond("rsi", ">=", 65.0, 20),
                _cond("bearish_reversal_confirm", ">=", 1.0, 20, True),
            ]},
            "risk_exit": _common_risk_exit(),
        },
        "順張り(トレンドフォロー)": {
            "schema_version": SCHEMA_VERSION,
            "allowed_regimes": ["UPTREND"],
            "group_caps": caps,
            "risk": {"support_buffer_atr": 0.5, "fallback_stop_atr": 1.5,
                     "fallback_target_atr": 2.5, "min_level_strength": 3},
            "buy": {"threshold": 75, "conditions": [
                _cond("sma50_above_200", ">=", 0.0, 20, True),
                _cond("sma200_slope", ">=", 0.0, 20, True),
                _cond("breakout_20d", ">=", 0.0, 20, True),
                _cond("macd_hist_improving_2", ">=", 1.0, 15),
                _cond("reversal_confirm", ">=", 1.0, 15, True),
                _cond("vol_ratio", ">=", 1.2, 10),
                _cond("risk_reward", ">=", 1.5, 15, True),
            ]},
            "take_profit": {"threshold": 50, "conditions": [
                _cond("dist_resistance", "<=", 1.5, 15),
                _cond("resistance_strength", ">=", 3.0, 15),
                _cond("rsi", ">=", 70.0, 20),
                _cond("bearish_reversal_confirm", ">=", 1.0, 20, True),
            ]},
            "risk_exit": _common_risk_exit(),
        },
        "逆張り(売られすぎ)": {
            "schema_version": SCHEMA_VERSION,
            "allowed_regimes": ["UPTREND", "RANGE"],
            "group_caps": caps,
            "risk": {"support_buffer_atr": 0.5, "fallback_stop_atr": 1.25,
                     "fallback_target_atr": 2.0, "min_level_strength": 3},
            "buy": {"threshold": 65, "conditions": [
                _cond("rsi", "<=", 35.0, 15, True),
                _cond("bb_b", "<=", 15.0, 15),
                _cond("stoch_k", "<=", 25.0, 15),
                _cond("reversal_confirm", ">=", 1.0, 20, True),
                _cond("dist_support", "<=", 2.5, 15, True),
                _cond("support_strength", ">=", 3.0, 15, True),
                _cond("vol_ratio", ">=", 0.8, 10),
                _cond("risk_reward", ">=", 1.8, 15, True),
            ]},
            "take_profit": {"threshold": 45, "conditions": [
                _cond("rsi", ">=", 60.0, 20),
                _cond("bb_b", ">=", 85.0, 15),
                _cond("dist_resistance", "<=", 2.0, 15),
                _cond("bearish_reversal_confirm", ">=", 1.0, 20, True),
            ]},
            "risk_exit": _common_risk_exit(),
        },
    }
    return {name: upgrade_rule(rule) for name, rule in rules.items()}


# ------------------------------------------------------------- スキーマ移行

def _normalise_condition(condition: dict) -> dict:
    out = copy.deepcopy(condition) if isinstance(condition, dict) else {}
    required = out.get("required", False)
    if isinstance(required, str):
        required = required.strip().lower() in {"true", "1", "yes", "y", "必須", "はい", "✓"}
    out["required"] = bool(required)
    return out


def _normalise_side(side: dict | None) -> dict:
    out = copy.deepcopy(side) if isinstance(side, dict) else {}
    conditions = out.get("conditions")
    out["conditions"] = [_normalise_condition(c) for c in conditions
                         if isinstance(c, dict)] if isinstance(conditions, list) else []
    try:
        out["threshold"] = int(out.get("threshold", 0))
    except (TypeError, ValueError):
        out["threshold"] = 0
    return out


def upgrade_rule(rule: dict | None) -> dict:
    """旧 buy/sell 形式を、条件を失わずv2へ移行したコピーを返す。"""
    source = copy.deepcopy(rule) if isinstance(rule, dict) else {}
    try:
        was_v2 = int(source.get("schema_version", 1) or 1) >= SCHEMA_VERSION
    except (TypeError, ValueError):
        was_v2 = False
    out = source
    out["schema_version"] = SCHEMA_VERSION
    out["buy"] = _normalise_side(out.get("buy"))

    # v1 の sell は利確条件として保存する。v2では take_profit を正とする。
    take_profit_source = (out.get("take_profit")
                          if was_v2 or "take_profit" in out else out.get("sell"))
    if take_profit_source is None:
        take_profit_source = out.get("sell")
    out["take_profit"] = _normalise_side(take_profit_source)
    out["risk_exit"] = _normalise_side(out.get("risk_exit"))
    out["sell"] = copy.deepcopy(out["take_profit"])  # 旧UI向けalias

    caps = out.get("group_caps")
    out["group_caps"] = copy.deepcopy(caps) if isinstance(caps, dict) else {}
    allowed = out.get("allowed_regimes")
    if not isinstance(allowed, (list, tuple, set)):
        allowed = list(REGIMES)
    out["allowed_regimes"] = [str(item).upper() for item in allowed]
    risk = out.get("risk")
    out["risk"] = copy.deepcopy(risk) if isinstance(risk, dict) else {}
    return out


def upgrade_rules(saved_rules: dict) -> dict:
    """名前付きルール群を安全に移行する。

    配布済みv1既定値と完全一致する3ルールだけ新v2既定へ置換する。
    同名でも利用者が1箇所でも変更したルールはカスタムとして条件を保持する。
    """
    if not isinstance(saved_rules, dict):
        return {}
    legacy, current = _legacy_default_rules(), default_rules()
    upgraded = {}
    for name, rule in saved_rules.items():
        if name in legacy and isinstance(rule, dict) and rule == legacy[name]:
            upgraded[name] = copy.deepcopy(current[name])
        else:
            upgraded[name] = upgrade_rule(rule)
    return upgraded


# ------------------------------------------------------------- 評価

def _fmt(metric_id: str, value) -> str:
    metric = METRICS.get(metric_id, {})
    if value is None:
        return "取得できず"
    return f"{value:,.{metric.get('digits', 2)}f}{metric.get('unit', '')}"


def _coerce_cap(value) -> float | None:
    try:
        cap = float(value)
    except (TypeError, ValueError):
        return None
    return max(cap, 0.0) if np.isfinite(cap) else None


def evaluate_side(ctx: dict, side: dict,
                  group_caps: dict | None = None) -> dict:
    """片側を評価し、必須条件とグループ上限を含む内訳を返す。"""
    side = _normalise_side(side)
    conditions = side["conditions"]
    threshold = int(side.get("threshold", 0))
    merged_caps = dict(group_caps or {})
    if isinstance(side.get("group_caps"), dict):
        merged_caps.update(side["group_caps"])

    checks = []
    group_raw_scores: dict[str, float] = {}
    group_totals: dict[str, float] = {}
    required_unavailable: list[str] = []
    required_failed: list[str] = []

    for condition in conditions:
        metric_id = condition.get("metric")
        metric = METRICS.get(metric_id)
        try:
            points = max(int(condition.get("points", 0)), 0)
        except (TypeError, ValueError):
            points = 0
        required = bool(condition.get("required", False))
        group = metric.get("group", "other") if metric else "other"
        group_totals[group] = group_totals.get(group, 0) + points

        actual = None
        if metric is not None:
            try:
                actual = metric["fn"](ctx)
            except Exception:
                actual = None
        op = condition.get("op")
        try:
            target = float(condition.get("value"))
        except (TypeError, ValueError):
            target = None

        if actual is None or target is None or op not in OPS:
            ok, status = False, "unavailable"
        else:
            ok = actual >= target if op == ">=" else actual <= target
            status = "passed" if ok else "failed"
        if ok:
            group_raw_scores[group] = group_raw_scores.get(group, 0) + points
        if required and status == "unavailable":
            required_unavailable.append(metric.get("label", str(metric_id))
                                        if metric else str(metric_id))
        elif required and not ok:
            required_failed.append(metric.get("label", str(metric_id))
                                   if metric else str(metric_id))

        label = metric.get("label", str(metric_id)) if metric else str(metric_id)
        unit = metric.get("unit", "") if metric else ""
        requirement = (f"{target:,.2f}{unit}{OPS.get(op, op or '—')}"
                       if target is not None else "設定不正")
        checks.append({
            "metric": metric_id, "label": label, "group": group,
            "points": points, "op": op, "threshold": target,
            "ok": ok, "status": status, "required": required,
            "actual": actual, "actual_text": _fmt(metric_id, actual),
            "requirement": requirement,
        })

    effective_caps = {}
    group_scores = {}
    capped_totals = {}
    for group, total in group_totals.items():
        configured = _coerce_cap(merged_caps.get(group))
        cap = total if configured is None else configured
        effective_caps[group] = cap
        group_scores[group] = min(group_raw_scores.get(group, 0), cap)
        capped_totals[group] = min(total, cap)

    score = sum(group_scores.values())
    total = sum(capped_totals.values())
    invalid_reasons = []
    if conditions and threshold <= 0:
        invalid_reasons.append("合格点は1点以上が必要です")
    if conditions and total <= 0:
        invalid_reasons.append("有効な配点がありません")
    if conditions and threshold > total:
        invalid_reasons.append("合格点がグループ上限適用後の満点を超えています")

    enabled = bool(conditions)
    valid = enabled and not invalid_reasons
    passed = (valid and not required_unavailable and not required_failed
              and score >= threshold)
    return {
        "score": score, "raw_score": sum(group_raw_scores.values()),
        "total": total, "raw_total": sum(group_totals.values()),
        "threshold": threshold, "passed": passed, "enabled": enabled,
        "valid": valid, "available": not required_unavailable,
        "required_passed": not required_unavailable and not required_failed,
        "required_unavailable": required_unavailable,
        "required_failed": required_failed,
        "invalid_reasons": invalid_reasons, "checks": checks,
        "group_scores": group_scores, "group_raw_scores": group_raw_scores,
        "group_totals": group_totals, "group_caps": effective_caps,
    }


def _normalise_external_gates(external_gates: list[dict] | None) -> list[dict]:
    gates = []
    for index, gate in enumerate(external_gates or []):
        if not isinstance(gate, dict):
            continue
        passed = gate.get("passed")
        passed = bool(passed) if isinstance(passed, (bool, np.bool_)) else None
        gates.append({
            **copy.deepcopy(gate),
            "key": str(gate.get("key", f"external_{index}")),
            "label": str(gate.get("label", gate.get("key", f"外部条件{index + 1}"))),
            "passed": passed,
            "required": bool(gate.get("required", False)),
            "reason": str(gate.get("reason", "")),
            "source": "external",
        })
    return gates


def _risk_kwargs(rule: dict) -> dict:
    allowed = {"support_buffer_atr", "fallback_stop_atr", "fallback_target_atr",
               "min_level_strength"}
    return {key: value for key, value in rule.get("risk", {}).items() if key in allowed}


def evaluate(df: pd.DataFrame, rule: dict, levels: list | None = None,
             position_mode: str = "entry",
             external_gates: list[dict] | None = None) -> dict:
    """v2ルール全体を評価する。

    entry: BUY / NEUTRAL / WAIT
    holding: RISK_EXIT / TAKE_PROFIT / HOLD / WAIT

    risk_exit は保有中の他の判定・外部ゲートより優先する。
    """
    if position_mode not in {"entry", "holding"}:
        raise ValueError("position_mode は 'entry' または 'holding' を指定してください")

    upgraded = upgrade_rule(rule)
    level_list = levels or []
    risk_plan = build_risk_plan(df, level_list, **_risk_kwargs(upgraded))
    ctx = {"df": df, "levels": level_list, "risk_plan": risk_plan}
    caps = upgraded.get("group_caps", {})
    buy = evaluate_side(ctx, upgraded.get("buy", {}), caps)
    take_profit = evaluate_side(ctx, upgraded.get("take_profit", {}), caps)
    risk_exit = evaluate_side(ctx, upgraded.get("risk_exit", {}), caps)

    regime = market_regime(df)
    allowed = upgraded.get("allowed_regimes")
    if not isinstance(allowed, list):
        allowed = list(REGIMES)
    regime_passed = regime in allowed if position_mode == "entry" else True
    regime_gate = {
        "key": "regime", "label": "相場レジーム", "passed": regime_passed,
        "required": position_mode == "entry", "actual": regime,
        "allowed": list(allowed), "source": "engine",
        "reason": ("許可されたレジームです" if regime_passed else
                   f"{regime} は許可レジーム外です"),
    }
    gates = [regime_gate, *_normalise_external_gates(external_gates)]
    blocking_gates = [gate for gate in gates
                      if gate.get("required") and gate.get("passed") is False]

    if position_mode == "entry":
        if blocking_gates:
            verdict, summary = "WAIT", "必須ゲートが成立するまで待機します"
        elif not buy["enabled"] or not buy["valid"] or not buy["available"]:
            verdict, summary = "WAIT", "買い判定に必要な設定またはデータが不足しています"
        elif buy["passed"]:
            verdict, summary = "BUY", "必須条件と買いスコアが成立しています"
        else:
            verdict, summary = "NEUTRAL", "買い条件は成立していません"
    else:
        # downside保護が未設定なら、旧カスタムルールを安全に保有判定へ使えない。
        if not risk_exit["enabled"]:
            verdict, summary = "WAIT", "リスク退出条件が未設定のため保有判定を待機します"
        elif not risk_exit["valid"] or not risk_exit["available"]:
            verdict, summary = "WAIT", "リスク退出判定に必要な設定またはデータが不足しています"
        elif risk_exit["passed"]:
            verdict, summary = "RISK_EXIT", "リスク退出条件が成立しています"
        elif take_profit["enabled"] and take_profit["passed"]:
            verdict, summary = "TAKE_PROFIT", "利確条件が成立しています"
        elif (take_profit["enabled"]
              and (not take_profit["valid"] or not take_profit["available"])):
            verdict, summary = "WAIT", "利確判定に必要なデータが不足しています"
        elif blocking_gates:
            verdict, summary = "WAIT", "必須ゲートが成立するまで保有判定を待機します"
        else:
            verdict, summary = "HOLD", "退出条件は成立していません"

    return {
        "verdict": verdict, "summary": summary, "position_mode": position_mode,
        "schema_version": SCHEMA_VERSION, "buy": buy,
        "take_profit": take_profit, "risk_exit": risk_exit,
        "sell": take_profit,  # 旧表示コード向けalias
        "risk_plan": risk_plan, "regime": regime, "gates": gates,
    }


# ------------------------------------------------------------- 永続化・編集

def load() -> dict:
    """保存ルールを読み、メモリ上でv2へ移行して返す。"""
    if DATA_FILE.exists():
        try:
            saved = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            rules = saved.get("rules")
            if isinstance(rules, dict) and rules:
                upgraded = upgrade_rules(rules)
                active = saved.get("active")
                if active not in upgraded:
                    active = next(iter(upgraded))
                return {"rules": upgraded, "active": active}
        except Exception:
            pass
    rules = default_rules()
    return {"rules": rules, "active": next(iter(rules))}


def save(rules: dict, active: str) -> None:
    normalised = upgrade_rules(rules)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps({"rules": normalised, "active": active},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")


def _required_from_row(row: pd.Series) -> bool:
    value = row.get("必須", row.get("required", row.get("Required", False)))
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "必須", "はい", "✓"}
    return bool(value)


def conditions_from_table(table: pd.DataFrame) -> list[dict]:
    """編集表を条件へ変換する。旧4列とv2の「必須」列の両方に対応。"""
    labels = {value["label"]: key for key, value in METRICS.items()}
    out = []
    for _, row in table.iterrows():
        metric = labels.get(row.get("指標"), row.get("metric"))
        value = row.get("しきい値", row.get("value"))
        if metric not in METRICS or value is None or pd.isna(value):
            continue
        op = row.get("条件", row.get("op"))
        if op not in OPS:
            op = ">="
        points = row.get("配点", row.get("points"))
        points = 0 if points is None or pd.isna(points) else int(points)
        out.append({"metric": metric, "op": op, "value": float(value),
                    "points": max(points, 0), "required": _required_from_row(row)})
    return out


def _side_caps(rule: dict, side: dict) -> dict:
    caps = dict(rule.get("group_caps") or {})
    if isinstance(side.get("group_caps"), dict):
        caps.update(side["group_caps"])
    return caps


def validate(rule: dict) -> list[str]:
    """設定不備、無効な上限、相関重複、矛盾条件を一覧で返す。"""
    upgraded = upgrade_rule(rule)
    problems = []

    allowed = upgraded.get("allowed_regimes", [])
    unknown_regimes = [value for value in allowed if value not in REGIMES]
    if not allowed:
        problems.append("許可する相場レジームが1つもありません")
    if unknown_regimes:
        problems.append(f"未知の相場レジームがあります: {', '.join(unknown_regimes)}")

    for group, value in upgraded.get("group_caps", {}).items():
        if group not in GROUPS:
            problems.append(f"未知のグループ上限です: {group}")
        cap = _coerce_cap(value)
        if cap is None or cap <= 0:
            problems.append(f"{GROUP_LABELS.get(group, group)}のグループ上限は1点以上が必要です")

    side_specs = (("buy", "買い"), ("take_profit", "利確"),
                  ("risk_exit", "リスク退出"))
    for side_key, side_name in side_specs:
        side = upgraded.get(side_key, {})
        conditions = side.get("conditions", [])
        if not conditions:
            problems.append(f"{side_name}条件が1つもありません")
            continue

        try:
            threshold = int(side.get("threshold", 0))
        except (TypeError, ValueError):
            threshold = 0
        if threshold <= 0:
            problems.append(f"{side_name}の合格点は1点以上が必要です")

        group_totals: dict[str, int] = {}
        seen_pairs = set()
        by_metric: dict[str, list[dict]] = {}
        group_counts: dict[str, int] = {}
        for condition in conditions:
            metric_id = condition.get("metric")
            metric = METRICS.get(metric_id)
            if metric is None:
                problems.append(f"{side_name}条件に未知の指標があります: {metric_id}")
                continue
            group = metric["group"]
            try:
                points = int(condition.get("points", 0))
            except (TypeError, ValueError):
                points = 0
            if points < 0:
                problems.append(f"{side_name}条件「{metric['label']}」の配点が負です")
            group_totals[group] = group_totals.get(group, 0) + max(points, 0)
            group_counts[group] = group_counts.get(group, 0) + 1

            op = condition.get("op")
            if op not in OPS:
                problems.append(f"{side_name}条件「{metric['label']}」の比較演算子が不正です")
            pair = (metric_id, op)
            if pair in seen_pairs:
                problems.append(
                    f"{side_name}条件に「{metric['label']} {OPS.get(op, '')}」が重複しています")
            seen_pairs.add(pair)
            by_metric.setdefault(metric_id, []).append(condition)

        caps = _side_caps(upgraded, side)
        achievable = 0.0
        for group, total in group_totals.items():
            cap = _coerce_cap(caps.get(group))
            achievable += min(total, cap) if cap is not None else total
            if (group_counts.get(group, 0) >= 3
                    and (cap is None or cap <= 0 or cap >= total)):
                problems.append(
                    f"{side_name}条件の{GROUP_LABELS.get(group, group)}指標が相関重複しています。"
                    "グループ上限を配点合計より小さくしてください")
        if threshold > achievable:
            problems.append(
                f"{side_name}の合格点({threshold})がグループ上限適用後の満点"
                f"({achievable:g})を超えており、成立しません")

        for metric_id, same_metric in by_metric.items():
            lowers, uppers = [], []
            for condition in same_metric:
                try:
                    value = float(condition.get("value"))
                except (TypeError, ValueError):
                    continue
                (lowers if condition.get("op") == ">=" else uppers).append(value)
            if lowers and uppers and max(lowers) > min(uppers):
                label = METRICS[metric_id]["label"]
                problems.append(
                    f"{side_name}条件の「{label}」に同時成立できない上下限があります")

    return problems
