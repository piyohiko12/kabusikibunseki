"""Point-in-time walk-forward diagnostics for custom trading rules.

This module deliberately accepts an OHLCV ``DataFrame`` and never fetches data.
For every signal date it rebuilds indicators and support/resistance levels from
the rows that were available at that close.  Signals are therefore evaluated at
the close and filled at the next session's open.

The result is a diagnostic, not a parameter optimiser and not evidence that a
rule will remain profitable.  ``train_bars`` are used only as warm-up history;
all reported trades originate in a fold's out-of-sample test window.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd

from lib import indicators, levels, rules


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

TRADE_COLUMNS = [
    "fold", "entry_signal_time", "entry_time", "exit_signal_time", "exit_time",
    "entry_price", "entry_fill_price", "exit_price", "exit_fill_price",
    "stop_price", "target_price", "atr_at_entry", "holding_days", "exit_reason",
    "gross_pnl", "net_pnl", "gross_return", "net_return", "transaction_cost",
    "one_way_bps", "slippage_bps",
]

FOLD_COLUMNS = [
    "fold", "train_start", "train_end", "test_start", "test_end",
    "train_bars", "test_bars", "trades", "wins", "win_rate",
    "avg_net_pnl", "avg_net_return", "profit_factor", "max_drawdown",
    "expectancy", "status",
]

DEFAULT_ACCEPTANCE = {
    # These are screening gates, not a claim of statistical significance.
    "min_trades": 30,
    "min_folds": 3,
    "min_profit_factor": 1.10,
    "min_expectancy": 0.0,
    "max_drawdown": 0.25,
    "min_positive_fold_ratio": 0.50,
}


def build_folds(length: int, train_bars: int, test_bars: int,
                step_bars: int) -> list[dict[str, int]]:
    """Return rolling train/test positions, with end positions exclusive.

    A partial final test fold is retained when it contains at least two bars
    (one signal close and one next-session open).  Training rows are warm-up
    only and never generate reported trades.
    """
    values = {
        "length": length, "train_bars": train_bars,
        "test_bars": test_bars, "step_bars": step_bars,
    }
    for name, value in values.items():
        if not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")

    folds: list[dict[str, int]] = []
    test_start = int(train_bars)
    fold_no = 1
    while test_start < int(length):
        test_end = min(test_start + int(test_bars), int(length))
        if test_end - test_start < 2:
            break
        folds.append({
            "fold": fold_no,
            "train_start": max(0, test_start - int(train_bars)),
            "train_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
        })
        fold_no += 1
        test_start += int(step_bars)
    return folds


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(columns=TRADE_COLUMNS), pd.DataFrame(columns=FOLD_COLUMNS)


def _insufficient_summary(reasons: list[str], *, rows: int = 0,
                          criteria: dict | None = None) -> dict:
    return {
        "status": "INSUFFICIENT_DATA",
        "passed": False,
        "reasons": reasons,
        "warnings": [],
        "bars": int(rows),
        "fold_count": 0,
        "trade_count": 0,
        "wins": 0,
        "win_rate": np.nan,
        "avg_net_pnl": np.nan,
        "avg_net_return": np.nan,
        "profit_factor": np.nan,
        "max_drawdown": np.nan,
        "expectancy": np.nan,
        "positive_fold_ratio": np.nan,
        "evaluation_error_count": 0,
        "skipped_entries": 0,
        "acceptance_criteria": criteria or dict(DEFAULT_ACCEPTANCE),
        "note": "診断基準の通過は将来の収益性を保証しません。",
    }


def _prepare_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    problems: list[str] = []
    if not isinstance(df, pd.DataFrame):
        return None, ["OHLCVデータがDataFrameではありません"]
    if df.empty:
        return None, ["OHLCVデータが空です"]
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return None, ["必要な列がありません: " + ", ".join(missing)]
    if not isinstance(df.index, pd.DatetimeIndex):
        return None, ["時系列のindexはDatetimeIndexである必要があります"]
    if df.index.has_duplicates:
        problems.append("時系列indexに重複があります")
    if not df.index.is_monotonic_increasing:
        problems.append("時系列indexが昇順ではありません")
    if problems:
        return None, problems

    out = df.copy()
    for column in REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    values = out.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return None, ["OHLCVに欠損または非有限値があります"]
    if (out[["Open", "High", "Low", "Close"]] <= 0).any().any():
        return None, ["OHLC価格は正の値である必要があります"]
    if (out["Volume"] < 0).any():
        return None, ["出来高に負の値があります"]
    if (out["High"] < out[["Open", "Close", "Low"]].max(axis=1)).any():
        return None, ["Highが同日のOpen/Close/Lowを下回っています"]
    if (out["Low"] > out[["Open", "Close", "High"]].min(axis=1)).any():
        return None, ["Lowが同日のOpen/Close/Highを上回っています"]
    return out, []


def _atr_as_of(df: pd.DataFrame, period: int) -> float | None:
    if len(df) < period + 1:
        return None
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    value = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().iloc[-1]
    if not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _history_and_levels(df: pd.DataFrame, start_pos: int,
                        end_pos: int) -> tuple[pd.DataFrame, list[dict]]:
    """Build causal indicators and a 182-calendar-day level window."""
    raw_history = df.iloc[start_pos:end_pos + 1]
    history = indicators.add_indicators(raw_history)
    cutoff = history.index[-1] - pd.Timedelta(days=182)
    level_history = history.loc[history.index >= cutoff]
    return history, levels.find_levels(level_history)


def _supports_extended_evaluate(fn) -> bool:
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return (any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters)
            or any(p.name == "position_mode" for p in parameters))


def _evaluate(history: pd.DataFrame, rule: dict, current_levels: list[dict],
              position_mode: str) -> tuple[str, dict]:
    """Call the new rule API when available and retain old-API compatibility."""
    fn = rules.evaluate
    if _supports_extended_evaluate(fn):
        result = fn(history, rule, current_levels,
                    position_mode=position_mode, external_gates=None)
    else:
        result = fn(history, rule, current_levels)
    if not isinstance(result, dict):
        raise TypeError("rules.evaluate must return a dict")

    raw = str(result.get("verdict") or result.get("decision") or "WAIT").upper()
    if position_mode == "holding":
        if raw == "SELL":
            verdict = "RISK_EXIT"
        elif raw == "BUY":
            verdict = "HOLD"
        elif raw in {"RISK_EXIT", "TAKE_PROFIT", "HOLD", "NEUTRAL", "WAIT"}:
            verdict = raw
        else:
            verdict = "WAIT"
    else:
        verdict = "BUY" if raw == "BUY" else (raw if raw in {"NEUTRAL", "WAIT"} else "WAIT")
    return verdict, result


def _number(plan: dict, *keys: str) -> float | None:
    for key in keys:
        value = plan.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _entry_risk(plan: dict, entry_price: float, atr: float | None,
                stop_mult: float, target_mult: float) -> tuple[float, float] | None:
    explicit_stop = _number(plan, "stop", "stop_price", "hard_stop_price")
    explicit_target = _number(plan, "target", "target_price", "take_profit_price")
    plan_stop_mult = _number(plan, "atr_stop_mult", "atr_stop_multiple")
    plan_target_mult = _number(plan, "atr_target_mult", "atr_target_multiple")

    if explicit_stop is not None and 0 < explicit_stop < entry_price:
        stop = explicit_stop
    elif atr is not None:
        stop = entry_price - atr * (plan_stop_mult or stop_mult)
    else:
        return None

    if explicit_target is not None and explicit_target > entry_price:
        target = explicit_target
    elif atr is not None:
        target = entry_price + atr * (plan_target_mult or target_mult)
    else:
        return None
    if stop <= 0 or stop >= entry_price or target <= entry_price:
        return None
    return float(stop), float(target)


def _close_trade(position: dict, exit_time, exit_reference: float,
                 exit_signal_time, exit_reason: str, holding_days: int,
                 one_way_bps: float, slippage_bps: float) -> dict:
    cost_rate = one_way_bps / 10_000
    slip_rate = slippage_bps / 10_000
    entry_reference = float(position["entry_price"])
    entry_fill = entry_reference * (1 + slip_rate)
    exit_fill = float(exit_reference) * (1 - slip_rate)
    entry_cash = entry_fill * (1 + cost_rate)
    exit_cash = exit_fill * (1 - cost_rate)
    gross_pnl = float(exit_reference) - entry_reference
    net_pnl = exit_cash - entry_cash
    gross_return = float(exit_reference) / entry_reference - 1
    net_return = exit_cash / entry_cash - 1
    return {
        "fold": position["fold"],
        "entry_signal_time": position["entry_signal_time"],
        "entry_time": position["entry_time"],
        "exit_signal_time": exit_signal_time,
        "exit_time": exit_time,
        "entry_price": entry_reference,
        "entry_fill_price": entry_fill,
        "exit_price": float(exit_reference),
        "exit_fill_price": exit_fill,
        "stop_price": position["stop_price"],
        "target_price": position["target_price"],
        "atr_at_entry": position["atr_at_entry"],
        "holding_days": int(max(1, holding_days)),
        "exit_reason": exit_reason,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "gross_return": gross_return,
        "net_return": net_return,
        "transaction_cost": gross_pnl - net_pnl,
        "one_way_bps": float(one_way_bps),
        "slippage_bps": float(slippage_bps),
    }


def _profit_factor(net_pnl: pd.Series) -> float:
    if net_pnl.empty:
        return np.nan
    gains = float(net_pnl[net_pnl > 0].sum())
    losses = float(-net_pnl[net_pnl < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _max_drawdown(net_returns: pd.Series) -> float:
    if net_returns.empty:
        return np.nan
    equity = np.cumprod(1 + net_returns.to_numpy(dtype=float))
    equity = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1 - equity / peaks
    return float(np.max(drawdowns))


def _mark_to_market_drawdown(df: pd.DataFrame, trades: pd.DataFrame,
                             start_pos: int, end_pos: int) -> float:
    """OOS期間を日次終値で時価評価した最大ドローダウン。

    ポジション保有中も、その日の終値で手仕舞うと仮定した売却コストと
    スリッページを反映する。決済日の値は実際の仮想約定損益を使う。
    """
    if trades.empty or start_pos >= end_pos:
        return np.nan
    window = df.iloc[start_pos:end_pos]
    equity = pd.Series(1.0, index=window.index, dtype=float)
    current_equity = 1.0
    cursor = 0

    for _, trade in trades.sort_values("entry_time").iterrows():
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]
        try:
            entry_loc = int(window.index.get_loc(entry_time))
            exit_loc = int(window.index.get_loc(exit_time))
        except (KeyError, TypeError, ValueError):
            continue
        if entry_loc < cursor or exit_loc < entry_loc:
            continue

        equity.iloc[cursor:entry_loc] = current_equity
        base_equity = current_equity
        cost_rate = float(trade["one_way_bps"]) / 10_000
        slip_rate = float(trade["slippage_bps"]) / 10_000
        entry_cash = float(trade["entry_fill_price"]) * (1 + cost_rate)
        if exit_loc > entry_loc and entry_cash > 0:
            closes = pd.to_numeric(
                window["Close"].iloc[entry_loc:exit_loc], errors="coerce")
            liquidation = closes * (1 - slip_rate) * (1 - cost_rate)
            equity.iloc[entry_loc:exit_loc] = (
                base_equity * liquidation.to_numpy(dtype=float) / entry_cash)

        current_equity = base_equity * (1 + float(trade["net_return"]))
        equity.iloc[exit_loc] = current_equity
        cursor = exit_loc + 1

    equity.iloc[cursor:] = current_equity
    values = equity.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    values = np.concatenate(([1.0], values))
    peaks = np.maximum.accumulate(values)
    return float(np.max(1 - values / peaks))


def _stats(trades: pd.DataFrame) -> dict:
    count = len(trades)
    if count == 0:
        return {
            "trades": 0, "wins": 0, "win_rate": np.nan,
            "avg_net_pnl": np.nan, "avg_net_return": np.nan,
            "profit_factor": np.nan, "max_drawdown": np.nan,
            "expectancy": np.nan,
        }
    wins = int((trades["net_pnl"] > 0).sum())
    expectation = float(trades["net_return"].mean())
    return {
        "trades": count,
        "wins": wins,
        "win_rate": wins / count,
        "avg_net_pnl": float(trades["net_pnl"].mean()),
        "avg_net_return": expectation,
        "profit_factor": _profit_factor(trades["net_return"]),
        "max_drawdown": _max_drawdown(trades["net_return"]),
        "expectancy": expectation,
    }


def _qualification(stats: dict, fold_df: pd.DataFrame, criteria: dict,
                   evaluation_errors: list[str]) -> tuple[str, bool, list[str]]:
    insufficient = []
    if len(fold_df) < int(criteria["min_folds"]):
        insufficient.append(
            f"テストfoldが{len(fold_df)}件で、最低{int(criteria['min_folds'])}件に不足")
    if stats["trades"] < int(criteria["min_trades"]):
        insufficient.append(
            f"OOS取引が{stats['trades']}件で、最低{int(criteria['min_trades'])}件に不足")
    if insufficient:
        return "INSUFFICIENT_DATA", False, insufficient

    failures = []
    if evaluation_errors:
        failures.append(f"ルール評価エラーが{len(evaluation_errors)}件あります")
    if not np.isfinite(stats["expectancy"]) or stats["expectancy"] <= float(criteria["min_expectancy"]):
        failures.append(
            f"期待値が基準({float(criteria['min_expectancy']):.4f}超)を満たしません")
    pf = stats["profit_factor"]
    if np.isnan(pf) or pf < float(criteria["min_profit_factor"]):
        failures.append(
            f"Profit Factorが基準({float(criteria['min_profit_factor']):.2f}以上)を満たしません")
    if (not np.isfinite(stats["max_drawdown"])
            or stats["max_drawdown"] > float(criteria["max_drawdown"])):
        failures.append(
            f"最大ドローダウンが上限({float(criteria['max_drawdown']):.1%})を超えています")

    positive_ratio = float((fold_df["expectancy"] > 0).mean()) if len(fold_df) else np.nan
    if (not np.isfinite(positive_ratio)
            or positive_ratio < float(criteria["min_positive_fold_ratio"])):
        failures.append(
            "期待値がプラスのfold比率が基準"
            f"({float(criteria['min_positive_fold_ratio']):.0%}以上)を満たしません")
    if failures:
        return "DOES_NOT_MEET_DIAGNOSTIC_CRITERIA", False, failures
    return (
        "MEETS_DIAGNOSTIC_CRITERIA",
        True,
        ["事前設定した診断基準を満たしましたが、将来の収益性を保証する結果ではありません"],
    )


def run_walk_forward(
    df: pd.DataFrame,
    rule: dict,
    *,
    train_bars: int = 252,
    test_bars: int = 63,
    step_bars: int = 63,
    one_way_bps: float = 10,
    slippage_bps: float = 5,
    max_holding_days: int = 20,
    atr_period: int = 14,
    atr_stop_mult: float = 2,
    atr_target_mult: float = 3,
    acceptance: dict | None = None,
) -> dict[str, Any]:
    """Run deterministic rolling OOS diagnostics on supplied OHLCV data.

    Stops and targets are checked against each daily bar before the close signal.
    If both are touched in one bar, the stop is chosen conservatively.  Rule and
    maximum-holding exits are decided at the close and filled at the next open.
    A still-open position is marked out at the final test close solely to keep
    each fold's diagnostic self-contained.
    """
    criteria = {**DEFAULT_ACCEPTANCE, **(acceptance or {})}
    prepared, problems = _prepare_ohlcv(df)
    if problems or prepared is None:
        trades, folds = _empty_frames()
        return {
            "trades": trades,
            "folds": folds,
            "summary": _insufficient_summary(
                problems or ["OHLCVデータを利用できません"],
                rows=len(df) if isinstance(df, pd.DataFrame) else 0,
                criteria=criteria,
            ),
        }

    numeric_parameters = {
        "one_way_bps": one_way_bps,
        "slippage_bps": slippage_bps,
        "atr_stop_mult": atr_stop_mult,
        "atr_target_mult": atr_target_mult,
    }
    for name, value in numeric_parameters.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if one_way_bps >= 10_000 or slippage_bps >= 10_000:
        raise ValueError("cost and slippage must each be below 10000 bps")
    if max_holding_days <= 0 or atr_period <= 0:
        raise ValueError("max_holding_days and atr_period must be positive")
    if atr_stop_mult <= 0 or atr_target_mult <= 0:
        raise ValueError("ATR multiples must be positive")

    fold_specs = build_folds(len(prepared), train_bars, test_bars, step_bars)
    if int(step_bars) < int(test_bars):
        raise ValueError("step_bars must be greater than or equal to test_bars")
    if not fold_specs:
        trades, folds = _empty_frames()
        required = int(train_bars) + 2
        summary = _insufficient_summary(
            [f"時系列が{len(prepared)}本で、最低{required}本に不足しています"],
            rows=len(prepared), criteria=criteria,
        )
        return {"trades": trades, "folds": folds, "summary": summary}

    all_trades: list[dict] = []
    fold_rows: list[dict] = []
    evaluation_errors: list[str] = []
    skipped_entries = 0
    # Cached by as-of position only.  Each cached object was built from [:pos+1].
    asof_cache: dict[tuple[int, int], tuple[pd.DataFrame, list[dict]]] = {}

    for spec in fold_specs:
        before = len(all_trades)
        position: dict | None = None
        pending_entry: dict | None = None
        pending_exit: dict | None = None

        for pos in range(spec["test_start"], spec["test_end"]):
            timestamp = prepared.index[pos]
            row = prepared.iloc[pos]

            # Orders decided at the previous close execute at today's open.
            if pending_exit is not None and position is not None:
                all_trades.append(_close_trade(
                    position, timestamp, float(row["Open"]),
                    pending_exit["signal_time"], pending_exit["reason"],
                    max(1, pos - position["entry_pos"]),
                    one_way_bps, slippage_bps,
                ))
                position = None
                pending_exit = None

            if pending_entry is not None and position is None:
                entry_price = float(row["Open"])
                risk = _entry_risk(
                    pending_entry["risk_plan"], entry_price, pending_entry["atr"],
                    atr_stop_mult, atr_target_mult,
                )
                if risk is None:
                    skipped_entries += 1
                else:
                    position = {
                        "fold": spec["fold"],
                        "entry_signal_time": pending_entry["signal_time"],
                        "entry_time": timestamp,
                        "entry_pos": pos,
                        "entry_price": entry_price,
                        "stop_price": risk[0],
                        "target_price": risk[1],
                        "atr_at_entry": pending_entry["atr"],
                    }
                pending_entry = None

            # Intraday risk orders take priority over any close signal.  A bar
            # touching both boundaries is treated as a stop first.
            if position is not None:
                stop_hit = float(row["Low"]) <= position["stop_price"]
                target_hit = float(row["High"]) >= position["target_price"]
                if stop_hit:
                    reference = (float(row["Open"])
                                 if float(row["Open"]) <= position["stop_price"]
                                 else position["stop_price"])
                    all_trades.append(_close_trade(
                        position, timestamp, reference, None, "atr_stop",
                        pos - position["entry_pos"] + 1,
                        one_way_bps, slippage_bps,
                    ))
                    position = None
                elif target_hit:
                    reference = (float(row["Open"])
                                 if float(row["Open"]) >= position["target_price"]
                                 else position["target_price"])
                    all_trades.append(_close_trade(
                        position, timestamp, reference, None, "atr_target",
                        pos - position["entry_pos"] + 1,
                        one_way_bps, slippage_bps,
                    ))
                    position = None

            # Test-boundary liquidation is not a trading signal.  It prevents
            # any training or later-fold return from leaking into this fold.
            if pos == spec["test_end"] - 1:
                if position is not None:
                    all_trades.append(_close_trade(
                        position, timestamp, float(row["Close"]), None, "fold_end",
                        pos - position["entry_pos"] + 1,
                        one_way_bps, slippage_bps,
                    ))
                    position = None
                continue

            # A hard holding-period cap must not be bypassed by a later
            # indicator/level/rule evaluation failure.
            if (position is not None
                    and pos - position["entry_pos"] + 1 >= int(max_holding_days)):
                pending_exit = {"signal_time": timestamp, "reason": "max_holding"}
                continue

            cache_key = (spec["train_start"], pos)
            if cache_key not in asof_cache:
                try:
                    asof_cache[cache_key] = _history_and_levels(
                        prepared, spec["train_start"], pos,
                    )
                except Exception as exc:  # a failed date becomes WAIT, never a future lookup
                    evaluation_errors.append(f"{timestamp}: level/indicator error: {exc}")
                    continue
            history, current_levels = asof_cache[cache_key]
            mode = "holding" if position is not None else "entry"
            try:
                verdict, result = _evaluate(history, rule, current_levels, mode)
            except Exception as exc:
                evaluation_errors.append(f"{timestamp}: rule evaluation error: {exc}")
                continue

            if position is None and verdict == "BUY":
                pending_entry = {
                    "signal_time": timestamp,
                    "atr": _atr_as_of(history, atr_period),
                    "risk_plan": (result.get("risk_plan")
                                  if isinstance(result.get("risk_plan"), dict) else {}),
                }
            elif position is not None:
                if verdict == "RISK_EXIT":
                    pending_exit = {"signal_time": timestamp, "reason": "risk_exit"}
                elif verdict == "TAKE_PROFIT":
                    pending_exit = {"signal_time": timestamp, "reason": "take_profit"}

        fold_trades = pd.DataFrame(all_trades[before:], columns=TRADE_COLUMNS)
        fold_stats = _stats(fold_trades)
        if not fold_trades.empty:
            fold_stats["max_drawdown"] = _mark_to_market_drawdown(
                prepared, fold_trades, spec["test_start"], spec["test_end"])
        fold_rows.append({
            "fold": spec["fold"],
            "train_start": prepared.index[spec["train_start"]],
            "train_end": prepared.index[spec["train_end"] - 1],
            "test_start": prepared.index[spec["test_start"]],
            "test_end": prepared.index[spec["test_end"] - 1],
            "train_bars": spec["train_end"] - spec["train_start"],
            "test_bars": spec["test_end"] - spec["test_start"],
            **fold_stats,
            "status": "OK" if fold_stats["trades"] else "NO_TRADES",
        })

    trade_df = pd.DataFrame(all_trades, columns=TRADE_COLUMNS)
    fold_df = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
    overall = _stats(trade_df)
    if not trade_df.empty:
        overall["max_drawdown"] = _mark_to_market_drawdown(
            prepared, trade_df, fold_specs[0]["test_start"],
            fold_specs[-1]["test_end"])
    positive_fold_ratio = float((fold_df["expectancy"] > 0).mean()) if len(fold_df) else np.nan
    status, passed, reasons = _qualification(
        overall, fold_df, criteria, evaluation_errors,
    )
    warnings = []
    if skipped_entries:
        warnings.append(f"ATRまたは有効なrisk plan不足で{skipped_entries}件のエントリーを見送りました")
    if evaluation_errors:
        warnings.extend(evaluation_errors[:5])

    summary = {
        "status": status,
        "passed": passed,
        "reasons": reasons,
        "warnings": warnings,
        "bars": len(prepared),
        "fold_count": len(fold_df),
        "trade_count": overall["trades"],
        "wins": overall["wins"],
        "win_rate": overall["win_rate"],
        "avg_net_pnl": overall["avg_net_pnl"],
        "avg_net_return": overall["avg_net_return"],
        "profit_factor": overall["profit_factor"],
        "max_drawdown": overall["max_drawdown"],
        "expectancy": overall["expectancy"],
        "positive_fold_ratio": positive_fold_ratio,
        "evaluation_error_count": len(evaluation_errors),
        "skipped_entries": skipped_entries,
        "acceptance_criteria": criteria,
        "cost_model": {
            "one_way_bps": float(one_way_bps),
            "slippage_bps": float(slippage_bps),
            "signal": "daily close",
            "fill": "next open",
            "same_bar_stop_target": "stop first",
            "drawdown": "daily close mark-to-market including costs",
        },
        "note": ("最大DDはOOS期間を日次終値で時価評価し、コストとスリッページを"
                 "反映しています。診断基準の通過は将来の収益性を保証しません。"),
    }
    return {"trades": trade_df, "folds": fold_df, "summary": summary}
