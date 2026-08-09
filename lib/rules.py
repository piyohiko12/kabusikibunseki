"""ユーザーが定義する売買判定ルールの評価。

固定ロジックのV6判定とは別に、利用者が条件・しきい値・配点を自由に組み立てて
判定できるようにする。判定を表示するだけで、注文は一切行わない。

考え方:
- 「買い条件」と「売り条件」をそれぞれ独立に持つ
- 各条件は 指標 / 比較演算子 / しきい値 / 配点 の4つで表す
- 満たした条件の配点を合計し、合格点に達したかで判定する
- 買いと売りが同時に成立したら「判断保留」とし、どちらかに倒さない
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "rules.json"

OPS = {">=": "以上", "<=": "以下"}


# ------------------------------------------------------------- 指標の定義
# fn は指標値(float)かNoneを返す。Noneはデータ不足で「条件を満たさない」扱い。

def _last(series) -> float | None:
    if series is None or len(series) == 0:
        return None
    v = series.iloc[-1] if hasattr(series, "iloc") else series[-1]
    return float(v) if pd.notna(v) else None


def _gap_pct(df, col) -> float | None:
    if col not in df.columns:
        return None
    ma = _last(df[col])
    price = _last(df["Close"])
    if ma is None or price is None or ma == 0:
        return None
    return (price / ma - 1) * 100


def _return_pct(df, days) -> float | None:
    if len(df) <= days:
        return None
    base = float(df["Close"].iloc[-(days + 1)])
    if base == 0:
        return None
    return (float(df["Close"].iloc[-1]) / base - 1) * 100


def _atr_pct(df) -> float | None:
    if len(df) < 15:
        return None
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - df["Close"].shift()).abs(),
                    (df["Low"] - df["Close"].shift()).abs()], axis=1).max(axis=1)
    atr = _last(tr.ewm(alpha=1 / 14, adjust=False).mean())
    price = _last(df["Close"])
    if atr is None or price in (None, 0):
        return None
    return atr / price * 100


def _bb_percent_b(df) -> float | None:
    if "BB_up" not in df.columns or "BB_low" not in df.columns:
        return None
    up, low, price = _last(df["BB_up"]), _last(df["BB_low"]), _last(df["Close"])
    if None in (up, low, price) or up == low:
        return None
    return (price - low) / (up - low) * 100


def _drawdown_52w(df) -> float | None:
    year = df.loc[df.index >= df.index.max() - pd.Timedelta(days=365)]
    if year.empty:
        return None
    high = float(year["High"].max())
    price = _last(df["Close"])
    if high == 0 or price is None:
        return None
    return (price / high - 1) * 100


def _volume_ratio(df) -> float | None:
    if "VOL_MA20" not in df.columns or "Volume" not in df.columns:
        return None
    ma, vol = _last(df["VOL_MA20"]), _last(df["Volume"])
    if ma in (None, 0) or vol is None:
        return None
    return vol / ma


def _annual_vol(df) -> float | None:
    if len(df) < 30:
        return None
    rets = df["Close"].pct_change().dropna().tail(60)
    if rets.empty:
        return None
    return float(rets.std() * np.sqrt(252) * 100)


def _level_distance(ctx, kind) -> float | None:
    """直近のサポート/レジスタンスまでの距離(%)。サポートは正の値で返す。"""
    levels = ctx.get("levels") or []
    price = _last(ctx["df"]["Close"])
    if not levels or price is None:
        return None
    side = [l for l in levels if l["type"] == kind]
    if not side:
        return None
    if kind == "抵抗線":
        nearest = min(side, key=lambda l: l["price"])
        return (nearest["price"] / price - 1) * 100
    nearest = max(side, key=lambda l: l["price"])
    return (1 - nearest["price"] / price) * 100


METRICS: dict[str, dict] = {
    "rsi": {"label": "RSI(14)", "unit": "", "digits": 1,
            "help": "70超で買われすぎ、30未満で売られすぎとされる",
            "fn": lambda c: _last(c["df"].get("RSI"))},
    "sma20_gap": {"label": "SMA20との乖離率", "unit": "%", "digits": 2,
                  "help": "プラスは20日移動平均より上",
                  "fn": lambda c: _gap_pct(c["df"], "SMA20")},
    "sma50_gap": {"label": "SMA50との乖離率", "unit": "%", "digits": 2,
                  "help": "中期トレンドに対する位置",
                  "fn": lambda c: _gap_pct(c["df"], "SMA50")},
    "sma200_gap": {"label": "SMA200との乖離率", "unit": "%", "digits": 2,
                   "help": "長期トレンドに対する位置",
                   "fn": lambda c: _gap_pct(c["df"], "SMA200")},
    "macd_hist": {"label": "MACDヒストグラム", "unit": "", "digits": 3,
                  "help": "プラスは短期が長期を上回る",
                  "fn": lambda c: _last(c["df"].get("MACD_hist"))},
    "stoch_k": {"label": "ストキャスティクス%K", "unit": "", "digits": 1,
                "help": "80超で買われすぎ、20未満で売られすぎ",
                "fn": lambda c: _last(c["df"].get("STOCH_K"))},
    "bb_b": {"label": "ボリンジャー%B", "unit": "", "digits": 1,
             "help": "0=下限、100=上限",
             "fn": lambda c: _bb_percent_b(c["df"])},
    "vol_ratio": {"label": "出来高(20日平均比)", "unit": "倍", "digits": 2,
                  "help": "1.0で平均並み",
                  "fn": lambda c: _volume_ratio(c["df"])},
    "ret_1d": {"label": "前日比", "unit": "%", "digits": 2, "help": "",
               "fn": lambda c: _return_pct(c["df"], 1)},
    "ret_5d": {"label": "5日リターン", "unit": "%", "digits": 2, "help": "",
               "fn": lambda c: _return_pct(c["df"], 5)},
    "ret_20d": {"label": "20日リターン", "unit": "%", "digits": 2, "help": "",
                "fn": lambda c: _return_pct(c["df"], 20)},
    "dd_52w": {"label": "52週高値からの下落率", "unit": "%", "digits": 2,
               "help": "マイナスが大きいほど高値から離れている",
               "fn": lambda c: _drawdown_52w(c["df"])},
    "atr_pct": {"label": "ATR(価格比)", "unit": "%", "digits": 2,
                "help": "1日の平均的な値動きの大きさ",
                "fn": lambda c: _atr_pct(c["df"])},
    "ann_vol": {"label": "年率ボラティリティ", "unit": "%", "digits": 1,
                "help": "直近60日から算出",
                "fn": lambda c: _annual_vol(c["df"])},
    "dist_support": {"label": "直近サポートまでの距離", "unit": "%", "digits": 2,
                     "help": "小さいほどサポートに近い",
                     "fn": lambda c: _level_distance(c, "サポート")},
    "dist_resistance": {"label": "直近レジスタンスまでの距離", "unit": "%",
                        "digits": 2, "help": "小さいほど上値が近い",
                        "fn": lambda c: _level_distance(c, "抵抗線")},
}


# ------------------------------------------------------------- 既定ルール

def default_rules() -> dict:
    """最初から使える3つのルールセット。"""
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


# --------------------------------------------------------------- 評価

def _fmt(metric_id: str, value) -> str:
    m = METRICS.get(metric_id, {})
    if value is None:
        return "取得できず"
    return f"{value:,.{m.get('digits', 2)}f}{m.get('unit', '')}"


def evaluate_side(ctx: dict, side: dict) -> dict:
    """買い側または売り側の条件をまとめて評価する。"""
    checks = []
    score = 0
    total = 0
    for cond in side.get("conditions", []):
        m = METRICS.get(cond["metric"])
        pts = int(cond.get("points", 0))
        total += pts
        if m is None:
            continue
        try:
            actual = m["fn"](ctx)
        except Exception:
            actual = None
        ok = False
        if actual is not None:
            ok = actual >= cond["value"] if cond["op"] == ">=" else actual <= cond["value"]
        if ok:
            score += pts
        checks.append({
            "metric": cond["metric"], "label": m["label"], "points": pts,
            "op": cond["op"], "threshold": cond["value"], "ok": ok,
            "actual": actual, "actual_text": _fmt(cond["metric"], actual),
            "requirement": f"{cond['value']:,.2f}{m.get('unit', '')}"
                           f"{OPS[cond['op']]}",
        })
    threshold = int(side.get("threshold", 0))
    return {"score": score, "total": total, "threshold": threshold,
            "passed": score >= threshold and total > 0, "checks": checks}


def evaluate(df: pd.DataFrame, rule: dict, levels: list | None = None) -> dict:
    """ルール全体を評価して判定を返す。

    verdict: BUY / SELL / CONFLICT(両方成立) / NEUTRAL
    """
    ctx = {"df": df, "levels": levels or []}
    buy = evaluate_side(ctx, rule.get("buy", {}))
    sell = evaluate_side(ctx, rule.get("sell", {}))

    if buy["passed"] and sell["passed"]:
        verdict, summary = "CONFLICT", "買い条件と売り条件が同時に成立しています"
    elif buy["passed"]:
        verdict, summary = "BUY", "買い条件が成立しています"
    elif sell["passed"]:
        verdict, summary = "SELL", "売り条件が成立しています"
    else:
        verdict, summary = "NEUTRAL", "どちらの条件も成立していません"
    return {"verdict": verdict, "summary": summary, "buy": buy, "sell": sell}


# --------------------------------------------------------------- 永続化

def load() -> dict:
    """{"rules": {名前: ルール}, "active": 名前} を返す。無ければ既定を返す。"""
    if DATA_FILE.exists():
        try:
            saved = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            rules = saved.get("rules")
            if isinstance(rules, dict) and rules:
                active = saved.get("active")
                if active not in rules:
                    active = next(iter(rules))
                return {"rules": rules, "active": active}
        except Exception:
            pass
    rules = default_rules()
    return {"rules": rules, "active": next(iter(rules))}


def save(rules: dict, active: str) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps({"rules": rules, "active": active}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def conditions_from_table(table: pd.DataFrame) -> list[dict]:
    """編集表(指標/条件/しきい値/配点)を条件リストに変換する。

    行を追加した直後は各セルが未入力(NaN)になりうるため、
    NaNをそのまま int() に渡して落ちないよう、ここで丸ごと吸収する。
    """
    labels = {v["label"]: k for k, v in METRICS.items()}
    out = []
    for _, row in table.iterrows():
        metric = labels.get(row.get("指標"))
        value = row.get("しきい値")
        if metric is None or value is None or pd.isna(value):
            continue
        op = row.get("条件")
        if op not in OPS:
            op = ">="
        points = row.get("配点")
        points = 0 if points is None or pd.isna(points) else int(points)
        out.append({"metric": metric, "op": op, "value": float(value),
                    "points": max(points, 0)})
    return out


def validate(rule: dict) -> list[str]:
    """ルールの妥当性を確認し、問題点の一覧を返す(空なら問題なし)。"""
    problems = []
    for side_key, side_name in (("buy", "買い"), ("sell", "売り")):
        side = rule.get(side_key, {})
        conds = side.get("conditions", [])
        if not conds:
            problems.append(f"{side_name}条件が1つもありません")
            continue
        total = sum(int(c.get("points", 0)) for c in conds)
        if total <= 0:
            problems.append(f"{side_name}条件の配点合計が0です")
            continue
        if int(side.get("threshold", 0)) > total:
            problems.append(
                f"{side_name}の合格点({side['threshold']})が"
                f"配点合計({total})を超えており、絶対に成立しません")
        seen = set()
        for c in conds:
            key = (c.get("metric"), c.get("op"))
            if key in seen:
                m = METRICS.get(c.get("metric"), {}).get("label", c.get("metric"))
                problems.append(f"{side_name}条件に「{m} {OPS.get(c.get('op'), '')}」が重複しています")
            seen.add(key)
    return problems
