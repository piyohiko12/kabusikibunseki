"""株価アラートの定義・判定・永続化。

**重要な制約**: このアプリはブラウザからアクセスされたときだけ動きます。
アプリを閉じている間に監視して通知することはできません。
アラートは「アプリを開いている間に、画面上で知らせる」ものです。

判定は毎回その場で行い、成立の履歴は残しますが、成立を見逃さないための
バックグラウンド監視は行いません。
"""

import json
import uuid
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "alerts.json"

# 種類ごとの定義。fn(ctx, value) -> (成立したか, 実測値の文字列)
KINDS = {
    "price_above": {"label": "価格が上回る", "unit": "$", "needs_value": True},
    "price_below": {"label": "価格が下回る", "unit": "$", "needs_value": True},
    "change_above": {"label": "前日比が上回る", "unit": "%", "needs_value": True},
    "change_below": {"label": "前日比が下回る", "unit": "%", "needs_value": True},
    "rsi_above": {"label": "RSI(14)が上回る", "unit": "", "needs_value": True},
    "rsi_below": {"label": "RSI(14)が下回る", "unit": "", "needs_value": True},
    "near_support": {"label": "サポートに近づく(距離%以内)", "unit": "%",
                     "needs_value": True},
    "near_resistance": {"label": "レジスタンスに近づく(距離%以内)", "unit": "%",
                        "needs_value": True},
    "rule_buy": {"label": "売買ルールが買い判定になる", "unit": "",
                 "needs_value": False},
    "rule_take_profit": {"label": "保有ルールが利益確定判定になる", "unit": "",
                         "needs_value": False},
    "rule_risk_exit": {"label": "保有ルールがリスク退出判定になる", "unit": "",
                       "needs_value": False},
    # 旧保存データとの互換用。SELLを空売り開始とは解釈せず、いずれかの
    # ロング手仕舞い判定で成立させる。
    "rule_sell": {"label": "保有ルールが手仕舞い判定になる（旧形式）", "unit": "",
                  "needs_value": False},
}


def new_alert(ticker: str, kind: str, value: float | None = None,
              note: str = "") -> dict:
    return {"id": uuid.uuid4().hex[:8], "ticker": ticker.strip().upper(),
            "kind": kind, "value": None if value is None else float(value),
            "note": note.strip(), "enabled": True}


def _price(df) -> float | None:
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


def _change_pct(df) -> float | None:
    if df is None or len(df) < 2:
        return None
    prev = float(df["Close"].iloc[-2])
    if prev == 0:
        return None
    return (float(df["Close"].iloc[-1]) / prev - 1) * 100


def _rsi(df) -> float | None:
    if df is None or "RSI" not in df.columns or df.empty:
        return None
    v = df["RSI"].iloc[-1]
    return float(v) if pd.notna(v) else None


def _nearest_distance(levels, price, kind) -> float | None:
    side = [l for l in (levels or []) if l["type"] == kind]
    if not side or not price:
        return None
    if kind == "抵抗線":
        nearest = min(side, key=lambda l: l["price"])
        return abs(nearest["price"] / price - 1) * 100
    nearest = max(side, key=lambda l: l["price"])
    return abs(1 - nearest["price"] / price) * 100


def check(alert: dict, ctx: dict) -> dict:
    """1件のアラートを判定する。

    ctx: {"df": 指標付きDataFrame, "levels": [...], "rule_verdict": "BUY"など}
    戻り値: {"triggered": bool, "actual": str, "reason": str}
    """
    df = ctx.get("df")
    kind = alert.get("kind")
    value = alert.get("value")
    try:
        price = float(ctx.get("price")) if ctx.get("price") is not None else _price(df)
    except (TypeError, ValueError):
        price = _price(df)

    def out(trig, actual, reason=""):
        return {"triggered": bool(trig), "actual": actual, "reason": reason}

    if kind in ("price_above", "price_below"):
        if price is None:
            return out(False, "取得できず", "価格を取得できませんでした")
        hit = price > value if kind == "price_above" else price < value
        return out(hit, f"${price:,.2f}")

    if kind in ("change_above", "change_below"):
        previous_close = ctx.get("previous_close")
        try:
            chg = ((price / float(previous_close) - 1) * 100
                   if price is not None and float(previous_close) != 0 else None)
        except (TypeError, ValueError, ZeroDivisionError):
            chg = None
        if chg is None:
            chg = _change_pct(df)
        if chg is None:
            return out(False, "取得できず", "前日比を計算できませんでした")
        hit = chg > value if kind == "change_above" else chg < value
        return out(hit, f"{chg:+.2f}%")

    if kind in ("rsi_above", "rsi_below"):
        rsi = _rsi(df)
        if rsi is None:
            return out(False, "取得できず", "RSIを計算できませんでした")
        hit = rsi > value if kind == "rsi_above" else rsi < value
        return out(hit, f"{rsi:.1f}")

    if kind in ("near_support", "near_resistance"):
        side = "サポート" if kind == "near_support" else "抵抗線"
        dist = _nearest_distance(ctx.get("levels"), price, side)
        if dist is None:
            return out(False, "取得できず", f"{side}が検出されていません")
        return out(dist <= value, f"{dist:.2f}%")

    if kind in ("rule_buy", "rule_take_profit", "rule_risk_exit", "rule_sell"):
        verdict = (ctx.get("entry_verdict") if kind == "rule_buy"
                   else ctx.get("holding_verdict"))
        if verdict is None:  # 旧呼び出し側との互換
            verdict = ctx.get("rule_verdict")
        if verdict is None:
            return out(False, "未評価", "売買ルールを評価できませんでした")
        if kind == "rule_buy":
            hit = verdict == "BUY"
        elif kind == "rule_take_profit":
            hit = verdict == "TAKE_PROFIT"
        elif kind == "rule_risk_exit":
            hit = verdict == "RISK_EXIT"
        else:
            hit = verdict in {"SELL", "TAKE_PROFIT", "RISK_EXIT"}
        return out(hit, verdict)

    return out(False, "—", f"未知のアラート種別: {kind}")


def describe(alert: dict) -> str:
    """アラートの内容を1行の日本語にする。"""
    spec = KINDS.get(alert.get("kind"), {})
    label = spec.get("label", alert.get("kind"))
    if not spec.get("needs_value"):
        return f"{alert['ticker']}: {label}"
    unit = spec.get("unit", "")
    val = alert.get("value")
    shown = f"{unit}{val:,.2f}" if unit == "$" else f"{val:,.2f}{unit}"
    return f"{alert['ticker']}: {label} {shown}"


# --------------------------------------------------------------- 永続化

def load() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    try:
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("alerts") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out = []
    for a in items:
        if isinstance(a, dict) and a.get("ticker") and a.get("kind") in KINDS:
            a.setdefault("id", uuid.uuid4().hex[:8])
            a.setdefault("enabled", True)
            a.setdefault("note", "")
            out.append(a)
    return out


def save(alerts: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps({"alerts": alerts}, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def tickers(alerts: list[dict]) -> list[str]:
    """有効なアラートが対象にしている銘柄の一覧(重複なし)。"""
    seen = []
    for a in alerts:
        if a.get("enabled") and a["ticker"] not in seen:
            seen.append(a["ticker"])
    return seen
