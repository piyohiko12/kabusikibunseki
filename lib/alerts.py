"""株価アラートの定義・判定・永続化。

**重要な制約**: このアプリはブラウザからアクセスされたときだけ動きます。
アプリを閉じている間に監視して通知することはできません。
アラートは「アプリを開いている間に、画面上で知らせる」ものです。

判定は毎回その場で行い、成立の履歴は残しますが、成立を見逃さないための
バックグラウンド監視は行いません。
"""

import json
import math
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
    "rule_take_profit": {"label": "保有株を利益確定のため売る判定になる", "unit": "",
                         "needs_value": False},
    "rule_risk_exit": {"label": "保有株を損失抑制のため売る判定になる", "unit": "",
                       "needs_value": False},
    # 旧保存データとの互換用。SELLを空売り開始とは解釈せず、いずれかの
    # ロング手仕舞い判定で成立させる。
    "rule_sell": {"label": "保有株を売る判定になる（以前の設定）", "unit": "",
                  "needs_value": False},
}

RULE_ACTUAL_LABELS = {
    "BUY": "買い候補",
    "BUY_BLOCKED": "買い条件あり・今は待つ",
    "NEUTRAL": "今は買わない",
    "WAIT": "判断を保留",
    "RISK_EXIT": "保有株を売る候補（損失を抑える）",
    "TAKE_PROFIT": "保有株を売る候補（利益を確定する）",
    "HOLD": "そのまま保有",
    "SELL": "保有株を売る候補（以前の設定）",
}


def new_alert(ticker: str, kind: str, value: float | None = None,
              note: str = "") -> dict:
    return {"id": uuid.uuid4().hex[:8], "ticker": ticker.strip().upper(),
            "kind": kind, "value": None if value is None else float(value),
            "note": note.strip(), "enabled": True}


def _price(df) -> float | None:
    if df is None or df.empty or "Close" not in df:
        return None
    return _finite_number(df["Close"].iloc[-1], positive=True)


def _change_pct(df) -> float | None:
    if df is None or "Close" not in df or len(df) < 2:
        return None
    prev = _finite_number(df["Close"].iloc[-2], positive=True)
    current = _finite_number(df["Close"].iloc[-1], positive=True)
    if prev is None or current is None:
        return None
    return (current / prev - 1) * 100


def _rsi(df) -> float | None:
    if df is None or "RSI" not in df.columns or df.empty:
        return None
    return _finite_number(df["RSI"].iloc[-1])


def _finite_number(value, *, positive: bool = False) -> float | None:
    """bool・NaN・無限大を数値として受け入れない。"""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _valid_alert_value(kind: str, value) -> float | None:
    number = _finite_number(value)
    if number is None:
        return None
    if kind in {"price_above", "price_below"} and number <= 0:
        return None
    if kind in {"rsi_above", "rsi_below"} and not 0 <= number <= 100:
        return None
    if kind in {"near_support", "near_resistance"} and number < 0:
        return None
    return number


def _nearest_distance(levels, price, kind) -> float | None:
    current = _finite_number(price, positive=True)
    if current is None:
        return None
    side: list[float] = []
    for level in levels or []:
        if not isinstance(level, dict) or level.get("type") != kind:
            continue
        level_price = _finite_number(level.get("price"), positive=True)
        if level_price is None:
            continue
        # 通過済みの線を「次の」支持・抵抗として扱わない。現在値と同値は
        # その水準へ到達中なので有効（距離0%）とする。
        if ((kind == "抵抗線" and level_price >= current)
                or (kind == "サポート" and level_price <= current)):
            side.append(level_price)
    if not side:
        return None
    if kind == "抵抗線":
        nearest = min(side)
        return (nearest / current - 1) * 100
    nearest = max(side)
    return (1 - nearest / current) * 100


def check(alert: dict, ctx: dict) -> dict:
    """1件のアラートを判定する。

    ctx: {"df": 指標付きDataFrame, "levels": [...], "rule_verdict": "BUY"など}
    戻り値: {"triggered": bool, "actual": str, "reason": str}
    """
    df = ctx.get("df")
    kind = alert.get("kind")
    value = alert.get("value")
    # 呼び出し側がpriceキーを明示してNoneにした場合は、時間外専用価格などを
    # 取得できなかったという意味を保つ。確定日足へ黙って戻すと、古い立会価格で
    # 価格アラートが成立してしまうため、キー自体がない旧呼び出しだけdfへ補完する。
    price = (_finite_number(ctx.get("price"), positive=True)
             if "price" in ctx else _price(df))

    def out(trig, actual, reason=""):
        return {"triggered": bool(trig), "actual": actual, "reason": reason}

    spec = KINDS.get(kind)
    if spec and spec.get("needs_value"):
        value = _valid_alert_value(kind, value)
        if value is None:
            return out(False, "設定不正", "アラートの基準値が欠損または不正です")

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
        if chg is None and "price" not in ctx:
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
        if price is None:
            return out(False, "取得できず", "現在の取引セッションの価格を取得できませんでした")
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
        shown_verdict = ("BUY_BLOCKED" if kind == "rule_buy" and hit
                         and ctx.get("entry_blocked") is True else verdict)
        return out(hit, shown_verdict)

    return out(False, "—", f"未知のアラート種別: {kind}")


def describe(alert: dict) -> str:
    """アラートの内容を1行の日本語にする。"""
    if not isinstance(alert, dict):
        return "不正なアラート設定"
    spec = KINDS.get(alert.get("kind"), {})
    label = spec.get("label", str(alert.get("kind") or "不明な種類"))
    ticker = str(alert.get("ticker") or "銘柄不明")
    if not spec.get("needs_value"):
        return f"{ticker}: {label}"
    unit = spec.get("unit", "")
    val = _valid_alert_value(str(alert.get("kind") or ""), alert.get("value"))
    if val is None:
        return f"{ticker}: {label}（基準値が不正）"
    shown = f"{unit}{val:,.2f}" if unit == "$" else f"{val:,.2f}{unit}"
    return f"{ticker}: {label} {shown}"


def format_actual(alert: dict, actual: object) -> str:
    """アラートの実測値を、空売りと誤認しない日本語表示へ変換する。"""
    text = "—" if actual is None or actual == "" else str(actual)
    if str(alert.get("kind") or "").startswith("rule_"):
        return RULE_ACTUAL_LABELS.get(text.upper(), text)
    return text


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
        if not isinstance(a, dict):
            continue
        ticker = a.get("ticker")
        kind = a.get("kind")
        if not isinstance(ticker, str) or not ticker.strip() or kind not in KINDS:
            continue
        spec = KINDS[kind]
        value = (_valid_alert_value(kind, a.get("value"))
                 if spec.get("needs_value") else None)
        # 基準値なしの価格・指標アラートは成立させず、一覧表示でも落とさない。
        if spec.get("needs_value") and value is None:
            continue
        alert_id = a.get("id")
        note = a.get("note")
        enabled = a.get("enabled", True)
        out.append({
            **a,
            "id": (alert_id.strip() if isinstance(alert_id, str) and alert_id.strip()
                   else uuid.uuid4().hex[:8]),
            "ticker": ticker.strip().upper()[:32],
            "kind": kind,
            "value": value,
            "note": note.strip() if isinstance(note, str) else "",
            # 文字列"false"等を真として監視しない。破損時は安全側の無効。
            "enabled": enabled if isinstance(enabled, bool) else False,
        })
    return out


def save(alerts: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps({"alerts": alerts}, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def tickers(alerts: list[dict]) -> list[str]:
    """有効なアラートが対象にしている銘柄の一覧(重複なし)。"""
    seen = []
    for a in alerts:
        if not isinstance(a, dict) or a.get("enabled") is not True:
            continue
        ticker = a.get("ticker")
        if isinstance(ticker, str) and ticker and ticker not in seen:
            seen.append(ticker)
    return seen
