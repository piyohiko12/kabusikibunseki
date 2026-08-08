"""Route B V6 判定ロジック(判定表示のみ・発注機能なし)。

仕様書「Route B V6 セクター選定・銘柄選定・売買判断ルール」の計算を、
そのまま再現した決定論的なルールです。言語モデルによる裁量judgementは
一切入りません。固定された数式・点数・閾値・優先順位だけで動きます。

**このモジュールは注文を出しません。** moomooの取引APIも参照しません。
利用者が判断するための材料を計算して返すだけです。

仕様との相違点(データ源の違いによる、避けられない差):

1. 仕様はOpenDの現在値スナップショットと累積出来高を5分ごとに記録した
   近似時系列を使う。本実装はYahoo Financeの5分足OHLCVを使う。
   したがって「増分出来高」は差分ではなく足の出来高そのもので、
   仕様の意図する量に一致する(むしろ差分より正確)。
2. 「観測ドル出来高 = 最新価格 × 累積出来高」の累積出来高は、
   当日セッションの5分足出来高の累計とする。
3. 取引所公式の板・気配・約定は使わない(仕様も同様に使っていない)。
4. SIMULATE口座・ロック・起動マーカー・口座束縛といった発注側の安全条件は、
   発注しないため対象外。データ健全性の条件のみ実装する。
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 定数

BENCHMARK = "SPY"
MIN_SAMPLES = 24          # 判定に必要な最小サンプル数(5分足24本)
EMA_WINDOW = 21           # EMAを計算する対象サンプル数
EMA_FAST, EMA_SLOW = 5, 13
VOL_LOOKBACK = 12         # 増分出来高の中央値をとる区間数
RMS_LOOKBACK = 12         # 短期変動RMSの対象区間数
RSI_PERIOD = 14

# セクター定義。基準ETFとSPYは強さの物差しで、選定対象ではない。
SECTORS = [
    {"name": "TECHNOLOGY", "jp": "テクノロジー", "etf": "XLK",
     "members": ["AAPL", "MSFT", "NVDA", "AVGO"], "etf2x": "ROM"},
    {"name": "FINANCIALS", "jp": "金融", "etf": "XLF",
     "members": ["JPM", "BAC", "GS", "MS"], "etf2x": "UYG"},
    {"name": "HEALTH_CARE", "jp": "ヘルスケア", "etf": "XLV",
     "members": ["LLY", "UNH", "JNJ", "ABBV"], "etf2x": "RXL"},
    {"name": "CONSUMER_DISCRETIONARY", "jp": "一般消費財", "etf": "XLY",
     "members": ["AMZN", "TSLA", "HD", "MCD"], "etf2x": "UCC"},
    {"name": "COMMUNICATION_SERVICES", "jp": "通信サービス", "etf": "XLC",
     "members": ["META", "GOOGL", "NFLX", "DIS"], "etf2x": "LTL"},
    {"name": "INDUSTRIALS", "jp": "資本財", "etf": "XLI",
     "members": ["GE", "CAT", "RTX", "HON"], "etf2x": "UXI"},
    {"name": "ENERGY", "jp": "エネルギー", "etf": "XLE",
     "members": ["XOM", "CVX", "COP", "SLB"], "etf2x": "DIG"},
    {"name": "CONSUMER_STAPLES", "jp": "生活必需品", "etf": "XLP",
     "members": ["WMT", "COST", "PG", "KO"], "etf2x": "UGE"},
    {"name": "UTILITIES", "jp": "公益", "etf": "XLU",
     "members": ["NEE", "SO", "DUK", "AEP"], "etf2x": "UPW"},
    {"name": "MATERIALS", "jp": "素材", "etf": "XLB",
     "members": ["LIN", "SHW", "FCX", "APD"], "etf2x": "UYM"},
    {"name": "REAL_ESTATE", "jp": "不動産", "etf": "XLRE",
     "members": ["PLD", "AMT", "EQIX", "WELL"], "etf2x": "URE"},
]

# 閾値(仕様どおり。ここを変えると判定が変わるため定数として明示する)
SECTOR_PASS = 65
STOCK_PASS = 70
ETF2X_PASS = 80
STRONG_SECTOR_FOR_2X = 85
BUY_PASS = 72
SIGNAL_REVERSAL = 30

MIN_PRICE = 5.00
MIN_DOLLAR_VOLUME = 25_000_000

STOCK_RMS_MAX = 0.006
ETF2X_RMS_MAX = 0.012
BUY_RMS_MAX = 0.004

STOP_LOSS = -0.006
TAKE_PROFIT = 0.010
TRAIL_ARM = 0.005          # 最高値がこれ以上上がったらトレーリング作動
TRAIL_GIVEBACK = 0.0035    # 最高値からの下落がこれ以上で退出
MAX_HOLD_SAMPLES = 18

# 境界の判定に使う許容誤差。
# 例えば 99.4/100 - 1 は -0.005999999999999894 になり、ちょうど-0.60%でも
# 「-0.006以下」を満たさなくなる。仕様は境界を含むため、その分だけ緩める。
EPS = 1e-9

BUY_WINDOW = ("10:00", "15:30")
FORCED_EXIT = "15:50"


def universe() -> list[str]:
    """観測する67銘柄(SPY + セクターETF11 + 普通株44 + 日次2倍ETF11)。"""
    out = [BENCHMARK]
    for s in SECTORS:
        out.append(s["etf"])
        out.extend(s["members"])
        out.append(s["etf2x"])
    return out


# -------------------------------------------------------------- 計算部品

def ema_cross(prices: np.ndarray) -> tuple[float, float] | None:
    """直近EMA_WINDOW本でEMA5とEMA13を求める。初期値は区間の最初の価格。"""
    p = prices[-EMA_WINDOW:]
    if len(p) < EMA_SLOW:
        return None

    def run(period: int) -> float:
        alpha = 2.0 / (period + 1)
        cur = float(p[0])
        for v in p[1:]:
            cur = alpha * float(v) + (1 - alpha) * cur
        return cur

    return run(EMA_FAST), run(EMA_SLOW)


def rsi14(prices: np.ndarray) -> float | None:
    """RSI(14)。直近14本の値動きの単純平均を使う(Wilder平滑化ではない)。"""
    if len(prices) < RSI_PERIOD + 1:
        return None
    diffs = np.diff(prices[-(RSI_PERIOD + 1):].astype(float))
    gains = np.clip(diffs, 0, None)
    losses = np.clip(-diffs, 0, None)
    avg_gain = float(gains.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0:
        if avg_gain > 0:
            return 100.0
        return 50.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def ret_intervals(prices: np.ndarray, k: int) -> float | None:
    """k区間前からのリターン。5分足ならk=3で15分、k=6で30分。"""
    if len(prices) < k + 1:
        return None
    base = float(prices[-(k + 1)])
    if base <= 0:
        return None
    return float(prices[-1]) / base - 1


def volume_ratio(volumes: np.ndarray) -> float:
    """最新の増分出来高 / 直前最大12区間の正の増分出来高の中央値。

    過去側に正の出来高が1つもなければ0(=不合格)を返す。
    """
    if len(volumes) < 2:
        return 0.0
    latest = float(volumes[-1])
    prior = volumes[-(VOL_LOOKBACK + 1):-1].astype(float)
    positive = prior[prior > 0]
    if positive.size == 0:
        return 0.0
    med = float(np.median(positive))
    if med <= 0:
        return 0.0
    return latest / med


def vwap_reference(prices: np.ndarray, volumes: np.ndarray) -> float:
    """直近EMA_WINDOW本の出来高加重参照価格。正の出来高だけを重みにする。"""
    p = prices[-EMA_WINDOW:].astype(float)
    v = volumes[-EMA_WINDOW:].astype(float)
    mask = v > 0
    if not mask.any():
        return float(p[-1])
    return float((p[mask] * v[mask]).sum() / v[mask].sum())


def rms_return(prices: np.ndarray) -> float:
    """直近最大12区間の5分リターンの二乗平均平方根。"""
    p = prices[-(RMS_LOOKBACK + 1):].astype(float)
    if len(p) < 2:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(p) / p[:-1]
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(rets ** 2)))


def dollar_volume(prices: np.ndarray, volumes: np.ndarray) -> float:
    """観測ドル出来高 = 最新価格 × 観測期間の累積出来高。"""
    return float(prices[-1]) * float(np.nansum(volumes))


def _arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (df["Close"].to_numpy(float), df["Volume"].to_numpy(float))


# ---------------------------------------------------------- セクター採点

def score_sector(sector_df: pd.DataFrame, spy_df: pd.DataFrame,
                 member_dfs: dict[str, pd.DataFrame]) -> dict:
    """セクターを100点満点で採点する。各条件は満たしたときだけ加点する。"""
    p, v = _arrays(sector_df)
    spy_p, _ = _arrays(spy_df)
    checks = []

    r30 = ret_intervals(p, 6)
    spy30 = ret_intervals(spy_p, 6)
    ok_a = r30 is not None and spy30 is not None and (r30 - spy30) > 0
    checks.append({"key": "A", "label": "30分でSPYに勝っている", "points": 25,
                   "ok": bool(ok_a),
                   "detail": (f"セクター {r30 * 100:+.2f}% / SPY {spy30 * 100:+.2f}%"
                              if r30 is not None and spy30 is not None else "データ不足")})

    ema = ema_cross(p)
    ok_b = ema is not None and ema[0] > ema[1]
    checks.append({"key": "B", "label": "EMA5がEMA13を上回る", "points": 25,
                   "ok": bool(ok_b),
                   "detail": (f"EMA5 {ema[0]:,.2f} / EMA13 {ema[1]:,.2f}"
                              if ema else "データ不足")})

    r3 = ret_intervals(p, 3)
    ok_c = r3 is not None and r3 > 0
    checks.append({"key": "C", "label": "15分モメンタムが正", "points": 15,
                   "ok": bool(ok_c),
                   "detail": f"{r3 * 100:+.2f}%" if r3 is not None else "データ不足"})

    vr = volume_ratio(v)
    ok_d = vr >= 1.10
    checks.append({"key": "D", "label": "出来高比率 1.10以上", "points": 15,
                   "ok": bool(ok_d), "detail": f"{vr:.2f}倍"})

    advancing = 0
    for sym, mdf in member_dfs.items():
        mp, _ = _arrays(mdf)
        mr = ret_intervals(mp, 3)
        if mr is not None and mr > 0:
            advancing += 1
    ok_e = advancing >= 2
    checks.append({"key": "E", "label": "構成銘柄の半数以上が上昇", "points": 20,
                   "ok": bool(ok_e), "detail": f"{advancing}/{len(member_dfs)}銘柄"})

    score = sum(c["points"] for c in checks if c["ok"])
    return {"score": score, "checks": checks}


# -------------------------------------------------------------- 銘柄採点

def _liquidity(prices: np.ndarray, volumes: np.ndarray) -> dict:
    price = float(prices[-1])
    dv = dollar_volume(prices, volumes)
    return {"price": price, "dollar_volume": dv,
            "ok": price >= MIN_PRICE and dv >= MIN_DOLLAR_VOLUME}


def score_candidate(df: pd.DataFrame, sector_df: pd.DataFrame,
                    rms_max: float) -> dict:
    """普通株・日次2倍ETF共通の採点(閾値のみ異なる)。"""
    p, v = _arrays(df)
    sp, _ = _arrays(sector_df)
    liq = _liquidity(p, v)
    checks = []

    r15 = ret_intervals(p, 3)
    s15 = ret_intervals(sp, 3)
    ok_a = r15 is not None and s15 is not None and (r15 - s15) > 0
    checks.append({"key": "A", "label": "15分でセクターETFに勝っている", "points": 30,
                   "ok": bool(ok_a),
                   "detail": (f"銘柄 {r15 * 100:+.2f}% / セクター {s15 * 100:+.2f}%"
                              if r15 is not None and s15 is not None else "データ不足")})

    ema = ema_cross(p)
    ok_b = ema is not None and ema[0] > ema[1]
    checks.append({"key": "B", "label": "EMA5がEMA13を上回る", "points": 25,
                   "ok": bool(ok_b),
                   "detail": (f"EMA5 {ema[0]:,.2f} / EMA13 {ema[1]:,.2f}"
                              if ema else "データ不足")})

    vr = volume_ratio(v)
    ok_c = vr >= 1.15
    checks.append({"key": "C", "label": "出来高比率 1.15以上", "points": 20,
                   "ok": bool(ok_c), "detail": f"{vr:.2f}倍"})

    ref = vwap_reference(p, v)
    ok_d = float(p[-1]) >= ref
    checks.append({"key": "D", "label": "出来高加重参照価格以上", "points": 15,
                   "ok": bool(ok_d),
                   "detail": f"現値 ${p[-1]:,.2f} / 参照 ${ref:,.2f}"})

    rms = rms_return(p)
    ok_e = rms <= rms_max
    checks.append({"key": "E", "label": f"短期変動 {rms_max * 100:.2f}%以下",
                   "points": 10, "ok": bool(ok_e), "detail": f"{rms * 100:.2f}%"})

    score = sum(c["points"] for c in checks if c["ok"])
    return {"score": score, "checks": checks, "liquidity": liq}


# ------------------------------------------------------------ 買いスコア

def score_buy(df: pd.DataFrame) -> dict:
    """選択された1銘柄の買いスコア(100点満点)。"""
    p, v = _arrays(df)
    checks = []

    ema = ema_cross(p)
    ok_a = ema is not None and ema[0] > ema[1]
    checks.append({"key": "A", "label": "EMA5がEMA13を上回る", "points": 28,
                   "ok": bool(ok_a),
                   "detail": (f"EMA5 {ema[0]:,.2f} / EMA13 {ema[1]:,.2f}"
                              if ema else "データ不足")})

    r3 = ret_intervals(p, 3)
    ok_b = r3 is not None and r3 > 0
    checks.append({"key": "B", "label": "15分モメンタムが正", "points": 18,
                   "ok": bool(ok_b),
                   "detail": f"{r3 * 100:+.2f}%" if r3 is not None else "データ不足"})

    rsi = rsi14(p)
    ok_c = rsi is not None and 48 <= rsi <= 78
    checks.append({"key": "C", "label": "RSI(14)が48〜78", "points": 18,
                   "ok": bool(ok_c),
                   "detail": f"{rsi:.1f}" if rsi is not None else "データ不足"})

    vr = volume_ratio(v)
    ok_d = vr >= 1.15
    checks.append({"key": "D", "label": "出来高比率 1.15以上", "points": 16,
                   "ok": bool(ok_d), "detail": f"{vr:.2f}倍"})

    ref = vwap_reference(p, v)
    ok_e = float(p[-1]) >= ref
    checks.append({"key": "E", "label": "出来高加重参照価格以上", "points": 12,
                   "ok": bool(ok_e),
                   "detail": f"現値 ${p[-1]:,.2f} / 参照 ${ref:,.2f}"})

    rms = rms_return(p)
    ok_f = rms <= BUY_RMS_MAX
    checks.append({"key": "F", "label": "短期変動 0.40%以下", "points": 8,
                   "ok": bool(ok_f), "detail": f"{rms * 100:.2f}%"})

    score = sum(c["points"] for c in checks if c["ok"])
    return {"score": score, "checks": checks, "passed": score >= BUY_PASS}


# ---------------------------------------------------------------- 退出

def exit_signals(df: pd.DataFrame, entry_price: float, held_samples: int,
                 max_price: float | None = None,
                 now_et: pd.Timestamp | None = None) -> list[dict]:
    """成立している退出条件をすべて返す(仕様の A〜F)。"""
    p, _ = _arrays(df)
    last = float(p[-1])
    if entry_price <= 0:
        return []
    ret = last / entry_price - 1
    peak = float(max_price) if max_price else float(p.max())
    out = []

    if ret <= STOP_LOSS + EPS:
        out.append({"key": "A", "label": "ハードストップロス",
                    "detail": f"買値から {ret * 100:+.2f}%(−0.60%以下)"})
    if ret >= TAKE_PROFIT - EPS:
        out.append({"key": "B", "label": "テイクプロフィット",
                    "detail": f"買値から {ret * 100:+.2f}%(+1.00%以上)"})

    peak_gain = peak / entry_price - 1
    giveback = 1 - last / peak if peak > 0 else 0.0
    if peak_gain >= TRAIL_ARM - EPS and giveback >= TRAIL_GIVEBACK - EPS:
        out.append({"key": "C", "label": "トレーリングストップ",
                    "detail": (f"最高値 +{peak_gain * 100:.2f}% から "
                               f"{giveback * 100:.2f}% 下落")})
    if held_samples >= MAX_HOLD_SAMPLES:
        out.append({"key": "D", "label": "最大保有時間",
                    "detail": f"{held_samples}サンプル(18以上・約90分)"})

    if now_et is not None:
        hhmm = now_et.strftime("%H:%M")
        if hhmm >= FORCED_EXIT:
            out.append({"key": "E", "label": "セッション終了前の強制退出",
                        "detail": f"{hhmm} ET(15:50以降)"})

    bs = score_buy(df)
    if bs["score"] <= SIGNAL_REVERSAL:
        out.append({"key": "F", "label": "シグナル反転",
                    "detail": f"買いスコア {bs['score']}点(30点以下)"})
    return out


# ------------------------------------------------------------ データ検査

def check_data(frames: dict[str, pd.DataFrame], now_et: pd.Timestamp | None = None,
               max_age_sec: int = 900) -> dict:
    """発注前のデータ健全性に相当する検査(判定の信頼性チェック)。

    仕様の180秒はスナップショット前提。本実装は5分足のため、
    直近の足が閉じてからの経過を見る目的で900秒(15分)を既定とする。
    """
    issues = []
    need = set(universe())
    have = set(frames)
    missing = sorted(need - have)
    extra = sorted(have - need)
    if missing:
        issues.append(f"{len(missing)}銘柄のデータが取得できませんでした: "
                      + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else ""))
    if extra:
        issues.append(f"想定外の銘柄が含まれています: {', '.join(extra[:5])}")

    short = [s for s, d in frames.items() if len(d) < MIN_SAMPLES]
    if short:
        issues.append(f"{len(short)}銘柄が{MIN_SAMPLES}サンプル未満です: "
                      + ", ".join(sorted(short)[:8])
                      + ("…" if len(short) > 8 else ""))

    bad = []
    for s, d in frames.items():
        if d.empty:
            continue
        if not d.index.is_monotonic_increasing or d.index.has_duplicates:
            bad.append(f"{s}(時刻の重複/逆転)")
            continue
        c = d["Close"].to_numpy(float)
        v = d["Volume"].to_numpy(float)
        if not np.all(np.isfinite(c)) or np.any(c <= 0):
            bad.append(f"{s}(価格が不正)")
        elif not np.all(np.isfinite(v)) or np.any(v < 0):
            bad.append(f"{s}(出来高が不正)")
    if bad:
        issues.append("データが不正な銘柄: " + ", ".join(bad[:6])
                      + ("…" if len(bad) > 6 else ""))

    latest = None
    if frames:
        stamps = [d.index[-1] for d in frames.values() if not d.empty]
        if stamps:
            latest = max(stamps)
            if len({str(s) for s in stamps}) > 1:
                issues.append("銘柄間で最新時刻がそろっていません(非同期)")
            if now_et is not None:
                age = (now_et - latest.tz_convert(now_et.tz)
                       if latest.tzinfo else now_et - latest).total_seconds()
                if age > max_age_sec:
                    issues.append(f"データが古すぎます(最新から{age / 60:.0f}分経過)")

    return {"ok": not issues, "issues": issues, "latest": latest,
            "samples": min((len(d) for d in frames.values()), default=0)}


# ------------------------------------------------------------ 総合判定

def evaluate(frames: dict[str, pd.DataFrame],
             now_et: pd.Timestamp | None = None) -> dict:
    """セクター選定 → 銘柄選定 → 買い判定を一括で行う。

    戻り値は表示用の辞書。注文は一切行わない。
    """
    data = check_data(frames, now_et)
    result = {"data": data, "sectors": [], "selected_sector": None,
              "stock": None, "etf2x": None, "choice": None,
              "buy": None, "decision": "WAIT", "reasons": []}

    if BENCHMARK not in frames or len(frames[BENCHMARK]) < MIN_SAMPLES:
        result["reasons"].append(f"{BENCHMARK}のデータが不足しています")
        return result

    spy = frames[BENCHMARK]
    rows = []
    for spec in SECTORS:
        if spec["etf"] not in frames:
            continue
        members = {m: frames[m] for m in spec["members"] if m in frames}
        if len(frames[spec["etf"]]) < MIN_SAMPLES or not members:
            continue
        sc = score_sector(frames[spec["etf"]], spy, members)
        rows.append({**spec, **sc})
    if not rows:
        result["reasons"].append("セクターを採点できるデータがありません")
        return result

    # 得点降順、同点はセクター名のアルファベット順
    rows.sort(key=lambda r: (-r["score"], r["name"]))
    result["sectors"] = rows
    top = rows[0]
    result["selected_sector"] = top

    if data["samples"] < MIN_SAMPLES:
        result["reasons"].append(
            f"サンプルが{data['samples']}本しかありません({MIN_SAMPLES}本必要)")
        return result
    if top["score"] < SECTOR_PASS:
        result["reasons"].append(
            f"最高セクター得点が{top['score']}点で、{SECTOR_PASS}点に届きません")
        return result

    sector_df = frames[top["etf"]]

    # 普通株の採点(流動性フィルターを通ったものだけ順位付け)
    stock_rows = []
    for sym in top["members"]:
        if sym not in frames or len(frames[sym]) < MIN_SAMPLES:
            continue
        sc = score_candidate(frames[sym], sector_df, STOCK_RMS_MAX)
        stock_rows.append({"symbol": sym, **sc})
    passed = [r for r in stock_rows if r["liquidity"]["ok"]]
    passed.sort(key=lambda r: (-r["score"], -r["liquidity"]["dollar_volume"],
                               r["symbol"]))
    best_stock = passed[0] if passed else None
    result["stock"] = {"ranked": stock_rows, "best": best_stock,
                       "passed": bool(best_stock and best_stock["score"] >= STOCK_PASS)}

    # 日次2倍ETFの採点
    etf_row = None
    sym2x = top["etf2x"]
    if sym2x in frames and len(frames[sym2x]) >= MIN_SAMPLES:
        sc = score_candidate(frames[sym2x], sector_df, ETF2X_RMS_MAX)
        etf_row = {"symbol": sym2x, **sc}
    strong = top["score"] >= STRONG_SECTOR_FOR_2X
    etf_ok = bool(etf_row and etf_row["liquidity"]["ok"]
                  and etf_row["score"] >= ETF2X_PASS)
    result["etf2x"] = {"row": etf_row, "strong_sector": strong,
                       "passed": strong and etf_ok}

    # 優先順位: 強セクター条件を満たす2倍ETF > 70点以上の普通株
    if strong and etf_ok:
        result["choice"] = {"symbol": sym2x, "kind": "日次2倍ETF",
                            "score": etf_row["score"],
                            "why": (f"セクター{top['score']}点(85点以上)かつ"
                                    f"2倍ETF{etf_row['score']}点(80点以上)のため優先")}
    elif best_stock and best_stock["score"] >= STOCK_PASS:
        result["choice"] = {"symbol": best_stock["symbol"], "kind": "普通株",
                            "score": best_stock["score"],
                            "why": f"セクター内で最上位かつ{STOCK_PASS}点以上"}
    else:
        if not best_stock:
            result["reasons"].append("普通株が流動性条件を満たしません")
        else:
            result["reasons"].append(
                f"最高の普通株が{best_stock['score']}点で{STOCK_PASS}点に届かず、"
                "2倍ETFの優先条件も満たしません")
        return result

    # 買いスコア
    chosen = result["choice"]["symbol"]
    buy = score_buy(frames[chosen])
    result["buy"] = buy
    if not buy["passed"]:
        result["reasons"].append(
            f"{chosen}の買いスコアが{buy['score']}点で、{BUY_PASS}点に届きません")
        return result

    # 時間帯(仕様の買付時間)
    if now_et is not None:
        hhmm = now_et.strftime("%H:%M")
        if not (BUY_WINDOW[0] <= hhmm <= BUY_WINDOW[1]):
            result["reasons"].append(
                f"買付時間({BUY_WINDOW[0]}〜{BUY_WINDOW[1]} ET)の外です(現在{hhmm})")
            return result
    if not data["ok"]:
        result["reasons"].append("データ健全性の条件を満たしません")
        return result

    result["decision"] = "BUY"
    return result
