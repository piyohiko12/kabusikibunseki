"""サポート/レジスタンス(支持線・抵抗線)の検出と参考指値の算出。

手法:
1. スイング高値/安値をフラクタル法で検出し、ATRの0.5倍幅でクラスタリング(ゾーン化)
2. 各ゾーンへの接近イベントを「ヒゲ拒絶」「反発」「突破」に分類する。
   ヒゲだけがゾーンに刺さって実体は入らなかった接触(拒絶)は、
   反発の証拠として最も強いため重く採点する
3. イベントは経過時間(半減期)と出来高(20日平均比)で重み付けする
4. ゾーン内で実際に商われた出来高の割合、期間高値/安値、
   フィボナッチ・キリ番・SMA・ピボットとの合流(コンフルエンス)で加点し、
   直近に実体で破られたレベルは減点する

強さ(★1〜5)の較正:
S&P500構成銘柄・2013〜2018の日次データ(505銘柄)を使い、6ヶ月窓でレベルを算出して
40本先までの結末を実測するウォークフォワード検証で境界を決めた。
較正に使っていない80銘柄での検証では、★5の反発率65.0%に対し★1は58.9%(差+6.1pt)。
各★にほぼ均等(約20%ずつ)に分かれる。

⚠️ 重要な限界: レベルの「当たり外れ」自体は、同じ距離にランダムに置いた線と
ほぼ変わらない(層別調整後のエッジはほぼ0)。★はあくまで**レベル同士の優劣**を
並べるための相対評価であり、反発を予測するものではない。
"""

import numpy as np
import pandas as pd

FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
ROUND_STEPS = (1000, 500, 250, 100, 50, 25, 10, 5, 2.5, 1)

# 素点 → ★ の境界(上記の検証で較正)。各★がほぼ均等に分かれる。
STRENGTH_CUTS = (1.13, 1.52, 1.83, 2.12)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _swing_points(df: pd.DataFrame, window: int) -> list[tuple]:
    roll = 2 * window + 1
    highs, lows = df["High"], df["Low"]
    high_mask = (highs == highs.rolling(roll, center=True).max()).fillna(False)
    low_mask = (lows == lows.rolling(roll, center=True).min()).fillna(False)
    points = [(idx, float(p), "high") for idx, p in highs[high_mask].items()]
    points += [(idx, float(p), "low") for idx, p in lows[low_mask].items()]
    return points


def volume_profile(df: pd.DataFrame, bins: int = 40) -> pd.Series:
    """価格帯別出来高。各バーの出来高を高値-安値レンジに均等配分する。

    戻り値: index=価格帯の中心、values=出来高。計算できなければ空Series。
    """
    if "Volume" not in df.columns or df["Volume"].sum() <= 0:
        return pd.Series(dtype=float)
    lo, hi = float(df["Low"].min()), float(df["High"].max())
    if hi <= lo:
        return pd.Series(dtype=float)
    edges = np.linspace(lo, hi, bins + 1)
    profile = np.zeros(bins)
    l_arr, h_arr = df["Low"].to_numpy(float), df["High"].to_numpy(float)
    v_arr = df["Volume"].to_numpy(float)
    for i in range(len(df)):
        if v_arr[i] <= 0:
            continue
        bar_lo, bar_hi = l_arr[i], max(h_arr[i], l_arr[i] + 1e-9)
        overlap = np.clip(
            (np.minimum(edges[1:], bar_hi) - np.maximum(edges[:-1], bar_lo))
            / (bar_hi - bar_lo), 0, None)
        profile += v_arr[i] * overlap
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.Series(profile, index=centers)


def _volume_nodes(df: pd.DataFrame, bins: int = 50, top: int = 5) -> list[float]:
    """出来高が特に集中している価格帯の中心。"""
    prof = volume_profile(df, bins)
    if prof.empty:
        return []
    return [float(v) for v in prof.nlargest(top).index]


def _zone_volume_share(df: pd.DataFrame, zlo: float, zhi: float) -> float:
    """ゾーン内で商われた出来高が全体に占める割合。"""
    if "Volume" not in df.columns:
        return 0.0
    lo = df["Low"].to_numpy(float)
    hi = np.maximum(df["High"].to_numpy(float), lo + 1e-9)
    vol = df["Volume"].to_numpy(float)
    total = vol.sum()
    if total <= 0:
        return 0.0
    overlap = np.clip(np.minimum(hi, zhi) - np.maximum(lo, zlo), 0, None) / (hi - lo)
    return float((vol * overlap).sum() / total)


def _events(df: pd.DataFrame, zlo: float, zhi: float, atr: float,
            min_gap: int = 3) -> list[dict]:
    """ゾーンへの接触を、実体とヒゲを区別して分類する。

    outcome:
      reject … ヒゲはゾーンに入ったが実体は入らず、来た側へ戻した(最も強い反発)
      bounce … 実体がゾーンに入ったが反対側へは抜けず、来た側へ戻した
      break  … 反対側へ実体で抜けた
    """
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    lo_ = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    n = len(df)
    if "VOL_MA20" in df.columns and "Volume" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            vr = np.nan_to_num(df["Volume"].to_numpy(float)
                               / df["VOL_MA20"].to_numpy(float), nan=1.0, posinf=1.0)
    else:
        vr = np.ones(n)

    out_up = zhi + 0.25 * atr
    out_dn = zlo - 0.25 * atr

    events: list[dict] = []
    i = 1
    while i < n:
        if not (lo_[i] <= zhi and h[i] >= zlo):
            i += 1
            continue
        # 直前の位置で接近方向を決める
        side = "support" if c[i - 1] > zhi else ("resistance" if c[i - 1] < zlo else None)
        if side is None:
            i += 1
            continue

        body_in = False
        vmax = vr[i]
        j = i
        outcome = None
        while j < n:
            body_lo, body_hi = min(o[j], c[j]), max(o[j], c[j])
            if body_lo <= zhi and body_hi >= zlo:
                body_in = True
            vmax = max(vmax, vr[j])
            if c[j] > out_up:
                outcome = "break" if side == "resistance" else (
                    "bounce" if body_in else "reject")
                break
            if c[j] < out_dn:
                outcome = "break" if side == "support" else (
                    "bounce" if body_in else "reject")
                break
            j += 1
        if outcome is None:
            break
        events.append({"end": j, "side": side, "outcome": outcome, "vol": vmax})
        i = j + min_gap
    return events


class _ConfluenceCtx:
    """フィボナッチ・キリ番・SMA・ピボットの照合用コンテキスト。"""

    def __init__(self, df: pd.DataFrame, current: float):
        hi, lo = float(df["High"].max()), float(df["Low"].min())
        rng = hi - lo
        self.fib = {f"フィボ{r * 100:.1f}%": hi - rng * r for r in FIB_RATIOS} if rng > 0 else {}
        self.sma = {}
        for name in ("SMA50", "SMA200"):
            if name in df.columns and pd.notna(df[name].iloc[-1]):
                self.sma[name] = float(df[name].iloc[-1])
        piv = pivot_points(df) or {}
        self.piv = {f"ピボット{k.split('(')[0]}": v for k, v in piv.items()}
        self.steps = [s for s in ROUND_STEPS
                      if current * 0.004 <= s <= current * 0.15]

    def tags(self, level: float, tol: float) -> list[str]:
        found = []
        for name, v in {**self.fib, **self.sma, **self.piv}.items():
            if abs(level - v) <= tol:
                found.append(name)
        for step in self.steps:
            r = round(level / step) * step
            if abs(level - r) <= tol * 0.6:
                found.append(f"キリ番${r:,.0f}")
                break
        return found[:2]


def find_levels(df: pd.DataFrame, max_per_side: int = 4) -> list[dict]:
    """サポート/レジスタンスのゾーン一覧を強度順に返す。

    各レベル: {type, price, zone_low, zone_high, distance_pct, bounces, breaks,
               touches, rejects, bounce_rate, raw_score, strength, vol_share,
               last_touch, basis, confluence, last_event, swings}
    """
    if df is None or len(df) < 20:
        return []

    n = len(df)
    current = float(df["Close"].iloc[-1])
    atr = float(_atr(df).iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return []
    tol = max(atr * 0.5, current * 0.003)
    skip_band = current * 0.002
    window = max(2, min(5, n // 15))
    half_life = max(20, n // 4)
    conf = _ConfluenceCtx(df, current)
    prof = volume_profile(df, 50)
    nodes = [float(v) for v in prof.nlargest(5).index] if not prof.empty else []
    dates = df.index

    raw: list[dict] = []

    def build(center, zone_lo, zone_hi, basis, swings, seed_last, seed_score):
        if abs(center - current) <= skip_band:
            return None
        zone_lo = min(zone_lo, center - atr * 0.25)
        zone_hi = max(zone_hi, center + atr * 0.25)

        evs = _events(df, zone_lo, zone_hi, atr)
        w_pos = w_neg = 0.0
        rejects = bounces = breaks = 0
        role_sides = set()
        last_i = None
        for e in evs:
            w = 0.5 ** ((n - 1 - e["end"]) / half_life)
            vf = min(max(e["vol"], 0.5), 2.0)
            if e["outcome"] == "reject":
                w_pos += w * vf * 1.4   # ヒゲ拒絶は最も強い反発の証拠
                rejects += 1
                role_sides.add(e["side"])
            elif e["outcome"] == "bounce":
                w_pos += w * vf
                bounces += 1
                role_sides.add(e["side"])
            else:
                w_neg += w * vf
                breaks += 1
            last_i = e["end"]

        touches = len(evs)
        good = rejects + bounces
        vol_share = _zone_volume_share(df, zone_lo, zone_hi)
        near_node = any(zone_lo <= v <= zone_hi for v in nodes)
        tags = [t for t in conf.tags(center, (zone_hi - zone_lo) / 2) if t != basis]
        if len(role_sides) == 2:
            tags = ["役割転換"] + tags

        # 直近に実体で破られたレベルは、同じ役割としては弱い
        fresh_break = 0.0
        if evs and evs[-1]["outcome"] == "break":
            bars = n - 1 - evs[-1]["end"]
            fresh_break = max(0.0, 1.0 - bars / 20)

        score = (seed_score
                 + 0.9 * np.tanh(w_pos / 2.0)
                 + 0.5 * (good / touches if touches else 0.0)
                 + 0.6 * min(vol_share / 0.10, 1.0)
                 + 0.25 * min(swings / 3, 1.0)
                 + 0.20 * min(len(tags), 2)
                 + (0.15 if near_node else 0.0)
                 - 0.7 * np.tanh(w_neg / 2.0)
                 - 0.6 * fresh_break)

        return {
            "type": "抵抗線" if center > current else "サポート",
            "price": center, "zone_low": zone_lo, "zone_high": zone_hi,
            "distance_pct": (center / current - 1) * 100,
            "bounces": good, "breaks": breaks, "touches": touches,
            "rejects": rejects, "swings": swings,
            "bounce_rate": good / touches * 100 if touches else None,
            "raw_score": float(score),
            "vol_share": vol_share,
            "last_touch": str(dates[last_i])[:10] if last_i is not None else seed_last,
            "basis": basis + ("+出来高集中" if near_node and "出来高" not in basis else ""),
            "confluence": tags,
            "last_event": ({"outcome": evs[-1]["outcome"], "side": evs[-1]["side"],
                            "bars_ago": n - 1 - evs[-1]["end"]} if evs else None),
        }

    # 1) スイングのクラスタ
    points = sorted(_swing_points(df, window), key=lambda x: x[1])
    clusters: list[list[tuple]] = []
    for pt in points:
        if clusters and pt[1] - clusters[-1][-1][1] <= tol:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])
    for cl in clusters:
        prices = [p for _, p, _ in cl]
        lv = build(float(pd.Series(prices).median()), min(prices), max(prices),
                   "スイング", len(cl), str(max(i for i, _, _ in cl))[:10], 0.25)
        if lv:
            raw.append(lv)

    # 2) 期間高値/安値
    for price, date in ((float(df["High"].max()), str(df["High"].idxmax())[:10]),
                        (float(df["Low"].min()), str(df["Low"].idxmin())[:10])):
        if any(abs(price - lv["price"]) <= tol for lv in raw):
            continue
        lv = build(price, price, price,
                   "期間高値" if price > current else "期間安値", 0, date, 0.45)
        if lv:
            raw.append(lv)

    # 3) 出来高集中帯
    for v in nodes:
        if any(abs(v - lv["price"]) <= tol for lv in raw):
            continue
        lv = build(float(v), v, v, "出来高集中帯", 0, "—", 0.30)
        if lv:
            raw.append(lv)

    # 4) 未テストの節目候補で補完
    hi, lo = float(df["High"].max()), float(df["Low"].min())
    rng_ = hi - lo
    fwd: dict[str, float] = {}
    if rng_ > 0:
        fwd.update({f"フィボ拡張{r * 100:.0f}%": lo + rng_ * r for r in (1.272, 1.618)})
    fwd.update({f"ピボット{k}": v for k, v in (pivot_points(df) or {}).items()
                if not k.startswith("P")})
    if conf.steps:
        step = conf.steps[0]
        base = round(current / step) * step
        for k in range(-3, 4):
            v = base + k * step
            if v > 0:
                fwd[f"キリ番${v:,.0f}"] = float(v)
    for side_type, sign in (("抵抗線", 1), ("サポート", -1)):
        need = 2 - sum(1 for l in raw if l["type"] == side_type)
        if need <= 0:
            continue
        cands = sorted([(nm, v) for nm, v in fwd.items()
                        if (v - current) * sign > skip_band],
                       key=lambda x: abs(x[1] - current))
        added = 0
        for nm, v in cands:
            if added >= need:
                break
            if any(abs(v - l["price"]) <= tol for l in raw):
                continue
            lv = build(v, v, v, nm, 0, "—", 0.10)
            if lv:
                raw.append(lv)
                added += 1

    # 素点 → ★1〜5(固定境界で較正済み)
    for lv in raw:
        s = lv["raw_score"]
        lv["strength"] = 1 + sum(1 for cut in STRENGTH_CUTS if s >= cut)

    def _pick(side: list[dict]) -> list[dict]:
        if not side:
            return []
        chosen = sorted(side, key=lambda l: (-l["raw_score"],
                                             abs(l["distance_pct"])))[:max_per_side]
        nearest = min(side, key=lambda l: abs(l["distance_pct"]))
        if nearest not in chosen:
            chosen = chosen[:-1] + [nearest]
        return chosen

    res = _pick([l for l in raw if l["type"] == "抵抗線"])
    sup = _pick([l for l in raw if l["type"] == "サポート"])
    return sorted(res + sup, key=lambda l: -l["price"])


def merge_mtf(levels: list[dict], htf_levels: list[dict], label: str) -> list[dict]:
    """上位足のレベルをマージする。

    一致するレベルには「{label}合流」タグ+強さ加点、未カバーの強い上位足レベル
    (★3以上)は各サイド最大2本まで追加する。
    """
    if not htf_levels:
        return levels

    def near(a: dict, b: dict) -> bool:
        band = max(b["zone_high"] - b["zone_low"], a["zone_high"] - a["zone_low"]) / 2
        return abs(a["price"] - b["price"]) <= max(band, a["price"] * 0.003)

    tag = f"{label}合流"
    for lv in levels:
        if any(near(lv, h) for h in htf_levels):
            lv["confluence"] = ([tag] + [t for t in lv["confluence"] if t != tag])[:3]
            lv["strength"] = min(5, lv["strength"] + 1)

    added = {"抵抗線": 0, "サポート": 0}
    out = list(levels)
    for h in sorted(htf_levels, key=lambda x: -x["strength"]):
        if h["strength"] < 3 or added[h["type"]] >= 2:
            continue
        if any(near(h, lv) for lv in out):
            continue
        clone = dict(h)
        clone["basis"] = f"{label}: {h['basis']}"
        clone["confluence"] = list(h["confluence"])[:2]
        out.append(clone)
        added[h["type"]] += 1
    return sorted(out, key=lambda l: -l["price"])


def level_alerts(levels: list[dict], current: float) -> list[tuple[str, str]]:
    """現在地に関するアラート(kind, メッセージ)を返す。

    kind: "testing"(ゾーン内でテスト中) / "break"(直近でブレイク)
    """
    alerts = []
    for lv in levels:
        rate = (f"反発{lv['bounces']}/{lv['touches']}回" if lv["touches"]
                else "テスト実績なし")
        if lv["zone_low"] <= current <= lv["zone_high"]:
            alerts.append(("testing",
                           f"現在、{lv['type']} ${lv['price']:,.2f} のゾーン内で"
                           f"攻防中です({rate}、強さ{'★' * lv['strength']})。"))
        le = lv.get("last_event")
        if le and le["outcome"] == "break" and le["bars_ago"] <= 5:
            direction = "上抜け" if current > lv["price"] else "下抜け"
            alerts.append(("break",
                           f"{le['bars_ago']}本前に {lv['type']} ${lv['price']:,.2f} を"
                           f"{direction}しました(役割転換に注意)。"))
    return alerts[:3]


def nearest_levels(levels: list[dict]) -> tuple[dict | None, dict | None]:
    """(直近の抵抗線, 直近のサポート)を返す。"""
    res = [l for l in levels if l["type"] == "抵抗線"]
    sup = [l for l in levels if l["type"] == "サポート"]
    nearest_r = min(res, key=lambda l: l["price"]) if res else None
    nearest_s = max(sup, key=lambda l: l["price"]) if sup else None
    return nearest_r, nearest_s


def pivot_points(df: pd.DataFrame) -> dict | None:
    """クラシック・ピボットポイント(直近の完結した1本から算出)。"""
    if df is None or len(df) < 2:
        return None
    prev = df.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    p = (h + l + c) / 3
    return {
        "P(ピボット)": p,
        "R1": 2 * p - l, "R2": p + (h - l), "R3": h + 2 * (p - l),
        "S1": 2 * p - h, "S2": p - (h - l), "S3": l - 2 * (h - p),
    }


def _level_note(lv: dict) -> str:
    rate = (f"反発{lv['bounces']}/{lv['touches']}回({lv['bounce_rate']:.0f}%)"
            if lv["touches"] else "テスト実績なし")
    return f"{lv['type']} ${lv['price']:,.2f}(強さ★{lv['strength']}、{rate})"


def suggest_limit_orders(df: pd.DataFrame, levels: list[dict],
                         risk_amount: float | None = None) -> list[dict]:
    """サポート/レジスタンスから参考指値(買い/売り候補)を機械的に算出する。

    risk_amount(1トレードの許容損失額、ドル)を渡すと、損切り幅から
    推奨株数と想定利益も計算する。
    """
    if df is None or df.empty or not levels:
        return []
    current = float(df["Close"].iloc[-1])
    atr = float(_atr(df).iloc[-1])
    res = sorted([l for l in levels if l["type"] == "抵抗線"], key=lambda l: l["price"])
    sup = sorted([l for l in levels if l["type"] == "サポート"],
                 key=lambda l: -l["price"])

    suggestions = []

    def buy_row(label: str, lv: dict) -> dict:
        entry = lv["zone_high"]
        stop = lv["zone_low"] - atr * 0.5
        target = res[0]["zone_low"] if res else current + 2 * (entry - stop)
        rr = (target - entry) / (entry - stop) if entry > stop and target > entry else None
        shares = est_profit = None
        if risk_amount and entry > stop:
            shares = int(risk_amount // (entry - stop))
            if shares > 0 and target > entry:
                est_profit = shares * (target - entry)
        return {
            "scenario": label, "side": "買い", "price": entry,
            "distance_pct": (entry / current - 1) * 100,
            "stop": stop, "target": target, "rr": rr,
            "shares": shares, "est_profit": est_profit,
            "basis": _level_note(lv),
        }

    if sup:
        suggestions.append(buy_row("押し目買い(直近サポート)", sup[0]))
        deeper = [l for l in sup[1:] if l["strength"] > sup[0]["strength"]]
        if deeper:
            best = max(deeper, key=lambda l: l["strength"])
            suggestions.append(buy_row("押し目買い(強サポート・深押し)", best))

    if res:
        r = res[0]
        suggestions.append({
            "scenario": "利確売り(直近抵抗)", "side": "売り", "price": r["zone_low"],
            "distance_pct": (r["zone_low"] / current - 1) * 100,
            "stop": None, "target": None, "rr": None,
            "shares": None, "est_profit": None, "basis": _level_note(r),
        })
    return suggestions
