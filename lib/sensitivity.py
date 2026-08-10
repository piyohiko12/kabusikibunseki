"""ファンダメンタルイベント・マクロ要因への感応度評価。

- 決算発表・FOMC: 実際のイベント日の株価変動を平常時と比較(イベントスタディ)
- マクロ要因: 市場指数・金利・ドル・原油・VIXの日次リターンとの相関/ベータ
"""

import pandas as pd

from lib import data_fetcher

# FOMC政策金利決定日(公表スケジュール、各会合2日目)
FOMC_DATES = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]

SECTOR_ETF = {
    "Technology": ("XLK", "テクノロジー株指数(XLK)"),
    "Communication Services": ("XLC", "通信サービス株指数(XLC)"),
    "Consumer Cyclical": ("XLY", "一般消費財株指数(XLY)"),
    "Consumer Defensive": ("XLP", "生活必需品株指数(XLP)"),
    "Energy": ("XLE", "エネルギー株指数(XLE)"),
    "Financial Services": ("XLF", "金融株指数(XLF)"),
    "Healthcare": ("XLV", "ヘルスケア株指数(XLV)"),
    "Industrials": ("XLI", "資本財株指数(XLI)"),
    "Basic Materials": ("XLB", "素材株指数(XLB)"),
    "Real Estate": ("XLRE", "不動産株指数(XLRE)"),
    "Utilities": ("XLU", "公益株指数(XLU)"),
}


def _naive_daily(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """タイムゾーンを外して日付に正規化する(naiveな指数にも対応)。"""
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize()


def _stars_from_ratio(ratio: float) -> int:
    for th, stars in ((4.0, 5), (3.0, 4), (2.0, 3), (1.3, 2)):
        if ratio >= th:
            return stars
    return 1


def _stars_from_corr(corr: float) -> int:
    a = abs(corr)
    for th, stars in ((0.7, 5), (0.5, 4), (0.3, 3), (0.15, 2)):
        if a >= th:
            return stars
    return 1


def _event_moves(hist: pd.DataFrame, dates: list[str]) -> list[float]:
    """各イベント日直後の日次変動率(%、符号つき)。

    イベント当日と翌営業日のうち、値動きの大きかったほうを符号つきで返す。
    絶対値ではなく符号を残すことで、上下どちらに振れやすいかも測れる。
    """
    ret = hist["Close"].pct_change() * 100
    ret = pd.Series(ret.values, index=_naive_daily(hist.index))
    moves = []
    for d in dates:
        ts = pd.Timestamp(d)
        after = ret.loc[ret.index >= ts]
        if len(after) >= 2 and (after.index[0] - ts).days <= 5:
            pair = [float(after.iloc[0]), float(after.iloc[1])]
            moves.append(max(pair, key=abs))
    return moves


def _direction(moves: list[float]) -> dict:
    """イベント後の値動きに上下の偏りがあるかを判定する。

    n が小さいと偏りは簡単に出るので、二項分布の目安として
    「偏りと呼ぶには 65%以上 かつ 5回以上」を条件にする。
    """
    n = len(moves)
    ups = sum(1 for m in moves if m > 0)
    up_rate = ups / n * 100 if n else 0.0
    avg_signed = sum(moves) / n if n else 0.0
    up_moves = [m for m in moves if m > 0]
    down_moves = [m for m in moves if m < 0]
    biased = n >= 5 and (up_rate >= 65 or up_rate <= 35)
    if not biased:
        label, tone = "上下どちらとも言えない", "flat"
    elif up_rate >= 65:
        label, tone = f"上に振れやすい({up_rate:.0f}%)", "up"
    else:
        label, tone = f"下に振れやすい({100 - up_rate:.0f}%)", "down"
    return {
        "n": n,
        "up_rate": up_rate,
        "avg_signed": avg_signed,
        "avg_up": sum(up_moves) / len(up_moves) if up_moves else 0.0,
        "avg_down": sum(down_moves) / len(down_moves) if down_moves else 0.0,
        "biased": biased,
        "direction_label": label,
        "direction_tone": tone,
    }


def _event_row(name: str, hist: pd.DataFrame, dates: list[str],
               base_move: float, note: str) -> dict | None:
    moves = _event_moves(hist, dates)
    if len(moves) < 3:
        return None
    avg_abs = sum(abs(m) for m in moves) / len(moves)
    ratio = avg_abs / base_move if base_move else 0
    d = _direction(moves)
    return {
        "イベント・要因": name,
        "感応度": "★" * _stars_from_ratio(ratio),
        "実測値": f"平均±{avg_abs:.1f}%(平常時の{ratio:.1f}倍)",
        "方向": d["direction_label"],
        "上昇時/下落時": f"+{d['avg_up']:.1f}% / {d['avg_down']:.1f}%",
        "解説": f"{note}(過去{len(moves)}回)",
        "_detail": {**d, "avg_abs": avg_abs, "ratio": ratio, "event": True},
    }


def _factor_row(name: str, symbol: str, stock_ret: pd.Series,
                notes: tuple[str, str, str], with_beta: bool) -> dict | None:
    """notes = (正の相関が強いとき, 負の相関が強いとき, 相関が弱いとき)"""
    try:
        f_hist = data_fetcher.fetch_history(symbol, "2y")
    except data_fetcher.FetchError:
        return None
    if f_hist.empty:
        return None
    f_ret = f_hist["Close"].pct_change()
    f_ret.index = _naive_daily(f_ret.index)
    pair = pd.concat([stock_ret, f_ret], axis=1, join="inner").dropna()
    if len(pair) < 60:
        return None
    corr = float(pair.corr().iloc[0, 1])
    value = f"相関 {corr:+.2f}"
    if with_beta:
        var = float(pair.iloc[:, 1].var())
        beta = float(pair.cov().iloc[0, 1]) / var if var else 0.0
        value = f"β {beta:.2f} / 相関 {corr:+.2f}"
    note = notes[2] if abs(corr) < 0.15 else (notes[0] if corr > 0 else notes[1])
    # マクロ要因は「その要因が1%動いたとき、この銘柄が平均何%動いたか」を
    # 方向として示す。相関が弱いときは方向を出さない。
    slope = float(pair.cov().iloc[0, 1] / pair.iloc[:, 1].var()) \
        if pair.iloc[:, 1].var() else 0.0
    if abs(corr) < 0.15:
        direction = "連動は弱い"
    elif corr > 0:
        direction = f"同方向(要因+1%で {slope * 1:+.2f}%)"
    else:
        direction = f"逆方向(要因+1%で {slope * 1:+.2f}%)"
    return {
        "イベント・要因": name,
        "感応度": "★" * _stars_from_corr(corr),
        "実測値": value,
        "方向": direction,
        "上昇時/下落時": "—",
        "解説": note,
        "_detail": {"corr": corr, "slope": slope, "n": int(len(pair)),
                    "event": False},
    }


def evaluate(hist: pd.DataFrame, info: dict, earnings_dates: list[str]) -> list[dict]:
    """感応度評価テーブルの行リストを返す(取得できない要因はスキップ)。"""
    if hist is None or len(hist) < 120:
        return []
    stock_ret = hist["Close"].pct_change()
    stock_ret.index = _naive_daily(stock_ret.index)
    base_move = float(stock_ret.abs().median() * 100)

    rows = []
    row = _event_row("決算発表", hist, earnings_dates, base_move,
                     "決算直後の株価変動の大きさ")
    if row:
        rows.append(row)
    row = _event_row("FOMC(米金融政策)", hist, FOMC_DATES, base_move,
                     "政策金利決定日の変動の大きさ")
    if row:
        rows.append(row)

    factors = [
        ("市場全体(S&P500)", "^GSPC", True,
         ("市場と同方向に動く(ベータが感応度)", "市場と逆方向に動く傾向", "市場との連動は弱い")),
        ("長期金利(米10年債)", "^TNX", False,
         ("金利上昇時に上がりやすい", "金利上昇時に下がりやすい(グロース株的)", "金利の影響は限定的")),
        ("ドル指数", "DX-Y.NYB", False,
         ("ドル高で上がりやすい", "ドル高で下がりやすい(海外売上比率の影響など)", "為替の影響は限定的")),
        ("原油価格", "CL=F", False,
         ("原油高で上がりやすい", "原油高で下がりやすい(コスト増要因)", "原油の影響は限定的")),
        ("リスクオフ(VIX)", "^VIX", False,
         ("市場不安時に買われる(ディフェンシブ)", "市場不安時に売られやすい", "VIXとの連動は弱い")),
    ]
    sector = info.get("sector")
    if sector in SECTOR_ETF:
        sym, label = SECTOR_ETF[sector]
        factors.insert(1, (f"セクター({label})", sym, True,
                           ("セクター全体と連動して動く", "セクターと逆行する傾向", "セクターとの連動は弱い")))

    for name, symbol, with_beta, notes in factors:
        row = _factor_row(name, symbol, stock_ret, notes, with_beta)
        if row:
            rows.append(row)
    return rows
