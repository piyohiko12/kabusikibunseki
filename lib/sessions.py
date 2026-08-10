"""米国株の取引セッション(プレ・立会・アフター・夜間)の判定と集計。

米国東部時間(ET)を基準に、1本のバーがどのセッションに属するかを決める。
夏時間はZoneInfoが自動で扱うので、コード側でオフセットを持たない。

    夜間(overnight)  20:00〜04:00 ET  … 24時間取引・OTC延長。取扱いは証券会社次第
    プレ(pre)         04:00〜09:30 ET
    立会(regular)     09:30〜16:00 ET
    アフター(after)   16:00〜20:00 ET

yfinanceは prepost=True で pre/after を返すが、夜間まで返すことはほぼない。
夜間の値は moomoo など別ソースで補う想定で、区分だけ先に用意しておく。
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

# セッションID → (表示名, 開始時刻, 終了時刻, チャートの背景色)
# 時刻は「その日のET」での境界。overnightだけ日をまたぐので別扱いにする。
PRE = "pre"
REGULAR = "regular"
AFTER = "after"
OVERNIGHT = "overnight"

SESSION_LABELS = {
    PRE: "プレマーケット",
    REGULAR: "立会",
    AFTER: "アフターマーケット",
    OVERNIGHT: "夜間",
}

SESSION_SHORT = {
    PRE: "プレ",
    REGULAR: "立会",
    AFTER: "アフター",
    OVERNIGHT: "夜間",
}

# 境界(ET・時間の小数表現)。09:30 = 9.5
BOUNDS = {
    PRE: (4.0, 9.5),
    REGULAR: (9.5, 16.0),
    AFTER: (16.0, 20.0),
}

# チャートの背景色(薄いほうがローソク足を邪魔しない)
SESSION_BG = {
    PRE: "rgba(80,140,255,0.07)",
    AFTER: "rgba(255,170,60,0.07)",
    OVERNIGHT: "rgba(150,150,170,0.10)",
}

# 立会以外をまとめて指すときの並び順(UIのタブ順と合わせる)
EXTENDED = (PRE, AFTER, OVERNIGHT)
ALL_SESSIONS = (OVERNIGHT, PRE, REGULAR, AFTER)


def to_et(index) -> pd.DatetimeIndex:
    """任意のDatetimeIndexを東部時間に変換する。

    tz情報が無いものはすでにETとみなす(yfinanceの日足など)。
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        return idx.tz_localize(ET, nonexistent="shift_forward",
                               ambiguous="NaT")
    return idx.tz_convert(ET)


def _hours(idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(idx.hour + idx.minute / 60 + idx.second / 3600,
                     index=range(len(idx)))


def classify(index) -> pd.Series:
    """各バーのセッションIDを返す(indexは元のindexのまま)。

    土日は夜間扱いにする。米国株は原則休みだが、24時間取引の商品や
    データ側の都合で値が入ることがあり、立会と混ぜたくないため。
    """
    idx = to_et(index)
    hours = _hours(idx).to_numpy()
    out = pd.Series(OVERNIGHT, index=pd.Index(range(len(idx))), dtype=object)
    for name in (PRE, REGULAR, AFTER):
        lo, hi = BOUNDS[name]
        out[(hours >= lo) & (hours < hi)] = name
    weekend = pd.Series(idx.dayofweek >= 5, index=out.index)
    out[weekend] = OVERNIGHT
    out.index = pd.Index(index)
    return out


def add_session(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameに 'session' 列と 'et' 列(東部時間)を付ける。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["session"] = classify(out.index).to_numpy()
    out["et"] = to_et(out.index)
    return out


def trading_day(index) -> pd.Series:
    """バーが属する「取引日」(ETの暦日)を返す。

    20:00 ET以降の夜間バーは翌営業日の取引に向けた動きなので、翌日に寄せる。
    こうしないと、寄付前の夜間の動きが前日の集計に混ざってしまう。
    """
    idx = to_et(index)
    hours = idx.hour + idx.minute / 60
    days = pd.Series(idx.normalize().tz_localize(None), index=range(len(idx)))
    days[hours >= 20.0] = days[hours >= 20.0] + pd.Timedelta(days=1)
    days.index = pd.Index(index)
    return days


def latest_day_slice(df: pd.DataFrame) -> pd.DataFrame:
    """直近の取引日ぶんだけを切り出す。"""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    days = trading_day(df.index)
    return df[days == days.iloc[-1]]


def session_slice(df: pd.DataFrame, names) -> pd.DataFrame:
    """指定セッションのバーだけを残す。"""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    keep = set(names if not isinstance(names, str) else [names])
    return df[classify(df.index).isin(keep).to_numpy()]


def filter_sessions(df: pd.DataFrame, show_extended: bool) -> pd.DataFrame:
    """時間外を表示しない設定のとき、立会のバーだけに絞る。"""
    if show_extended:
        return df
    return session_slice(df, [REGULAR])


def session_summary(df: pd.DataFrame, prev_close: float | None = None) -> list[dict]:
    """直近取引日のセッション別の騰落をまとめる。

    各セッションの始値・終値・高値・安値・出来高と、そのセッション中の
    変化率を返す。基準は「1つ前のセッションの終値」で、最初のセッションだけ
    prev_close(前日の立会終値)を使う。
    """
    day = latest_day_slice(df)
    if day is None or day.empty:
        return []
    sess = classify(day.index)
    rows = []
    base = prev_close
    for name in ALL_SESSIONS:
        part = day[(sess == name).to_numpy()]
        if part.empty:
            continue
        open_, close = float(part["Open"].iloc[0]), float(part["Close"].iloc[-1])
        ref = base if base else open_
        rows.append({
            "session": name,
            "label": SESSION_LABELS[name],
            "open": open_,
            "close": close,
            "high": float(part["High"].max()),
            "low": float(part["Low"].min()),
            "volume": float(part["Volume"].sum()) if "Volume" in part else 0.0,
            "bars": int(len(part)),
            "change_pct": (close / ref - 1) * 100 if ref else 0.0,
            "start": part.index[0],
            "end": part.index[-1],
        })
        base = close
    return rows


def now_session(now: pd.Timestamp | None = None) -> str:
    """現在(ET)のセッションIDを返す。"""
    ts = now or pd.Timestamp.now(tz=ET)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ET)
    return str(classify(pd.DatetimeIndex([ts])).iloc[0])


def rangebreaks(interval: str, show_extended: bool) -> list[dict]:
    """Plotlyのrangebreaks(空白時間を詰める設定)を組み立てる。

    時間外を表示するときは 20:00〜04:00 だけを詰める。立会だけのときは
    従来どおり 16:00〜09:30 を詰める。
    """
    breaks = [dict(bounds=["sat", "mon"])]
    if interval not in INTRADAY:
        return breaks
    if show_extended:
        breaks.append(dict(bounds=[20, 4], pattern="hour"))
    else:
        breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
    return breaks


INTRADAY = ("1m", "2m", "5m", "15m", "30m", "1h", "60m", "90m")
