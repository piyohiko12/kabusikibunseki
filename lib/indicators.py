"""テクニカル指標の計算(pandasのみで実装)。"""

import pandas as pd

SMA_WINDOWS = (20, 50, 200)
EMA_SPANS = (20, 50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標の列を追加したコピーを返す。

    SMA(20/50/200)、EMA(20/50)、ボリンジャーバンド(20, ±2σ)、RSI(14)、
    MACD(12,26,9)、スローストキャスティクス(14,3,3)、一目均衡表(9,26,52)、
    出来高20日平均。
    """
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    for w in SMA_WINDOWS:
        out[f"SMA{w}"] = close.rolling(w).mean()
    for s in EMA_SPANS:
        out[f"EMA{s}"] = close.ewm(span=s, adjust=False).mean()

    # ボリンジャーバンド(20期間, ±2σ)
    std20 = close.rolling(20).std()
    out["BB_up"] = out["SMA20"] + 2 * std20
    out["BB_low"] = out["SMA20"] - 2 * std20

    # RSI(14): Wilder方式(初期14本はNaN)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    out["RSI"] = 100 - 100 / (1 + gain / loss)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # スローストキャスティクス(14, 3, 3)
    ll14 = low.rolling(14).min()
    hh14 = high.rolling(14).max()
    fast_k = (close - ll14) / (hh14 - ll14) * 100
    out["STOCH_K"] = fast_k.rolling(3).mean()
    out["STOCH_D"] = out["STOCH_K"].rolling(3).mean()

    # 一目均衡表(9, 26, 52)。先行スパンは26期間先行(表示は既存日付範囲内)
    out["ICHI_TENKAN"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["ICHI_KIJUN"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["ICHI_SPAN_A"] = ((out["ICHI_TENKAN"] + out["ICHI_KIJUN"]) / 2).shift(26)
    out["ICHI_SPAN_B"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    out["ICHI_CHIKOU"] = close.shift(-26)

    if "Volume" in out.columns:
        out["VOL_MA20"] = out["Volume"].rolling(20).mean()
        # VWAP(日ごとにリセット)。分足・時間足での利用を想定
        tp = (out["High"] + out["Low"] + out["Close"]) / 3
        day = out.index.date
        pv_cum = (tp * out["Volume"]).groupby(day).cumsum()
        vol_cum = out["Volume"].groupby(day).cumsum()
        out["VWAP"] = pv_cum / vol_cum.replace(0, pd.NA)

    return out


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """平均足(Heikin-Ashi)のOHLCを計算する。"""
    ha = pd.DataFrame(index=df.index)
    ha["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
    opens = [float(df["Open"].iloc[0])]
    ha_close = ha["Close"].to_numpy()
    for i in range(1, len(df)):
        opens.append((opens[-1] + float(ha_close[i - 1])) / 2)
    ha["Open"] = opens
    ha["High"] = pd.concat([df["High"], ha["Open"], ha["Close"]], axis=1).max(axis=1)
    ha["Low"] = pd.concat([df["Low"], ha["Open"], ha["Close"]], axis=1).min(axis=1)
    return ha


def slice_display(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """指標計算済みのDataFrameから直近days日分を切り出す。"""
    if df.empty:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df.loc[df.index >= cutoff]
