"""テクニカル指標の計算(pandasのみで実装)。"""

import pandas as pd

SMA_WINDOWS = (20, 50, 200)
EMA_SPANS = (20, 50)

DEFAULT_PARAMS = {
    "sma_periods": SMA_WINDOWS,
    "ema_periods": EMA_SPANS,
    "boll_period": 20,
    "boll_std": 2.0,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "stoch_period": 14,
    "stoch_k": 3,
    "stoch_d": 3,
    "volume_ma": 20,
}


def indicator_params(overrides: dict | None = None) -> dict:
    """チャート用指標パラメータを正規化する。"""
    params = {**DEFAULT_PARAMS, **(overrides or {})}
    params["sma_periods"] = tuple(max(1, int(v)) for v in params["sma_periods"])
    params["ema_periods"] = tuple(max(1, int(v)) for v in params["ema_periods"])
    for key in ("boll_period", "rsi_period", "macd_fast", "macd_slow",
                "macd_signal", "stoch_period", "stoch_k", "stoch_d",
                "volume_ma"):
        params[key] = max(1, int(params[key]))
    params["boll_std"] = max(0.1, float(params["boll_std"]))
    return params


def add_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """テクニカル指標の列を追加したコピーを返す。

    既定はSMA(20/50/200)、EMA(20/50)、ボリンジャーバンド(20, ±2σ)、
    RSI(14)、MACD(12,26,9)、スローストキャスティクス(14,3,3)、
    一目均衡表(9,26,52)、出来高20日平均。paramsで期間を変更できる。
    """
    cfg = indicator_params(params)
    out = df.copy()
    close, high, low = out["Close"], out["High"], out["Low"]

    for w in dict.fromkeys(cfg["sma_periods"]):
        out[f"SMA{w}"] = close.rolling(w).mean()
    for s in dict.fromkeys(cfg["ema_periods"]):
        out[f"EMA{s}"] = close.ewm(span=s, adjust=False).mean()

    # ボリンジャーバンド
    boll_mid = close.rolling(cfg["boll_period"]).mean()
    boll_std = close.rolling(cfg["boll_period"]).std()
    out["BB_mid"] = boll_mid
    out["BB_up"] = boll_mid + cfg["boll_std"] * boll_std
    out["BB_low"] = boll_mid - cfg["boll_std"] * boll_std

    # RSI: Wilder方式
    rsi_period = cfg["rsi_period"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    out["RSI"] = 100 - 100 / (1 + gain / loss)

    ema_fast = close.ewm(span=cfg["macd_fast"], adjust=False).mean()
    ema_slow = close.ewm(span=cfg["macd_slow"], adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_signal"] = out["MACD"].ewm(
        span=cfg["macd_signal"], adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # スローストキャスティクス
    lowest = low.rolling(cfg["stoch_period"]).min()
    highest = high.rolling(cfg["stoch_period"]).max()
    fast_k = (close - lowest) / (highest - lowest) * 100
    out["STOCH_K"] = fast_k.rolling(cfg["stoch_k"]).mean()
    out["STOCH_D"] = out["STOCH_K"].rolling(cfg["stoch_d"]).mean()

    # 一目均衡表(9, 26, 52)。先行スパンは26期間先行(表示は既存日付範囲内)
    out["ICHI_TENKAN"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["ICHI_KIJUN"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["ICHI_SPAN_A"] = ((out["ICHI_TENKAN"] + out["ICHI_KIJUN"]) / 2).shift(26)
    out["ICHI_SPAN_B"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    out["ICHI_CHIKOU"] = close.shift(-26)

    if "Volume" in out.columns:
        out["VOL_MA"] = out["Volume"].rolling(cfg["volume_ma"]).mean()
        # 既存ページとの後方互換性
        if cfg["volume_ma"] == 20:
            out["VOL_MA20"] = out["VOL_MA"]
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
