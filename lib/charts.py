"""plotlyチャートの生成。

配色は検証済みのカテゴリカルパレット(dataviz参照パレット)に従う。
ローソク足・出来高・MACDヒストグラムは上昇/下降のステータス色(good/critical)、
重ね描きする線にはステータス色と衝突しないスロットを割り当てる。
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib import indicators
from lib import levels as levels_mod
from lib import sessions

UP = "#00a86b"        # 上昇(status: good)
DOWN = "#ef5350"      # 下降(status: critical)
PRIMARY = "#2a78d6"   # 単一系列の既定色(categorical slot 1)
SECONDARY = "#1baf7a" # 2つ目の単一系列(categorical slot 2)
MUTED = "#898781"     # 基準線・補助要素
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

CHART_THEMES = {
    "ダーク": {
        "paper": "#111827", "plot": "#111827", "text": "#dbe3ef",
        "grid": "#263244", "muted": "#8390a5", "panel": "#182233",
    },
    "ライト": {
        "paper": "#ffffff", "plot": "#ffffff", "text": "#263244",
        "grid": "#e4e8ef", "muted": "#7c8799", "panel": "#f5f7fa",
    },
}

SMA_COLORS = ("#2a78d6", "#eda100", "#9b7de3")
EMA_COLORS = ("#00b894", "#e87ba4")

INTRADAY_INTERVALS = ("1m", "5m", "15m", "1h")

OSC_TITLES = {
    "出来高": "出来高",
    "RSI": "RSI(14)",
    "MACD": "MACD(12, 26, 9)",
    "ストキャスティクス": "ストキャスティクス(14, 3, 3)",
}

DEFAULT_OPTS = {
    "chart_type": "ローソク足",
    "interval": "1d",
    "overlays": ["移動平均線(SMA)"],
    "oscillators": ["出来高", "RSI", "MACD"],
    "events": True,
    "log_scale": False,
    "indicator_params": indicators.DEFAULT_PARAMS,
    "theme": "ダーク",
    "color_scheme": "緑上昇 / 赤下落",
    "grid": True,
    "range_slider": False,
    "range_selector": True,
    "current_price_line": True,
    "signals": False,
    "compact_sessions": True,
    "extended_hours": False,
    "interaction": "クロスヘア",
    "current_price": None,
}


def _price_colors(opts: dict) -> tuple[str, str]:
    if opts.get("color_scheme") == "赤上昇 / 緑下落":
        return "#ef5350", "#00a86b"
    return UP, DOWN


def _last_name(label: str, series: pd.Series) -> str:
    values = series.dropna()
    return label if values.empty else f"{label}  {values.iloc[-1]:,.2f}"


def _add_price_traces(fig, df, ticker, chart_type, up_color, down_color):
    if chart_type == "平均足":
        ha = indicators.heikin_ashi(df)
        fig.add_trace(go.Candlestick(
            x=ha.index, open=ha["Open"], high=ha["High"],
            low=ha["Low"], close=ha["Close"], name=f"{ticker}(平均足)",
            increasing_line_color=up_color, increasing_fillcolor=up_color,
            decreasing_line_color=down_color, decreasing_fillcolor=down_color,
            showlegend=False,
        ), row=1, col=1)
    elif chart_type == "ライン":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name=ticker,
            line=dict(color="#4c9aff", width=2), showlegend=False,
            hovertemplate="%{x}<br>Close %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
    elif chart_type == "エリア":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name=ticker,
            line=dict(color="#4c9aff", width=2),
            fill="tozeroy", fillcolor="rgba(76,154,255,0.16)",
            showlegend=False,
            hovertemplate="%{x}<br>Close %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
    elif chart_type == "OHLCバー":
        fig.add_trace(go.Ohlc(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker,
            increasing_line_color=up_color, decreasing_line_color=down_color,
            showlegend=False,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker,
            increasing_line_color=up_color, increasing_fillcolor=up_color,
            decreasing_line_color=down_color, decreasing_fillcolor=down_color,
            showlegend=False,
        ), row=1, col=1)


def _add_overlays(fig, df, overlays, opts):
    params = indicators.indicator_params(opts.get("indicator_params"))
    if "移動平均線(SMA)" in overlays:
        for i, period in enumerate(params["sma_periods"]):
            name = f"SMA{period}"
            if name not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df[name], name=_last_name(name, df[name]),
                line=dict(color=SMA_COLORS[i % len(SMA_COLORS)], width=1.5),
                hovertemplate=f"{name} %{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)

    if "指数移動平均線(EMA)" in overlays:
        for i, period in enumerate(params["ema_periods"]):
            name = f"EMA{period}"
            if name not in df.columns:
                continue
            fig.add_trace(go.Scatter(
                x=df.index, y=df[name], name=_last_name(name, df[name]),
                line=dict(color=EMA_COLORS[i % len(EMA_COLORS)], width=1.5,
                          dash="dash"),
                hovertemplate=f"{name} %{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)

    # 旧プリセットとの互換性
    legacy_lines = {
        "SMA20": ("#2a78d6", None), "SMA50": ("#eda100", None),
        "SMA200": ("#9b7de3", None), "EMA20": ("#00b894", "dash"),
        "EMA50": ("#e87ba4", "dash"),
    }
    for name, (color, dash) in legacy_lines.items():
        if name in overlays and name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[name], name=_last_name(name, df[name]),
                line=dict(color=color, width=1.5, dash=dash),
                hovertemplate=f"{name} %{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)

    if "VWAP(日中)" in overlays and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name=_last_name("VWAP", df["VWAP"]),
            line=dict(color="#00c2ff", width=1.7),
            hovertemplate="VWAP %{y:,.2f}<extra></extra>",
        ), row=1, col=1)

    if "ボリンジャーバンド" in overlays and "BB_up" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_up"],
            name=f"BOLL({params['boll_period']}, {params['boll_std']:g})",
            line=dict(color="#7d8ca3", width=1, dash="dot"),
            hovertemplate="BOLL上限 %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_low"], showlegend=False,
            line=dict(color="#7d8ca3", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(125,140,163,0.10)",
            hovertemplate="BOLL下限 %{y:,.2f}<extra></extra>",
        ), row=1, col=1)
        if "BB_mid" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["BB_mid"], showlegend=False,
                line=dict(color="#7d8ca3", width=1),
                hovertemplate="BOLL中心 %{y:,.2f}<extra></extra>",
            ), row=1, col=1)

    if "一目均衡表" in overlays and "ICHI_SPAN_A" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ICHI_SPAN_A"], name="一目均衡表(雲)",
            line=dict(color="rgba(137,135,129,0.5)", width=1),
            legendgroup="ichimoku", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ICHI_SPAN_B"], showlegend=False,
            line=dict(color="rgba(137,135,129,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(137,135,129,0.16)",
            legendgroup="ichimoku", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ICHI_TENKAN"], name="転換線",
            line=dict(color="#eb6834", width=1.5),
            legendgroup="ichimoku", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ICHI_KIJUN"], name="基準線",
            line=dict(color="#184f95", width=1.5),
            legendgroup="ichimoku", hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ICHI_CHIKOU"], name="遅行スパン",
            line=dict(color=MUTED, width=1, dash="dot"),
            legendgroup="ichimoku", hoverinfo="skip",
        ), row=1, col=1)


def _add_event_markers(fig, df):
    if "Dividends" in df.columns:
        div = df[df["Dividends"] > 0]
        if not div.empty:
            fig.add_trace(go.Scatter(
                x=div.index, y=div["Low"] * 0.985, mode="markers", name="配当",
                marker=dict(symbol="diamond", size=9, color="#eda100",
                            line=dict(color="#fcfcfb", width=1)),
                customdata=div["Dividends"],
                hovertemplate="配当 $%{customdata:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
            ), row=1, col=1)
    if "Stock Splits" in df.columns:
        splits = df[df["Stock Splits"] != 0]
        if not splits.empty:
            fig.add_trace(go.Scatter(
                x=splits.index, y=splits["High"] * 1.015, mode="markers", name="分割",
                marker=dict(symbol="star", size=11, color="#4a3aa7",
                            line=dict(color="#fcfcfb", width=1)),
                customdata=splits["Stock Splits"],
                hovertemplate="株式分割 %{customdata}:1<br>%{x|%Y-%m-%d}<extra></extra>",
            ), row=1, col=1)


def _add_oscillator(fig, df, name, row, opts):
    up_color, down_color = _price_colors(opts)
    params = indicators.indicator_params(opts.get("indicator_params"))
    if name == "出来高" and "Volume" in df.columns:
        colors = [up_color if c >= o else down_color
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="出来高",
            marker_color=colors, marker_line_width=0, opacity=0.55,
            showlegend=False,
        ), row=row, col=1)
        if "VOL_MA" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["VOL_MA"],
                name=f"出来高MA{params['volume_ma']}",
                line=dict(color="#f5b642", width=1.3), showlegend=False,
            ), row=row, col=1)
    elif name == "RSI":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name=_last_name(f"RSI{params['rsi_period']}", df["RSI"]),
            line=dict(color=PRIMARY, width=2), showlegend=False,
        ), row=row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)",
                      line_width=0, row=row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,168,107,0.06)",
                      line_width=0, row=row, col=1)
        for level in (30, 70):
            fig.add_hline(y=level, line=dict(color="#7d8ca3", width=1, dash="dot"),
                          row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    elif name == "MACD":
        hist_colors = [up_color if v >= 0 else down_color
                       for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"], name="ヒストグラム",
            marker_color=hist_colors, marker_line_width=0, opacity=0.5,
            showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"],
            name=_last_name("DIF", df["MACD"]),
            line=dict(color=PRIMARY, width=1.6), showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"],
            name=_last_name("DEA", df["MACD_signal"]),
            line=dict(color="#eda100", width=1.6), showlegend=False,
        ), row=row, col=1)
        fig.add_hline(y=0, line=dict(color="#7d8ca3", width=1), row=row, col=1)
    elif name == "ストキャスティクス":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["STOCH_K"], name="%K",
            line=dict(color=PRIMARY, width=2), showlegend=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["STOCH_D"], name="%D",
            line=dict(color="#eda100", width=2), showlegend=False,
        ), row=row, col=1)
        for level in (20, 80):
            fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"),
                          row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)


def _add_signal_markers(fig, df):
    """MACDクロスとRSIの閾値通過を価格上に表示する。"""
    if {"MACD", "MACD_signal"}.issubset(df.columns):
        bull = (df["MACD"] > df["MACD_signal"]) & (
            df["MACD"].shift(1) <= df["MACD_signal"].shift(1))
        bear = (df["MACD"] < df["MACD_signal"]) & (
            df["MACD"].shift(1) >= df["MACD_signal"].shift(1))
        for mask, symbol, color, label, factor in (
            (bull, "triangle-up", UP, "MACDゴールデンクロス", 0.992),
            (bear, "triangle-down", DOWN, "MACDデッドクロス", 1.008),
        ):
            points = df[mask].tail(12)
            if points.empty:
                continue
            base = points["Low"] if factor < 1 else points["High"]
            fig.add_trace(go.Scatter(
                x=points.index, y=base * factor, mode="markers",
                marker=dict(symbol=symbol, size=9, color=color),
                name=label, showlegend=False,
                hovertemplate=f"{label}<br>%{{x}}<extra></extra>",
            ), row=1, col=1)

    if "RSI" in df.columns:
        oversold = (df["RSI"] >= 30) & (df["RSI"].shift(1) < 30)
        overbought = (df["RSI"] <= 70) & (df["RSI"].shift(1) > 70)
        for mask, symbol, color, label, factor in (
            (oversold, "circle", UP, "RSIが売られすぎ圏から回復", 0.986),
            (overbought, "circle", DOWN, "RSIが買われすぎ圏から低下", 1.014),
        ):
            points = df[mask].tail(8)
            if points.empty:
                continue
            base = points["Low"] if factor < 1 else points["High"]
            fig.add_trace(go.Scatter(
                x=points.index, y=base * factor, mode="markers",
                marker=dict(symbol=symbol, size=7, color=color,
                            line=dict(color="#ffffff", width=1)),
                name=label, showlegend=False,
                hovertemplate=f"{label}<br>%{{x}}<extra></extra>",
            ), row=1, col=1)


def _shade_sessions(fig, df: pd.DataFrame, rows: int) -> None:
    """時間外セッションの区間に薄い背景を敷き、立会と一目で区別できるようにする。

    連続する同一セッションのバーをひとまとめにして矩形を1つ描く。
    バー単位で描くと図形が数百個になり描画が重くなるため。
    """
    if df is None or df.empty:
        return
    labels = sessions.classify(df.index).to_numpy()
    index = df.index
    start = 0
    labelled = set()
    for i in range(1, len(labels) + 1):
        if i < len(labels) and labels[i] == labels[start]:
            continue
        name = labels[start]
        if name in sessions.SESSION_BG:
            # 端のバーの幅ぶん外側に広げて、隣の区間と隙間ができないようにする
            x0 = index[start]
            x1 = index[i] if i < len(labels) else index[i - 1]
            fig.add_vrect(x0=x0, x1=x1, fillcolor=sessions.SESSION_BG[name],
                          line_width=0, layer="below", row="all", col=1)
            # 凡例代わりのラベルは価格パネルに1セッション1回だけ置く
            if name not in labelled:
                fig.add_annotation(
                    x=x0, y=1, xref="x", yref="y domain",
                    text=sessions.SESSION_SHORT[name], showarrow=False,
                    xanchor="left", yanchor="top",
                    font=dict(size=9, color=MUTED), row=1, col=1)
                labelled.add(name)
        start = i


def price_chart(df: pd.DataFrame, ticker: str, opts: dict | None = None) -> go.Figure:
    """メインチャート(価格+選択したオーバーレイ・サブチャート)。"""
    o = {**DEFAULT_OPTS, **(opts or {})}
    params = indicators.indicator_params(o.get("indicator_params"))
    o["indicator_params"] = params
    oscillators = [n for n in o["oscillators"] if n in OSC_TITLES]
    theme = CHART_THEMES.get(o["theme"], CHART_THEMES["ダーク"])
    up_color, down_color = _price_colors(o)

    rows = 1 + len(oscillators)
    weights = [4.2] + [1.25] * len(oscillators)
    osc_titles = {
        "出来高": f"出来高 / MA{params['volume_ma']}",
        "RSI": f"RSI({params['rsi_period']})",
        "MACD": (f"MACD({params['macd_fast']}, {params['macd_slow']}, "
                 f"{params['macd_signal']})"),
        "ストキャスティクス": (f"STOCH({params['stoch_period']}, "
                          f"{params['stoch_k']}, {params['stoch_d']})"),
    }
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[w / sum(weights) for w in weights],
        vertical_spacing=0.025,
        subplot_titles=[""] + [osc_titles[n] for n in oscillators],
    )

    _add_price_traces(fig, df, ticker, o["chart_type"], up_color, down_color)
    _add_overlays(fig, df, set(o["overlays"]), o)
    if o["events"]:
        _add_event_markers(fig, df)
    if o["signals"]:
        _add_signal_markers(fig, df)

    if "フィボナッチ" in set(o["overlays"]) and not df.empty:
        hi, lo = float(df["High"].max()), float(df["Low"].min())
        if hi > lo:
            for r in (0.236, 0.382, 0.5, 0.618, 0.786):
                y = hi - (hi - lo) * r
                fig.add_hline(
                    y=y, line=dict(color="#9085e9", width=1, dash="dot"),
                    opacity=0.8, row=1, col=1,
                    annotation_text=f"Fib {r * 100:.1f}%",
                    annotation_position="right",
                    annotation_font=dict(color="#9085e9", size=9),
                )

    if "出来高プロファイル" in set(o["overlays"]):
        prof = levels_mod.volume_profile(df)
        if not prof.empty and prof.max() > 0:
            share = prof / prof.sum() * 100
            # サブプロットはxaxis1..xaxis{rows}を使うので、重ね描き用の軸は
            # その次の番号を使う。x2を使うと出来高パネルの軸を壊してしまう。
            axis_id = rows + 1
            fig.add_trace(go.Bar(
                x=(prof / prof.max()).values, y=prof.index, orientation="h",
                marker_color="rgba(42,120,214,0.22)", marker_line_width=0,
                name="出来高プロファイル", xaxis=f"x{axis_id}", yaxis="y",
                customdata=share.values,
                hovertemplate="$%{y:,.2f}帯: 出来高シェア %{customdata:.1f}%<extra></extra>",
            ))
            fig.update_layout(**{f"xaxis{axis_id}": dict(
                overlaying="x", anchor="y", side="top", range=[0, 6],
                visible=False, fixedrange=True,
            )})

    if "サポレジライン" in set(o["overlays"]):
        for lv in (o.get("levels") or []):
            is_res = lv["type"] == "抵抗線"
            color = DOWN if is_res else UP
            strength = int(lv.get("strength") or 0)
            # 強いレベルほど濃く・太く描き、一目で優先順位が分かるようにする
            alpha = 0.06 + 0.035 * strength
            width = 1 + (1 if strength >= 4 else 0)
            if lv.get("zone_high", 0) > lv.get("zone_low", 0):
                rgb = "208,59,59" if is_res else "12,163,12"
                fig.add_hrect(
                    y0=lv["zone_low"], y1=lv["zone_high"],
                    fillcolor=f"rgba({rgb},{alpha:.3f})",
                    line_width=0, row=1, col=1,
                )
            label = f"${lv['price']:,.2f}"
            if strength:
                label += " " + "★" * strength
            # ラベルは枠の内側に置く。外側(left)だとY軸の目盛りと重なって読めない
            fig.add_hline(
                y=lv["price"], line=dict(color=color, width=width, dash="dash"),
                opacity=0.7, row=1, col=1,
                annotation_text=label,
                annotation_position="top left",
                annotation_font=dict(color=color, size=10),
                annotation_bgcolor="rgba(252,252,251,0.72)",
            )

    if not df.empty and o["current_price_line"]:
        last = float(o.get("current_price") or df["Close"].iloc[-1])
        previous = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        price_color = up_color if last >= previous else down_color
        fig.add_hline(y=last, line=dict(color=price_color, width=1, dash="dot"),
                      row=1, col=1,
                      annotation_text=f" {last:,.2f} ",
                      annotation_position="right",
                      annotation_bgcolor=price_color,
                      annotation_bordercolor=price_color,
                      annotation_font=dict(color="#ffffff", size=11))

    for i, name in enumerate(oscillators):
        _add_oscillator(fig, df, name, row=2 + i, opts=o)

    if o["compact_sessions"]:
        fig.update_xaxes(rangebreaks=sessions.rangebreaks(
            o["interval"], bool(o.get("extended_hours"))))
    if o.get("extended_hours") and o["interval"] in INTRADAY_INTERVALS:
        _shade_sessions(fig, df, rows)
    if o["log_scale"]:
        fig.update_yaxes(type="log", row=1, col=1)

    drag_modes = {"クロスヘア": "zoom", "ズーム": "zoom", "移動": "pan",
                  "ライン描画": "drawline"}
    crosshair = o["interaction"] == "クロスヘア"
    if o["range_selector"] and o["interval"] not in INTRADAY_INTERVALS:
        fig.update_xaxes(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ],
                bgcolor=theme["panel"], activecolor="#2a78d6",
                font=dict(color=theme["text"], size=10), x=0, y=1.04,
            ), row=1, col=1,
        )
    # ローソク足はrangesliderの既定がTrueなので、まず全行で明示的に消す。
    # 消さないと1行目のスライダーがサブチャート領域に価格チャートの縮小版を
    # 重ね描きしてしまう。表示するのは最下段だけ。
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(rangeslider_visible=o["range_slider"], row=rows, col=1)

    last_row = df.iloc[-1] if not df.empty else None
    ohlc_text = ""
    if last_row is not None:
        bar_change = float(last_row["Close"] - last_row["Open"])
        ohlc_text = (f"{ticker}  O {last_row['Open']:,.2f}  H {last_row['High']:,.2f}  "
                     f"L {last_row['Low']:,.2f}  C {last_row['Close']:,.2f}  "
                     f"{bar_change:+,.2f}")

    base_h = int(o.get("height") or 520)
    fig.update_layout(
        height=base_h + 150 * len(oscillators),
        hovermode="x unified" if crosshair else "closest",
        dragmode=o.get("dragmode") or drag_modes.get(o["interaction"], "zoom"),
        paper_bgcolor=theme["paper"], plot_bgcolor=theme["plot"],
        font=dict(color=theme["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="right",
                    x=1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=74, b=28, l=24, r=68),
        hoverlabel=dict(bgcolor=theme["panel"], font_color=theme["text"]),
        hoverdistance=80, spikedistance=-1,
        # uirevisionが同じ間、Plotlyは前回のズーム位置などUI状態を引き継ぐ。
        # インジケーターやテーマを切り替えてもズームが外れないようにしたいので、
        # 銘柄・足種・チャート種別だけをキーにする。
        uirevision=f"{ticker}-{o['interval']}-{o['chart_type']}",
        newshape=dict(line=dict(color="#f5b642", width=2), opacity=0.9),
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=theme["muted"],
                     activecolor="#4c9aff"),
        annotations=list(fig.layout.annotations) + ([dict(
            x=0, y=1.08, xref="paper", yref="paper", text=ohlc_text,
            showarrow=False, xanchor="left",
            font=dict(size=12, color=theme["text"]),
        )] if ohlc_text else []),
    )
    fig.update_xaxes(
        showgrid=o["grid"], gridcolor=theme["grid"], zeroline=False,
        showspikes=crosshair, spikemode="across", spikesnap="cursor",
        spikecolor=theme["muted"], spikethickness=1, spikedash="dot",
        tickfont=dict(size=10),
    )
    fig.update_yaxes(
        side="right", showgrid=o["grid"], gridcolor=theme["grid"],
        zeroline=False, fixedrange=False, tickfont=dict(size=10),
        showspikes=crosshair, spikemode="across", spikesnap="cursor",
        spikecolor=theme["muted"], spikethickness=1, spikedash="dot",
    )
    for annotation in fig.layout.annotations:
        if annotation.text in osc_titles.values():
            annotation.update(x=0.005, xanchor="left", font=dict(size=10,
                              color=theme["muted"]))
    return fig


def mini_price_chart(df: pd.DataFrame, label: str, interval: str = "1d",
                     theme_name: str = "ダーク") -> go.Figure:
    """マルチタイムフレーム用の軽量チャート。"""
    theme = CHART_THEMES.get(theme_name, CHART_THEMES["ダーク"])
    fig = go.Figure(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
        close=df["Close"], increasing_line_color=UP, increasing_fillcolor=UP,
        decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
        name=label, showlegend=False,
    ))
    for period, color in ((20, "#2a78d6"), (50, "#eda100")):
        column = f"SMA{period}"
        if column in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[column], line=dict(color=color, width=1.2),
                name=column, hoverinfo="skip", showlegend=False,
            ))
    if interval in {"1d", "1wk"}:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(
        title=dict(text=label, x=0.02, y=0.95, font=dict(size=13)),
        height=310, margin=dict(t=36, b=12, l=10, r=48),
        paper_bgcolor=theme["paper"], plot_bgcolor=theme["plot"],
        font=dict(color=theme["text"]), hovermode="x unified",
        xaxis_rangeslider_visible=False, dragmode="pan",
        uirevision=f"mini-{label}-{interval}",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9))
    fig.update_yaxes(side="right", showgrid=True, gridcolor=theme["grid"],
                     tickfont=dict(size=9))
    return fig


def comparison_chart(series_map: dict[str, pd.Series], interval: str = "1d") -> go.Figure:
    """複数銘柄のパフォーマンス比較(期間始点=100に正規化)。"""
    fig = go.Figure()
    for i, (label, s) in enumerate(series_map.items()):
        s = s.dropna()
        if s.empty:
            continue
        norm = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=label,
            line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2),
            hovertemplate=f"{label}: %{{y:.1f}}<extra></extra>",
        ))
    fig.add_hline(y=100, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(
        height=300, hovermode="x unified",
        yaxis_title="始点=100",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, b=20),
    )
    if interval == "1d":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    elif interval in INTRADAY_INTERVALS:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                      dict(bounds=[16, 9.5], pattern="hour")])
    return fig


def revenue_chart(fin: pd.DataFrame) -> go.Figure:
    """年次売上高の棒グラフ(単位: 10億ドル)。"""
    fig = go.Figure(go.Bar(
        x=fin.index.astype(str), y=fin["revenue"] / 1e9,
        marker_color=PRIMARY, marker_line_width=0,
        texttemplate="%{y:,.1f}", textposition="outside",
    ))
    fig.update_layout(
        title="売上高(年次)", yaxis_title="10億ドル",
        height=320, margin=dict(t=50, b=20), showlegend=False,
    )
    return fig


def eps_chart(fin: pd.DataFrame) -> go.Figure:
    """年次EPSの棒グラフ。"""
    fig = go.Figure(go.Bar(
        x=fin.index.astype(str), y=fin["eps"],
        marker_color=SECONDARY, marker_line_width=0,
        texttemplate="%{y:.2f}", textposition="outside",
    ))
    fig.update_layout(
        title="EPS(年次)", yaxis_title="ドル",
        height=320, margin=dict(t=50, b=20), showlegend=False,
    )
    return fig


def mood_gauge(score: float) -> go.Figure:
    """恐怖・強欲スコア(0〜100)の半円ゲージ。"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(font=dict(size=42)),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 25, 45, 55, 75, 100],
                      tickfont=dict(size=10, color=MUTED)),
            bar=dict(color="#0b0b0b", thickness=0.22),
            borderwidth=0,
            steps=[
                dict(range=[0, 25], color="#d03b3b"),
                dict(range=[25, 45], color="#ec835a"),
                dict(range=[45, 55], color="#c3c2b7"),
                dict(range=[55, 75], color="#54a054"),
                dict(range=[75, 100], color="#0ca30c"),
            ],
        ),
    ))
    fig.update_layout(height=240, margin=dict(t=30, b=10, l=30, r=30),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def sparkline(s: pd.Series) -> go.Figure:
    """メトリクスカード用の小さな推移チャート。始点比で色分け。"""
    s = s.dropna()
    color = UP if len(s) > 1 and s.iloc[-1] >= s.iloc[0] else DOWN
    fig = go.Figure(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        line=dict(color=color, width=1.5), hoverinfo="skip",
    ))
    fig.update_layout(
        height=48, margin=dict(t=2, b=2, l=0, r=0), showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def pl_bar(pl: pd.Series) -> go.Figure:
    """銘柄別損益の棒グラフ(プラス緑/マイナス赤)。"""
    colors = [UP if v >= 0 else DOWN for v in pl.values]
    fig = go.Figure(go.Bar(
        x=pl.index, y=pl.values, marker_color=colors, marker_line_width=0,
        text=[f"{v:+,.0f}" for v in pl.values], textposition="outside",
        hovertemplate="%{x}: $%{y:+,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(title="銘柄別損益(ドル)", height=320,
                      margin=dict(t=50, b=20), showlegend=False)
    return fig


def perf_bar(changes: pd.Series, title: str) -> go.Figure:
    """変化率の横棒グラフ(ソート済み、プラス緑/マイナス赤)。"""
    s = changes.sort_values()
    colors = [UP if v >= 0 else DOWN for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.2f}%" for v in s.values], textposition="outside",
        hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(title=title, height=400, margin=dict(t=50, b=20),
                      showlegend=False, xaxis_title="前日比(%)")
    return fig


def depth_chart(bids: list[tuple], asks: list[tuple]) -> go.Figure:
    """板情報の横棒グラフ(上=売り気配 赤、下=買い気配 緑)。

    bids / asks は (価格, 数量, 注文数) のリスト(価格の良い順)。
    """
    rows = [(p, v, "売り") for p, v, _n in reversed(asks)] + \
           [(p, v, "買い") for p, v, _n in bids]
    if not rows:
        return go.Figure()
    labels = [f"{p:,.2f}" for p, _v, _s in rows]
    fig = go.Figure(go.Bar(
        x=[v for _p, v, _s in rows], y=labels, orientation="h",
        marker_color=[DOWN if s == "売り" else UP for _p, _v, s in rows],
        marker_line_width=0,
        text=[f"{v:,}" for _p, v, _s in rows], textposition="outside",
        hovertemplate="%{y}: %{x:,}株<extra></extra>",
    ))
    # 売りと買いの境目(スプレッド)に線を引く
    if asks and bids:
        fig.add_hline(y=len(asks) - 0.5, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(
        title="板情報(気配)", height=max(260, 26 * len(rows) + 90),
        margin=dict(t=50, b=20), showlegend=False,
        xaxis_title="数量(株)", yaxis_title="価格($)",
        yaxis=dict(autorange="reversed", type="category"),
    )
    return fig


def capital_bar(tiers: list[tuple]) -> go.Figure:
    """資金流入の横棒グラフ。tiers は (区分, 純額, 流入, 流出) のリスト。"""
    labels = [t[0] for t in tiers]
    values = [t[1] for t in tiers]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=[UP if v >= 0 else DOWN for v in values], marker_line_width=0,
        text=[f"{v/1e6:+,.1f}M" for v in values], textposition="outside",
        hovertemplate="%{y}: $%{x:+,.0f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(title="資金流入(本日・純額)", height=260,
                      margin=dict(t=50, b=20), showlegend=False,
                      xaxis_title="純流入額(ドル)",
                      yaxis=dict(autorange="reversed"))
    return fig


def iv_hv_chart(series: pd.DataFrame) -> go.Figure:
    """オプションのIV(予想変動率)とHV(実績変動率)の推移。"""
    fig = go.Figure()
    for col, color, name in (("IV", PRIMARY, "IV(予想変動率)"),
                             ("HV", SECONDARY, "HV(実績変動率)")):
        if col in series.columns and series[col].notna().any():
            fig.add_trace(go.Scatter(
                x=series["日付"], y=series[col], name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"%{{x}}<br>{name}: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(title="IVとHVの推移", height=320, margin=dict(t=50, b=20),
                      yaxis_title="変動率(%)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


def short_interest_chart(df: pd.DataFrame) -> go.Figure:
    """空売り残高(棒)と浮動株に対する比率(線)の推移。"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["日付"], y=df["空売り株数"], name="空売り株数",
        marker_color=MUTED, marker_line_width=0,
        hovertemplate="%{x}<br>空売り: %{y:,.0f}株<extra></extra>"), secondary_y=False)
    if df["浮動株比率"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["日付"], y=df["浮動株比率"], name="浮動株比率",
            line=dict(color=DOWN, width=2),
            hovertemplate="%{x}<br>比率: %{y:.2f}%<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="空売り株数", secondary_y=False)
    fig.update_yaxes(title_text="浮動株比率(%)", secondary_y=True, showgrid=False)
    fig.update_layout(title="空売り残高の推移", height=320, margin=dict(t=50, b=20),
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


def institution_chart(df: pd.DataFrame) -> go.Figure:
    """機関投資家の保有比率(線)と機関数(棒)の推移。"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["報告期"], y=df["機関数"], name="機関数",
        marker_color="#c3c2b7", marker_line_width=0,
        hovertemplate="%{x}<br>機関数: %{y:,.0f}<extra></extra>"), secondary_y=False)
    if df["保有比率"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["報告期"], y=df["保有比率"], name="保有比率",
            line=dict(color=PRIMARY, width=2),
            hovertemplate="%{x}<br>保有比率: %{y:.2f}%<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title_text="機関数", secondary_y=False)
    fig.update_yaxes(title_text="保有比率(%)", secondary_y=True, showgrid=False)
    fig.update_layout(title="機関投資家の保有推移", height=320, margin=dict(t=50, b=20),
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


def put_call_chart(df: pd.DataFrame) -> go.Figure:
    """米国株オプション市場のPut/Callレシオ。1.0超で弱気寄り。"""
    fig = go.Figure(go.Scatter(
        x=df["日付"], y=df["Put/Call"], name="Put/Call",
        line=dict(color=PRIMARY, width=2),
        hovertemplate="%{x}<br>Put/Call: %{y:.2f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text="1.0(強気と弱気の境目)",
                  annotation_position="top left")
    fig.update_layout(title="Put/Callレシオ(米国株オプション・出来高ベース)",
                      height=300, margin=dict(t=50, b=20), showlegend=False,
                      yaxis_title="Put/Call")
    return fig


def fed_watch_chart(df: pd.DataFrame, meeting: str) -> go.Figure:
    """FedWatchの織り込み確率。指定の会合について金利レンジ別の確率を横棒で示す。"""
    sub = df[df["meeting_date"] == meeting].copy()
    sub["probability"] = pd.to_numeric(sub["probability"], errors="coerce")
    sub = sub.dropna(subset=["probability"]).sort_values("probability")
    if sub.empty:
        return go.Figure()
    top = sub["probability"].max()
    colors = [PRIMARY if v == top else "#c3c2b7" for v in sub["probability"]]
    fig = go.Figure(go.Bar(
        x=sub["probability"], y=sub["target_range"], orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:.1f}%" for v in sub["probability"]], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(title=f"{meeting} 会合の織り込み確率",
                      height=max(240, 40 * len(sub) + 90),
                      margin=dict(t=50, b=20), showlegend=False,
                      xaxis_title="確率(%)", yaxis_title="政策金利レンジ")
    return fig


def value_chart(total: pd.Series, cost: float) -> go.Figure:
    """ポートフォリオ評価額の推移(取得額の水平線付き)。"""
    fig = go.Figure(go.Scatter(
        x=total.index, y=total.values, name="評価額",
        line=dict(color=PRIMARY, width=2),
        hovertemplate="$%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=cost, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"取得額 ${cost:,.0f}",
                  annotation_position="bottom right",
                  annotation_font=dict(color=MUTED, size=11))
    fig.update_layout(
        height=340, hovermode="x unified", showlegend=False,
        margin=dict(t=30, b=20), yaxis_title="評価額(ドル)",
        xaxis=dict(rangebreaks=[dict(bounds=["sat", "mon"])]),
    )
    return fig


def sector_pie(sector_values: pd.Series) -> go.Figure:
    """セクター別の評価額配分の円グラフ。9セクター以上は「その他」に集約。"""
    s = sector_values.sort_values(ascending=False)
    if len(s) > len(CATEGORICAL):
        head = s.iloc[:len(CATEGORICAL) - 1]
        head["その他"] = s.iloc[len(CATEGORICAL) - 1:].sum()
        s = head

    fig = go.Figure(go.Pie(
        labels=s.index, values=s.values,
        marker=dict(colors=CATEGORICAL[:len(s)],
                    line=dict(color="#ffffff", width=2)),
        hole=0.4, sort=False,
        texttemplate="%{label}<br>%{percent}",
        hovertemplate="%{label}<br>$%{value:,.2f}(%{percent})<extra></extra>",
    ))
    fig.update_layout(height=420, margin=dict(t=30, b=20), showlegend=False)
    return fig
