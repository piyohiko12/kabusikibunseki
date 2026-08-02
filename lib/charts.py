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

UP = "#0ca30c"        # 上昇(status: good)
DOWN = "#d03b3b"      # 下降(status: critical)
PRIMARY = "#2a78d6"   # 単一系列の既定色(categorical slot 1)
SECONDARY = "#1baf7a" # 2つ目の単一系列(categorical slot 2)
MUTED = "#898781"     # 基準線・補助要素
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]

# オーバーレイ線の配色(同時表示されても区別できるように割り当て)
OVERLAY_LINES = {
    "SMA20": ("#2a78d6", None),
    "SMA50": ("#eda100", None),
    "SMA200": ("#4a3aa7", None),
    "EMA20": ("#1baf7a", "dash"),
    "EMA50": ("#e87ba4", "dash"),
}

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
    "overlays": ["SMA20", "SMA50", "SMA200"],
    "oscillators": ["出来高", "RSI", "MACD"],
    "events": True,
    "log_scale": False,
}


def _add_price_traces(fig, df, ticker, chart_type):
    if chart_type == "平均足":
        ha = indicators.heikin_ashi(df)
        fig.add_trace(go.Candlestick(
            x=ha.index, open=ha["Open"], high=ha["High"],
            low=ha["Low"], close=ha["Close"], name=f"{ticker}(平均足)",
            increasing_line_color=UP, increasing_fillcolor=UP,
            decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
            showlegend=False,
        ), row=1, col=1)
    elif chart_type == "ライン":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], name=ticker,
            line=dict(color=PRIMARY, width=2), showlegend=False,
        ), row=1, col=1)
    else:
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker,
            increasing_line_color=UP, increasing_fillcolor=UP,
            decreasing_line_color=DOWN, decreasing_fillcolor=DOWN,
            showlegend=False,
        ), row=1, col=1)


def _add_overlays(fig, df, overlays):
    for name, (color, dash) in OVERLAY_LINES.items():
        if name in overlays and name in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[name], name=name,
                line=dict(color=color, width=2, dash=dash), hoverinfo="skip",
            ), row=1, col=1)

    if "VWAP(日中)" in overlays and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#184f95", width=2), hoverinfo="skip",
        ), row=1, col=1)

    if "ボリンジャーバンド" in overlays and "BB_up" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_up"], name="ボリンジャー(±2σ)",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_low"], showlegend=False,
            line=dict(color=MUTED, width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(137,135,129,0.12)", hoverinfo="skip",
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


def _add_oscillator(fig, df, name, row):
    if name == "出来高" and "Volume" in df.columns:
        colors = [UP if c >= o else DOWN
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="出来高",
            marker_color=colors, marker_line_width=0, opacity=0.5,
            showlegend=False,
        ), row=row, col=1)
        if "VOL_MA20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["VOL_MA20"], name="出来高MA20",
                line=dict(color=MUTED, width=1.5), showlegend=False,
            ), row=row, col=1)
    elif name == "RSI":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color=PRIMARY, width=2), showlegend=False,
        ), row=row, col=1)
        for level in (30, 70):
            fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"),
                          row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    elif name == "MACD":
        hist_colors = [UP if v >= 0 else DOWN for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"], name="ヒストグラム",
            marker_color=hist_colors, marker_line_width=0, opacity=0.5,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=PRIMARY, width=2),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="シグナル",
            line=dict(color="#eda100", width=2),
        ), row=row, col=1)
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


def price_chart(df: pd.DataFrame, ticker: str, opts: dict | None = None) -> go.Figure:
    """メインチャート(価格+選択したオーバーレイ・サブチャート)。"""
    o = {**DEFAULT_OPTS, **(opts or {})}
    oscillators = [n for n in o["oscillators"] if n in OSC_TITLES]

    rows = 1 + len(oscillators)
    weights = [3.0] + [1.0] * len(oscillators)
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[w / sum(weights) for w in weights],
        vertical_spacing=0.24 / max(rows, 2),
        subplot_titles=[""] + [OSC_TITLES[n] for n in oscillators],
    )

    _add_price_traces(fig, df, ticker, o["chart_type"])
    _add_overlays(fig, df, set(o["overlays"]))
    if o["events"]:
        _add_event_markers(fig, df)

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
            fig.add_trace(go.Bar(
                x=(prof / prof.max()).values, y=prof.index, orientation="h",
                marker_color="rgba(42,120,214,0.22)", marker_line_width=0,
                name="出来高プロファイル", xaxis="x2", yaxis="y",
                customdata=share.values,
                hovertemplate="$%{y:,.2f}帯: 出来高シェア %{customdata:.1f}%<extra></extra>",
            ))
            fig.update_layout(xaxis2=dict(
                overlaying="x", side="top", range=[0, 6],
                visible=False, fixedrange=True,
            ))

    if "サポレジライン" in set(o["overlays"]):
        for lv in (o.get("levels") or []):
            is_res = lv["type"] == "抵抗線"
            color = DOWN if is_res else UP
            if lv.get("zone_high", 0) > lv.get("zone_low", 0):
                fig.add_hrect(
                    y0=lv["zone_low"], y1=lv["zone_high"],
                    fillcolor=("rgba(208,59,59,0.08)" if is_res
                               else "rgba(12,163,12,0.08)"),
                    line_width=0, row=1, col=1,
                )
            fig.add_hline(
                y=lv["price"], line=dict(color=color, width=1, dash="dash"),
                opacity=0.65, row=1, col=1,
                annotation_text=f"${lv['price']:,.2f}",
                annotation_position="left",
                annotation_font=dict(color=color, size=10),
            )

    if not df.empty:
        last = float(df["Close"].iloc[-1])
        fig.add_hline(y=last, line=dict(color=MUTED, width=1, dash="dot"),
                      row=1, col=1,
                      annotation_text=f"${last:,.2f}",
                      annotation_position="top right",
                      annotation_font=dict(color=MUTED, size=11))

    for i, name in enumerate(oscillators):
        _add_oscillator(fig, df, name, row=2 + i)

    if o["interval"] == "1d":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    elif o["interval"] in INTRADAY_INTERVALS:
        # 週末と場外時間(米国株の通常セッション9:30〜16:00 ET)を除去
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]),
                                      dict(bounds=[16, 9.5], pattern="hour")])
    if o["log_scale"]:
        fig.update_yaxes(type="log", row=1, col=1)

    fig.update_layout(
        height=430 + 140 * len(oscillators),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(t=30, b=20),
    )
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
