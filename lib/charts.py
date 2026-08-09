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

    # 十字カーソル。価格を目で追いやすくする(サブチャートとx軸は共有)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor=MUTED, spikethickness=1, spikedash="dot")
    fig.update_yaxes(showspikes=True, spikemode="toaxis", spikesnap="cursor",
                     spikecolor=MUTED, spikethickness=1, spikedash="dot",
                     row=1, col=1)

    base_h = int(o.get("height") or 430)
    fig.update_layout(
        height=base_h + 140 * len(oscillators),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        spikedistance=-1,
        dragmode=o.get("dragmode") or "zoom",
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
