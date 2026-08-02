"""UI部品: カラーチップ、相対時刻、センチメントバー。

配色はチャートと同じ検証済みパレット(lib/charts.py)に合わせる。
"""

from datetime import datetime, timezone

import plotly.graph_objects as go

# (背景色, 文字色) — パレット各色の淡色チップ
_CHIP_STYLES = {
    "blue": ("#e6effa", "#1c5cab"),
    "green": ("#e2f4e2", "#006300"),
    "red": ("#fbe7e7", "#a32d2d"),
    "orange": ("#fdeee6", "#b04516"),
    "violet": ("#eceafd", "#3a2f86"),
    "gray": ("#f0efec", "#52514e"),
}

SOURCE_COLORS = {
    "Stocktwits": "blue",
    "Hacker News": "orange",
    "Mastodon": "violet",
}

SENTIMENT_LABELS = {"Bullish": "強気", "Bearish": "弱気"}


def chip(text: str, color: str = "gray") -> str:
    bg, fg = _CHIP_STYLES.get(color, _CHIP_STYLES["gray"])
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:12px;font-size:0.78rem;font-weight:600;'
            f'white-space:nowrap;">{text}</span>')


def source_chip(source: str) -> str:
    return chip(source, SOURCE_COLORS.get(source, "gray"))


def sentiment_chip(sentiment: str | None) -> str:
    if sentiment == "Bullish":
        return chip("🐂 強気", "green")
    if sentiment == "Bearish":
        return chip("🐻 弱気", "red")
    return ""


def relative_time(utc_str: str) -> str:
    """"YYYY-MM-DD HH:MM"(UTC)を「◯分前/◯時間前/◯日前」に変換する。"""
    try:
        dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return utc_str
    delta = datetime.now(timezone.utc) - dt
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "たった今"
    if minutes < 60:
        return f"{minutes}分前"
    if minutes < 60 * 24:
        return f"{minutes // 60}時間前"
    return f"{minutes // (60 * 24)}日前"


def stacked_bar(segments: list[tuple[str, int, str, str]]) -> go.Figure:
    """(ラベル, 値, 塗り色, 文字色) の並びを1本の積み上げ横棒で示す。"""
    fig = go.Figure()
    for label, value, color, text_color in segments:
        fig.add_trace(go.Bar(
            x=[value], y=[""], name=label, orientation="h",
            marker=dict(color=color, line=dict(color="#fcfcfb", width=2)),
            text=f"{label} {value}" if value else "",
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color=text_color, size=13),
            hovertemplate=f"{label}: {value}件<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=56, showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def sentiment_bar(bullish: int, bearish: int, neutral: int) -> go.Figure:
    """強気/弱気/表明なしの内訳バー。"""
    return stacked_bar([
        ("強気", bullish, "#0ca30c", "#ffffff"),
        ("弱気", bearish, "#d03b3b", "#ffffff"),
        ("表明なし", neutral, "#c3c2b7", "#0b0b0b"),
    ])


def rating_bar(counts: dict) -> go.Figure:
    """アナリストレーティング分布バー(強い買い→強い売り)。"""
    return stacked_bar([
        ("強い買い", counts.get("strongBuy", 0), "#006300", "#ffffff"),
        ("買い", counts.get("buy", 0), "#0ca30c", "#ffffff"),
        ("中立", counts.get("hold", 0), "#c3c2b7", "#0b0b0b"),
        ("売り", counts.get("sell", 0), "#ec835a", "#ffffff"),
        ("強い売り", counts.get("strongSell", 0), "#d03b3b", "#ffffff"),
    ])
