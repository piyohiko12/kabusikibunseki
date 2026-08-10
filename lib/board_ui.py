"""読み取り専用の銘柄情報掲示板UI。"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from zoneinfo import ZoneInfo

import streamlit as st

from lib import ui


CATEGORY_COLORS = {
    "trade": "violet",
    "alert": "orange",
    "event": "red",
    "news": "blue",
    "price": "green",
    "level": "gray",
    "analyst": "violet",
}
IMPORTANCE_COLORS = {
    "critical": "red",
    "high": "orange",
    "medium": "blue",
    "low": "gray",
    "info": "gray",
}
SORT_LABELS = {
    "重要度順": "importance",
    "新着順": "newest",
    "カテゴリ順": "category",
}


def _plain_markdown(value: object) -> str:
    """外部データをMarkdown/HTMLとして実行させず、改行だけ維持する。"""
    text = str(value or "")
    text = re.sub(r"([\\`*_{}\[\]()<>#+\-.!|>$])", r"\\\1", text)
    return text.replace("\n", "  \n")


def _timestamp(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        try:
            stamp = datetime.strptime(str(value)[:10], "%Y-%m-%d")
        except (TypeError, ValueError):
            return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _format_time(value: object) -> str:
    raw = str(value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw.replace("-", "/") + "（時刻未定）"
    stamp = _timestamp(value)
    if stamp is None:
        return "時刻不明"
    return stamp.astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y/%m/%d %H:%M JST")


def _item_sort_key(item: dict, mode: str):
    stamp = _timestamp(item.get("occurred_at"))
    timestamp = stamp.timestamp() if stamp else float("-inf")
    rank = int(item.get("importance_rank") or 0)
    if mode == "newest":
        return (-timestamp, -rank, str(item.get("id") or ""))
    if mode == "category":
        return (str(item.get("category_label_ja") or ""), -rank, -timestamp)
    return (-rank, -timestamp, str(item.get("id") or ""))


def _filter_items(items: list[dict], categories: list[str],
                  importance: list[str], search: str) -> list[dict]:
    needle = search.strip().casefold()
    output = []
    for item in items:
        if categories and item.get("category_label_ja") not in categories:
            continue
        if importance and item.get("importance_label_ja") not in importance:
            continue
        haystack = " ".join([
            str(item.get("title_ja") or ""),
            str(item.get("summary_ja") or ""),
            str(item.get("source") or ""),
            " ".join(str(tag) for tag in item.get("tags") or []),
        ]).casefold()
        if needle and needle not in haystack:
            continue
        output.append(item)
    return output


def render_information_board(report: dict, *, show_heading: bool = True,
                             key_prefix: str = "information_board") -> None:
    """正規化済みreportを、投稿機能のない一覧として描画する。"""
    report = report if isinstance(report, dict) else {}
    ticker = str(report.get("ticker") or "—")
    items = [item for item in report.get("items") or [] if isinstance(item, dict)]
    if show_heading:
        st.subheader(f"📋 {ticker} 情報掲示板")
    st.caption(
        "価格・判定・アラート・イベント・ニュースを読み取り専用で一覧化しています。"
        "投稿、返信、データ保存、外部送信、注文は行いません。")

    warnings = [str(value) for value in report.get("warnings") or [] if str(value).strip()]
    if warnings:
        with st.expander(f"⚠️ 取得・整理上の注意 {len(warnings)}件"):
            for warning in warnings:
                st.warning(warning)

    high_items = [item for item in items
                  if item.get("importance") in {"critical", "high"}]
    event_items = [item for item in items if item.get("category") == "event"]
    active_alerts = [item for item in items
                     if item.get("category") == "alert"
                     and (item.get("data") or {}).get("triggered") is True]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("全情報", f"{len(items)}件", border=True)
    m2.metric("重要", f"{len(high_items)}件", border=True)
    m3.metric("イベント", f"{len(event_items)}件", border=True)
    m4.metric("成立アラート", f"{len(active_alerts)}件", border=True)

    categories = sorted({str(item.get("category_label_ja")) for item in items
                         if item.get("category_label_ja")})
    importance_labels = sorted(
        {str(item.get("importance_label_ja")) for item in items
         if item.get("importance_label_ja")},
        key=lambda label: {"最重要": 0, "重要": 1, "要確認": 2,
                           "参考": 3, "情報": 4}.get(label, 9))
    f1, f2, f3 = st.columns([1.3, 1.1, 1.4])
    selected_categories = f1.multiselect(
        "カテゴリ", categories, default=categories,
        key=f"{key_prefix}_categories")
    selected_importance = f2.multiselect(
        "重要度", importance_labels, default=importance_labels,
        key=f"{key_prefix}_importance")
    search = f3.text_input(
        "キーワード", placeholder="見出し・内容・情報源を検索",
        key=f"{key_prefix}_search")
    sort_label = st.radio(
        "並び順", list(SORT_LABELS), horizontal=True,
        key=f"{key_prefix}_sort")

    shown_items = _filter_items(
        items, selected_categories, selected_importance, search)
    shown_items.sort(key=lambda item: _item_sort_key(item, SORT_LABELS[sort_label]))
    st.caption(f"{len(shown_items)} / {len(items)}件を表示")

    if not shown_items:
        if items:
            st.info("条件に一致する情報はありません。絞り込みを変更してください。")
        else:
            st.info("表示できる情報がありません。取得上の注意を確認してください。")
        return

    for item in shown_items:
        with st.container(border=True):
            category = str(item.get("category") or "")
            importance = str(item.get("importance") or "info")
            chips = [
                ui.chip(str(item.get("category_label_ja") or "情報"),
                        CATEGORY_COLORS.get(category, "gray")),
                ui.chip(f"重要度: {item.get('importance_label_ja') or '情報'}",
                        IMPORTANCE_COLORS.get(importance, "gray")),
            ]
            st.markdown(" ".join(chips), unsafe_allow_html=True)
            st.markdown(f"#### {_plain_markdown(item.get('title_ja'))}")
            if item.get("summary_ja"):
                st.markdown(_plain_markdown(item.get("summary_ja")))
            source = str(item.get("source") or "情報源不明")
            st.caption(f"{_plain_markdown(source)} ・ {_format_time(item.get('occurred_at'))}")
            tags = [str(tag) for tag in item.get("tags") or [] if str(tag).strip()]
            if tags:
                st.caption("タグ: " + " / ".join(_plain_markdown(tag) for tag in tags[:6]))
            url = item.get("url")
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                st.link_button("原文・詳細を開く ↗", url)

    metadata = report.get("metadata") or {}
    quota_used = bool(
        metadata.get("uses_moomoo_history_quota")
        or metadata.get("moomoo_history_quota_used")
    )
    st.caption(
        "この一覧は表示専用です。売買判定へ自動加点せず、"
        "この掲示板によるmoomoo過去K線の追加取得: "
        f"{'あり' if quota_used else 'なし'}。")


__all__ = ["render_information_board"]
