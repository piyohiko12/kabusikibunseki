"""Codex・Claude・手動編集で共有する読み取り専用の編集ログ画面。"""

import streamlit as st

from lib import edit_log as edit_log_lib


@st.cache_data(ttl=5, show_spinner=False)
def _load_edit_log() -> dict:
    return edit_log_lib.load_entries()


st.title("📝 編集ログ")
st.caption(
    "Codex・Claude・手動編集による変更内容と確認結果を、Gitで共有しています。")
st.info(
    "これは分析ツールの開発履歴です。株価、注文、保有銘柄の履歴ではありません。",
    icon="ℹ️",
)

report = _load_edit_log()
entries = report["entries"]
warnings = report["warnings"]

if warnings:
    st.warning(
        f"読み込めない編集ログが {len(warnings)} 件あります。"
        "正常なログは引き続き表示しています。",
        icon="⚠️",
    )
    with st.expander("読み込み警告を見る"):
        for warning in warnings:
            st.text(warning)

editor_count = len({entry["editor"] for entry in entries})
verified_count = sum(
    entry["verification_status"] == "passed" for entry in entries)
follow_up_count = sum(bool(entry["follow_ups"]) for entry in entries)
m1, m2, m3, m4 = st.columns(4)
m1.metric("記録", f"{len(entries)}件")
m2.metric("編集者", f"{editor_count}種類")
m3.metric("確認済み", f"{verified_count}件")
m4.metric("残課題あり", f"{follow_up_count}件")

st.subheader("変更を探す")
query = st.text_input(
    "キーワード", placeholder="例: チャート、アラート、時間外価格",
    key="edit_log_query",
)
f1, f2 = st.columns(2)
editor_options = [""] + sorted({entry["editor"] for entry in entries})
selected_editor = f1.selectbox(
    "編集者", editor_options,
    format_func=lambda value: (
        "すべて" if not value else edit_log_lib.EDITOR_LABELS.get(value, value)),
    key="edit_log_editor",
)
status_options = [""] + list(edit_log_lib.STATUSES)
selected_status = f2.selectbox(
    "作業状態", status_options,
    format_func=lambda value: (
        "すべて" if not value else edit_log_lib.STATUS_LABELS.get(value, value)),
    key="edit_log_status",
)

filtered = edit_log_lib.filter_entries(
    entries, editor=selected_editor, status=selected_status, query=query)
st.caption(f"新しい順に {len(filtered)} / {len(entries)} 件を表示")

if not entries:
    st.info("編集ログはまだありません。最初の変更から記録を開始します。")
elif not filtered:
    st.info("条件に一致する編集ログはありません。検索条件を変更してください。")

for entry in filtered:
    editor_label = edit_log_lib.EDITOR_LABELS.get(entry["editor"], entry["editor"])
    status_label = edit_log_lib.STATUS_LABELS.get(entry["status"], entry["status"])
    verification_label = edit_log_lib.VERIFICATION_STATUS_LABELS.get(
        entry["verification_status"], entry["verification_status"])
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.subheader(edit_log_lib.escape_markdown(entry["summary"]), anchor=False)
        right.markdown(f"**{status_label}**")
        st.caption(edit_log_lib.escape_markdown(
            f"{edit_log_lib.format_created_at_jst(entry['created_at'])}"
            f" ／ 編集者: {editor_label} ／ 確認: {verification_label}"))
        for change in entry["changes"]:
            st.markdown(edit_log_lib.escape_markdown(f"・{change}"))

        with st.expander("詳しい内容"):
            if entry["files"]:
                st.markdown("**変更ファイル**")
                st.code("\n".join(entry["files"]), language=None)
            st.markdown("**確認結果**")
            if entry["validation"]:
                for item in entry["validation"]:
                    st.markdown(edit_log_lib.escape_markdown(f"・{item}"))
            else:
                st.write("・確認結果は記録されていません")
            if entry["follow_ups"]:
                st.markdown("**残っている作業・注意点**")
                for item in entry["follow_ups"]:
                    st.markdown(edit_log_lib.escape_markdown(f"・{item}"))
            context = [
                f"作業ブランチ: {entry['branch']}" if entry["branch"] else None,
                f"変更開始時: {entry['base_commit']}" if entry["base_commit"] else None,
                f"変更コミット: {entry['commit']}" if entry["commit"] else None,
                f"訂正対象: {entry['supersedes']}" if entry["supersedes"] else None,
                f"ログID: {entry['id']}",
            ]
            st.caption(edit_log_lib.escape_markdown(
                " ／ ".join(item for item in context if item)))

with st.expander("Codex・Claudeから記録する方法"):
    st.write(
        "コード変更と同じコミットへ、新しいログファイルを1件追加します。"
        "既存のログは編集しません。")
    st.code(
        ".venv/bin/python scripts/add_edit_log.py \\\n  --editor codex \\\n  --summary \"変更の短い要約\" \\\n  --change \"実装した内容\" \\\n  --file views/example.py \\\n  --verification-status passed \\\n  --validation \"全テスト成功\"",
        language="bash",
    )
    st.caption("詳しいルールは docs/edit-log/README.md を参照してください。")
