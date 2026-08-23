#!/usr/bin/env python3
"""Codex・Claude・手動編集で共用する編集ログ追記コマンド。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib import edit_log  # noqa: E402


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gitで共有する編集ログを1件追加します。既存ログは変更しません。"))
    parser.add_argument("--editor", required=True, choices=edit_log.EDITORS,
                        help="編集者: codex / claude / manual / other")
    parser.add_argument("--summary", required=True,
                        help="今回の変更を表す短い日本語の要約")
    parser.add_argument("--change", action="append", required=True,
                        dest="changes", help="変更点。複数回指定できます")
    parser.add_argument("--file", action="append", required=True,
                        dest="files", help="変更したGit相対パス。複数回指定できます")
    parser.add_argument("--validation", action="append", default=[],
                        help="実行した確認と結果。複数回指定できます")
    parser.add_argument("--follow-up", action="append", default=[],
                        dest="follow_ups", help="残課題。複数回指定できます")
    parser.add_argument("--status", choices=edit_log.STATUSES,
                        default="completed", help="作業自体の状態")
    parser.add_argument(
        "--verification-status", choices=edit_log.VERIFICATION_STATUSES,
        default="not_run", help="テスト・確認の状態")
    parser.add_argument("--base-commit",
                        help="変更開始時のコミット。省略時は現在のHEAD")
    parser.add_argument("--commit", help="変更を含むコミット（既知の場合のみ）")
    parser.add_argument("--branch", help="作業ブランチ。省略時は現在のブランチ")
    parser.add_argument("--supersedes", help="このログが訂正する過去ログのid")
    return parser


def main(argv: list[str] | None = None, *,
         entry_dir: str | Path | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        path, entry = edit_log.append_entry(
            editor=args.editor,
            summary=args.summary,
            changes=args.changes,
            files=args.files,
            validation=args.validation,
            follow_ups=args.follow_ups,
            status=args.status,
            verification_status=args.verification_status,
            base_commit=args.base_commit or _git_value("rev-parse", "--short=12", "HEAD"),
            commit=args.commit,
            branch=args.branch or _git_value("branch", "--show-current"),
            supersedes=args.supersedes,
            entry_dir=entry_dir or edit_log.ENTRY_DIR,
        )
    except edit_log.EditLogValidationError as exc:
        print(f"入力エラー: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"保存エラー: {exc}", file=sys.stderr)
        return 1

    try:
        display_path = path.relative_to(REPO_ROOT)
    except ValueError:
        # テスト用の任意保存先でも端末固有の絶対パスを標準出力へ出さない。
        display_path = Path("docs/edit-log/entries") / path.name
    print(f"編集ログを追加しました: {display_path}")
    print(f"id: {entry['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
