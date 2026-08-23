"""Gitで共有する追記専用の編集ログ。

1作業を1つのJSONファイルへ保存し、CodexとClaudeが同時に追記しても
同じファイルを編集しない構成にする。アプリ側は読み取り専用で利用する。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
import re
import tempfile
from typing import Iterable, Mapping
import unicodedata
import uuid
from zoneinfo import ZoneInfo


SCHEMA_VERSION = 1
MAX_ENTRY_BYTES = 256 * 1024
REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY_DIR = REPO_ROOT / "docs" / "edit-log" / "entries"

ENTRY_FIELDS = frozenset({
    "schema_version", "id", "created_at", "editor", "status",
    "verification_status", "summary", "changes", "files", "validation",
    "follow_ups", "base_commit", "commit", "branch", "supersedes",
})

EDITORS = ("codex", "claude", "manual", "other")
EDITOR_LABELS = {
    "codex": "Codex",
    "claude": "Claude",
    "manual": "手動編集",
    "other": "その他",
}
STATUSES = ("completed", "in_progress", "blocked")
STATUS_LABELS = {
    "completed": "完了",
    "in_progress": "作業中",
    "blocked": "保留・要確認",
}
VERIFICATION_STATUSES = ("passed", "partial", "failed", "not_run")
VERIFICATION_STATUS_LABELS = {
    "passed": "確認済み",
    "partial": "一部確認",
    "failed": "未解決あり",
    "not_run": "未確認",
}

_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{7,127}$")
_PRIVATE_PATH_ROOTS = frozenset({".git", ".venv", "data", "reports"})
_MARKDOWN_RE = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|>$])")
_SENSITIVE_PATTERNS = (
    re.compile(
        r"(?i)(?:^|[\s\"'`])(?:/Users/|/home/|/root/|/private/|"
        r"[A-Z]:[\\/]Users[\\/]|\\\\[^\\\s]+\\[^\\\s]+)"),
    re.compile(
        r"(?i)\b(?:api[_ -]?key|access[_ -]?token|refresh[_ -]?token|token|"
        r"password|passwd|secret|cookie|authorization)\b\s*[:=]\s*\S+"),
    re.compile(
        r"(?i)\b(?:sk-[a-z0-9_-]{16,}|gh[pousr]_[a-z0-9]{16,}|"
        r"github_pat_[a-z0-9_]{16,}|xox[baprs]-[a-z0-9-]{10,})\b"),
    re.compile(
        r"(?i)\b(?:account[_ -]?id|acc[_ -]?id|口座ID)\b\s*[:=]\s*[a-z0-9-]+"),
)


class EditLogValidationError(ValueError):
    """編集ログの入力または保存形式が不正。"""


def escape_markdown(value: object) -> str:
    """ログ本文をMarkdownのリンク・画像・HTMLとして解釈させない。"""
    text = str(value or "")
    return _MARKDOWN_RE.sub(r"\\\1", text).replace("\n", "  \n")


def _text(value: object, field: str, *, required: bool = False,
          max_length: int = 500) -> str | None:
    if value is None:
        if required:
            raise EditLogValidationError(f"{field} は必須です")
        return None
    if not isinstance(value, str):
        raise EditLogValidationError(f"{field} は文字列で指定してください")
    cleaned = value.strip()
    if required and not cleaned:
        raise EditLogValidationError(f"{field} は空にできません")
    if not cleaned:
        return None
    if len(cleaned) > max_length:
        raise EditLogValidationError(
            f"{field} は {max_length} 文字以内で指定してください")
    if any(ord(char) < 32 or unicodedata.category(char) == "Cf"
           for char in cleaned):
        raise EditLogValidationError(f"{field} に制御文字は使用できません")
    if any(pattern.search(cleaned) for pattern in _SENSITIVE_PATTERNS):
        raise EditLogValidationError(
            f"{field} に端末固有パスまたは秘密情報らしき値は記録できません")
    return cleaned


def _text_list(value: object, field: str, *, required: bool = False,
               max_items: int = 100, max_length: int = 500) -> list[str]:
    if value is None:
        items: list[object] = []
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise EditLogValidationError(f"{field} は配列で指定してください")
    if len(items) > max_items:
        raise EditLogValidationError(
            f"{field} は {max_items} 件以内で指定してください")
    cleaned = [
        _text(item, f"{field}[{index}]", required=True,
              max_length=max_length)
        for index, item in enumerate(items)
    ]
    result = [item for item in cleaned if item is not None]
    if required and not result:
        raise EditLogValidationError(f"{field} は1件以上必要です")
    return result


def _file_list(value: object) -> list[str]:
    files = _text_list(value, "files", required=True, max_length=300)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in files:
        if "\\" in value:
            raise EditLogValidationError("files は / 区切りのリポジトリ相対パスにしてください")
        path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        if (path.is_absolute() or windows_path.is_absolute()
                or windows_path.drive or ".." in path.parts
                or value.startswith("./")):
            raise EditLogValidationError(
                "files に絶対パスやリポジトリ外のパスは使用できません")
        normalized_value = path.as_posix()
        if normalized_value in ("", "."):
            raise EditLogValidationError("files に空のパスは使用できません")
        if path.parts and path.parts[0] in _PRIVATE_PATH_ROOTS:
            raise EditLogValidationError(
                f"files に非公開・Git管理外のパスは使用できません: {path.parts[0]}")
        if normalized_value in seen:
            raise EditLogValidationError(f"files が重複しています: {normalized_value}")
        seen.add(normalized_value)
        normalized.append(normalized_value)
    return normalized


def _utc_iso(value: object) -> str:
    if isinstance(value, datetime):
        stamp = value
    elif isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            stamp = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise EditLogValidationError(
                "created_at はISO 8601形式で指定してください") from exc
    else:
        raise EditLogValidationError(
            "created_at はタイムゾーン付き日時で指定してください")
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise EditLogValidationError("created_at にはタイムゾーンが必要です")
    return stamp.astimezone(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


def normalize_entry(raw: Mapping[str, object]) -> dict:
    """保存済みエントリを検証し、安定した公開形へ正規化する。"""
    if not isinstance(raw, Mapping):
        raise EditLogValidationError("編集ログはJSONオブジェクトで保存してください")
    if (type(raw.get("schema_version")) is not int
            or raw.get("schema_version") != SCHEMA_VERSION):
        raise EditLogValidationError(
            f"schema_version は {SCHEMA_VERSION} が必要です")
    unknown_fields = sorted(set(raw) - ENTRY_FIELDS)
    if unknown_fields:
        raise EditLogValidationError(
            "未対応の項目があります: " + ", ".join(unknown_fields))

    entry_id = _text(raw.get("id"), "id", required=True, max_length=128)
    if entry_id is None or not _ID_RE.fullmatch(entry_id):
        raise EditLogValidationError("id の形式が正しくありません")

    editor = _text(raw.get("editor"), "editor", required=True,
                   max_length=20)
    editor = str(editor).lower()
    if editor not in EDITORS:
        raise EditLogValidationError(
            f"editor は {', '.join(EDITORS)} から選んでください")

    status = _text(raw.get("status"), "status", required=True,
                   max_length=20)
    status = str(status).lower()
    if status not in STATUSES:
        raise EditLogValidationError(
            f"status は {', '.join(STATUSES)} から選んでください")

    verification_status = _text(
        raw.get("verification_status"), "verification_status",
        required=True, max_length=20)
    verification_status = str(verification_status).lower()
    if verification_status not in VERIFICATION_STATUSES:
        raise EditLogValidationError(
            "verification_status は "
            f"{', '.join(VERIFICATION_STATUSES)} から選んでください")
    validation = _text_list(raw.get("validation"), "validation")
    if verification_status != "not_run" and not validation:
        raise EditLogValidationError(
            "確認状態を記録する場合は validation も1件以上必要です")

    supersedes = _text(raw.get("supersedes"), "supersedes", max_length=128)
    if supersedes is not None and not _ID_RE.fullmatch(supersedes):
        raise EditLogValidationError("supersedes の形式が正しくありません")
    if supersedes == entry_id:
        raise EditLogValidationError("自分自身を supersedes に指定できません")

    return {
        "schema_version": SCHEMA_VERSION,
        "id": entry_id,
        "created_at": _utc_iso(raw.get("created_at")),
        "editor": editor,
        "status": status,
        "verification_status": verification_status,
        "summary": _text(raw.get("summary"), "summary", required=True,
                         max_length=200),
        "changes": _text_list(raw.get("changes"), "changes", required=True),
        "files": _file_list(raw.get("files")),
        "validation": validation,
        "follow_ups": _text_list(raw.get("follow_ups"), "follow_ups"),
        "base_commit": _text(raw.get("base_commit"), "base_commit",
                             max_length=80),
        "commit": _text(raw.get("commit"), "commit", max_length=80),
        "branch": _text(raw.get("branch"), "branch", max_length=200),
        "supersedes": supersedes,
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "editor"


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise EditLogValidationError(f"JSONの項目が重複しています: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise EditLogValidationError(f"JSONに非有限値は使用できません: {value}")


def append_entry(*, editor: str, summary: str,
                 changes: Iterable[str], files: Iterable[str],
                 validation: Iterable[str] = (),
                 follow_ups: Iterable[str] = (), status: str = "completed",
                 verification_status: str = "not_run",
                 base_commit: str | None = None, commit: str | None = None,
                 branch: str | None = None, supersedes: str | None = None,
                 now: datetime | None = None,
                 entry_dir: str | Path = ENTRY_DIR) -> tuple[Path, dict]:
    """新しいログを一意なファイルへ原子的に追加する。既存ログは変更しない。"""
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise EditLogValidationError("now にはタイムゾーンが必要です")
    entry_id = (
        stamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{_slug(editor)}-{uuid.uuid4().hex[:8]}"
    )
    entry = normalize_entry({
        "schema_version": SCHEMA_VERSION,
        "id": entry_id,
        "created_at": stamp,
        "editor": editor,
        "status": status,
        "verification_status": verification_status,
        "summary": summary,
        "changes": list(changes),
        "files": list(files),
        "validation": list(validation),
        "follow_ups": list(follow_ups),
        "base_commit": base_commit,
        "commit": commit,
        "branch": branch,
        "supersedes": supersedes,
    })

    destination_dir = Path(entry_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{entry_id}.json"
    if destination.exists():
        raise FileExistsError(f"編集ログが既に存在します: {destination.name}")
    payload = (json.dumps(entry, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8")
    if len(payload) > MAX_ENTRY_BYTES:
        raise EditLogValidationError(
            f"編集ログは {MAX_ENTRY_BYTES} bytes 以内で指定してください")

    temp_path: Path | None = None
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    lock_fd: int | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination_dir,
                prefix=f".{entry_id}.", suffix=".tmp", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        # ロックを排他的に作成して同じIDの並行公開を防ぎ、同一ディレクトリの
        # 一時ファイルを原子的に公開する。os.replaceはWindows/macOS/Linux対応。
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        if destination.exists():
            raise FileExistsError(f"編集ログが既に存在します: {destination.name}")
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_fd is not None and lock_path.exists():
            try:
                lock_path.unlink()
            except OSError:
                pass
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return destination, entry


def load_entries(entry_dir: str | Path = ENTRY_DIR) -> dict:
    """有効なログを新しい順に読み、破損ファイルは警告として分離する。"""
    directory = Path(entry_dir)
    if not directory.exists():
        return {"entries": [], "warnings": []}
    if directory.is_symlink():
        return {
            "entries": [],
            "warnings": ["編集ログの保存先がシンボリックリンクのため読み込めません"],
        }
    if not directory.is_dir():
        return {
            "entries": [],
            "warnings": ["編集ログの保存先がディレクトリではありません"],
        }

    entries: list[dict] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError as exc:
        return {
            "entries": [],
            "warnings": [f"編集ログの一覧を読み込めません: {exc}"],
        }
    for path in paths:
        try:
            if path.is_symlink():
                raise EditLogValidationError("シンボリックリンクは読み込めません")
            if path.stat().st_size > MAX_ENTRY_BYTES:
                raise EditLogValidationError(
                    f"ファイルサイズが上限 {MAX_ENTRY_BYTES} bytes を超えています")
            raw = json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
            entry = normalize_entry(raw)
            if path.stem != entry["id"]:
                raise EditLogValidationError("ファイル名とidが一致しません")
            if entry["id"] in seen_ids:
                raise EditLogValidationError("idが重複しています")
        except (OSError, UnicodeError, json.JSONDecodeError,
                EditLogValidationError) as exc:
            warnings.append(f"{path.name}: {exc}")
            continue
        seen_ids.add(entry["id"])
        entries.append(entry)
    entries.sort(key=lambda item: (item["created_at"], item["id"]), reverse=True)
    available_ids = {entry["id"] for entry in entries}
    for entry in entries:
        target = entry.get("supersedes")
        if target and target not in available_ids:
            warnings.append(
                f"{entry['id']}.json: 訂正対象 {target} が見つかりません")
    return {"entries": entries, "warnings": warnings}


def filter_entries(entries: Iterable[Mapping[str, object]], *,
                   editor: str | None = None, status: str | None = None,
                   query: str | None = None) -> list[dict]:
    """編集者・状態・文字列で絞り込み、入力順を維持する。"""
    editor_filter = str(editor or "").strip().lower()
    status_filter = str(status or "").strip().lower()
    query_filter = str(query or "").strip().casefold()
    result: list[dict] = []
    for source in entries:
        entry = dict(source)
        if editor_filter and entry.get("editor") != editor_filter:
            continue
        if status_filter and entry.get("status") != status_filter:
            continue
        if query_filter:
            values = [
                entry.get("summary"), entry.get("editor"), entry.get("branch"),
                entry.get("base_commit"), entry.get("commit"),
                entry.get("supersedes"), entry.get("verification_status"),
                *(entry.get("changes") or []), *(entry.get("files") or []),
                *(entry.get("validation") or []),
                *(entry.get("follow_ups") or []),
            ]
            haystack = "\n".join(str(value or "") for value in values).casefold()
            if query_filter not in haystack:
                continue
        result.append(entry)
    return result


def format_created_at_jst(value: str) -> str:
    """UTC保存時刻を日本時間の読みやすい形式へ変換する。"""
    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    stamp = datetime.fromisoformat(raw).astimezone(ZoneInfo("Asia/Tokyo"))
    return stamp.strftime("%Y/%m/%d %H:%M JST")
