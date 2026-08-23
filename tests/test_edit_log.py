import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from lib import edit_log
from scripts import add_edit_log


def _entry(**overrides):
    value = {
        "schema_version": 1,
        "id": "20260823T120000Z-codex-12345678",
        "created_at": "2026-08-23T21:00:00+09:00",
        "editor": "codex",
        "status": "completed",
        "verification_status": "passed",
        "summary": "編集ログを追加",
        "changes": ["日本語の変更内容を記録"],
        "files": ["lib/edit_log.py"],
        "validation": ["ユニットテスト成功"],
        "follow_ups": [],
        "base_commit": "86741c6",
        "commit": None,
        "branch": "feature/edit-log",
        "supersedes": None,
    }
    value.update(overrides)
    return value


class EntryValidationTests(unittest.TestCase):
    def test_unicode_round_trip_utc_normalization_and_jst_display(self):
        raw = _entry()
        original = copy.deepcopy(raw)
        normalized = edit_log.normalize_entry(raw)
        self.assertEqual(raw, original)
        self.assertEqual(normalized["summary"], "編集ログを追加")
        self.assertEqual(normalized["created_at"], "2026-08-23T12:00:00Z")
        self.assertEqual(
            edit_log.format_created_at_jst(normalized["created_at"]),
            "2026/08/23 21:00 JST",
        )

    def test_schema_requires_exact_integer_and_known_fields(self):
        for raw in (
            _entry(schema_version=True),
            _entry(schema_version=2),
            _entry(unexpected="秘密を置けない"),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(edit_log.EditLogValidationError):
                    edit_log.normalize_entry(raw)

    def test_validation_is_required_when_marked_as_checked(self):
        with self.assertRaises(edit_log.EditLogValidationError):
            edit_log.normalize_entry(_entry(validation=[]))
        value = edit_log.normalize_entry(_entry(
            verification_status="not_run", validation=[]))
        self.assertEqual(value["validation"], [])

    def test_paths_must_be_public_repo_relative_posix_paths(self):
        invalid = (
            "/Users/name/project.py", "C:/Users/name/project.py",
            r"\\server\share\project.py", "../project.py", "a/../../project.py",
            "./project.py", ".git/config", ".venv/bin/python",
            "data/settings.json", "reports/result.json", "a\nb.py",
        )
        for path in invalid:
            with self.subTest(path=path):
                with self.assertRaises(edit_log.EditLogValidationError):
                    edit_log.normalize_entry(_entry(files=[path]))

    def test_files_are_required_and_duplicates_are_rejected(self):
        for files in ([], ["lib/edit_log.py", "lib/edit_log.py"]):
            with self.subTest(files=files):
                with self.assertRaises(edit_log.EditLogValidationError):
                    edit_log.normalize_entry(_entry(files=files))

    def test_control_and_bidirectional_characters_are_rejected(self):
        for summary in ("改行\nあり", "見た目\u202e反転"):
            with self.subTest(summary=summary):
                with self.assertRaises(edit_log.EditLogValidationError):
                    edit_log.normalize_entry(_entry(summary=summary))

    def test_sensitive_values_and_local_paths_are_rejected_in_text_fields(self):
        invalid = (
            {"summary": "ログは /Users/person/project に保存"},
            {"changes": [r"C:\Users\person\secret.txt を参照"]},
            {"validation": ["api_key=super-secret-value"]},
            {"follow_ups": ["token: abcdefghijklmnop"]},
            {"branch": "password=hunter2"},
            {"summary": "GitHub token ghp_abcdefghijklmnop"},
            {"summary": "口座ID: 123456"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaises(edit_log.EditLogValidationError):
                    edit_log.normalize_entry(_entry(**overrides))

    def test_naive_time_and_self_supersedes_are_rejected(self):
        with self.assertRaises(edit_log.EditLogValidationError):
            edit_log.normalize_entry(_entry(created_at="2026-08-23T12:00:00"))
        raw = _entry()
        raw["supersedes"] = raw["id"]
        with self.assertRaises(edit_log.EditLogValidationError):
            edit_log.normalize_entry(raw)

    def test_markdown_and_html_are_rendered_as_plain_text(self):
        value = edit_log.escape_markdown(
            "<script>alert(1)</script>\n![画像](https://example.com/a.png)")
        self.assertNotIn("<script>", value)
        self.assertNotIn("![画像]", value)
        self.assertIn(r"\<script\>", value)
        self.assertIn(r"\!\[画像\]", value)


class EditLogPersistenceTests(unittest.TestCase):
    def test_same_second_appends_are_distinct_and_sorted(self):
        with TemporaryDirectory() as folder:
            now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
            first_path, first = edit_log.append_entry(
                editor="codex", summary="一件目", changes=["変更1"],
                files=["lib/a.py"], now=now, entry_dir=folder)
            second_path, second = edit_log.append_entry(
                editor="claude", summary="二件目", changes=["変更2"],
                files=["lib/b.py"], now=now, entry_dir=folder)
            self.assertNotEqual(first_path, second_path)
            self.assertNotEqual(first["id"], second["id"])
            report = edit_log.load_entries(folder)
            self.assertEqual(len(report["entries"]), 2)
            self.assertEqual(report["warnings"], [])

    def test_atomic_failure_preserves_existing_entries_and_cleans_temp(self):
        with TemporaryDirectory() as folder:
            first_path, _ = edit_log.append_entry(
                editor="codex", summary="既存", changes=["既存変更"],
                files=["lib/existing.py"], entry_dir=folder)
            before = first_path.read_bytes()
            with mock.patch.object(edit_log.os, "replace", side_effect=OSError("failed")):
                with self.assertRaises(OSError):
                    edit_log.append_entry(
                        editor="claude", summary="失敗", changes=["失敗変更"],
                        files=["lib/new.py"], entry_dir=folder)
            self.assertEqual(first_path.read_bytes(), before)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)

    def test_writer_rejects_payload_that_loader_would_hide_as_oversize(self):
        with TemporaryDirectory() as folder:
            huge_changes = ["あ" * 500 for _ in range(100)]
            huge_validation = ["確認" * 250 for _ in range(100)]
            with self.assertRaises(edit_log.EditLogValidationError):
                edit_log.append_entry(
                    editor="codex", summary="巨大ログ", changes=huge_changes,
                    files=["lib/a.py"], validation=huge_validation,
                    verification_status="passed", entry_dir=folder)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_id_collision_does_not_overwrite(self):
        with TemporaryDirectory() as folder:
            now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
            fake_uuid = mock.Mock(hex="12345678abcdef")
            with mock.patch.object(edit_log.uuid, "uuid4", return_value=fake_uuid):
                path, _ = edit_log.append_entry(
                    editor="codex", summary="既存", changes=["変更"],
                    files=["lib/a.py"], now=now, entry_dir=folder)
                before = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    edit_log.append_entry(
                        editor="codex", summary="上書き不可", changes=["変更"],
                        files=["lib/b.py"], now=now, entry_dir=folder)
            self.assertEqual(path.read_bytes(), before)

    def test_existing_parallel_writer_lock_is_not_removed(self):
        with TemporaryDirectory() as folder:
            now = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
            fake_uuid = mock.Mock(hex="12345678abcdef")
            entry_id = "20260823T120000Z-codex-12345678"
            lock = Path(folder) / f"{entry_id}.json.lock"
            lock.write_text("other writer", encoding="utf-8")
            with mock.patch.object(edit_log.uuid, "uuid4", return_value=fake_uuid):
                with self.assertRaises(FileExistsError):
                    edit_log.append_entry(
                        editor="codex", summary="並行", changes=["変更"],
                        files=["lib/a.py"], now=now, entry_dir=folder)
            self.assertEqual(lock.read_text(encoding="utf-8"), "other writer")
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_corrupt_utf8_oversize_symlink_duplicate_and_nan_are_skipped(self):
        with TemporaryDirectory() as folder:
            valid_path, _ = edit_log.append_entry(
                editor="codex", summary="正常", changes=["変更"],
                files=["lib/a.py"], entry_dir=folder)
            directory = Path(folder)
            (directory / "broken.json").write_text("{", encoding="utf-8")
            (directory / "utf8.json").write_bytes(b"\xff\xfe")
            with (directory / "large.json").open("wb") as handle:
                handle.truncate(edit_log.MAX_ENTRY_BYTES + 1)
            expected_warnings = 5
            try:
                os.symlink(valid_path, directory / "linked.json")
            except (OSError, NotImplementedError):
                pass
            else:
                expected_warnings += 1
            duplicate = json.dumps(_entry(), ensure_ascii=False)
            duplicate = duplicate.replace(
                '"schema_version": 1',
                '"schema_version": 1, "schema_version": 1',
                1,
            )
            (directory / "duplicate.json").write_text(duplicate, encoding="utf-8")
            nan_value = json.dumps(_entry(), ensure_ascii=False)[:-1] + ', "x": NaN}'
            (directory / "nan.json").write_text(nan_value, encoding="utf-8")
            report = edit_log.load_entries(folder)
            self.assertEqual([item["summary"] for item in report["entries"]], ["正常"])
            self.assertEqual(len(report["warnings"]), expected_warnings)

    def test_directory_symlink_and_non_directory_fail_closed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "target"
            target.mkdir()
            link = root / "entries"
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError):
                pass
            else:
                self.assertEqual(edit_log.load_entries(link)["entries"], [])
                self.assertTrue(edit_log.load_entries(link)["warnings"])
            regular_file = root / "file"
            regular_file.write_text("x", encoding="utf-8")
            self.assertTrue(edit_log.load_entries(regular_file)["warnings"])

    def test_missing_superseded_entry_is_reported_without_hiding_entry(self):
        with TemporaryDirectory() as folder:
            _, item = edit_log.append_entry(
                editor="claude", summary="訂正", changes=["訂正内容"],
                files=["lib/a.py"], supersedes="missing-entry-1234",
                entry_dir=folder)
            report = edit_log.load_entries(folder)
            self.assertEqual(report["entries"][0]["id"], item["id"])
            self.assertIn("見つかりません", report["warnings"][0])

    def test_filter_combines_editor_status_and_japanese_query(self):
        entries = [
            _entry(summary="チャート改善", editor="codex"),
            _entry(id="20260823T130000Z-claude-abcdef12", editor="claude",
                   status="in_progress", summary="アラート修正"),
        ]
        result = edit_log.filter_entries(
            entries, editor="claude", status="in_progress", query="アラート")
        self.assertEqual([item["editor"] for item in result], ["claude"])


class EditLogCliTests(unittest.TestCase):
    def _args(self):
        return [
            "--editor", "codex", "--summary", "CLI追加",
            "--change", "ログを追加", "--file", "lib/edit_log.py",
            "--verification-status", "passed",
            "--validation", "テスト成功",
        ]

    def test_cli_success_uses_zero_and_never_prints_absolute_temp_path(self):
        with TemporaryDirectory() as folder:
            stdout = io.StringIO()
            with mock.patch.object(add_edit_log, "_git_value",
                                   side_effect=["86741c6", "feature/log"]), \
                    mock.patch("sys.stdout", stdout):
                code = add_edit_log.main(self._args(), entry_dir=folder)
            self.assertEqual(code, 0)
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)
            self.assertNotIn(folder, stdout.getvalue())
            self.assertIn("docs/edit-log/entries/", stdout.getvalue())

    def test_cli_validation_error_is_two(self):
        args = self._args()
        args[args.index("lib/edit_log.py")] = "data/settings.json"
        with mock.patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(add_edit_log.main(args), 2)

    def test_cli_io_error_is_one(self):
        with mock.patch.object(edit_log, "append_entry", side_effect=OSError("disk")), \
                mock.patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(add_edit_log.main(self._args()), 1)


if __name__ == "__main__":
    unittest.main()
