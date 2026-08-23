from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EditLogUiContractTests(unittest.TestCase):
    def test_page_is_registered_and_read_only(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        page = (ROOT / "views" / "edit_log.py").read_text(encoding="utf-8")
        self.assertIn('st.Page("views/edit_log.py", title="編集ログ"', app)
        self.assertIn("edit_log_lib.load_entries", page)
        self.assertNotIn("unsafe_allow_html=True", page)
        for forbidden in (
            "append_entry", "settings_store.save", "data_fetcher",
            "moomoo", "yfinance", "place_order", "subprocess",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, page)

    def test_every_dynamic_log_text_uses_plain_rendering(self):
        page = (ROOT / "views" / "edit_log.py").read_text(encoding="utf-8")
        self.assertIn("edit_log_lib.escape_markdown(entry[\"summary\"])", page)
        self.assertIn("edit_log_lib.escape_markdown(f\"・{change}\")", page)
        self.assertIn("st.code(\"\\n\".join(entry[\"files\"])", page)
        self.assertNotIn('left.subheader(entry["summary"]', page)
        self.assertNotIn("st.write(f\"・{change}\")", page)

    def test_cli_example_has_no_diff_markers(self):
        page = (ROOT / "views" / "edit_log.py").read_text(encoding="utf-8")
        self.assertNotIn(r"\n+  --editor", page)
        self.assertIn(r"\n  --editor codex", page)

    def test_both_agents_share_the_same_protocol(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
        guide = (ROOT / "docs" / "edit-log" / "README.md").read_text(
            encoding="utf-8")
        self.assertIn("scripts/add_edit_log.py --editor codex", agents)
        self.assertIn("scripts/add_edit_log.py --editor claude", claude)
        self.assertIn(r".venv\Scripts\python.exe", agents)
        self.assertIn(r".venv\Scripts\python.exe", claude)
        self.assertIn("1つのまとまった作業につき", guide)
        self.assertIn("秘密情報", guide)
        self.assertIn("同じコミット", guide)

    def test_log_destination_is_not_ignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertNotIn("docs/edit-log", gitignore)
        self.assertIn("data/", gitignore)


if __name__ == "__main__":
    unittest.main()
