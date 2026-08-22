import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from lib import settings_store


class SettingsStoreSafetyTests(unittest.TestCase):
    def test_non_mapping_json_is_ignored(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            with mock.patch.object(settings_store, "DATA_FILE", path):
                for value in (None, [], "bad"):
                    path.write_text(json.dumps(value), encoding="utf-8")
                    self.assertEqual(settings_store.load(), {})

    def test_invalid_runtime_values_are_removed(self):
        raw = {
            "moomoo_enabled": "yes",
            "moomoo_port": "bad",
            "moomoo_history_reserve": -1,
            "advanced_chart": [],
            "default_ticker": " skhy ",
        }
        cleaned = settings_store._normalize(raw)
        self.assertEqual(cleaned, {"default_ticker": "SKHY"})

    def test_numeric_strings_are_normalized_and_saved_safely(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            with mock.patch.object(settings_store, "DATA_FILE", path):
                saved = settings_store.save(
                    moomoo_port="11111", moomoo_history_reserve="10",
                    advanced_chart={"preset": "スイング"})
                self.assertEqual(saved["moomoo_port"], 11111)
                self.assertEqual(saved["moomoo_history_reserve"], 10)
                self.assertIsInstance(settings_store.load()["advanced_chart"], dict)


if __name__ == "__main__":
    unittest.main()
