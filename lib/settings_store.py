"""アプリ設定(既定銘柄など)のJSON永続化。"""

import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "settings.json"


def load() -> dict:
    if not DATA_FILE.exists():
        return {}
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(**updates) -> dict:
    """キーワード引数で渡された設定をマージして保存する。"""
    settings = load()
    settings.update(updates)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return settings
