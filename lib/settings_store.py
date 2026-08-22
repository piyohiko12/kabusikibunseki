"""アプリ設定(既定銘柄など)のJSON永続化。"""

import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "settings.json"


def _normalize(raw) -> dict:
    """破損・手編集された値でアプリ全体を停止させない。"""
    if not isinstance(raw, dict):
        return {}
    settings = dict(raw)
    for key in ("moomoo_enabled", "moomoo_chart_history"):
        if key in settings and not isinstance(settings[key], bool):
            settings.pop(key)
    if "moomoo_host" in settings:
        host = settings["moomoo_host"]
        if not isinstance(host, str) or not host.strip():
            settings.pop("moomoo_host")
        else:
            settings["moomoo_host"] = host.strip()
    for key, lower, upper in (
            ("moomoo_port", 1, 65535),
            ("moomoo_history_reserve", 0, 2000)):
        if key not in settings:
            continue
        value = settings[key]
        if isinstance(value, bool):
            settings.pop(key)
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            settings.pop(key)
            continue
        if lower <= value <= upper:
            settings[key] = value
        else:
            settings.pop(key)
    if "advanced_chart" in settings and not isinstance(settings["advanced_chart"], dict):
        settings.pop("advanced_chart")
    if "default_ticker" in settings:
        ticker = settings["default_ticker"]
        if not isinstance(ticker, str) or not ticker.strip():
            settings.pop("default_ticker")
        else:
            settings["default_ticker"] = ticker.strip().upper()[:32]
    return settings


def load() -> dict:
    if not DATA_FILE.exists():
        return {}
    try:
        return _normalize(json.loads(DATA_FILE.read_text(encoding="utf-8")))
    except Exception:
        return {}


def save(**updates) -> dict:
    """キーワード引数で渡された設定をマージして保存する。"""
    settings = load()
    settings.update(updates)
    settings = _normalize(settings)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return settings
