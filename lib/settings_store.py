"""アプリ設定(既定銘柄など)のJSON永続化。"""

import json
import math
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "settings.json"

_ADV_ENUMS = {
    "preset": {"標準", "デイトレ", "スイング", "長期"},
    "chart_type": {"ローソク足", "平均足", "OHLCバー", "ライン", "エリア"},
    "bar_label": {"1分", "5分", "15分", "1時間", "日足", "週足", "月足"},
    "interaction": {"クロスヘア", "ズーム", "移動", "ライン描画"},
    "theme": {"ダーク", "ライト"},
    "color_scheme": {"緑上昇 / 赤下落", "赤上昇 / 緑下落"},
}
_ADV_LISTS = {
    "overlays": {
        "移動平均線(SMA)", "指数移動平均線(EMA)", "VWAP(日中)",
        "ボリンジャーバンド", "一目均衡表", "サポレジライン",
        "フィボナッチ", "出来高プロファイル",
    },
    "oscillators": {"出来高", "RSI", "MACD", "ストキャスティクス"},
    "benchmarks": {"S&P500", "NASDAQ総合", "ダウ平均"},
}
_ADV_BOOLS = {
    "show_signals", "events", "current_price_line", "grid", "range_selector",
    "range_slider", "compact_sessions", "log_scale", "multi_timeframe",
    "show_order_book",
}
_INDICATOR_DEFAULTS = {
    "sma_periods": [20, 50, 200],
    "ema_periods": [20, 50],
    "boll_period": 20,
    "boll_std": 2.0,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "stoch_period": 14,
    "stoch_k": 3,
    "stoch_d": 3,
    "volume_ma": 20,
}


def _finite(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _bounded_int(value, low: int, high: int) -> int | None:
    number = _finite(value)
    if number is None or not number.is_integer():
        return None
    integer = int(number)
    return integer if low <= integer <= high else None


def _bounded_float(value, low: float, high: float) -> float | None:
    number = _finite(value)
    return number if number is not None and low <= number <= high else None


def _normalise_periods(value, defaults: list[int],
                       bounds: tuple[tuple[int, int], ...]) -> list[int]:
    """文字列・不足配列を既定値へ戻し、UIの添字参照を必ず安全にする。"""
    if not isinstance(value, (list, tuple)) or len(value) != len(defaults):
        return list(defaults)
    result = [_bounded_int(item, *limit) for item, limit in zip(value, bounds)]
    return ([int(item) for item in result] if all(item is not None for item in result)
            else list(defaults))


def _normalise_indicator_params(raw) -> dict:
    if not isinstance(raw, dict):
        return dict(_INDICATOR_DEFAULTS)
    clean = {
        "sma_periods": _normalise_periods(
            raw.get("sma_periods"), _INDICATOR_DEFAULTS["sma_periods"],
            ((2, 100), (5, 200), (20, 500))),
        "ema_periods": _normalise_periods(
            raw.get("ema_periods"), _INDICATOR_DEFAULTS["ema_periods"],
            ((2, 100), (3, 200))),
    }
    integer_bounds = {
        "boll_period": (5, 100), "rsi_period": (2, 50),
        "macd_fast": (2, 50), "macd_slow": (3, 100),
        "macd_signal": (2, 50), "stoch_period": (3, 50),
        "stoch_k": (1, 50), "stoch_d": (1, 50), "volume_ma": (2, 100),
    }
    for key, bounds in integer_bounds.items():
        clean[key] = (_bounded_int(raw.get(key), *bounds)
                      if key in raw else _INDICATOR_DEFAULTS[key])
        if clean[key] is None:
            clean[key] = _INDICATOR_DEFAULTS[key]
    boll_std = (_bounded_float(raw.get("boll_std"), 0.5, 4.0)
                if "boll_std" in raw else _INDICATOR_DEFAULTS["boll_std"])
    clean["boll_std"] = (_INDICATOR_DEFAULTS["boll_std"]
                         if boll_std is None else boll_std)
    # MACD長期はUI側と同様に短期より必ず1以上長くする。
    clean["macd_slow"] = max(clean["macd_fast"] + 1, clean["macd_slow"])
    return clean


def _normalise_advanced_chart(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    clean: dict = {}
    for key, allowed in _ADV_ENUMS.items():
        value = raw.get(key)
        if isinstance(value, str) and value in allowed:
            clean[key] = value
    for key, allowed in _ADV_LISTS.items():
        value = raw.get(key)
        if not isinstance(value, (list, tuple)):
            continue
        # 順序を維持したまま未知値と重複を除く。
        clean[key] = list(dict.fromkeys(
            item for item in value if isinstance(item, str) and item in allowed))
    for key in _ADV_BOOLS:
        if isinstance(raw.get(key), bool):
            clean[key] = raw[key]
    if "indicator_params" in raw:
        clean["indicator_params"] = _normalise_indicator_params(
            raw.get("indicator_params"))
    if "height" in raw:
        height = _bounded_int(raw.get("height"), 360, 780)
        if height in {360, 430, 520, 640, 780}:
            clean["height"] = height
    return clean


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
    if "advanced_chart" in settings:
        advanced = _normalise_advanced_chart(settings["advanced_chart"])
        if advanced is None:
            settings.pop("advanced_chart")
        else:
            settings["advanced_chart"] = advanced
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
