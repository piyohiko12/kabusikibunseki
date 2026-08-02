"""ウォッチリスト(ティッカーのリスト)のCSV永続化。"""

from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "watchlist.csv"


def load() -> list[str]:
    if not DATA_FILE.exists():
        return []
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        return []
    if "ticker" not in df.columns:
        return []
    return [str(t).strip().upper() for t in df["ticker"].dropna() if str(t).strip()]


def save(tickers: list[str]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in cleaned:
            cleaned.append(t)
    pd.DataFrame({"ticker": cleaned}).to_csv(DATA_FILE, index=False)
