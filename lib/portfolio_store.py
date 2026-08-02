"""ポートフォリオのCSV永続化。"""

from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "portfolio.csv"
COLUMNS = ["ticker", "shares", "avg_cost"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"ticker": pd.Series(dtype=str),
                         "shares": pd.Series(dtype=float),
                         "avg_cost": pd.Series(dtype=float)})


def load() -> pd.DataFrame:
    if not DATA_FILE.exists():
        return _empty()
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        return _empty()
    return df.reindex(columns=COLUMNS)


def save(df: pd.DataFrame) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=COLUMNS).to_csv(DATA_FILE, index=False)


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """編集結果を検証・正規化する。戻り値: (正規化済みDataFrame, エラーメッセージのリスト)"""
    errors: list[str] = []
    df = df.reindex(columns=COLUMNS).copy()

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[~df["ticker"].isin(["", "NAN", "NONE"])]

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")

    for row in df.itertuples():
        if pd.isna(row.shares) or row.shares <= 0:
            errors.append(f"{row.ticker}: 株数は0より大きい数値を入力してください。")
        if pd.isna(row.avg_cost) or row.avg_cost <= 0:
            errors.append(f"{row.ticker}: 取得単価は0より大きい数値を入力してください。")

    dup = df["ticker"][df["ticker"].duplicated()].unique()
    for t in dup:
        errors.append(f"{t}: 同じティッカーが複数行あります。1行にまとめてください。")

    return df.reset_index(drop=True), errors
