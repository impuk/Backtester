"""Загрузка CSV со свечами и сигналами."""
import io

import numpy as np
import pandas as pd

PRICE_COLS = ("open", "high", "low", "close")
TRUE_VALUES = {"true", "1", "1.0", "yes", "y", "t", "да"}


def read_csv(raw: bytes) -> pd.DataFrame:
    """Читает CSV, разделитель — запятая или точка с запятой."""
    head = raw[:4096].decode("utf-8", errors="ignore")
    first = head.splitlines()[0] if head else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(io.BytesIO(raw), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_price_columns(df):
    lower = {c.lower(): c for c in df.columns}
    missing = [p for p in PRICE_COLS if p not in lower]
    if missing:
        raise ValueError("В файле нет колонок: " + ", ".join(missing))
    return {p: lower[p] for p in PRICE_COLS}


def guess_time_column(df):
    cols = list(df.columns)
    for c in cols:
        if c.lower() in ("open_time_utc", "open_time", "time", "timestamp", "date", "datetime"):
            return c
    for c in cols:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return cols[0]


def signal_candidates(df):
    """Колонки, похожие на булевы сигналы."""
    out = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        if s.dtype == bool:
            out.append(c)
            continue
        vals = set(s.astype(str).str.strip().str.lower().unique()[:10])
        if vals and vals <= {"true", "false", "0", "1", "0.0", "1.0", "yes", "no", "t", "f"}:
            out.append(c)
    return out


def default_signal_column(cands):
    for c in cands:
        if c.lower() == "long_signal":
            return c
    return cands[0] if cands else None


def parse_signal(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.strip().str.lower().isin(TRUE_VALUES).to_numpy()


def parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.iloc[0] > 1e11 else "s"
        return pd.to_datetime(s, unit=unit)
    return pd.to_datetime(s)


def prepare(df, time_col, signal_col):
    """Возвращает таблицу time/open/high/low/close/signal, отсортированную по времени."""
    pc = find_price_columns(df)
    out = pd.DataFrame({
        "time": parse_time(df[time_col]),
        "open": pd.to_numeric(df[pc["open"]], errors="coerce"),
        "high": pd.to_numeric(df[pc["high"]], errors="coerce"),
        "low": pd.to_numeric(df[pc["low"]], errors="coerce"),
        "close": pd.to_numeric(df[pc["close"]], errors="coerce"),
        "signal": parse_signal(df[signal_col]),
    })
    bad = int(out[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out, bad


def timeframe_label(times: pd.Series) -> str:
    if len(times) < 2:
        return "?"
    d = times.diff().dropna().median()
    mins = d.total_seconds() / 60
    if mins < 60:
        return f"{mins:g}m"
    if mins < 1440:
        return f"{mins / 60:g}h"
    return f"{mins / 1440:g}d"
