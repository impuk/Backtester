"""Загрузка свечей и сигналов из CSV или Parquet."""
import io

import numpy as np
import pandas as pd

PRICE_COLS = ("open", "high", "low", "close")
TRUE_VALUES = {"true", "1", "1.0", "yes", "y", "t", "да"}
PARQUET_MAGIC = b"PAR1"


def read_csv(raw: bytes) -> pd.DataFrame:
    """Читает CSV, разделитель — запятая или точка с запятой."""
    head = raw[:4096].decode("utf-8", errors="ignore")
    first = head.splitlines()[0] if head else ""
    sep = ";" if first.count(";") > first.count(",") else ","
    df = pd.read_csv(io.BytesIO(raw), sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_parquet(raw: bytes) -> pd.DataFrame:
    """Читает Parquet (например, файл загрузчика BingX 2.0 с добавленной колонкой сигнала)."""
    df = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def is_parquet(name: str, raw: bytes) -> bool:
    return name.lower().endswith(".parquet") or raw[:4] == PARQUET_MAGIC


def read_file(name: str, raw: bytes) -> pd.DataFrame:
    """CSV или Parquet — по расширению файла (или по сигнатуре Parquet в начале файла)."""
    return read_parquet(raw) if is_parquet(name, raw) else read_csv(raw)


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
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            # без перевода в строки: на миллионах строк Parquet это заметно быстрее
            if s.isin([0, 1]).all():
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
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) == 1).to_numpy()
    return s.astype(str).str.strip().str.lower().isin(TRUE_VALUES).to_numpy()


# Явные форматы времени, по порядку проверки. Первый — формат загрузчика BingX
# (ДД-ММ-ГГГГ ЧЧ:ММ:СС), второй — прежние выгрузки (ГГГГ-ММ-ДД ЧЧ:ММ:СС).
TIME_FORMATS = (
    "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%Y-%m-%d %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%Y-%m-%dT%H:%M:%S",
)


def parse_time(s: pd.Series) -> pd.Series:
    """Разбирает время. Строки — только по явным форматам, без угадывания.

    pd.to_datetime без формата угадывает его по первой строке и при дне <= 12
    может тихо прочитать «05-01-2022» как 1 мая. Поэтому формат подбирается
    так, чтобы им разбирались ВСЕ строки колонки.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        # Parquet: уже метка времени; с часовым поясом (UTC у загрузчика) -> время UTC без пояса,
        # в тех же единицах (мкс), что даёт разбор строк CSV
        if s.dt.tz is not None:
            s = s.dt.tz_convert(None)
        return s.astype("datetime64[us]")
    if pd.api.types.is_numeric_dtype(s):
        unit = "ms" if s.iloc[0] > 1e11 else "s"
        return pd.to_datetime(s, unit=unit)
    txt = s.astype(str).str.strip()
    nonempty = txt[(txt != "") & (txt.str.lower() != "nan")]
    for fmt in TIME_FORMATS:
        parsed = pd.to_datetime(txt, format=fmt, errors="coerce")
        if parsed[nonempty.index].notna().all():
            return parsed
    # запасной вариант: ISO 8601 (например, с часовым поясом или долями секунды)
    try:
        return pd.to_datetime(txt, format="ISO8601", utc=True).dt.tz_localize(None)
    except (ValueError, TypeError):
        example = nonempty.iloc[0] if len(nonempty) else ""
        raise ValueError(
            f"Не удалось разобрать время (пример: «{example}»). Ожидается формат "
            "ДД-ММ-ГГГГ ЧЧ:ММ:СС или ГГГГ-ММ-ДД ЧЧ:ММ:СС.") from None


def to_number(s: pd.Series) -> pd.Series:
    """Число из колонки; понимает десятичную запятую (CSV из русского Excel)."""
    if not pd.api.types.is_numeric_dtype(s):
        s = (s.astype(str).str.strip().str.replace("\u00a0", "", regex=False)
             .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def prepare(df, time_col, signal_col):
    """Возвращает таблицу time/open/high/low/close/signal, отсортированную по времени."""
    pc = find_price_columns(df)
    out = pd.DataFrame({
        "time": parse_time(df[time_col]),
        "open": to_number(df[pc["open"]]),
        "high": to_number(df[pc["high"]]),
        "low": to_number(df[pc["low"]]),
        "close": to_number(df[pc["close"]]),
        "signal": parse_signal(df[signal_col]),
    })
    bad = int(out[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if out.empty:
        raise ValueError("после разбора не осталось ни одной свечи — проверьте колонки цен и времени")
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
