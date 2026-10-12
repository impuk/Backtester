"""Проверки чтения файлов и разбора времени в data.py."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

import data as D


def csv_bytes(times):
    rows = ["open_time_utc,open,high,low,close,long_signal"]
    for i, t in enumerate(times):
        rows.append(f"{t},{100 + i},{101 + i},{99 + i},{100.5 + i},{'true' if i % 2 else 'false'}")
    return ("\n".join(rows) + "\n").encode()


def check(times, expected):
    df = D.read_csv(csv_bytes(times))
    bars, bad = D.prepare(df, "open_time_utc", "long_signal")
    got = [t.strftime("%Y-%m-%d %H:%M:%S") for t in bars.time]
    assert got == sorted(expected), (got, expected)
    assert bad == 0


def test_new_format_day_first():
    # первая строка — день <= 12: раньше читалось как 1 мая
    check(["05-01-2022 00:00:00", "05-01-2022 01:00:00", "13-01-2022 00:00:00"],
          ["2022-01-05 00:00:00", "2022-01-05 01:00:00", "2022-01-13 00:00:00"])


def test_new_format_all_ambiguous():
    # во всём файле день <= 12 — должен остаться формат ДД-ММ-ГГГГ
    check(["01-02-2022 12:00:00", "02-02-2022 12:00:00", "12-02-2022 12:00:00"],
          ["2022-02-01 12:00:00", "2022-02-02 12:00:00", "2022-02-12 12:00:00"])


def test_old_format():
    check(["2026-03-27 06:45:00", "2026-03-27 07:00:00"],
          ["2026-03-27 06:45:00", "2026-03-27 07:00:00"])


def test_iso_fallback():
    check(["2026-03-27T06:45:00.000Z", "2026-03-27T07:00:00.000Z"],
          ["2026-03-27 06:45:00", "2026-03-27 07:00:00"])


def test_milliseconds():
    s = pd.Series([1643457600000, 1643461200000])
    got = D.parse_time(s).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    assert got == ["2022-01-29 12:00:00", "2022-01-29 13:00:00"]


def test_garbage_raises():
    try:
        D.parse_time(pd.Series(["вчера", "сегодня"]))
    except ValueError as e:
        assert "Не удалось разобрать время" in str(e)
    else:
        raise AssertionError("ожидалась ошибка")


def parquet_bytes(df):
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="zstd")
    return buf.getvalue()


def bingx_parquet(signal):
    # как у загрузчика BingX 2.0: время — метка UTC в мс, цены float64, без close_time
    t = pd.to_datetime([1643457600000, 1643461200000, 1643464800000], unit="ms", utc=True)
    df = pd.DataFrame({"open_time_utc": t.astype("datetime64[ms, UTC]"),
                       "open": [100.0, 101.0, 102.0], "high": [101.0, 102.0, 103.0],
                       "low": [99.0, 100.0, 101.0], "close": [100.5, 101.5, 102.5],
                       "volume_btc": [1.5, 2.5, 3.5]})
    if signal is not None:
        df["long_signal"] = signal
    return parquet_bytes(df)


def test_parquet_bool_signal():
    raw = bingx_parquet([False, True, False])
    df = D.read_file("BTC-USDT_1h.parquet", raw)
    assert D.guess_time_column(df) == "open_time_utc"
    assert D.signal_candidates(df) == ["long_signal"]
    bars, bad = D.prepare(df, "open_time_utc", "long_signal")
    assert bars.time.dt.tz is None
    assert [t.strftime("%d-%m-%Y %H:%M:%S") for t in bars.time] == [
        "29-01-2022 12:00:00", "29-01-2022 13:00:00", "29-01-2022 14:00:00"]
    assert bars.signal.tolist() == [False, True, False] and bad == 0
    assert D.timeframe_label(bars.time) == "1h"


def test_parquet_int_signal_and_magic():
    raw = bingx_parquet([0, 1, 1])
    df = D.read_file("без_расширения", raw)          # узнаётся по сигнатуре PAR1
    assert D.signal_candidates(df) == ["long_signal"]
    bars, _ = D.prepare(df, "open_time_utc", "long_signal")
    assert bars.signal.tolist() == [False, True, True]


def test_parquet_without_signal():
    df = D.read_file("BTC-USDT_1h.parquet", bingx_parquet(None))
    assert D.signal_candidates(df) == []


def test_csv_still_read_by_name():
    df = D.read_file("x.csv", csv_bytes(["05-01-2022 00:00:00", "05-01-2022 01:00:00"]))
    assert list(df.columns)[0] == "open_time_utc"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
