"""Проверки разбора времени в data.py."""
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
