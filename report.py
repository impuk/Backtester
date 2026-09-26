"""Таблицы и графики для интерфейса."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import engine as E

# Палитра (валидированная эталонная палитра dataviz-навыка)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GOOD = "#0ca30c"
CRIT = "#d03b3b"
GRAY_MID = "#f0efec"
DIVERGING = [[0.0, "#b3261e"], [0.25, "#e34948"], [0.5, GRAY_MID],
             [0.75, "#3987e5"], [1.0, "#184f95"]]
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]

REASON_NAMES = {E.REASON_OPEN: "открыта", E.REASON_TP: "TP", E.REASON_SL: "SL"}

METRIC_COLS = [
    (E.M_PROFIT, "Прибыль USDT"),
    (E.M_PROFIT_PCT, "Прибыль %"),
    (E.M_TRADES, "Сделок"),
    (E.M_WINRATE, "Винрейт %"),
    (E.M_MAXDD_PCT, "Макс. просадка %"),
    (E.M_MAXDD_USDT, "Макс. просадка USDT"),
    (E.M_PF, "Profit factor"),
    (E.M_AVG_ENTRIES, "Ср. входов в позиции"),
    (E.M_MAX_ENTRIES, "Макс. входов"),
    (E.M_MAX_CAPITAL, "Макс. задействовано USDT"),
    (E.M_EXECUTED, "Исполнено входов"),
    (E.M_SKIPPED, "Пропущено сигналов"),
    (E.M_OPEN_END, "Открыто на конце"),
    (E.M_OPEN_UNREAL, "Нереализ. на конце USDT"),
    (E.M_TIME_IN_MKT, "Время в рынке %"),
]
TEST_PREFIX = "Проверка: "


def max_entries_label(mode):
    return "Макс. входов в позиции" if mode == E.MODE_AVERAGE else "Макс. одновременно открытых"


def grid_frame(tps, sls, metrics, mode, test_metrics=None):
    df = pd.DataFrame({"TP %": np.round(tps * 100, 2), "SL %": np.round(sls * 100, 2)})
    for idx, name in METRIC_COLS:
        if idx == E.M_AVG_ENTRIES and mode != E.MODE_AVERAGE:
            continue
        col = max_entries_label(mode) if idx == E.M_MAX_ENTRIES else name
        df[col] = metrics[:, idx]
    if test_metrics is not None:
        for idx, name in METRIC_COLS:
            if idx == E.M_AVG_ENTRIES and mode != E.MODE_AVERAGE:
                continue
            col = max_entries_label(mode) if idx == E.M_MAX_ENTRIES else name
            df[TEST_PREFIX + col] = test_metrics[:, idx]
    return df


def apply_filters(df, min_trades, max_dd):
    mask = np.ones(len(df), bool)
    if min_trades is not None:
        mask &= df["Сделок"].to_numpy() >= min_trades
    if max_dd is not None:
        mask &= df["Макс. просадка %"].to_numpy() <= max_dd
    return mask


# ------------------------------------------------------------------ heatmaps
def pivot(df, col):
    p = df.pivot(index="SL %", columns="TP %", values=col)
    return p.sort_index().sort_index(axis=1)


def stability(p):
    """Среднее по клетке и соседям ±1 шаг (без учёта пустых)."""
    a = p.to_numpy(dtype=float)
    a = np.where(np.isfinite(a), a, np.nan)
    pad = np.pad(a, 1, constant_values=np.nan)
    stack = np.stack([pad[i:i + a.shape[0], j:j + a.shape[1]]
                      for i in range(3) for j in range(3)])
    with np.errstate(all="ignore"):
        cnt = np.isfinite(stack).sum(0)
        s = np.nansum(stack, 0)
        m = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(m, index=p.index, columns=p.columns)


def heatmap(p, title, diverging, fmt=".2f", best=None):
    z = p.to_numpy(dtype=float)
    z = np.where(np.isfinite(z), z, np.nan)
    kw = {}
    if diverging:
        lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
        kw = dict(colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim)
    else:
        kw = dict(colorscale=SEQUENTIAL)
    fig = go.Figure(go.Heatmap(
        z=z, x=p.columns.astype(float), y=p.index.astype(float), xgap=1, ygap=1,
        hovertemplate="TP %{x}%<br>SL %{y}%<br>" + title + ": %{z:" + fmt + "}<extra></extra>",
        colorbar=dict(title=""), **kw))
    if best is not None:
        fig.add_trace(go.Scatter(x=[best[0]], y=[best[1]], mode="markers",
                                 marker=dict(symbol="star", size=14, color="#ffffff",
                                             line=dict(color="#0b0b0b", width=1.5)),
                                 name="Лучшая", hoverinfo="skip", showlegend=False))
    fig.update_layout(title=title, xaxis_title="TP, %", yaxis_title="SL, %",
                      height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ------------------------------------------------------------------ trades
def trades_frame(det, times, mode, tf_minutes):
    t = det["trades"]
    n = len(t["entry_bar"])
    if n == 0:
        return pd.DataFrame()
    tarr = times.to_numpy()
    eb = t["entry_bar"]
    xb = t["exit_bar"]
    closed = xb >= 0
    exit_time = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_time[closed] = tarr[xb[closed]]
    last = len(tarr) - 1
    dur = np.where(closed, xb - eb, last - eb)
    fills = det["fills"]
    df = pd.DataFrame({
        "№": np.arange(1, n + 1),
        "Вход": tarr[eb],
    })
    if mode == E.MODE_AVERAGE:
        cnt = np.bincount(fills["group"], minlength=n)[:n]
        df["Входов"] = cnt
        df["Средняя цена"] = t["entry_price"]
    else:
        df["Цена входа"] = t["entry_price"]
    df["Выход"] = exit_time
    df["Цена выхода"] = t["exit_price"]
    df["Причина"] = [REASON_NAMES[int(r)] for r in t["reason"]]
    df["Кол-во"] = t["qty"]
    df["Вложено USDT"] = t["cost"]
    df["PnL USDT"] = t["pnl"]
    df["PnL %"] = t["pnl"] / t["cost"] * 100
    df["Длительность, свечей"] = dur
    df["Длительность, ч"] = dur * tf_minutes / 60
    df["_entry_bar"] = eb
    df["_exit_bar"] = xb
    return df


def buy_hold(ds, capital, fee, slip):
    entry = ds.o[0] * (1 + slip)
    qty = capital / (1 + fee) / entry
    final = qty * ds.c[-1] * (1 - slip) * (1 - fee)
    return final - capital


def equity_figure(times, equity, capital, split_time=None):
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    step = max(1, len(equity) // 20000)  # прореживание для скорости браузера
    x = times.iloc[::step]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=x, y=equity[::step], name="Капитал", line=dict(color=BLUE, width=2),
                             hovertemplate="%{x}<br>%{y:,.2f} USDT<extra></extra>"), 1, 1)
    fig.add_hline(y=capital, line=dict(color="#8a8984", width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=dd[::step], name="Просадка", fill="tozeroy",
                             line=dict(color=CRIT, width=1.5),
                             hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>"), 2, 1)
    if split_time is not None:
        fig.add_vline(x=split_time, line=dict(color=ORANGE, width=2, dash="dash"))
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="Просадка, %", row=2, col=1)
    fig.update_layout(height=520, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                      hovermode="x unified")
    return fig


def candle_figure(df_bars, det, mode, tp, sl, trade_row, pad=60, max_bars=3000):
    """Свечи вокруг выбранной сделки с отметками входов, докупок и выходов."""
    n = len(df_bars)
    eb = int(trade_row["_entry_bar"])
    xb = int(trade_row["_exit_bar"])
    end_bar = xb if xb >= 0 else n - 1
    a = max(0, eb - pad)
    b = min(n - 1, end_bar + pad)
    truncated = False
    if b - a + 1 > max_bars:
        b = a + max_bars - 1
        truncated = True
    w = df_bars.iloc[a:b + 1]
    fig = go.Figure(go.Candlestick(x=w["time"], open=w["open"], high=w["high"], low=w["low"],
                                   close=w["close"], name="Цена",
                                   increasing_line_color="#8a8984", decreasing_line_color="#3b3a37",
                                   increasing_fillcolor="#ffffff", decreasing_fillcolor="#3b3a37"))
    times = df_bars["time"].to_numpy()
    t = det["trades"]
    f = det["fills"]
    # все входы в окне
    fm = (f["bar"] >= a) & (f["bar"] <= b)
    fig.add_trace(go.Scatter(x=times[f["bar"][fm]], y=f["price"][fm], mode="markers", name="Вход",
                             marker=dict(symbol="triangle-up", size=10, color=BLUE,
                                         line=dict(color="#ffffff", width=1))))
    # все выходы в окне
    xm = (t["exit_bar"] >= a) & (t["exit_bar"] <= b)
    for reason, color, sym, name in ((E.REASON_TP, GOOD, "circle", "Выход TP"),
                                     (E.REASON_SL, CRIT, "x", "Выход SL")):
        m = xm & (t["reason"] == reason)
        fig.add_trace(go.Scatter(x=times[t["exit_bar"][m]], y=t["exit_price"][m], mode="markers",
                                 name=name, marker=dict(symbol=sym, size=10, color=color,
                                                        line=dict(color="#ffffff", width=1))))
    # уровни выбранной сделки
    gi = int(trade_row["№"]) - 1
    gb = f["bar"][f["group"] == gi]
    ga = f["avg"][f["group"] == gi]
    stop = min(end_bar, b)
    xs, avg_y = [], []
    for j in range(len(gb)):
        s0 = gb[j]
        s1 = gb[j + 1] if j + 1 < len(gb) else stop
        if s0 > b:
            break
        xs += [times[s0], times[min(s1, b)], None]
        avg_y += [ga[j], ga[j], None]
    avg_arr = np.array([v if v is not None else np.nan for v in avg_y], dtype=float)
    lbl = "Средняя цена" if mode == E.MODE_AVERAGE else "Цена входа"
    fig.add_trace(go.Scatter(x=xs, y=avg_arr, mode="lines", name=lbl,
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=xs, y=avg_arr * (1 + tp), mode="lines", name="TP",
                             line=dict(color=GOOD, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=xs, y=avg_arr * (1 - sl), mode="lines", name="SL",
                             line=dict(color=CRIT, width=2, dash="dash")))
    fig.update_layout(height=560, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig, truncated


def pnl_histogram(tdf, bins=40):
    closed = tdf[tdf["Причина"] != "открыта"]["PnL USDT"].to_numpy()
    fig = go.Figure()
    if len(closed) == 0:
        return fig
    lo, hi = closed.min(), closed.max()
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    cnt, _ = np.histogram(closed, edges)
    mids = (edges[:-1] + edges[1:]) / 2
    for name, color, m in (("Прибыльные", BLUE, mids > 0), ("Убыточные", CRIT, mids <= 0)):
        fig.add_trace(go.Bar(x=mids[m], y=cnt[m], width=(edges[1] - edges[0]) * 0.9, name=name,
                             marker_color=color,
                             hovertemplate="PnL ≈ %{x:.2f} USDT<br>Сделок: %{y}<extra></extra>"))
    fig.update_layout(height=360, xaxis_title="PnL сделки, USDT", yaxis_title="Сделок",
                      margin=dict(l=10, r=10, t=10, b=10), bargap=0,
                      legend=dict(orientation="h", y=1.05, x=0))
    return fig


def monthly_returns(times, equity, capital):
    s = pd.Series(equity, index=pd.DatetimeIndex(times))
    me = s.resample("ME").last()
    prev = me.shift(1)
    prev.iloc[0] = capital
    r = (me / prev - 1) * 100
    tab = pd.DataFrame({"Год": r.index.year, "Месяц": r.index.month, "r": r.values})
    p = tab.pivot(index="Год", columns="Месяц", values="r")
    names = {1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр", 5: "Май", 6: "Июн", 7: "Июл",
             8: "Авг", 9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек"}
    p = p.rename(columns=names)
    ye = s.resample("YE").last()
    prev_y = ye.shift(1)
    prev_y.iloc[0] = capital
    yearly = pd.Series(((ye / prev_y - 1) * 100).values, index=ye.index.year)
    p["Год, %"] = yearly.reindex(p.index).values
    return p


def concurrency(det, n):
    """Сколько входов открыто на каждой свече (режим «Независимые»)."""
    t = det["trades"]
    d = np.zeros(n + 1, np.int64)
    np.add.at(d, t["entry_bar"], 1)
    xb = np.where(t["exit_bar"] >= 0, t["exit_bar"] + 1, n)
    np.add.at(d, xb, -1)
    return np.cumsum(d)[:n]


def bar_chart(x, y, xtitle, ytitle, color=BLUE):
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=color,
                           marker_line=dict(color="#ffffff", width=1),
                           hovertemplate=xtitle + ": %{x}<br>" + ytitle + ": %{y}<extra></extra>"))
    fig.update_layout(height=320, xaxis_title=xtitle, yaxis_title=ytitle,
                      margin=dict(l=10, r=10, t=10, b=10), xaxis_type="category")
    return fig


def stitched_pnl_figure(t_train, eq_train, t_test, eq_test, capital):
    p1 = eq_train - capital
    p2 = eq_test - capital + p1[-1]
    s1 = max(1, len(p1) // 15000)
    s2 = max(1, len(p2) // 15000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_train.iloc[::s1], y=p1[::s1], name="Подбор",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=t_test.iloc[::s2], y=p2[::s2], name="Проверка",
                             line=dict(color=ORANGE, width=2)))
    fig.add_vline(x=t_test.iloc[0], line=dict(color="#8a8984", width=1, dash="dash"))
    fig.update_layout(height=420, yaxis_title="Накопленный результат, USDT",
                      margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.05, x=0))
    return fig


# ------------------------------------------------------------------ export
def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            out = df.drop(columns=[c for c in df.columns if str(c).startswith("_")])
            out = out.replace([np.inf, -np.inf], np.nan)
            out.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


def to_csv(df) -> bytes:
    out = df.drop(columns=[c for c in df.columns if str(c).startswith("_")])
    return out.to_csv(index=False).encode("utf-8-sig")


def return_cell_style(v):
    """Цвет ячейки помесячной доходности: синий — плюс, красный — минус."""
    if v is None or not np.isfinite(v) or v == 0:
        return ""
    a = min(abs(v) / 5.0, 1.0) * 0.55 + 0.08  # насыщенность растёт до ±5%
    rgb = "42,120,214" if v > 0 else "227,73,72"
    return f"background-color: rgba({rgb},{a:.2f})"


INT_COLS = {"Сделок", "Исполнено входов", "Пропущено сигналов", "Открыто на конце",
            "Макс. входов в позиции", "Макс. одновременно открытых"}


def column_format(col):
    base = col[len(TEST_PREFIX):] if col.startswith(TEST_PREFIX) else col
    return "%d" if base in INT_COLS else "%.2f"
