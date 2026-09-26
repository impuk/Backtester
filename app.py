"""Бэктестер сигналов: перебор TP/SL по CSV со свечами BingX.

Запуск: streamlit run app.py
"""
import hashlib
import time

import numpy as np
import pandas as pd
import streamlit as st

import data as D
import engine as E
import report as R

st.set_page_config(page_title="Бэктестер сигналов", layout="wide")

MODES = {"Независимые": E.MODE_INDEPENDENT, "Средняя": E.MODE_AVERAGE}
VALIDATION = {"Все данные": "all", "Подбор + проверка": "split"}
HEAT_METRICS = {
    "Прибыль USDT": ("Прибыль USDT", True, ",.2f"),
    "Макс. просадка %": ("Макс. просадка %", False, ".2f"),
    "Винрейт %": ("Винрейт %", False, ".1f"),
    "Число сделок": ("Сделок", False, ",.0f"),
    "Profit factor": ("Profit factor", False, ".2f"),
}


@st.cache_data(show_spinner="Читаю CSV…", max_entries=3)
def load_raw(raw: bytes):
    return D.read_csv(raw)


@st.cache_data(show_spinner="Готовлю данные…", max_entries=3)
def load_prepared(raw: bytes, time_col: str, signal_col: str):
    return D.prepare(load_raw(raw), time_col, signal_col)


def value_range(a, b, step):
    if step <= 0 or b < a:
        return np.array([])
    n = int(round((b - a) / step)) + 1
    return np.round(a + np.arange(n) * step, 4)


def make_datasets(bars, validation, split_pct):
    full = E.Dataset(bars.open, bars.high, bars.low, bars.close, bars.signal)
    sets = {"Весь файл": (full, bars)}
    if validation == "split":
        k = int(len(bars) * split_pct / 100)
        tr = bars.iloc[:k].reset_index(drop=True)
        te = bars.iloc[k:].reset_index(drop=True)
        sets["Подбор"] = (E.Dataset(tr.open, tr.high, tr.low, tr.close, tr.signal), tr)
        sets["Проверка"] = (E.Dataset(te.open, te.high, te.low, te.close, te.signal), te)
    return sets


def run_with_progress(ds, tps, sls, p, bar, lo, hi):
    out = np.empty((len(tps), E.N_METRICS))
    chunk = 96
    for s in range(0, len(tps), chunk):
        out[s:s + chunk] = E.run_grid(ds, tps[s:s + chunk], sls[s:s + chunk], p["capital"],
                                      p["amount"], p["fee"], p["slip"], p["mode"])
        done = min(len(tps), s + chunk) / len(tps)
        bar.progress(lo + (hi - lo) * done, text=f"Перебор комбинаций… {int((lo + (hi - lo) * done) * 100)}%")
    return out


# ================================================================== sidebar
st.sidebar.title("Настройки")
up = st.sidebar.file_uploader("CSV со свечами и сигналом", type=["csv"])
if up is None:
    st.title("Бэктестер сигналов")
    st.info("Загрузите CSV слева. Нужны колонки open, high, low, close, время и колонка "
            "сигнала со значениями true/false.")
    st.stop()

raw = up.getvalue()
file_id = hashlib.md5(raw).hexdigest()
try:
    df_raw = load_raw(raw)
    D.find_price_columns(df_raw)
except Exception as ex:  # noqa: BLE001
    st.error(f"Не удалось прочитать файл: {ex}")
    st.stop()

cols = list(df_raw.columns)
time_col = st.sidebar.selectbox("Колонка времени", cols, index=cols.index(D.guess_time_column(df_raw)))
sig_cands = D.signal_candidates(df_raw)
if not sig_cands:
    st.error("Не нашёл колонку сигнала со значениями true/false.")
    st.stop()
sig_col = st.sidebar.selectbox("Колонка сигнала (long)", sig_cands,
                               index=sig_cands.index(D.default_signal_column(sig_cands)))
try:
    bars, bad_rows = load_prepared(raw, time_col, sig_col)
except Exception as ex:  # noqa: BLE001
    st.error(f"Ошибка разбора данных: {ex}")
    st.stop()
tf = D.timeframe_label(bars.time)
tf_minutes = bars.time.diff().dropna().median().total_seconds() / 60 if len(bars) > 1 else 1

st.sidebar.caption(f"Свечей: {len(bars):,} · таймфрейм {tf} · сигналов: {int(bars.signal.sum()):,}\n\n"
                   f"{bars.time.iloc[0]:%Y-%m-%d %H:%M} — {bars.time.iloc[-1]:%Y-%m-%d %H:%M}")
if bad_rows:
    st.sidebar.warning(f"Пропущено строк с пустыми ценами: {bad_rows}")

st.sidebar.subheader("Режимы")
mode_name = st.sidebar.radio("Выход из сделок", list(MODES), index=0,
                             help="Независимые — у каждого входа свои TP/SL. "
                                  "Средняя — TP/SL от средней цены, позиция закрывается целиком.")
val_name = st.sidebar.radio("Проверка", list(VALIDATION), index=0)
split_pct = 70
if VALIDATION[val_name] == "split":
    split_pct = st.sidebar.slider("Доля данных для подбора, %", 50, 90, 70, 5)

st.sidebar.subheader("Деньги")
capital = st.sidebar.number_input("Начальный капитал, USDT", 1.0, 1e9, 100000.0, 1000.0)
amount = st.sidebar.number_input("Сумма одного входа, USDT", 0.01, 1e9, 100.0, 10.0)
fee_pct = st.sidebar.number_input("Комиссия за сделку, %", 0.0, 5.0, 0.1, 0.01, format="%.3f",
                                  help="Берётся и при входе, и при выходе. BingX spot VIP0: 0,1%.")
slip_pct = st.sidebar.number_input("Проскальзывание, %", 0.0, 5.0, 0.0, 0.01, format="%.3f")

st.sidebar.subheader("Перебор TP / SL, %")
c1, c2, c3 = st.sidebar.columns(3)
tp_a = c1.number_input("TP от", 0.1, 100.0, 1.0, 0.1)
tp_b = c2.number_input("TP до", 0.1, 100.0, 10.0, 0.1)
tp_s = c3.number_input("TP шаг", 0.01, 10.0, 0.2, 0.05)
c1, c2, c3 = st.sidebar.columns(3)
sl_a = c1.number_input("SL от", 0.1, 99.0, 1.0, 0.1)
sl_b = c2.number_input("SL до", 0.1, 99.0, 10.0, 0.1)
sl_s = c3.number_input("SL шаг", 0.01, 10.0, 0.2, 0.05)
tp_vals = value_range(tp_a, tp_b, tp_s)
sl_vals = value_range(sl_a, sl_b, sl_s)
st.sidebar.caption(f"Комбинаций: {len(tp_vals) * len(sl_vals):,}")

st.sidebar.subheader("Фильтры лучшей комбинации")
f_trades_on = st.sidebar.checkbox("Минимум сделок", True)
f_trades = st.sidebar.number_input("Сделок не меньше", 1, 10_000_000, 30, 1, disabled=not f_trades_on)
f_dd_on = st.sidebar.checkbox("Максимальная просадка", True)
f_dd = st.sidebar.number_input("Просадка не больше, %", 0.1, 100.0, 30.0, 1.0, disabled=not f_dd_on)
top_n = st.sidebar.number_input("Строк в топе (N)", 5, 5000, 50, 5)

params = dict(file=file_id, time_col=time_col, sig_col=sig_col, mode=MODES[mode_name],
              validation=VALIDATION[val_name], split=split_pct, capital=capital, amount=amount,
              fee=fee_pct / 100, slip=slip_pct / 100,
              tps=tuple(tp_vals), sls=tuple(sl_vals))

run = st.sidebar.button("Запустить перебор", type="primary", width="stretch")

# ================================================================== run
if run:
    if len(tp_vals) == 0 or len(sl_vals) == 0:
        st.error("Проверьте диапазоны TP/SL.")
        st.stop()
    t0 = time.time()
    bar = st.progress(0.0, text="Подготовка… (первый запуск компилирует движок, это до минуты)")
    sets = make_datasets(bars, params["validation"], split_pct)
    TP, SL = np.meshgrid(tp_vals / 100, sl_vals / 100, indexing="ij")
    tps, sls = TP.ravel(), SL.ravel()
    if params["validation"] == "split":
        m_tr = run_with_progress(sets["Подбор"][0], tps, sls, params, bar, 0.0, 0.5)
        m_te = run_with_progress(sets["Проверка"][0], tps, sls, params, bar, 0.5, 1.0)
        grid = R.grid_frame(tps, sls, m_tr, params["mode"], m_te)
    else:
        m_all = run_with_progress(sets["Весь файл"][0], tps, sls, params, bar, 0.0, 1.0)
        grid = R.grid_frame(tps, sls, m_all, params["mode"])
    bar.empty()
    st.session_state["res"] = dict(params=params, grid=grid, sets=sets, elapsed=time.time() - t0,
                                   tf_minutes=tf_minutes)
    st.session_state.pop("sel", None)
    st.session_state["detail_cache"] = {}

res = st.session_state.get("res")
if res is None:
    st.title("Бэктестер сигналов")
    st.info("Настройте параметры и нажмите «Запустить перебор».")
    st.stop()
if res["params"] != params:
    st.warning("Настройки изменились после последнего перебора — результаты ниже для старых "
               "настроек. Нажмите «Запустить перебор», чтобы пересчитать.")

P = res["params"]
grid = res["grid"]
sets = res["sets"]
mode = P["mode"]
split = P["validation"] == "split"
main_period = "Подбор" if split else "Весь файл"

mask = R.apply_filters(grid, f_trades if f_trades_on else None, f_dd if f_dd_on else None)
passed = grid[mask]
best = None
if len(passed):
    best = passed.loc[passed["Прибыль USDT"].idxmax()]

st.title("Бэктестер сигналов")
st.caption(f"Режим: {[k for k, v in MODES.items() if v == mode][0]} · "
           f"{'подбор на ' + str(P['split']) + '% + проверка' if split else 'все данные'} · "
           f"комбинаций {len(grid):,} · расчёт {res['elapsed']:.1f} с")

tabs = st.tabs(["Перебор", "Комбинация", "Проверка 70/30" if split else "Проверка", "Экспорт"])

# ================================================================== tab: grid
with tabs[0]:
    if best is None:
        st.warning("Ни одна комбинация не прошла фильтры. Ослабьте или отключите фильтры слева.")
    else:
        st.markdown("**Лучшая комбинация**")
        cols4 = st.columns(5)
        cols4[0].metric("TP", f"{best['TP %']:g}%")
        cols4[1].metric("SL", f"{best['SL %']:g}%")
        cols4[2].metric("Прибыль, USDT" + (" (подбор)" if split else ""), f"{best['Прибыль USDT']:,.2f}",
                        f"{best['Прибыль %']:.2f}%")
        cols4[3].metric("Сделок", f"{int(best['Сделок']):,}")
        cols4[4].metric("Макс. просадка", f"{best['Макс. просадка %']:.2f}%")
        if split:
            st.caption(f"На проверочной части: прибыль {best[R.TEST_PREFIX + 'Прибыль USDT']:,.2f} USDT, "
                       f"сделок {int(best[R.TEST_PREFIX + 'Сделок'])}, "
                       f"просадка {best[R.TEST_PREFIX + 'Макс. просадка %']:.2f}%")
    st.caption(f"Прошли фильтры: {int(mask.sum()):,} из {len(grid):,}")

    h1, h2, h3 = st.columns([2, 2, 1])
    hm_name = h1.selectbox("Метрика тепловой карты", list(HEAT_METRICS))
    hm_src = h2.radio("Данные", ["Подбор", "Проверка"], horizontal=True) if split else "Подбор"
    hide = h3.checkbox("Скрыть не прошедшие фильтры", False)
    col, diverging, fmt = HEAT_METRICS[hm_name]
    src_col = (R.TEST_PREFIX + col) if hm_src == "Проверка" else col
    g2 = grid.copy()
    if hide:
        g2.loc[~mask, src_col] = np.nan
    best_xy = (best["TP %"], best["SL %"]) if best is not None else None
    p = R.pivot(g2, src_col)
    st.plotly_chart(R.heatmap(p, hm_name, diverging, fmt, best_xy), width="stretch")

    st.markdown("**Карта устойчивости** — средняя прибыль клетки и её соседей (±1 шаг TP и SL). "
                "Широкая синяя зона надёжнее одиночного пика.")
    pst = R.stability(R.pivot(g2, (R.TEST_PREFIX if hm_src == "Проверка" else "") + "Прибыль USDT"))
    st.plotly_chart(R.heatmap(pst, "Средняя прибыль соседей, USDT", True, ",.2f", best_xy),
                    width="stretch")

    st.subheader(f"Топ-{int(top_n)} комбинаций")
    s1, s2, s3 = st.columns([2, 1, 1])
    sort_cols = [c for c in grid.columns]
    sort_by = s1.selectbox("Отбирать топ по колонке", sort_cols, index=sort_cols.index("Прибыль USDT"))
    ascending = s2.radio("Порядок", ["по убыванию", "по возрастанию"], horizontal=True) == "по возрастанию"
    only_passed = s3.checkbox("Только прошедшие фильтры", True)
    base = passed if only_passed else grid
    top = base.sort_values(sort_by, ascending=ascending, na_position="last").head(int(top_n))
    top = top.reset_index(drop=True)
    num_cfg = {c: st.column_config.NumberColumn(format=R.column_format(c)) for c in top.columns
               if c not in ("TP %", "SL %")}
    ev = st.dataframe(top, width="stretch", hide_index=True, height=420,
                      on_select="rerun", selection_mode="single-row", column_config=num_cfg,
                      key="top_table")
    st.caption("Клик по заголовку колонки сортирует таблицу. Выберите строку, чтобы открыть "
               "её на вкладке «Комбинация».")
    rows = ev.selection.rows if ev is not None else []
    if rows:
        r = top.iloc[rows[0]]
        st.session_state["sel"] = (float(r["TP %"]), float(r["SL %"]))

# ================================================================== detail helpers
tp_list = sorted(grid["TP %"].unique())
sl_list = sorted(grid["SL %"].unique())
default_sel = st.session_state.get("sel") or ((best["TP %"], best["SL %"]) if best is not None
                                              else (tp_list[0], sl_list[0]))


def get_detail(period, tp, sl):
    key = (period, tp, sl)
    cache = st.session_state.setdefault("detail_cache", {})
    if key not in cache:
        ds, _ = sets[period]
        cache[key] = E.run_detail(ds, tp / 100, sl / 100, P["capital"], P["amount"], P["fee"],
                                  P["slip"], mode)
        if len(cache) > 12:
            cache.pop(next(iter(cache)))
    return cache[key]


# ================================================================== tab: combination
with tabs[1]:
    d1, d2, d3 = st.columns(3)
    tp_sel = d1.selectbox("TP, %", tp_list, index=tp_list.index(default_sel[0]))
    sl_sel = d2.selectbox("SL, %", sl_list, index=sl_list.index(default_sel[1]))
    periods = ["Весь файл", "Подбор", "Проверка"] if split else ["Весь файл"]
    period = d3.selectbox("Период", periods, index=0)
    ds, pbars = sets[period]
    det = get_detail(period, tp_sel, sl_sel)
    m = det["metrics"]
    bh = R.buy_hold(ds, P["capital"], P["fee"], P["slip"])

    k = st.columns(4)
    k[0].metric("Прибыль, USDT", f"{m[E.M_PROFIT]:,.2f}", f"{m[E.M_PROFIT_PCT]:.2f}%")
    k[1].metric("Сделок", f"{int(m[E.M_TRADES]):,}",
                f"винрейт {m[E.M_WINRATE]:.1f}%" if np.isfinite(m[E.M_WINRATE]) else None,
                delta_color="off", delta_arrow="off")
    k[2].metric("Макс. просадка", f"{m[E.M_MAXDD_PCT]:.2f}%", f"{m[E.M_MAXDD_USDT]:,.2f} USDT",
                delta_color="off", delta_arrow="off")
    pf = m[E.M_PF]
    k[3].metric("Profit factor", "∞" if np.isinf(pf) else (f"{pf:.2f}" if np.isfinite(pf) else "—"))
    k = st.columns(4)
    k[0].metric("«Купил и держал», USDT", f"{bh:,.2f}", f"{bh / P['capital'] * 100:.2f}%")
    k[1].metric("Макс. задействовано, USDT", f"{m[E.M_MAX_CAPITAL]:,.2f}")
    k[2].metric(R.max_entries_label(mode), f"{int(m[E.M_MAX_ENTRIES]):,}")
    k[3].metric("Время в рынке", f"{m[E.M_TIME_IN_MKT]:.1f}%")
    if m[E.M_OPEN_END] > 0:
        st.warning(f"На конце данных открыто входов: {int(m[E.M_OPEN_END])}. Нереализованный "
                   f"результат {m[E.M_OPEN_UNREAL]:,.2f} USDT — в прибыль не включён.")

    st.subheader("Кривая капитала и просадка")
    st.caption("Капитал с учётом открытых позиций по цене закрытия свечи.")
    split_time = sets["Проверка"][1].time.iloc[0] if (split and period == "Весь файл") else None
    st.plotly_chart(R.equity_figure(pbars.time, det["equity"], P["capital"], split_time),
                    width="stretch")

    tdf = R.trades_frame(det, pbars.time, mode, res["tf_minutes"])
    st.subheader("Сделки" if mode == E.MODE_INDEPENDENT else "Позиции")
    if tdf.empty:
        st.info("Сделок нет.")
    else:
        shown = tdf.drop(columns=["_entry_bar", "_exit_bar"])
        cfg = {c: st.column_config.NumberColumn(format="%.8f" if c == "Кол-во" else
                                                ("%.4f" if "цена" in c.lower() else "%.2f"))
               for c in shown.columns if shown[c].dtype.kind == "f"}
        tev = st.dataframe(shown, width="stretch", hide_index=True, height=360,
                           on_select="rerun", selection_mode="single-row", column_config=cfg,
                           key=f"trades_{period}_{tp_sel}_{sl_sel}")
        trow = tdf.iloc[tev.selection.rows[0]] if tev.selection.rows else tdf.iloc[0]
        st.subheader(f"График: {'сделка' if mode == E.MODE_INDEPENDENT else 'позиция'} №{int(trow['№'])}")
        st.caption("Выберите строку в таблице выше, чтобы перейти к другой сделке.")
        fig, trunc = R.candle_figure(pbars, det, mode, tp_sel / 100, sl_sel / 100, trow)
        if trunc:
            st.caption("Сделка длинная — показаны первые 3000 свечей окна.")
        st.plotly_chart(fig, width="stretch")

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Распределение PnL")
            st.plotly_chart(R.pnl_histogram(tdf), width="stretch")
        with g2:
            st.subheader("Докупки")
            if mode == E.MODE_AVERAGE:
                vc = tdf[tdf["Причина"] != "открыта"]["Входов"].value_counts().sort_index()
                st.plotly_chart(R.bar_chart(vc.index.astype(str), vc.values, "Входов в позиции",
                                            "Позиций"), width="stretch")
            else:
                conc = R.concurrency(det, len(pbars))
                vc = pd.Series(conc[conc > 0]).value_counts().sort_index()
                if len(vc) > 40:
                    vc = pd.Series(conc[conc > 0]).value_counts(bins=20).sort_index()
                    labels = [f"{int(i.left) + 1}–{int(i.right)}" for i in vc.index]
                else:
                    labels = vc.index.astype(str)
                st.plotly_chart(R.bar_chart(labels, vc.values, "Открыто входов одновременно",
                                            "Свечей"), width="stretch")
            st.caption(f"Макс. задействованный капитал: {m[E.M_MAX_CAPITAL]:,.2f} USDT "
                       f"({m[E.M_MAX_CAPITAL] / P['capital'] * 100:.1f}% капитала)")

        st.subheader("Помесячная доходность, %")
        mr = R.monthly_returns(pbars.time, det["equity"], P["capital"])
        st.dataframe(mr.style.format(lambda v: "" if pd.isna(v) else f"{v:.2f}").map(R.return_cell_style),
                     width="stretch")

        s1, s2 = st.columns(2)
        with s1:
            st.subheader("Сигналы")
            total = int(pbars.signal.sum())
            last_sig = int(bool(pbars.signal.iloc[-1]))
            st.table(pd.DataFrame({"Значение": [
                total, int(m[E.M_EXECUTED]), int(m[E.M_SKIPPED]), last_sig]},
                index=["Всего сигналов", "Исполнено входов", "Пропущено (не хватило денег)",
                       "На последней свече (некуда входить)"]))
        with s2:
            st.subheader("Время в рынке")
            closed = tdf[tdf["Причина"] != "открыта"]
            st.table(pd.DataFrame({"Значение": [
                f"{m[E.M_TIME_IN_MKT]:.1f}%",
                f"{closed['Длительность, ч'].mean():.2f} ч" if len(closed) else "—",
                f"{closed['Длительность, ч'].median():.2f} ч" if len(closed) else "—",
                f"{closed['Длительность, ч'].max():.2f} ч" if len(closed) else "—"]},
                index=["Доля свечей с позицией", "Средняя длительность", "Медианная длительность",
                       "Максимальная длительность"]))

# ================================================================== tab: validation
with tabs[2]:
    if not split:
        st.info("Включите режим «Подбор + проверка» слева и перезапустите перебор, "
                "чтобы сравнить результат на данных подбора и на проверочных данных.")
    else:
        v1, v2 = st.columns(2)
        vtp = v1.selectbox("TP, %", tp_list, index=tp_list.index(default_sel[0]), key="vtp")
        vsl = v2.selectbox("SL, %", sl_list, index=sl_list.index(default_sel[1]), key="vsl")
        dtr = get_detail("Подбор", vtp, vsl)
        dte = get_detail("Проверка", vtp, vsl)
        rows_ = [(E.M_PROFIT, "Прибыль USDT", "{:,.2f}"), (E.M_PROFIT_PCT, "Прибыль %", "{:.2f}"),
                 (E.M_TRADES, "Сделок", "{:,.0f}"), (E.M_WINRATE, "Винрейт %", "{:.1f}"),
                 (E.M_MAXDD_PCT, "Макс. просадка %", "{:.2f}"), (E.M_PF, "Profit factor", "{:.2f}"),
                 (E.M_MAX_CAPITAL, "Макс. задействовано USDT", "{:,.2f}"),
                 (E.M_TIME_IN_MKT, "Время в рынке %", "{:.1f}")]
        tr_bars, te_bars = sets["Подбор"][1], sets["Проверка"][1]
        comp = pd.DataFrame({
            f"Подбор ({tr_bars.time.iloc[0]:%Y-%m-%d} — {tr_bars.time.iloc[-1]:%Y-%m-%d})":
                [f.format(dtr["metrics"][i]) for i, _, f in rows_],
            f"Проверка ({te_bars.time.iloc[0]:%Y-%m-%d} — {te_bars.time.iloc[-1]:%Y-%m-%d})":
                [f.format(dte["metrics"][i]) for i, _, f in rows_],
        }, index=[n for _, n, _ in rows_])
        st.table(comp)
        st.caption("Каждая часть считается отдельно с полным начальным капиталом. "
                   "Если на проверке результат резко хуже — параметры подогнаны под историю.")
        st.plotly_chart(R.stitched_pnl_figure(tr_bars.time, dtr["equity"], te_bars.time,
                                              dte["equity"], P["capital"]),
                        width="stretch")

# ================================================================== tab: export
with tabs[3]:
    st.subheader("Результаты перебора (все комбинации)")
    e1, e2 = st.columns(2)
    e1.download_button("Скачать CSV", R.to_csv(grid), "grid.csv", "text/csv")
    if e2.button("Подготовить Excel"):
        st.session_state["grid_xlsx"] = R.to_excel({"Перебор": grid})
    if "grid_xlsx" in st.session_state:
        e2.download_button("Скачать Excel", st.session_state["grid_xlsx"], "grid.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Сделки выбранной комбинации")
    x1, x2, x3 = st.columns(3)
    etp = x1.selectbox("TP, %", tp_list, index=tp_list.index(default_sel[0]), key="etp")
    esl = x2.selectbox("SL, %", sl_list, index=sl_list.index(default_sel[1]), key="esl")
    eper = x3.selectbox("Период", ["Весь файл", "Подбор", "Проверка"] if split else ["Весь файл"],
                        key="eper")
    edet = get_detail(eper, etp, esl)
    etdf = R.trades_frame(edet, sets[eper][1].time, mode, res["tf_minutes"])
    name = f"trades_tp{etp:g}_sl{esl:g}"
    y1, y2 = st.columns(2)
    y1.download_button("Скачать CSV", R.to_csv(etdf) if len(etdf) else b"", f"{name}.csv", "text/csv",
                       disabled=etdf.empty)
    if y2.button("Подготовить Excel", key="tx", disabled=etdf.empty):
        st.session_state["trades_xlsx"] = (name, R.to_excel({"Сделки": etdf}))
    if "trades_xlsx" in st.session_state and st.session_state["trades_xlsx"][0] == name:
        y2.download_button("Скачать Excel", st.session_state["trades_xlsx"][1], f"{name}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="txd")
