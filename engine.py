"""Ядро бэктеста: симуляция long-сделок по сигналам с TP/SL и докупками.

Два режима выхода:
  MODE_INDEPENDENT (0) — у каждого входа свои TP/SL от его цены, закрывается сам.
  MODE_AVERAGE     (1) — TP/SL от средней цены позиции, позиция закрывается целиком.

Правила (согласованы в ТЗ):
  * сигнал на свече i -> вход по open свечи i+1 (сигнал на последней свече игнорируется);
  * каждая свеча с True — отдельный сигнал, докупка без лимита, пока хватает кэша;
  * фиксированная сумма входа, комиссия берётся с входа и с выхода, проскальзывание
    ухудшает цену входа (вверх) и выхода (вниз);
  * если на одной свече задеты и TP, и SL — считается SL;
  * если свеча открылась гэпом за уровнем — выход по open этой свечи;
  * позиция, открытая на конце данных, в прибыль не входит (показывается отдельно).
"""
import numba
import numpy as np
from numba import njit, prange

MODE_INDEPENDENT = 0
MODE_AVERAGE = 1

REASON_OPEN = 0
REASON_TP = 1
REASON_SL = 2

# индексы в массиве метрик
M_PROFIT = 0          # реализованная прибыль, USDT
M_PROFIT_PCT = 1      # % от начального капитала
M_TRADES = 2          # закрытых сделок
M_WINRATE = 3         # %
M_MAXDD_PCT = 4       # макс. просадка, % от пика капитала
M_MAXDD_USDT = 5
M_PF = 6              # profit factor (inf, если убыточных нет)
M_AVG_ENTRIES = 7     # режим «Средняя»: среднее число входов в позиции
M_MAX_ENTRIES = 8     # «Средняя»: макс. входов в позиции; «Независимые»: макс. одновременно открытых
M_MAX_CAPITAL = 9     # макс. одновременно задействованный капитал, USDT
M_SKIPPED = 10        # сигналов пропущено из-за нехватки денег
M_EXECUTED = 11       # входов исполнено
M_OPEN_END = 12       # входов, открытых на конце данных
M_OPEN_UNREAL = 13    # нереализованный результат открытых на конце, USDT
M_TIME_IN_MKT = 14    # доля свечей с открытой позицией, %
N_METRICS = 15

SCAN = 128  # длина линейного поиска выхода перед деревом отрезков


# ---------------------------------------------------------------- segment tree
@njit(cache=True)
def build_max_tree(a):
    n = a.shape[0]
    size = 1
    while size < n:
        size *= 2
    tree = np.full(2 * size, -np.inf)
    for i in range(n):
        tree[size + i] = a[i]
    for i in range(size - 1, 0, -1):
        tree[i] = max(tree[2 * i], tree[2 * i + 1])
    return tree, size


@njit(cache=True)
def first_ge(tree, size, l, x):
    """Первый индекс >= l, где значение >= x. -1, если такого нет."""
    if l >= size:
        return -1
    i = l + size
    while True:
        if tree[i] >= x:
            break
        while i & 1:
            i >>= 1
        if i == 0:
            return -1
        i += 1
    while i < size:
        i = 2 * i
        if tree[i] < x:
            i += 1
    return i - size


# ---------------------------------------------------------------- simulations
@njit(cache=True)
def _simulate_independent(o, h, l, c, cand, tp, sl, capital, amount, fee, slip,
                          hi_tree, lo_tree, size,
                          equity, in_mkt, t_entry, t_exit, t_eprice, t_xprice,
                          t_qty, t_cost, t_pnl, t_reason, t_group, skipped_bars,
                          ex_proc, ex_qty, ex_cost, ex_cnt):
    # ex_* — корзины выходов по свечам; на входе нули, после прогона снова нули
    n = o.shape[0]

    cash = capital
    open_qty = 0.0
    open_cost = 0.0
    open_cnt = 0
    max_cnt = 0
    max_cost = 0.0
    nt = 0            # записанных входов
    ns = 0            # пропущенных сигналов
    ci = 0
    ncand = cand.shape[0]
    cost_in = amount * (1.0 + fee)

    for k in range(n):
        if ci < ncand and cand[ci] == k:
            ci += 1
            if cash + 1e-9 >= cost_in:
                p = o[k] * (1.0 + slip)
                qty = amount / p
                tp_lvl = p * (1.0 + tp)
                sl_lvl = p * (1.0 - sl)
                # сначала короткий линейный поиск (обычно выход близко),
                # затем — дерево отрезков для дальних выходов
                kt = n
                ks = n
                lim = min(n, k + SCAN)
                for j in range(k, lim):
                    if l[j] <= sl_lvl:
                        ks = j
                    if h[j] >= tp_lvl:
                        kt = j
                    if ks < n or kt < n:
                        break
                if ks == n and kt == n and lim < n:
                    kt = first_ge(hi_tree, size, lim, tp_lvl)
                    ks = first_ge(lo_tree, size, lim, -sl_lvl)
                    if kt < 0 or kt >= n:
                        kt = n
                    if ks < 0 or ks >= n:
                        ks = n
                kx = min(kt, ks)
                reason = REASON_OPEN
                xp = np.nan
                if kx < n:
                    if kx > k and o[kx] >= tp_lvl:
                        xp = o[kx]
                        reason = REASON_TP
                    elif kx > k and o[kx] <= sl_lvl:
                        xp = o[kx]
                        reason = REASON_SL
                    elif ks == kx:
                        xp = sl_lvl
                        reason = REASON_SL
                    else:
                        xp = tp_lvl
                        reason = REASON_TP
                cash -= cost_in
                open_qty += qty
                open_cost += cost_in
                open_cnt += 1
                t_entry[nt] = k
                t_eprice[nt] = p
                t_qty[nt] = qty
                t_cost[nt] = cost_in
                t_reason[nt] = reason
                t_group[nt] = nt
                if reason != REASON_OPEN:
                    xexec = xp * (1.0 - slip)
                    proceeds = qty * xexec * (1.0 - fee)
                    t_exit[nt] = kx
                    t_xprice[nt] = xexec
                    t_pnl[nt] = proceeds - cost_in
                    ex_proc[kx] += proceeds
                    ex_qty[kx] += qty
                    ex_cost[kx] += cost_in
                    ex_cnt[kx] += 1
                else:
                    t_exit[nt] = -1
                    t_xprice[nt] = np.nan
                    t_pnl[nt] = qty * c[n - 1] * (1.0 - fee) - cost_in
                nt += 1
                if open_cnt > max_cnt:
                    max_cnt = open_cnt
                if open_cost > max_cost:
                    max_cost = open_cost
            else:
                skipped_bars[ns] = k
                ns += 1
        in_mkt[k] = open_cnt > 0
        cash += ex_proc[k]
        open_qty -= ex_qty[k]
        open_cost -= ex_cost[k]
        open_cnt -= ex_cnt[k]
        ex_proc[k] = 0.0
        ex_qty[k] = 0.0
        ex_cost[k] = 0.0
        ex_cnt[k] = 0
        if open_cnt == 0:
            open_qty = 0.0
            open_cost = 0.0
        equity[k] = cash + open_qty * c[k]
    return nt, ns, max_cnt, max_cost


@njit(cache=True)
def _close_avg(pos_qty, pos_cost, price, slip, fee):
    xexec = price * (1.0 - slip)
    proceeds = pos_qty * xexec * (1.0 - fee)
    return xexec, proceeds


@njit(cache=True)
def _simulate_average(o, h, l, c, cand, tp, sl, capital, amount, fee, slip,
                      equity, in_mkt, t_entry, t_exit, t_eprice, t_xprice,
                      t_qty, t_cost, t_pnl, t_reason, t_group, skipped_bars,
                      f_bar, f_price, f_avg, f_group):
    """Записи t_* — позиции (группа входов). f_* — отдельные исполнения входов."""
    n = o.shape[0]
    cash = capital
    pos_qty = 0.0
    pos_cost = 0.0
    pos_amount = 0.0
    pos_n = 0
    pos_first = -1
    tp_lvl = 0.0
    sl_lvl = 0.0
    max_n = 0
    max_cost = 0.0
    nt = 0
    ns = 0
    nf = 0
    ci = 0
    ncand = cand.shape[0]
    cost_in = amount * (1.0 + fee)

    for k in range(n):
        # 1) гэп на открытии относительно уровней, выставленных до этой свечи
        if pos_n > 0:
            px = np.nan
            reason = REASON_OPEN
            if o[k] >= tp_lvl:
                px = o[k]
                reason = REASON_TP
            elif o[k] <= sl_lvl:
                px = o[k]
                reason = REASON_SL
            if reason != REASON_OPEN:
                xexec, proceeds = _close_avg(pos_qty, pos_cost, px, slip, fee)
                cash += proceeds
                t_exit[nt] = k
                t_xprice[nt] = xexec
                t_pnl[nt] = proceeds - pos_cost
                t_reason[nt] = reason
                nt += 1
                pos_qty = 0.0
                pos_cost = 0.0
                pos_amount = 0.0
                pos_n = 0
        # 2) вход / докупка по open
        if ci < ncand and cand[ci] == k:
            ci += 1
            if cash + 1e-9 >= cost_in:
                p = o[k] * (1.0 + slip)
                qty = amount / p
                if pos_n == 0:
                    pos_first = k
                    t_entry[nt] = k
                    t_group[nt] = nt
                cash -= cost_in
                pos_qty += qty
                pos_cost += cost_in
                pos_amount += amount
                pos_n += 1
                avg = pos_amount / pos_qty
                tp_lvl = avg * (1.0 + tp)
                sl_lvl = avg * (1.0 - sl)
                t_eprice[nt] = avg
                t_qty[nt] = pos_qty
                t_cost[nt] = pos_cost
                f_bar[nf] = k
                f_price[nf] = p
                f_avg[nf] = avg
                f_group[nf] = nt
                nf += 1
                if pos_n > max_n:
                    max_n = pos_n
                if pos_cost > max_cost:
                    max_cost = pos_cost
            else:
                skipped_bars[ns] = k
                ns += 1
        # 3) внутри свечи: SL приоритетнее TP
        if pos_n > 0:
            in_mkt[k] = True
            reason = REASON_OPEN
            px = np.nan
            if l[k] <= sl_lvl:
                px = sl_lvl
                reason = REASON_SL
            elif h[k] >= tp_lvl:
                px = tp_lvl
                reason = REASON_TP
            if reason != REASON_OPEN:
                xexec, proceeds = _close_avg(pos_qty, pos_cost, px, slip, fee)
                cash += proceeds
                t_exit[nt] = k
                t_xprice[nt] = xexec
                t_pnl[nt] = proceeds - pos_cost
                t_reason[nt] = reason
                nt += 1
                pos_qty = 0.0
                pos_cost = 0.0
                pos_amount = 0.0
                pos_n = 0
        else:
            in_mkt[k] = False
        equity[k] = cash + pos_qty * c[k]

    if pos_n > 0:  # открыта на конце данных
        t_exit[nt] = -1
        t_xprice[nt] = np.nan
        t_pnl[nt] = pos_qty * c[n - 1] * (1.0 - fee) - pos_cost
        t_reason[nt] = REASON_OPEN
        nt += 1
    return nt, ns, nf, max_n, max_cost


# ---------------------------------------------------------------- metrics
@njit(cache=True)
def _metrics(equity, in_mkt, t_exit, t_pnl, nt, ns, nf, max_entries, max_cost,
             capital, mode):
    m = np.full(N_METRICS, np.nan)
    n = equity.shape[0]
    profit = 0.0
    wins = 0
    closed = 0
    gw = 0.0
    gl = 0.0
    open_end = 0
    unreal = 0.0
    for i in range(nt):
        if t_exit[i] >= 0:
            closed += 1
            pnl = t_pnl[i]
            profit += pnl
            if pnl > 0:
                wins += 1
                gw += pnl
            else:
                gl -= pnl
        else:
            open_end += 1
            unreal += t_pnl[i]
    peak = -np.inf
    mdd = 0.0
    mdd_usdt = 0.0
    inm = 0
    for k in range(n):
        e = equity[k]
        if e > peak:
            peak = e
        dd = peak - e
        if dd > mdd_usdt:
            mdd_usdt = dd
        if peak > 0 and dd / peak > mdd:
            mdd = dd / peak
        if in_mkt[k]:
            inm += 1
    m[M_PROFIT] = profit
    m[M_PROFIT_PCT] = profit / capital * 100.0
    m[M_TRADES] = closed
    m[M_WINRATE] = wins / closed * 100.0 if closed > 0 else np.nan
    m[M_MAXDD_PCT] = mdd * 100.0
    m[M_MAXDD_USDT] = mdd_usdt
    if gl > 0:
        m[M_PF] = gw / gl
    elif gw > 0:
        m[M_PF] = np.inf
    m[M_MAX_ENTRIES] = max_entries
    if mode == MODE_AVERAGE and nt > 0:
        m[M_AVG_ENTRIES] = nf / nt
    m[M_MAX_CAPITAL] = max_cost
    m[M_SKIPPED] = ns
    m[M_EXECUTED] = nf
    m[M_OPEN_END] = open_end
    m[M_OPEN_UNREAL] = unreal
    m[M_TIME_IN_MKT] = inm / n * 100.0 if n > 0 else np.nan
    return m


@njit(cache=True)
def _alloc(n, ncand):
    E = ncand + 1
    return (np.empty(n), np.zeros(n, np.bool_),
            np.full(E, -1, np.int64), np.full(E, -1, np.int64),
            np.full(E, np.nan), np.full(E, np.nan), np.zeros(E), np.zeros(E),
            np.zeros(E), np.zeros(E, np.int64), np.zeros(E, np.int64),
            np.full(E, -1, np.int64), np.full(E, -1, np.int64), np.full(E, np.nan),
            np.full(E, np.nan), np.full(E, -1, np.int64),
            np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, np.int64))


@njit(cache=True)
def _run_one(o, h, l, c, cand, tp, sl, capital, amount, fee, slip, mode,
             hi_tree, lo_tree, size):
    b = _alloc(o.shape[0], cand.shape[0])
    return _run_buf(o, h, l, c, cand, tp, sl, capital, amount, fee, slip, mode,
                    hi_tree, lo_tree, size, b)


@njit(cache=True)
def _run_buf(o, h, l, c, cand, tp, sl, capital, amount, fee, slip, mode,
             hi_tree, lo_tree, size, b):
    (equity, in_mkt, t_entry, t_exit, t_eprice, t_xprice, t_qty, t_cost, t_pnl,
     t_reason, t_group, skipped, f_bar, f_price, f_avg, f_group,
     ex_proc, ex_qty, ex_cost, ex_cnt) = b
    if mode == MODE_INDEPENDENT:
        nt, ns, max_e, max_cost = _simulate_independent(
            o, h, l, c, cand, tp, sl, capital, amount, fee, slip,
            hi_tree, lo_tree, size, equity, in_mkt, t_entry, t_exit, t_eprice,
            t_xprice, t_qty, t_cost, t_pnl, t_reason, t_group, skipped,
            ex_proc, ex_qty, ex_cost, ex_cnt)
        nf = nt
        for i in range(nt):
            f_bar[i] = t_entry[i]
            f_price[i] = t_eprice[i]
            f_avg[i] = t_eprice[i]
            f_group[i] = i
    else:
        nt, ns, nf, max_e, max_cost = _simulate_average(
            o, h, l, c, cand, tp, sl, capital, amount, fee, slip,
            equity, in_mkt, t_entry, t_exit, t_eprice, t_xprice, t_qty, t_cost,
            t_pnl, t_reason, t_group, skipped, f_bar, f_price, f_avg, f_group)
    m = _metrics(equity, in_mkt, t_exit, t_pnl, nt, ns, nf, max_e, max_cost,
                 capital, mode)
    return (m, equity, in_mkt, nt, t_entry, t_exit, t_eprice, t_xprice, t_qty,
            t_cost, t_pnl, t_reason, ns, skipped, nf, f_bar, f_price, f_avg, f_group)


@njit(parallel=True, cache=True)
def _grid(o, h, l, c, cand, tps, sls, capital, amount, fee, slip, mode,
          hi_tree, lo_tree, size, nthreads):
    nc = tps.shape[0]
    out = np.empty((nc, N_METRICS))
    nw = min(nthreads, nc)
    for w in prange(nw):
        b = _alloc(o.shape[0], cand.shape[0])  # буферы выделяются один раз на поток
        for i in range(w, nc, nw):
            res = _run_buf(o, h, l, c, cand, tps[i], sls[i], capital, amount, fee,
                           slip, mode, hi_tree, lo_tree, size, b)
            out[i, :] = res[0]
    return out


# ---------------------------------------------------------------- Python API
class Dataset:
    """Подготовленные массивы свечей и сигналов для одного отрезка данных."""

    def __init__(self, o, h, l, c, signal):
        self.o = np.ascontiguousarray(o, dtype=np.float64)
        self.h = np.ascontiguousarray(h, dtype=np.float64)
        self.l = np.ascontiguousarray(l, dtype=np.float64)
        self.c = np.ascontiguousarray(c, dtype=np.float64)
        sig = np.asarray(signal, dtype=np.bool_)
        n = self.o.shape[0]
        idx = np.nonzero(sig[: max(n - 1, 0)])[0] + 1  # вход на следующей свече
        self.cand = idx.astype(np.int64)
        self.n_signals = int(sig.sum())
        self.hi_tree, self.size = build_max_tree(self.h)
        self.lo_tree, _ = build_max_tree(-self.l)


def run_grid(ds, tps, sls, capital, amount, fee, slip, mode):
    """tps, sls — одномерные массивы долей (0.01 = 1%) одинаковой длины."""
    return _grid(ds.o, ds.h, ds.l, ds.c, ds.cand,
                 np.ascontiguousarray(tps, np.float64),
                 np.ascontiguousarray(sls, np.float64),
                 float(capital), float(amount), float(fee), float(slip), int(mode),
                 ds.hi_tree, ds.lo_tree, ds.size, max(1, numba.get_num_threads()))


def run_detail(ds, tp, sl, capital, amount, fee, slip, mode):
    r = _run_one(ds.o, ds.h, ds.l, ds.c, ds.cand, float(tp), float(sl),
                 float(capital), float(amount), float(fee), float(slip), int(mode),
                 ds.hi_tree, ds.lo_tree, ds.size)
    (m, equity, in_mkt, nt, t_entry, t_exit, t_eprice, t_xprice, t_qty, t_cost,
     t_pnl, t_reason, ns, skipped, nf, f_bar, f_price, f_avg, f_group) = r
    trades = {
        "entry_bar": t_entry[:nt], "exit_bar": t_exit[:nt],
        "entry_price": t_eprice[:nt], "exit_price": t_xprice[:nt],
        "qty": t_qty[:nt], "cost": t_cost[:nt], "pnl": t_pnl[:nt],
        "reason": t_reason[:nt],
    }
    fills = {"bar": f_bar[:nf], "price": f_price[:nf], "avg": f_avg[:nf],
             "group": f_group[:nf]}
    return {"metrics": m, "equity": equity, "in_mkt": in_mkt, "trades": trades,
            "fills": fills, "skipped_bars": skipped[:ns]}
