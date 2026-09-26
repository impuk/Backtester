"""Сверка Numba-движка с простой (медленной) эталонной реализацией."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import engine as E


def ref_independent(o, h, l, c, sig, tp, sl, cap, amt, fee, slip):
    n = len(o); cash = cap; opn = []; trades = []; skipped = 0; eq = np.empty(n)
    cost_in = amt * (1 + fee)
    for k in range(n):
        if k >= 1 and sig[k - 1]:
            if cash + 1e-9 >= cost_in:
                p = o[k] * (1 + slip)
                opn.append(dict(k=k, p=p, q=amt / p, tp=p * (1 + tp), sl=p * (1 - sl)))
                cash -= cost_in
            else:
                skipped += 1
        keep = []
        for t in opn:
            px = None
            if k > t['k'] and o[k] >= t['tp']: px = o[k]
            elif k > t['k'] and o[k] <= t['sl']: px = o[k]
            elif l[k] <= t['sl']: px = t['sl']
            elif h[k] >= t['tp']: px = t['tp']
            if px is None: keep.append(t); continue
            proc = t['q'] * px * (1 - slip) * (1 - fee)
            cash += proc; trades.append(proc - cost_in)
        opn = keep
        eq[k] = cash + sum(t['q'] for t in opn) * c[k]
    return sum(trades), len(trades), skipped, len(opn), eq


def ref_average(o, h, l, c, sig, tp, sl, cap, amt, fee, slip):
    n = len(o); cash = cap; q = cost = amount = 0.0; ne = 0; trades = []; skipped = 0
    eq = np.empty(n); cost_in = amt * (1 + fee); tpl = sll = 0
    def close(px):
        nonlocal cash, q, cost, amount, ne
        proc = q * px * (1 - slip) * (1 - fee); cash += proc; trades.append(proc - cost)
        q = cost = amount = 0.0; ne = 0
    for k in range(n):
        if ne and o[k] >= tpl: close(o[k])
        elif ne and o[k] <= sll: close(o[k])
        if k >= 1 and sig[k - 1]:
            if cash + 1e-9 >= cost_in:
                p = o[k] * (1 + slip); q += amt / p; cost += cost_in; amount += amt; ne += 1
                cash -= cost_in; avg = amount / q; tpl = avg * (1 + tp); sll = avg * (1 - sl)
            else:
                skipped += 1
        if ne:
            if l[k] <= sll: close(sll)
            elif h[k] >= tpl: close(tpl)
        eq[k] = cash + q * c[k]
    return sum(trades), len(trades), skipped, 1 if ne else 0, eq


def make_data(n, seed, p_sig):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.004, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]] * np.exp(rng.normal(0, 0.001, n))  # небольшие гэпы
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    sig = rng.random(n) < p_sig
    return o, h, l, c, sig


def test_against_reference():
    cases = 0
    for seed in range(12):
        n = 3000
        o, h, l, c, sig = make_data(n, seed, [0.02, 0.1, 0.4][seed % 3])
        ds = E.Dataset(o, h, l, c, sig)
        cap = [100000, 3000, 800][seed % 3]
        for mode, ref in ((E.MODE_INDEPENDENT, ref_independent), (E.MODE_AVERAGE, ref_average)):
            for tp, sl in ((0.01, 0.01), (0.03, 0.012), (0.1, 0.05), (0.014, 0.1)):
                fee, slip = 0.001, 0.0005 * (seed % 2)
                d = E.run_detail(ds, tp, sl, cap, 100, fee, slip, mode)
                rp, rn, rs, ro, req = ref(o, h, l, c, sig, tp, sl, cap, 100, fee, slip)
                m = d['metrics']
                assert abs(m[E.M_PROFIT] - rp) < 1e-6, (seed, mode, tp, sl, m[E.M_PROFIT], rp)
                assert m[E.M_TRADES] == rn
                assert m[E.M_SKIPPED] == rs
                assert m[E.M_OPEN_END] == ro, (m[E.M_OPEN_END], ro)
                assert np.allclose(d['equity'], req, atol=1e-6)
                g = E.run_grid(ds, np.array([tp]), np.array([sl]), cap, 100, fee, slip, mode)[0]
                assert np.allclose(np.nan_to_num(g, posinf=1e18), np.nan_to_num(m, posinf=1e18))
                cases += 1
    print("OK, cases:", cases)


def test_first_ge():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 64, 1000):
        a = rng.random(n)
        tree, size = E.build_max_tree(a)
        for _ in range(300):
            lft = rng.integers(0, n); x = rng.random() * 1.1
            idx = np.nonzero(a[lft:] >= x)[0]
            exp = lft + idx[0] if len(idx) else -1
            assert E.first_ge(tree, size, lft, x) == exp
    print("OK first_ge")


if __name__ == "__main__":
    test_first_ge()
    test_against_reference()
