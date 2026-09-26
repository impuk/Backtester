"""Предварительная компиляция движка (запускается установщиком)."""
import numpy as np

import engine as E

rng = np.random.default_rng(0)
n = 500
c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
o = np.r_[100, c[:-1]]
h = np.maximum(o, c) * 1.005
l = np.minimum(o, c) * 0.995
ds = E.Dataset(o, h, l, c, rng.random(n) < 0.1)
for mode in (E.MODE_INDEPENDENT, E.MODE_AVERAGE):
    E.run_grid(ds, np.array([0.01, 0.02]), np.array([0.01, 0.02]), 1000, 10, 0.001, 0, mode)
    E.run_detail(ds, 0.01, 0.01, 1000, 10, 0.001, 0, mode)
print("Движок скомпилирован.")
