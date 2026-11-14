"""
src/oracle.py
Cota superior: LP amb informació perfecta (consum, solar, preu i CO2 de tota la setmana coneguts).
Cap agent real pot superar aquest score; ens diu quant marge ens queda.

Ús:
    python src/oracle.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, tril, vstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_hourly, split_weeks  # noqa: E402

CAP, PMAX = 2000.0, 500.0
ETA = 0.9 ** 0.5


def solve_week(net_load, w, soc0=1000.0):
    """Minimitza sum(w * grid) amb grid = net_load + c - d >= 0. Retorna p = c - d."""
    H = len(w)
    L = tril(csr_matrix(np.ones((H, H))), format="csr")
    A_soc = hstack([ETA * L, -L / ETA])          # soc_k - soc0
    A_ub = vstack([A_soc, -A_soc]).tocsr()
    b_ub = np.concatenate([np.full(H, CAP - soc0), np.full(H, soc0)])
    w = np.asarray(w) / np.mean(w)               # escala ~1: evita que HiGHS ho tracti com a zero
    c = np.concatenate([w, -w])
    dmax = np.clip(net_load, 0, PMAX)            # descarregar més que el consum no aporta res
    bounds = [(0, PMAX)] * H + [(0, d) for d in dmax]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return r.x[:H] - r.x[H:]


def main(week_hours=168):
    df = load_hourly()
    chunks = [b.iloc[i:i + week_hours] for b in split_weeks(df)
              for i in range(0, len(b) - 24 + 1, week_hours)]
    base = [np.maximum(c.load_kw.values - c.pv_kw.values, 0) for c in chunks]
    co2_b = sum((b * c.co2_kg_per_kwh.values).sum() for b, c in zip(base, chunks))
    cost_b = sum((b * c.precio_eur_per_kwh.values).sum() for b, c in zip(base, chunks))

    co2_a = cost_a = 0.0
    for b, c in zip(base, chunks):
        net = c.load_kw.values - c.pv_kw.values
        w = 0.7 * c.co2_kg_per_kwh.values / co2_b + 0.3 * c.precio_eur_per_kwh.values / cost_b
        p = solve_week(net, w)
        g = np.maximum(net + p, 0)
        co2_a += (g * c.co2_kg_per_kwh.values).sum()
        cost_a += (g * c.precio_eur_per_kwh.values).sum()

    d_co2, d_cost = 1 - co2_a / co2_b, 1 - cost_a / cost_b
    print(f"ORACLE ({len(chunks)} setmanes): score={100*(0.7*d_co2+0.3*d_cost):.3f}  "
          f"dCO2={100*d_co2:.3f}%  dCost={100*d_cost:.3f}%")
    print(f"Rati CO2_base/Cost_base = {co2_b/cost_b:.4f} kg/EUR")


if __name__ == "__main__":
    main()
