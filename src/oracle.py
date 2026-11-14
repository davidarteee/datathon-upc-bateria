"""
src/oracle.py
Cota superior: LP amb informació perfecta (consum, solar, preu i CO2 de tota la setmana coneguts).
Cap agent real pot superar aquest score; ens diu quant marge ens queda.

Ús:
    python src/oracle.py            # Mode Estàndard
    python src/oracle.py --hard     # Mode Difícil (6.000 kWh, ±1.500 kW, solar ×12)
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_hourly, split_weeks  # noqa: E402
from hard_env import to_hard  # noqa: E402

ETA = 0.9 ** 0.5


def solve_lp(net, w, soc0, cap, pmax, v_term=0.0):
    """min sum w_t·g_t - v_term·soc_H
    amb g_t >= net_t + c_t - d_t, g_t >= 0 (l'excedent exportat no val res).
    Variables: [c (càrrega), d (descàrrega), g (importació)]. Retorna (c, d, g)."""
    H = len(w)
    scale = np.mean(w) if np.mean(w) > 0 else 1.0
    w = np.asarray(w) / scale                    # escala ~1: evita que HiGHS ho tracti com a zero
    v = v_term / scale
    L = csr_matrix(np.tril(np.ones((H, H))))
    Z = csr_matrix((H, H))
    I = identity(H, format="csr")
    A_soc = hstack([ETA * L, -L / ETA, Z])       # soc_k - soc0
    A_bal = hstack([I, -I, -I])                  # c - d - g <= -net
    A_ub = vstack([A_soc, -A_soc, A_bal]).tocsr()
    b_ub = np.concatenate([np.full(H, cap - soc0), np.full(H, soc0), -np.asarray(net)])
    cost = np.concatenate([np.full(H, -v * ETA), np.full(H, v / ETA), w])
    bounds = [(0, pmax)] * (2 * H) + [(0, None)] * H
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if r.status != 0:
        return None
    return r.x[:H], r.x[H:2 * H], r.x[2 * H:]


def main(hard=False, week_hours=168):
    df = load_hourly()
    cap, pmax = (6000.0, 1500.0) if hard else (2000.0, 500.0)
    if hard:
        df = to_hard(df)
    chunks = [b.iloc[i:i + week_hours] for b in split_weeks(df)
              for i in range(0, len(b) - 24 + 1, week_hours)]
    base = [np.maximum(c.load_kw.values - c.pv_kw.values, 0) for c in chunks]
    co2_b = sum((b * c.co2_kg_per_kwh.values).sum() for b, c in zip(base, chunks))
    cost_b = sum((b * c.precio_eur_per_kwh.values).sum() for b, c in zip(base, chunks))

    co2_a = cost_a = 0.0
    for c in chunks:
        net = c.load_kw.values - c.pv_kw.values
        w = 0.7 * c.co2_kg_per_kwh.values / co2_b + 0.3 * c.precio_eur_per_kwh.values / cost_b
        ch, dis, _ = solve_lp(net, w, 0.5 * cap, cap, pmax)
        g = np.maximum(net + ch - dis, 0)
        co2_a += (g * c.co2_kg_per_kwh.values).sum()
        cost_a += (g * c.precio_eur_per_kwh.values).sum()

    d_co2, d_cost = 1 - co2_a / co2_b, 1 - cost_a / cost_b
    print(f"ORACLE {'DIFÍCIL' if hard else 'ESTÀNDARD'} ({len(chunks)} setmanes): "
          f"score={100*(0.7*d_co2+0.3*d_cost):.3f}  dCO2={100*d_co2:.3f}%  dCost={100*d_cost:.3f}%")
    print(f"Rati CO2_base/Cost_base = {co2_b/cost_b:.4f} kg/EUR  -> K = {(0.3/0.7)*co2_b/cost_b:.4f}")


if __name__ == "__main__":
    main(hard="--hard" in sys.argv)
