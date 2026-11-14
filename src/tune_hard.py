"""
src/tune_hard.py
Diagnòstic i ajust de l'agent del Mode Difícil.

  python src/tune_hard.py diag    # quant perdem per cada predicció (substituint-la per la real)
  python src/tune_hard.py grid    # graella de paràmetres de la política
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import aggregate, load_agent_class, run, week_chunks  # noqa: E402
from data import load_hourly  # noqa: E402
from hard_env import to_hard  # noqa: E402

Agent = load_agent_class(ROOT / "submission_hard" / "agent.py")
DF = to_hard(load_hourly())
CHUNKS = week_chunks(DF)
COL = {"precio": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh", "load": "load_kw", "pv": "pv_kw"}


def make(perfect=(), **kw):
    class A(Agent):
        def _forecast(self, key, t, offset, n_known):
            if key in perfect:
                idx = pd.date_range(t, periods=self.H, freq="1h")
                s = DF[COL[key]].reindex(idx)
                base = super()._forecast(key, t, offset, n_known)
                return np.where(s.isna(), base, s.values)
            return super()._forecast(key, t, offset, n_known)

    def f():
        a = A()
        for k, v in kw.items():
            setattr(a, k, v)
        return a
    return f


def score(factory):
    return aggregate([run(factory(), c, hard=True) for c in CHUNKS])


def show(name, r):
    print(f"{name:45s} score={r['score_sense_penal']:.3f}  CO2={r['co2_estalviat_pct']:.2f}%  "
          f"cost={r['cost_estalviat_pct']:.2f}%  infr={r['infraccions']}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "diag":
        show("v1 (prediccions pròpies)", score(make()))
        for p in [("precio", "co2"), ("load",), ("pv",), ("load", "pv"), ("precio", "co2", "load", "pv")]:
            show(f"perfecte: {'+'.join(p)}", score(make(perfect=p)))
    else:
        for kw in [dict(term_alpha=a) for a in (0.5, 0.8, 1.0)] + \
                  [dict(discharge_margin=m) for m in (50, 150)]:
            show(str(kw), score(make(**kw)))
