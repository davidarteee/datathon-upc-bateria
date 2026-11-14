"""
src/cv_hard.py
Validació creuada HONESTA del Mode Difícil, imitant el test real:
el test són els últims 7 dies de cada mes, just després del train d'aquell mes.
Per a cada bloc contigu del train: test = les seves últimes 168 h; s'entrenen les
prediccions amb TOTA la resta de dades i s'avalua l'agent només en aquella setmana.

Ús:
    python src/cv_hard.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import aggregate, load_agent_class, run  # noqa: E402
from data import load_hourly, split_weeks  # noqa: E402
from hard_env import to_hard  # noqa: E402
import train_forecast  # noqa: E402

DF = to_hard(load_hourly())


def folds(min_block=168 + 72):
    for b in split_weeks(DF):
        if len(b) >= min_block:
            test = b.iloc[-168:]
            yield DF.drop(test.index), test


_CACHE = {}


def cv(agent_path=ROOT / "submission_hard" / "agent.py", fit=None, verbose=True, **kw):
    Agent = load_agent_class(Path(agent_path))
    fit = fit or train_forecast.fit_all
    envs = []
    for i, (train, test) in enumerate(folds()):
        ck = (getattr(fit, "__name__", str(fit)), i)
        if ck not in _CACHE:
            _CACHE[ck] = fit(train)
        params = _CACHE[ck]
        a = Agent(params=params)
        for k, v in kw.items():
            setattr(a, k, v)
        envs.append(run(a, test, hard=True))
    r = aggregate(envs)
    if verbose:
        print(f"CV ({len(envs)} setmanes) {kw}: score={r['score_sense_penal']:.3f} "
              f"CO2={r['co2_estalviat_pct']:.2f}% cost={r['cost_estalviat_pct']:.2f}% infr={r['infraccions']}",
              flush=True)
    return r


if __name__ == "__main__":
    t0 = time.time()
    cv()
    print(f"{time.time() - t0:.0f}s")


def oracle_cv():
    """Cota superior sobre les mateixes setmanes de la CV."""
    import numpy as np
    from oracle import solve_lp
    tests = [t for _, t in folds()]
    base = [np.maximum(c.load_kw.values - c.pv_kw.values, 0) for c in tests]
    co2_b = sum((b * c.co2_kg_per_kwh.values).sum() for b, c in zip(base, tests))
    cost_b = sum((b * c.precio_eur_per_kwh.values).sum() for b, c in zip(base, tests))
    co2_a = cost_a = 0.0
    for c in tests:
        net = c.load_kw.values - c.pv_kw.values
        w = 0.7 * c.co2_kg_per_kwh.values / co2_b + 0.3 * c.precio_eur_per_kwh.values / cost_b
        ch, dis, _ = solve_lp(net, w, 3000.0, 6000.0, 1500.0)
        g = np.maximum(net + ch - dis, 0)
        co2_a += (g * c.co2_kg_per_kwh.values).sum()
        cost_a += (g * c.precio_eur_per_kwh.values).sum()
    d1, d2 = 1 - co2_a / co2_b, 1 - cost_a / cost_b
    print(f"ORACLE CV: score={100*(0.7*d1+0.3*d2):.3f}")
