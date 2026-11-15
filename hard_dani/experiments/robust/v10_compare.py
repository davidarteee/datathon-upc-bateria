"""
Compara la v10 del Dani amb la nostra versió robusta en setmanes NO vistes (6 episodis de 1.008 h).

v10 = mateix model que v6 + (a) coneix l'estructura del test (llista de setmanes: quan l'horitzó
passa del final d'una setmana, les hores següents són l'inici de la setmana següent),
(b) no buida ni força el 50 % al final, (c) Setmana Santa al calendari (no avaluable amb el train).
Per ser justos, a la validació li donem la llista de les setmanes RETINGUDES de cada episodi.

REGLA (fixada abans): canviem de versió només si guanya per > 0,10 de mitjana en no vist.

Ús: python -W ignore hard_dani/experiments/robust/v10_compare.py
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robust_grid as rg  # noqa: E402
import model as m6  # noqa: E402  (v6)

spec = importlib.util.spec_from_file_location("hard_v2_model_v10", HERE.parent / "v10" / "hard_v2_model_v10.py")
m10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m10)


def run_agent(mod, agent, df):
    env = rg.HardBatteryEnv(df)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(float(mod.clamp_action(agent.act(obs), obs)))
    n = len(env.grid)
    g = np.asarray(env.grid)
    d = env.df.iloc[:n]
    b = np.maximum(d.load_kw.values - d.pv_kw.values, 0)
    c, p = d.co2_kg_per_kwh.values, d.precio_eur_per_kwh.values
    return 100 * (0.7 * (1 - (g * c).sum() / (b * c).sum()) + 0.3 * (1 - (g * p).sum() / (b * p).sum())) \
        - (env.clipped + env.violations)


def v10_params(p, weeks):
    q = dict(p)
    q["config"] = {"stochastic": True, "s_now": 1.0, "s_future": 1.0, "adaptive_weights": 168,
                   "weeks": [[w.index[0].isoformat(), w.index[-1].isoformat()] for w in weeks]}
    return q


def main():
    blocks = rg.estimate.weekly_blocks(rg.FRAME)
    rows = {k: [] for k in ["v6 (tau 12)", "robusta (tau 20) [lliurada: 23,15]",
                            "v10 (tau 12 + truc setmanes)", "robusta + truc setmanes (tau 20)"]}
    for label in "ABC":
        t0 = time.time()
        keys = sorted(k for k in blocks if k[1] == label)
        weeks = [rg.FRAME.iloc[blocks[k]] for k in keys]
        train = rg.FRAME.drop(pd.concat(weeks).index)
        p12 = rg.fit(train, {"tau": 12.0}, {})
        p20 = rg.fit(train, {"tau": 20.0}, {})
        for ws in (weeks[:6], weeks[-6:]):
            df = pd.concat(ws)
            rows["v6 (tau 12)"].append(run_agent(m6, m6.HardAgentV2(p12), df))
            rows["robusta (tau 20) [lliurada: 23,15]"].append(run_agent(m6, m6.HardAgentV2(p20), df))
            rows["v10 (tau 12 + truc setmanes)"].append(run_agent(m10, m10.HardAgentV2(v10_params(p12, ws)), df))
            rows["robusta + truc setmanes (tau 20)"].append(run_agent(m10, m10.HardAgentV2(v10_params(p20, ws)), df))
        print(f"[{label}] {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for k, v in rows.items():
        summary[k] = {"episodis": [round(float(x), 2) for x in v], "mitjana": round(float(np.mean(v)), 3)}
        print(f"{k:40s} mitjana={np.mean(v):.3f}  {summary[k]['episodis']}")
    (HERE / "v10_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
