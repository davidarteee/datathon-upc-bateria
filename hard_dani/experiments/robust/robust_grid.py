"""
Variants regularitzades de l'agent del Dani i validació amb setmanes retingudes.

Per a cada variant:
  - "no vist": entrenem sense les setmanes A (o B) de cada mes i avaluem en episodis continus de
    1.008 h (6 setmanes) fets amb aquestes setmanes -> 4 episodis (A 1-6, A 6-11, B 1-6, B 6-11).
  - "vist": mateixos episodis, però entrenant amb TOT el train.
Triem la variant amb millor mitjana "no vist" i diferència vist - no vist petita.

Ús (des de l'arrel del repo):
    python -W ignore hard_dani/experiments/robust/robust_grid.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
DANI = REPO / "hard_dani"
for p in (REPO / "src", DANI, DANI / "experiments/dani_hard_score_estimation", DANI / "experiments/dani_hard_v2"):
    sys.path.insert(0, str(p))

from data import load_hourly  # noqa: E402
from hard_env import HardBatteryEnv, to_hard  # noqa: E402
from oracle import solve_lp  # noqa: E402
import estimate  # noqa: E402
import model as dmodel  # noqa: E402
import train as dtrain  # noqa: E402

FRAME = to_hard(load_hourly())
COLS = ["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]
SIMPLE = {"load": [0, 1, 2], "pv": [0, 1, 2], "price": [0, 1, 2], "co2": [0, 1, 2]}

# Variants: (nom, kwargs de fit, canvis de config)
VARIANTS = [
    ("v6 Dani (referència)", {}, {}),
    ("ridge 0,1", {"ridge": 0.1}, {}),
    ("ridge 1", {"ridge": 1.0}, {}),
    ("perfils llisos tau 20", {"tau": 20.0}, {}),
    ("perfils llisos tau 30", {"tau": 30.0}, {}),
    ("menys variables", {"use": SIMPLE}, {}),
    ("tau 20 + ridge 0,1", {"tau": 20.0, "ridge": 0.1}, {}),
    ("tau 20 + ridge 0,1 + menys variables", {"tau": 20.0, "ridge": 0.1, "use": SIMPLE}, {}),
]


def fit(train, fit_kw, cfg):
    market = pd.DataFrame({"price": train.precio_eur_per_kwh, "co2": train.co2_kg_per_kwh})
    kw = {**dtrain.FINAL, **fit_kw}
    params = dtrain.fit(train[COLS], market, load_full=train.load_kw, **kw)
    params["config"] = {**dtrain.FINAL_CONFIG, **cfg}
    return params


def run(params, df):
    agent = dmodel.HardAgentV2(params)
    env = HardBatteryEnv(df)
    obs = env.reset()
    done = False
    while not done:
        from model import clamp_action
        obs, done = env.step(float(clamp_action(agent.act(obs), obs)))
    n = len(env.grid)
    g = np.asarray(env.grid)
    d = env.df.iloc[:n]
    b = np.maximum(d.load_kw.values - d.pv_kw.values, 0)
    c, p = d.co2_kg_per_kwh.values, d.precio_eur_per_kwh.values
    infr = env.clipped + env.violations
    return 100 * (0.7 * (1 - (g * c).sum() / (b * c).sum()) + 0.3 * (1 - (g * p).sum() / (b * p).sum())) - infr


def oracle(df):
    net = df.load_kw.values - df.pv_kw.values
    b = np.maximum(net, 0)
    c, p = df.co2_kg_per_kwh.values, df.precio_eur_per_kwh.values
    w = 0.7 * c / (b * c).sum() + 0.3 * p / (b * p).sum()
    ch, dis, _ = solve_lp(net, w, 3000.0, 6000.0, 1500.0)
    g = np.maximum(net + ch - dis, 0)
    return 100 * (0.7 * (1 - (g * c).sum() / (b * c).sum()) + 0.3 * (1 - (g * p).sum() / (b * p).sum()))


def episodes():
    blocks = estimate.weekly_blocks(FRAME)
    out = []
    for label in "AB":
        keys = sorted(k for k in blocks if k[1] == label)
        weeks = [FRAME.iloc[blocks[k]] for k in keys]
        test_idx = pd.concat(weeks).index
        out.append((label, FRAME.drop(test_idx), [("1-6", pd.concat(weeks[:6])), ("6-11", pd.concat(weeks[-6:]))]))
    return out


def main():
    eps = episodes()
    orc = [oracle(df) for _, _, es in eps for _, df in es]
    print("Màxim possible (oracle) per episodi:", [round(o, 2) for o in orc], flush=True)
    results = []
    for name, fit_kw, cfg in VARIANTS:
        t0 = time.time()
        p_all = fit(FRAME, fit_kw, cfg)
        unseen, seen = [], []
        for label, train, es in eps:
            p_out = fit(train, fit_kw, cfg)
            for _, df in es:
                unseen.append(run(p_out, df))
                seen.append(run(p_all, df))
        row = {"variant": name, "no_vist": [round(x, 2) for x in unseen], "vist": [round(x, 2) for x in seen],
               "mitjana_no_vist": round(float(np.mean(unseen)), 3), "mitjana_vist": round(float(np.mean(seen)), 3),
               "diferencia": round(float(np.mean(seen) - np.mean(unseen)), 3),
               "pitjor_no_vist_vs_oracle": round(float(min(u / o for u, o in zip(unseen, orc))), 3)}
        results.append(row)
        print(f"{name:42s} no vist={row['mitjana_no_vist']:.3f}  vist={row['mitjana_vist']:.3f}  "
              f"dif={row['diferencia']:.3f}  {row['no_vist']}  ({time.time() - t0:.0f}s)", flush=True)
    (Path(__file__).parent / "robust_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()
