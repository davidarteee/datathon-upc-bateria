"""
Comitè de models (ensemble) vs versió robusta, en setmanes NO vistes.

REGLA DE DECISIÓ (fixada ABANS de veure resultats):
  adoptem el comitè només si la seva mitjana en setmanes no vistes supera la de la
  versió robusta (tau 20) en MÉS de 0,10 punts. Si no, ens quedem la robusta.

Validació: 3 grups de setmanes retingudes (A, B, C = la 1a, 2a i 3a setmana abans del final
de cada mes). Per a cada grup s'entrena sense aquelles setmanes i s'avalua en 2 episodis continus
de 1.008 h -> 6 episodis.

Ús (des de l'arrel del repo):
    python -W ignore hard_dani/experiments/robust/ensemble_test.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust_grid as rg  # noqa: E402  (mateixes dades, entorn i funcions d'ajust)
import model as dmodel  # noqa: E402

TAUS = (12.0, 20.0, 30.0)
THRESHOLD = 0.10


class EnsembleForecaster:
    """Mitjana de les prediccions de diversos Forecasters (un per tau)."""

    def __init__(self, params_list, horizon):
        self.members = [dmodel.Forecaster(p, horizon) for p in params_list]

    def __getattr__(self, name):            # previous, obs_net, obs_market... del primer membre
        return getattr(self.members[0], name)

    def forecast(self, obs):
        outs = [m.forecast(obs) for m in self.members]
        res = dict(outs[0])
        for var in dmodel.VARS:
            res[var] = np.mean([o[var] for o in outs], axis=0)
        return res


class EnsembleAgent(dmodel.HardAgentV2):
    def __init__(self, params_list):
        self.params_list = params_list
        super().__init__(params_list[1])    # sigma, pesos i config de tau 20
        self.forecaster = EnsembleForecaster(params_list, self.c["horizon"])

    def new_episode(self):
        super().new_episode()
        self.forecaster = EnsembleForecaster(self.params_list, self.c["horizon"])


def run_agent(agent, df):
    env = rg.HardBatteryEnv(df)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(float(dmodel.clamp_action(agent.act(obs), obs)))
    n = len(env.grid)
    g = np.asarray(env.grid)
    d = env.df.iloc[:n]
    b = np.maximum(d.load_kw.values - d.pv_kw.values, 0)
    c, p = d.co2_kg_per_kwh.values, d.precio_eur_per_kwh.values
    return 100 * (0.7 * (1 - (g * c).sum() / (b * c).sum()) + 0.3 * (1 - (g * p).sum() / (b * p).sum())) \
        - (env.clipped + env.violations)


def episodes():
    blocks = rg.estimate.weekly_blocks(rg.FRAME)
    out = []
    for label in "ABC":
        keys = sorted(k for k in blocks if k[1] == label)
        weeks = [rg.FRAME.iloc[blocks[k]] for k in keys]
        test_idx = pd.concat(weeks).index
        out.append((label, rg.FRAME.drop(test_idx), [pd.concat(weeks[:6]), pd.concat(weeks[-6:])]))
    return out


def main():
    rows = {"v6 Dani (tau 12)": [], "robusta (tau 20)": [], "comitè (tau 12+20+30)": []}
    orc = []
    for label, train, eps in episodes():
        t0 = time.time()
        p = {tau: rg.fit(train, {"tau": tau}, {}) for tau in TAUS}
        for df in eps:
            orc.append(rg.oracle(df))
            rows["v6 Dani (tau 12)"].append(run_agent(dmodel.HardAgentV2(p[12.0]), df))
            rows["robusta (tau 20)"].append(run_agent(dmodel.HardAgentV2(p[20.0]), df))
            rows["comitè (tau 12+20+30)"].append(run_agent(EnsembleAgent([p[t] for t in TAUS]), df))
        print(f"[{label}] fet en {time.time() - t0:.0f}s", flush=True)
    print("oracle:", [round(o, 2) for o in orc])
    summary = {}
    for name, v in rows.items():
        summary[name] = {"episodis": [round(x, 2) for x in v], "mitjana": round(float(np.mean(v)), 3),
                         "pitjor": round(float(np.min(v)), 2)}
        print(f"{name:26s} mitjana no vist={np.mean(v):.3f}  {summary[name]['episodis']}")
    gain = summary["comitè (tau 12+20+30)"]["mitjana"] - summary["robusta (tau 20)"]["mitjana"]
    wins = int(sum(e > r for e, r in zip(rows["comitè (tau 12+20+30)"], rows["robusta (tau 20)"])))
    verdict = "ADOPTAR comitè" if gain > THRESHOLD else "QUEDAR-SE la robusta"
    print(f"Guany comitè vs robusta: {gain:+.3f} (guanya {wins}/6 episodis) -> {verdict}")
    summary["decisio"] = {"guany": round(gain, 3), "episodis_guanyats": wins, "veredicte": verdict}
    (Path(__file__).parent / "ensemble_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
