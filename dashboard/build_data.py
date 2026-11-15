"""
dashboard/build_data.py
Genera les dades de l'app "El Torn de la Bateria" per als dos modes:
  - Fàcil:   bateria 2.000 kWh / ±500 kW, solar real    -> agent submission_easy_final (v1 MPC-LP)
  - Difícil: bateria 6.000 kWh / ±1.500 kW, solar ×12   -> agent submission_hard_final (v10 del Dani)

Per a cada dia triat:
  1. L'agent final juga la setmana anterior i el dia (així té context, com en el test), amb SOC inicial 50 %.
  2. Es desa el dia: consum, solar, preu, CO2, consigna de l'agent i SOC.
  3. Es calcula el màxim possible (LP amb informació perfecta) partint del mateix SOC.
Tot el que l'app puntua (jugador, agent, màxim) es recalcula a l'app amb la mateixa física i fórmula.

Ús:
    python -W ignore dashboard/build_data.py      # escriu dashboard/data.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter_kit"))

from data import load_hourly  # noqa: E402
from environment import CampusBatteryEnv  # noqa: E402
from evaluate import load_agent_class  # noqa: E402
from hard_env import HardBatteryEnv, to_hard  # noqa: E402
from oracle import solve_lp  # noqa: E402

ETA = 0.9 ** 0.5
DAYS = [
    ("2026-05-21", "Primavera assolellada", "Dijous de maig: molt de sol i el campus a ple rendiment."),
    ("2026-01-20", "Hivern ennuvolat", "Dimarts de gener: gairebé sense sol i consum alt."),
    ("2026-03-15", "Diumenge de sol", "Campus quasi buit i sol fort: sobra energia."),
    ("2026-08-12", "Agost a mig gas", "Estiu amb activitat reduïda i molt de sol."),
]
MODES = {
    "facil": {"label": "Fàcil", "cap": 2000.0, "pmax": 500.0, "pv_scale": 1.0, "horizon": 24,
              "agent": ROOT / "submission_easy_final" / "agent.py", "env": CampusBatteryEnv},
    "dificil": {"label": "Difícil", "cap": 6000.0, "pmax": 1500.0, "pv_scale": 12.0, "horizon": 3,
                "agent": ROOT / "submission_hard_final" / "agent.py", "env": HardBatteryEnv},
}


def clamp(p, soc, cap, pmax):
    max_c = max(0.0, min(pmax, (cap - soc) / ETA - 1e-4))
    max_d = max(0.0, min(pmax, soc * ETA - 1e-4))
    return float(np.clip(p, -max_d, max_c))


def replay(actions, soc0, cap, pmax):
    soc, out, socs = soc0, [], []
    for p in actions:
        p = clamp(p, soc, cap, pmax)
        soc = soc + p * ETA if p >= 0 else soc + p / ETA
        out.append(p)
        socs.append(soc)
    return out, socs


def run_agent(mode, frame, day):
    cfg = MODES[mode]
    start = pd.Timestamp(day) - pd.Timedelta(days=7)
    end = pd.Timestamp(day) + pd.Timedelta(days=2) - pd.Timedelta(hours=1)
    window = frame.loc[start:end]
    assert len(window) == 9 * 24, (mode, day, len(window))
    Agent = load_agent_class(cfg["agent"])
    agent = Agent()
    env = cfg["env"](window) if mode == "dificil" else cfg["env"](window)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    i0 = 7 * 24
    soc_series = [env.soc_init] + list(env.soc_series)
    actions = env.actions[i0:i0 + 24]
    return window, actions, float(soc_series[i0]), env.clipped + env.violations


def main():
    base = load_hourly()
    out = {"modes": {}}
    for mode, cfg in MODES.items():
        frame = to_hard(base) if mode == "dificil" else base
        days = []
        for day, title, blurb in DAYS:
            window, actions, soc0, infr = run_agent(mode, frame, day)
            d = window.loc[day]
            nxt = window.loc[pd.Timestamp(day) + pd.Timedelta(days=1):].iloc[:24]
            prev = window.loc[:pd.Timestamp(day) - pd.Timedelta(hours=1)].iloc[-1]
            load, pv = d.load_kw.values, d.pv_kw.values
            price, co2 = d.precio_eur_per_kwh.values, d.co2_kg_per_kwh.values
            net = load - pv
            b = np.maximum(net, 0)
            co2_b, cost_b = float((b * co2).sum()), float((b * price).sum())
            avg_c, avg_p = co2_b / b.sum(), cost_b / b.sum()
            w = 0.7 * co2 / co2_b + 0.3 * price / cost_b
            v_term = ETA * (0.7 * avg_c / co2_b + 0.3 * avg_p / cost_b)
            ch, dis, _ = solve_lp(net, w, soc0, cfg["cap"], cfg["pmax"], v_term=v_term)
            best, _ = replay(list(ch - dis), soc0, cfg["cap"], cfg["pmax"])
            agent, _ = replay(actions, soc0, cfg["cap"], cfg["pmax"])
            days.append({
                "date": day, "title": title, "blurb": blurb, "soc0": round(soc0, 1),
                "load": np.round(load, 1).tolist(), "pv": np.round(pv, 1).tolist(),
                "price": np.round(np.r_[price, nxt.precio_eur_per_kwh.values], 5).tolist(),
                "co2": np.round(np.r_[co2, nxt.co2_kg_per_kwh.values], 5).tolist(),
                "prev_load": round(float(prev.load_kw), 1), "prev_pv": round(float(prev.pv_kw), 1),
                "agent": np.round(agent, 1).tolist(), "best": np.round(best, 1).tolist(),
                "agent_infractions": int(infr),
            })
            print(f"{mode:8s} {day}  SOC0={soc0:7.0f}  infraccions setmana={infr}", flush=True)
        out["modes"][mode] = {k: v for k, v in cfg.items() if k in ("label", "cap", "pmax", "pv_scale", "horizon")}
        out["modes"][mode]["days"] = days
    path = Path(__file__).parent / "data.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Desat {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
