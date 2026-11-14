"""
src/evaluate.py
Avalua un agent sobre les dades de train amb el simulador oficial (starter_kit/environment.py).

Ús:
    python src/evaluate.py                          # avalua submission/agent.py
    python src/evaluate.py starter_kit/agent.py     # avalua qualsevol agent.py

Dos modes:
  - "complet": tot el train seguit (una sola simulació).
  - "per setmanes": simula l'avaluació oficial — trossos de 7 dies (168 h) començant cada un
    amb SOC = 50% i un agent nou. És el número que hem de mirar per comparar versions.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))
sys.path.insert(0, str(ROOT / "src"))

from environment import CampusBatteryEnv  # noqa: E402
from data import load_hourly, split_weeks  # noqa: E402


def load_agent_class(path: Path):
    spec = importlib.util.spec_from_file_location(f"agent_{path.stem}_{id(path)}", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod.BatteryAgent


def run(agent, df: pd.DataFrame) -> CampusBatteryEnv:
    env = CampusBatteryEnv(df)
    obs = env.reset()
    done = False
    while not done:
        obs, done = env.step(agent.act(obs))
    return env


def aggregate(envs):
    """Agrega diverses simulacions com si fossin un únic període d'avaluació."""
    g, b, pr, co, infr = [], [], [], [], 0
    for e in envs:
        n = len(e.grid)
        g.append(np.asarray(e.grid))
        b.append(np.maximum(e.df["load_kw"].values[:n] - e.df["pv_kw"].values[:n], 0))
        pr.append(e.df["precio_eur_per_kwh"].values[:n])
        co.append(e.df["co2_kg_per_kwh"].values[:n])
        infr += e.clipped + e.violations
    g, b, pr, co = map(np.concatenate, (g, b, pr, co))
    d_co2 = 1 - (g * co).sum() / (b * co).sum()
    d_cost = 1 - (g * pr).sum() / (b * pr).sum()
    d_co2, d_cost = float(d_co2), float(d_cost)
    raw = 0.7 * d_co2 + 0.3 * d_cost
    return {
        "score_sense_penal": round(100 * raw, 3),
        "score_penal_local(0.2)": round(max(0, 100 * raw - 0.2 * infr), 3),
        "score_penal_enunciat(1.0)": round(max(0, 100 * raw - 1.0 * infr), 3),
        "co2_estalviat_pct": round(100 * d_co2, 3),
        "cost_estalviat_pct": round(100 * d_cost, 3),
        "infraccions": int(infr),
    }


def evaluate(agent_path: Path, week_hours: int = 168):
    AgentCls = load_agent_class(agent_path)
    df = load_hourly()

    t0 = time.time()
    full = aggregate([run(AgentCls(), df)])
    t_full = time.time() - t0

    chunks = []
    for block in split_weeks(df):
        for i in range(0, len(block) - 24 + 1, week_hours):
            chunks.append(block.iloc[i:i + week_hours])
    t0 = time.time()
    weekly = aggregate([run(AgentCls(), c) for c in chunks])
    t_week = time.time() - t0

    print(f"\n=== {agent_path} ===")
    print(f"Complet ({len(df)} h, {t_full:.1f}s): {full}")
    print(f"Per setmanes ({len(chunks)} trossos, {t_week:.1f}s): {weekly}")
    return full, weekly


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "submission" / "agent.py"
    evaluate(target.resolve())
