"""Estimació de robustesa de la candidata Hard congelada en finestres de 1.008 observacions.

NO modifica ni recalibra res de `experiments/oriol_hard_mode`: només importa el
motor oficial, l'adaptador `hard_observation`, la candidata congelada
(`candidate_agent.BatteryAgent` + JSON originals) i les polítiques del protocol.

Ús (des de l'arrel del repositori):
    python -B experiments/dani_hard_score_estimation/estimate.py verify
    python -B experiments/dani_hard_score_estimation/estimate.py audit
    python -B experiments/dani_hard_score_estimation/estimate.py reproduce
    python -B experiments/dani_hard_score_estimation/estimate.py windows --set primary
    python -B experiments/dani_hard_score_estimation/estimate.py windows --set dense --jobs 6
    python -B experiments/dani_hard_score_estimation/estimate.py windows --set weekly --jobs 8
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HARD = ROOT / "experiments/oriol_hard_mode"
OUT = HERE / "results"
sys.path.insert(0, str(ROOT))

from experiments.oriol_hard_mode import lab  # noqa: E402
from experiments.oriol_hard_mode.candidate_agent import BatteryAgent  # noqa: E402
from experiments.oriol_hard_mode.policies import make_policy, solve_hard_plan  # noqa: E402

WINDOW = 1008
SEED = 20261114  # declarada; tot el protocol és determinista (no hi ha mostreig aleatori)
PRIMARY_COUNT = 30
DENSE_STRIDE = 24
REFERENCE_SCORE = 23.03
FROZEN_ACTIONS_SHA = "a88ce6b3de14ea719e360ec343576b757607cc83b235ec17f56f4c2ba46b4081"
# Polítiques comparades; totes amb training.json congelat, sense cap ajust.
POLICIES = ["mpc24_annual", "standby", "solar", "mpc3_shadow090", "naive3", "mpc24_scenarios"]
SEASONS = {12: "hivern", 1: "hivern", 2: "hivern", 3: "primavera", 4: "primavera", 5: "primavera",
           6: "estiu", 7: "estiu", 8: "estiu", 9: "tardor", 10: "tardor", 11: "tardor"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def frozen_training():
    return lab.read(HARD / "training.json")


def build_policy(name):
    """Candidata = punt d'entrada real del ZIP; la resta = configuracions del protocol congelat."""
    if name == "mpc24_annual":
        return BatteryAgent()
    if name == "standby":
        return make_policy({"id": "standby", "kind": "standby"}, frozen_training())
    configuration = next(c for c in lab.read(HARD / "protocol.json")["models"] if c["id"] == name)
    return make_policy(configuration, frozen_training())


def controller_of(policy):
    return getattr(policy, "controller", policy)


# ---------------------------------------------------------------- verificacions

def verify():
    """Comprova que la candidata no s'ha tocat i que el training regenerable coincideix amb el congelat."""
    config = lab.read(HARD / "candidate_config.json")
    hashes = {name: sha(HARD / name) for name in config["source_sha256"]}
    mismatched = {n: h for n, h in hashes.items() if h != config["source_sha256"][n]}
    refit = lab.fit_training()  # en memòria; NO s'escriu enlloc
    frozen = frozen_training()
    diff = 0.0
    for key, value in frozen.items():
        if isinstance(value, (int, float)):
            diff = max(diff, abs(value - refit[key]))
        elif isinstance(value, list):
            diff = max(diff, float(np.max(np.abs(np.asarray(value, float) - np.asarray(refit[key], float)))))
    frame = lab.load_hard()
    env = lab.official_env()(frame, spec=lab.HARD_SPEC, export_price=0.0)
    raw = env.reset()
    adapted = lab.hard_observation(raw)
    report = {
        "frozen_hashes_match_candidate_config": not mismatched, "mismatched": mismatched,
        "training_refit_max_abs_diff_vs_frozen": diff,
        "official_environment_sha256": sha(lab.OFFICIAL / "starter_kit/environment.py"),
        "official_environment_sha256_seen_by_oriol": lab.read(HARD / "results.json")["environment_audit"]["official_environment_sha256"],
        "raw_market_vector_lengths": [len(raw["preus_properes_24h"]), len(raw["co2_properes_24h"])],
        "adapted_market_vector_lengths": [len(adapted["preus_properes_24h"]), len(adapted["co2_properes_24h"])],
        "raw_telemetry_keys": sorted(raw["noves_dades_5min"]),
        "spec": lab.HARD_SPEC, "env_capacity_kwh": env.cap, "env_power_kw": env.power, "env_efficiency": env.eta,
        "env_soc_init_kwh": env.soc_init, "export_price": env.export_price,
        "pv_scale_attr": frame.attrs.get("hard_pv_scale"),
        "pv_max_kw_hard": float(frame.pv_kw.max()), "rows": len(frame),
    }
    try:
        lab.scale_hard(frame)
        report["double_scaling_blocked"] = False
    except ValueError:
        report["double_scaling_blocked"] = True
    write_json(OUT / "verify.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- simulació

def oracle_score(frame, soc_init):
    """Cota superior NO causal: LP amb previsió perfecta i pesos exactes del score de la finestra."""
    net = (frame.load_kw - frame.pv_kw).to_numpy()
    base = np.maximum(net, 0)
    co2 = frame.co2_kg_per_kwh.to_numpy()
    price = frame.precio_eur_per_kwh.to_numpy()
    values = .7 * co2 / (base @ co2) + .3 * price / (base @ price)
    scale = 1.0 / values.mean()
    plan, _ = solve_hard_plan(values * scale, soc_init, net, lab.HARD_SPEC["capacity_kwh"], lab.HARD_SPEC["power_kw"])
    env = lab.official_env()(frame, spec=lab.HARD_SPEC, export_price=0.0)
    obs = env.reset()
    from experiments.oriol_hard_mode.safety import clamp_action
    for action in plan:
        obs, _ = env.step(clamp_action(action, lab.hard_observation(obs)))
    result = env.evaluate_results()
    return {"oracle_score": result["score"], "oracle_co2_pct": result["co2_estalviat_pct"],
            "oracle_cost_pct": result["cost_estalviat_pct"], "oracle_infraccions": result["infraccions"]}


def simulate(frame, policy_name, forecast_errors=False, monthly=False):
    """Un episodi Hard nou sobre `frame` (files consecutives en l'ordre original)."""
    started = perf_counter()
    policy = build_policy(policy_name)
    env = lab.official_env()(frame, spec=lab.HARD_SPEC, export_price=0.0)
    obs = lab.hard_observation(env.reset())
    index = frame.index
    load = frame.load_kw.to_numpy(); pv = frame.pv_kw.to_numpy()
    price = frame.precio_eur_per_kwh.to_numpy(); co2 = frame.co2_kg_per_kwh.to_numpy()
    errors = {k: [] for k in ["load0", "pv0", "load_ahead", "pv_ahead", "price_unknown", "co2_unknown"]}
    position = 0
    hour = pd.Timedelta(hours=1)
    while obs is not None:
        if len(obs["preus_properes_24h"]) > 3 or len(obs["co2_properes_24h"]) > 3:
            raise RuntimeError("Fuita: l'agent veuria més de 3 hores de mercat")
        action = policy.act(obs)
        if not np.isfinite(action):
            raise ValueError("Acció no finita")
        if forecast_errors:
            # Diagnòstic POSTERIOR a la decisió: els targets reals mai tornen a l'agent.
            forecast = getattr(controller_of(policy), "last_forecasts", None)
            if forecast is not None:
                errors["load0"].append(abs(forecast["load_kw"][0] - load[position]))
                errors["pv0"].append(abs(forecast["pv_kw"][0] - pv[position]))
                for lead in range(1, 24):
                    j = position + lead
                    if j >= len(frame) or index[j] - index[position] != lead * hour:
                        break
                    errors["load_ahead"].append(abs(forecast["load_kw"][lead] - load[j]))
                    errors["pv_ahead"].append(abs(forecast["pv_kw"][lead] - pv[j]))
                    if lead >= 3:
                        errors["price_unknown"].append(abs(forecast["price"][lead] - price[j]))
                        errors["co2_unknown"].append(abs(forecast["carbon"][lead] - co2[j]))
        raw, _ = env.step(action)
        obs = lab.hard_observation(raw)
        position += 1
    metrics = env.evaluate_results()
    net = load - pv
    base = np.maximum(net, 0)
    grid = np.asarray(env.grid); actions = np.asarray(env.actions)
    exports = np.asarray(env.exports); soc = np.asarray(env.soc_series)
    capacity = lab.HARD_SPEC["capacity_kwh"]; power = lab.HARD_SPEC["power_kw"]
    capture = np.minimum(np.maximum(actions, 0), np.maximum(-net, 0))
    waste = np.maximum(-actions - np.maximum(net, 0), 0)
    controller = controller_of(policy)
    row = {
        "policy": policy_name,
        "score": metrics["score"],
        "score_precise": float(100 * (.7 * (1 - (grid @ co2) / (base @ co2)) + .3 * (1 - (grid @ price) / (base @ price)))
                               - 0.2 * metrics["infraccions"]),
        "co2_pct": metrics["co2_estalviat_pct"], "cost_pct": metrics["cost_estalviat_pct"],
        "co2_saved_kg": metrics["co2_estalviat_kg"], "cost_saved_eur": metrics["cost_estalviat_eur"],
        "co2_baseline_kg": metrics["co2_baseline_kg"], "cost_baseline_eur": metrics["cost_baseline_eur"],
        "infraccions": metrics["infraccions"],
        "failures": int(getattr(controller, "failures", 0)), "lp_calls": int(getattr(controller, "lp_calls", 0)),
        "soc_initial_kwh": float(env.soc_init), "soc_final_kwh": float(env.soc),
        "soc_final_pct": float(env.soc / capacity),
        "hours_soc_empty": int(np.sum(soc < .01)), "hours_soc_full": int(np.sum(soc > capacity - .01)),
        "hours_action_max": int(np.sum(np.abs(actions) > power - 1e-3)),
        "hours_action_partial": int(np.sum((np.abs(actions) > 1e-6) & (np.abs(actions) <= power - 1e-3))),
        "hours_action_zero": int(np.sum(np.abs(actions) <= 1e-6)),
        "charge_kwh": float(np.maximum(actions, 0).sum()), "discharge_kwh": float(np.maximum(-actions, 0).sum()),
        "equivalent_cycles": float(np.maximum(-actions, 0).sum() / capacity),
        "exported_kwh": float(exports.sum()),
        "surplus_captured_kwh": float(capture.sum()),
        "battery_export_waste_kwh": float(waste.sum()),
        "grid_import_kwh": float(grid.sum()),
        "actions_sha256": hashlib.sha256(actions.astype("<f8").tobytes()).hexdigest(),
        "runtime_seconds": perf_counter() - started,
    }
    weekend = np.asarray(index.dayofweek >= 5)
    for label, mask in [("weekend", weekend), ("weekday", ~weekend)]:
        part = env.evaluate_results(mask=mask)
        row["score_" + label] = part["score"]
    if monthly:
        periods = index.to_period("M")
        row["monthly"] = {str(m): {k: v for k, v in env.evaluate_results(mask=np.asarray(periods == m)).items()
                                   if k in ("score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions")}
                          for m in periods.unique()}
    if forecast_errors:
        for key, values in errors.items():
            row["mae_" + key] = float(np.mean(values)) if values else None
    return row


def window_features(frame, window_id, rows_index):
    """Descriptors exògens de la finestra (independents de la política)."""
    index = frame.index
    deltas = np.asarray((index[1:] - index[:-1]) / pd.Timedelta(hours=1), dtype=float)
    gaps = deltas[deltas != 1]
    net = frame.load_kw - frame.pv_kw
    co2 = frame.co2_kg_per_kwh; price = frame.precio_eur_per_kwh
    value = .7 * co2 / co2.mean() + .3 * price / price.mean()
    day = index.normalize()
    daily = lambda s: s.groupby(day)
    full_days = daily(co2).count() == 24
    months = pd.Series(index.month).value_counts(normalize=True).sort_index()
    seasons = pd.Series([SEASONS[m] for m in index.month]).value_counts(normalize=True)
    base = net.clip(lower=0)
    return {
        "window_id": window_id, "start_index": int(rows_index[0]), "end_index": int(rows_index[-1]),
        "rows": len(frame), "first_timestamp": str(index[0]), "last_timestamp": str(index[-1]),
        "calendar_span_hours": float((index[-1] - index[0]) / pd.Timedelta(hours=1)) + 1,
        "gap_count": int(len(gaps)), "gap_missing_hours": float((gaps - 1).sum()) if len(gaps) else 0.0,
        "max_gap_hours": float(gaps.max()) if len(gaps) else 1.0,
        "month_share": {str(k): round(float(v), 4) for k, v in months.items()},
        "season_share": {k: round(float(v), 4) for k, v in seasons.items()},
        "dominant_season": seasons.idxmax(),
        "weekend_share": float(np.mean(index.dayofweek >= 5)),
        "pv_kwh": float(frame.pv_kw.sum()), "load_kwh": float(frame.load_kw.sum()),
        "load_mean_kw": float(frame.load_kw.mean()), "net_mean_kw": float(net.mean()),
        "pv_to_load_ratio": float(frame.pv_kw.sum() / frame.load_kw.sum()),
        "negative_net_hours": int((net < 0).sum()), "surplus_base_kwh": float((-net).clip(lower=0).sum()),
        "baseline_import_kwh": float(base.sum()),
        "price_mean": float(price.mean()), "price_std": float(price.std()),
        "co2_mean": float(co2.mean()), "co2_std": float(co2.std()),
        "price_daily_spread_mean": float((daily(price).max() - daily(price).min())[full_days].mean()),
        "co2_daily_spread_mean": float((daily(co2).max() - daily(co2).min())[full_days].mean()),
        "value_daily_spread_mean": float((daily(value).max() - daily(value).min())[full_days].mean()),
        "value_p90_minus_p10": float(value.quantile(.9) - value.quantile(.1)),
        "import_weighted_co2": float((base * co2).sum() / base.sum()),
        "import_weighted_price": float((base * price).sum() / base.sum()),
    }


def run_window(task):
    window_id, rows_index, policies, with_oracle = task
    frame = _FRAME.iloc[rows_index]
    features = window_features(frame, window_id, rows_index)
    rows = []
    for name in policies:
        row = {**features, **simulate(frame, name, forecast_errors=(name == "mpc24_annual"))}
        if name == "mpc24_annual" and with_oracle:
            row.update(oracle_score(frame, lab.HARD_SPEC["soc_init"] * lab.HARD_SPEC["capacity_kwh"]))
        rows.append(row)
    return rows


_FRAME = None


def init_worker():
    global _FRAME
    _FRAME = lab.load_hard()


def continuous_blocks(index):
    breaks = np.flatnonzero(np.diff(index.values) != np.timedelta64(1, "h")) + 1
    edges = [0, *breaks.tolist(), len(index)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


def window_starts(kind, frame):
    last = len(frame) - WINDOW
    if kind == "primary":  # 30 inicis equiespaiats sobre tot el rang possible; cap criteri de score
        return sorted({int(round(x)) for x in np.linspace(0, last, PRIMARY_COUNT)})
    if kind == "dense":  # sensibilitat: tots els inicis cada 24 files
        return list(range(0, last + 1, DENSE_STRIDE))
    raise ValueError(kind)


def weekly_blocks(frame):
    """Blocs de 168 files per mes, comptats des del final de les dades disponibles del mes.

    A = les 168 files immediatament anteriors a la setmana retinguda pel test oficial
    (últims 7 dies de cada mes); B i C = les dues setmanes anteriors. Cap criteri de score.
    """
    blocks = {}
    periods = frame.index.to_period("M")
    for month in periods.unique():
        rows = np.flatnonzero(periods == month)
        for k, label in enumerate("ABC"):
            if len(rows) >= 168 * (k + 1):
                end = len(rows) - 168 * k
                blocks[(str(month), label)] = rows[end - 168:end]
    return blocks


def weekly_tasks(frame, policies):
    """ESCENARIS HIPOTÈTICS amb l'estructura declarada a l'enunciat (setmanes de 7 dies de mesos diferents).

    Per a cada posició (A/B/C): totes les combinacions de 6 mesos (1.008 files, ordre cronològic,
    timestamps reals, salts reals) i l'episodi amb tots els mesos disponibles.
    """
    from itertools import combinations
    blocks = weekly_blocks(frame)
    tasks = []
    for label in "ABC":
        months = sorted(m for m, l in blocks if l == label)
        for subset in [tuple(months), *combinations(months, 6)]:
            rows = np.concatenate([blocks[(m, label)] for m in subset])
            name = "all" if len(subset) == len(months) else "+".join(m[2:] for m in subset)
            tasks.append((f"wk{label}:{name}", rows, policies, True))
    return tasks


def windows(kind, jobs):
    from multiprocessing import Pool
    init_worker()
    policies = POLICIES if kind == "primary" else ["mpc24_annual", "solar"]
    if kind == "weekly":
        tasks = weekly_tasks(_FRAME, policies)
        starts = [t[0] for t in tasks]
    else:
        starts = window_starts(kind, _FRAME)
        tasks = [(f"w{s:04d}", np.arange(s, s + WINDOW), policies, True) for s in starts]
    began = perf_counter()
    if jobs <= 1:
        results = [run_window(t) for t in tasks]
    else:
        with Pool(jobs, initializer=init_worker) as pool:
            results = pool.map(run_window, tasks, chunksize=1)
    rows = [row for group in results for row in group]
    payload = {"set": kind, "window_rows": WINDOW, "seed": SEED, "starts": starts, "policies": policies,
               "jobs": jobs, "wall_seconds": perf_counter() - began,
               "soc_protocol": "Cada finestra: entorn Hard nou, SOC inicial 50% (3.000 kWh), agent nou; SOC conservat als salts interns com fa el motor oficial.",
               "rows": rows}
    write_json(OUT / f"windows_{kind}.json", payload)
    flat = pd.json_normalize(rows)
    flat.to_csv(OUT / f"windows_{kind}.csv", index=False)
    if kind == "weekly":  # 1.134 escenaris: el JSON complet seria redundant amb el CSV
        payload["rows"] = [r for r in rows if r["window_id"].endswith(":all")]
        write_json(OUT / f"windows_{kind}.json", payload)
    mpc = flat[flat.policy == "mpc24_annual"]
    print(f"{kind}: {len(starts)} finestres, {payload['wall_seconds']:.1f}s; MPC score mitjà {mpc.score.mean():.2f} "
          f"min {mpc.score.min():.2f} max {mpc.score.max():.2f}; infraccions {int(flat.infraccions.sum())}")


def audit():
    frame = lab.load_hard()
    index = frame.index
    deltas = (index[1:] - index[:-1]) / pd.Timedelta(hours=1)
    gap_rows = [{"row_before": int(i), "last_before": str(index[i]), "first_after": str(index[i + 1]),
                 "delta_hours": float(d), "missing_hours": float(d - 1)}
                for i, d in enumerate(deltas) if d != 1]
    blocks = [{"start_index": a, "end_index": b - 1, "rows": b - a, "first": str(index[a]), "last": str(index[b - 1]),
               "possible_contiguous_1008_windows": max(0, b - a - WINDOW + 1)} for a, b in continuous_blocks(index)]
    last = len(frame) - WINDOW
    breaks = np.asarray([g["row_before"] for g in gap_rows])
    crossing = np.asarray([int(np.sum((breaks >= s) & (breaks < s + WINDOW - 1))) for s in range(last + 1)])
    span = np.asarray([(index[s + WINDOW - 1] - index[s]) / pd.Timedelta(hours=1) + 1 for s in range(last + 1)])
    report = {
        "rows": len(frame), "first": str(index[0]), "last": str(index[-1]),
        "calendar_hours_first_to_last": float((index[-1] - index[0]) / pd.Timedelta(hours=1)) + 1,
        "coverage_of_calendar": len(frame) / (float((index[-1] - index[0]) / pd.Timedelta(hours=1)) + 1),
        "gap_count": len(gap_rows), "gaps": gap_rows, "continuous_blocks": blocks,
        "rows_per_month": {str(k): int(v) for k, v in frame.groupby(index.to_period("M")).size().items()},
        "possible_1008_row_windows": last + 1,
        "windows_without_gap": int(np.sum(crossing == 0)),
        "windows_crossing_gap": int(np.sum(crossing > 0)),
        "gaps_crossed_distribution": {str(k): int(v) for k, v in pd.Series(crossing).value_counts().sort_index().items()},
        "calendar_span_hours_of_1008_row_windows": {"min": float(span.min()), "median": float(np.median(span)), "max": float(span.max())},
        "soc_at_gaps": "environment.py no coneix els timestamps a step(): el SOC es conserva entre files consecutives encara que hi hagi salt; reset() només a l'inici de l'episodi.",
        "negative_net_hours": int((frame.load_kw - frame.pv_kw < 0).sum()),
    }
    write_json(OUT / "temporal_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def reproduce():
    frame = lab.load_hard()
    rows = []
    for name in ["mpc24_annual", "standby", "solar", "naive3"]:
        row = simulate(frame, name, monthly=True)
        rows.append(row)
        print(f"{name:16} score={row['score']:.2f} CO2={row['co2_pct']:.2f}% cost={row['cost_pct']:.2f}% "
              f"infr={row['infraccions']} fallades={row['failures']} t={row['runtime_seconds']:.2f}s sha={row['actions_sha256'][:12]}")
    candidate = rows[0]
    expected = lab.read(HARD / "results.json")["full_mpc24_annual"]
    report = {"steps": len(frame), "rows": rows,
              "expected": {k: expected[k] for k in ["score", "co2_estalviat_pct", "cost_estalviat_pct", "infraccions"]},
              "expected_actions_sha256": FROZEN_ACTIONS_SHA,
              "actions_sha256_identical": candidate["actions_sha256"] == FROZEN_ACTIONS_SHA,
              "score_identical": candidate["score"] == expected["score"]}
    write_json(OUT / "reproduce_full.json", report)
    print("hash d'accions idèntic al congelat:", report["actions_sha256_identical"], "| score idèntic:", report["score_identical"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["verify", "audit", "reproduce", "windows"])
    parser.add_argument("--set", default="primary", choices=["primary", "dense", "weekly"])
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    {"verify": verify, "audit": audit, "reproduce": reproduce}.get(args.phase, lambda: windows(args.set, args.jobs))()


if __name__ == "__main__":
    main()
