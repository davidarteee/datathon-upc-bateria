"""Entrenament offline dels paràmetres de l'agent v2 i banc de validació amb setmanes retingudes.

    python -B experiments/dani_hard_v2/train.py validate      # hold-out de blocs A i B (com el test: setmanes senceres)
    python -B experiments/dani_hard_v2/train.py final         # entrena amb tot el train -> params.json
    python -B experiments/dani_hard_v2/train.py robust        # versió robusta (tau 20) -> params_robust.json
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/dani_hard_score_estimation"))
sys.path.insert(0, str(HERE))

import estimate  # noqa: E402  (blocs setmanals i oracle)
import model  # noqa: E402
from experiments.oriol_hard_mode import lab  # noqa: E402  (motor oficial + dades; mai dins del ZIP)


def market_frame():
    data = lab.OFFICIAL / "data/train"
    price = pd.read_csv(data / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")["precio_eur_per_kwh"]
    carbon = pd.read_csv(data / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")["co2_kg_per_kwh"]
    return pd.DataFrame({"price": price, "co2": carbon}).dropna()


def fill_months(table):
    """Mesos sense dades (novembre per a demanda/solar): mitjana dels veïns disponibles."""
    for m in range(12):
        if table[m] is None:
            before = next(table[(m - j) % 12] for j in range(1, 12) if table[(m - j) % 12] is not None)
            after = next(table[(m + j) % 12] for j in range(1, 12) if table[(m + j) % 12] is not None)
            table[m] = (np.asarray(before) + np.asarray(after)) / 2
    return [np.asarray(t).tolist() for t in table]


def fit_profiles(frame, market):
    params = {}
    dtype = np.asarray([model.day_type(ts.to_pydatetime()) for ts in frame.index])
    load = [None] * 12; pv = [None] * 12; clear = [None] * 12
    for m in range(12):
        rows = frame.index.month == m + 1
        if rows.sum() < 24 * 7:
            continue
        sub = frame[rows]; types = dtype[rows]
        by_type = []
        for t in range(3):
            pick = sub[types == t]
            hours = pick.load_kw.groupby(pick.index.hour).mean().reindex(range(24))
            by_type.append(hours)
        # tipus de dia sense mostres: hereta del més semblant
        if by_type[2].isna().any():
            by_type[2] = by_type[2].fillna(by_type[1])
        if by_type[1].isna().any():
            by_type[1] = by_type[1].fillna(by_type[2])
        for t in range(3):
            by_type[t] = by_type[t].fillna(sub.load_kw.groupby(sub.index.hour).mean().reindex(range(24))).fillna(sub.load_kw.mean())
        load[m] = [b.tolist() for b in by_type]
        pv[m] = sub.pv_kw.groupby(sub.index.hour).mean().reindex(range(24)).fillna(0.0).tolist()
        clear[m] = sub.pv_kw.groupby(sub.index.hour).quantile(0.9).reindex(range(24)).fillna(0.0).tolist()
    params["load_profile"] = fill_months(load)
    params["pv_profile"] = fill_months(pv)
    params["pv_clear"] = fill_months(clear)
    params["pv_kbar"] = [float(np.sum(params["pv_profile"][m]) / max(1.0, np.sum(params["pv_clear"][m]))) for m in range(12)]
    mtype = np.asarray([model.market_type(ts.to_pydatetime()) for ts in market.index])
    for var in ["price", "co2"]:
        table = [None] * 12
        for m in range(12):
            rows = market.index.month == m + 1
            if rows.sum() < 24 * 7:
                continue
            sub = market[rows]; types = mtype[rows]
            both = []
            for t in range(2):
                pick = sub[types == t][var]
                both.append(pick.groupby(pick.index.hour).mean().reindex(range(24))
                            .fillna(sub[var].groupby(sub.index.hour).mean().reindex(range(24))).tolist())
            table[m] = both
        params[var + "_profile"] = fill_months(table)
    grid = (frame.load_kw - frame.pv_kw).clip(lower=0)
    params["mean_carbon"] = float((grid * frame.co2_kg_per_kwh).sum() / grid.sum())
    params["mean_price"] = float((grid * frame.precio_eur_per_kwh).sum() / grid.sum())
    return params


def full_series():
    """Sèries horàries amb la cobertura pròpia de cada sensor (la demanda arriba fins al 24/08; la solar fins al 16/08)."""
    data = lab.OFFICIAL / "data/train"
    meter = pd.read_csv(data / "contador.csv", parse_dates=["Fecha"], usecols=["Fecha", "Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).set_index("Fecha")
    load = (-meter.iloc[:, 0]).clip(lower=0, upper=5000).resample("1h").mean().dropna()
    return load


def kernel_table(series, groups, n_groups, tau, quantile=None, window=15):
    """Taula [dia de l'any][grup][hora] amb mitjana ponderada per proximitat cíclica de dates."""
    doy = series.index.dayofyear.to_numpy() - 1
    hours = series.index.hour.to_numpy()
    values = series.to_numpy()
    table = np.zeros((365, n_groups, 24))
    days = np.arange(365)
    for g in range(n_groups):
        for h in range(24):
            pick = (groups == g) & (hours == h)
            if not pick.any():
                table[:, g, h] = np.nan
                continue
            delta = np.abs(days[:, None] - np.minimum(doy[pick], 364)[None, :])
            delta = np.minimum(delta, 365 - delta)
            if quantile is None:
                for width in (tau, 3 * tau, 10 * tau):
                    w = np.exp(-0.5 * (delta / width) ** 2)
                    total = w.sum(axis=1)
                    estimate_ = (w * values[pick][None, :]).sum(axis=1) / np.maximum(total, 1e-12)
                    if width == tau:
                        table[:, g, h] = estimate_; enough = total >= 2.0
                    else:
                        table[~enough, g, h] = estimate_[~enough]; enough = enough | (total >= 2.0)
            else:
                v = values[pick]
                for d in range(365):
                    for width in (window, 2 * window, 4 * window, 400):
                        near = v[delta[d] <= width]
                        if len(near) >= 8:
                            break
                    table[d, g, h] = np.quantile(near, quantile)
    return table


def fit_profiles_daily(frame, market, load_full, tau=10.0):
    params = {"daily": True}
    taus = tau if isinstance(tau, dict) else {"load": tau, "pv": tau, "price": tau, "co2": tau}
    ltype = np.asarray([model.day_type(ts.to_pydatetime()) for ts in load_full.index])
    load = kernel_table(load_full, ltype, 3, taus["load"])
    for t in (2, 1):   # tipus de dia sense mostres: hereta
        other = 1 if t == 2 else 2
        bad = np.isnan(load[:, t, :]); load[:, t, :][bad] = load[:, other, :][bad]
    bad = np.isnan(load); load[bad] = np.broadcast_to(load[:, :1, :], load.shape)[bad]
    params["load_profile"] = np.round(load, 2).tolist()
    zeros = np.zeros(len(frame), int)
    pv = kernel_table(frame.pv_kw, zeros, 1, taus["pv"])[:, 0, :]
    clear = kernel_table(frame.pv_kw, zeros, 1, taus["pv"], quantile=0.9)[:, 0, :]
    params["pv_profile"] = np.round(pv, 2).tolist()
    params["pv_clear"] = np.round(clear, 2).tolist()
    params["pv_kbar"] = [float(pv[d].sum() / max(1.0, clear[d].sum())) for d in range(365)]
    mtype = np.asarray([model.market_type(ts.to_pydatetime()) for ts in market.index])
    for var in ["price", "co2"]:
        params[var + "_profile"] = np.round(kernel_table(market[var], mtype, 2, taus[var]), 6).tolist()
    grid = (frame.load_kw - frame.pv_kw).clip(lower=0)
    params["mean_carbon"] = float((grid * frame.co2_kg_per_kwh).sum() / grid.sum())
    params["mean_price"] = float((grid * frame.precio_eur_per_kwh).sum() / grid.sum())
    return params


def synthetic_obs(frame, i):
    """Mateixa informació que dona l'entorn Hard a la fila i (sense SOC real: no cal per a la previsió)."""
    index = frame.index
    prev = frame.iloc[i - 1] if i else frame.iloc[0]
    end = min(i + 3, len(frame))
    return {"timestamp": index[i], "soc_kwh": 3000.0,
            "preus_properes_24h": frame.precio_eur_per_kwh.iloc[i:end].tolist(),
            "co2_properes_24h": frame.co2_kg_per_kwh.iloc[i:end].tolist(),
            "noves_dades_5min": {"ultim_consum_kw": float(prev.load_kw), "ultima_solar_kw": float(prev.pv_kw)}}


DEFAULT_USE = {"load": [0, 1, 2], "pv": [0, 1, 2], "price": [0, 1, 2], "co2": [0, 1, 2]}


def collect(frame, params, horizon=24):
    """Passada causal pel train: per variable i horitzó, [bucket, residu objectiu, totes les característiques]."""
    forecaster = model.Forecaster({**params, "coef": {}}, horizon)
    position = {ts.to_pydatetime(): i for i, ts in enumerate(frame.index)}
    truth = {"load": frame.load_kw.to_numpy(), "pv": frame.pv_kw.to_numpy(),
             "price": frame.precio_eur_per_kwh.to_numpy(), "co2": frame.co2_kg_per_kwh.to_numpy()}
    samples = {v: [[] for _ in range(horizon)] for v in model.VARS}
    for i in range(len(frame)):
        now, known, prices, carbon = forecaster.update(synthetic_obs(frame, i))
        targets, base, clear, feats = forecaster.features(now, known, prices, carbon)
        for k, ts in enumerate(targets):
            j = position.get(ts)
            if j is None:
                continue
            for var in model.VARS:
                if var == "pv" and clear[k] <= 30.0:
                    continue
                samples[var][k].append([now.hour, truth[var][j] - base[var][k], *[f[k] for f in feats[var]],
                                        base[var][k], ts.timetuple().tm_yday, model.day_type(ts), clear[k]])
    return {v: [np.asarray(s, float) for s in samples[v]] for v in model.VARS}


def fit_coefficients(samples, use, bucket_hours=24, horizon=24, ridge=1e-3, minimum=150):
    coef = {}
    for var in model.VARS:
        cols_all = use[var]
        coef[var] = []
        for b in range(24 // bucket_hours):
            tables = []
            for k in range(horizon):
                table = {}
                data = samples[var][k]
                if len(data):
                    data = data[(data[:, 0] // bucket_hours) == b]
                if len(data):
                    y, x = data[:, 1], data[:, 2:-4][:, cols_all]
                    available = ~np.isnan(x)
                    patterns = {tuple(row) for row in available}
                    for pattern in patterns:
                        cols = [j for j in range(len(cols_all)) if pattern[j]]
                        if not cols:
                            continue
                        ok = available[:, cols].all(axis=1)
                        if ok.sum() < minimum:
                            continue
                        a = x[ok][:, cols]; t = y[ok]
                        scale = (a ** 2).mean(axis=0) + 1e-9
                        w = np.linalg.solve(a.T @ a / len(a) + ridge * np.diag(scale), a.T @ t / len(a))
                        full = [0.0] * len(cols_all)
                        for j, value in zip(cols, w):
                            full[j] = float(value)
                        table["".join("1" if v else "0" for v in pattern)] = full
                tables.append(table)
            coef[var].append(tables)
    return coef


def fit(frame, market, horizon=24, use=None, bucket_hours=24, ridge=1e-3, samples=None, tau=None, load_full=None):
    params = fit_profiles(frame, market) if tau is None else fit_profiles_daily(frame, market, load_full, tau)
    use = use or DEFAULT_USE
    if samples is None:
        samples = collect(frame, params, horizon)
    params["use"] = use
    params["bucket_hours"] = bucket_hours
    params["coef"] = fit_coefficients(samples, use, bucket_hours, horizon, ridge)
    params["horizon"] = horizon
    params["net_sigma"] = net_sigma(frame, params)
    return params


def net_sigma(frame, params):
    """RMSE de l'error de previsió de demanda neta per horitzó i hora del dia objectiu."""
    horizon = params["horizon"]
    forecaster = model.Forecaster(params, horizon)
    errors = [[[] for _ in range(24)] for _ in range(horizon)]
    net = (frame.load_kw - frame.pv_kw).to_numpy()
    index = frame.index; hour = pd.Timedelta(hours=1)
    for i in range(len(frame)):
        out = forecaster.forecast(synthetic_obs(frame, i))
        for k in range(horizon):
            j = i + k
            if j >= len(frame) or index[j] - index[i] != k * hour:
                break
            errors[k][index[j].hour].append(net[j] - (out["load"][k] - out["pv"][k]))
    table = [[float(np.sqrt(np.mean(np.square(e)))) if len(e) >= 20 else 0.0 for e in row] for row in errors]
    params["net_sigma_lead"] = table
    return table[0]


def run_episode(agent, frame, forecast_errors=True):
    """Episodi Hard oficial amb `hard_observation`; errors de previsió mesurats després de decidir."""
    env = lab.official_env()(frame, spec=lab.HARD_SPEC, export_price=0.0)
    obs = lab.hard_observation(env.reset())
    index = frame.index; hour = pd.Timedelta(hours=1)
    truth = {"load": frame.load_kw.to_numpy(), "pv": frame.pv_kw.to_numpy(),
             "price": frame.precio_eur_per_kwh.to_numpy(), "co2": frame.co2_kg_per_kwh.to_numpy()}
    errors = {v: [] for v in model.VARS}
    i = 0; started = perf_counter()
    while obs is not None:
        action = agent.act(obs)
        forecast = getattr(agent, "last_forecasts", None)
        if forecast_errors and forecast is not None:
            names = {"load": "load", "pv": "pv", "price": "price", "co2": "co2"}
            if "load_kw" in forecast:
                names = {"load": "load_kw", "pv": "pv_kw", "price": "price", "co2": "carbon"}
            for lead in range(0, 24):
                j = i + lead
                if j >= len(frame) or index[j] - index[i] != lead * hour:
                    break
                for var, key in names.items():
                    if var in ("price", "co2") and lead < 3:
                        continue
                    errors[var].append(abs(forecast[key][lead] - truth[var][j]))
        raw, _ = env.step(action); obs = lab.hard_observation(raw); i += 1
    result = env.evaluate_results()
    out = {"score": result["score"], "co2": result["co2_estalviat_pct"], "cost": result["cost_estalviat_pct"],
           "infr": result["infraccions"], "failures": getattr(agent, "failures", 0), "seconds": round(perf_counter() - started, 2)}
    out.update({"mae_" + v: round(float(np.mean(e)), 4) for v, e in errors.items() if e})
    return out


def holdout(label):
    frame = lab.load_hard()
    blocks = estimate.weekly_blocks(frame)
    rows = np.concatenate([blocks[k] for k in sorted(blocks) if k[1] == label])
    keep = np.ones(len(frame), bool); keep[rows] = False
    test = frame.iloc[rows]
    market = market_frame()
    market = market[~market.index.isin(test.index)]
    global LOAD_FULL
    load_full = full_series()
    LOAD_FULL = load_full[~load_full.index.isin(test.index)]
    return frame[keep], market, test


LOAD_FULL = None


def validate(configs=None):
    from experiments.oriol_hard_mode.candidate_agent import BatteryAgent
    report = {}
    for label in "AB":
        train, market, test = holdout(label)
        began = perf_counter()
        params = fit(train, market)
        fit_seconds = perf_counter() - began
        rows = {"v1_frozen": run_episode(BatteryAgent().controller, test),
                "v2": run_episode(model.HardAgentV2(params), test),
                "v2_profiles_only": run_episode(model.HardAgentV2({**params, "coef": {}}), test)}
        oracle = estimate.oracle_score(test, 3000.0)["oracle_score"]
        report[label] = {"oracle": oracle, "fit_seconds": round(fit_seconds, 1), **rows}
        print(label, "oracle", oracle, "fit", round(fit_seconds, 1), "s")
        for name, row in rows.items():
            print("  ", name, row)
    (HERE / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


FINAL = {"use": {"load": [0, 1, 2], "pv": [0, 1, 2], "price": [0, 1, 2, 3, 4, 5, 6], "co2": [0, 1, 2, 3, 4, 5, 6, 7]},
         "tau": 12.0}
FINAL_CONFIG = {"stochastic": True, "s_now": 1.0, "s_future": 1.0, "adaptive_weights": 168, "episode_lengths": [1008, 2016], "end_cost": 1.0}


ROBUST = {**FINAL, "tau": 20.0}   # versió robusta: perfils més llisos (validat a experiments/robust/)


def robust():
    """Com final(), però amb perfils per dia de l'any més llisos (tau 20 en lloc de 12).
    A la validació amb setmanes retingudes té el millor score en setmanes no vistes (20,72 vs 20,69)
    i la meitat de diferència vist/no vist (0,11 vs 0,21)."""
    frame = lab.load_hard()
    params = fit(frame, market_frame(), load_full=full_series(), **ROBUST)
    params["config"] = FINAL_CONFIG
    out = HERE / "params_robust.json"
    out.write_text(json.dumps(params, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(out.name, out.stat().st_size, "bytes")


def final():
    """Paràmetres definitius: mateix disseny validat, entrenat amb tot el train."""
    frame = lab.load_hard()
    params = fit(frame, market_frame(), load_full=full_series(), **FINAL)
    params["config"] = FINAL_CONFIG
    (HERE / "params.json").write_text(json.dumps(params, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print("params.json", (HERE / "params.json").stat().st_size, "bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate", "final", "robust"])
    args = parser.parse_args()
    {"validate": validate, "final": final, "robust": robust}[args.phase]()
