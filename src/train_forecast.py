"""
src/train_forecast.py
Entrena els models de predicció del Mode Difícil i els desa a submission_hard/params.json
(només números: l'agent els llegeix amb json, sense pickle).

Per a cada variable i cada horitzó ("lead") ajustem una regressió lineal (Ridge) amb features
que l'agent pot calcular en temps real a partir del que ha vist:
  - perfil típic (mes × [laborable/cap de setmana] × hora) del valor a predir
  - valor d'ahir a la mateixa hora
  - darrer valor observat, expressat com a desviació respecte al seu perfil
  - (solar) irradiància de la darrera hora i índex de claredat

Ús:
    python src/train_forecast.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_hourly  # noqa: E402
from hard_env import to_hard  # noqa: E402

OUT = ROOT / "submission_hard" / "params.json"
H = 24


def daytype(idx):
    return (idx.dayofweek >= 5).astype(int)          # 0 laborable, 1 cap de setmana


def profile(s: pd.Series, by_daytype: bool):
    """Taula [12 mesos][2 tipus de dia][24 h] (mitjana). Buits -> perfil mensual/global."""
    df = pd.DataFrame({"v": s, "m": s.index.month - 1, "d": daytype(s.index) if by_daytype else 0,
                       "h": s.index.hour})
    tab = np.full((12, 2, 24), np.nan)
    for (m, d, h), v in df.groupby(["m", "d", "h"])["v"].mean().items():
        tab[m, d, h] = v
        if not by_daytype:
            tab[m, 1, h] = v
    glob = df.groupby("h")["v"].mean().reindex(range(24)).values
    for m in range(12):
        for d in range(2):
            row = tab[m, d]
            row[np.isnan(row)] = glob[np.isnan(row)]
    return tab


def prof_at(tab, idx):
    return tab[idx.month - 1, daytype(idx), idx.hour]


def fit_var(s: pd.Series, tab, first_lead, last_lead, last_obs_offset, extra=None):
    """s: sèrie horària (índex continu, NaN als buits).
    last_obs_offset: posició del darrer valor observat respecte a t (preu/CO2: +2, consum/solar: -1).
    Features per a target ts = t+h (mateixes que calcula l'agent, vegeu BatteryAgent._features):
      py  perfil(ts) | y1 valor(ts-24) | y2 valor(ts-48) | d0, d1 desviació dels 2 darrers observats
      dm  desviació mitjana de les darreres 24 h observades | (solar) claredat · py"""
    idx = s.index
    p = pd.Series(prof_at(tab, idx), index=idx)
    dev = (s - p)
    dm_all = dev.rolling(24, min_periods=1).mean()
    coefs, maes, preds = {}, {}, {}
    for h in range(first_lead + last_obs_offset, last_lead + last_obs_offset + 1):
        y = s.shift(-h)
        py = p.shift(-h)
        y1 = s.shift(-(h - 24)).fillna(py)
        y2 = s.shift(-(h - 48)).fillna(py)
        d0 = dev.shift(-last_obs_offset).fillna(0)
        d1 = dev.shift(-(last_obs_offset - 1)).fillna(0)
        dm = dm_all.shift(-last_obs_offset).fillna(0)
        cols = [py, y1, y2, d0, d1, dm]
        if extra is not None:
            for e in extra:
                cols.append(e(h))
        X = pd.concat(cols, axis=1)
        m = y.notna() & X.notna().all(axis=1)
        X_, y_ = X[m].values, y[m].values
        r = Ridge(alpha=1.0).fit(X_, y_)
        pred = r.predict(X_)
        preds[h] = pd.Series(pred, index=X[m].index)
        lead = h - last_obs_offset
        coefs[str(h)] = [float(r.intercept_)] + [float(c) for c in r.coef_]
        maes[lead] = (float(np.mean(np.abs(pred - y_))), float(np.mean(np.abs(py[m].values - y_))))
    return coefs, maes, preds


def fit_params(df: pd.DataFrame, verbose: bool = False) -> dict:
    """Entrena tots els models amb el DataFrame horari (ja en Mode Difícil) i retorna params."""
    full = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="1h"))

    params = {"pv_scale": 12.0}
    tabs = {
        "precio": profile(df["precio_eur_per_kwh"], by_daytype=True),
        "co2": profile(df["co2_kg_per_kwh"], by_daytype=True),
        "load": profile(df["load_kw"], by_daytype=True),
        "pv": profile(df["pv_kw"], by_daytype=False),
        "irr": profile(df["irradiancia_wm2"], by_daytype=False),
    }
    params["profiles"] = {k: np.round(v, 6).tolist() for k, v in tabs.items()}

    report = {}
    preds_all = {}
    # Preu i CO2: coneixem t, t+1, t+2 -> cal predir t+3..t+23
    for key, col in [("precio", "precio_eur_per_kwh"), ("co2", "co2_kg_per_kwh")]:
        c, m, pr = fit_var(full[col], tabs[key], 1, 21, 2)
        preds_all[key] = pr
        params[f"coef_{key}"] = c
        report[key] = m

    # Consum: darrer observat t-1 -> predir t..t+23
    c, m, pred_load = fit_var(full["load_kw"], tabs["load"], 1, 24, -1)
    params["coef_load"] = c
    report["load"] = m

    # Solar: + claredat de la darrera hora (pv observat / pv típic) escalada al perfil del target
    pv = full["pv_kw"]
    p_pv = pd.Series(prof_at(tabs["pv"], pv.index), index=pv.index)
    irr = full["irradiancia_wm2"]
    p_irr = pd.Series(prof_at(tabs["irr"], irr.index), index=irr.index)
    clear_last = ((irr.shift(1) + 1) / (p_irr.shift(1) + 1)).clip(0, 2).fillna(1.0)
    extra = [lambda h: (clear_last * p_pv.shift(-h))]
    c, m, pred_pv = fit_var(pv, tabs["pv"], 1, 24, -1, extra=extra)
    params["coef_pv"] = c
    report["pv"] = m

    preds_all["load"], preds_all["pv"] = pred_load, pred_pv
    params["scen"] = residual_paths(full, preds_all)

    # Error de la predicció del consum net de l'hora actual (h=0): quantils per hora del dia.
    # L'agent no descarrega més que (consum net previst - quantil de sobreestimació), perquè
    # l'energia descarregada per sobre del consum real es perd (exportació a 0 €).
    net_hat = (pred_load[0] - pred_pv[0]).dropna()
    net_true = (full["load_kw"] - full["pv_kw"]).reindex(net_hat.index)
    err = (net_hat - net_true).dropna()
    params["net_err_q"] = {str(q): [float(max(0.0, err[err.index.hour == h].quantile(q))) for h in range(24)]
                           for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)}

    # Història inicial: les últimes 72 h de cada bloc de train. El test (últims 7 dies de cada mes)
    # comença just quan acaba cada bloc, així l'agent no comença "a cegues".
    hist = {}
    gap = df.index.to_series().diff() > pd.Timedelta(hours=1)
    for _, b in df.groupby(gap.cumsum()):
        tail = b.iloc[-72:]
        for key, col in [("precio", "precio_eur_per_kwh"), ("co2", "co2_kg_per_kwh"), ("load", "load_kw"),
                         ("pv", "pv_kw"), ("irr", "irradiancia_wm2")]:
            hist.setdefault(key, {}).update({ts.strftime("%Y-%m-%d %H:%M"): round(float(v), 5)
                                             for ts, v in tail[col].items()})
    params["hist"] = hist

    if verbose:
        for k, m in report.items():
            leads = sorted(m)
            show = [leads[0], leads[len(leads) // 2], leads[-1]]
            print(k, {l: f"model {m[l][0]:.4g} vs perfil {m[l][1]:.4g}" for l in show})
    return params


def residual_paths(full: pd.DataFrame, preds_all: dict, per_hour: int = 12, seed: int = 0) -> list:
    """Camins d'error històrics (real - predit) de les 4 variables, presos al MATEIX instant t
    (així es conserva la correlació entre variables i entre hores). Per a cada hora del dia es
    guarden `per_hour` camins de 24 h. L'agent els suma a la seva predicció per crear escenaris."""
    cols = {"precio": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh", "load": "load_kw", "pv": "pv_kw"}
    R = {}
    for key, pr in preds_all.items():
        mat = {}
        for h in range(24):
            if h in pr:
                truth = full[cols[key]].shift(-h).reindex(pr[h].index)
                mat[h] = truth - pr[h]
            else:
                mat[h] = pd.Series(0.0, index=full.index)   # hores conegudes (preu/CO2 h<3)
        R[key] = pd.DataFrame(mat).reindex(columns=range(24))
    common = None
    for df in R.values():
        ok = df.dropna().index
        common = ok if common is None else common.intersection(ok)
    rng = np.random.default_rng(seed)
    out = []
    for hr in range(24):
        idx = [t for t in common if t.hour == hr]
        pick = rng.choice(len(idx), size=min(per_hour, len(idx)), replace=False) if idx else []
        out.append([{k: np.round(R[k].loc[idx[i]].values, 5).tolist() for k in R} for i in pick])
    return out


LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
LGB_ROUNDS = 400
COLS = {"precio": "precio_eur_per_kwh", "co2": "co2_kg_per_kwh", "load": "load_kw", "pv": "pv_kw",
        "irr": "irradiancia_wm2", "temp": "temp_ambient_c"}


def fit_lgb(df: pd.DataFrame, params: dict, step: int = 3) -> dict:
    """Entrena un LightGBM per variable (preu, CO2, consum, solar) que prediu la desviació respecte
    al perfil típic, amb les features EXACTES de BatteryAgent._feats (importem l'agent)."""
    import importlib.util
    import lightgbm as lgb
    spec = importlib.util.spec_from_file_location("hard_agent", ROOT / "submission_hard" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    agent = mod.BatteryAgent(params={k: v for k, v in params.items() if k != "lgb"})
    for k, col in COLS.items():
        agent.hist[k] = {ts: float(v) for ts, v in df[col].items() if v == v}

    one = pd.Timedelta(hours=1)
    out = {}
    for key, offset, hs in [("precio", 2, range(3, 24)), ("co2", 2, range(3, 24)),
                            ("load", -1, range(0, 24)), ("pv", -1, range(0, 24))]:
        hist = agent.hist[key]
        Xs, ys = [], []
        for t in df.index[::step]:
            if (t + offset * one) not in hist:
                continue
            hs_ok = [h for h in hs if (t + h * one) in hist]
            if not hs_ok:
                continue
            X, py = agent._feats(key, t, offset, hs_ok)
            Xs.append(X)
            ys.append(np.array([hist[t + h * one] for h in hs_ok]) - py)
        X, y = np.vstack(Xs), np.concatenate(ys)
        booster = lgb.train(LGB_PARAMS, lgb.Dataset(X, y), num_boost_round=LGB_ROUNDS)
        out[key] = booster.model_to_string()
    return out


def fit_all(df: pd.DataFrame, verbose: bool = False) -> dict:
    params = fit_params(df, verbose=verbose)
    params["lgb"] = fit_lgb(df, params)
    return params


def main():
    params = fit_all(to_hard(load_hourly()), verbose=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(params), encoding="utf-8")
    print(f"Desat {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
