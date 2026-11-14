"""
agent.py — La MasIA · Repte UPC Sostenible · Datathon FME 2026 · MODE DIFÍCIL

Política: MPC (planificació de 24 h amb programació lineal) + prediccions.

A cada hora:
  1. Guardem el que hem observat (preu, CO2, consum, solar, meteo) a la memòria.
  2. Predim les properes 24 h:
       - preu i CO2: les 3 primeres hores són reals (obs), de la 4a a la 24a les predim;
       - consum i solar: les predim totes (només sabem l'hora anterior).
     Models: regressions lineals per horitzó entrenades al train (params.json).
  3. Resolem un LP que minimitza  sum_t w_t · importació_t  amb
       importació_t >= consum_t - solar_t + P_t ,  importació_t >= 0
     (així el model sap que l'excedent solar és gratis i que descarregar més del consum es perd).
       w_t = co2_t + K · preu_t      (pes oficial 70 % CO2 / 30 % cost)
  4. Apliquem només la decisió d'aquesta hora, amb un marge de seguretat i clamping físic.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack

try:  # LightGBM és a la llista de llibreries permeses; si faltés, l'agent usa les regressions lineals
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

ETA = 0.9 ** 0.5
H = 24


class BatteryAgent:
    def __init__(self, params=None):
        p = params if params is not None else json.loads(
            (Path(__file__).parent / "params.json").read_text(encoding="utf-8"))
        self.prof = {k: np.asarray(v) for k, v in p["profiles"].items()}
        self.coef = {k: {int(h): np.asarray(c) for h, c in p[f"coef_{k}"].items()}
                     for k in ("precio", "co2", "load", "pv")}
        # Paràmetres de la política (ajustats al train amb src/evaluate.py --hard)
        self.H = 24                   # hores de planificació
        self.K = 0.2617               # (0,3/0,7)·CO2_base/Cost_base al train del Mode Difícil
        self.term_alpha = 0.0         # valor de l'energia que queda al final de l'horitzó
        self.discharge_margin = 25.0  # kW de marge fix en descarregar (error de predicció del consum)
        self.margin_pv = 0.4          # marge extra proporcional a la solar prevista (més incerta)
        # Final de l'episodi: el test són els últims 7 dies de cada mes -> s'acaba a final de mes.
        # (src/evaluate.py el fixa al final de cada setmana local per simular-ho.)
        self.episode_end = None
        self.hist = {k: {} for k in ("precio", "co2", "load", "pv", "irr", "temp")}
        # Història precarregada (últimes 72 h de cada bloc de train, just abans de cada setmana de test)
        for k, d in (p.get("hist") or {}).items():
            if k in self.hist:
                self.hist[k].update({pd.Timestamp(ts): float(v) for ts, v in d.items()})
        self._mats = {}
        self.lgb = {}
        self.lgb_keys = ("precio", "co2")   # variables on fem servir LightGBM (validat amb src/cv_hard.py)
        self.lgb_w = 0.5                     # pes de LightGBM en la mitjana amb la regressió lineal
        if lgb is not None:
            for k, s in (p.get("lgb") or {}).items():
                self.lgb[k] = lgb.Booster(model_str=s)

    # ------------------------------------------------------------------ memòria
    def _remember(self, obs, t):
        prices, co2 = obs.get("preus_properes_24h") or [], obs.get("co2_properes_24h") or []
        self.hist["precio"][t] = float(obs["preu_actual"])
        self.hist["co2"][t] = float(obs["co2_actual"])
        for i, v in enumerate(prices):
            self.hist["precio"][t + pd.Timedelta(hours=i)] = float(v)
        for i, v in enumerate(co2):
            self.hist["co2"][t + pd.Timedelta(hours=i)] = float(v)
        d = obs.get("noves_dades_5min") or {}
        tp = t - pd.Timedelta(hours=1)
        if "ultim_consum_kw" in d:
            self.hist["load"][tp] = float(d["ultim_consum_kw"])
        if "ultima_solar_kw" in d:
            self.hist["pv"][tp] = float(d["ultima_solar_kw"])
        if "irradiancia_wm2" in d and d["irradiancia_wm2"] is not None:
            self.hist["irr"][tp] = max(0.0, float(d["irradiancia_wm2"]))
        if "temp_ambient_c" in d and d["temp_ambient_c"] is not None:
            self.hist["temp"][tp] = float(d["temp_ambient_c"])

    # ------------------------------------------------------------------ prediccions
    def _p(self, key, ts):
        return float(self.prof[key][ts.month - 1, int(ts.dayofweek >= 5), ts.hour])

    def _val(self, key, ts):
        return self.hist[key].get(ts)

    def _dev(self, key, ts):
        v = self.hist[key].get(ts)
        return None if v is None else v - self._p(key, ts)

    def _feats(self, key, t, offset, hs):
        """Matriu de features per al model LightGBM (una fila per hora vista endavant h).
        La fa servir TAMBÉ src/train_forecast.py per entrenar -> mateixes features garantides.
        Retorna (X, py): py = perfil típic, el model prediu la desviació respecte a py."""
        hist = self.hist[key]
        one = pd.Timedelta(hours=1)
        t_last = t + offset * one
        devs = [self._dev(key, t_last - k * one) for k in range(24)]
        d0, d1, d2 = [(devs[k] if devs[k] is not None else np.nan) for k in range(3)]
        dv = [d for d in devs if d is not None]
        dm = float(np.mean(dv)) if dv else np.nan
        irr = self.hist["irr"].get(t - one, np.nan)
        irr_p = self._p("irr", t - one)
        clear = min(2.0, max(0.0, (irr + 1) / (irr_p + 1))) if irr == irr else np.nan
        temp = self.hist["temp"].get(t - one, np.nan)
        rows, pys = [], []
        for h in hs:
            ts = t + h * one
            py = self._p(key, ts)
            y1 = hist.get(ts - 24 * one, np.nan)
            y2 = hist.get(ts - 48 * one, np.nan)
            y7 = hist.get(ts - 168 * one, np.nan)
            rows.append([h, ts.hour, ts.dayofweek, ts.month, py, y1 - py, y2 - py, y7 - py,
                         d0, d1, d2, dm, t.hour, clear, irr, temp])
            pys.append(py)
        return np.asarray(rows, dtype=float), np.asarray(pys)

    def _forecast(self, key, t, offset, n_known):
        """Retorna H valors (t..t+H-1). Mitjana ponderada LightGBM / regressió lineal."""
        lin = self._forecast_lin(key, t, offset, n_known)
        if key in self.lgb and key in self.lgb_keys and self.lgb_w > 0:
            return self.lgb_w * self._forecast_lgb(key, t, offset, n_known) + (1 - self.lgb_w) * lin
        return lin

    def _forecast_lgb(self, key, t, offset, n_known):
        hist = self.hist[key]
        one = pd.Timedelta(hours=1)
        out = np.empty(self.H)
        todo = []
        for h in range(self.H):
            ts = t + h * one
            if h < n_known and ts in hist:
                out[h] = hist[ts]
            else:
                todo.append(h)
        if todo:
            X, py = self._feats(key, t, offset, todo)
            X[:, 0] = np.minimum(X[:, 0], 23)
            out[todo] = py + self.lgb[key].predict(X)
        return np.maximum(out, 0.0)

    def _forecast_lin(self, key, t, offset, n_known):
        hist = self.hist[key]
        one = pd.Timedelta(hours=1)
        t_last = t + offset * one
        d0 = self._dev(key, t_last) or 0.0
        d1 = self._dev(key, t_last - one) or 0.0
        devs = [self._dev(key, t_last - k * one) for k in range(24)]
        devs = [d for d in devs if d is not None]
        dm = float(np.mean(devs)) if devs else 0.0
        clear = 1.0
        if key == "pv":
            irr = self.hist["irr"].get(t_last)
            if irr is not None:
                clear = min(2.0, max(0.0, (irr + 1) / (self._p("irr", t_last) + 1)))
        out = np.empty(self.H)
        for h in range(self.H):
            ts = t + h * one
            if h < n_known and ts in hist:
                out[h] = hist[ts]
                continue
            py = self._p(key, ts)
            c = self.coef[key].get(h)
            if c is None:
                out[h] = py
                continue
            y1 = hist.get(ts - 24 * one, py)
            y2 = hist.get(ts - 48 * one, py)
            x = [py, y1, y2, d0, d1, dm] + ([clear * py] if key == "pv" else [])
            out[h] = c[0] + float(np.dot(c[1:], x))
        return np.maximum(out, 0.0)

    # ------------------------------------------------------------------ LP
    def _matrices(self, n):
        if n not in self._mats:
            L = csr_matrix(np.tril(np.ones((n, n))))
            Z = csr_matrix((n, n))
            I = identity(n, format="csr")
            A_soc = hstack([ETA * L, -L / ETA, Z])
            A_bal = hstack([I, -I, -I])
            self._mats[n] = vstack([A_soc, -A_soc, A_bal]).tocsr()
        return self._mats[n]

    def _plan(self, soc, net, w, cap, pmax, v_term):
        n = len(w)
        scale = float(np.mean(w)) if np.mean(w) > 0 else 1.0
        w = w / scale
        v = v_term / scale
        A = self._matrices(n)
        b = np.concatenate([np.full(n, cap - soc), np.full(n, soc), -net])
        cost = np.concatenate([np.full(n, -v * ETA), np.full(n, v / ETA), w])
        bounds = [(0, pmax)] * (2 * n) + [(0, None)] * n
        r = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if r.status != 0:
            return 0.0
        return float(r.x[0] - r.x[n])

    # ------------------------------------------------------------------ política
    def act(self, obs: dict) -> float:
        cap = float(obs.get("cap_max_kwh", 6000.0))
        pmax = float(obs.get("power_max_kw", 1500.0))
        soc = float(obs["soc_kwh"])
        try:
            t = pd.Timestamp(obs["timestamp"]).floor("h")
        except Exception:
            t = pd.Timestamp("2026-01-01") + pd.Timedelta(hours=len(self.hist["precio"]))
        self._remember(obs, t)

        n_known = max(1, len(obs.get("preus_properes_24h") or []))
        price = self._forecast("precio", t, n_known - 1, n_known)
        co2 = self._forecast("co2", t, n_known - 1, n_known)
        load = self._forecast("load", t, -1, 0)
        pv = self._forecast("pv", t, -1, 0)

        # Horitzó: 24 h, però sense passar del final de mes (el test són els últims 7 dies de cada
        # mes) ni del final de dades (si la llista de preus ve curta, s'acaba l'episodi).
        n = self.H
        end = self.episode_end if self.episode_end is not None else (t + pd.offsets.MonthBegin(1)).normalize()
        n = max(1, min(n, int((end - t) / pd.Timedelta(hours=1))))
        if n_known < 3:
            n = min(n, n_known)
        w = co2[:n] + self.K * price[:n]
        net = load[:n] - pv[:n]
        v_term = 0.0 if n < self.H else self.term_alpha * float(np.mean(w)) * ETA

        p = self._plan(soc, net, w, cap, pmax, v_term)

        # Marge: no descarreguem més que el consum net previst menys un marge (l'excés es perdria).
        if p < 0:
            margin = self.discharge_margin + self.margin_pv * pv[0]
            p = -min(-p, max(0.0, net[0] - margin))
        return self._safe(p, soc, cap, pmax)

    @staticmethod
    def _safe(p, soc, cap, pmax):
        eps = 1e-4
        max_c = max(0.0, min(pmax, (cap - soc) / ETA - eps))
        max_d = max(0.0, min(pmax, soc * ETA - eps))
        p = float(np.clip(p, -max_d, max_c))
        return 0.0 if abs(p) < 1e-3 else p
