"""
agent.py — La MasIA · Repte UPC Sostenible · Datathon FME 2026

Política: MPC (Model Predictive Control) amb programació lineal.
A cada hora resolem un LP sobre l'horitzó conegut (preus i CO2 de les properes 24 h)
i apliquem només la primera decisió.

Idea clau: el campus sempre importa > 500 kW de la xarxa, així que l'efecte de la bateria
sobre la xarxa és exactament P_t. L'objectiu oficial es pot escriure com
    min  sum_t  w_t · P_t,      w_t = 0,7·co2_t/CO2_base + 0,3·preu_t/Cost_base
                                    ∝ co2_t + K·preu_t,   K = (0,3/0,7)·(CO2_base/Cost_base)
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

CAP = 2000.0
PMAX = 500.0
ETA = 0.9 ** 0.5          # rendiment exacte de càrrega/descàrrega (no 0.9487!)
SAFE = 1e-6               # marge per no fer mai infraccions per arrodoniment


class BatteryAgent:
    def __init__(self):
        # Rati CO2_base / Cost_base estimat al train (kg/€) -> pes relatiu del preu
        self.K = (0.3 / 0.7) * 0.5873
        # Valor terminal de l'energia que queda a l'acabar l'horitzó (fracció d'un quantil de w)
        self.term_q = 0.5
        self.term_alpha = 1.0
        # Final de l'episodi: el test són els últims 7 dies de cada mes -> s'acaba a final de mes.
        # (src/evaluate.py el fixa al final de cada setmana local per simular-ho.)
        self.episode_end = None
        self._mats = {}

    # ---------- LP ----------
    def _matrices(self, H):
        if H not in self._mats:
            L = np.tril(np.ones((H, H)))
            A_soc = np.hstack([ETA * L, -L / ETA])
            self._mats[H] = np.vstack([A_soc, -A_soc])
        return self._mats[H]

    def _plan(self, soc, w, v_term, dmax):
        H = len(w)
        scale = np.mean(w)
        w = w / scale
        v = v_term / scale
        # min sum w(c-d) - v·soc_H   amb soc_H = soc + sum(ETA c - d/ETA)
        cost = np.concatenate([w - v * ETA, -w + v / ETA])
        A = self._matrices(H)
        b = np.concatenate([np.full(H, CAP - soc), np.full(H, soc)])
        bounds = [(0.0, PMAX)] * H + [(0.0, dmax)] * H
        r = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if r.status != 0:
            return 0.0
        return float(r.x[0] - r.x[H])

    # ---------- política ----------
    def act(self, obs: dict) -> float:
        soc = float(obs["soc_kwh"])
        prices = np.asarray(obs["preus_properes_24h"], dtype=float)
        co2 = np.asarray(obs["co2_properes_24h"], dtype=float)
        H = min(len(prices), len(co2))
        if H == 0:
            prices = np.array([obs["preu_actual"]])
            co2 = np.array([obs["co2_actual"]])
            H = 1
        # No planifiquem més enllà del final de l'episodi (final de mes): l'energia que hi quedi no puntua.
        try:
            t = pd.Timestamp(obs["timestamp"]).floor("h")
            end = self.episode_end if self.episode_end is not None else (t + pd.offsets.MonthBegin(1)).normalize()
            H = max(1, min(H, int((end - t) / pd.Timedelta(hours=1))))
        except Exception:
            pass
        w = co2[:H] + self.K * prices[:H]

        # Si l'horitzó és curt, s'acaba el període d'avaluació: l'energia restant no val res.
        v_term = 0.0 if H < 24 else self.term_alpha * np.quantile(w, self.term_q) * ETA

        # No té sentit descarregar més que el consum net (l'exportació no es paga).
        d5 = obs.get("noves_dades_5min", {}) or {}
        net = float(d5.get("ultim_consum_kw", 1e4)) - float(d5.get("ultima_solar_kw", 0.0))
        dmax = float(np.clip(net, 0.0, PMAX))

        p = self._plan(soc, w, v_term, dmax)
        return self._safe(p, soc)

    @staticmethod
    def _safe(p, soc):
        max_c = max(0.0, min(PMAX, (CAP - soc) / ETA) * (1 - SAFE) - SAFE)
        max_d = max(0.0, min(PMAX, soc * ETA) * (1 - SAFE) - SAFE)
        p = float(np.clip(p, -max_d, max_c))
        if abs(p) < 1e-3:
            p = 0.0
        return p
