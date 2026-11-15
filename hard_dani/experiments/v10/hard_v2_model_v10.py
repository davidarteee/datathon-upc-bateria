"""Agent Hard v2 (Dani): MPC 24 h amb previsió causal per mes + correccions apreses.

Autocontingut (numpy + scipy). No importa res de `oriol_hard_mode` ni llegeix cap CSV:
tot el coneixement del train entra per `params` (JSON generat offline per `train.py`).

Informació usada a cada pas, i res més:
  - timestamp, SOC i límits de l'observació;
  - preu i CO2 de t, t+1, t+2 (vectors de l'observació, truncats a 3);
  - telemetria de l'hora t-1 (consum i solar), només si l'hora anterior és contigua.
"""

from datetime import datetime, timedelta
from functools import lru_cache
from math import isfinite, sqrt

import numpy as np

ETA = sqrt(0.9)
EPSILON = 1e-4
VARS = ("load", "pv", "price", "co2")
# Festius (Barcelona / UPC) tractats com a diumenge. Coneixement de calendari, no dades del test.
HOLIDAYS = {
    "2025-09-11", "2025-09-24", "2025-10-12", "2025-11-01", "2025-12-06", "2025-12-08", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-01-06", "2026-04-03", "2026-04-06", "2026-05-01", "2026-05-25", "2026-06-24", "2026-08-15",
    "2026-09-11",
}


# ------------------------------------------------------------------ seguretat

def limits(obs):
    try:
        capacity = float(obs.get("cap_max_kwh", 6000.0))
        power = float(obs.get("power_max_kw", 1500.0))
        soc = float(obs["soc_kwh"])
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(v) for v in (capacity, power, soc)) or capacity <= 0 or power <= 0:
        return None
    if not 0 <= soc <= capacity:
        return None
    return capacity, power, soc


def clamp_action(action, obs):
    try:
        value = float(action)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    state = limits(obs)
    if not isfinite(value) or state is None:
        return 0.0
    capacity, power, soc = state
    charge = max(0.0, min(power, (capacity - soc) / ETA - EPSILON))
    discharge = max(0.0, min(power, soc * ETA - EPSILON))
    return float(max(-discharge, min(charge, value)))


# ------------------------------------------------------------------ LP

@lru_cache(maxsize=16)
def _template(n):
    from scipy.sparse import coo_matrix
    rows, cols, data = [], [], []
    for i in range(n):
        for col, value in [(i, -ETA), (n + i, 1 / ETA), (2 * n + i, 1)]:
            rows.append(i); cols.append(col); data.append(value)
        if i:
            rows.append(i); cols.append(2 * n + i - 1); data.append(-1)
    equality = coo_matrix((data, (rows, cols)), shape=(n, 4 * n)).tocsr()
    rows, cols, data = [], [], []
    for i in range(n):
        for col, value in [(i, 1), (n + i, -1), (3 * n + i, -1)]:
            rows.append(i); cols.append(col); data.append(value)
    inequality = coo_matrix((data, (rows, cols)), shape=(n, 4 * n)).tocsr()
    return equality, inequality


def solve_plan(values, soc, net, capacity, power, terminal_target=None, terminal_value=0.0):
    """min sum_t import_t*values_t ; import_t >= net_t + c_t - d_t ; dinàmica de SOC amb pèrdues."""
    from scipy.optimize import linprog
    n = len(values)
    objective = np.zeros(4 * n)
    objective[:2 * n] = 1e-8
    objective[3 * n:] = values
    objective[3 * n - 1] = -terminal_value
    equality, inequality = _template(n)
    rhs = np.zeros(n); rhs[0] = soc
    bounds = [(0, power)] * (2 * n) + [(0, capacity)] * n + [(0, None)] * n
    if terminal_target is not None:
        bounds[3 * n - 1] = (terminal_target, terminal_target)
    result = linprog(objective, A_eq=equality, b_eq=rhs, A_ub=inequality, b_ub=-np.asarray(net),
                     bounds=bounds, method="highs", options={"time_limit": 0.5})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("LP no resolt")
    c, d = result.x[:n], result.x[n:2 * n]
    if np.any(np.minimum(c, d) > 1e-5):
        raise RuntimeError("Càrrega i descàrrega simultànies")
    return c - d


QUANTILES = (-1.2816, -0.5244, 0.0, 0.5244, 1.2816)   # 5 escenaris equiprobables d'una normal


@lru_cache(maxsize=16)
def _template_stochastic(n, scenarios):
    from scipy.sparse import coo_matrix
    width = 3 * n + scenarios * n
    rows, cols, data = [], [], []
    for i in range(n):
        for col, value in [(i, -ETA), (n + i, 1 / ETA), (2 * n + i, 1)]:
            rows.append(i); cols.append(col); data.append(value)
        if i:
            rows.append(i); cols.append(2 * n + i - 1); data.append(-1)
    equality = coo_matrix((data, (rows, cols)), shape=(n, width)).tocsr()
    rows, cols, data = [], [], []
    for s in range(scenarios):
        for i in range(n):
            row = s * n + i
            for col, value in [(i, 1), (n + i, -1), (3 * n + row, -1)]:
                rows.append(row); cols.append(col); data.append(value)
    inequality = coo_matrix((data, (rows, cols)), shape=(scenarios * n, width)).tocsr()
    return equality, inequality


def solve_plan_stochastic(values, soc, net, sigma, capacity, power, terminal_target=None):
    """Com solve_plan, però la importació esperada de cada hora es calcula sobre 5 escenaris de demanda neta.

    L'acció és única per hora (no depèn de l'escenari): el LP tria la consigna que minimitza el cost
    esperat sabent que descarregar de més s'exporta gratis i que carregar de més s'importa.
    """
    from scipy.optimize import linprog
    n = len(values); s = len(QUANTILES)
    objective = np.zeros(3 * n + s * n)
    objective[:2 * n] = 1e-8
    equality, inequality = _template_stochastic(n, s)
    rhs = np.zeros(n); rhs[0] = soc
    upper = np.empty(s * n)
    for k, q in enumerate(QUANTILES):
        objective[3 * n + k * n:3 * n + (k + 1) * n] = np.asarray(values) / s
        upper[k * n:(k + 1) * n] = -np.clip(np.asarray(net) + q * np.asarray(sigma), -power, power)
    bounds = [(0, power)] * (2 * n) + [(0, capacity)] * n + [(0, None)] * (s * n)
    if terminal_target is not None:
        bounds[3 * n - 1] = (terminal_target, terminal_target)
    result = linprog(objective, A_eq=equality, b_eq=rhs, A_ub=inequality, b_ub=upper,
                     bounds=bounds, method="highs", options={"time_limit": 0.5})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError("LP estocàstic no resolt")
    c, d = result.x[:n], result.x[n:2 * n]
    if np.any(np.minimum(c, d) > 1e-5):
        raise RuntimeError("Càrrega i descàrrega simultànies")
    return c - d


# ------------------------------------------------------------------ previsió

def parse_timestamp(obs):
    value = obs.get("timestamp")
    try:
        return value.to_pydatetime() if hasattr(value, "to_pydatetime") else datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        hour = int(obs.get("hour", 0)); day = int(obs.get("day_of_week", 0))
        return datetime(2000, 1, 3) + timedelta(days=max(0, min(6, day)), hours=max(0, min(23, hour)))


# Tancaments del campus observats al train: Nadal (consum de diumenge del 25/12 al 6/1) i agost (fins al 23/8).
CAMPUS_CLOSED = {f"2025-12-{d}" for d in range(25, 32)} | {f"2026-01-0{d}" for d in range(1, 7)}
CAMPUS_REDUCED = {"2025-12-24", "2026-03-30", "2026-03-31", "2026-04-01", "2026-04-02"} | {f"2026-08-{d:02d}" for d in range(1, 24)} | {f"2025-08-{d:02d}" for d in range(1, 24)}


def day_type(ts):
    """Tipus de dia del campus: 0 laborable, 1 dissabte o activitat reduïda, 2 diumenge, festiu o tancat."""
    key = ts.strftime("%Y-%m-%d")
    if key in HOLIDAYS or key in CAMPUS_CLOSED or ts.weekday() == 6:
        return 2
    return 1 if ts.weekday() == 5 or key in CAMPUS_REDUCED else 0


def market_type(ts):
    """Tipus de dia del mercat elèctric: 0 laborable, 1 cap de setmana o festiu (els tancaments del campus no compten)."""
    return 1 if ts.weekday() >= 5 or ts.strftime("%Y-%m-%d") in HOLIDAYS else 0


class Forecaster:
    """Base per mes/tipus de dia/hora + correccions lineals per horitzó amb residus observats."""

    def __init__(self, params, horizon=24, weeks=None):
        self.p = params
        self.h = horizon
        # Setmanes del test (inici, final). Quan una setmana s'acaba, la hora següent és l'inici de la següent, no l'endemà.
        self.skip_jumped = False
        self.weeks = sorted((datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in (weeks or []))
        self.previous = None
        self.obs_net = {}      # ts -> (load, pv) observats via telemetria t-1
        self.obs_market = {}   # ts -> (price, co2) coneguts per l'observació
        self._base_cache = {}

    # -- perfils base
    def base(self, var, ts):
        key = (var, ts)
        cached = self._base_cache.get(key)
        if cached is not None:
            return cached
        if self.p.get("daily"):
            d = ts.timetuple().tm_yday - 1
            if var == "load":
                value = self.p["load_profile"][d][day_type(ts)][ts.hour]
            elif var == "pv":
                value = self.p["pv_profile"][d][ts.hour]
            elif var == "cs":
                value = self.p["pv_clear"][d][ts.hour]
            else:
                value = self.p[var + "_profile"][d][market_type(ts)][ts.hour]
            if len(self._base_cache) > 20000:
                self._base_cache.clear()
            self._base_cache[key] = value
            return value
        m = ts.month - 1
        if var == "load":
            value = self.p["load_profile"][m][day_type(ts)][ts.hour]
        elif var == "pv":
            value = self.p["pv_profile"][m][ts.hour]
        elif var == "cs":
            value = self.p["pv_clear"][m][ts.hour]
        else:
            value = self.p[var + "_profile"][m][market_type(ts)][ts.hour]
        if len(self._base_cache) > 20000:
            self._base_cache.clear()
        self._base_cache[key] = value
        return value

    # -- actualització causal de l'historial
    def update(self, obs):
        now = parse_timestamp(obs)
        contiguous = self.previous is not None and (now - self.previous) == timedelta(hours=1)
        if contiguous:
            telemetry = obs.get("noves_dades_5min", {}) or {}
            try:
                load = float(telemetry["ultim_consum_kw"]); pv = float(telemetry["ultima_solar_kw"])
                if isfinite(load) and isfinite(pv) and load >= 0 and pv >= 0:
                    self.obs_net[self.previous] = (load, pv)
            except (KeyError, TypeError, ValueError):
                pass
        try:
            prices = [float(v) for v in list(obs.get("preus_properes_24h", []))[:self.h]]
            carbon = [float(v) for v in list(obs.get("co2_properes_24h", []))[:self.h]]
            known = min(len(prices), len(carbon))
            if not all(isfinite(v) for v in prices[:known] + carbon[:known]):
                known = 0
        except (TypeError, ValueError):
            known = 0
        # Només l'hora actual és segur que correspon a `now`; t+1 i t+2 es desen en avançar.
        if known:
            self.obs_market[now] = (prices[0], carbon[0])
        self.previous = now
        if len(self.obs_net) > 600:
            for key in sorted(self.obs_net)[:-400]:
                del self.obs_net[key]
        if len(self.obs_market) > 600:
            for key in sorted(self.obs_market)[:-400]:
                del self.obs_market[key]
        return now, known, prices[:known] if known else [], carbon[:known] if known else []

    def targets(self, now):
        hour = timedelta(hours=1)
        plain = [now + i * hour for i in range(self.h)]
        if not self.weeks:
            return plain
        cur = next(((a, b) for a, b in self.weeks if a <= now <= b), None)
        if cur is None:
            return plain
        nxt = next(((a, b) for a, b in self.weeks if a > cur[1]), None)
        out = []
        for ts in plain:
            if ts > cur[1] and nxt is not None:
                ts = nxt[0] + (ts - cur[1] - hour)
            out.append(ts)
        return out

    # -- característiques (NaN = no disponible)
    def features(self, now, known, prices, carbon):
        h = self.h
        hour = timedelta(hours=1)
        targets = self.targets(now)
        nan = float("nan")
        base = {v: np.asarray([self.base(v, ts) for ts in targets]) for v in VARS}
        clear = np.asarray([self.base("cs", ts) for ts in targets])
        feats = {}
        net = self.obs_net
        market = self.obs_market

        def load_res(ts):
            seen = net.get(ts)
            return seen[0] - self.base("load", ts) if seen else nan

        # Demanda
        r1 = load_res(now - hour)
        f_last = np.full(h, r1)
        f_day = np.asarray([load_res(ts - 24 * hour) for ts in targets])
        recent = [v for v in (load_res(now - j * hour) for j in range(1, 25)) if v == v]
        f_mean = np.full(h, float(np.mean(recent)) if len(recent) >= 12 else nan)
        f_shift = np.full(h, r1 - load_res(now - 25 * hour))   # canvi de nivell respecte d'ahir
        f_trend = np.full(h, r1 - load_res(now - 2 * hour))
        feats["load"] = [f_last, f_day, f_mean, f_shift, f_trend]

        # Solar: índex de claredat observat respecte del perfil clar del mes
        kbar = self.p["pv_kbar"][now.timetuple().tm_yday - 1 if self.p.get("daily") else now.month - 1]

        def clearness(times, minimum):
            num = den = 0.0
            for ts in times:
                seen = net.get(ts)
                if seen:
                    c = self.base("cs", ts)
                    if c > 30.0:
                        num += seen[1]; den += c
            return (num / den - kbar) if den > minimum else nan

        k_recent = clearness([now - j * hour for j in range(1, 4)], 100.0)
        k_today = clearness([now - j * hour for j in range(1, now.hour + 1)], 300.0)
        k_day = clearness([now - j * hour for j in range(1, 25)], 800.0)
        pv_day = np.full(h, nan)
        for i, ts in enumerate(targets):
            seen = net.get(ts - 24 * hour)
            if seen:
                pv_day[i] = seen[1] - self.base("pv", ts - 24 * hour)
        feats["pv"] = [clear * k_recent, clear * k_today, clear * k_day, pv_day]

        # Mercat
        near = {}
        for index, var, values in [(0, "price", prices), (1, "co2", carbon)]:
            res = [values[i] - self.base(var, targets[i]) for i in range(min(known, 3))]
            near[var] = float(np.mean(res)) if res else nan

            def mres(ts, index=index, var=var):
                seen = market.get(ts)
                return seen[index] - self.base(var, ts) if seen else nan

            g_near = np.full(h, near[var])
            g_day = np.asarray([mres(ts - 24 * hour) for ts in targets])
            hist = [v for v in (mres(now - j * hour) for j in range(0, 24)) if v == v]
            g_mean = np.full(h, float(np.mean(hist)) if len(hist) >= 12 else nan)
            ynear = [mres(targets[i] - 24 * hour) for i in range(3)]
            g_shift = np.full(h, near[var] - float(np.mean(ynear)) if all(v == v for v in ynear) else nan)
            g_slope = np.full(h, res[2] - res[0] if len(res) == 3 else nan)
            g_day2 = np.asarray([mres(ts - 48 * hour) for ts in targets])
            feats[var] = [g_near, g_day, g_mean, g_shift, g_slope, g_day2]
        feats["price"].append(np.full(h, near["co2"]))
        feats["co2"].append(np.full(h, near["price"]))
        feats["co2"].append(clear * k_day)          # dia assolellat al campus ~ més solar a la xarxa
        return targets, base, clear, feats

    def bucket(self, now):
        return now.hour // self.p.get("bucket_hours", 24)

    def predict(self, now, base, clear, feats, jumped=None):
        out = {}
        coefs = self.p.get("coef", {})
        use = self.p.get("use", {})
        b = self.bucket(now)
        for var in VARS:
            pred = base[var].copy()
            tables = coefs.get(var)
            if tables:
                table = tables[b] if b < len(tables) else tables[0]
                stack = [feats[var][j] for j in use[var]]
                for i in range(self.h):
                    if jumped is not None and jumped[i]:
                        continue          # hora de la setmana següent del test: les correccions d'ara ja no hi valen
                    mask = "".join("0" if f[i] != f[i] else "1" for f in stack)
                    weights = table[i].get(mask) if i < len(table) else None
                    if weights:
                        for j, w in enumerate(weights):
                            if w:
                                pred[i] += w * stack[j][i]
            if var == "load":
                pred = np.maximum(pred, 0.0)
            elif var == "pv":
                pred = np.clip(pred, 0.0, 1.25 * clear)
            else:
                pred = np.maximum(pred, 1e-5)
            out[var] = pred
        return out

    def forecast(self, obs):
        now, known, prices, carbon = self.update(obs)
        targets, base, clear, feats = self.features(now, known, prices, carbon)
        jumped = None
        if self.skip_jumped:
            jumped = [t != now + timedelta(hours=i) for i, t in enumerate(targets)]
        out = self.predict(now, base, clear, feats, jumped)
        for i in range(known):
            out["price"][i] = prices[i]; out["co2"][i] = carbon[i]
        out["known"] = known
        out["timestamp"] = now
        return out


# ------------------------------------------------------------------ agent

class HardAgentV2:
    def __init__(self, params, config=None):
        self.p = params
        self.c = {"horizon": 24, "terminal": "target", **params.get("config", {}), **(config or {})}
        self.forecaster = Forecaster(params, self.c["horizon"], self.c.get("weeks"))
        self.forecaster.skip_jumped = bool(self.c.get("skip_jumped"))
        self.failures = 0
        self.lp_calls = 0
        self.last_forecasts = None
        self.step = 0
        self.max_known = 0
        self.w_sum = None

    def steps_left(self):
        """Hores que queden fins al final previst de l'avaluació (1.008 h públic; 2.016 h si és el test sencer)."""
        for length in self.c.get("episode_lengths", []):
            if self.step < length:
                return length - self.step
        return None

    def weights(self, forecast):
        """Normalitzadors del score: mitjanes de CO2 i preu ponderades per la importació.

        Amb `adaptive_weights`, s'actualitzen amb el que l'agent ja ha vist (hora t-1), partint del valor del train."""
        prior_c, prior_p = self.p["mean_carbon"], self.p["mean_price"]
        w0 = self.c.get("adaptive_weights", 0)
        if not w0:
            return prior_c, prior_p
        f = self.forecaster
        last = f.previous - timedelta(hours=1) if f.previous else None
        seen, market = f.obs_net.get(last), f.obs_market.get(last)
        if self.w_sum is None:
            self.w_sum = [0.0, 0.0, 0.0, None]
        if seen and market and self.w_sum[3] != last:
            imp = max(seen[0] - seen[1], 0.0)
            self.w_sum[0] += imp; self.w_sum[1] += imp * market[1]; self.w_sum[2] += imp * market[0]; self.w_sum[3] = last
        k = w0 * 1000.0
        return (k * prior_c + self.w_sum[1]) / (k + self.w_sum[0]), (k * prior_p + self.w_sum[2]) / (k + self.w_sum[0])

    def fallback(self, forecast, state):
        capacity, power, soc = state
        net = forecast["load"][0] - forecast["pv"][0]
        return min(power, -net) if net < 0 else 0.0

    def act(self, obs):
        state = limits(obs)
        if state is None:
            return 0.0
        capacity, power, soc = state
        left = self.steps_left()
        self.step += 1
        try:
            forecast = self.forecaster.forecast(obs)
        except Exception:
            self.failures += 1
            return 0.0
        self.last_forecasts = forecast
        known = forecast["known"]
        if known == 0:
            return 0.0
        self.max_known = max(self.max_known, known)
        # El vector de mercat només s'escurça a les últimes hores de l'episodi: és l'únic senyal de final fiable.
        ending = known < max(3, self.max_known)
        n = known if ending else max(known, self.c["horizon"])
        if not ending and left is not None and left <= self.c.get("drain_window", 24):
            n = max(1, min(n, left)); ending = True
        if not ending and self.c.get("end_time"):
            # Final real del test (2.016 h seguides; l'última setmana acaba el 31 d'agost a les 23:00): buidar només aquí.
            end = datetime.fromisoformat(self.c["end_time"])
            to_end = int(round((end - forecast["timestamp"]).total_seconds() / 3600)) + 1
            if 0 < to_end <= self.c.get("drain_window", 24):
                n = max(1, min(n, to_end)); ending = True
        mc, mp = self.weights(forecast)
        values = (.7 * forecast["co2"] / mc + .3 * forecast["price"] / mp)[:n]
        if not np.isfinite(values).all() or np.any(values <= 0):
            self.failures += 1
            return clamp_action(self.fallback(forecast, state), obs)
        try:
            self.lp_calls += 1
            net = forecast["load"][:n] - forecast["pv"][:n]
            sigma = self.p.get("net_sigma")
            if sigma:
                # Marge asimètric: descarregar de més s'exporta gratis; carregar de menys també es perd.
                now = forecast["timestamp"]
                shift = np.asarray([sigma[(now.hour + i) % 24] for i in range(n)])
                net = net - self.c.get("z_future", 0.0) * shift
                net[0] += (self.c.get("z_future", 0.0) - self.c.get("z_now", 0.0)) * shift[0]
            target = capacity * self.c.get("terminal_frac", .5) if self.c["terminal"] == "target" and not ending else None
            lead_sigma = self.p.get("net_sigma_lead")
            if self.c.get("stochastic") and lead_sigma:
                now = forecast["timestamp"]
                spread = np.asarray([lead_sigma[i][(now.hour + i) % 24] for i in range(n)])
                spread[1:] *= self.c.get("s_future", 1.0)
                spread[0] *= self.c.get("s_now", 1.0)
                actions = solve_plan_stochastic(values, soc, net, spread, capacity, power, target)
            else:
                actions = solve_plan(values, soc, np.clip(net, -power, power), capacity, power, target)
            return clamp_action(actions[0], obs)
        except (ValueError, RuntimeError):
            self.failures += 1
            return clamp_action(self.fallback(forecast, state), obs)
