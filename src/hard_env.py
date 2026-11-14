"""
src/hard_env.py
Simulador local del MODE DIFÍCIL, seguint les indicacions de la web oficial:
  - Bateria 6.000 kWh, ±1.500 kW, rendiment 90 %, SOC inicial 50 %.
  - Solar ×12 (pv_kw * 12) -> hi ha excedents a migdia (exportar val 0 € i 0 kg).
  - obs["preus_properes_24h"] i obs["co2_properes_24h"] només tenen 3 hores (actual + 2).
  - obs["noves_dades_5min"] inclou la meteo de la darrera hora.
  - obs inclou "cap_max_kwh" i "power_max_kw".
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))
from environment import CampusBatteryEnv  # noqa: E402

SPEC_HARD = {
    "capacity_kwh": 6000.0, "power_kw": 1500.0, "efficiency": 0.90, "soc_init": 0.50,
    "soc_min": 0.0, "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": 3,
}
PV_SCALE = 12.0
METEO = ["irradiancia_wm2", "temp_ambient_c", "temp_panel_c", "vent_ms"]


def to_hard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pv_kw"] = df["pv_kw"] * PV_SCALE
    return df


class HardBatteryEnv(CampusBatteryEnv):
    def __init__(self, hourly_data: pd.DataFrame, spec: dict = None):
        spec = spec or SPEC_HARD
        self.horizon = int(spec.get("forecast_horizon", 3))
        super().__init__(hourly_data, spec=spec)

    def _get_obs(self):
        obs = super()._get_obs()
        if obs is None:
            return None
        obs["preus_properes_24h"] = obs["preus_properes_24h"][: self.horizon]
        obs["co2_properes_24h"] = obs["co2_properes_24h"][: self.horizon]
        obs["cap_max_kwh"] = self.cap
        obs["power_max_kw"] = self.power
        i = self.t - 1 if self.t > 0 else self.t
        for c in METEO:
            if c in self.df.columns:
                obs["noves_dades_5min"][c] = float(self.df[c].iloc[i])
        return obs
