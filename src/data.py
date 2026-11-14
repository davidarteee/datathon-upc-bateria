"""
src/data.py
Construeix la taula horària del campus a partir de les dades de train.

Ús:
    python src/data.py            # extreu el zip (si cal) i genera data/processed/hourly.csv

Columnes de sortida (les que espera starter_kit/environment.py):
    Fecha, load_kw, pv_kw, precio_eur_per_kwh, co2_kg_per_kwh
"""

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ZIP = ROOT / "data" / "raw" / "paquet_participants_datathon_upc.zip"
TRAIN_DIR = ROOT / "data" / "train"
PROCESSED = ROOT / "data" / "processed" / "hourly.csv"


def ensure_train_data() -> Path:
    """Extreu data/train/*.csv del zip oficial si encara no hi són."""
    if TRAIN_DIR.exists() and any(TRAIN_DIR.glob("*.csv")):
        return TRAIN_DIR
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP) as z:
        for name in z.namelist():
            if "data/train/" in name.replace("\\", "/") and name.endswith(".csv"):
                (TRAIN_DIR / Path(name).name).write_bytes(z.read(name))
    return TRAIN_DIR


def build_hourly(train_dir: Path = None) -> pd.DataFrame:
    """Mateix preprocessament que el tutorial oficial (per ser coherents amb l'avaluador)."""
    d = train_dir or ensure_train_data()

    cont = pd.read_csv(d / "contador.csv", parse_dates=["Fecha"], low_memory=False).set_index("Fecha")
    load = (-cont["Contador - POTENCIA ACTIVA EXPORTADA (kW)"]).clip(lower=0, upper=5000)
    load_h = load.resample("1h").mean()

    inv = pd.read_csv(d / "inversor.csv", parse_dates=["Fecha"], low_memory=False).set_index("Fecha")
    cols = [c for c in inv.columns
            if "POTENCIA ACTIVA" in c.upper() and "FASE" not in c.upper() and c.upper().startswith("INVERSOR")]
    pv = inv[cols].fillna(0).sum(axis=1).clip(lower=0)
    pv_h = pv.resample("1h").mean()

    pvpc = pd.read_csv(d / "precio_pvpc.csv", parse_dates=["Fecha"]).set_index("Fecha")
    co2 = pd.read_csv(d / "co2_red.csv", parse_dates=["Fecha"]).set_index("Fecha")

    df = pd.DataFrame({
        "load_kw": load_h,
        "pv_kw": pv_h,
        "precio_eur_per_kwh": pvpc["precio_eur_per_kwh"],
        "co2_kg_per_kwh": co2["co2_kg_per_kwh"],
    }).dropna()

    # Meteo (Mode Difícil): s'afegeix sense eliminar hores (buits -> interpolació / 0 de nit)
    met = pd.read_csv(d / "meteo.csv", parse_dates=["Fecha"], low_memory=False).set_index("Fecha")
    met_cols = {
        "irradiancia_wm2": "Atersa - IRRADIANCIA (W/m2)",
        "temp_ambient_c": "Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)",
        "temp_panel_c": "Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)",
        "vent_ms": "Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)",
    }
    met_h = pd.DataFrame({k: pd.to_numeric(met[v], errors="coerce") for k, v in met_cols.items()}).resample("1h").mean()
    met_h["irradiancia_wm2"] = met_h["irradiancia_wm2"].clip(lower=0)
    df = df.join(met_h, how="left")
    df[list(met_cols)] = df[list(met_cols)].interpolate(limit_direction="both")
    df.index.name = "Fecha"
    return df


def load_hourly(rebuild: bool = False) -> pd.DataFrame:
    if PROCESSED.exists() and not rebuild:
        return pd.read_csv(PROCESSED, parse_dates=["Fecha"], index_col="Fecha")
    df = build_hourly()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED)
    return df


def split_weeks(df: pd.DataFrame, min_hours: int = 72):
    """Divideix la taula en blocs contigus (un per bloc de train de cada mes).
    Serveix per simular l'avaluació oficial, que reinicia la bateria al 50% a cada període."""
    gap = df.index.to_series().diff() > pd.Timedelta(hours=1)
    block_id = gap.cumsum()
    return [g for _, g in df.groupby(block_id) if len(g) >= min_hours]


if __name__ == "__main__":
    df = load_hourly(rebuild=True)
    print(df.describe().round(3))
    print(f"\n{len(df)} hores, de {df.index.min()} a {df.index.max()}")
    print(f"Blocs contigus: {[len(b) for b in split_weeks(df)]}")
