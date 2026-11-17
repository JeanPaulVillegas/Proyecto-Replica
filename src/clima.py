"""Clima EXTERIOR de Chennai para el gemelo digital.

Prioridad:
  1. data_reales/clima_chennai.csv  (datos horarios reales; se bajan con src/descargar_clima.py)
  2. Respaldo: clima sintético armado con normales climáticas de Chennai (aproximado).

El simulador convierte este clima exterior en clima del cuarto de cultivo (ver simulator.py):
el edificio amortigua y retrasa la oscilación diaria, y la humedad interior sale del vapor de
agua del aire exterior más la transpiración de las plantas.

Fuentes del respaldo:
  - Temperatura media mensual: normales 1991-2020 de la JMA para Chennai/Minambakkam (estación 43279).
  - Máximas y mínimas medias: weather2travel (resumen de normales).
  - Punto de rocío mensual: APROXIMADO (no se encontró una fuente con valores numéricos);
    se reemplaza automáticamente por el dato real cuando existe clima_chennai.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_REAL = ROOT / "data_reales" / "clima_chennai.csv"

#                    ene   feb   mar   abr   may   jun   jul   ago   sep   oct   nov   dic
T_MEDIA_JMA = [25.4, 26.7, 28.7, 31.0, 33.0, 32.3, 31.0, 30.3, 29.8, 28.5, 26.7, 25.6]
T_MAX = [29, 31, 33, 35, 38, 38, 36, 35, 34, 32, 29, 29]
T_MIN = [20, 20, 22, 26, 28, 27, 26, 26, 25, 24, 22, 21]
T_ROCIO_APROX = [19.0, 19.5, 21.5, 24.0, 24.0, 22.0, 22.0, 22.5, 23.0, 23.5, 22.5, 20.5]


def presion_saturacion_kpa(t_c):
    """Ecuación de Magnus-Tetens (kPa)."""
    t_c = np.asarray(t_c, dtype=float)
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def hr_desde_rocio(t_c, td_c):
    return np.clip(100 * presion_saturacion_kpa(td_c) / presion_saturacion_kpa(t_c), 5, 100)


def _sintetico(inicio: pd.Timestamp, horas: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = inicio + pd.to_timedelta(np.arange(horas), unit="h")
    # normales interpoladas día a día (centradas a mitad de mes)
    doy = t.dayofyear.to_numpy() + t.hour.to_numpy() / 24
    mid = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    xp = np.concatenate([mid - 365, mid, mid + 365])

    def interp(v):
        return np.interp(doy, xp, np.tile(v, 3))

    t_mean, t_max, t_min, td = interp(T_MEDIA_JMA), interp(T_MAX), interp(T_MIN), interp(T_ROCIO_APROX)
    # forma diaria asimétrica: mínimo ~06 h, máximo ~14 h
    h = t.hour.to_numpy()
    fase = np.where((h >= 6) & (h < 14), (h - 6) / 8 * np.pi, np.pi + ((h - 14) % 24) / 16 * np.pi)
    forma = -np.cos(fase)                                  # -1 a las 06 h, +1 a las 14 h
    amp = (t_max - t_min) / 2
    # anomalías día a día (AR(1)): frentes, nubosidad, lluvias
    n_dias = horas // 24 + 2
    an_t, an_td = np.zeros(n_dias), np.zeros(n_dias)
    for i in range(1, n_dias):
        an_t[i] = 0.7 * an_t[i - 1] + rng.normal(0, 0.8)
        an_td[i] = 0.7 * an_td[i - 1] + rng.normal(0, 1.0)
    dia = (np.arange(horas) // 24)
    temp = t_mean + amp * forma + an_t[dia] + rng.normal(0, 0.3, horas)
    rocio = np.minimum(td + an_td[dia] + 0.6 * forma + rng.normal(0, 0.3, horas), temp - 0.5)
    return pd.DataFrame(dict(time=t, temperature_2m=temp, dew_point_2m=rocio,
                             relative_humidity_2m=hr_desde_rocio(temp, rocio)))


def cargar_clima_exterior(inicio: str, dias: int, seed: int = 0) -> tuple[pd.DataFrame, str]:
    """DataFrame horario (time, temperature_2m, relative_humidity_2m, dew_point_2m) y su origen."""
    t0 = pd.Timestamp(inicio)
    horas = (dias + 2) * 24
    if CSV_REAL.exists():
        df = pd.read_csv(CSV_REAL, parse_dates=["time"])
        df = df[df.time >= t0].reset_index(drop=True)
        if len(df) >= dias * 24:
            if "dew_point_2m" not in df or df.dew_point_2m.isna().all():
                e = df.relative_humidity_2m / 100 * presion_saturacion_kpa(df.temperature_2m)
                df["dew_point_2m"] = 237.3 * np.log(e / 0.6108) / (17.27 - np.log(e / 0.6108))
            df = df[["time", "temperature_2m", "relative_humidity_2m", "dew_point_2m"]]
            df = df.interpolate(limit_direction="both")
            if len(df) < horas:                      # rellena las 48 h de margen repitiendo el final
                df = pd.concat([df, df.iloc[-(horas - len(df)):]], ignore_index=True)
            return df.iloc[:horas].reset_index(drop=True), f"real ({CSV_REAL.name}, Open-Meteo/ERA5)"
        print(f"[clima] {CSV_REAL.name} no cubre {dias} días desde {inicio}; uso el respaldo climatológico")
    return _sintetico(t0, horas, seed), "sintético (normales climáticas de Chennai, aproximado)"
