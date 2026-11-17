"""Pipeline de preprocesamiento del artículo (secciones 3.1-3.2) y armado de ventanas 60 × 6.

Orden: Ec. 1 (pH) → Ec. 2 (EC a 25 °C) → Ec. 3 (media móvil N=10, en el ESP32) →
Ec. 5 (outliers IQR, reemplazo por último valor válido) → interpolación lineal →
Ec. 4 (z-score, ajustado solo con entrenamiento) → ventanas deslizantes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["temp_air", "rh", "ph", "ec", "nitrogen", "lux"]   # las 6 del artículo
TARGETS = ["ec", "ph"]


# --- Ec. 1: calibración lineal de pH con buffers 4,00 y 7,00 -----------------------------
def calibrate_ph(v_buffer4: float, v_buffer7: float) -> tuple[float, float]:
    a = (7.0 - 4.0) / (v_buffer7 - v_buffer4)
    b = 4.0 - a * v_buffer4
    return a, b


def ph_from_voltage(v, a, b):
    return a * np.asarray(v) + b


def _simulated_buffer_calibration(seed: int = 123) -> tuple[float, float]:
    """Calibración simulada: la sonda (ver simulator.Params) leída en buffers 4 y 7 con ruido."""
    rng = np.random.default_rng(seed)
    slope, offset = -5.70, 21.34
    v4 = np.mean((4.0 - offset) / slope + rng.normal(0, 0.003, 30))
    v7 = np.mean((7.0 - offset) / slope + rng.normal(0, 0.003, 30))
    return calibrate_ph(v4, v7)


PH_CAL = _simulated_buffer_calibration()


# --- Ec. 2: compensación de EC por temperatura -------------------------------------------
def ec_compensate(ec_measured, t_water, alpha: float = 0.019):
    return np.asarray(ec_measured) / (1 + alpha * (np.asarray(t_water) - 25))


# --- Ec. 3: media móvil ----------------------------------------------------------------
def moving_average(df: pd.DataFrame, cols, n: int = 10) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].rolling(n, min_periods=1).mean()
    return out


# --- Ec. 5: outliers por IQR --------------------------------------------------------------
def iqr_bounds(df: pd.DataFrame, cols, k: float = 1.5) -> dict:
    b = {}
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        b[c] = (q1 - k * iqr, q3 + k * iqr)
    return b


def remove_outliers(df: pd.DataFrame, bounds: dict) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    n_out = {}
    for c, (lo, hi) in bounds.items():
        mask = (out[c] < lo) | (out[c] > hi)
        n_out[c] = int(mask.sum())
        out.loc[mask, c] = np.nan
        out[c] = out[c].ffill()          # último valor válido
    return out, pd.Series(n_out, name="outliers")


# --- Ec. 4: z-score ------------------------------------------------------------------------
class ZScore:
    def fit(self, df: pd.DataFrame, cols):
        self.cols = list(cols)
        self.mu = df[self.cols].mean()
        self.sd = df[self.cols].std().replace(0, 1)
        return self

    def transform(self, df):
        out = df.copy()
        out[self.cols] = (out[self.cols] - self.mu) / self.sd
        return out

    def inverse(self, arr: np.ndarray, cols) -> np.ndarray:
        return arr * self.sd[cols].values + self.mu[cols].values


# --- Pipeline completo -----------------------------------------------------------------------
def to_engineering_units(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["ph"] = ph_from_voltage(df["ph_v"], *PH_CAL)
    df["ec"] = ec_compensate(df["ec_raw"], df["t_water"])
    return df


def preprocess(raw: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Devuelve (df limpio y normalizado, df limpio sin normalizar, scaler, informe, índices de corte)."""
    df = to_engineering_units(raw)
    cols = FEATURES + ["t_water", "co2"]
    df = moving_average(df, cols, n=10)                    # Ec. 3 (en el ESP32)
    n = len(df)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
    bounds = iqr_bounds(df.iloc[:i_tr], FEATURES)          # límites solo con train
    df, n_out = remove_outliers(df, bounds)                # Ec. 5
    n_gaps = int(df[FEATURES].isna().any(axis=1).sum())
    df[cols] = df[cols].interpolate(method="linear", limit_direction="both")
    clean = df.copy()
    scaler = ZScore().fit(df.iloc[:i_tr], FEATURES)        # Ec. 4 (sin fuga)
    df = scaler.transform(df)
    report = dict(rows=n, gap_rows_interpolated=n_gaps, outliers=n_out.to_dict(), split=(i_tr, i_va))
    return df, clean, scaler, report


def make_windows(df_norm: pd.DataFrame, clean: pd.DataFrame, window: int = 60, horizon: int = 15,
                 stride: int = 1, target: str = "forecast"):
    """Ventanas X de forma (n, 60, 6).

    target="forecast" (variante A): EC y pH medidos (limpios, sin normalizar) en t+horizon.
    target="setpoint" (variante B): consigna del día en t+horizon (requiere columnas ec_target/ph_target).
    """
    X_all = df_norm[FEATURES].to_numpy(np.float32)
    if target == "forecast":
        Y_all = clean[TARGETS].to_numpy(np.float32)
    else:
        Y_all = clean[["ec_target", "ph_target"]].to_numpy(np.float32)
    idx = np.arange(window, len(df_norm) - horizon + 1, stride)
    X = np.stack([X_all[i - window:i] for i in idx])
    Y = Y_all[idx + horizon - 1]
    last = clean[TARGETS].to_numpy(np.float32)[idx - 1]   # para el baseline de persistencia
    return X, Y, last, idx


def split_windows(idx: np.ndarray, split: tuple[int, int], window: int = 60, horizon: int = 15):
    """Partición cronológica 70/15/15 sin solapamiento entre conjuntos."""
    i_tr, i_va = split
    end = idx + horizon - 1
    tr = end < i_tr
    va = (idx - window >= i_tr) & (end < i_va)
    te = idx - window >= i_va
    return tr, va, te
