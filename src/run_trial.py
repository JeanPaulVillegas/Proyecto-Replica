"""Fase 3: ensayo virtual de 60 días en lazo cerrado con el LSTM de la fase 2.

Uso (desde la carpeta del proyecto, con el entorno activo):
    python src/run_trial.py              # 10 semillas por tratamiento (~15-30 min)
    python src/run_trial.py --quick      # 2 semillas, para probar (~3-5 min)
    python src/run_trial.py --seeds 30   # como dice el plan (más lento)

Tratamientos (todos sobre el mismo simulador y las mismas semillas):
  1. Manual            una corrección al día con error de operador
  2. Umbrales          consigna fija EC 1,8 ±0,2 y pH 6,0 ±0,2
  3. Consigna sin LSTM lazo de la sección 3.4 siguiendo la consigna del día (ablación: ¿qué aporta el LSTM?)
  4. LSTM              igual que 3, pero decide con el pronóstico del LSTM a 15 min:
                       δ = consigna − ŷ(t+15)  →  dosifica antes de que EC/pH se salgan del rango

Requiere: fase 1 (data/) y fase 2 (resultados/lstm_articulo.keras o .tflite).
Salidas: resultados/fase3_*.csv, resultados/fase3_resumen.json, figs/fase3_*.png
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from simulator import run, Params, MIN_PER_DAY, ec_setpoint, ph_setpoint
from controllers import ManualController, ThresholdController, SmartController, ScheduleForecaster
from pipeline import ph_from_voltage, ec_compensate, PH_CAL, FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGS, RES = ROOT / "data", ROOT / "figs", ROOT / "resultados"
HARVEST_G = 250.0     # peso objetivo para "días a cosecha" (supuesto; el artículo no lo define)


# ----------------------------------------------------------------------------- LSTM como predictor
class LSTMForecaster:
    """Predictor para SmartController con la interfaz predict(minute, history).

    Replica el pipeline del edge: convierte cada lectura (Ec. 1 y 2), aplica la media móvil de 10
    (Ec. 3), rellena huecos con el último valor, agrupa los últimos 500 minutos en 60 pasos de 500 s
    (el formato con que se entrenó el modelo de la fase 2), normaliza con el z-score de entrenamiento
    (Ec. 4) y ejecuta el modelo TFLite. Devuelve (consigna EC, consigna pH) del día, pero
    "corrigiendo" la medida con el pronóstico: el controlador compara la consigna con lo que el
    LSTM espera que haya dentro de 15 min, no con lo medido ahora.
    """

    def __init__(self, tflite_path: Path, npz_path: Path, tf):
        w = np.load(npz_path)
        self.mu, self.sd = w["mu"].astype(np.float32), w["sd"].astype(np.float32)
        self.ymu, self.ysd = w["Y_train"].mean(0), w["Y_train"].std(0) + 1e-8
        self.interp = tf.lite.Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        self.i_in = self.interp.get_input_details()[0]["index"]
        self.i_out = self.interp.get_output_details()[0]["index"]
        self.rows: list[np.ndarray] = []
        self.last = None
        self.n_calls, self.ms = 0, 0.0
        self.last_forecast = None

    def observe(self, reading: dict):
        if np.isnan(reading.get("ec_raw", np.nan)):
            row = self.last if self.last is not None else None
        else:
            row = np.array([reading["temp_air"], reading["rh"],
                            float(ph_from_voltage(reading["ph_v"], *PH_CAL)),
                            float(ec_compensate(reading["ec_raw"], reading["t_water"])),
                            reading["nitrogen"], reading["lux"]], dtype=np.float32)
            # descarta picos físicamente imposibles (equivalente al filtro de outliers del edge)
            if self.last is not None and not (0.2 < row[3] < 5 and 3 < row[2] < 9):
                row = self.last
            if self.last is not None and np.isnan(row).any():   # falla puntual de un sensor (DHT22, NPK)
                row = np.where(np.isnan(row), self.last, row)
        if row is None:
            return
        self.last = row
        self.rows.append(row)
        if len(self.rows) > 520:
            self.rows.pop(0)

    def forecast(self):
        if len(self.rows) < 510:
            return None
        arr = np.array(self.rows)
        sm = pd.DataFrame(arr).rolling(10, min_periods=1).mean().to_numpy()[-500:]   # Ec. 3
        bins = (np.arange(500) * 60) // 500                                           # 60 pasos de 500 s
        win = np.stack([sm[bins == b].mean(0) for b in range(60)])
        x = ((win - self.mu) / self.sd).astype(np.float32)[None]                      # Ec. 4
        t0 = time.perf_counter()
        self.interp.set_tensor(self.i_in, x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.i_out)[0] * self.ysd + self.ymu
        self.ms += (time.perf_counter() - t0) * 1000
        self.n_calls += 1
        self.last_forecast = (float(y[0]), float(y[1]))
        self.last_ms = (time.perf_counter() - t0) * 1000
        return self.last_forecast


class LSTMSmartController(SmartController):
    """Lazo de la sección 3.4 con δ = consigna − pronóstico LSTM (Ec. 9 aplicada al estado futuro)."""

    def __init__(self, forecaster: LSTMForecaster):
        super().__init__(predictor=ScheduleForecaster())
        self.fc = forecaster

    def act(self, unit, reading, minute):
        self.fc.observe(reading)
        s = self._smoothed(reading)
        self._climate_hysteresis(unit, reading)
        if s is None:
            return
        ec, ph = s
        if minute % 15 == 0:
            day = (minute + 15) / MIN_PER_DAY
            ec_t, ph_t = ec_setpoint(day), ph_setpoint(day)
            f = self.fc.forecast()
            if f is not None:
                # combina lo medido con lo pronosticado: actúa si cualquiera de los dos se sale
                ec_f, ph_f = f
                ec = ec_f if abs(ec_t - ec_f) > abs(ec_t - ec) else ec
                ph = ph_f if abs(ph_t - ph_f) > abs(ph_t - ph) else ph
            self.target = (ec_t, ph_t)
            self._dose_to(unit, ec, ph, ec_t, ph_t, 0.2, 0.1)
            self.pending_check = minute + 5
        elif self.pending_check is not None and minute >= self.pending_check:
            ec_t, ph_t = self.target
            self._dose_to(unit, ec, ph, ec_t, ph_t, 0.1, 0.1)
            self.pending_check = None


# ----------------------------------------------------------------------------- utilidades
def load_tflite(tf) -> Path:
    tfl = RES / "lstm_articulo.tflite"
    if tfl.exists():
        return tfl
    keras_path = RES / "lstm_articulo.keras"
    if not keras_path.exists():
        raise SystemExit("No encuentro el modelo de la fase 2. Ejecuta antes: python src/train_models.py --quick")
    from train_models import tflite_latency
    model = tf.keras.models.load_model(keras_path)
    X = np.load(DATA / "windows_500s_forecast.npz")["X_test"]
    tflite_latency(model, tf, X, n=20)      # crea resultados/lstm_articulo.tflite
    return tfl


def summarize(name, seed, df, s) -> dict:
    days = df.minute / MIN_PER_DAY
    reach = df.index[df.biomass_mean_g >= HARVEST_G]
    light = df.light.astype(bool)
    return dict(
        tratamiento=name, semilla=seed,
        rendimiento_g=s["yield_g_mean"], desvio_g=s["yield_g_std"], cv_pct=s["yield_cv_pct"],
        dias_a_cosecha=float(days[reach[0]]) if len(reach) else np.nan,
        ec_error=float((df.true_ec - df.ec_target).abs().mean()),
        ph_error=float((df.true_ph - df.ph_target).abs().mean()),
        ec_fuera_rango_pct=float(((df.true_ec - df.ec_target).abs() > 0.2).mean() * 100),
        ph_fuera_rango_pct=float(((df.true_ph - df.ph_target).abs() > 0.2).mean() * 100),
        temp_desv=float((df.true_t_air - 25).abs().mean()), hr_desv=float((df.true_rh - 60).abs().mean()),
        concentrado_ml=s["dosed_ab_ml"], acido_base_ml=s["dosed_acid_ml"] + s["dosed_base_ml"],
        nutriente_descartado=s["nutr_discarded"],
        nutriente_usado=s["dosed_ab_ml"] / 1000 * 200 + s["nutr_discarded"],
        agua_l=s["water_added_l"], energia_kwh=s["energy_kwh"],
        bombas_min_dia=s["pump_min_per_day"], luz_h_dia=float(light.sum() / 60 / (len(df) / MIN_PER_DAY)),
    )


# ----------------------------------------------------------------------------- main
def main(a):
    import tensorflow as tf
    RES.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    tfl = load_tflite(tf)
    npz = DATA / "windows_500s_forecast.npz"
    seeds = list(range(100, 100 + (2 if a.quick else a.seeds)))   # semillas distintas a la de entrenamiento (0)
    print(f"Ensayo virtual · {len(seeds)} semillas × 4 tratamientos · 60 días cada uno")

    rows, traces, lat = [], {}, []
    t_start = time.time()
    for i, seed in enumerate(seeds):
        p = Params(seed=seed)
        fc = LSTMForecaster(tfl, npz, tf)
        treatments = [("Manual", ManualController(seed)), ("Umbrales", ThresholdController()),
                      ("Consigna sin LSTM", SmartController()), ("LSTM", LSTMSmartController(fc))]
        for name, ctrl in treatments:
            df, s = run(ctrl, p)
            rows.append(summarize(name, seed, df, s))
            if seed == seeds[0]:
                traces[name] = df[["minute", "true_ec", "ec_target", "true_ph", "ph_target", "biomass_mean_g"]].copy()
        if fc.n_calls:
            lat.append(fc.ms / fc.n_calls)
        el = time.time() - t_start
        print(f"  semilla {seed} lista ({i + 1}/{len(seeds)}) · {el / 60:.1f} min transcurridos")

    df = pd.DataFrame(rows)
    df.to_csv(RES / "fase3_por_semilla.csv", index=False)
    order = ["Manual", "Umbrales", "Consigna sin LSTM", "LSTM"]
    g = df.groupby("tratamiento").agg(["mean", "std"]).drop(columns="semilla", level=0).reindex(order)
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    g.round(4).to_csv(RES / "fase3_resumen_tratamientos.csv")

    # Equivalentes de las tablas del artículo
    m = df.groupby("tratamiento").mean(numeric_only=True).reindex(order)
    t2 = m[["rendimiento_g", "desvio_g", "cv_pct", "dias_a_cosecha"]].round(1)
    t2["articulo_g"] = [297, 323, np.nan, 368]
    t4 = m[["energia_kwh", "agua_l", "bombas_min_dia", "luz_h_dia"]].round(2)
    base_u, base_m = m.loc["Umbrales", "nutriente_usado"], m.loc["Manual", "nutriente_usado"]
    nut = pd.DataFrame({
        "nutriente_usado": m.nutriente_usado.round(1),
        "reduccion_vs_umbrales_%": ((base_u - m.nutriente_usado) / base_u * 100).round(1),
        "reduccion_vs_manual_%": ((base_m - m.nutriente_usado) / base_m * 100).round(1),
        "ec_error": m.ec_error.round(3), "ph_error": m.ph_error.round(3),
        "ec_fuera_rango_%": m.ec_fuera_rango_pct.round(1), "ph_fuera_rango_%": m.ph_fuera_rango_pct.round(1)})
    t2.to_csv(RES / "fase3_tabla2_rendimiento.csv")
    nut.to_csv(RES / "fase3_tabla1_nutrientes.csv")
    t4.to_csv(RES / "fase3_tabla4_recursos.csv")

    # Estadística: ANOVA de una vía + Tukey sobre el rendimiento (unidad = semilla)
    groups = [df[df.tratamiento == t].rendimiento_g.values for t in order]
    anova = stats.f_oneway(*groups) if len(seeds) > 1 else None
    tk = stats.tukey_hsd(*groups) if len(seeds) > 1 else None
    tukey_rows = []
    if tk is not None:
        for i in range(4):
            for j in range(i + 1, 4):
                tukey_rows.append(dict(a=order[i], b=order[j], diff_g=round(float(groups[j].mean() - groups[i].mean()), 2),
                                       p=round(float(tk.pvalue[i, j]), 4)))
        pd.DataFrame(tukey_rows).to_csv(RES / "fase3_tukey_rendimiento.csv", index=False)

    lstm_vs_cons = df[df.tratamiento == "LSTM"].set_index("semilla")
    cons = df[df.tratamiento == "Consigna sin LSTM"].set_index("semilla")
    summary = dict(seeds=seeds, latencia_tflite_ms=float(np.mean(lat)) if lat else None,
                   anova_rendimiento=dict(F=float(anova.statistic), p=float(anova.pvalue)) if anova else None,
                   tukey=tukey_rows,
                   lstm_menos_consigna=dict(
                       rendimiento_g=float((lstm_vs_cons.rendimiento_g - cons.rendimiento_g).mean()),
                       nutriente_usado=float((lstm_vs_cons.nutriente_usado - cons.nutriente_usado).mean()),
                       ec_error=float((lstm_vs_cons.ec_error - cons.ec_error).mean()),
                       ph_error=float((lstm_vs_cons.ph_error - cons.ph_error).mean())))
    (RES / "fase3_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Figuras
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {"Manual": "#c46a1f", "Umbrales": "#888", "Consigna sin LSTM": "#2a8a4a", "LSTM": "#1f5fa8"}
    for name, d in traces.items():
        x = d.minute / MIN_PER_DAY
        ax[0].plot(x, d.true_ec, lw=0.8, color=colors[name], label=name)
        ax[1].plot(x, d.true_ph, lw=0.8, color=colors[name])
        ax[2].plot(x, d.biomass_mean_g, lw=1.5, color=colors[name])
    d0 = next(iter(traces.values()))
    ax[0].plot(d0.minute / MIN_PER_DAY, d0.ec_target, "k--", lw=1, label="Consigna")
    ax[1].plot(d0.minute / MIN_PER_DAY, d0.ph_target, "k--", lw=1)
    ax[0].set_ylabel("EC (mS/cm)"); ax[1].set_ylabel("pH"); ax[2].set_ylabel("Biomasa (g/planta)")
    ax[2].set_xlabel("Día de cultivo"); ax[0].legend(fontsize=8, ncol=5)
    fig.suptitle(f"Fase 3 · ensayo virtual (semilla {seeds[0]})")
    fig.tight_layout(); fig.savefig(FIGS / "fase3_series.png", dpi=120)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for k, (col, lab) in enumerate([("rendimiento_g", "Rendimiento (g/planta)"),
                                    ("nutriente_usado", "Nutriente usado (mS/cm·L)"),
                                    ("ec_error", "Error medio de EC (mS/cm)")]):
        ax[k].boxplot([df[df.tratamiento == t][col] for t in order])
        ax[k].set_xticks([1, 2, 3, 4], ["Manual", "Umbr.", "Consigna", "LSTM"])
        ax[k].set_title(lab, fontsize=10)
    fig.suptitle(f"Fase 3 · {len(seeds)} semillas por tratamiento")
    fig.tight_layout(); fig.savefig(FIGS / "fase3_comparacion.png", dpi=120)

    pd.set_option("display.width", 200)
    print("\n=== Tipo Tabla 2: rendimiento ===\n" + t2.to_string())
    print("\n=== Tipo Tabla 1: nutrientes y seguimiento ===\n" + nut.to_string())
    print("\n=== Tipo Tabla 4: recursos ===\n" + t4.to_string())
    if anova:
        print(f"\nANOVA rendimiento: F = {anova.statistic:.2f}, p = {anova.pvalue:.4g}")
        print(pd.DataFrame(tukey_rows).to_string(index=False))
    print(f"\nLSTM − Consigna sin LSTM (media por semilla): {summary['lstm_menos_consigna']}")
    print(f"Latencia media TFLite: {summary['latencia_tflite_ms']:.3f} ms" if lat else "")
    print(f"\nResultados guardados en {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())
