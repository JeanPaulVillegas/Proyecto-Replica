"""Fase 5: ¿qué tendría que ser cierto para obtener las cifras del artículo?

Corre la réplica completa (datos → LSTM → ensayo de 4 tratamientos) en dos escenarios y los compara
con el artículo:

  realista   el simulador tal cual (clima de Chennai sin aire acondicionado, óptimos agronómicos
             en meseta, sensores con la precisión de su hoja de datos, umbrales sin desperdicio).
  favorable  cuatro supuestos que acercan los resultados al artículo:
               1. cuarto climatizado a 24 °C (plausible: el artículo no lo descarta)
               2. sensores más ruidosos y con más deriva (plausible: sensores baratos)
               3. manual y umbrales cambian toda la solución cada 30 días (práctica común)
               4. la lechuga crece mejor cuanto más cerca esté de la consigna EXACTA de la Tabla 1
                  (sin evidencia: es lo que haría falta para que seguir la consigna dé rendimiento)

Uso:  python src/escenarios.py                 # 5 semillas por tratamiento y escenario (~25-40 min)
      python src/escenarios.py --quick         # 2 semillas, para probar (~10-15 min)
      python src/escenarios.py --solo favorable
Salidas: resultados/fase5/<escenario>/..., resultados/fase5_comparacion.csv, figs/fase5_escenarios.png
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
from sklearn.linear_model import LinearRegression

import controllers
import generate_dataset
import train_models
from controllers import ManualController, ThresholdController, SmartController
from run_trial import LSTMForecaster, LSTMSmartController, summarize
from simulator import run, Params

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "resultados", ROOT / "figs"

ESCENARIOS = {
    "realista": dict(params={}, cambio_solucion=False),
    "favorable": dict(
        params=dict(ac_setpoint_c=24.0,                                        # 1
                    noise_ph=0.15, noise_ec_rel=0.04, ph_sensor_drift_per_week=0.10,  # 2
                    crecimiento_sigue_consigna=True, extra=dict(ancho_ec=0.7, ancho_ph=0.6)),  # 4
        cambio_solucion=True),                                                  # 3
}
ARTICULO = {
    "Rendimiento manual (g)": 297, "Rendimiento umbrales (g)": 323, "Rendimiento LSTM (g)": 368,
    "LSTM vs umbrales, rendimiento (%)": 13.9, "Umbrales vs manual, rendimiento (%)": 8.8,
    "Ahorro de nutrientes LSTM vs umbrales (%)": 27.6,
    "Error medio EC con LSTM (mS/cm)": 0.03, "Error medio pH con LSTM": 0.05,
    "RMSE EC del LSTM, escala normalizada": 0.25, "RMSE pH del LSTM, escala normalizada": 0.18,
    "R² EC del LSTM": 0.91, "R² pH del LSTM": 0.94,
}


def entrenar(nombre, cfg, out, tf, epochs):
    """Datos de entrenamiento (unidad con umbrales, semilla 0) → ventanas 500 s → LSTM → TFLite."""
    p = Params(seed=0, **cfg["params"])
    raw, _ = run(ThresholdController(cfg["cambio_solucion"]), p)
    rs = raw.set_index("timestamp").resample("500s").mean(numeric_only=True).reset_index()
    generate_dataset.DATA = out
    generate_dataset.build(rs, "500s", window=60, horizon=2)
    npz = out / "windows_500s_forecast.npz"
    d = dict(np.load(npz))
    ysd = d["Y_train"].std(0)
    model, pred, info = train_models.fit_nn("LSTM (artículo)", d, 0, tf, epochs, 10, 128, augment_train=True)
    m = train_models.metrics(d["Y_test"], pred)
    lin = LinearRegression().fit(d["X_train"].reshape(len(d["X_train"]), -1), d["Y_train"])
    m_lin = train_models.metrics(d["Y_test"], lin.predict(d["X_test"].reshape(len(d["X_test"]), -1)))
    m_per = train_models.metrics(d["Y_test"], d["last_test"])
    train_models.RES = out
    lat = train_models.tflite_latency(model, tf, d["X_test"], n=50)
    met = dict(lstm=m, lineal=m_lin, persistencia=m_per, ysd_train=ysd.tolist(), tflite=lat, **info)
    (out / "metricas_lstm.json").write_text(json.dumps(met, indent=2, ensure_ascii=False))
    print(f"  [{nombre}] LSTM RMSE EC {m['RMSE_EC']:.4f} (z {m['RMSE_EC'] / ysd[0]:.3f}) · "
          f"pH {m['RMSE_pH']:.4f} (z {m['RMSE_pH'] / ysd[1]:.3f}) · R² {m['R2_EC']:.3f}/{m['R2_pH']:.3f}")
    return out / "lstm_articulo.tflite", npz, met


def ensayo(nombre, cfg, tfl, npz, tf, seeds):
    rows, traces = [], {}
    for i, seed in enumerate(seeds):
        p = Params(seed=seed, **cfg["params"])
        cs = cfg["cambio_solucion"]
        tratamientos = [("Manual", ManualController(seed, cs)), ("Umbrales", ThresholdController(cs)),
                        ("Consigna sin LSTM", SmartController()),
                        ("LSTM", LSTMSmartController(LSTMForecaster(tfl, npz, tf)))]
        for name, ctrl in tratamientos:
            df, s = run(ctrl, p)
            rows.append(summarize(name, seed, df, s))
            if i == 0:
                traces[name] = df[["minute", "biomass_mean_g"]].copy()
        print(f"  [{nombre}] semilla {seed} lista ({i + 1}/{len(seeds)})")
    return pd.DataFrame(rows), traces


def resumen(df, met):
    m = df.groupby("tratamiento").mean(numeric_only=True)
    y = m.rendimiento_g
    ysd = met["ysd_train"]
    return {
        "Rendimiento manual (g)": y["Manual"], "Rendimiento umbrales (g)": y["Umbrales"],
        "Rendimiento LSTM (g)": y["LSTM"],
        "LSTM vs umbrales, rendimiento (%)": (y["LSTM"] / y["Umbrales"] - 1) * 100,
        "Umbrales vs manual, rendimiento (%)": (y["Umbrales"] / y["Manual"] - 1) * 100,
        "Ahorro de nutrientes LSTM vs umbrales (%)": (1 - m.nutriente_usado["LSTM"] / m.nutriente_usado["Umbrales"]) * 100,
        "Error medio EC con LSTM (mS/cm)": m.ec_error["LSTM"], "Error medio pH con LSTM": m.ph_error["LSTM"],
        "RMSE EC del LSTM, escala normalizada": met["lstm"]["RMSE_EC"] / ysd[0],
        "RMSE pH del LSTM, escala normalizada": met["lstm"]["RMSE_pH"] / ysd[1],
        "R² EC del LSTM": met["lstm"]["R2_EC"], "R² pH del LSTM": met["lstm"]["R2_pH"],
        "LSTM vs consigna sin LSTM, rendimiento (%)": (y["LSTM"] / y["Consigna sin LSTM"] - 1) * 100,
        "LSTM supera a la regresión lineal (RMSE EC)": "sí" if met["lstm"]["RMSE_EC"] < met["lineal"]["RMSE_EC"] else "no",
    }


def main(a):
    import tensorflow as tf
    seeds = list(range(100, 100 + (2 if a.quick else a.seeds)))
    epochs = 10 if a.quick else a.epochs
    nombres = [a.solo] if a.solo else list(ESCENARIOS)
    tabla, curvas = {}, {}
    t0 = time.time()
    for nombre in nombres:
        cfg = ESCENARIOS[nombre]
        out = RES / "fase5" / nombre
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Escenario {nombre} ===")
        tfl, npz, met = entrenar(nombre, cfg, out, tf, epochs)
        df, traces = ensayo(nombre, cfg, tfl, npz, tf, seeds)
        df.to_csv(out / "ensayo_por_semilla.csv", index=False)
        tabla[nombre] = resumen(df, met)
        curvas[nombre] = traces
        print(f"  [{nombre}] listo · {(time.time() - t0) / 60:.1f} min transcurridos")

    comp = pd.DataFrame({"artículo": pd.Series(ARTICULO), **{k: pd.Series(v) for k, v in tabla.items()}})
    comp = comp.reindex(list(next(iter(tabla.values())).keys()))
    comp = comp.apply(lambda c: c.map(lambda v: round(v, 3) if isinstance(v, float) else v))
    comp.index.name = "métrica"
    comp.to_csv(RES / "fase5_comparacion.csv")
    pd.set_option("display.width", 200)
    print("\n=== Fase 5 · artículo vs escenarios ===")
    print(comp.to_string())

    fig, ax = plt.subplots(1, len(curvas), figsize=(6 * len(curvas), 4.2), sharey=True, squeeze=False)
    colors = {"Manual": "#c46a1f", "Umbrales": "#888", "Consigna sin LSTM": "#2a8a4a", "LSTM": "#1f5fa8"}
    for k, (nombre, tr) in enumerate(curvas.items()):
        for name, d in tr.items():
            ax[0, k].plot(d.minute / 1440, d.biomass_mean_g, color=colors[name], lw=1.5,
                          ls="--" if name == "LSTM" else "-", label=name)
        for name, v in [("Manual", 297), ("Umbrales", 323), ("LSTM", 368)]:
            ax[0, k].scatter([60], [v], color=colors[name], marker="*", s=120, zorder=5)
        ax[0, k].set_title(f"Escenario {nombre} (estrellas: artículo)")
        ax[0, k].set_xlabel("Día de cultivo")
    ax[0, 0].set_ylabel("Biomasa media (g/planta)")
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "fase5_escenarios.png", dpi=120)
    print(f"\nGuardado en {RES / 'fase5_comparacion.csv'} y {FIGS / 'fase5_escenarios.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--solo", choices=list(ESCENARIOS))
    main(ap.parse_args())
