"""Matrices de confusión del pronóstico de EC y pH a 15 minutos (complemento de la fase 2).

El LSTM es un modelo de regresión: predice números, no clases. Para armar una matriz de confusión,
cada valor se clasifica según la decisión que tomaría el lazo de control de la sección 3.4:

    baja      valor < consigna − tolerancia      (hay que subir: nutriente A/B, o base para el pH)
    en rango  dentro de consigna ± tolerancia    (no hay que hacer nada)
    alta      valor > consigna + tolerancia      (hay que bajar: diluir, o ácido para el pH)

Tolerancias del lazo: 0,2 mS/cm para la EC y 0,1 para el pH (las ε de controllers.py).
Clase real     = EC o pH medidos (procesados) 15 min después.
Clase predicha = lo que pronosticó cada modelo para ese mismo momento.

Se comparan tres modelos sobre el mismo conjunto de prueba: persistencia, regresión lineal y el
LSTM del artículo (el .tflite de la fase 2).

Requiere: fase 1 (data/windows_500s_*.npz) y fase 2 (resultados/lstm_articulo.tflite).
Uso:     python src/matriz_confusion.py
Salidas: figs/matriz_confusion.png y resultados/matriz_confusion_metricas.csv
"""
from __future__ import annotations

import os
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
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA, RES, FIGS = ROOT / "data", ROOT / "resultados", ROOT / "figs"
CLASES = ["baja", "en rango", "alta"]
VARIABLES = [("EC", 0, 0.2, "mS/cm"), ("pH", 1, 0.1, "")]


def clasificar(valor, consigna, tol):
    """0 = baja, 1 = en rango, 2 = alta."""
    return np.where(valor < consigna - tol, 0, np.where(valor > consigna + tol, 2, 1))


def predecir_lstm(d, tfl_path):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tfl_path))
    interp.allocate_tensors()
    i_in, i_out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
    ym, ys = d["Y_train"].mean(0), d["Y_train"].std(0) + 1e-8      # la red se entrenó con la salida normalizada
    out = np.empty((len(d["X_test"]), 2), np.float32)
    for k, x in enumerate(d["X_test"]):
        interp.set_tensor(i_in, x[None].astype(np.float32))
        interp.invoke()
        out[k] = interp.get_tensor(i_out)[0]
    return out * ys + ym


def main():
    f_fc, f_sp = DATA / "windows_500s_forecast.npz", DATA / "windows_500s_setpoint.npz"
    tfl = RES / "lstm_articulo.tflite"
    for f, fase in [(f_fc, "1 (python src/generate_dataset.py)"), (f_sp, "1 (python src/generate_dataset.py)"),
                    (tfl, "2 (python src/train_models.py)")]:
        if not f.exists():
            raise SystemExit(f"Falta {f}. Corre primero la fase {fase}.")
    d = dict(np.load(f_fc))
    consigna = np.load(f_sp)["Y_test"]       # mismas ventanas: la consigna del día en ese mismo momento
    real = d["Y_test"]

    Xf_tr, Xf_te = d["X_train"].reshape(len(d["X_train"]), -1), d["X_test"].reshape(len(d["X_test"]), -1)
    modelos = {
        "Persistencia": d["last_test"],
        "Regresión lineal": LinearRegression().fit(Xf_tr, d["Y_train"]).predict(Xf_te),
        "LSTM (artículo)": predecir_lstm(d, tfl),
    }

    filas = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2))
    for r, (var, j, tol, _) in enumerate(VARIABLES):
        y_real = clasificar(real[:, j], consigna[:, j], tol)
        for c, (nombre, pred) in enumerate(modelos.items()):
            y_pred = clasificar(pred[:, j], consigna[:, j], tol)
            cm = confusion_matrix(y_real, y_pred, labels=[0, 1, 2])
            fila = dict(variable=var, modelo=nombre, n=int(cm.sum()),
                        exactitud_pct=round(100 * np.trace(cm) / cm.sum(), 1))
            for k, clase in enumerate(CLASES):
                n_real, n_pred = cm[k].sum(), cm[:, k].sum()
                fila[f"sensibilidad_{clase}_pct"] = round(100 * cm[k, k] / n_real, 1) if n_real else None
                fila[f"precision_{clase}_pct"] = round(100 * cm[k, k] / n_pred, 1) if n_pred else None
                fila[f"casos_reales_{clase}"] = int(n_real)
            filas.append(fila)

            ax = axes[r, c]
            pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100       # % de cada clase real
            ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
            for a in range(3):
                for b in range(3):
                    ax.text(b, a, f"{cm[a, b]}\n{pct[a, b]:.0f} %", ha="center", va="center", fontsize=10,
                            color="white" if pct[a, b] > 60 else "#1f2937")
            ax.set_xticks(range(3), CLASES)
            ax.set_yticks(range(3), CLASES)
            ax.set_xlabel("Clase predicha")
            if c == 0:
                ax.set_ylabel(f"{var}: clase real")
            ax.set_title(f"{nombre} · {var}\nexactitud {fila['exactitud_pct']:.1f} %", fontsize=11)

    fig.suptitle("Matrices de confusión del pronóstico a 15 min (conjunto de prueba)\n"
                 "Clases según la consigna del día: EC ± 0,2 mS/cm, pH ± 0,1", fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    fig.savefig(FIGS / "matriz_confusion.png", dpi=150)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(RES / "matriz_confusion_metricas.csv", index=False)

    cols = ["variable", "modelo", "exactitud_pct", "sensibilidad_baja_pct", "sensibilidad_en rango_pct",
            "sensibilidad_alta_pct"]
    print(tabla[cols].to_string(index=False))
    print(f"\nCasos reales en prueba: " + "; ".join(
        f"{f['variable']} baja {f['casos_reales_baja']}, en rango {f['casos_reales_en rango']}, alta {f['casos_reales_alta']}"
        for f in filas[::3]))
    print(f"Guardado: {FIGS / 'matriz_confusion.png'} y {RES / 'matriz_confusion_metricas.csv'}")


if __name__ == "__main__":
    main()
