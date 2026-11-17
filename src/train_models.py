"""Fase 2: entrena el LSTM del artículo y los modelos de comparación sobre las ventanas de la fase 1.

Uso (desde la carpeta del proyecto, con el entorno activo):
    python src/train_models.py                    # dataset 500 s (10 368 filas, como el artículo), 3 semillas
    python src/train_models.py --quick            # prueba rápida: 1 semilla, menos épocas
    python src/train_models.py --dataset 1min     # dataset a 1 min (86 400 filas, más lento)
    python src/train_models.py --target setpoint  # variante B del objetivo

Modelos: persistencia, regresión lineal, Random Forest, CNN 1D, CNN-LSTM y LSTM del artículo
(entrada 60×6, LSTM 64 → LSTM 32 → Dropout 0,2 → Dense 16 ReLU → Dense 2 lineal; Adam 0,001; MSE;
150 épocas con early stopping). El LSTM se entrena con y sin aumentación (jitter + time warping).
Métricas en el conjunto de prueba (Ec. 6–8): RMSE, MAPE y R², en unidades originales.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGS, RES = ROOT / "data", ROOT / "figs", ROOT / "resultados"
TARGET_NAMES = ["EC", "pH"]


# ----------------------------------------------------------------------------- métricas
def metrics(y, yhat) -> dict:
    out = {}
    for j, name in enumerate(TARGET_NAMES):
        e = y[:, j] - yhat[:, j]
        out[f"RMSE_{name}"] = float(np.sqrt(np.mean(e ** 2)))                       # Ec. 7
        out[f"MAPE_{name}_%"] = float(np.mean(np.abs(e / y[:, j])) * 100)           # Ec. 8
        out[f"R2_{name}"] = float(1 - np.sum(e ** 2) / np.sum((y[:, j] - y[:, j].mean()) ** 2))
    return out


# ----------------------------------------------------------------------------- aumentación
def jitter(X, sigma, rng):
    return X + rng.normal(0, sigma, X.shape).astype(np.float32)


def time_warp(X, max_warp, rng):
    """Estira o comprime cada ventana en el tiempo (±max_warp) y la re-muestrea a 60 pasos."""
    n, T, F = X.shape
    out = np.empty_like(X)
    base = np.arange(T)
    for i in range(n):
        f = 1 + rng.uniform(-max_warp, max_warp)
        src = np.clip(T - 1 - (T - 1 - base) * f, 0, T - 1)   # el último paso queda fijo
        for k in range(F):
            out[i, :, k] = np.interp(src, base, X[i, :, k])
    return out


def augment(X, Y, rng, sigma=0.03, warp=0.10):
    Xa = np.concatenate([X, jitter(X, sigma, rng), time_warp(X, warp, rng)])
    Ya = np.concatenate([Y, Y, Y])
    return Xa, Ya


# ----------------------------------------------------------------------------- modelos clásicos
def window_stats(X):
    """Rasgos para Random Forest: último valor, media, pendiente, mín. y máx. de cada variable."""
    t = np.arange(X.shape[1]) - (X.shape[1] - 1) / 2
    slope = (X * t[None, :, None]).sum(1) / (t ** 2).sum()
    return np.concatenate([X[:, -1], X.mean(1), slope, X.min(1), X.max(1)], axis=1)


# ----------------------------------------------------------------------------- modelos Keras
def build_nn(kind: str, tf):
    L = tf.keras.layers
    inp = L.Input(shape=(60, 6))
    if kind == "LSTM (artículo)":
        x = L.LSTM(64, return_sequences=True)(inp)
        x = L.LSTM(32)(x)
        x = L.Dropout(0.2)(x)
        x = L.Dense(16, activation="relu")(x)
    elif kind == "CNN 1D":
        x = L.Conv1D(32, 3, activation="relu", padding="same")(inp)
        x = L.Conv1D(32, 3, activation="relu", padding="same")(x)
        x = L.GlobalAveragePooling1D()(x)
        x = L.Dense(16, activation="relu")(x)
    elif kind == "CNN-LSTM":
        x = L.Conv1D(32, 3, activation="relu", padding="same")(inp)
        x = L.LSTM(32)(x)
        x = L.Dense(16, activation="relu")(x)
    else:
        raise ValueError(kind)
    out = L.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def fit_nn(kind, d, seed, tf, epochs, patience, batch, augment_train=False):
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    # La salida se normaliza con media/desvío de entrenamiento y se des-normaliza al evaluar
    ym, ys = d["Y_train"].mean(0), d["Y_train"].std(0) + 1e-8
    Xtr, Ytr = d["X_train"], (d["Y_train"] - ym) / ys
    if augment_train:
        Xtr, Ytr = augment(Xtr, Ytr, rng)
    model = build_nn(kind, tf)
    cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(Xtr, Ytr, validation_data=(d["X_val"], (d["Y_val"] - ym) / ys), epochs=epochs,
                     batch_size=batch, callbacks=[cb], verbose=0, shuffle=True)
    pred = model.predict(d["X_test"], verbose=0) * ys + ym
    return model, pred, dict(epochs_run=len(hist.history["loss"]), train_s=round(time.time() - t0, 1))


def tflite_latency(model, tf, X, n=200):
    """Convierte el LSTM a TFLite (como el artículo) y mide la latencia de una inferencia en esta CPU.

    Para que el conversor use solo operaciones nativas de TFLite, se arma una copia del modelo con
    las capas LSTM desenrolladas (unroll=True) y lote fijo de 1, con los mismos pesos.
    """
    L = tf.keras.layers
    inp = L.Input(shape=(60, 6), batch_size=1)
    x = L.LSTM(64, return_sequences=True, unroll=True)(inp)
    x = L.LSTM(32, unroll=True)(x)
    x = L.Dropout(0.2)(x)
    x = L.Dense(16, activation="relu")(x)
    edge = tf.keras.Model(inp, L.Dense(2)(x))
    edge.set_weights(model.get_weights())
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):   # el conversor imprime mucho texto interno
        tfl = tf.lite.TFLiteConverter.from_keras_model(edge).convert()
    (RES / "lstm_articulo.tflite").write_bytes(tfl)
    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    i_in, i_out = interp.get_input_details()[0]["index"], interp.get_output_details()[0]["index"]
    times = []
    for k in range(n):
        interp.set_tensor(i_in, X[k % len(X)][None].astype(np.float32))
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    out = interp.get_tensor(i_out)
    same = float(np.abs(out - model.predict(X[(n - 1) % len(X)][None], verbose=0)).max())
    return dict(tflite_kb=round(len(tfl) / 1024, 1), latency_ms_median=round(float(np.median(times)), 3),
                latency_ms_p95=round(float(np.percentile(times, 95)), 3), max_diff_vs_keras=same)


# ----------------------------------------------------------------------------- main
def main(a):
    import tensorflow as tf
    RES.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    f = DATA / f"windows_{a.dataset}_{a.target}.npz"
    if not f.exists():
        raise SystemExit(f"No encuentro {f}. Ejecuta primero la fase 1: python src/generate_dataset.py")
    d = dict(np.load(f))
    seeds = [0] if a.quick else list(range(a.seeds))
    epochs = 10 if a.quick else a.epochs
    print(f"TensorFlow {tf.__version__} · dataset {a.dataset} · objetivo {a.target} · "
          f"train/val/test = {len(d['X_train'])}/{len(d['X_val'])}/{len(d['X_test'])} · semillas {seeds}")

    rows, preds = [], {}

    def add(model, seed, pred, extra=None):
        r = dict(modelo=model, semilla=seed, **metrics(d["Y_test"], pred), **(extra or {}))
        rows.append(r)
        preds.setdefault(model, pred)
        print(f"  {model:<28} semilla {seed}  RMSE EC {r['RMSE_EC']:.4f}  RMSE pH {r['RMSE_pH']:.4f}  "
              f"R² EC {r['R2_EC']:.3f}  R² pH {r['R2_pH']:.3f}")

    # Persistencia (solo tiene sentido para la variante A, pero se informa igual)
    add("Persistencia", 0, d["last_test"])
    Xf_tr, Xf_te = d["X_train"].reshape(len(d["X_train"]), -1), d["X_test"].reshape(len(d["X_test"]), -1)
    add("Regresión lineal", 0, LinearRegression().fit(Xf_tr, d["Y_train"]).predict(Xf_te))
    for s in seeds:
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=s)
        add("Random Forest", s, rf.fit(window_stats(d["X_train"]), d["Y_train"]).predict(window_stats(d["X_test"])))

    lstm_model = None
    for s in seeds:
        for kind, aug in [("LSTM (artículo)", True), ("LSTM (artículo)", False), ("CNN 1D", True), ("CNN-LSTM", True)]:
            name = kind + ("" if aug else " sin aumentación")
            model, pred, info = fit_nn(kind, d, s, tf, epochs, a.patience, a.batch, augment_train=aug)
            add(name, s, pred, info)
            if kind == "LSTM (artículo)" and aug and lstm_model is None:
                lstm_model = model
                lstm_model.save(RES / "lstm_articulo.keras")

    df = pd.DataFrame(rows)
    df.to_csv(RES / f"metricas_fase2_{a.dataset}_{a.target}_por_semilla.csv", index=False)
    cols = [c for c in df.columns if c.startswith(("RMSE", "MAPE", "R2"))]
    summ = df.groupby("modelo", sort=False)[cols].agg(["mean", "std"]).round(4)
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.to_csv(RES / f"metricas_fase2_{a.dataset}_{a.target}.csv")

    lat = {}
    try:
        lat = tflite_latency(lstm_model, tf, d["X_test"])
    except Exception as e:  # la conversión puede fallar según la versión de TF
        lat = {"error": str(e)[:300]}

    ref = {"RMSE_EC": 0.25, "RMSE_pH": 0.18, "MAPE_EC_%": 6.2, "MAPE_pH_%": 4.7, "R2_EC": 0.91, "R2_pH": 0.94}
    lstm = df[df.modelo == "LSTM (artículo)"][list(ref)].mean()
    comp = pd.DataFrame({"artículo": ref, "réplica (media)": lstm.round(4)})
    # Hipótesis: el artículo reportó el RMSE en la escala normalizada (z-score), no en mS/cm ni en pH.
    # El R² no cambia con la escala; el RMSE se divide por el desvío estándar de entrenamiento.
    ysd = d["Y_train"].std(0)
    comp["réplica en escala normalizada"] = [round(lstm["RMSE_EC"] / ysd[0], 3), round(lstm["RMSE_pH"] / ysd[1], 3),
                                             np.nan, np.nan, round(lstm["R2_EC"], 4), round(lstm["R2_pH"], 4)]
    comp.to_csv(RES / f"comparacion_articulo_{a.dataset}_{a.target}.csv")
    info = dict(dataset=a.dataset, target=a.target, seeds=seeds, epochs_max=epochs, tensorflow=tf.__version__,
                n_train=len(d["X_train"]), n_val=len(d["X_val"]), n_test=len(d["X_test"]),
                y_test_std=d["Y_test"].std(0).round(4).tolist(), tflite=lat)
    (RES / f"info_fase2_{a.dataset}_{a.target}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))

    # Figura: predicción vs real en el último tramo del test
    n = min(len(d["Y_test"]), 600)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for j, name in enumerate(TARGET_NAMES):
        ax[j].plot(d["Y_test"][-n:, j], color="k", lw=1.5, label="Real")
        for m, c in [("Persistencia", "#999"), ("Random Forest", "#2a8a4a"), ("LSTM (artículo)", "#1f5fa8")]:
            ax[j].plot(preds[m][-n:, j], lw=1, color=c, label=m)
        ax[j].set_ylabel(name)
    ax[0].legend(fontsize=8, ncol=4)
    ax[1].set_xlabel("Ventana del conjunto de prueba")
    fig.suptitle(f"Fase 2 · predicción a 15 min (dataset {a.dataset}, objetivo {a.target})")
    fig.tight_layout()
    fig.savefig(FIGS / f"fase2_prediccion_{a.dataset}_{a.target}.png", dpi=120)

    pd.set_option("display.width", 200)
    print("\n=== Media ± desvío por modelo (conjunto de prueba) ===")
    print(summ[[c for c in summ.columns if c.startswith(("RMSE", "R2"))]].to_string())
    print("\n=== LSTM: artículo vs réplica ===")
    print(comp.to_string())
    print(f"\nTFLite: {lat}")
    print(f"Desvío estándar del objetivo en test (EC, pH): {info['y_test_std']}")
    print(f"\nResultados guardados en {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="500s", choices=["500s", "1min"])
    ap.add_argument("--target", default="forecast", choices=["forecast", "setpoint"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())
