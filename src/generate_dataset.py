"""Fase 1: genera el dataset sintético de 60 días, lo preprocesa y arma las ventanas 60 × 6.

Uso:  python generate_dataset.py [--seed 0]
Salidas en ../data y ../figs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulator import run, Params, MIN_PER_DAY
from controllers import ThresholdController, ManualController, SmartController
from pipeline import preprocess, make_windows, split_windows, FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGS = ROOT / "data", ROOT / "figs"


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build(raw: pd.DataFrame, name: str, window=60, horizon=15):
    df_norm, clean, scaler, rep = preprocess(raw)
    out = {}
    for target in ["forecast", "setpoint"]:
        X, Y, last, idx = make_windows(df_norm, clean, window, horizon, target=target)
        tr, va, te = split_windows(idx, rep["split"], window, horizon)
        np.savez_compressed(DATA / f"windows_{name}_{target}.npz", X_train=X[tr], Y_train=Y[tr], X_val=X[va],
                            Y_val=Y[va], X_test=X[te], Y_test=Y[te], last_test=last[te],
                            mu=scaler.mu.values, sd=scaler.sd.values, features=np.array(FEATURES))
        pers = {"rmse_ec": rmse(last[te][:, 0], Y[te][:, 0]), "rmse_ph": rmse(last[te][:, 1], Y[te][:, 1])}
        out[target] = dict(n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
                           X_shape=list(X.shape), persistence_test=pers,
                           y_std_test=[float(Y[te][:, 0].std()), float(Y[te][:, 1].std())])
    clean.to_csv(DATA / f"clean_{name}.csv.gz", index=False)
    rep["windows"] = out
    return rep


def main(seed: int):
    DATA.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    p = Params(seed=seed)

    # 1) Dataset de entrenamiento: unidad con control por umbrales (como el "histórico" del artículo)
    raw, summ = run(ThresholdController(), p)
    raw.to_csv(DATA / "raw_threshold_1min.csv.gz", index=False)

    # 2) Versión remuestreada: 500 s → exactamente 10 368 filas en 60 días (cifra del artículo)
    rs = raw.set_index("timestamp").resample("500s").mean(numeric_only=True).reset_index()
    rs.to_csv(DATA / "raw_threshold_500s.csv.gz", index=False)

    report = {"params": {k: v for k, v in p.__dict__.items() if k != "extra"}, "threshold_summary": summ}
    report["dataset_1min"] = build(raw, "1min", window=60, horizon=15)
    # en 500 s, 60 pasos = 8,3 h y 15 min ≈ 2 pasos
    report["dataset_500s"] = build(rs, "500s", window=60, horizon=2)

    # 3) Corrida rápida de los tres controladores (misma semilla) para revisar el simulador
    runs = {"Manual": (raw_m := run(ManualController(seed), p))[0], "Umbrales": raw,
            "Inteligente (provisional)": (raw_s := run(SmartController(), p))[0]}
    summaries = {"Manual": raw_m[1], "Umbrales": summ, "Inteligente (provisional)": raw_s[1]}
    rows = []
    for k, d in runs.items():
        s = summaries[k]
        rows.append(dict(
            tratamiento=k,
            rendimiento_g=round(s["yield_g_mean"], 1), cv_pct=round(s["yield_cv_pct"], 1),
            temp_desv_media_C=round((d.true_t_air - 25).abs().mean(), 2),
            temp_sd_C=round(d.true_t_air.std(), 2),
            hr_desv_media_pct=round((d.true_rh - 60).abs().mean(), 2),
            ec_error_medio=round((d.true_ec - d.ec_target).abs().mean(), 3),
            ph_error_medio=round((d.true_ph - d.ph_target).abs().mean(), 3),
            ab_dosificado_ml=round(s["dosed_ab_ml"], 0), acido_ml=round(s["dosed_acid_ml"], 0),
            base_ml=round(s["dosed_base_ml"], 0), agua_l=round(s["water_added_l"], 1),
            energia_kwh=round(s["energy_kwh"], 1), bombas_min_dia=round(s["pump_min_per_day"], 2),
        ))
    comp = pd.DataFrame(rows)
    comp.to_csv(DATA / "chequeo_tres_controladores.csv", index=False)
    report["three_controllers_check"] = rows

    # Tabla 1 del artículo: EC/pH cada 10 días (tratamiento inteligente provisional)
    d = runs["Inteligente (provisional)"]
    t1 = []
    for day in [10, 20, 30, 40, 50, 60]:
        w = d[(d.minute >= (day - 1) * MIN_PER_DAY) & (d.minute < day * MIN_PER_DAY)]
        t1.append(dict(dia=day, ec_obj=round(w.ec_target.mean(), 2), ec_med=round(w.true_ec.mean(), 3),
                       ph_obj=round(w.ph_target.mean(), 2), ph_med=round(w.true_ph.mean(), 3)))
    report["table1_like"] = t1

    with open(DATA / "reporte_fase1.json", "w") as f:
        json.dump(report, f, indent=2, default=float, ensure_ascii=False)

    # 4) Figuras
    days = raw.minute / MIN_PER_DAY
    fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    clean = pd.read_csv(DATA / "clean_1min.csv.gz")
    ax[0].plot(days, clean.ec, lw=0.5, color="#888", label="EC medida (procesada)")
    ax[0].plot(days, raw.true_ec, lw=1, color="#1f5fa8", label="EC real")
    ax[0].plot(days, raw.ec_target, "--", color="k", lw=1, label="Consigna (Tabla 1)")
    ax[0].axhline(1.8, color="#c44", lw=0.8, ls=":", label="Umbral fijo 1,8")
    ax[0].set_ylabel("EC (mS/cm)"); ax[0].legend(fontsize=8, ncol=4)
    ax[1].plot(days, clean.ph, lw=0.5, color="#888", label="pH medido (procesado)")
    ax[1].plot(days, raw.true_ph, lw=1, color="#2a8a4a", label="pH real")
    ax[1].plot(days, raw.ph_target, "--", color="k", lw=1, label="Consigna")
    ax[1].set_ylabel("pH"); ax[1].legend(fontsize=8, ncol=3)
    ax[2].plot(days, raw.t_out, lw=0.5, color="#bbb", label="Temperatura exterior, Chennai (°C)")
    ax[2].plot(days, raw.true_t_air, lw=0.6, color="#c46a1f", label="Temperatura del cuarto (°C)")
    ax[2].plot(days, raw.true_rh / 2.4, lw=0.6, color="#6a4ab0", alpha=0.7, label="HR / 2,4 (%)")
    ax[2].axhline(25, color="k", ls="--", lw=0.8); ax[2].legend(fontsize=8)
    for k, dd in runs.items():
        ax[3].plot(dd.minute / MIN_PER_DAY, dd.biomass_mean_g, label=k)
    ax[3].set_ylabel("Biomasa media (g/planta)"); ax[3].set_xlabel("Día de cultivo"); ax[3].legend(fontsize=8)
    fig.suptitle("Fase 1 · dataset sintético (control por umbrales) y chequeo de los tres controladores")
    fig.tight_layout()
    fig.savefig(FIGS / "fase1_series.png", dpi=120)
    print(json.dumps({k: report[k] for k in ["three_controllers_check", "table1_like"]}, indent=1, ensure_ascii=False))
    print(json.dumps({"1min": report["dataset_1min"]["windows"], "500s": report["dataset_500s"]["windows"]}, indent=1))
    print("gaps/outliers:", report["dataset_1min"]["gap_rows_interpolated"], report["dataset_1min"]["outliers"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
