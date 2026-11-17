"""Valida que los datos simulados sean realistas: cada variable contra un rango de referencia con fuente.

Se corre DESPUÉS de la fase 1 (usa data/raw_threshold_1min.csv.gz y data/chequeo_tres_controladores.csv).
Uso:  python src/validar_realismo.py
Salida: tabla en pantalla y resultados/validacion_realismo.csv  (OK / FUERA por cada chequeo)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clima import T_MEDIA_JMA, T_MAX, T_MIN, CSV_REAL
from simulator import run, Params, MIN_PER_DAY

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "resultados"

F = dict(
    jma="https://ds.data.jma.go.jp/gmd/tcc/tcc/products/climate/normal/parts/NrmMonth_e.php?stn=43279",
    w2t="https://weather2travel.com/india/chennai/climate",
    cornell="https://bpb-us-e1.wpmucdn.com/blogs.cornell.edu/dist/8/8824/files/2019/06/GPNOctober2016_LeafyGreens.pdf",
    gg="https://www.greenhousegrower.com/crops/varieties/manage-ph-and-soluble-salts-in-hydroponics/",
    hd="https://www.hortidaily.com/article/9878963/evaluating-lettuce-production-in-hydroponic-systems/",
    barbosa="https://pmc.ncbi.nlm.nih.gov/articles/PMC4483736/",
    dht="https://cdn.sparkfun.com/assets/f/7/d/9/c/DHT22.pdf",
    propio="criterio físico de la réplica",
)

rows = []


def chk(grupo, nombre, valor, lo, hi, fuente, fmt="{:.2f}"):
    ok = (lo is None or valor >= lo) and (hi is None or valor <= hi)
    rango = f"{'' if lo is None else lo} – {'' if hi is None else hi}"
    rows.append(dict(grupo=grupo, chequeo=nombre, simulado=fmt.format(valor), referencia=rango,
                     resultado="OK" if ok else "FUERA", fuente=F.get(fuente, fuente)))


def main():
    raw_path = DATA / "raw_threshold_1min.csv.gz"
    if not raw_path.exists():
        sys.exit("Falta data/raw_threshold_1min.csv.gz: corre primero  python src/generate_dataset.py")
    d = pd.read_csv(raw_path, parse_dates=["timestamp"])
    if "t_out" not in d:
        sys.exit("El dataset es de la versión anterior del simulador: vuelve a correr  python src/generate_dataset.py")
    comp = pd.read_csv(DATA / "chequeo_tres_controladores.csv")
    d["dia"] = d.minute // MIN_PER_DAY
    g = d.groupby("dia")
    origen = "real (Open-Meteo)" if CSV_REAL.exists() else "respaldo climatológico"
    print(f"Clima exterior usado: {origen}\n")

    # --- Clima exterior de Chennai ------------------------------------------------------------
    for mes, w in d.groupby(d.timestamp.dt.month):
        if len(w) < 10 * MIN_PER_DAY:
            continue
        m = mes - 1
        chk("Exterior", f"T media mes {mes} (°C)", w.t_out.mean(), T_MEDIA_JMA[m] - 1.5, T_MEDIA_JMA[m] + 1.5, "jma")
        gm = w.groupby("dia").t_out
        chk("Exterior", f"T máx diaria media mes {mes} (°C)", gm.max().mean(), T_MAX[m] - 2, T_MAX[m] + 2, "w2t")
        chk("Exterior", f"T mín diaria media mes {mes} (°C)", gm.min().mean(), T_MIN[m] - 2, T_MIN[m] + 2, "w2t")
    chk("Exterior", "HR exterior mínima (%)", d.rh_out.min(), 25, None, "propio", "{:.0f}")

    # --- Cuarto de cultivo -----------------------------------------------------------------------
    chk("Cuarto", "T aire interior mínima (°C)", d.true_t_air.min(), 17, None, "propio")
    chk("Cuarto", "T aire interior máxima (°C)", d.true_t_air.max(), None, 35, "propio")
    chk("Cuarto", "Oscilación diaria interior < exterior (°C)",
        (g.true_t_air.max() - g.true_t_air.min()).mean() - (g.t_out.max() - g.t_out.min()).mean(), None, 0, "propio")
    chk("Cuarto", "Cambio máximo de T aire en 1 min (°C)", d.true_t_air.diff().abs().max(), None, 0.5, "propio")
    chk("Cuarto", "HR interior media (%)", d.true_rh.mean(), 45, 80, "cornell", "{:.1f}")
    chk("Cuarto", "Cambio máximo de HR en 1 min (%)", d.true_rh.diff().abs().max(), None, 3, "propio")
    chk("Cuarto", "CO2 interior medio (ppm)", d.true_co2.mean(), 400, 1000, "propio", "{:.0f}")
    chk("Cuarto", "T agua mín (°C)", d.true_t_water.min(), 18, None, "propio")
    chk("Cuarto", "T agua máx (°C)", d.true_t_water.max(), None, 30, "cornell")
    chk("Cuarto", "Oscilación diaria agua < aire (°C)",
        (g.true_t_water.max() - g.true_t_water.min()).mean() - (g.true_t_air.max() - g.true_t_air.min()).mean(), None, 0, "cornell")
    pct_t = ((d.true_t_air >= 18) & (d.true_t_air <= 27)).mean() * 100
    pct_h = ((d.true_rh >= 50) & (d.true_rh <= 70)).mean() * 100
    print(f"(info) tiempo con aire entre 18 y 27 °C: {pct_t:.0f} %  ·  con HR entre 50 y 70 %: {pct_h:.0f} %")

    # --- Solución nutritiva (control por umbrales) -------------------------------------------------
    chk("Solución", "pH mínimo con control", d.true_ph.min(), 5.5, None, "gg")
    chk("Solución", "pH máximo con control", d.true_ph.max(), None, 6.5, "gg")
    chk("Solución", "EC mínima (mS/cm)", d.true_ec.min(), 1.0, None, "gg")
    chk("Solución", "EC máxima (mS/cm)", d.true_ec.max(), None, 2.4, "gg")
    th = comp[comp.tratamiento == "Umbrales"].iloc[0]
    chk("Solución", "Base / ácido usados (el pH tiende a subir)", th.base_ml / max(th.acido_ml, 1), None, 0.2, "gg")
    # pH sin ningún control: debe subir, pero no más de 1-2 unidades por día
    class SinControl:
        def act(self, u, r, m): pass
    nc, _ = run(SinControl(), Params(days=30), log_truth=True)
    dph = nc.groupby(nc.minute // MIN_PER_DAY).true_ph.last().diff().dropna()
    chk("Solución", "pH sin control: subida media (pH/día)", dph.mean(), 0.0, None, "gg", "{:.3f}")
    chk("Solución", "pH sin control: cambio máximo (pH/día)", dph.abs().max(), None, 1.0, "gg", "{:.3f}")

    # --- Cultivo -----------------------------------------------------------------------------------
    for _, r in comp.iterrows():
        chk("Cultivo", f"Peso fresco por planta, {r.tratamiento} (g)", r.rendimiento_g, 100, 350, "hd", "{:.0f}")
    l_kg = th.agua_l / (th.rendimiento_g * 10 / 1000)
    chk("Cultivo", "Agua repuesta por kg de lechuga (L/kg)", l_kg, 10, 45, "barbosa", "{:.1f}")

    # --- Sensores ----------------------------------------------------------------------------------
    t = d.temp_air.dropna()
    chk("Sensores", "DHT22 temperatura en pasos de 0,1 °C (% de lecturas)",
        100 * np.isclose(t * 10, np.round(t * 10), atol=1e-6).mean(), 99.9, None, "dht", "{:.1f}")
    chk("Sensores", "DHT22 lecturas fallidas (%)", 100 * d.temp_air.isna().mean(), None, 3, "dht", "{:.2f}")
    chk("Sensores", "BH1750 lux máximo (límite 65535)", d.lux.max(), None, 65535, "propio", "{:.0f}")
    chk("Sensores", "NPK en enteros (% de lecturas)",
        100 * (d.nitrogen.dropna() == np.round(d.nitrogen.dropna())).mean(), 99.9, None, "propio", "{:.1f}")

    tab = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    tab.to_csv(RES / "validacion_realismo.csv", index=False)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(tab[["grupo", "chequeo", "simulado", "referencia", "resultado"]].to_string(index=False))
    n_bad = (tab.resultado == "FUERA").sum()
    print(f"\n{len(tab) - n_bad} de {len(tab)} chequeos dentro del rango de referencia."
          f" Fuentes en resultados/validacion_realismo.csv")


if __name__ == "__main__":
    main()
