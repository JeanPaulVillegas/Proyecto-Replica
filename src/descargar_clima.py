"""Descarga clima horario REAL de Chennai (Open-Meteo, archivo histórico basado en ERA5).

No necesita librerías extra ni cuenta. Guarda data_reales/clima_chennai.csv, que el simulador
usa automáticamente en lugar del clima de respaldo.

Uso:  python src/descargar_clima.py                       # 2025-01-01, 62 días (temporada fresca)
      python src/descargar_clima.py --inicio 2025-11-01   # otra temporada
Datos: Open-Meteo (https://open-meteo.com), licencia CC BY 4.0.
"""
from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_reales" / "clima_chennai.csv"
VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "shortwave_radiation"]


def main(a):
    ini = date.fromisoformat(a.inicio)
    fin = ini + timedelta(days=a.dias - 1)
    q = dict(latitude=a.lat, longitude=a.lon, start_date=ini.isoformat(), end_date=fin.isoformat(),
             hourly=",".join(VARS), timezone="Asia/Kolkata")
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q)
    print("Descargando:", url)
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.load(r)
    h = d["hourly"]
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time"] + VARS)
        for i, t in enumerate(h["time"]):
            w.writerow([t] + [h[v][i] for v in VARS])
    temps = [x for x in h["temperature_2m"] if x is not None]
    hr = [x for x in h["relative_humidity_2m"] if x is not None]
    print(f"Guardado {OUT} · {len(h['time'])} horas")
    print(f"Temperatura: mín {min(temps):.1f} °C · media {sum(temps) / len(temps):.1f} °C · máx {max(temps):.1f} °C")
    print(f"Humedad relativa: mín {min(hr):.0f} % · media {sum(hr) / len(hr):.0f} % · máx {max(hr):.0f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", default="2025-01-01")
    ap.add_argument("--dias", type=int, default=62)
    ap.add_argument("--lat", type=float, default=13.08)
    ap.add_argument("--lon", type=float, default=80.27)
    main(ap.parse_args())
