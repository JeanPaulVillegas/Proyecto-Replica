"""Fase 4 · demostración completa del IoT emulado con pruebas de falla.

Requiere los contenedores encendidos:   docker compose -f iot/docker-compose.yml up -d

Lanza el servicio edge y el nodo ESP32 virtual como procesos separados, y durante la corrida:
  1. desconecta los sensores 3 veces (5 s cada vez)       → detección y recuperación de sensor
  2. apaga el broker MQTT 10 s (docker stop/start)         → reconexión y recuperación del buffer
Al final imprime y guarda resultados/fase4_reporte.json (métricas tipo Tabla 5).

Uso:  python src/iot_demo.py                 # 2 días simulados, ~5 min
      python src/iot_demo.py --dias 5        # más largo, para ver más en Grafana
      python src/iot_demo.py --sin-fallas    # solo el flujo normal
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent


def docker(*args):
    r = subprocess.run(["docker", *args], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def main(a):
    ok, out = docker("inspect", "-f", "{{.State.Running}}", "replica-mosquitto")
    if not ok or "true" not in out:
        sys.exit("El broker no está corriendo. Primero: docker compose -f iot/docker-compose.yml up -d")
    py = sys.executable
    edge_args = [py, str(SRC / "edge.py")] + (["--sin-influx"] if a.sin_influx else [])
    print("Iniciando el edge (carga TensorFlow, unos segundos)…")
    edge = subprocess.Popen(edge_args)
    time.sleep(a.espera_edge)
    print("Iniciando el nodo ESP32 virtual…")
    node = subprocess.Popen([py, str(SRC / "iot_node.py"), "--dias", str(a.dias), "--velocidad", str(a.velocidad)])
    total = a.dias * 1440 / a.velocidad
    print(f"Duración aproximada: {total / 60:.1f} min. Grafana: http://localhost:3000")

    if not a.sin_fallas:
        cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pruebas-falla")
        cli.connect("localhost", 1883)
        cli.loop_start()
        t0 = time.time()

        def wait_until(frac):
            time.sleep(max(0.0, t0 + total * frac - time.time()))

        for frac in (0.45, 0.55, 0.65):
            wait_until(frac)
            print(f"\n>>> Falla 1: sensores desconectados 5 s")
            cli.publish("hydro/u1/fault", json.dumps({"sensor_off_s": 5}), qos=1)
        cli.loop_stop()
        cli.disconnect()
        wait_until(0.80)
        print("\n>>> Falla 2: broker MQTT apagado 10 s")
        docker("stop", "replica-mosquitto")
        time.sleep(10)
        docker("start", "replica-mosquitto")
        print(">>> broker encendido de nuevo")

    node.wait()
    try:
        edge.wait(timeout=30)
    except subprocess.TimeoutExpired:
        edge.terminate()
    rep = ROOT / "resultados" / "fase4_reporte.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        print("\n=== Comparación con la Tabla 5 del artículo (sistema inteligente) ===")
        rows = [
            ("Detección → acción", "3,2 s", f"{r['deteccion_accion_ms']['media']:.0f} ms" if r.get("deteccion_accion_ms") else "—"),
            ("Ejecución en el edge", "0,8 s", f"{r['procesamiento_edge_ms']['media']:.1f} ms" if r.get("procesamiento_edge_ms") else "—"),
            ("Inferencia LSTM (TFLite)", "~300 ms", f"{r['inferencia_tflite_ms']['media']:.2f} ms" if r.get("inferencia_tflite_ms") else "—"),
            ("Reconexión de sensor", "58 s", ", ".join(f"{e['desconectado_s']} s" for e in r.get("sensor_desconexiones", [])) or "—"),
            ("Reconexión MQTT (nodo)", "88 s", ", ".join(f"{x:.1f} s" for x in (r.get("reconexion_mqtt_nodo_s") or [])) or "—"),
            ("Registros recuperados del buffer", "94", ", ".join(map(str, r.get("registros_recuperados_buffer") or [])) or "—"),
            ("Disponibilidad", "98,7 %", f"{r['disponibilidad_pct']} %"),
        ]
        w = max(len(x[0]) for x in rows)
        print(f"{'Métrica':<{w}}  {'Artículo':>9}  Réplica")
        for k, p, v in rows:
            print(f"{k:<{w}}  {p:>9}  {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=float, default=2.0)
    ap.add_argument("--velocidad", type=float, default=10.0, help="minutos simulados por segundo real")
    ap.add_argument("--espera-edge", type=float, default=12.0)
    ap.add_argument("--sin-fallas", action="store_true")
    ap.add_argument("--sin-influx", action="store_true")
    main(ap.parse_args())
