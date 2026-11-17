"""Fase 4 · servicio edge (el papel de la Raspberry Pi 4 del artículo).

- Se suscribe a las lecturas del nodo por MQTT (Mosquitto).
- Guarda todo en InfluxDB (sensores, pronósticos, acciones, alertas, latencias).
- Cada 15 minutos simulados ejecuta el LSTM en TFLite y decide la dosificación con el mismo
  lazo de la fase 3 (Ec. 9 y 10); envía la orden al nodo por MQTT.
- Alertas por umbral (Ec. 11) y detección de sensor desconectado.
- Al terminar, escribe resultados/fase4_reporte.json con las métricas tipo Tabla 5.

Uso:  python src/edge.py                (requiere Mosquitto e InfluxDB de iot/docker-compose.yml)
      python src/edge.py --sin-influx   (solo MQTT, sin base de datos)
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

import numpy as np
import paho.mqtt.client as mqtt

from simulator import MIN_PER_DAY, ec_setpoint, ph_setpoint
from run_trial import LSTMForecaster, LSTMSmartController, load_tflite, DATA, RES

TOPIC = "hydro/{u}/{kind}"
ALERT_EC, ALERT_PH = 0.3, 0.3          # δ de la Ec. 11 (supuesto: más amplio que la tolerancia de control)
SENSOR_DOWN_AFTER = 3                  # lecturas vacías seguidas para declarar sensor desconectado


class RemoteUnit:
    """Se hace pasar por la unidad física para el controlador: junta las dosis en una orden MQTT."""

    def __init__(self):
        self.reset()
        self.fan = False
        self.heater = False

    def reset(self):
        self.cmd = dict(ab_ml=0.0, acido_ml=0.0, base_ml=0.0, diluir_l=0.0)

    def dose_ab(self, ml): self.cmd["ab_ml"] += max(0.0, ml)
    def dose_acid(self, ml): self.cmd["acido_ml"] += max(0.0, ml)
    def dose_base(self, ml): self.cmd["base_ml"] += max(0.0, ml)
    def dilute(self, l): self.cmd["diluir_l"] += max(0.0, l)

    def has_cmd(self):
        return any(v > 0 for v in self.cmd.values())


class Edge:
    def __init__(self, a, tf):
        self.a = a
        self.fc = LSTMForecaster(load_tflite(tf), DATA / "windows_500s_forecast.npz", tf)
        self.ctrl = LSTMSmartController(self.fc)
        self.unit = RemoteUnit()
        self.write = None
        if not a.sin_influx:
            from influxdb_client import InfluxDBClient, Point, WritePrecision
            from influxdb_client.client.write_api import WriteOptions
            self.Point, self.WP = Point, WritePrecision
            self.influx = InfluxDBClient(url=a.influx, token=a.token, org="replica")
            self.write = self.influx.write_api(write_options=WriteOptions(batch_size=500, flush_interval=1000))
        self.n_ok = self.n_total = 0
        self.max_minute = -1
        self.missing_run = 0
        self.sensor_down_since = None
        self.sensor_events = []          # (detección_s, recuperación_s)
        self.latencies, self.edge_ms, self.inference_ms = [], [], []
        self.alerts = 0
        self.node_stats = None
        self.done = False
        self.mqtt_disconnected_at = None
        self.mqtt_reconnects = []
        self.cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"edge-{a.unidad}")
        self.cli.on_connect = self.on_connect
        self.cli.on_disconnect = self.on_disconnect
        self.cli.on_message = self.on_message
        self.cli.reconnect_delay_set(min_delay=1, max_delay=4)

    def point(self, meas, t, **fields):
        if self.write is None:
            return
        p = self.Point(meas).tag("unidad", self.a.unidad).time(int(t * 1e9), self.WP.NS)
        for k, v in fields.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                p = p.field(k, float(v))
        self.write.write(bucket="hidroponia", record=p)

    # --- MQTT -------------------------------------------------------------------------------
    def on_connect(self, cli, userdata, flags, rc, props=None):
        for kind in ["sensors", "ack", "status"]:
            cli.subscribe(TOPIC.format(u=self.a.unidad, kind=kind), qos=1)
        if self.mqtt_disconnected_at is not None:
            self.mqtt_reconnects.append(time.time() - self.mqtt_disconnected_at)
            self.mqtt_disconnected_at = None
        print("[edge] conectado al broker MQTT")

    def on_disconnect(self, cli, userdata, flags, rc, props=None):
        if self.mqtt_disconnected_at is None:
            self.mqtt_disconnected_at = time.time()
            print("[edge] conexión MQTT perdida, reintentando…")

    def on_message(self, cli, userdata, msg):
        d = json.loads(msg.payload)
        kind = msg.topic.rsplit("/", 1)[-1]
        if kind == "ack":
            if d.get("t_reading"):
                lat = (d["t_ack"] - d["t_reading"]) * 1000
                self.latencies.append(lat)
                self.point("latencia", d["t_ack"], deteccion_accion_ms=lat)
            return
        if kind == "status":
            if d.get("evento") == "fin":
                self.node_stats = d
                self.done = True
            return
        self.on_reading(d)

    def on_reading(self, d):
        t_recv = time.time()
        m = d["minute"]
        reading = {k: (np.nan if d.get(k) is None else d[k])
                   for k in ["temp_air", "rh", "ph_v", "ec_raw", "t_water", "nitrogen", "lux", "co2"]}
        valid = not np.isnan(reading["ec_raw"])
        self.n_total += 1
        self.n_ok += valid
        self.max_minute = max(self.max_minute, m)
        # Conversión para guardar en unidades de ingeniería (Ec. 1 y 2)
        from pipeline import ph_from_voltage, ec_compensate, PH_CAL
        ph = float(ph_from_voltage(reading["ph_v"], *PH_CAL)) if valid else None
        ec = float(ec_compensate(reading["ec_raw"], reading["t_water"])) if valid else None
        self.point("sensores", d["t"], ec=ec, ph=ph, temp_air=reading["temp_air"], rh=reading["rh"],
                   t_water=reading["t_water"], nitrogen=reading["nitrogen"], lux=reading["lux"], co2=reading["co2"],
                   ec_real=d.get("true_ec"), ph_real=d.get("true_ph"))
        if d.get("buffered"):
            return        # los registros recuperados del buffer se guardan, pero no se usan para controlar

        # Sensor desconectado: detección y recuperación
        if not valid:
            self.missing_run += 1
            if self.missing_run == SENSOR_DOWN_AFTER:
                self.sensor_down_since = (d["t"], t_recv)
                print(f"[edge] ALERTA: sensores sin datos (minuto {m})")
                self.point("alertas", t_recv, sensor_desconectado=1)
        else:
            if self.sensor_down_since is not None:
                t_first_missing = self.sensor_down_since[0]
                self.sensor_events.append(dict(desconectado_s=round(t_recv - t_first_missing, 2)))
                print(f"[edge] sensores recuperados tras {t_recv - t_first_missing:.1f} s")
                self.sensor_down_since = None
            self.missing_run = 0

        # Control (sección 3.4): el controlador escribe las dosis en RemoteUnit
        self.unit.reset()
        n_before = self.fc.n_calls
        self.ctrl.act(self.unit, reading, m)
        if self.fc.n_calls > n_before:
            self.inference_ms.append(self.fc.last_ms)
            day = (m + 15) / MIN_PER_DAY
            ecf, phf = self.fc.last_forecast
            self.point("control", d["t"], ec_consigna=ec_setpoint(day), ph_consigna=ph_setpoint(day),
                       ec_pronostico=ecf, ph_pronostico=phf)
            self.point("latencia", t_recv, inferencia_ms=self.fc.last_ms)
        if self.unit.has_cmd():
            t_cmd = time.time()
            self.edge_ms.append((t_cmd - t_recv) * 1000)
            self.cli.publish(TOPIC.format(u=self.a.unidad, kind="cmd"),
                             json.dumps(dict(**self.unit.cmd, t_reading=d["t"], t_cmd=t_cmd)), qos=1)
            self.point("acciones", t_cmd, ab_ml=self.unit.cmd["ab_ml"], acido_ml=self.unit.cmd["acido_ml"],
                       base_ml=self.unit.cmd["base_ml"])

        # Alertas por umbral (Ec. 11)
        if valid:
            day = m / MIN_PER_DAY
            dev_ec, dev_ph = abs(ec - ec_setpoint(day)), abs(ph - ph_setpoint(day))
            if dev_ec > ALERT_EC or dev_ph > ALERT_PH:
                self.alerts += 1
                self.point("alertas", d["t"], ec_dev=dev_ec, ph_dev=dev_ph)

    # --- informe ----------------------------------------------------------------------------
    def report(self):
        def stat(x):
            return dict(media=round(float(np.mean(x)), 3), p95=round(float(np.percentile(x, 95)), 3), n=len(x)) if x else None
        ns = self.node_stats or {}
        expected = self.max_minute + 1
        rep = dict(
            minutos_simulados=expected,
            lecturas_recibidas=self.n_total,
            disponibilidad_pct=round(100 * self.n_ok / expected, 2) if expected > 0 else None,
            deteccion_accion_ms=stat(self.latencies),
            procesamiento_edge_ms=stat(self.edge_ms),
            inferencia_tflite_ms=stat(self.inference_ms),
            sensor_desconexiones=self.sensor_events,
            reconexion_mqtt_nodo_s=ns.get("reconexiones"),
            registros_recuperados_buffer=ns.get("registros_recuperados"),
            reconexion_mqtt_edge_s=[round(x, 2) for x in self.mqtt_reconnects],
            ordenes_ejecutadas=ns.get("ordenes"),
            alertas=self.alerts,
        )
        RES.mkdir(exist_ok=True)
        (RES / "fase4_reporte.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print("\n=== Fase 4 · métricas tipo Tabla 5 ===")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return rep

    def run(self):
        self.cli.connect_async(self.a.broker, self.a.puerto, keepalive=5)
        self.cli.loop_start()
        print("[edge] esperando lecturas del nodo… (Ctrl+C para terminar)")
        try:
            while not self.done:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        time.sleep(1)
        if self.write is not None:
            self.write.flush()
            self.write.close()
        self.cli.loop_stop()
        return self.report()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unidad", default="u1")
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--puerto", type=int, default=1883)
    ap.add_argument("--influx", default="http://localhost:8086")
    ap.add_argument("--token", default="replica-token-local")
    ap.add_argument("--sin-influx", action="store_true")
    a = ap.parse_args()
    import tensorflow as tf
    Edge(a, tf).run()
