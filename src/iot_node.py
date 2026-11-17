"""Fase 4 · nodo ESP32 virtual.

Hace el papel del ESP32 del artículo: "lee" los sensores de la unidad simulada cada minuto
simulado, publica un JSON por MQTT (sección 3.1.2) y ejecuta las órdenes de dosificación que
le envía el edge. El clima (ventilador 27/25 °C, calentador, luz 16/8) se controla localmente,
como respaldo que funciona aunque se caiga la red.

Si el broker no está disponible, guarda las lecturas en un buffer local y las reenvía al
reconectar (el "MQTT buffer sync recovery" de la Tabla 5).

Uso:  python src/iot_node.py --velocidad 120 --dias 1
      (120 = 120 minutos simulados por segundo real... ver --velocidad)
"""
from __future__ import annotations

import argparse
import collections
import json
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from simulator import Params, HydroponicUnit, Sensors, MIN_PER_DAY

TOPIC = "hydro/{u}/{kind}"


class Node:
    def __init__(self, a):
        self.a = a
        p = Params(seed=a.semilla)
        rng = np.random.default_rng(a.semilla)
        self.unit = HydroponicUnit(p, rng)
        self.sensors = Sensors(p, np.random.default_rng(a.semilla + 10_000))
        self.buffer = collections.deque(maxlen=5000)
        self.lock = threading.Lock()
        self.sensor_off_until = 0.0
        self.connected = False
        self.disconnected_at = None
        self.stats = dict(reconexiones=[], registros_recuperados=[], ordenes=0)
        self.cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"esp32-{a.unidad}")
        self.cli.on_connect = self.on_connect
        self.cli.on_disconnect = self.on_disconnect
        self.cli.on_message = self.on_message
        self.cli.reconnect_delay_set(min_delay=1, max_delay=4)

    # --- MQTT -------------------------------------------------------------------------------
    def on_connect(self, cli, userdata, flags, rc, props=None):
        self.connected = True
        cli.subscribe(TOPIC.format(u=self.a.unidad, kind="cmd"), qos=1)
        cli.subscribe(TOPIC.format(u=self.a.unidad, kind="fault"), qos=1)
        with self.lock:
            pending = list(self.buffer)
            self.buffer.clear()
        for msg in pending:
            msg["buffered"] = True
            cli.publish(TOPIC.format(u=self.a.unidad, kind="sensors"), json.dumps(msg), qos=1)
        if self.disconnected_at is not None:
            dt = time.time() - self.disconnected_at
            self.stats["reconexiones"].append(round(dt, 2))
            self.stats["registros_recuperados"].append(len(pending))
            cli.publish(TOPIC.format(u=self.a.unidad, kind="status"),
                        json.dumps(dict(evento="reconectado", segundos=dt, recuperados=len(pending), t=time.time())), qos=1)
            print(f"[nodo] reconectado tras {dt:.1f} s · {len(pending)} registros reenviados desde el buffer")
            self.disconnected_at = None

    def on_disconnect(self, cli, userdata, flags, rc, props=None):
        if self.connected:
            self.disconnected_at = time.time()
            print("[nodo] conexión MQTT perdida: guardando lecturas en el buffer local")
        self.connected = False

    def on_message(self, cli, userdata, msg):
        d = json.loads(msg.payload)
        if msg.topic.endswith("/fault"):
            self.sensor_off_until = time.time() + float(d.get("sensor_off_s", 0))
            print(f"[nodo] falla inyectada: sensores desconectados {d.get('sensor_off_s')} s")
            return
        with self.lock:
            if d.get("ab_ml"):
                self.unit.dose_ab(d["ab_ml"])
            if d.get("acido_ml"):
                self.unit.dose_acid(d["acido_ml"])
            if d.get("base_ml"):
                self.unit.dose_base(d["base_ml"])
            if d.get("diluir_l"):
                self.unit.dilute(d["diluir_l"])
        self.stats["ordenes"] += 1
        # acuse de recibo con la marca de tiempo de la orden: el edge calcula la latencia
        cli.publish(TOPIC.format(u=self.a.unidad, kind="ack"),
                    json.dumps(dict(t_reading=d.get("t_reading"), t_cmd=d.get("t_cmd"), t_ack=time.time())), qos=1)

    # --- bucle principal -----------------------------------------------------------------------
    def climate(self, r):
        if not np.isnan(r["temp_air"]):
            if r["temp_air"] > 27:
                self.unit.fan = True
            elif r["temp_air"] < 25:
                self.unit.fan = False
        if not np.isnan(r["t_water"]):
            self.unit.heater = r["t_water"] < 20

    def run(self):
        # connect_async + loop_start: paho reintenta solo si el broker no está o se cae
        self.cli.connect_async(self.a.broker, self.a.puerto, keepalive=5)
        self.cli.loop_start()
        n = int(self.a.dias * MIN_PER_DAY)
        dt = 1.0 / self.a.velocidad
        t_next = time.time()
        print(f"[nodo {self.a.unidad}] simulando {self.a.dias} días a {self.a.velocidad} min simulados por segundo")
        for m in range(n):
            with self.lock:
                self.unit.step(m)
                r = self.sensors.read(self.unit, m)
                self.climate(r)
            if time.time() < self.sensor_off_until:
                r = {k: float("nan") for k in r}
            msg = dict(minute=m, t=time.time(), **{k: (None if np.isnan(v) else float(v)) for k, v in r.items()},
                       true_ec=self.unit.ec, true_ph=self.unit.ph, fan=bool(self.unit.fan), light=bool(self.unit.light))
            if self.connected:
                self.cli.publish(TOPIC.format(u=self.a.unidad, kind="sensors"), json.dumps(msg), qos=1)
            else:
                with self.lock:
                    self.buffer.append(msg)
            t_next += dt
            time.sleep(max(0.0, t_next - time.time()))
        self.cli.publish(TOPIC.format(u=self.a.unidad, kind="status"),
                         json.dumps(dict(evento="fin", t=time.time(), **self.stats)), qos=1)
        time.sleep(1)
        self.cli.loop_stop()
        print(f"[nodo] fin · {json.dumps(self.stats)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unidad", default="u1")
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--puerto", type=int, default=1883)
    ap.add_argument("--dias", type=float, default=1.0)
    ap.add_argument("--velocidad", type=float, default=60.0, help="minutos simulados por segundo real")
    ap.add_argument("--semilla", type=int, default=200)
    Node(ap.parse_args()).run()
