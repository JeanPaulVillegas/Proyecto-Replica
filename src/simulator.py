"""Gemelo digital de una unidad hidropónica DWC de lechuga (réplica virtual de Devaraju et al., 2026).

Paso de 1 minuto. Modela clima, luz, CO2, crecimiento de 10 plantas, agua, EC, pH,
temperatura del agua y los sensores del artículo (con ruido, deriva, picos y huecos).

Todos los parámetros son SUPUESTOS de la réplica (el artículo no los da); están
agrupados en `Params` para poder hacer análisis de sensibilidad.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from clima import cargar_clima_exterior, presion_saturacion_kpa

MIN_PER_DAY = 1440


@dataclass
class Params:
    days: int = 60
    n_plants: int = 10
    # Agua y nutrientes
    v_nominal_l: float = 80.0          # volumen del depósito (supuesto, 50-100 L)
    v_refill_trigger_l: float = 76.0   # la electroválvula repone agua bajo este nivel
    ec_refill_water: float = 0.1       # EC del agua de reposición (mS/cm)
    c_solution: float = 200.0          # "concentración" de la mezcla A+B (mS/cm equivalentes, Ec. 10)
    uptake_ec_l_per_g: float = 0.035   # mS/cm·L de nutrientes absorbidos por g de biomasa fresca
    et_l_per_g_day: float = 0.0021     # L/día de evapotranspiración por g de biomasa (luz encendida)
    # pH
    ph_rise_per_ecl: float = 0.05      # subida de pH por absorción de nitrato (calibrado: el pH sube y se corrige casi solo con ácido)
    ph_drift_per_day: float = 0.01     # deriva química aleatoria (pequeña: la tendencia real es al alza por nitrato)
    acid_ph_per_ml_per_l: float = 0.02 # bajada de pH por mL de ácido por litro de solución
    base_ph_per_ml_per_l: float = 0.02
    dose_ph_raise_per_ml_per_l: float = 0.0004  # efecto secundario de A+B sobre el pH (leve acidez)
    # Crecimiento (logístico, solo con luz)
    w0_g: float = 5.0
    wmax_g: float = 430.0
    wmax_cv: float = 0.10
    r_per_light_day: float = 0.175
    r_cv: float = 0.06
    # Clima: exterior REAL de Chennai (data_reales/clima_chennai.csv) o normales climáticas (clima.py)
    start_date: str = "2025-01-01"     # temporada fresca de Chennai (ene-feb), la más apta para lechuga
    clima_seed: int = 0                # el mismo clima para todos los tratamientos (corren a la vez en el mismo cuarto)
    tau_building_h: float = 4.0        # inercia térmica del edificio (amortigua y retrasa el ciclo diario)
    internal_gain_c: float = 0.5       # calor interno del cuarto (equipos, personas)
    t_fan_target: float = 23.5
    fan_cooling_rate: float = 0.04     # fracción por minuto hacia t_fan_target
    fan_max_delta_c: float = 3.0       # capacidad limitada: no enfría más de 3 °C bajo el cuarto
    room_relax_rate: float = 0.02
    led_heat_c: float = 1.2            # calor que agregan las luces encendidas
    transp_vapor_kpa: float = 0.25     # vapor que agregan las plantas adultas al aire del cuarto
    ac_setpoint_c: float | None = None # fase 5: cuarto climatizado (None = sin aire acondicionado)
    ac_rh_max: float = 65.0            # el aire acondicionado deshumidifica hasta este límite
    crecimiento_sigue_consigna: bool = False  # fase 5 (escenario favorable): el óptimo es la consigna del día
    # Luz
    light_on_hour: int = 6
    photoperiod_h: int = 16
    lux_on: float = 18000.0
    # CO2
    co2_ambient: float = 820.0         # cuarto cerrado y ocupado (exterior ~425 ppm)
    # Potencias nominales (W) para el cálculo de energía
    p_led_w: float = 60.0
    p_fan_w: float = 12.0
    p_heater_w: float = 50.0
    p_pump_recirc_w: float = 8.0
    p_dosing_pump_w: float = 6.0
    dosing_ml_per_min: float = 50.0
    # Sensores (precisión de la tabla del artículo)
    noise_temp: float = 0.25           # ±0.5 °C -> sigma ≈ 0.25
    noise_rh: float = 1.5              # ±2-5 %
    noise_ph: float = 0.05             # ±0.1 pH
    noise_ec_rel: float = 0.01         # ±2 % FS
    noise_n_rel: float = 0.025         # ±5 %
    noise_lux_rel: float = 0.10        # ±20 %
    noise_co2: float = 15.0
    ph_sensor_drift_per_week: float = 0.03
    spike_prob: float = 0.0008         # tasa base de fallas puntuales (cada sensor tiene su modo de falla real)
    dht_fail_prob: float = 0.005       # DHT22: lectura fallida por checksum/tiempo (devuelve NaN)
    gap_prob: float = 0.0015
    gap_max_min: int = 12
    # pH sensor eléctrico: pH = a*V + b (Ec. 1); valores "verdaderos" de la sonda
    ph_probe_slope: float = -5.70      # pH por voltio
    ph_probe_offset: float = 21.34
    seed: int = 0
    extra: dict = field(default_factory=dict)


def ec_setpoint(day: float) -> float:
    """Consigna de EC por día de cultivo (interpolación de la Tabla 1 del artículo)."""
    d = np.array([0, 10, 20, 30, 40, 50, 60])
    ec = np.array([1.5, 1.6, 1.7, 1.8, 1.8, 1.9, 2.0])
    return float(np.interp(day, d, ec))


def ph_setpoint(day: float) -> float:
    d = np.array([0, 10, 20, 30, 40, 50, 60])
    ph = np.array([6.0, 6.0, 6.0, 6.1, 6.2, 6.2, 6.3])
    return float(np.interp(day, d, ph))


def _plateau(x, lo, hi, width):
    """1 dentro de [lo, hi]; decae como gaussiana fuera."""
    if x < lo:
        return float(np.exp(-((x - lo) / width) ** 2))
    if x > hi:
        return float(np.exp(-((x - hi) / width) ** 2))
    return 1.0


class HydroponicUnit:
    """Estado físico "verdadero" de la unidad. Los controladores actúan por métodos dose_* y set_*."""

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.v = p.v_nominal_l
        self.ec = ec_setpoint(0)
        self.nutr = self.ec * self.v          # "masa" de nutrientes en mS/cm·L
        self.ph = 6.0
        clima, self.clima_origen = cargar_clima_exterior(p.start_date, p.days, p.clima_seed)
        self._t_out_h = clima.temperature_2m.to_numpy(float)
        self._e_out_h = presion_saturacion_kpa(clima.dew_point_2m.to_numpy(float))
        self.t_out, self.rh_out = self._t_out_h[0], float(clima.relative_humidity_2m.iloc[0])
        self.t_bld = float(self._t_out_h[:24].mean())       # edificio arranca en la media del primer día
        self.e_bld = float(self._e_out_h[:24].mean())
        self.t_air = self.t_bld + p.internal_gain_c
        self.t_water = self.t_air - 1.0
        self.rh = float(100 * self.e_bld / presion_saturacion_kpa(self.t_air))
        self.co2 = p.co2_ambient
        self.w = np.full(p.n_plants, p.w0_g) * rng.normal(1, 0.05, p.n_plants)
        self.wmax = p.wmax_g * rng.normal(1, p.wmax_cv, p.n_plants)
        self.r_plant = p.r_per_light_day * rng.normal(1, p.r_cv, p.n_plants)
        self.fan = False
        self.heater = False
        self.light = False
        # contabilidad
        self.dosed_ab_ml = 0.0
        self.dosed_acid_ml = 0.0
        self.dosed_base_ml = 0.0
        self.water_added_l = 0.0
        self.water_discarded_l = 0.0
        self.nutr_discarded = 0.0
        self.energy_wh = 0.0
        self.pump_on_min = 0.0
        self.light_on_min = 0.0
        self.ph_walk = 0.0

    # --- acciones -----------------------------------------------------------
    def dose_ab(self, ml: float):
        ml = max(0.0, ml)
        self.nutr += ml / 1000.0 * self.p.c_solution
        self.v += ml / 1000.0
        self.ph -= self.p.dose_ph_raise_per_ml_per_l * ml / self.v * 10
        self.dosed_ab_ml += ml
        self._pump_time(ml)

    def dose_acid(self, ml: float):
        ml = max(0.0, ml)
        self.ph -= self.p.acid_ph_per_ml_per_l * ml / self.v * 10
        self.dosed_acid_ml += ml
        self._pump_time(ml)

    def dose_base(self, ml: float):
        ml = max(0.0, ml)
        self.ph += self.p.base_ph_per_ml_per_l * ml / self.v * 10
        self.dosed_base_ml += ml
        self._pump_time(ml)

    def dilute(self, liters: float):
        """Descarta solución y repone con agua (corrige EC alta). Cuenta como desperdicio."""
        liters = min(max(0.0, liters), self.v * 0.5)
        frac = liters / self.v
        self.nutr_discarded += self.nutr * frac
        self.nutr *= (1 - frac)
        self.water_discarded_l += liters
        self.nutr += liters * self.p.ec_refill_water
        self.water_added_l += liters
        self.pump_on_min += liters / 5.0  # bomba de 5 L/min

    def replace_solution(self, ec_target: float, ph_target: float = 6.0):
        """Cambio completo de solución (práctica común en sistemas manuales): se descarta todo y se
        prepara solución nueva con agua y concentrado A+B hasta la EC objetivo."""
        self.nutr_discarded += self.nutr
        self.water_discarded_l += self.v
        self.water_added_l += self.p.v_nominal_l
        self.v = self.p.v_nominal_l
        self.nutr = self.v * self.p.ec_refill_water
        need = max(0.0, ec_target * self.v - self.nutr)
        ml = need / self.p.c_solution * 1000
        self.nutr += need
        self.dosed_ab_ml += ml
        self.ph = ph_target
        self.ec = self.nutr / self.v
        self.pump_on_min += self.v / 5.0

    def _pump_time(self, ml: float):
        minutes = ml / self.p.dosing_ml_per_min
        self.pump_on_min += minutes
        self.energy_wh += self.p.p_dosing_pump_w * minutes / 60

    # --- dinámica -----------------------------------------------------------
    def step(self, minute: int):
        p, rng = self.p, self.rng
        day = minute / MIN_PER_DAY
        hour = (minute % MIN_PER_DAY) / 60
        self.light = p.light_on_hour <= hour < p.light_on_hour + p.photoperiod_h

        # Clima exterior (horario, interpolado al minuto)
        i, f = minute // 60, (minute % 60) / 60
        self.t_out = self._t_out_h[i] * (1 - f) + self._t_out_h[i + 1] * f
        e_out = self._e_out_h[i] * (1 - f) + self._e_out_h[i + 1] * f
        self.rh_out = float(100 * e_out / presion_saturacion_kpa(self.t_out))
        # Edificio: filtro de primer orden (amortigua y retrasa la onda diaria)
        k = 1 / (p.tau_building_h * 60)
        self.t_bld += k * (self.t_out - self.t_bld)
        self.e_bld += k * (e_out - self.e_bld)
        t_eq = self.t_bld + p.internal_gain_c + (p.led_heat_c if self.light else 0.0)
        if p.ac_setpoint_c is not None:   # el equipo retira el calor; queda algo del de las luces
            t_eq = p.ac_setpoint_c + (0.3 * p.led_heat_c if self.light else 0.0)
        self.t_air += p.room_relax_rate * (t_eq - self.t_air) + rng.normal(0, 0.03)
        if self.fan:   # enfriamiento con capacidad limitada
            target = max(p.t_fan_target, t_eq - p.fan_max_delta_c)
            self.t_air += p.fan_cooling_rate * (target - self.t_air)
        # Humedad: vapor del aire exterior (vía edificio) + transpiración; el ventilador renueva aire
        transp = self.w.sum() / (p.n_plants * p.wmax_g)
        e_in = self.e_bld + p.transp_vapor_kpa * transp * (1.0 if self.light else 0.3) * (0.5 if self.fan else 1.0)
        rh_eq = 100 * e_in / presion_saturacion_kpa(self.t_air)
        if p.ac_setpoint_c is not None:
            rh_eq = min(rh_eq, p.ac_rh_max)
        self.rh += 0.05 * (rh_eq - self.rh) + rng.normal(0, 0.1)
        self.rh = float(np.clip(self.rh, 20, 99))

        # Agua (sigue al aire con retardo; el calentador la sube)
        self.t_water += (self.t_air - 1.0 - self.t_water) / 360 + (0.01 if self.heater else 0.0)  # evaporación: ~1 °C bajo el aire

        # CO2: baja con fotosíntesis
        uptake_co2 = 120 * transp if self.light else -20
        self.co2 += 0.05 * (p.co2_ambient - uptake_co2 - self.co2) + rng.normal(0, 2)

        # Crecimiento logístico con factores de estrés
        if self.light:
            r = self.r_plant / MIN_PER_DAY
            f_t = np.exp(-((self.t_air - 23.0) / 7.0) ** 2)
            # Óptimos agronómicos FIJOS (no dependen de la consigna de ningún controlador,
            # para no favorecer por diseño a un tratamiento): meseta y caída gaussiana fuera
            f_ec = _plateau(self.ec, 1.4, 2.2, 0.35)
            f_ph = _plateau(self.ph, 5.6, 6.4, 0.45)
            if p.crecimiento_sigue_consigna:
                # SUPUESTO FAVORABLE AL ARTÍCULO (sin evidencia): crece mejor cuanto más cerca esté de la
                # consigna exacta de la Tabla 1. Solo se usa en el escenario "favorable" de la fase 5.
                f_ec = float(np.exp(-((self.ec - ec_setpoint(day)) / p.extra.get("ancho_ec", 0.35)) ** 2))
                f_ph = float(np.exp(-((self.ph - ph_setpoint(day)) / p.extra.get("ancho_ph", 0.30)) ** 2))
            f_rh = _plateau(self.rh, 50, 70, 20)          # 50-70 % recomendado (Cornell CEA)
            dw = r * self.w * (1 - self.w / self.wmax) * f_t * f_ec * f_ph * f_rh
            dw = np.maximum(dw, 0)
            self.w += dw
            # Absorción de nutrientes y subida de pH por nitrato
            upt = p.uptake_ec_l_per_g * dw.sum()
            self.nutr = max(self.nutr - upt, 0.0)
            self.ph += p.ph_rise_per_ecl * upt / self.v * 80
            # Evapotranspiración
            vpd_factor = 1 + 0.02 * (self.t_air - 24) - 0.01 * (self.rh - 60)
            et = p.et_l_per_g_day / MIN_PER_DAY * self.w.sum() * max(vpd_factor, 0.3)
            self.v -= et
            self.light_on_min += 1
        # Deriva química lenta del pH (paseo aleatorio acotado)
        self.ph_walk = 0.999 * self.ph_walk + rng.normal(0, p.ph_drift_per_day / 40)
        self.ph += self.ph_walk / MIN_PER_DAY * 10

        # Reposición de agua por electroválvula
        if self.v < p.v_refill_trigger_l:
            add = p.v_nominal_l - self.v
            self.v += add
            self.nutr += add * p.ec_refill_water
            self.water_added_l += add
        self.ec = self.nutr / self.v
        self.ph = float(np.clip(self.ph, 3.5, 9.0))       # límites químicos de una solución nutritiva

        # Energía
        e = p.p_pump_recirc_w + (p.p_led_w if self.light else 0) + (p.p_fan_w if self.fan else 0) \
            + (p.p_heater_w if self.heater else 0)
        self.energy_wh += e / 60


class Sensors:
    """Lecturas crudas del ESP32 según la tabla de sensores del artículo."""

    def __init__(self, p: Params, rng: np.random.Generator):
        self.p, self.rng = p, rng
        self.gap_left = 0

    def read(self, u: HydroponicUnit, minute: int) -> dict:
        p, rng = self.p, self.rng
        week = minute / (7 * MIN_PER_DAY)
        # deriva de la sonda; se recalibra cada 2 semanas (supuesto del plan)
        ph_meas = u.ph + rng.normal(0, p.noise_ph) + p.ph_sensor_drift_per_week * (week % 2)
        # La sonda entrega voltaje; se convierte con la calibración de la Ec. 1 en el pipeline
        ph_v = (ph_meas - p.ph_probe_offset) / p.ph_probe_slope
        # La sonda EC mide a la temperatura del agua (sin compensar), Ec. 2 se aplica en el pipeline
        ec_raw = u.ec * (1 + 0.019 * (u.t_water - 25)) * (1 + rng.normal(0, p.noise_ec_rel))
        n_mgl = 140 * u.ec / 1.8 * (1 + rng.normal(0, p.noise_n_rel))
        q = lambda x, step: float(np.round(x / step) * step)   # resolución del sensor / ADC
        adc_lsb = 3.3 / 4095                                     # ESP32: ADC de 12 bits a 3,3 V
        r = dict(
            temp_air=q(u.t_air + rng.normal(0, p.noise_temp), 0.1),            # DHT22: 0,1 °C
            rh=q(float(np.clip(u.rh + rng.normal(0, p.noise_rh), 0, 100)), 0.1),  # DHT22: 0,1 %
            ph_v=q(ph_v, adc_lsb),
            ec_raw=max(0.0, q(ec_raw, 0.01)),                                   # módulo EC: 0,01 mS/cm
            t_water=q(u.t_water + rng.normal(0, 0.1), 0.0625),                  # DS18B20: 0,0625 °C
            nitrogen=float(max(0, round(n_mgl))),                               # NPK RS485: entero (mg/kg)
            lux=float(min(65535, max(0, round((p.lux_on if u.light else 5.0) * (1 + rng.normal(0, p.noise_lux_rel)))))),  # BH1750
            co2=float(round(u.co2 + rng.normal(0, p.noise_co2))),               # NDIR: 1 ppm
        )
        # Fallas puntuales con los modos de falla reales de cada sensor (en vez de multiplicar por 3)
        if rng.random() < p.dht_fail_prob:                     # DHT22: checksum/tiempo agotado
            r["temp_air"] = r["rh"] = np.nan
        if rng.random() < p.spike_prob:                         # burbuja de aire en la celda EC: lee bajo
            r["ec_raw"] = q(r["ec_raw"] * rng.uniform(0.3, 0.7), 0.01)
        if rng.random() < p.spike_prob:                         # interferencia eléctrica en la sonda de pH
            r["ph_v"] = q(r["ph_v"] + rng.choice([-1, 1]) * rng.uniform(0.3, 1.0) / abs(p.ph_probe_slope), adc_lsb)
        if rng.random() < p.spike_prob / 2:                     # DS18B20: 85,0 °C = valor de reinicio
            r["t_water"] = 85.0
        if rng.random() < p.spike_prob:                         # NPK: trama RS485 corrupta
            r["nitrogen"] = np.nan
        if rng.random() < p.spike_prob / 2:                     # CO2: alguien respira cerca del sensor
            r["co2"] = r["co2"] + float(round(rng.uniform(100, 400)))
        # Huecos de comunicación (todas las variables a NaN)
        if self.gap_left == 0 and rng.random() < p.gap_prob:
            self.gap_left = int(rng.integers(1, p.gap_max_min + 1))
        if self.gap_left > 0:
            self.gap_left -= 1
            r = {k: np.nan for k in r}
        return r


def run(controller, p: Params | None = None, seed: int | None = None, log_truth: bool = True) -> tuple[pd.DataFrame, dict]:
    """Corre una simulación de p.days días con el controlador dado.

    `controller` es un objeto con `act(unit, reading, minute)` (ver controllers.py).
    Devuelve (DataFrame minuto a minuto con lecturas y estado verdadero, resumen).
    """
    p = p or Params()
    seed = p.seed if seed is None else seed
    rng = np.random.default_rng(seed)
    unit = HydroponicUnit(p, rng)
    sensors = Sensors(p, np.random.default_rng(seed + 10_000))
    n = p.days * MIN_PER_DAY
    rows = []
    for m in range(n):
        unit.step(m)
        reading = sensors.read(unit, m)
        controller.act(unit, reading, m)
        row = {"minute": m, **reading}
        if log_truth:
            row.update(
                true_ec=unit.ec, true_ph=unit.ph, true_t_air=unit.t_air, true_rh=unit.rh,
                t_out=unit.t_out, rh_out=unit.rh_out, true_co2=unit.co2,
                true_t_water=unit.t_water, volume_l=unit.v, biomass_mean_g=float(unit.w.mean()),
                fan=unit.fan, heater=unit.heater, light=unit.light,
                ec_target=ec_setpoint(m / MIN_PER_DAY), ph_target=ph_setpoint(m / MIN_PER_DAY),
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.Timestamp(p.start_date) + pd.to_timedelta(df["minute"], unit="min")
    summary = dict(
        controller=type(controller).__name__, seed=seed, clima=unit.clima_origen,
        yield_g_mean=float(unit.w.mean()), yield_g_std=float(unit.w.std(ddof=1)),
        yield_cv_pct=float(unit.w.std(ddof=1) / unit.w.mean() * 100),
        dosed_ab_ml=unit.dosed_ab_ml, dosed_acid_ml=unit.dosed_acid_ml, dosed_base_ml=unit.dosed_base_ml,
        nutr_discarded=unit.nutr_discarded, water_added_l=unit.water_added_l,
        water_discarded_l=unit.water_discarded_l, energy_kwh=unit.energy_wh / 1000,
        pump_min_per_day=unit.pump_on_min / p.days, light_h_per_day=unit.light_on_min / 60 / p.days,
    )
    return df, summary
