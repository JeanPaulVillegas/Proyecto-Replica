"""Controladores de los tres tratamientos del artículo: manual, por umbrales e inteligente.

El inteligente usa la lógica de la sección 3.4 (Ec. 9 y 10): cada 15 min obtiene una
consigna/predicción de EC y pH, compara con lo medido, dosifica si |δ| ≥ ε y verifica a
los 5 min. En la fase 1 la predicción viene de un `predictor` intercambiable; el LSTM se
conecta en la fase 3.
"""
from __future__ import annotations

import numpy as np

from simulator import MIN_PER_DAY, ec_setpoint, ph_setpoint
from pipeline import ph_from_voltage, ec_compensate, PH_CAL


def _meas(reading: dict) -> tuple[float, float]:
    """Conversión a bordo (Ec. 1 y 2) usada por los controladores."""
    ph = ph_from_voltage(reading["ph_v"], *PH_CAL)
    ec = ec_compensate(reading["ec_raw"], reading["t_water"])
    return ec, ph


class _Base:
    v_total_l = 80.0
    c_solution = 200.0
    ml_acid_per_ph = 320.0  # mL de ácido 1:10 por unidad de pH en 80 L (supuesto; corrige ~80 % del desvío)

    cambio_semanal = False   # fase 5: cambio completo de solución periódico (solo manual y umbrales)
    dias_cambio = 30         # cada cuántos días se cambia la solución si cambio_semanal=True

    def __init__(self):
        self.buf_ec, self.buf_ph = [], []

    def _weekly_change(self, unit, minute, ec_target):
        per = int(self.dias_cambio * MIN_PER_DAY)
        if self.cambio_semanal and minute >= per and minute % per == 10 * 60:
            unit.replace_solution(ec_target)
            self.buf_ec, self.buf_ph = [], []

    def _smoothed(self, reading):
        if np.isnan(reading.get("ec_raw", np.nan)):
            return None
        ec, ph = _meas(reading)
        # Validación en el edge: descarta lecturas físicamente imposibles (picos)
        if 0.2 < ec < 5.0 and 3.0 < ph < 9.0:
            self.buf_ec = (self.buf_ec + [ec])[-10:]
            self.buf_ph = (self.buf_ph + [ph])[-10:]
        if len(self.buf_ec) < 5:
            return None   # espera a tener ventana suficiente tras una dosis
        # Mediana: equivalente robusto de la media móvil de 10 muestras (Ec. 3) con filtro de outliers
        return float(np.median(self.buf_ec)), float(np.median(self.buf_ph))

    def _dose_to(self, unit, ec_meas, ph_meas, ec_tgt, ph_tgt, eps_ec, eps_ph, error=0.0, rng=None):
        d_ec = ec_tgt - ec_meas
        dosed = abs(d_ec) >= eps_ec or abs(ph_tgt - ph_meas) >= eps_ph
        if dosed:
            # tras dosificar, la media móvil vieja ya no representa la solución
            self.buf_ec, self.buf_ph = [], []
        if d_ec >= eps_ec:
            ml = d_ec * self.v_total_l * 1000 / self.c_solution        # Ec. 10 (en mL)
            if error and rng is not None:
                ml *= 1 + rng.normal(0, error)
            unit.dose_ab(ml)
        elif d_ec <= -2 * eps_ec:
            # EC demasiado alta: descartar parte y reponer con agua
            unit.dilute(self.v_total_l * (-d_ec) / max(ec_meas, 0.1))
        d_ph = ph_tgt - ph_meas
        if abs(d_ph) >= eps_ph:
            ml = abs(d_ph) * self.ml_acid_per_ph
            if error and rng is not None:
                ml *= 1 + rng.normal(0, error)
            (unit.dose_acid if d_ph < 0 else unit.dose_base)(ml)

    def _climate_hysteresis(self, unit, reading):
        t = reading.get("temp_air", np.nan)
        if not np.isnan(t):
            if t > 27:
                unit.fan = True
            elif t < 25:
                unit.fan = False
        tw = reading.get("t_water", np.nan)
        if not np.isnan(tw):
            unit.heater = tw < 20


class ManualController(_Base):
    """Operador: mide con medidor portátil una vez al día (9:00) y corrige con error.
    Ventilador encendido a mano en horario fijo (11:00-17:00)."""

    def __init__(self, seed: int = 0, cambio_semanal: bool = False):
        super().__init__()
        self.rng = np.random.default_rng(seed + 777)
        self.cambio_semanal = cambio_semanal

    def act(self, unit, reading, minute):
        hour = (minute % MIN_PER_DAY) / 60
        unit.fan = 11 <= hour < 17
        unit.heater = False
        self._weekly_change(unit, minute, ec_setpoint(minute / MIN_PER_DAY))
        if minute % MIN_PER_DAY == 9 * 60:
            ec = unit.ec * (1 + self.rng.normal(0, 0.03))
            ph = unit.ph + self.rng.normal(0, 0.1)
            day = minute / MIN_PER_DAY
            self._dose_to(unit, ec, ph, ec_setpoint(day), ph_setpoint(day), 0.1, 0.15, error=0.25, rng=self.rng)


class ThresholdController(_Base):
    """Control por umbrales: consigna fija EC 1,8 ±0,2 mS/cm y pH 6,0 ±0,2, revisión cada 15 min,
    histéresis 27/25 °C para el ventilador."""

    ec_target, ph_target = 1.8, 6.0

    def __init__(self, cambio_semanal: bool = False):
        super().__init__()
        self.cambio_semanal = cambio_semanal

    def act(self, unit, reading, minute):
        self._weekly_change(unit, minute, self.ec_target)
        s = self._smoothed(reading)
        self._climate_hysteresis(unit, reading)
        if s is None or minute % 15:
            return
        ec, ph = s
        if abs(ec - self.ec_target) > 0.2 or abs(ph - self.ph_target) > 0.2:
            self._dose_to(unit, ec, ph, self.ec_target, self.ph_target, 0.2, 0.2)


class ScheduleForecaster:
    """Predictor provisional de la fase 1: devuelve la consigna del día (variante B).
    Se reemplaza por el LSTM en la fase 3 con la misma interfaz."""

    def predict(self, minute: int, history) -> tuple[float, float]:
        day = (minute + 15) / MIN_PER_DAY
        return ec_setpoint(day), ph_setpoint(day)


class SmartController(_Base):
    """Lazo de la sección 3.4: predicción cada 15 min, δ = ŷ − medido (Ec. 9), tolerancias
    ε = 0,2 mS/cm y 0,1 pH, dosis con Ec. 10, verificación a los 5 min."""

    def __init__(self, predictor=None):
        super().__init__()
        self.predictor = predictor or ScheduleForecaster()
        self.pending_check = None
        self.history = []

    def act(self, unit, reading, minute):
        self.history.append(reading)
        if len(self.history) > 120:
            self.history.pop(0)
        s = self._smoothed(reading)
        self._climate_hysteresis(unit, reading)
        if s is None:
            return
        ec, ph = s
        if minute % 15 == 0:
            ec_t, ph_t = self.predictor.predict(minute, self.history)
            self.target = (ec_t, ph_t)
            self._dose_to(unit, ec, ph, ec_t, ph_t, 0.2, 0.1)
            self.pending_check = minute + 5
        elif self.pending_check is not None and minute >= self.pending_check:
            ec_t, ph_t = self.target
            # ajuste fino con tolerancia más estrecha
            self._dose_to(unit, ec, ph, ec_t, ph_t, 0.1, 0.1)
            self.pending_check = None
