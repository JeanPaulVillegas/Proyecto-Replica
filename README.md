# Réplica virtual: Devaraju et al. (2026), hidroponía IoT + LSTM

Este proyecto reproduce en simulación el sistema del artículo *Smart hydroponic farming system using IoT-integrated deep learning for real-time nutrient monitoring and climate control* (Ecohydrology & Hydrobiology 26, 2026, 100754).

Plan completo de la réplica: https://claude.ai/artifact/8KZiKBegcz9dp7hKJX8e1r

La réplica avanza por fases. Cada fase trae su código y sus instrucciones. Tú ejecutas cada fase en tu computadora y compruebas que tus resultados coincidan con los de la carpeta `resultados_esperados/`.

## 0. Datos realistas: clima de Chennai y validación (actualización del 13-11-2026)

El simulador ya no usa un clima inventado. Ahora:

- **Clima exterior de Chennai.** Si existe `data_reales/clima_chennai.csv`, usa ese clima horario real. Si no, usa un respaldo armado con las normales climáticas de Chennai (temperatura media de la JMA, máximas y mínimas medias, punto de rocío aproximado). Por defecto simula del 1 de enero al 1 de marzo de 2025, la temporada fresca, que es la más apta para lechuga.
- **Clima del cuarto a partir del exterior.** El edificio amortigua y retrasa la onda diaria (constante de 4 h), las luces suman calor, el ventilador enfría como máximo 3 °C, y la humedad sale del vapor del aire exterior más la transpiración de las plantas.
- **pH que sube con la absorción de nitrato**, como en un cultivo real: se corrige casi solo con ácido.
- **Sensores con su resolución real** (DHT22 0,1 °C; ADC de 12 bits del ESP32; DS18B20 0,0625 °C; NPK en enteros; BH1750 hasta 65 535 lux) y **sus modos de falla reales**: lectura fallida del DHT22, burbuja en la celda EC, interferencia en la sonda de pH, el 85,0 °C del DS18B20 al reiniciarse, trama corrupta del NPK.
- Límites físicos (pH entre 3,5 y 9; HR entre 20 y 99 %).

**Paso 1 (opcional, recomendado): descargar el clima real.** Necesita internet, no necesita librerías extra:

```
python src\descargar_clima.py
```

Baja 62 días de temperatura, humedad, punto de rocío y radiación horarios de Chennai desde Open-Meteo (archivo histórico basado en ERA5, licencia CC BY 4.0) y los guarda en `data_reales\clima_chennai.csv`. Muestra al final la mínima, media y máxima: para enero y febrero deberías ver unos 18–21 °C de mínima y 31–34 °C de máxima. Para otra temporada: `python src\descargar_clima.py --inicio 2025-11-01`. Si lo descargas, cambia también `start_date` en `Params` (simulator.py) a la misma fecha.

**Paso 2: regenerar la fase 1 y validar.**

```
python src\generate_dataset.py
python src\validar_realismo.py
```

`validar_realismo.py` compara cada variable con un rango de referencia y escribe `resultados\validacion_realismo.csv` con la fuente de cada rango. Con el clima de respaldo da **32 de 32 chequeos dentro de rango** (`resultados_esperados\validacion_realismo.csv`). Las fuentes son:

- Clima de Chennai: [normales de la JMA](https://ds.data.jma.go.jp/gmd/tcc/tcc/products/climate/normal/parts/NrmMonth_e.php?stn=43279) y [máximas y mínimas medias](https://weather2travel.com/india/chennai/climate).
- Lechuga: [Cornell CEA, aire 24/19 °C, HR 50–70 %, oxígeno disuelto](https://bpb-us-e1.wpmucdn.com/blogs.cornell.edu/dist/8/8824/files/2019/06/GPNOctober2016_LeafyGreens.pdf); [pH 5,5–6,5, EC 1,0–2,0 y tendencia del pH a subir](https://www.greenhousegrower.com/crops/varieties/manage-ph-and-soluble-salts-in-hydroponics/); [peso fresco en DWC](https://www.hortidaily.com/article/9878963/evaluating-lettuce-production-in-hydroponic-systems/); [agua por kg de lechuga hidropónica](https://pmc.ncbi.nlm.nih.gov/articles/PMC4483736/).
- Sensores: [hoja de datos del DHT22](https://cdn.sparkfun.com/assets/f/7/d/9/c/DHT22.pdf).

Algunos rangos son criterio físico de la réplica (por ejemplo, que el aire no cambie más de 0,5 °C en un minuto); el CSV lo indica.

**Importante:** como cambió el simulador, hay que volver a correr las fases 1, 2 y 3 en ese orden. Los resultados esperados de abajo ya están actualizados.

| Fase | Estado | Qué produce |
| --- | --- | --- |
| 1. Simulador y pipeline | Hecha (actualizada: datos realistas, sección 0) | Dataset sintético de 60 días y ventanas 60 × 6 |
| 2. Modelo LSTM y comparaciones | Hecha | Métricas RMSE, MAPE y R² |
| 3. Control en lazo cerrado | Hecha | Ensayo virtual con 3 tratamientos |
| 4. IoT emulado | Lista para ejecutar | MQTT, InfluxDB, Grafana, pruebas de falla |
| 5. Escenarios: realista vs. favorable al artículo | Lista para ejecutar | Qué supuestos hacen falta para obtener las cifras del artículo |
| 6. Informe | Pendiente | Informe final de la réplica |

---

## 1. Requisitos

- Python 3.10, 3.11 o 3.12 (comprueba tu versión con `python --version`). Si no lo tienes, descárgalo de https://www.python.org/downloads/ y, en Windows, marca la casilla "Add Python to PATH" durante la instalación.
- Unos 200 MB libres en disco.
- Ninguna GPU por ahora. La fase 2 (LSTM) instalará TensorFlow aparte.

## 2. Instalación (solo la primera vez)

Abre una terminal **dentro de la carpeta del proyecto** (la que contiene este README).

**Windows (PowerShell o CMD):**

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Sabrás que el entorno está activo porque la línea de la terminal empieza con `(.venv)`. Cada vez que abras una terminal nueva, repite solo la línea de `activate`.

## 3. Ejecutar la fase 1

Con el entorno activo y desde la carpeta del proyecto:

```
python src/generate_dataset.py --seed 0
```

Tarda entre 30 segundos y 2 minutos. Simula 60 días minuto a minuto (86 400 pasos) tres veces: una por cada controlador.

## 4. Qué deberías obtener

Se crean dos carpetas:

| Archivo | Qué es |
| --- | --- |
| `data/raw_threshold_1min.csv.gz` | 86 400 lecturas crudas de sensores más el estado "real" de la unidad (control por umbrales) |
| `data/raw_threshold_500s.csv.gz` | Remuestreo cada 500 s: exactamente 10 368 filas, la cifra del artículo |
| `data/clean_*.csv.gz` | Datos después del preprocesamiento (Ec. 1 a 5) |
| `data/windows_*_forecast.npz` | Ventanas 60 × 6 con objetivo = EC y pH 15 min después (variante A) |
| `data/windows_*_setpoint.npz` | Ventanas 60 × 6 con objetivo = consigna del día (variante B) |
| `data/chequeo_tres_controladores.csv` | Resumen de los tres tratamientos |
| `data/reporte_fase1.json` | Parámetros usados, outliers, tamaños de los conjuntos, baseline de persistencia |
| `figs/fase1_series.png` | Gráfico de EC, pH, clima y biomasa |

**Comprobación:** con `--seed 0`, tu `data/chequeo_tres_controladores.csv` debería coincidir con `resultados_esperados/chequeo_tres_controladores.csv` (pueden aparecer diferencias mínimas en el último decimal según la versión de numpy). Los valores clave son:

| Tratamiento | Rendimiento (g/planta) | Error medio EC (mS/cm) | Error medio pH |
| --- | --- | --- | --- |
| Manual | 195,1 | 0,044 | 0,088 |
| Umbrales | 230,7 | 0,140 | 0,100 |
| Inteligente (provisional) | 230,7 | 0,036 | 0,029 |

Y al final de la salida en pantalla, la persistencia (repetir el último valor) en el conjunto de prueba a 1 min debe dar un RMSE de unos 0,017 mS/cm para EC y 0,029 para pH.

Estos valores son con el clima de respaldo. Si descargaste el clima real (sección 0), los números cambian y no hay que compararlos con esta tabla; en ese caso la comprobación es `validar_realismo.py`.

Si algo no coincide o aparece un error, copia el mensaje completo de la terminal y envíalo.

## 5. Qué hay en el código

| Archivo | Contenido |
| --- | --- |
| `src/clima.py` | Clima exterior de Chennai: el real si existe `data_reales/clima_chennai.csv`, si no el de respaldo con normales climáticas. |
| `src/descargar_clima.py` | Baja el clima horario real de Chennai (Open-Meteo). |
| `src/validar_realismo.py` | Compara los datos simulados con rangos de referencia con fuente. |
| `src/simulator.py` | Gemelo digital de una unidad DWC de lechuga (10 plantas, 80 L) y los sensores del artículo con ruido, deriva, picos y huecos. **Todos los supuestos están en la clase `Params`**: puedes cambiarlos para hacer análisis de sensibilidad. |
| `src/controllers.py` | Controladores manual, por umbrales e inteligente (lazo de la sección 3.4, Ec. 9 y 10). El inteligente usa por ahora la consigna del día; el LSTM se conecta en la fase 3. |
| `src/pipeline.py` | Ec. 1 (calibración de pH), Ec. 2 (EC a 25 °C), Ec. 3 (media móvil de 10), Ec. 5 (outliers IQR), interpolación, Ec. 4 (z-score ajustado solo con entrenamiento), ventanas 60 × 6 y partición cronológica 70/15/15. |
| `src/generate_dataset.py` | El script que ejecutas: genera todo lo de `data/` y `figs/`. |

Para probar otra semilla aleatoria: `python src/generate_dataset.py --seed 1`.

## 6. Hallazgos de la fase 1

1. **Las métricas del LSTM del artículo no cuadran con la variabilidad de EC y pH.** En el conjunto de prueba, la desviación estándar de EC a 15 min es 0,07 mS/cm y la de pH 0,05. Para tener R² = 0,91 haría falta un RMSE de unos 0,02 mS/cm, no de 0,25. La simple persistencia ya da 0,017 (EC) y 0,029 (pH).
2. **El error de seguimiento de la Tabla 1 (≤0,03 EC, ≤0,05 pH) se alcanza casi sin LSTM.** Basta con seguir la consigna del día con el lazo de la sección 3.4: error medio de 0,036 mS/cm y 0,029 de pH.
3. **El IQR global de la Ec. 5 marca como outliers el 1,6 % de las temperaturas**, que son horas de calor reales y no fallas. Conviene probar una versión con ventana móvil.
4. Con óptimos agronómicos fijos (EC 1,4–2,2; pH 5,6–6,4), el control por umbrales y el inteligente provisional dan el mismo rendimiento (231 g/planta); el manual da 195 g. El calor de Chennai (el cuarto pasa de 27 °C muchas tardes) es lo que más limita el crecimiento.
5. Energía: la Tabla 4 dice 8,5 kWh en total y 141 W de potencia media diaria, que serían 203 kWh en 60 días. Es otra contradicción interna del artículo; con un LED de 60 W, el simulador da unos 78 kWh.

---

# Fase 2: LSTM y modelos de comparación

Entrena el LSTM del artículo con las ventanas de la fase 1 y lo compara con cinco alternativas. Necesitas haber corrido antes la fase 1 (debe existir la carpeta `data`).

## 2.1 Instalar TensorFlow (solo la primera vez)

Con el entorno activo (`.venv\Scripts\activate`) y desde la carpeta del proyecto:

```
pip install -r requirements-fase2.txt
```

Descarga unos 500 MB. Para comprobar que quedó bien: `python -c "import tensorflow as tf; print(tf.__version__)"` debe mostrar 2.16 o superior.

## 2.2 Ejecutar

Primero una prueba rápida (1 semilla, 10 épocas, unos 3 minutos):

```
python src\train_models.py --quick
```

Si termina sin errores, la corrida completa (3 semillas, hasta 150 épocas con early stopping). En una PC sin GPU puede tardar entre 30 y 90 minutos:

```
python src\train_models.py
```

Mientras corre, va imprimiendo una línea por modelo entrenado. Puedes dejarlo trabajando y volver después.

Opciones: `--seeds 5` (más semillas), `--dataset 1min` (las 86 400 filas, mucho más lento), `--target setpoint` (variante B: predecir la consigna del día).

## 2.3 Qué hace

| Modelo | Configuración |
| --- | --- |
| Persistencia | Predice que EC y pH en 15 min serán iguales al último valor. No está en el artículo, pero es la referencia mínima que cualquier modelo debe superar. |
| Regresión lineal | Sobre la ventana completa aplanada (60 × 6 = 360 entradas) |
| Random Forest | 300 árboles sobre estadísticas de la ventana (último valor, media, pendiente, mínimo, máximo) |
| CNN 1D | Dos Conv1D de 32 filtros, pooling global, Dense 16 |
| CNN-LSTM | Conv1D de 32 filtros + LSTM 32 + Dense 16 |
| **LSTM (artículo)** | Exacto al artículo: LSTM 64 → LSTM 32 → Dropout 0,2 → Dense 16 ReLU → Dense 2. Adam 0,001, MSE, máx. 150 épocas con early stopping. Entrenado con aumentación (jitter σ = 0,03 y time warping ±10 %) y sin ella. |

Supuestos propios (el artículo no los da): lote de 128, paciencia de 10 épocas, partición cronológica 70/15/15. El lote de 128 en vez de 32 reduce el tiempo a la mitad en CPU sin cambiar el resultado de forma apreciable.

Al final convierte el LSTM a TensorFlow Lite y mide la latencia de una predicción, como en la sección 3.7 del artículo (~300 ms en una Raspberry Pi 4).

## 2.4 Qué deberías obtener

Se crea la carpeta `resultados` con:

| Archivo | Contenido |
| --- | --- |
| `metricas_fase2_500s_forecast.csv` | RMSE, MAPE y R² de cada modelo (media y desvío entre semillas) |
| `metricas_fase2_500s_forecast_por_semilla.csv` | Lo mismo, semilla por semilla |
| `comparacion_articulo_500s_forecast.csv` | LSTM del artículo vs. LSTM de la réplica |
| `info_fase2_500s_forecast.json` | Versión de TF, tamaños de los conjuntos, latencia TFLite |
| `lstm_articulo.keras` y `lstm_articulo.tflite` | El modelo entrenado (se usará en la fase 3) |

y la figura `figs\fase2_prediccion_500s_forecast.png`.

**Comprobación.** Las redes neuronales varían un poco entre computadoras y versiones de TensorFlow, así que compara con rangos, no con decimales exactos. Referencia de nuestra prueba con el simulador realista (TensorFlow 2.21, CPU, `--quick`, semilla 0) en `resultados_esperados/fase2_referencia_500s_forecast.csv`:

| Modelo | RMSE EC (mS/cm) | RMSE pH | R² EC | R² pH |
| --- | --- | --- | --- | --- |
| Persistencia | 0,0047 (exacto) | 0,0082 (exacto) | 0,995 | 0,967 |
| Regresión lineal | 0,0032 (exacto) | 0,0051 (exacto) | 0,998 | 0,987 |
| Random Forest | ~0,014 | ~0,012 | ~0,96 | ~0,93 |
| LSTM (artículo) | 0,005–0,009 | 0,007–0,010 | 0,98–0,99 | 0,95–0,98 |
| LSTM sin aumentación | 0,008–0,013 | 0,009–0,014 | 0,96–0,99 | 0,90–0,96 |
| CNN 1D | 0,02–0,035 | 0,03–0,045 | 0,7–0,9 | 0,0–0,4 |
| CNN-LSTM | 0,005–0,008 | 0,007–0,010 | ~0,99 | ~0,97 |

En escala normalizada, el LSTM da 0,12 (EC) y 0,12 (pH), frente a 0,25 y 0,18 del artículo.

Latencia TFLite del LSTM: alrededor de 0,3–1 ms por predicción en una PC (en una Raspberry Pi 4 sería varias veces más).

## 2.5 Cómo leer los resultados

1. **El LSTM del artículo no supera a los modelos simples.** La regresión lineal y la persistencia predicen EC y pH a 15 min tan bien o mejor que el LSTM. Tiene sentido: en 15 minutos EC y pH casi no cambian.
2. **Nuestro LSTM tiene un error unas 25 veces menor que el del artículo** (0,01 frente a 0,25 mS/cm), con R² parecido o mayor. Con la variabilidad real de EC y pH, un RMSE de 0,25 sería peor que no predecir nada; esto refuerza que las métricas del artículo no son coherentes.
3. **La aumentación ayuda un poco** al LSTM (jitter + time warping), en línea con lo que dice el artículo.
4. El MAPE de nuestra réplica queda por debajo del 0,5 %, frente al 6,2 % y 4,7 % reportados.

5. **Probable explicación de la diferencia: el artículo reportó el RMSE en la escala normalizada (z-score).** Si se divide nuestro RMSE por el desvío estándar de entrenamiento, el LSTM da unos 0,16 (EC) y 0,14 (pH), frente a 0,25 y 0,18 del artículo, con R² casi iguales (el R² no depende de la escala). Con esa lectura, las tres métricas del artículo vuelven a ser coherentes entre sí. El script lo muestra en la columna "réplica en escala normalizada" de `comparacion_articulo_*.csv`. Lo que no explica es el MAPE: en escala normalizada no tiene sentido, porque los valores rondan 0.

Estos resultados son con datos simulados: dicen si el método del artículo es coherente, no cómo se comportaría con datos reales.

**¿GPU?** No hace falta: el modelo es pequeño y la corrida completa ya cabe en una hora de CPU. Además, TensorFlow dejó de soportar GPU en Windows nativo desde la versión 2.10; para usarla habría que instalar WSL2 (Linux dentro de Windows) y `pip install tensorflow[and-cuda]`. Solo vale la pena si más adelante entrenas con `--dataset 1min` o con muchas semillas.


---

# Fase 3: control en lazo cerrado (ensayo virtual)

Conecta el LSTM de la fase 2 al controlador y simula el ensayo de 60 días con cuatro tratamientos, como la sección 4 del artículo. Requiere haber corrido la fase 1 y la fase 2 (al menos `--quick`, que crea `resultados\lstm_articulo.keras`).

## 3.1 Ejecutar

Con `(.venv)` activo y desde la carpeta del proyecto:

```
python src\run_trial.py --quick
python src\run_trial.py
```

`--quick` usa 2 semillas (unos 2 minutos). La corrida normal usa 10 semillas por tratamiento (unos 6–15 minutos). Con `--seeds 30` haces las 30 repeticiones del plan.

## 3.2 Tratamientos

| Tratamiento | Qué hace |
| --- | --- |
| Manual | Un operador corrige EC y pH una vez al día, con error de medición y de dosis |
| Umbrales | Consigna fija EC 1,8 ±0,2 mS/cm y pH 6,0 ±0,2 |
| Consigna sin LSTM | El lazo de la sección 3.4 siguiendo la consigna del día (Tabla 1), pero sin modelo. Sirve para separar qué aporta el LSTM y qué aporta el lazo |
| LSTM | Igual, pero decide con el pronóstico del LSTM a 15 min: si el LSTM prevé que EC o pH se saldrán de la consigna, dosifica antes. El modelo corre en TFLite, como en el artículo |

Las semillas del ensayo (100 en adelante) son distintas de la que generó los datos de entrenamiento (0), así que el LSTM se prueba en condiciones que no vio.

## 3.3 Qué deberías obtener

En `resultados` aparecen `fase3_tabla1_nutrientes.csv`, `fase3_tabla2_rendimiento.csv`, `fase3_tabla4_recursos.csv`, `fase3_tukey_rendimiento.csv`, `fase3_por_semilla.csv` y `fase3_resumen.json`; en `figs`, `fase3_series.png` y `fase3_comparacion.png`.

Referencia de nuestra prueba con 10 semillas y el simulador realista (`resultados_esperados\fase3_resumen_tratamientos.csv`):

| Tratamiento | Rendimiento (g/planta) | CV (%) | Días a 250 g | Error medio EC | Error medio pH | Nutriente usado vs. umbrales |
| --- | --- | --- | --- | --- | --- | --- |
| Manual | 202,6 | 13,5 | no llega | 0,050 | 0,102 | +16,7 % |
| Umbrales | 240,0 | 12,9 | 59,1 | 0,139 | 0,102 | — |
| Consigna sin LSTM | 240,0 | 12,9 | 59,1 | 0,034 | 0,031 | +20,0 % |
| LSTM | 240,0 | 12,9 | 59,1 | 0,034 | 0,031 | +20,0 % |

ANOVA del rendimiento: p < 0,001, pero según Tukey la única diferencia significativa es la del manual frente a los otros tres. Latencia TFLite: unos 0,5 ms por predicción. "Días a 250 g" es el promedio de las semillas en que la planta media llega a 250 g; con el calor de Chennai, el manual no llega en ninguna.

## 3.4 Cómo leer los resultados

1. **El LSTM no mejora el rendimiento frente al control por umbrales** (240,0 g en ambos). En el simulador, la lechuga crece igual mientras EC y pH estén dentro de su rango agronómico (EC 1,4–2,2; pH 5,6–6,4), y los dos controles automáticos lo cumplen.
2. **El LSTM no aporta frente al mismo lazo sin LSTM.** Las diferencias son del orden de milésimas: lo que reduce el error de EC y pH (EC de 0,14 a 0,03; pH de 0,10 a 0,03) es seguir la consigna del día, no el modelo.
3. **No reproducimos el 27,6 % de ahorro de nutrientes:** seguir una consigna que sube de 1,6 a 2,0 mS/cm gasta más concentrado (+20 %) que mantener 1,8 fijo.
4. **La energía no cambia** entre los tratamientos automáticos, porque el clima y la luz se controlan igual. El 30 % de ahorro de la Tabla 4 no puede venir del control de nutrientes.
5. Para obtener una ventaja del LSTM en rendimiento, el crecimiento tendría que depender de seguir la consigna exacta de la Tabla 1. Eso sería diseñar el simulador para que gane el LSTM (el riesgo de circularidad del plan). Lo dejamos como análisis de sensibilidad para la fase 5.

---

# Fase 4: IoT emulado (MQTT, InfluxDB, Grafana y pruebas de falla)

Reproduce la arquitectura de la sección 3 del artículo con software en vez de hardware:

| En el artículo | En la réplica |
| --- | --- |
| ESP32 con sensores y relés | `src\iot_node.py`: lee el simulador, publica cada lectura por MQTT, ejecuta las órdenes de dosificación y controla el clima localmente. Guarda las lecturas en un buffer si se cae la red |
| Raspberry Pi 4 (edge) | `src\edge.py`: recibe por MQTT, guarda en InfluxDB, ejecuta el LSTM en TFLite cada 15 min, decide la dosis (Ec. 9 y 10), envía la orden y genera alertas (Ec. 11) |
| Mosquitto, InfluxDB, Grafana | Contenedores Docker (`iot\docker-compose.yml`), con el tablero ya configurado |
| Firebase, ThingSpeak, Google Cloud | Se omiten: no influyen en las métricas que se comparan |

El tiempo va acelerado: por defecto, 10 minutos simulados por segundo real (2 días simulados ≈ 5 minutos). Las fallas sí se miden en segundos reales.

## 4.1 Requisitos (solo la primera vez)

1. **Docker Desktop** para Windows: https://www.docker.com/products/docker-desktop/ . Instálalo, reinicia si lo pide y ábrelo. Debe quedar en verde ("Engine running").
2. Las librerías de MQTT e InfluxDB, con `(.venv)` activo:

```
pip install -r requirements-fase4.txt
```

## 4.2 Ejecutar

Desde la carpeta del proyecto, con `(.venv)` activo:

```
docker compose -f iot\docker-compose.yml up -d
python src\iot_demo.py
```

La primera línea descarga y enciende Mosquitto, InfluxDB y Grafana (la primera vez descarga ~1 GB). La segunda corre la demostración completa, de unos 5 minutos:

1. Arranca el edge y el nodo virtual.
2. Durante el primer minuto el LSTM no predice, porque necesita 500 minutos simulados de historia.
3. A mitad de la corrida desconecta los sensores 3 veces (5 s cada vez).
4. Cerca del final apaga el broker MQTT 10 s y lo vuelve a encender.
5. Imprime las métricas tipo Tabla 5 y las guarda en `resultados\fase4_reporte.json`.

**Mientras corre**, abre http://localhost:3000 en el navegador. Verás el tablero "Réplica hidroponía IoT" actualizándose cada 5 s: EC y pH (medidos, consigna y pronóstico del LSTM), temperatura, humedad, dosis, latencias, alertas y luz. Referencia: `resultados_esperados\fase4_grafana.png`.

Para apagar los contenedores al terminar: `docker compose -f iot\docker-compose.yml down`.

Si el puerto 1883, 8086 o 3000 ya está ocupado en tu PC, `docker compose up` lo dirá. Avísame y cambiamos el puerto.

## 4.3 Qué deberías obtener

Referencia de nuestra prueba (los tiempos dependen de tu PC):

| Métrica (Tabla 5) | Artículo | Réplica |
| --- | --- | --- |
| Detección → acción | 3,2 s | ~3 ms |
| Ejecución en el edge | 0,8 s | ~2 ms |
| Inferencia LSTM en TFLite | ~300 ms (Raspberry Pi) | ~1 ms (PC) |
| Recuperación tras desconectar sensores 5 s | 58 s | ~5 s (lo que dura la falla) |
| Reconexión MQTT tras 10 s sin broker | 88 s | ~11 s (≈1 s después de volver el broker) |
| Registros recuperados del buffer | 94 | ~110 (todo lo generado durante el corte) |
| Disponibilidad | 98,7 % | ~94 % |

## 4.4 Cómo leer los resultados

1. **La arquitectura del artículo funciona y se puede reproducir con software libre.** El flujo completo sensor → MQTT → edge → LSTM → orden → actuador anda, y el tablero muestra lo mismo que describe la sección 3.5.2.
2. **Los tiempos dependen de cómo se implemente la reconexión, no del LSTM.** Con reintentos cada 1–4 s, la recuperación tarda alrededor de un segundo más que la falla. Los 58 s y 88 s del artículo sugieren reintentos mucho más lentos, que no se describen. La comparación sirve de referencia, no como prueba.
3. **La disponibilidad (~94 %) baja por las fallas inyectadas a propósito** (3 cortes de sensores) más los huecos aleatorios del simulador. Sin `--sin-fallas` no se puede comparar directamente con el 98,7 %.
4. En el tablero se ve que **el pronóstico del LSTM sale un poco alto (~0,05 mS/cm) al inicio del cultivo.** Se entrenó con datos del control por umbrales, con la EC alrededor de 1,8, y al inicio la EC está en 1,5: fuera de lo que vio en el entrenamiento. Es una limitación real de entrenar con datos de un solo tratamiento.

Otras opciones: `python src\iot_demo.py --dias 5` (más largo), `--sin-fallas` (solo flujo normal), `--velocidad 30` (más rápido).

---

# Fase 5: escenario realista vs. escenario favorable al artículo

Pregunta: **¿qué tendría que ser cierto para obtener las cifras del artículo?** El script corre la réplica completa (datos, entrenamiento del LSTM y ensayo de 4 tratamientos) dos veces y compara ambas corridas con el artículo.

| Escenario | Supuestos |
| --- | --- |
| Realista | El simulador tal cual: Chennai sin aire acondicionado, lechuga que crece igual mientras EC y pH estén en su rango sano, sensores con la precisión de su hoja de datos, umbrales sin desperdicio |
| Favorable | 1. Cuarto climatizado a 24 °C (plausible). 2. Sensores más ruidosos y con más deriva: pH ±0,15, EC ±4 %, deriva de 0,1 pH por semana (plausible con sensores baratos). 3. Manual y umbrales cambian toda la solución cada 30 días (práctica común). 4. La lechuga crece mejor cuanto más cerca esté de la consigna exacta de la Tabla 1 (**sin evidencia**: es lo que haría falta para que seguir la consigna dé más rendimiento) |

Los supuestos están en `ESCENARIOS`, al comienzo de `src\escenarios.py`, por si quieres probar otros.

## 5.1 Ejecutar

Requiere la fase 1 ya corrida con el simulador nuevo. Con `(.venv)` activo:

```
python src\escenarios.py --quick
python src\escenarios.py
```

`--quick` usa 2 semillas y 10 épocas (unos 8–15 minutos). La corrida normal usa 5 semillas y hasta 40 épocas (unos 15–30 minutos). Cada escenario entrena su propio LSTM con sus propios datos y lo guarda en `resultados\fase5\<escenario>\`. No toca el modelo de la fase 2.

## 5.2 Qué deberías obtener

`resultados\fase5_comparacion.csv` y `figs\fase5_escenarios.png`. Referencia de nuestra corrida con 5 semillas (`resultados_esperados\fase5_comparacion.csv`). Las cifras de rendimiento y nutrientes deberían salir casi iguales; las del LSTM pueden variar un poco según la computadora:

| Métrica | Artículo | Realista | Favorable |
| --- | --- | --- | --- |
| Rendimiento manual (g) | 297 | 203 | 376 |
| Rendimiento umbrales (g) | 323 | 240 | 333 |
| Rendimiento LSTM (g) | 368 | 240 | 379 |
| LSTM vs. umbrales | +13,9 % | 0 % | +13,9 % |
| Umbrales vs. manual | +8,8 % | +18,3 % | −11,5 % |
| Ahorro de nutrientes LSTM vs. umbrales | 27,6 % | −20,0 % | 25,0 % |
| Error medio EC / pH con LSTM | 0,03 / 0,05 | 0,034 / 0,030 | 0,040 / 0,099 |
| RMSE del LSTM en escala normalizada, EC / pH | 0,25 / 0,18 | 0,13 / 0,12 | 0,17 / 0,27 |
| R² del LSTM, EC / pH | 0,91 / 0,94 | 0,99 / 0,98 | 0,99 / 0,95 |
| LSTM vs. el mismo lazo sin LSTM | — | 0 % | +0,04 % |
| ¿El LSTM supera a la regresión lineal? | — | no | no |

## 5.3 Cómo leer los resultados

1. **Con los cuatro supuestos se reproducen las cifras principales:** los rendimientos de umbrales y LSTM, el +13,9 % y un ahorro de nutrientes del 25 %, cerca del 27,6 %.
2. **Pero la ventaja no viene del LSTM.** El mismo lazo sin LSTM da el mismo rendimiento (diferencia de 0,04 %). La ventaja sale de seguir la consigna del día (supuesto 4) y de no tirar la solución (supuesto 3).
3. **El supuesto 4 es el que decide, y no tiene respaldo.** Sin él (escenario realista), todos los controles automáticos rinden igual.
4. **El escenario favorable contradice al artículo en el manual.** El operador también sigue la consigna del día, así que con el supuesto 4 rinde más que los umbrales (376 g frente a 333 g). En el artículo el manual es el peor. Para que eso pase, el manual tendría que tener mucho más error que el que suponemos.
5. **Con sensores más ruidosos, el error de pH del LSTM se acerca al del artículo** (0,27 frente a 0,18 en escala normalizada), pero el error de seguimiento del pH empeora a 0,10, el doble de lo que reporta el artículo. Con los mismos sensores no se pueden tener a la vez el RMSE y el seguimiento que reporta el artículo.
6. En ningún escenario el LSTM supera a una regresión lineal para predecir EC a 15 minutos.

Conclusión: las cifras del artículo se pueden obtener, pero solo con supuestos que el artículo no declara. La mejora que atribuye al LSTM se explica por el lazo que sigue la consigna, no por el modelo.

#   P r o y e c t o - R e p l i c a  
 