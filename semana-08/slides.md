# Semana 08: Suavizado Exponencial y Benchmark Integrador
### Holt-Winters Amortiguado, Validación con Origen Móvil y Torneo de Modelos
**MA3001B: Modelos y TDA • Profesor B**  
Tecnológico de Monterrey • 8vo Semestre

Note:
[00:00 - 00:05]
Bienvenida a la última sesión del módulo (Semana 8).
Hoy integramos los dos mundos: deep learning (Semana 4) y estadística clásica (Semanas 5 a 7), agregando la familia de suavizado exponencial y ejecutando el gran torneo final de modelos.
Duración: 180 min.

---

## Agenda de la Sesión (180 min)

| Minutos | Bloque | Actividad y Enfoque |
| :--- | :--- | :--- |
| **00:00–00:20** | **Detonador** | La tabla de un solo corte: ¿evidencia real o suerte? |
| **00:20–01:00** | **Teoría** | Suavizado simple, Holt, Holt-Winters, MASE y origen móvil |
| **01:00–01:10** | **Descanso** | Pausa intermedia |
| **01:10–01:40** | **Paquetes** | `ExponentialSmoothing`, parámetros $\alpha, \beta, \gamma, \phi$ |
| **01:40–02:45** | **Benchmark** | Torneo de 5 familias sobre la serie del Reto |
| **02:45–03:00** | **Cierre** | Mini-presentaciones y recomendación ejecutiva final |

Note:
[00:05 - 00:10]
Explicar que la meta de hoy no es coronar a un modelo favorito, sino evaluar con métricas imparciales cuál conviene implementar en producción.

---

## Problema Detonador (00:00–00:20)

<div class="alert-box">
  <strong>El espejismo del corte único:</strong> Un equipo presenta un informe donde la LSTM obtiene un RMSE de 0.38 frente a 0.41 de SARIMA en una sola partición de prueba de 12 meses. Con base en esto, solicitan servidores con GPU para producción. El área de auditoría de modelos cuestiona: <em>¿Qué nos garantiza que en el próximo año la LSTM no falle estrepitosamente debido a la volatilidad de un solo corte favorable?</em>
</div>

--

### Código del Problema Detonador

```python [1-5|7-12]
# Evaluación en un solo corte temporal arbitrario:
corte = len(y) - 12
train, test = y.iloc[:corte], y.iloc[corte:]

# Supongamos que en este año específico la LSTM tuvo suerte:
# RMSE LSTM: 0.38 ppm | RMSE SARIMA: 0.41 ppm

# Pero si desplazamos el corte 12 meses atrás:
corte_previo = len(y) - 24
# En ese ciclo, ¡la LSTM tuvo RMSE de 1.85 y SARIMA 0.43!
# Un único corte es insuficiente para certificar robustez en producción.
```

--

### Preguntas para los Equipos (15 min)

<div class="highlight-box">
  <ol>
    <li>¿Por qué la validación cruzada aleatoria clásica de K-Fold no es aplicable a series de tiempo?</li>
    <li>¿Cómo funciona la validación con **origen móvil** (rolling-origin cross-validation) para simular múltiples despliegues reales en el tiempo?</li>
    <li>¿Cómo medimos el desempeño entre diferentes series con magnitudes distintas sin que el porcentaje se distorsione al dividir entre valores cercanos a cero?</li>
  </ol>
</div>

Note:
[00:15 - 00:20]
Presentar la necesidad de métricas robustas como MASE (Mean Absolute Scaled Error).

---

## Objetivos de Aprendizaje

- Dominar la formulación de **Suavizado Exponencial Simple (SES)**, **Holt lineal** y **Holt-Winters** (aditivo y multiplicativo). <!-- .element: class="fragment" -->
- Implementar la **amortiguación de tendencia** ($\phi$) para prevenir pronósticos divergentes a largo plazo. <!-- .element: class="fragment" -->
- Comprender y computar la métrica estandarizada **MASE**. <!-- .element: class="fragment" -->
- Diseñar un protocolo de validación riguroso con **5 orígenes móviles consecutivos**. <!-- .element: class="fragment" -->
- Sintetizar un informe de recomendación técnica considerando precisión, coste computacional, mantenimiento y explicabilidad. <!-- .element: class="fragment" -->

---

## Teoría: Suavizado Exponencial

El principio intuitivo es que las observaciones más recientes contienen mayor información que el pasado remoto:

- **Suavizado Exponencial Simple (SES):**
  $$\hat{y}_{t+1|t} = \alpha y_t + (1 - \alpha) \hat{y}_{t|t-1} = \sum_{j=0}^{\infty} \alpha (1 - \alpha)^j y_{t-j}$$
  Donde $\alpha \in (0, 1)$ es el factor de suavizado del nivel.

--

### Método de Holt: Nivel y Tendencia

Añade una ecuación para la pendiente estimada $b_t$, regulada por el parámetro $\beta \in (0, 1)$:

$$
\ell_t = \alpha y_t + (1 - \alpha)(\ell_{t-1} + b_{t-1})
$$

$$
b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}
$$

$$
\hat{y}_{t+h|t} = \ell_t + h b_t
$$

--

### Método de Holt-Winters Completo

Incorpora la ecuación estacional con parámetro $\gamma \in (0, 1)$ y periodo $m$:

$$
\ell_t = \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1})
$$

$$
b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}
$$

$$
s_t = \gamma (y_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-m}
$$

$$
\hat{y}_{t+h|t} = \ell_t + h b_t + s_{t+h-m(k+1)}
$$

--

### Tendencia Amortiguada (Damped Trend)

Las tendencias lineales puras tienden a sobrestimar la realidad a horizontes largos ($h > 12$). Introducimos un parámetro de amortiguamiento $\phi \in (0, 1)$:

$$
\hat{y}_{t+h|t} = \ell_t + \left(\sum_{i=1}^h \phi^i\right) b_t + s_{t+h-m(k+1)}
$$

- Cuando $h \to \infty$, la suma converge a $\frac{\phi}{1 - \phi} b_t$, impidiendo que el pronóstico explote hacia el infinito. <!-- .element: class="fragment" -->

Note:
[00:45 - 00:55]
El modelo amortiguado aditivo (`damped_trend=True`) es históricamente uno de los más difíciles de vencer en competiciones M3 y M4.

---

## Validación con Origen Móvil (Rolling Origin)

En lugar de evaluar en un único corte, simulamos $K$ momentos históricos consecutivos en los que el modelo genera un pronóstico hacia adelante de horizonte $h$:

<svg viewBox="0 0 650 200" style="width:100%; max-height:210px; margin: 10px auto; display:block;">
  <!-- Origen 1 -->
  <rect x="20" y="25" width="300" height="22" rx="4" fill="#EEE8D5" stroke="#073642"/>
  <rect x="320" y="25" width="80" height="22" rx="4" fill="#CB4B16"/>
  <text x="170" y="40" text-anchor="middle" font-family="Inter" font-size="11" fill="#073642">Entrenamiento 1</text>
  <text x="360" y="40" text-anchor="middle" font-family="Inter" font-size="11" fill="#FFF">Prueba 1</text>

  <!-- Origen 2 -->
  <rect x="20" y="60" width="340" height="22" rx="4" fill="#EEE8D5" stroke="#073642"/>
  <rect x="360" y="60" width="80" height="22" rx="4" fill="#CB4B16"/>
  <text x="190" y="75" text-anchor="middle" font-family="Inter" font-size="11" fill="#073642">Entrenamiento 2</text>
  <text x="400" y="75" text-anchor="middle" font-family="Inter" font-size="11" fill="#FFF">Prueba 2</text>

  <!-- Origen 3 -->
  <rect x="20" y="95" width="380" height="22" rx="4" fill="#EEE8D5" stroke="#073642"/>
  <rect x="440" y="95" width="80" height="22" rx="4" fill="#CB4B16"/>
  <text x="210" y="110" text-anchor="middle" font-family="Inter" font-size="11" fill="#073642">Entrenamiento 3</text>
  <text x="480" y="110" text-anchor="middle" font-family="Inter" font-size="11" fill="#FFF">Prueba 3</text>

  <!-- Origen 4 -->
  <rect x="20" y="130" width="420" height="22" rx="4" fill="#EEE8D5" stroke="#073642"/>
  <rect x="520" y="130" width="80" height="22" rx="4" fill="#CB4B16"/>
  <text x="230" y="145" text-anchor="middle" font-family="Inter" font-size="11" fill="#073642">Entrenamiento 4</text>
  <text x="560" y="145" text-anchor="middle" font-family="Inter" font-size="11" fill="#FFF">Prueba 4</text>
</svg>

- Cada modelo se reentrena (o actualiza) y se promedian los errores a través de todos los orígenes.

---

## La Métrica MASE (Mean Absolute Scaled Error)

Propuesta por Hyndman y Koehler para comparar modelos de manera justa e independiente de escala:

$$
\operatorname{MASE} = \frac{\frac{1}{h} \sum_{j=1}^h |y_{T+j} - \hat{y}_{T+j}|}{\frac{1}{T - m} \sum_{t=m+1}^T |y_t - y_{t-m}|}
$$

- El numerador es el **MAE del modelo evaluado en el conjunto de prueba**.
- El denominador es el **MAE del baseline ingenuo estacional dentro de la muestra de entrenamiento**.
- **Interpretación clave:**
  - $\operatorname{MASE} < 1$: El modelo es **superior** al baseline ingenuo estacional.
  - $\operatorname{MASE} > 1$: El modelo es **peor** que simplemente copiar los datos del año pasado.

---

## Paquetes en Python: Holt-Winters

```python [1-6|8-12]
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Ajuste con estimación óptima de parámetros iniciales:
hw_model = ExponentialSmoothing(
    y_train,
    trend='add',
    damped_trend=True,
    seasonal='add',
    seasonal_periods=12,
    initialization_method='estimated'
).fit()

# Parámetros estimados:
# hw_model.params['smoothing_level']    -> alpha
# hw_model.params['smoothing_trend']    -> beta
# hw_model.params['smoothing_seasonal'] -> gamma
# hw_model.params['damping_trend']      -> phi
```

Note:
[01:25 - 01:40]
El método `.fit()` en statsmodels ejecuta optimización L-BFGS-B en menos de 0.05 segundos en CPU.

---

## El Gran Torneo del Reto (5 Familias)

Competencia sobre la serie del Reto a lo largo de 5 orígenes móviles con horizonte $h=12$:

<div class="container">
  <div class="col">
    <h4>1. Ingenuo Estacional</h4>
    <p style="font-size:0.75em;">Línea base sin parámetros. Repite el último ciclo observado.</p>
  </div>
  <div class="col">
    <h4>2. Holt-Winters</h4>
    <p style="font-size:0.75em;">Tendencia amortiguada y estacionalidad aditiva.</p>
  </div>
  <div class="col">
    <h4>3. SARIMA</h4>
    <p style="font-size:0.75em;">Modelo econométrico identificado con Box-Jenkins.</p>
  </div>
</div>

<div class="container" style="margin-top:10px;">
  <div class="col">
    <h4>4. LSTM Recurrente</h4>
    <p style="font-size:0.75em;">Red neuronal con compuertas y pronóstico recursivo.</p>
  </div>
  <div class="col">
    <h4>5. Transformer</h4>
    <p style="font-size:0.75em;">Atención multi-cabeza y codificación posicional.</p>
  </div>
</div>

---

## Tabla Resumen de Decisión Multicriterio

| Dimensión de Negocio | Ingenuo Est. | Holt-Winters | SARIMA | LSTM / Transformer |
| :--- | :---: | :---: | :---: | :---: |
| **Tiempo de Ajuste** | Inmediato | < 0.1 s | ~ 1–3 s | ~ 10–60 s |
| **Intervalos Teóricos** | No | Limitados | **Exactos** | Vía Dropout / Bootstrapping |
| **Actualización en Línea** | Trivial | **Instantánea** | Requiere filtro | Costosa |
| **Interpretabilidad** | Total | Alta | Alta | Caja Negra |
| **Desempeño en Series Univariadas** | Referencia | **Excelente** | **Excelente** | Propenso a sobreajuste |

Note:
[02:15 - 02:30]
Destacar a los alumnos: en ciencia de datos industrial, el modelo más complejo rara vez es el modelo más adecuado para producción.

---

## Actividad del Reto y Entregable Final (01:40–02:45)

<div class="highlight-box">
  <strong>Misión Integrada:</strong> Abrir <code>notebooks/N5_HoltWinters_Benchmark.ipynb</code>.
</div>

- Ejecutar el benchmark sobre los 5 orígenes de pronóstico en la serie del Reto.
- Construir la tabla final con **MAE, RMSE y MASE promedio** por modelo.
- Responder con evidencia: ¿Logró el Deep Learning vencer a los modelos estadísticos clásicos?
- Preparar la **presentación ejecutiva de 2 minutos** por equipo: modelo elegido y justificación financiera y técnica.

---

## Cierre del Módulo y Conclusiones (02:45–03:00)

<div class="success-box">
  <strong>Conclusión Global de Modelos y TDA (Profesor B):</strong>
  <ol>
    <li>El Deep Learning brilla en contextos multivariados masivos; en series de tiempo univariadas con estacionalidad, los modelos de estructura estadística (SARIMA, Holt-Winters) suelen ser más precisos, estables y económicos.</li>
    <li>El verdadero valor de un ingeniero de datos radica en seleccionar el modelo adecuado con base en evidencia empírica rigurosa y métricas imparciales.</li>
  </ol>
</div>

Note:
[02:55 - 03:00]
Felicitaciones a todos los equipos por completar el módulo de Modelos y TDA en MA3001B.
Enlace directo: notebooks/html/N5_HoltWinters_Benchmark.html
