# Semana 07: Atípicos, ARIMA/SARIMA y AIC/BIC
### Detección Robusta, Metodología Box-Jenkins e Intervalos de Pronóstico
**MA3001B: Modelos y TDA • Profesor B**  
Tecnológico de Monterrey • 8vo Semestre

Note:
[00:00 - 00:05]
Bienvenida a la Semana 7.
Hoy convertimos toda la información aprendida (estacionariedad, rezagos, ciclos) en modelos econométricos formales: SARIMAX con intervalos de confianza e inferencia matemática rigurosa.
Duración: 180 min.

---

## Agenda de la Sesión (180 min)

| Minutos | Bloque | Actividad y Enfoque |
| :--- | :--- | :--- |
| **00:00–00:25** | **Detonador** | Pronóstico puntual sin bandas y picos atípicos destructivos |
| **00:25–01:15** | **Teoría** | Atípicos con MAD, Box-Jenkins, SARIMA, AIC y BIC |
| **01:15–01:25** | **Descanso** | Pausa intermedia |
| **01:25–02:00** | **Paquetes** | `SARIMAX`, `plot_diagnostics`, `acorr_ljungbox`, `get_forecast` |
| **02:00–02:50** | **Reto** | Rejilla SARIMA, diagnósticos y bandas de incertidumbre |
| **02:50–03:00** | **Cierre** | Defensa del mejor modelo y pregunta para Semana 8 |

Note:
[00:05 - 00:10]
Enfatizar el valor industrial de los intervalos de confianza: las decisiones de negocio requieren cuantificar el riesgo, no solo el valor esperado puntual.

---

## Problema Detonador (00:00–00:25)

<div class="alert-box">
  <strong>La junta directiva exige rangos de riesgo:</strong> La dirección general solicita un pronóstico a 24 meses que incluya intervalos de confianza del 95% para dimensionar inventarios. Nuestras redes (LSTM/Transformer) solo entregan un número flotante puntual por mes. Para colmo, dos picos anómalos en el historial están sesgando todas las predicciones hacia arriba.
</div>

--

### Código del Problema Detonador

```python [1-5|7-12]
# La LSTM solo produce predicciones puntuales escalares:
y_pred_lstm = lstm_model(X_test)  # [372.4, 373.1, 373.8, ...]
# ¿Cuál es el límite inferior y superior al 95% de probabilidad?
# La red clásica no modela la varianza condicional del error.

# Además, dos observaciones anómalas (+8 y -8 ppm) distorsionan el ajuste:
y_contaminada = y.copy()
y_contaminada.iloc[[100, 250]] += [8.0, -8.0]
# Si ajustamos mínimos cuadrados ordinarios sobre la serie con picos,
# los coeficientes autorregresivos se sesgan severamente hacia cero.
```

--

### Preguntas para los Equipos (15 min)

<div class="highlight-box">
  <ol>
    <li>¿Por qué detectar atípicos directamente sobre la serie en nivel ($y_t > 3\sigma$) confunde la estacionalidad normal con anomalías?</li>
    <li>¿Cómo podemos usar la descomposición STL para aislar el residuo puro y detectar anomalías sobre él?</li>
    <li>¿Por qué no podemos simplemente borrar las observaciones anómalas dejando huecos en la serie temporal sin un modelo que soporte datos faltantes?</li>
  </ol>
</div>

Note:
[00:15 - 00:25]
Explicar que `SARIMAX` en statsmodels maneja valores `NaN` de forma nativa a través de las ecuaciones de filtro de Kalman.

---

## Objetivos de Aprendizaje

- Identificar atípicos aditivos e innovacionales mediante la **mediana absoluta de las desviaciones (MAD)** sobre el residuo STL. <!-- .element: class="fragment" -->
- Formular procesos **AR($p$)**, **MA($q$)**, **ARIMA($p, d, q$)** y **SARIMA($p, d, q$)($P, D, Q$)$_s$**. <!-- .element: class="fragment" -->
- Ejecutar el **ciclo iterativo de Box-Jenkins**: identificación, estimación, diagnóstico y pronóstico. <!-- .element: class="fragment" -->
- Seleccionar modelos de forma parsimoniosa utilizando los criterios de información **AIC** y **BIC**. <!-- .element: class="fragment" -->
- Validar residuos mediante la prueba de autocorrelación **Ljung-Box** y normalidad visual. <!-- .element: class="fragment" -->

---

## Detección Robusta con la Regla de la MAD

Calcular la media y desviación estándar tradicional es vulnerable al fenómeno de *enmascaramiento*, pues los mismos atípicos inflan $\sigma$. En su lugar, utilizamos el **residuo de STL** $r_t$ y estimadores robustos:

$$
\operatorname{MAD}(r) = \operatorname{mediana}\left( |r_t - \operatorname{mediana}(r)| \right)
$$

$$
z_t = \frac{0.6745 \cdot (r_t - \operatorname{mediana}(r))}{\operatorname{MAD}(r)}
$$

- Criterio de atípico severo: **$|z_t| > 3.5$**. <!-- .element: class="fragment" -->
- Al marcar el valor como `np.nan`, el filtro de Kalman de `SARIMAX` interpola óptimamente el estado sin distorsionar los parámetros autorregresivos. <!-- .element: class="fragment" -->

---

## Modelos Autorregresivos y de Medias Móviles

- **AR($p$):** El valor actual depende linealmente de sus propios $p$ valores pasados más un choque aleatorio:
  $$y_t = c + \phi_1 y_{t-1} + \dots + \phi_p y_{t-p} + \varepsilon_t$$

- **MA($q$):** El valor actual depende de los $q$ errores o innovaciones pasadas:
  $$y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}$$

--

### Notación con el Operador de Rezago $B$

Definiendo $B y_t = y_{t-1}$ y $B^k y_t = y_{t-k}$:

$$
\phi(B) = 1 - \phi_1 B - \phi_2 B^2 - \dots - \phi_p B^p
$$

$$
\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + \dots + \theta_q B^q
$$

El modelo **ARIMA($p, d, q$)** con $d$ diferencias toma la forma compacta:

$$
\phi(B) (1 - B)^d y_t = c + \theta(B) \varepsilon_t
$$

--

### Modelo SARIMA Estacional Completo

Añade polinomios autorregresivos y de medias móviles estacionales con periodo $s$ y $D$ diferencias estacionales:

$$
\Phi_P(B^s) \phi_p(B) (1 - B)^d (1 - B^s)^D y_t = c + \Theta_Q(B^s) \theta_q(B) \varepsilon_t
$$

- Por ejemplo, en datos mensuales ($s=12$), $\operatorname{SARIMA}(1, 1, 1)(1, 1, 1)_{12}$:
  $$(1 - \Phi_1 B^{12})(1 - \phi_1 B)(1 - B)(1 - B^{12}) y_t = (1 + \Theta_1 B^{12})(1 + \theta_1 B) \varepsilon_t$$

Note:
[00:50 - 01:00]
Hacer notar que este modelo lineal tiene apenas 4 coeficientes libres pero captura simultáneamente tendencia y ciclo anual completo.

---

## El Ciclo de Box-Jenkins

<svg viewBox="0 0 650 180" style="width:100%; max-height:200px; margin: 10px auto; display:block;">
  <rect x="20" y="60" width="120" height="60" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="80" y="85" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#073642">1. Identificar</text>
  <text x="80" y="105" text-anchor="middle" font-family="Inter" font-size="11" fill="#586E75">ADF, ACF, PACF</text>
  <path d="M 140 90 L 180 90" stroke="#CB4B16" stroke-width="2"/>

  <rect x="180" y="60" width="120" height="60" rx="8" fill="#EEE8D5" stroke="#CB4B16" stroke-width="2"/>
  <text x="240" y="85" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#073642">2. Estimar</text>
  <text x="240" y="105" text-anchor="middle" font-family="Inter" font-size="11" fill="#586E75">Máxima Verosim.</text>
  <path d="M 300 90 L 340 90" stroke="#CB4B16" stroke-width="2"/>

  <rect x="340" y="60" width="130" height="60" rx="8" fill="#EEE8D5" stroke="#2AA198" stroke-width="2"/>
  <text x="405" y="85" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#073642">3. Diagnosticar</text>
  <text x="405" y="105" text-anchor="middle" font-family="Inter" font-size="11" fill="#586E75">Ljung-Box, Residuos</text>
  <path d="M 470 90 L 510 90" stroke="#2AA198" stroke-width="2"/>

  <!-- Flecha de feedback si falla diagnóstico -->
  <path d="M 405 60 L 405 30 L 80 30 L 80 60" fill="none" stroke="#DC322F" stroke-width="2" stroke-dasharray="4"/>
  <text x="240" y="24" text-anchor="middle" font-family="Inter" font-size="11" fill="#DC322F">Si los residuos no son ruido blanco</text>

  <rect x="510" y="60" width="120" height="60" rx="8" fill="#FDF6E3" stroke="#859900" stroke-width="3"/>
  <text x="570" y="85" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#859900">4. Pronosticar</text>
  <text x="570" y="105" text-anchor="middle" font-family="Inter" font-size="11" fill="#073642">Bandas al 95%</text>
</svg>

---

## Criterios de Selección: AIC vs. BIC

Penalizan la función de máxima verosimilitud $\ln \hat{L}$ en función del número de parámetros libres $k$ y el tamaño muestral $n$:

$$
\operatorname{AIC} = 2k - 2\ln \hat{L}
$$

$$
\operatorname{BIC} = k \ln n - 2\ln \hat{L}
$$

- **BIC penaliza con más severidad** la complejidad cuando $n$ es grande ($\ln n > 2$ para $n > 8$). Tiende a preferir modelos más parsimoniosos. <!-- .element: class="fragment" -->
- **¡Regla Inquebrantable!** Solo se pueden comparar valores de AIC/BIC entre modelos estimados con **exactamente los mismos datos y el mismo orden de diferenciación ($d, D$)**. <!-- .element: class="fragment" -->

---

## Diagnóstico de Residuos: Prueba Ljung-Box

Evalúa si las autocorrelaciones de los residuos hasta el rezago $m$ son simultáneamente cero:

$$
Q = n(n + 2) \sum_{k=1}^m \frac{\hat{r}_k^2}{n - k} \sim \chi^2(m - p - q)
$$

- **Hipótesis nula ($H_0$):** Los residuos son ruido blanco independiente (sin autocorrelación restante).
- Buscamos un **$p$-valor $> 0.05$**. Si $p < 0.05$, rechazamos $H_0$, lo que indica que el modelo no ha extraído toda la dinámica temporal y se debe aumentar $p$ o $q$.

---

## Paquetes en Python: SARIMAX

```python [1-5|7-11|13-16]
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# 1) Ajuste del modelo:
model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False
).fit(disp=False)

# 2) Diagnóstico y bondad de ajuste:
print(f"AIC: {model.aic:.1f} | BIC: {model.bic:.1f}")
lb_test = acorr_ljungbox(model.resid.dropna(), lags=[12, 24], return_df=True)

# 3) Pronóstico con intervalos de confianza al 95%:
fc = model.get_forecast(steps=24).summary_frame(alpha=0.05)
# Columnas: mean, mean_se, mean_ci_lower, mean_ci_upper
```

--

### Diagnóstico Gráfico Automatizado

El método `model.plot_diagnostics(figsize=(10, 6))` produce 4 paneles indispensables:

<div class="highlight-box">
  <ul>
    <li><strong>Residuos estandarizados en el tiempo:</strong> Deben lucir como ruido blanco centrado en cero sin varianza variable.</li>
    <li><strong>Histograma y densidad KDE:</strong> Comparación directa con la curva normal estándar $N(0, 1)$.</li>
    <li><strong>Gráfica Q-Q Normal:</strong> Puntos alineados sobre la diagonal recta de 45 grados.</li>
    <li><strong>Correlograma (ACF) del residuo:</strong> Ninguna barra debe traspasar los límites de significancia.</li>
  </ul>
</div>

Note:
[01:45 - 02:00]
Si los puntos en la gráfica Q-Q se desvían fuertemente en las colas, los intervalos de confianza pueden subestimar la probabilidad de eventos extremos.

---

## Actividad del Reto (02:00–02:50)

<div class="highlight-box">
  <strong>Laboratorio:</strong> Abrir <code>notebooks/N4_Atipicos_ARIMA_SARIMA.ipynb</code>.
</div>

- Detectar atípicos con la regla de la MAD y tratarlos con máscara `NaN`.
- Ejecutar una rejilla sistemática de modelos SARIMA fijando $d=1, D=1, s=12$ y variando $p, q \in \{0, 1, 2\}$ y $P, Q \in \{0, 1\}$.
- Ordenar la tabla de modelos por **BIC** y seleccionar el mejor candidato.
- Realizar el diagnóstico integral (Ljung-Box y `plot_diagnostics`).
- Generar el pronóstico a 24 pasos y graficar la serie real junto a la media estimada y su banda sombreada de incertidumbre al 95%.

---

## Cierre y Pregunta Abierta (02:50–03:00)

<div class="def-box">
  <strong>Logro de hoy:</strong> Hemos construido un modelo que no solo supera en interpretabilidad a las cajas negras neuronales, sino que provee una cuantificación probabilística rigurosa del riesgo futuro.
</div>

- **Pregunta para la Semana 08:** Si un directivo de planta nos dice: *"Necesito actualizar el pronóstico todos los lunes por la mañana en milisegundos con cada nueva lectura sin tener que reestimar una matriz hessiana pesada"*, ¿qué familia de modelos de suavizado adaptativo puede responder a esta necesidad, y quién ganará el gran benchmark final?

Note:
[02:55 - 03:00]
Entregable: tabla comparativa de la rejilla SARIMA, diagnósticos del modelo seleccionado y gráfico de pronóstico con intervalo.
Enlace: notebooks/html/N4_Atipicos_ARIMA_SARIMA.html
