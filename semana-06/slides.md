# Semana 06: Estacionalidad, ACF/PACF y Análisis Espectral
### Dinámica Temporal, Correlación Cruzada y Dominio de Frecuencia
**MA3001B: Modelos y TDA • Profesor B**  
Tecnológico de Monterrey • 8vo Semestre

Note:
[00:00 - 00:05]
Bienvenida a la Semana 6.
Hoy sustituimos las decisiones arbitrarias ("a ojo") de arquitectura por evidencia estadística directa proveniente del análisis temporal y espectral.
Duración: 180 min.

---

## Agenda de la Sesión (180 min)

| Minutos | Bloque | Actividad y Enfoque |
| :--- | :--- | :--- |
| **00:00–00:25** | **Detonador** | La ventana arbitraria y el ciclo oculto sospechado |
| **00:25–01:15** | **Teoría** | Autocovarianza, ACF, PACF, CCF, Fourier y Periodograma |
| **01:15–01:25** | **Descanso** | Pausa intermedia |
| **01:25–02:00** | **Paquetes** | `plot_acf`, `plot_pacf`, `ccf`, `scipy.signal.periodogram` |
| **02:00–02:50** | **Reto** | Signo de CCF, atención vs. PACF y reentrenamiento de red |
| **02:50–03:00** | **Cierre** | Justificación rigurosa de $L$ y pregunta para Semana 7 |

Note:
[00:05 - 00:10]
Presentar los dos mundos complementarios: el dominio del tiempo (ACF/PACF) y el dominio de la frecuencia (espectro de Fourier).

---

## Problema Detonador (00:00–00:25)

<div class="alert-box">
  <strong>¿Ventana científica o intuición a ciegas?</strong> En la Semana 4 fijamos $L=24$ porque "parecía razonable para dos años". Además, el equipo de operaciones sospecha que una variable externa económica se adelanta a nuestra serie y que existe una periodicidad que no es obvia a simple vista.
</div>

--

### Código del Problema Detonador

```python [1-6|8-13]
# ¿Cómo elegimos L en la práctica?
L_arbitrario = 24  # ¿Por qué no 12, 17 o 36?

# Si L es muy pequeño: perdemos dependencias lejanas de largo plazo.
# Si L es muy grande: inflamos parámetros, agregamos ruido y costo de cómputo.

# Además: ¿esta variable macroeconómica x predice a y?
correlacion_bruta = np.corrcoef(serie_x, serie_y)[0, 1]
print("Correlación instantánea:", correlacion_bruta)  # 0.85
# ¡Pero 0.85 no nos dice si x adelanta a y, si y adelanta a x,
# o si ambas suben por una tendencia compartida espuria!
```

--

### Preguntas para los Equipos (15 min)

<div class="highlight-box">
  <ol>
    <li>¿Cómo distinguimos entre una correlación directa entre $y_t$ y $y_{t-k}$ y la correlación indirecta transmitida a través de los pasos intermedios?</li>
    <li>¿Por qué correlacionar dos series no estacionarias produce correlaciones espurias cercanas a $\pm 1$ aunque las variables no tengan ninguna relación causal?</li>
    <li>¿Cómo calculamos matemáticamente el periodo exacto de oscilación de una señal sin necesidad de contar picos manualmente?</li>
  </ol>
</div>

Note:
[00:15 - 00:25]
Discusión guiada hacia la definición de autocorrelación parcial y descomposición de Fourier.

---

## Objetivos de Aprendizaje

- Interpretar la **Función de Autocorrelación (ACF)** y la **Autocorrelación Parcial (PACF)**. <!-- .element: class="fragment" -->
- Reconocer las firmas teóricas de procesos autorregresivos (AR), de medias móviles (MA) y estacionales. <!-- .element: class="fragment" -->
- Aplicar la **Correlación Cruzada (CCF)** para identificar indicadores líderes, evitando correlaciones espurias. <!-- .element: class="fragment" -->
- Dominar el **análisis espectral**: Transformada Discreta de Fourier (DFT), periodograma y ventana Welch. <!-- .element: class="fragment" -->
- Contrastar empíricamente los pesos de atención del Transformer con la magnitud de la PACF. <!-- .element: class="fragment" -->

---

## Teoría: ACF y PACF

Para un proceso estacionario con media $\mu$, la autocovarianza y la autocorrelación al rezago $k$ se definen como:

$$
\gamma(k) = E[(y_t - \mu)(y_{t-k} - \mu)], \qquad \rho(k) = \frac{\gamma(k)}{\gamma(0)}
$$

La **ACF** mide la correlación total entre $y_t$ y $y_{t-k}$.

--

### Función de Autocorrelación Parcial (PACF)

La **PACF** ($\phi_{kk}$) mide la correlación condicional entre $y_t$ y $y_{t-k}$ habiendo eliminado el efecto lineal de todos los rezagos intermedios $\{y_{t-1}, y_{t-2}, \dots, y_{t-k+1}\}$:

$$
y_t = \phi_{k1} y_{t-1} + \phi_{k2} y_{t-2} + \dots + \phi_{kk} y_{t-k} + e_t
$$

- Las bandas de significancia estadística a un nivel del 95% para una muestra de tamaño $n$ se sitúan en:
  $$\pm \frac{1.96}{\sqrt{n}}$$
- Toda barra que exceda esta banda representa una correlación estadísticamente no nula.

--

### Patrones Teóricos de Identificación

| Proceso | Función ACF | Función PACF |
| :--- | :--- | :--- |
| **AR($p$)** | Decae exponencialmente o sinusoide amortiguada | **Se trunca abruptamente** después del rezago $p$ |
| **MA($q$)** | **Se trunca abruptamente** después del rezago $q$ | Decae exponencialmente o sinusoide amortiguada |
| **ARMA($p, q$)** | Decaimiento gradual infinito | Decaimiento gradual infinito |
| **Estacional $s$** | Picos prominentes en rezagos $s, 2s, 3s, \dots$ | Pico marcado en el rezago estacional $s$ |
| **Tendencia sin filtrar** | Decae con extrema lentitud, cercano a 1 | Barra gigantesca en rezago 1, resto insignificante |

Note:
[00:45 - 00:55]
Enfatizar: si la ACF no decae rápido a cero, ¡la serie aún no está estacionarizada!

---

## Correlación Cruzada (CCF)

Mide la dependencia lineal mutua entre dos series estacionarias $\{x_t\}$ e $\{y_t\}$ separadas por un desplazamiento temporal $k$:

$$
\rho_{xy}(k) = \frac{E[(x_{t+k} - \mu_x)(y_t - \mu_y)]}{\sigma_x \sigma_y}
$$

- **¡Regla de Oro!** Ambas series deben pre-blanquearse o diferenciarse antes de calcular la CCF. Si se introducen series con tendencia común, la CCF mostrará correlación alta en todos los rezagos por puro artificio. <!-- .element: class="fragment" -->

--

### La Convención de Signo en statsmodels

En `statsmodels.tsa.stattools.ccf(a, b, adjusted=False)[k]`:

- Se calcula la correlación entre $a_{t+k}$ y $b_t$.
- Si una variable líder $x$ precede a $y$ por 3 periodos ($y_t \approx x_{t-3}$), entonces al evaluar $y_{t+3} \approx x_t$.
- Por lo tanto, la llamada `ccf(y, x)` mostrará su valor máximo exactamente en el rezago **$k=3$**.

Note:
[01:05 - 01:15]
Verificamos este resultado con simulaciones controladas: para probar si x adelanta a y, colocamos y como primer argumento en ccf.

---

## Análisis Espectral y Periodograma

Cualquier serie temporal finita de longitud $N$ puede descomponerse en una suma ponderada de senos y cosenos ortogonales mediante la **Transformada Discreta de Fourier (DFT)**:

$$
y_t = \sum_{j=0}^{\lfloor N/2 \rfloor} \left[ A_j \cos(2\pi f_j t) + B_j \sin(2\pi f_j t) \right]
$$

Con frecuencias armónicas discretas:

$$
f_j = \frac{j}{N}, \qquad j = 0, 1, \dots, \lfloor N/2 \rfloor
$$

--

### Frecuencia de Nyquist y Periodo

- La frecuencia máxima captable sin aliasing es la **frecuencia de Nyquist**:
  $$f_{\text{Nyquist}} = \frac{f_s}{2} = 0.5 \quad \text{(ciclos por muestra)}$$
- La relación física fundamental entre frecuencia $f$ y periodo temporal $T$ es:
  $$T = \frac{1}{f} \quad \text{(muestras por ciclo completo)}$$
  *(Ejemplo: Si $f = 1/12 \approx 0.0833$ en datos mensuales, el periodo es $T = 12$ meses).*

--

### El Periodograma y la Fuga Espectral

El periodograma cuantifica la densidad espectral de potencia en cada frecuencia:

$$
I(f_j) = \frac{1}{N} \left| \sum_{t=1}^N y_t e^{-2\pi i f_j t} \right|^2
$$

- **Fuga Espectral (Spectral Leakage):** Si la periodicidad verdadera no es un divisor exacto de $N$, la energía se desparrama hacia armónicos vecinos.
- **Mitigación:** Aplicar una ventana de suavizado (`window='hann'`) o el método de **Welch** (`scipy.signal.welch`), promediando subsegmentos solapados.

---

## Paquetes en Python

```python [1-5|7-12]
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf
from scipy.signal import periodogram, find_peaks

# 1) Diagnóstico en dominio del tiempo:
fig_acf = plot_acf(y_diff, lags=36, alpha=0.05)
fig_pacf = plot_pacf(y_diff, lags=36, method='ywm', alpha=0.05)

# 2) Análisis espectral en dominio de frecuencia:
# fs = 1.0 (1 dato por mes)
freqs, potencias = periodogram(y_sin_tendencia, fs=1.0, window='hann')
picos, _ = find_peaks(potencias, prominence=potencias.max() * 0.05)
periodos_dominantes = 1.0 / freqs[picos]
```

--

### Errores Comunes de Análisis Dinámico

<div class="alert-box">
  <ul>
    <li><strong>Confundir frecuencia con periodo:</strong> Una frecuencia $f = 0.083$ no significa 0.083 meses; el ciclo es $1 / 0.083 \approx 12$ meses.</li>
    <li><strong>Calcular ACF sobre series no estacionarias:</strong> Las barras se mantendrán pegadas a 1.0 por decenas de rezagos, ocultando la estructura interna.</li>
    <li><strong>Omitir el parámetro de frecuencia <code>fs</code>:</strong> Si los datos son trimestrales o diarios y no se especifica la escala de tiempo, la unidad de $1/f$ quedará en pasos muestrales discretos.</li>
  </ul>
</div>

Note:
[01:50 - 02:00]
Recordar que las barras de la PACF en statsmodels en lag 0 siempre valen 1 por definición.

---

## Actividad del Reto (02:00–02:50)

<div class="highlight-box">
  <strong>Laboratorio:</strong> Abrir <code>notebooks/N3_ACF_PACF_CCF_Espectral.ipynb</code>.
</div>

- Identificar rezagos significativos con ACF y PACF de la serie diferenciada.
- Obtener el periodograma y confirmar el ciclo estacional con `find_peaks`.
- **Experimento clave: Atención vs. PACF:**
  Superponer los pesos de auto-atención `attn_profile` (extraídos en la Semana 4) con $|\text{PACF}(k)|$.
- Discutir: ¿Coinciden los rezagos donde el Transformer pone peso con los rezagos donde la PACF muestra correlación parcial?
- Reentrenar la LSTM con la ventana óptima $L$ justificada estadísticamente.

--

### Atención vs. PACF: Consideración Teórica

<div class="def-box">
  <strong>Precaución de interpretación:</strong>
  <ul>
    <li>La <strong>PACF</strong> mide correlación <em>lineal</em> aislada e invariante.</li>
    <li>La <strong>atención</strong> del Transformer es un ponderador dinámico no lineal aprendido condicionalmente a la entrada observada.</li>
    <li>Que coincidan valida la intuición física del modelo; que discrepen no implica necesariamente que uno esté equivocado.</li>
  </ul>
</div>

---

## Cierre y Pregunta Abierta (02:50–03:00)

<div class="def-box">
  <strong>Síntesis de la sesión:</strong> Ahora tenemos evidencia matemática irrefutable: conocemos el orden de integración $d$, el ciclo estacional $s=12$, y los rezagos directos $p$.
</div>

- **Pregunta para la Semana 07:** Si ya conocemos exactamente los rezagos autorregresivos $p$, las diferencias $d$ y el ciclo estacional $s$, ¿cuál es el modelo paramétrico lineal explícito que aprovecha esta estructura entregando intervalos de pronóstico exactos y criterios objetivos (AIC/BIC) de selección?

Note:
[02:55 - 03:00]
Entregable: reporte con gráfica superpuesta PACF vs Atención y los periodos calculados con el periodograma.
Enlace: notebooks/html/N3_ACF_PACF_CCF_Espectral.html
