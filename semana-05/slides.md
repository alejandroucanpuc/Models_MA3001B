# Semana 05: Procesos Estocásticos y Descomposición
### Estacionariedad Débil, Pruebas ADF/KPSS y Descomposición STL
**MA3001B: Modelos y TDA • Profesor B**  
Tecnológico de Monterrey • 8vo Semestre

Note:
[00:00 - 00:05]
Bienvenida a la Semana 5.
Hoy conectamos el fallo empírico observado en la Semana 4 (el aplanamiento de las redes) con la teoría matemática rigurosa de procesos estocásticos y estacionariedad.
Duración: 180 min.

---

## Agenda de la Sesión (180 min)

| Minutos | Bloque | Actividad y Enfoque |
| :--- | :--- | :--- |
| **00:00–00:25** | **Detonador** | Pronóstico aplanado y el engaño del split aleatorio |
| **00:25–01:15** | **Teoría** | Estacionariedad débil, pruebas ADF y KPSS, Box-Cox |
| **01:15–01:25** | **Descanso** | Pausa intermedia |
| **01:25–02:00** | **Paquetes** | `adfuller`, `kpss`, `seasonal_decompose`, `STL` |
| **02:00–02:50** | **Reto** | Diagnóstico y LSTM con inversión de diferencias |
| **02:50–03:00** | **Cierre** | Declaración de transformaciones y pregunta para Sem. 6 |

Note:
[00:05 - 00:10]
Explicar el valor práctico: ninguna empresa pronostica series en bruto sin antes realizar un análisis de estacionariedad.

---

## Problema Detonador (00:00–00:25)

<div class="alert-box">
  <strong>El doble engaño del modelado ingenuo:</strong>
  <ol>
    <li>La LSTM entrenada en la Semana 4 se aplana en un valor medio constante en el conjunto de prueba mientras la serie real continúa subiendo.</li>
    <li>Un alumno propone usar una regresión lineal con <code>train_test_split(shuffle=True)</code> y obtiene un $R^2 = 0.98$; sin embargo, en producción el error es colosal.</li>
  </ol>
</div>

--

### Código del Problema Detonador

```python [1-5|7-12]
# Error típico: split aleatorio en series temporales
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_idx = np.arange(len(y)).reshape(-1, 1)
X_tr, X_te, y_tr, y_te = train_test_split(X_idx, y, test_size=0.2, shuffle=True)
reg = LinearRegression().fit(X_tr, y_tr)
print("R2 en test con shuffle:", reg.score(X_te, y_te))  # ¡0.987!

# Pero en la realidad el test es el futuro:
X_futuro = np.arange(len(y), len(y) + 24).reshape(-1, 1)
# Si la serie cambia de pendiente o tiene ciclos, el split aleatorio
# aprendió a interpolar puntos vecinos, no a pronosticar hacia adelante.
```

--

### Preguntas para los Equipos (15 min)

<div class="highlight-box">
  <ol>
    <li>¿Por qué mezclar observaciones con <code>shuffle=True</code> constituye una violación matemática de la hipótesis de independencia temporal?</li>
    <li>¿Por qué las funciones de activación acotadas ($\tanh, \sigma$) o capas lineales entrenadas en un rango $[y_{\min}, y_{\max}]$ no pueden extrapolar una tendencia estrictamente creciente?</li>
    <li>¿Qué transformación matemática elemental permitiría convertir una serie creciente en una serie oscilatoria alrededor de cero?</li>
  </ol>
</div>

Note:
[00:15 - 00:25]
Guiar a los alumnos hacia la idea de diferenciación $y_t - y_{t-1}$.

---

## Objetivos de Aprendizaje

- Distinguir entre un **proceso estocástico** y una **realización observada**. <!-- .element: class="fragment" -->
- Definir con rigor la **estacionariedad en sentido débil** (media, varianza y autocovarianza invariantes). <!-- .element: class="fragment" -->
- Ejecutar e interpretar conjuntamente las pruebas de hipótesis **ADF** y **KPSS**. <!-- .element: class="fragment" -->
- Estacionarizar series mediante transformaciones de potencia (Box-Cox) y operadores de diferencias ($\Delta$). <!-- .element: class="fragment" -->
- Aplicar descomposición de series (clásica aditiva/multiplicativa y **STL**) para aislar tendencia y estacionalidad. <!-- .element: class="fragment" -->

---

## Teoría: Procesos Estocásticos

Un proceso estocástico es una familia de variables aleatorias $\{Y_t : t \in T\}$ indexadas por el tiempo. La serie temporal registrada en datos es una **única realización** muestral de dicho proceso.

<svg viewBox="0 0 650 180" style="width:100%; max-height:200px; margin: 10px auto; display:block;">
  <rect x="20" y="20" width="610" height="140" rx="10" fill="#FDF6E3" stroke="#B58900" stroke-width="2"/>
  <text x="40" y="45" font-family="Crimson Text" font-weight="bold" font-size="16" fill="#073642">Espacio Muestral vs. Realización Observada</text>

  <!-- Trayectoria 1 fantasma -->
  <path d="M 40 100 Q 120 70, 200 110 T 360 80 T 520 120 T 600 70" fill="none" stroke="#93A1A1" stroke-width="1.5" stroke-dasharray="3"/>
  <!-- Trayectoria 2 fantasma -->
  <path d="M 40 120 Q 120 140, 200 90 T 360 130 T 520 70 T 600 110" fill="none" stroke="#93A1A1" stroke-width="1.5" stroke-dasharray="3"/>
  <!-- Trayectoria observada -->
  <path d="M 40 110 Q 120 90, 200 100 T 360 95 T 520 105 T 600 90" fill="none" stroke="#CB4B16" stroke-width="3"/>
  <text x="500" y="55" font-family="Inter" font-weight="bold" font-size="13" fill="#CB4B16">Realización observada (Datos)</text>
</svg>

--

### Estacionariedad Débil (o de Segundo Orden)

Un proceso $\{y_t\}$ es **débilmente estacionario** si satisface simultáneamente tres condiciones para todo $t$ y para todo rezago $k$:

$$
1.\quad E[y_t] = \mu \quad \text{(Media constante en el tiempo)}
$$

$$
2.\quad \operatorname{Var}(y_t) = \sigma^2 < \infty \quad \text{(Varianza finita y constante)}
$$

$$
3.\quad \operatorname{Cov}(y_t, y_{t+k}) = \gamma(k) \quad \text{(Covarianza dependiente solo de } k\text{)}
$$

--

### Ejemplos Canónicos

- **Ruido Blanco ($\varepsilon_t \sim WN(0, \sigma^2)$):**
  $$E[\varepsilon_t] = 0, \quad \operatorname{Var}(\varepsilon_t) = \sigma^2, \quad \operatorname{Cov}(\varepsilon_t, \varepsilon_{t+k}) = 0 \quad (k \neq 0)$$
  Es estrictamente estacionario. <!-- .element: class="fragment" -->

- **Caminata Aleatoria ($y_t = y_{t-1} + \varepsilon_t$):**
  $$y_t = y_0 + \sum_{i=1}^t \varepsilon_i \implies \operatorname{Var}(y_t) = t \sigma^2$$
  La varianza diverge con $t \to \infty$. **No es estacionaria**. <!-- .element: class="fragment" -->
  Sin embargo, su primera diferencia $\Delta y_t = y_t - y_{t-1} = \varepsilon_t$ sí lo es. <!-- .element: class="fragment" -->

Note:
[00:40 - 00:50]
Destacar que la caminata aleatoria posee una raíz unitaria ($\phi=1$).

---

## Pruebas de Hipótesis: ADF vs. KPSS

Para diagnosticar estacionariedad se deben contrastar dos pruebas con hipótesis nulas opuestas:

- **Augmented Dickey-Fuller (ADF):**
  $$H_0: \text{Posee raíz unitaria (No estacionaria)} \quad \text{vs.} \quad H_1: \text{Es estacionaria}$$

- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin):**
  $$H_0: \text{La serie es estacionaria} \quad \text{vs.} \quad H_1: \text{Posee raíz unitaria}$$

--

### Matriz de Decisión Cruzada

| ADF ($H_0$: raíz unitaria) | KPSS ($H_0$: estacionaria) | Diagnóstico Riguroso | Acción Recomendada |
| :---: | :---: | :--- | :--- |
| **Rechaza** ($p < 0.05$) | **No rechaza** ($p \ge 0.05$) | **Estacionaria** | Modelar directamente |
| **No rechaza** ($p \ge 0.05$) | **Rechaza** ($p < 0.05$) | **No estacionaria** (raíz unitaria) | Diferenciar ($\Delta y_t$) |
| **Rechaza** ($p < 0.05$) | **Rechaza** ($p < 0.05$) | **Trend-stationarity** | Remover tendencia lineal |
| **No rechaza** ($p \ge 0.05$) | **No rechaza** ($p \ge 0.05$) | **Evidencia inconclusa** | Inspeccionar gráfica |

Note:
[00:55 - 01:05]
Advertir a los alumnos: un p-valor bajo en ADF significa estacionariedad, pero en KPSS significa no estacionariedad.

---

## Transformaciones de Estacionarización

Si la varianza aumenta con el nivel de la serie, aplicar primero **Box-Cox**:

$$
w_t = \begin{cases}
\dfrac{y_t^\lambda - 1}{\lambda}, & \text{si } \lambda \neq 0 \\
\ln(y_t), & \text{si } \lambda = 0
\end{cases}
$$

--

### El Operador de Diferencias $\Delta$

$$
\Delta y_t = y_t - y_{t-1} = (1 - B) y_t
$$

- **Diferencia Estacional:**
  $$\Delta_s y_t = y_t - y_{t-s} = (1 - B^s) y_t$$
  Elimina el ciclo de periodo $s$ (e.g. $s=12$ en series mensuales).
- **Sobrediferenciación:**
  Si se diferencia de más una serie ya estacionaria, se induce autocorrelación negativa espuria y varianza inflada.

---

## Descomposición de Series de Tiempo

Separa la serie observada en tres componentes no observables:

$$
\text{Aditiva:} \quad y_t = T_t + S_t + R_t
$$

$$
\text{Multiplicativa:} \quad y_t = T_t \times S_t \times R_t \iff \ln y_t = \ln T_t + \ln S_t + \ln R_t
$$

- **Aditiva:** La amplitud de las oscilaciones estacionales es constante en el tiempo.
- **Multiplicativa:** La amplitud crece proporcionalmente al nivel de la tendencia.

--

### Descomposición Clásica vs. STL

- **Clásica (`seasonal_decompose`):**
  Estima $T_t$ con promedios móviles simétricos y $S_t$ promediando cada periodo del año. Pierde observaciones al inicio y final.
- **STL (Seasonal and Trend decomposition using Loess):**
  Utiliza regresiones locales ponderadas (Loess). Es versátil, admite estacionalidades dinámicas y ofrece modo robusto (`robust=True`) ante atípicos.

---

## Comandos en Python: statsmodels

```python [1-6|8-13]
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# 1) Pruebas de hipótesis:
adf_stat, adf_p, _, _, _, _ = adfuller(y, regression='c', autolag='AIC')
kpss_stat, kpss_p, _, _ = kpss(y, regression='c', nlags='auto')

# 2) Descomposición de la serie:
desc_add = seasonal_decompose(y, model='additive', period=12)
stl_res = STL(y, period=12, robust=True).fit()

# Acceso a componentes:
trend = stl_res.trend
season = stl_res.seasonal
resid = stl_res.resid
```

--

### Errores Típicos en la Práctica

<div class="alert-box">
  <ul>
    <li><strong>Confundir hipótesis nulas:</strong> Interpretar $p < 0.05$ en KPSS como "la serie es estacionaria" (es exactamente al revés).</li>
    <li><strong>Olvidar invertir la transformación:</strong> Entrenar un modelo sobre $\Delta y_t$ y calcular el RMSE directamente contra $y_t$ sin reconstruir la serie sumando acumulativamente:
    $$\hat{y}_t = y_{t-1} + \widehat{\Delta y}_t$$</li>
    <li><strong>Diferenciar dos veces sin necesidad:</strong> Genera un proceso MA(1) artificial con raíz cerca del círculo unitario.</li>
  </ul>
</div>

Note:
[01:50 - 02:00]
Enfatizar el paso de reconstrucción del nivel.

---

## Actividad del Reto (02:00–02:50)

<div class="highlight-box">
  <strong>Laboratorio:</strong> Abrir <code>notebooks/N2_Estacionariedad_Descomposicion.ipynb</code>.
</div>

- Generar el reporte comparativo ADF y KPSS para: nivel original, primera diferencia $\Delta y_t$, y diferencia estacional $\Delta_{12} y_t$. <!-- .element: class="fragment" -->
- Ajustar `seasonal_decompose` y `STL`, graficar los residuos y justificar si la serie del Reto es aditiva o multiplicativa. <!-- .element: class="fragment" -->
- Reentrenar la LSTM de la Semana 4 sobre la serie estacionarizada e **invertir la transformación** para reportar el RMSE final en la escala original. <!-- .element: class="fragment" -->

---

## Cierre y Pregunta Abierta (02:50–03:00)

<div class="def-box">
  <strong>Conclusión de la sesión:</strong> Transformar la serie a estacionaria resuelve el problema del aplanamiento en redes y estabiliza las distribuciones de entrada para cualquier modelo probabilístico.
</div>

- **Pregunta para la Semana 06:** Una vez que eliminamos la tendencia y la serie es estacionaria en media, ¿cuántos rezagos del pasado ($y_{t-1}, y_{t-2}, \dots$) tienen correlación directa con el presente, y cuál es el periodo exacto de los ciclos oscilatorios remanentes?

Note:
[02:55 - 03:00]
Entregable: tabla ADF/KPSS y gráfica de componentes STL con explicación en la bitácora.
Enlace: notebooks/html/N2_Estacionariedad_Descomposicion.html
