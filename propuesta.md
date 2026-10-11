# Propuesta adaptada Profesor B: Redes Secuenciales, Transformers y Series de Tiempo (Semanas 4-8)

Oct 10, 2026 · @Alejandro Ucan-Puc

Tus 5 sesiones de 3 h (15 h) quedan así: la semana 4 cubre redes secuenciales y Transformers, y las semanas 5 a 8 cubren series de tiempo completas, con el TDA ya visto con el Profesor A.

## 1. Resumen de la adaptación y supuestos

Cada sesión conserva la lógica ABP del documento original (arranca con un problema detonador sobre el Reto), pero ahora se organiza en cuatro bloques: teoría básica, descripción del paquete (comandos y parámetros), ejemplo aplicado y actividad del Reto.

**Qué cambió respecto a la propuesta original:**

- RNN, LSTM, GRU y Transformer, que estaban repartidos en tres semanas, se comprimen en una sola sesión (semana 4). Es la semana más densa; en la sección de la semana 4 propongo material previo para aliviarla.
- Como el deep learning ahora llega *antes* que la estadística clásica, la semana 4 deja una pregunta abierta ("¿le ganamos a un modelo simple?") que se resuelve en la semana 8 con un benchmark de todos los modelos.
- La comparación "atención del Transformer vs. PACF" pasa a la semana 6, cuando ya se conoce la PACF.
- Todo el código se escribe en Python con PyTorch (redes) y statsmodels, scipy y scikit-learn (series de tiempo).

## 2. Mapa general de las semanas 4 a 8

La secuencia va de "modelo que aprende solo" a "modelo con estructura estadística" y termina comparando ambos sobre la misma serie del Reto.

| Sem. | Tema | Problema detonador (resumen) | Paquetes principales | Entregable (bitácora) |
| --- | --- | --- | --- | --- |
| 4 | RNN, GRU, LSTM y Transformers | Un MLP y una RNN simple se estancan en loss alto con ventanas largas | torch (nn.RNN, nn.LSTM, nn.GRU, nn.TransformerEncoder), scikit-learn | Notebook con LSTM, GRU y Transformer vs. baseline ingenuo |
| 5 | Procesos estocásticos, estacionariedad y descomposición | La red no extrapola la tendencia y la regresión con split aleatorio engaña | statsmodels (adfuller, kpss, seasonal\_decompose, STL), pandas | Diagnóstico de estacionariedad y descomposición de la serie del Reto |
| 6 | Estacionalidad, ACF/PACF, correlación cruzada y análisis espectral | La ventana de la red se eligió "a ojo"; ¿cuál es el ciclo y cuántos rezagos importan? | statsmodels (plot\_acf, plot\_pacf, ccf), scipy.signal (periodogram, welch) | Rezagos y periodos justificados; atención vs. PACF |
| 7 | Atípicos, AR, ARIMA/SARIMA y AIC/BIC | El modelo no da intervalos y los picos sesgan el ajuste | statsmodels (SARIMAX), sklearn (TimeSeriesSplit), pmdarima (opcional) | Mejor SARIMA con diagnóstico y pronóstico con intervalo |
| 8 | Suavizado exponencial y benchmark integrador | Gerencia necesita un pronóstico actualizable y saber qué modelo gana | statsmodels (ExponentialSmoothing), sklearn.metrics | Informe de benchmark: ingenuo, SARIMA, Holt-Winters, LSTM, Transformer |

## 3. Estructura tipo de una sesión de 3 horas

Las semanas 5 a 8 siguen este ritmo de 180 minutos; la semana 4 lo ajusta (se detalla ahí) porque debe cubrir dos familias de modelos.

| Minutos | Bloque | Qué ocurre |
| --- | --- | --- |
| 0–25 | Problema detonador | Se abre con código roto, una gráfica incomprensible o un modelo estancado. Los equipos (3 o 4 personas) lo diagnostican y proponen hipótesis antes de ver teoría. |
| 25–75 | Teoría básica | Mini-clase en dos tramos de 25 min con preguntas intercaladas: definiciones, fórmulas clave y cómo se lee el resultado. |
| 75–85 | Descanso |  |
| 85–120 | Paquete en vivo | Demostración de comandos y parámetros con una tabla de referencia; se muestran los errores típicos de uso. |
| 120–170 | Ejemplo aplicado y actividad del Reto | Primero se resuelve un ejemplo guiado con datos de práctica; después cada equipo lo replica en la serie del Reto. |
| 170–180 | Cierre | Boleto de salida: una conclusión por equipo y la pregunta que queda abierta para la siguiente semana. |

**Material previo sugerido (30 a 45 min, opcional):** un notebook de arranque y una lectura corta, para que la sesión se dedique a diagnosticar y aplicar, no a instalar paquetes.

## 4. Semana 4: RNN, GRU, LSTM y Transformers

Es la sesión más densa: cubre dos familias de modelos en 180 minutos, por lo que la teoría se limita a lo necesario para entender los parámetros del código.

### Problema detonador

Se entrega un notebook que entrena un MLP sobre ventanas de la serie del Reto y una RNN simple con ventanas largas (L = 100 o más). Ambos terminan con una pérdida alta y estancada, y su pronóstico es casi una línea recta. Pregunta al grupo: ¿por qué la red no aprende, y qué información del pasado debería conservar?

### Objetivos

- Explicar por qué una RNN simple pierde información a largo plazo (desvanecimiento del gradiente) y cómo lo atenúan LSTM y GRU.
- Describir la auto-atención como una alternativa que mira todos los pasos de la ventana a la vez.
- Entrenar RNN, GRU, LSTM y un Transformer en PyTorch y compararlos contra un baseline ingenuo con un corte cronológico.

### Cronograma de la sesión

| Minutos | Bloque | Contenido |
| --- | --- | --- |
| 0–20 | Detonador | Notebook "roto": MLP y RNN simple estancados. Hipótesis por equipo. |
| 20–55 | Teoría 1 | RNN, tipos (many-to-one, many-to-many), gradiente que se desvanece o explota, compuertas de LSTM y GRU. |
| 55–85 | Paquete en vivo 1 | nn.RNN, nn.LSTM, nn.GRU: parámetros, formas de entrada y salida, recorte de gradiente. |
| 85–95 | Descanso |  |
| 95–120 | Teoría 2 | Auto-atención, Q/K/V, codificación posicional, máscara causal, multi-cabeza. |
| 120–150 | Paquete en vivo 2 | nn.TransformerEncoderLayer, nn.TransformerEncoder, nn.MultiheadAttention. |
| 150–170 | Actividad del Reto | Entrenar LSTM, GRU y Transformer en la serie del Reto; tabla de RMSE contra el baseline ingenuo. |
| 170–180 | Cierre | ¿Qué modelo ganó? ¿Con qué evidencia? Pregunta abierta: ¿le ganamos a un modelo estadístico? |

**Material previo sugerido:** un notebook corto con `torch.tensor`, `DataLoader` y un MLP de regresión, para que nadie pierda tiempo con lo básico de PyTorch.

### Teoría básica

**RNN.** Procesa la secuencia paso a paso y arrastra un estado oculto que resume el pasado. En *many-to-one* se usa solo el último estado (pronóstico del siguiente valor); en *many-to-many* se usa el estado de cada paso (pronóstico multi-paso o etiquetado).

```latex
h_t = \tanh\left(W_x x_t + W_h h_{t-1} + b\right), \qquad \hat{y} = W_y h_T + c
```

**Desvanecimiento del gradiente.** Al retropropagar en el tiempo, el gradiente hacia un paso lejano k es un producto de (T − k) factores que dependen de W\_h. Si esos factores tienen norma menor que 1, el gradiente se desvanece y la red no aprende dependencias largas; si es mayor que 1, explota (se mitiga con recorte de gradiente).

**LSTM y GRU.** Agregan compuertas que deciden cuánto del pasado se conserva. La LSTM tiene un estado de celda c\_t con compuertas de olvido (f), entrada (i) y salida (o); la GRU funde olvido y entrada en una compuerta de actualización (z) y añade una de reinicio (r).

```latex
\begin{aligned}
f_t &= \sigma(W_f[h_{t-1},x_t]+b_f), \; i_t = \sigma(W_i[h_{t-1},x_t]+b_i), \; o_t = \sigma(W_o[h_{t-1},x_t]+b_o)\\
\tilde{c}_t &= \tanh(W_c[h_{t-1},x_t]+b_c), \; c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t, \; h_t = o_t \odot \tanh(c_t)\\
z_t &= \sigma(W_z[h_{t-1},x_t]), \; r_t = \sigma(W_r[h_{t-1},x_t]), \; \tilde{h}_t=\tanh(W[r_t\odot h_{t-1},x_t]), \; h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde{h}_t
\end{aligned}
```

| Característica | RNN simple | LSTM | GRU |
| --- | --- | --- | --- |
| Estados | h | h y c | h |
| Compuertas | 0 | 3 (olvido, entrada, salida) | 2 (actualización, reinicio) |
| Parámetros relativos | 1× | 4× | 3× |
| Cuándo usarla | Ventanas cortas, línea base | Dependencias largas, más datos | Casi igual que LSTM con menos parámetros |

**Auto-atención.** Cada paso de la ventana genera una consulta (Q), una clave (K) y un valor (V). El peso entre dos pasos es qué tan parecidas son su consulta y su clave; el resultado es un promedio ponderado de los valores. Como no hay recurrencia, el modelo necesita una codificación posicional para saber el orden, y una máscara causal si predice varios pasos para no "ver el futuro".

```latex
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V, \qquad PE_{(pos,2i)}=\sin\frac{pos}{10000^{2i/d}},\; PE_{(pos,2i+1)}=\cos\frac{pos}{10000^{2i/d}}
```

### Paquetes: comandos y parámetros (PyTorch)

| Comando | Parámetros clave | Entrada y salida |
| --- | --- | --- |
| `nn.RNN` / `nn.GRU` / `nn.LSTM` | `input_size` (variables por paso), `hidden_size`, `num_layers=1`, `batch_first=False`, `dropout=0.0`, `bidirectional=False`; solo RNN: `nonlinearity='tanh'` | Con `batch_first=True` la entrada es (lote, L, variables). Devuelve `output` (lote, L, hidden) y `h_n` (en LSTM, `(h_n, c_n)`). |
| `nn.Linear(in, out)` | Cabeza que convierte el último estado en el pronóstico | (lote, hidden) a (lote, 1) |
| `nn.TransformerEncoderLayer` | `d_model`, `nhead` (divide a d\_model), `dim_feedforward=2048`, `dropout=0.1`, `batch_first=False`, `norm_first=False` | Una capa de auto-atención más red feed-forward |
| `nn.TransformerEncoder` | `encoder_layer`, `num_layers`, `norm=None`; al llamar: `mask`, `is_causal` | (lote, L, d\_model) con `batch_first=True` |
| `nn.MultiheadAttention` | `embed_dim`, `num_heads`, `batch_first`; al llamar: `need_weights=True`, `attn_mask` | Devuelve la salida y los pesos de atención (lote, L, L) |
| `nn.Transformer.generate_square_subsequent_mask(L)` | Tamaño L | Máscara causal para pronóstico multi-paso |
| `torch.optim.Adam` | `lr=1e-3`, `weight_decay=0` | Optimizador |
| `nn.utils.clip_grad_norm_` | `parameters`, `max_norm` | Recorte de gradiente (típico: 1.0) |
| `DataLoader` | `batch_size`, `shuffle` | Lotes; mezclar ventanas de *train* es válido, pero el *test* siempre va después en el tiempo |
| `StandardScaler` | `fit` solo con *train* | Evita fuga de información del futuro |

Ejercicio rápido para la clase: con `hidden_size=32` y una variable por paso, comprobar con `sum(p.numel() for p in model.rnn.parameters())` que RNN, GRU y LSTM tienen 1120, 3360 y 4480 parámetros (la proporción 1×, 3×, 4× de la tabla de teoría).

### Ejemplo aplicado: RNN, GRU y LSTM contra un baseline ingenuo

Serie de práctica: CO₂ mensual de Mauna Loa (viene en statsmodels). Se sustituye por la serie del Reto.

```python
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

torch.manual_seed(0)
y = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
n_train = int(len(y) * 0.8)                       # corte cronológico
scaler = StandardScaler().fit(y.iloc[:n_train].to_frame())   # solo con train
z = scaler.transform(y.to_frame()).ravel().astype("float32")

def make_windows(z, L):                           # L pasos -> siguiente valor
    X = np.stack([z[i:i + L] for i in range(len(z) - L)])
    return torch.tensor(X).unsqueeze(-1), torch.tensor(z[L:]).unsqueeze(-1)

L = 24
X, t = make_windows(z, L)                         # (N, L, 1) y (N, 1)
split = n_train - L                               # primer objetivo de test = z[n_train]
train_dl = DataLoader(TensorDataset(X[:split], t[:split]), batch_size=32, shuffle=True)

class SeqModel(nn.Module):
    def __init__(self, kind="lstm", hidden=32):
        super().__init__()
        cell = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
        self.rnn = cell(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):                         # many-to-one
        out, _ = self.rnn(x)                      # (lote, L, hidden)
        return self.head(out[:, -1, :])

def fit(model, dl, epochs=60, lr=1e-3, clip=1.0):
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).squeeze(-1).numpy()

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
true = t[split:].squeeze(-1).numpy()
results = {"ingenuo (valor anterior)": rmse(true, X[split:, -1, 0].numpy())}
for kind in ["rnn", "gru", "lstm"]:
    results[kind] = rmse(true, predict(fit(SeqModel(kind), train_dl), X[split:]))
print(results)                                    # RMSE en unidades estandarizadas
```

**Qué observar:** el baseline ingenuo suele ser muy difícil de superar en una serie con tendencia. Además, el conjunto de prueba queda por encima del rango de entrenamiento, y las redes tienden a quedarse cortas ahí. Ese es el gancho para la semana 5.

### Ejemplo aplicado: Transformer con atención visible

```python
class PosEnc(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):                         # x: (lote, L, d_model)
        return x + self.pe[: x.size(1)]

class TSTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.inp, self.pos = nn.Linear(1, d_model), PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):                         # no hace falta máscara: la ventana ya es pasado
        return self.head(self.enc(self.pos(self.inp(x)))[:, -1, :])

results["transformer"] = rmse(true, predict(fit(TSTransformer(), train_dl), X[split:]))

# Versión de una sola capa para inspeccionar la atención (se usa en la semana 6)
class TinyAttn(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.inp, self.pos = nn.Linear(1, d_model), PosEnc(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.pos(self.inp(x))
        a, w = self.attn(h, h, h, need_weights=True)   # w: (lote, L, L)
        self.w = w.detach()
        return self.head(self.norm(h + a)[:, -1, :])

tiny = fit(TinyAttn(), train_dl)
predict(tiny, X[split:])
attn_last = tiny.w[:, -1, :].mean(0).numpy()      # a qué rezagos atiende el último paso
```

### Actividad del Reto y entregable

Cada equipo corre el notebook sobre la serie del Reto y entrega una bitácora con:

- Tabla de RMSE y MAE en el conjunto de prueba (ingenuo, RNN, GRU, LSTM, Transformer) y curva de pérdida de al menos dos modelos.
- Respuesta a: ¿cambia el resultado si la ventana pasa de L = 12 a L = 48? ¿Y si se mezcla antes de dividir (error que debe identificarse como fuga)?
- Gráfica de `attn_last` contra el número de rezago, que se retoma en la semana 6.

**Dónde se atoran los alumnos:** mezclar las dimensiones (lote, L, variables), olvidar `batch_first=True`, ajustar el escalador con toda la serie y comparar modelos sin un baseline.

## 5. Semana 5: Procesos estocásticos, estacionariedad y descomposición

La sesión explica por qué la red de la semana 4 no extrapola la tendencia y entrega las herramientas para transformar la serie antes de modelarla.

### Problema detonador

Se proyecta la gráfica de pronóstico de la semana 4: la LSTM se "aplana" mientras la serie real sigue subiendo. En paralelo, una regresión lineal con `train_test_split` aleatorio reporta un R² altísimo, pero falla al pronosticar el último año. Pregunta: ¿qué supuestos estamos violando y por qué el baseline ingenuo es tan duro de vencer?

### Objetivos

- Distinguir proceso estocástico de realización, y estacionariedad débil de no estacionariedad (tendencia determinista vs. raíz unitaria).
- Aplicar e interpretar las pruebas ADF y KPSS, y estacionarizar con diferencias, logaritmo o Box-Cox.
- Descomponer una serie en tendencia, estacionalidad y residuo (aditiva y multiplicativa; clásica y STL).

### Cronograma de la sesión

| Minutos | Bloque | Contenido |
| --- | --- | --- |
| 0–25 | Detonador | Revisión del pronóstico aplanado y del split aleatorio. Hipótesis por equipo. |
| 25–75 | Teoría básica | Proceso estocástico, ruido blanco, caminata aleatoria, estacionariedad débil, ADF y KPSS, diferenciación, modelos aditivo y multiplicativo. |
| 75–85 | Descanso |  |
| 85–120 | Paquete en vivo | adfuller, kpss, seasonal\_decompose, STL, diff, rolling, boxcox. |
| 120–170 | Ejemplo y actividad | Reporte de estacionariedad y descomposición de CO₂; luego la serie del Reto. |
| 170–180 | Cierre | Cada equipo declara su transformación y la justifica. Pregunta abierta: ¿qué ciclos y rezagos quedan en la serie? |

### Teoría básica

**Proceso estocástico.** Colección de variables aleatorias indexadas por el tiempo; la serie observada es una sola realización. Un proceso es débilmente estacionario si su media y su varianza son constantes y la covarianza entre dos instantes depende solo de la distancia (el rezago k).

```latex
E[y_t]=\mu,\qquad \mathrm{Var}(y_t)=\sigma^2,\qquad \mathrm{Cov}(y_t,y_{t+k})=\gamma(k)
```

**Ejemplos mínimos.** El *ruido blanco* es estacionario. La *caminata aleatoria* y\_t = y\_(t−1) + ε\_t no lo es: su varianza crece con t, y diferenciarla produce ruido blanco. Un AR(1) con |φ| < 1 es estacionario; con φ = 1 es la caminata aleatoria (raíz unitaria).

**ADF y KPSS** responden preguntas opuestas, por eso se usan juntas:

| ADF (H0: hay raíz unitaria) | KPSS (H0: es estacionaria) | Lectura |
| --- | --- | --- |
| Rechaza | No rechaza | Estacionaria |
| No rechaza | Rechaza | No estacionaria por raíz unitaria: diferenciar |
| Rechaza | Rechaza | Probable estacionariedad alrededor de una tendencia: quitar la tendencia |
| No rechaza | No rechaza | Poca evidencia: apoyarse en la gráfica y en más datos |

**Descomposición.** Separa la serie en tendencia (T), estacionalidad (S) y residuo (R). Es aditiva si la amplitud estacional no depende del nivel; es multiplicativa si crece con el nivel, y en ese caso el logaritmo la vuelve aditiva.

```latex
y_t = T_t + S_t + R_t \qquad \text{o} \qquad y_t = T_t \times S_t \times R_t \;\Longleftrightarrow\; \log y_t = \log T_t + \log S_t + \log R_t
```

### Paquetes: comandos y parámetros (statsmodels, pandas, scipy)

| Comando | Parámetros clave | Devuelve o hace |
| --- | --- | --- |
| `adfuller(x, maxlag=None, regression='c', autolag='AIC')` | `regression`: `'n'` sin constante, `'c'` constante, `'ct'` constante y tendencia, `'ctt'` con tendencia cuadrática | Estadístico, p-valor, rezagos usados, observaciones, valores críticos |
| `kpss(x, regression='c', nlags='auto')` | `regression`: `'c'` (estacionaria en nivel) o `'ct'` (alrededor de tendencia) | Estadístico, p-valor (truncado entre 0.01 y 0.10), rezagos, críticos |
| `seasonal_decompose(x, model='additive', period=None, two_sided=True, extrapolate_trend=0)` | `model`: `'additive'` o `'multiplicative'`; `period` obligatorio si el índice no tiene frecuencia | Objeto con `.trend`, `.seasonal`, `.resid` y `.plot()` |
| `STL(endog, period=None, seasonal=7, trend=None, robust=False).fit()` | `seasonal` (suavidad de la estacionalidad, impar), `robust=True` para atípicos | `.trend`, `.seasonal`, `.resid` |
| `Series.diff(periods=1)` | `periods=12` para diferencia estacional mensual | Diferencia a distancia `periods` |
| `Series.rolling(window).mean()` / `.std()` | `window` | Media y desviación móviles para revisar estabilidad visualmente |
| `scipy.stats.boxcox(x, lmbda=None)` | Solo datos positivos | Serie transformada y el λ estimado |

### Ejemplo aplicado: estacionarizar y descomponer CO₂

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL

y = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()

def reporte(x, nombre):
    x = x.dropna()
    adf_p = adfuller(x, regression="c", autolag="AIC")[1]
    kpss_p = kpss(x, regression="c", nlags="auto")[1]
    print(f"{nombre:<26} ADF p={adf_p:.3f} | KPSS p={kpss_p:.3f}")

reporte(y, "nivel")
reporte(y.diff(), "1ª diferencia")
reporte(y.diff().diff(12), "1ª dif. + dif. estacional")
reporte(np.log(y).diff(), "log + 1ª diferencia")

add = seasonal_decompose(y, model="additive", period=12)
mul = seasonal_decompose(y, model="multiplicative", period=12)
stl = STL(y, period=12, robust=True).fit()
add.plot(); mul.plot(); stl.plot()      # comparar el residuo de cada una
```

**Qué observar:** el nivel no es estacionario y la diferencia sí lo es. La amplitud estacional del CO₂ es casi constante, así que la descomposición aditiva basta y la multiplicativa no aporta. Conviene que los alumnos contrasten esto con una serie cuya amplitud sí crece con el nivel (por ejemplo, ventas de una temporada de alto crecimiento).

### Actividad del Reto y entregable

- Reporte ADF y KPSS de la serie del Reto antes y después de transformarla; decidir si usar diferencia, logaritmo o Box-Cox.
- Descomposición con `seasonal_decompose` y con `STL`; explicar cuál se elige (aditiva o multiplicativa) con una gráfica del residuo.
- Reentrenar la LSTM de la semana 4 sobre la serie estacionarizada (diferencia o residuo de STL) e **invertir** la transformación para comparar el RMSE en la escala original. Para una diferencia simple, el nivel se reconstruye sumando acumulativamente a partir del último valor conocido.

**Dónde se atoran los alumnos:** leer mal la dirección de las hipótesis de ADF y KPSS, diferenciar de más (el residuo queda con autocorrelación negativa fuerte) y olvidar invertir la transformación antes de calcular errores.

## 6. Semana 6: Estacionalidad, ACF/PACF, correlación cruzada y análisis espectral

La sesión reemplaza las decisiones "a ojo" de la semana 4 (ventana, ciclo, variables útiles) por evidencia estadística, y contrasta la atención del Transformer con la PACF.

### Problema detonador

Se muestra el mapa de atención del Transformer de la semana 4 y la ventana L = 24 que se eligió sin justificación. Además, operaciones sospecha que existe un ciclo oculto de periodo desconocido y que otra variable se adelanta a la serie del Reto. Pregunta: ¿cuántos rezagos importan, cuál es el periodo del ciclo, y cómo lo comprobamos con datos?

### Objetivos

- Calcular e interpretar ACF y PACF, y leer los patrones de AR, MA y estacionalidad.
- Medir dependencia entre dos series con correlación cruzada, evitando correlaciones espurias.
- Encontrar periodos dominantes con el periodograma y convertir frecuencia en periodo.

### Cronograma de la sesión

| Minutos | Bloque | Contenido |
| --- | --- | --- |
| 0–25 | Detonador | Mapa de atención y ventana arbitraria. Hipótesis por equipo. |
| 25–75 | Teoría básica | Autocovarianza, ACF, PACF y sus patrones; bandas de confianza; correlación cruzada; DFT, periodograma, frecuencia de Nyquist y fuga espectral. |
| 75–85 | Descanso |  |
| 85–120 | Paquete en vivo | plot\_acf, plot\_pacf, acf, pacf, ccf, periodogram, welch, find\_peaks, month\_plot. |
| 120–170 | Ejemplo y actividad | ACF/PACF y periodograma de CO₂; ccf en macrodata; luego la serie del Reto y atención vs. PACF. |
| 170–180 | Cierre | Cada equipo declara su ventana L y sus periodos, con evidencia. Pregunta abierta: ¿qué modelo explícito usa estos rezagos? |

### Teoría básica

**ACF y PACF.** La ACF mide la correlación de la serie consigo misma a k pasos; la PACF mide la correlación a k pasos *quitando* el efecto de los rezagos intermedios. Con n observaciones, las barras fuera de ±1.96/√n son significativas a 5% (aproximado).

```latex
\rho(k)=\frac{\gamma(k)}{\gamma(0)}, \qquad \rho_{xy}(k)=\frac{\mathrm{Cov}(x_{t+k},\,y_t)}{\sigma_x\sigma_y}
```

| Patrón | ACF | PACF |
| --- | --- | --- |
| AR(p) | Decae gradualmente | Se corta después del rezago p |
| MA(q) | Se corta después del rezago q | Decae gradualmente |
| ARMA(p, q) | Decae | Decae |
| Estacionalidad de periodo s | Picos en s, 2s, 3s | Pico en s |
| Tendencia sin quitar | Decae muy lento, cerca de 1 | Pico grande en el rezago 1 |

**Correlación cruzada.** Mide si una serie se adelanta o se atrasa respecto de otra. Si las series no son estacionarias, la correlación sale alta aunque no haya relación (correlación espuria); por eso primero se diferencian o se quitan sus tendencias.

**Análisis espectral.** La transformada discreta de Fourier descompone la serie en ondas de distintas frecuencias. El periodograma mide cuánta potencia hay en cada frecuencia f (ciclos por observación, hasta 0.5, la frecuencia de Nyquist); el periodo es 1/f.

```latex
I(f_j)=\frac{1}{N}\left|\sum_{t=1}^{N} y_t\, e^{-2\pi i f_j t}\right|^2,\quad f_j=\frac{j}{N},\quad \text{periodo}=\frac{1}{f}
```

Si el ciclo no cabe un número entero de veces en la muestra, la potencia se "derrama" a frecuencias vecinas (fuga espectral). Se atenúa con una ventana (`hann`) o con el método de Welch, que promedia varios segmentos.

### Paquetes: comandos y parámetros

| Comando | Parámetros clave | Devuelve o hace |
| --- | --- | --- |
| `plot_acf(x, lags=None, alpha=0.05, fft=False, bartlett_confint=True)` | `lags` (cuántos rezagos), `alpha` (nivel de la banda) | Gráfica de ACF con banda de confianza |
| `plot_pacf(x, lags=None, alpha=0.05, method='ywm')` | `method`: `'ywm'` (recomendado), `'ols'` | Gráfica de PACF |
| `acf(x, nlags=None, qstat=False, alpha=None)` y `pacf(x, nlags=None, method='ywm')` | `qstat=True` añade el estadístico Q de Ljung-Box | Valores numéricos por rezago |
| `ccf(x, y, adjusted=True)` | `adjusted=False` divide entre n | Correlación cruzada en rezagos 0, 1, 2, ... (verificar convención de signo, ver ejemplo) |
| `month_plot(x)` | Serie con índice de fechas con frecuencia | Gráfica por mes para ver la estacionalidad |
| `scipy.signal.periodogram(x, fs=1.0, window='boxcar', detrend='constant', scaling='density')` | `fs` = observaciones por unidad de tiempo; `window='hann'` reduce la fuga | Frecuencias y potencia |
| `scipy.signal.welch(x, fs=1.0, window='hann', nperseg=None)` | `nperseg` = tamaño de segmento (más pequeño, más suave y menos resolución) | Espectro promediado |
| `scipy.signal.find_peaks(x, height=None, distance=None, prominence=None)` | `prominence` filtra picos pequeños | Índices de los picos |
| `numpy.fft.rfft(x)` y `numpy.fft.rfftfreq(n, d=1.0)` | `d` = intervalo entre observaciones | Transformada y frecuencias asociadas |

### Ejemplo aplicado: rezagos, ciclos y una variable líder

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.seasonal import STL
from scipy.signal import periodogram, find_peaks

y = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()

# 1) ACF y PACF sobre la serie diferenciada (estacionaria en media)
x = y.diff().dropna()
plot_acf(x, lags=48); plot_pacf(x, lags=48, method="ywm")   # picos en 12, 24, 36 = estacionalidad

# 2) Periodograma de la serie sin tendencia; fs = 1 observación por mes
sin_tend = y - STL(y, period=12, robust=True).fit().trend
f, P = periodogram(sin_tend, fs=1.0, window="hann", detrend="constant")
picos, _ = find_peaks(P, prominence=P.max() * 0.05)
print("periodos dominantes (meses):", np.round(1 / f[picos], 1))   # se espera ~12 y armónicos

# 3) Correlación cruzada: crecimiento del PIB vs. cambio del desempleo (trimestral)
m = sm.datasets.macrodata.load_pandas().data
dg = np.log(m["realgdp"]).diff().dropna()
du = m["unemp"].diff().dropna()
print(np.round(ccf(dg, du, adjusted=False)[:9], 2))             # rezagos 0 a 8

# 4) Comprobar la convención de signo con una serie desplazada a propósito
rng = np.random.default_rng(0)
a = rng.normal(size=500)
b = np.roll(a, 3) + 0.1 * rng.normal(size=500)                  # b va 3 pasos detrás de a
print(np.argmax(ccf(a, b, adjusted=False)[:10]), np.argmax(ccf(b, a, adjusted=False)[:10]))
# la llamada que imprima 3 indica en qué orden hay que pasar (x, y) para detectar que a se adelanta
```

**Qué observar:** los picos de ACF y PACF en 12, 24 y 36 meses confirman un ciclo anual; el periodograma debe dar el mismo periodo, y la prueba con series sintéticas deja claro qué serie "va primero".

### Actividad del Reto y entregable

- ACF y PACF de la serie estacionarizada del Reto y periodograma; tabla con los periodos y rezagos significativos.
- Si el Reto tiene una variable exógena, correlación cruzada con rezagos y una conclusión sobre si es un indicador líder.
- Superponer `attn_last` (semana 4) y el valor absoluto de la PACF por rezago. Discutir en equipo si coinciden y por qué no deberían interpretarse como lo mismo: la PACF mide dependencia lineal parcial y la atención es un peso aprendido, no una medida de importancia causal.
- Elegir una ventana L justificada con ACF/PACF y reentrenar la LSTM de la semana 4 con ella.

**Dónde se atoran los alumnos:** correlacionar series con tendencia, confundir frecuencia con periodo (f = 1/12 es un ciclo cada 12 meses) y olvidar el parámetro `fs` en el periodograma.

## 7. Semana 7: Atípicos, AR, ARIMA/SARIMA y AIC/BIC

La sesión convierte los hallazgos de las semanas 5 y 6 (diferencias, periodo estacional, rezagos) en un modelo explícito que entrega intervalos de pronóstico y se compara con criterios objetivos.

### Problema detonador

El director del proyecto pide un rango de incertidumbre para el pronóstico, y la LSTM solo entrega un valor por mes. Además, dos picos extraños (captura errónea o evento real) distorsionan el ajuste de cualquier modelo. Pregunta: ¿cómo detectamos esos picos sin borrar señal real, y cómo obtenemos un pronóstico con intervalo y una justificación de por qué este modelo y no otro?

### Objetivos

- Detectar atípicos con un método robusto (residuo de STL y regla de la MAD) y decidir cómo tratarlos.
- Formular AR(p), MA(q), ARIMA(p, d, q) y SARIMA(p, d, q)(P, D, Q)ₛ, y ajustarlos con el ciclo de Box-Jenkins.
- Comparar modelos con AIC y BIC, diagnosticar residuos y evaluar con un corte cronológico.

### Cronograma de la sesión

| Minutos | Bloque | Contenido |
| --- | --- | --- |
| 0–25 | Detonador | Pronóstico sin intervalo y serie con picos. Hipótesis por equipo. |
| 25–75 | Teoría básica | Tipos de atípicos, AR, MA, ARIMA, SARIMA, operador de rezago, Box-Jenkins, AIC y BIC, diagnóstico de residuos. |
| 75–85 | Descanso |  |
| 85–120 | Paquete en vivo | SARIMAX, AutoReg, ar\_select\_order, acorr\_ljungbox, TimeSeriesSplit, auto\_arima (opcional). |
| 120–170 | Ejemplo y actividad | Detección de atípicos y mejor SARIMA para CO₂; luego la serie del Reto. |
| 170–180 | Cierre | Cada equipo defiende su modelo con AIC/BIC y diagnóstico. Pregunta abierta: ¿hay una alternativa más simple y actualizable? |

### Teoría básica

**Atípicos.** Pueden ser *aditivos* (un solo punto fuera de lugar), *cambios de nivel* (el nivel se desplaza para siempre) o *innovacionales* (un golpe que se propaga). Se detectan sobre el residuo de la descomposición, no sobre la serie cruda, para no confundir estacionalidad con anomalía. La regla robusta marca un punto si su puntuación z modificada pasa de 3.5. Antes de borrar un punto hay que investigar su causa: puede ser un evento real.

```latex
z_t = \frac{0.6745\,(r_t - \mathrm{mediana}(r))}{\mathrm{MAD}(r)}, \qquad \text{atípico si } |z_t| > 3.5
```

**AR, MA y ARIMA.** Un AR(p) pronostica con los p valores anteriores, un MA(q) con los q errores anteriores, y el ARIMA añade d diferencias para volver estacionaria la serie. Con el operador de rezago B (B y\_t = y\_(t−1)), el SARIMA suma la parte estacional con periodo s:

```latex
\phi(B)\,(1-B)^d\, y_t = c + \theta(B)\,\varepsilon_t \qquad\Rightarrow\qquad \Phi(B^s)\,\phi(B)\,(1-B)^d(1-B^s)^D\, y_t = c + \Theta(B^s)\,\theta(B)\,\varepsilon_t
```

**Ciclo de Box-Jenkins:** (1) identificar con ACF/PACF y las diferencias necesarias; (2) estimar los coeficientes; (3) diagnosticar (los residuos deben parecer ruido blanco: sin autocorrelación según Ljung-Box y aproximadamente normales); (4) pronosticar con intervalos.

**AIC y BIC** premian el ajuste y penalizan el número de parámetros k; el BIC penaliza más con muestras grandes. Se comparan solo entre modelos ajustados a *los mismos datos con el mismo d y D*, y se eligen los menores.

```latex
\mathrm{AIC}=2k-2\ln\hat{L}, \qquad \mathrm{BIC}=k\ln n-2\ln\hat{L}
```

### Paquetes: comandos y parámetros (statsmodels, scikit-learn)

| Comando | Parámetros clave | Devuelve o hace |
| --- | --- | --- |
| `SARIMAX(endog, exog=None, order=(p,d,q), seasonal_order=(P,D,Q,s), trend=None, enforce_stationarity=True, enforce_invertibility=True)` | `order`, `seasonal_order` (s = 12 en datos mensuales), `exog` para variables externas; acepta valores faltantes (NaN) | Modelo; `.fit(disp=False)` lo ajusta |
| Resultado de `.fit()` | `.summary()`, `.aic`, `.bic`, `.resid`, `.plot_diagnostics()` | Coeficientes, criterios y diagnóstico gráfico |
| `.get_forecast(steps).summary_frame(alpha=0.05)` | `steps` (horizonte), `alpha` (1 − nivel del intervalo) | Tabla con `mean`, `mean_ci_lower`, `mean_ci_upper` |
| `ar_select_order(endog, maxlag, ic='aic', trend='c')` | `maxlag`, `ic`: `'aic'` o `'bic'` | Rezagos elegidos en `.ar_lags` |
| `AutoReg(endog, lags, trend='c', seasonal=False).fit()` | `lags` (entero o lista), `trend` | AR puro |
| `acorr_ljungbox(resid, lags=[12, 24], return_df=True)` | `lags`, `model_df` (parámetros ARMA estimados) | p-valores; p > 0.05 sugiere residuo sin autocorrelación |
| `TimeSeriesSplit(n_splits=5, test_size=None, gap=0)` | `n_splits`, `test_size` (horizonte), `gap` | Pliegues con el pasado siempre antes del futuro |
| `mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error` (sklearn.metrics) | `y_true`, `y_pred` | Métricas de error |
| `pmdarima.auto_arima(y, seasonal=True, m=12, information_criterion='aic', stepwise=True)` | Rangos `max_p`, `max_q`, `max_P`, `max_Q` | Búsqueda automática; es opcional, se instala aparte y puede dar conflictos de versión, así que la rejilla de abajo es la alternativa |

### Ejemplo aplicado: atípicos, AR y mejor SARIMA para CO₂

```python
import itertools, numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.stats.diagnostic import acorr_ljungbox

y = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
y_pico = y.copy()
y_pico.iloc[[100, 250]] += [8, -8]                  # dos atípicos sembrados para la práctica

# 1) Detección: residuo de STL robusto + regla de la MAD
r = STL(y_pico, period=12, robust=True).fit().resid
z = 0.6745 * (r - r.median()) / np.median(np.abs(r - r.median()))
print(y_pico[np.abs(z) > 3.5])                      # deben aparecer los puntos sembrados
y_limpia = y_pico.mask(np.abs(z) > 3.5)             # SARIMAX acepta NaN

# 2) AR puro sobre la serie diferenciada; compararlo con la PACF de la semana 6
x = y.diff().dropna()
print(ar_select_order(x, maxlag=24, ic="aic", trend="c").ar_lags)

# 3) Rejilla pequeña de SARIMA con d = 1 y D = 1 fijos (tarda alrededor de un minuto)
train, test = y.iloc[:-24], y.iloc[-24:]
filas = []
for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2)):
    try:
        m = SARIMAX(train, order=(p, 1, q), seasonal_order=(P, 1, Q, 12)).fit(disp=False)
        filas.append(((p, 1, q), (P, 1, Q, 12), m.aic, m.bic))
    except Exception:
        pass
tabla = pd.DataFrame(filas, columns=["order", "seasonal_order", "AIC", "BIC"]).sort_values("BIC")
print(tabla.head())

# 4) Mejor modelo: diagnóstico, pronóstico con intervalo y error en el conjunto de prueba
mejor = tabla.iloc[0]
mod = SARIMAX(train, order=mejor["order"], seasonal_order=mejor["seasonal_order"]).fit(disp=False)
mod.plot_diagnostics(figsize=(10, 6))
print(acorr_ljungbox(mod.resid.iloc[13:], lags=[12, 24], return_df=True))
fc = mod.get_forecast(steps=24).summary_frame(alpha=0.05)
print("MAE prueba:", np.mean(np.abs(test.values - fc["mean"].values)))
```

**Qué observar:** el BIC suele preferir un modelo más pequeño que el AIC; ambos deben contrastarse con el diagnóstico de residuos, no usarse solos. El intervalo del pronóstico se ensancha con el horizonte, algo que la LSTM no entrega por defecto.

### Actividad del Reto y entregable

- Lista de atípicos detectados en la serie del Reto con su regla, y una decisión justificada (conservar, marcar o imputar).
- Tabla AIC/BIC de al menos 12 combinaciones de SARIMA y el modelo elegido con su diagnóstico (Ljung-Box y gráficas de residuos).
- Pronóstico con intervalo del 95% en un horizonte definido por el equipo y su MAE o RMSE en un conjunto de prueba cronológico.

**Dónde se atoran los alumnos:** comparar AIC entre modelos con distinto d o D, ajustar el modelo con los datos de prueba incluidos, y detener el diagnóstico en el AIC sin revisar residuos.

## 8. Semana 8: Suavizado exponencial y benchmark integrador

La última sesión agrega Holt-Winters y cierra el módulo respondiendo la pregunta abierta de la semana 4: ¿qué modelo conviene poner en producción, con qué evidencia?

### Problema detonador

Gerencia necesita un pronóstico mensual que se actualice con cada dato nuevo sin reajustar un modelo pesado, y quiere saber con evidencia cuál de los candidatos usar: ingenuo estacional, SARIMA, Holt-Winters, LSTM o Transformer. Se muestra una tabla con un solo corte de prueba en la que gana la LSTM y se pregunta: ¿este resultado es confiable o fue suerte del corte?

### Objetivos

- Formular y ajustar suavizado exponencial simple, Holt y Holt-Winters (aditivo y multiplicativo, con tendencia amortiguada).
- Evaluar con métricas comparables (MAE, RMSE, MASE) y validación de origen móvil.
- Redactar una recomendación de modelo que combine precisión, intervalos, costo de actualización e interpretabilidad.

### Cronograma de la sesión

| Minutos | Bloque | Contenido |
| --- | --- | --- |
| 0–20 | Detonador | Tabla de un solo corte de prueba: ¿confiable o suerte? |
| 20–60 | Teoría básica | Suavizado simple, Holt, Holt-Winters, tendencia amortiguada, métricas (MAE, RMSE, MAPE, MASE), validación de origen móvil. |
| 60–70 | Descanso |  |
| 70–100 | Paquete en vivo | ExponentialSmoothing, SimpleExpSmoothing, Holt, ETSModel (opcional); funciones del benchmark. |
| 100–165 | Benchmark del Reto | Cada equipo compara los cinco modelos con 5 orígenes de pronóstico. |
| 165–180 | Cierre del módulo | Mini-presentación de 2 minutos por equipo: modelo recomendado y por qué. |

### Teoría básica

**Suavizado exponencial.** El pronóstico es un promedio donde los datos recientes pesan más; el parámetro α (entre 0 y 1) controla qué tan rápido se olvida el pasado. Holt añade una pendiente (β) y Holt-Winters una componente estacional (γ). La tendencia amortiguada (φ < 1) frena el crecimiento a horizontes largos.

```latex
\begin{aligned}
\ell_t &= \alpha\,(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1})\\
b_t &= \beta\,(\ell_t-\ell_{t-1}) + (1-\beta)\,b_{t-1}\\
s_t &= \gamma\,(y_t-\ell_{t-1}-b_{t-1}) + (1-\gamma)\,s_{t-m}\\
\hat{y}_{t+h|t} &= \ell_t + h\,b_t + s_{t+h-m(k+1)}
\end{aligned}
```

El modelo del ejemplo es el aditivo. En el multiplicativo, la estacionalidad multiplica en lugar de sumar. Se elige igual que en la semana 5: según si la amplitud estacional crece con el nivel.

**Métricas y validación.** El MAE y el RMSE dependen de la escala de la serie. El MASE divide el error entre el MAE de un pronóstico ingenuo estacional dentro de la muestra de entrenamiento: menor que 1 significa que el modelo supera a ese baseline. Un solo corte de prueba puede favorecer a un modelo por azar, así que se repite el pronóstico desde varios orígenes consecutivos (origen móvil) y se promedia.

```latex
\mathrm{MASE}=\frac{\frac{1}{h}\sum_{j=1}^{h}|y_{T+j}-\hat{y}_{T+j}|}{\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t-y_{t-m}|}
```

### Paquetes: comandos y parámetros (statsmodels)

| Comando | Parámetros clave | Devuelve o hace |
| --- | --- | --- |
| `SimpleExpSmoothing(endog, initialization_method=None).fit(smoothing_level=None, optimized=True)` | `smoothing_level` (α fijo, o estimado si `optimized=True`) | Solo nivel |
| `Holt(endog, exponential=False, damped_trend=False, initialization_method=None)` | `exponential` (tendencia multiplicativa), `damped_trend` | Nivel y tendencia |
| `ExponentialSmoothing(endog, trend=None, damped_trend=False, seasonal=None, seasonal_periods=None, initialization_method=None, use_boxcox=False)` | `trend`/`seasonal`: `'add'` o `'mul'`; `seasonal_periods` (12 en mensual); `initialization_method='estimated'` | Holt-Winters completo |
| `.fit(smoothing_level=None, smoothing_trend=None, smoothing_seasonal=None, damping_trend=None, optimized=True)` | Fijar un parámetro lo excluye de la optimización | Resultado con `.params`, `.aic`, `.bic`, `.sse`, `.fittedvalues` |
| `.forecast(steps)` | `steps` | Pronóstico puntual (Serie) |
| `ETSModel(endog, error='add', trend='add', damped_trend=False, seasonal='add', seasonal_periods=12).fit()` | `error`: `'add'` o `'mul'` | Versión estadística de Holt-Winters con intervalos mediante `.get_prediction(start, end).summary_frame()` (opcional) |
| `TimeSeriesSplit(n_splits, test_size)` | Origen móvil automático | Pliegues cronológicos |

### Ejemplo aplicado: Holt-Winters y benchmark de cinco modelos

```python
import numpy as np, pandas as pd, statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Reutiliza make_windows, SeqModel, TSTransformer y fit de la semana 4

y = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()

# 1) Holt-Winters con tendencia amortiguada y estacionalidad aditiva
hw = ExponentialSmoothing(y.iloc[:-24], trend="add", damped_trend=True, seasonal="add",
                          seasonal_periods=12, initialization_method="estimated").fit()
print(hw.params, hw.aic)             # α, β, γ, φ estimados
print(hw.forecast(24).head())

# 2) Benchmark con 5 orígenes de pronóstico y horizonte de 12 meses
h = 12
origenes = range(len(y) - 5 * h, len(y) - h + 1, h)

def f_naive_estacional(tr, h, m=12):
    return np.tile(tr.values[-m:], int(np.ceil(h / m)))[:h]

def f_hw(tr, h):
    aj = ExponentialSmoothing(tr, trend="add", damped_trend=True, seasonal="add",
                              seasonal_periods=12, initialization_method="estimated").fit()
    return aj.forecast(h).values

def f_sarima(tr, h):                  # modelo "airline" como referencia
    return SARIMAX(tr, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).forecast(h).values

def f_red(tr, h, make_model, L=24, epochs=60):        # pronóstico recursivo de una red
    sc = StandardScaler().fit(tr.to_frame())
    z = sc.transform(tr.to_frame()).ravel().astype("float32")
    X, t = make_windows(z, L)
    modelo = fit(make_model(), DataLoader(TensorDataset(X, t), batch_size=32, shuffle=True), epochs=epochs)
    modelo.eval(); ventana, salida = list(z[-L:]), []
    with torch.no_grad():
        for _ in range(h):
            nxt = modelo(torch.tensor(ventana[-L:]).view(1, L, 1)).item()
            salida.append(nxt); ventana.append(nxt)
    return sc.inverse_transform(np.array(salida).reshape(-1, 1)).ravel()

def mase(real, pred, tr, m=12):
    return np.mean(np.abs(real - pred)) / np.mean(np.abs(tr.values[m:] - tr.values[:-m]))

modelos = {"naive estacional": f_naive_estacional, "Holt-Winters": f_hw, "SARIMA airline": f_sarima,
           "LSTM": lambda tr, h: f_red(tr, h, lambda: SeqModel("lstm")),
           "Transformer": lambda tr, h: f_red(tr, h, TSTransformer)}
filas = []
for nombre, f in modelos.items():
    for o in origenes:
        tr, real = y.iloc[:o], y.iloc[o:o + h].values
        pred = f(tr, h)
        filas.append((nombre, np.mean(np.abs(real - pred)), np.sqrt(np.mean((real - pred) ** 2)), mase(real, pred, tr)))
res = pd.DataFrame(filas, columns=["modelo", "MAE", "RMSE", "MASE"])
print(res.groupby("modelo").mean().sort_values("MASE"))
```

**Qué observar:** en series con tendencia y estacionalidad estables como el CO₂, los modelos estadísticos suelen ganar o empatar con las redes y cuestan mucho menos de entrenar y actualizar. Ese resultado es parte del aprendizaje: la red no es automáticamente mejor, y el benchmark es lo que lo demuestra. Si las redes se entrenan sobre la serie diferenciada (semana 5), el resultado cambia, y vale la pena que los equipos lo prueben.

### Actividad del Reto y entregable final

- Benchmark de los cinco modelos sobre la serie del Reto con al menos 5 orígenes de pronóstico; tabla de MAE, RMSE y MASE promedio.
- Informe corto con la recomendación de modelo y su justificación: precisión, intervalos de pronóstico, costo de entrenamiento y actualización, interpretabilidad. Debe responder la pregunta de la semana 4: ¿el deep learning le ganó a los modelos estadísticos?
- Mini-presentación de 2 minutos en la sesión.

**Extensión opcional con TDA (para equipos que terminen antes y solo si el Profesor A cubrió Takens y homología persistente):** convertir ventanas de la serie en nubes de puntos con encaje de Takens, calcular su homología persistente y usar un resumen (por ejemplo, la entropía de persistencia) como variable adicional de un modelo. Esquema con giotto-tda; los parámetros `time_delay` y `dimension` funcionan como cotas máximas cuando `parameters_type="search"`:

```python
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy

emb = SingleTakensEmbedding(parameters_type="search", time_delay=12, dimension=6, stride=1)
nube = emb.fit_transform(y.diff().dropna().values)             # (puntos, dimensión)
dgm = VietorisRipsPersistence(homology_dimensions=[0, 1]).fit_transform(nube[None, :, :])
entropia = PersistenceEntropy().fit_transform(dgm)
```

**Dónde se atoran los alumnos:** comparar modelos con cortes distintos, omitir `initialization_method` o `seasonal_periods`, y dar por ganadora a la red por un solo origen favorable.

## 9. Evaluación y entregables

Se mantiene la evaluación formativa del documento original: cada actividad es una bitácora corta que ancla la teoría al código del Reto, y la semana 8 funciona como entregable integrador. Los pesos de calificación los defines tú.

| Semana | Entregable | Evidencia mínima |
| --- | --- | --- |
| 4 | Notebook de redes secuenciales | Tabla de RMSE y MAE (ingenuo, RNN, GRU, LSTM, Transformer), curvas de pérdida y gráfica de atención |
| 5 | Diagnóstico de estacionariedad y descomposición | Reporte ADF y KPSS antes y después, descomposición elegida y justificada, RMSE de la red en escala original |
| 6 | Rezagos, ciclos y atención vs. PACF | ACF/PACF, periodograma con periodos, ventana L justificada, comparación con `attn_last` |
| 7 | Modelo SARIMA con diagnóstico | Lista de atípicos, tabla AIC/BIC, Ljung-Box, pronóstico con intervalo y error en prueba |
| 8 | Informe de benchmark y mini-presentación | MAE, RMSE y MASE promedio de cinco modelos con 5 orígenes, y recomendación razonada |

**Criterios transversales de revisión:**

- **Validación sin fuga:** corte cronológico, escalador ajustado solo con entrenamiento, ninguna mezcla de datos de prueba.
- **Baseline presente:** todo resultado se compara con el modelo ingenuo (o ingenuo estacional).
- **Interpretación:** cada número va acompañado de una frase que dice qué significa para el Reto.
- **Reproducibilidad:** semilla fija, notebook que corre de principio a fin y versiones de paquetes anotadas.

## 10. Referencias, paquetes y recomendaciones

Las referencias se citan por título (sin enlaces, porque no las abrí aquí); verifica ediciones y acceso antes de compartirlas con el grupo.

**Libros y artículos base:**

- *Deep Learning* (Goodfellow, Bengio, Courville): capítulos de redes recurrentes y atención.
- *Time Series Analysis: Forecasting and Control* (Box, Jenkins, Reinsel, Ljung): ARIMA y Box-Jenkins.
- *Forecasting: Principles and Practice* (Hyndman y Athanasopoulos): descomposición, suavizado exponencial y métricas como MASE.
- *Time Series Analysis and Its Applications* (Shumway y Stoffer): análisis espectral.
- Hochreiter y Schmidhuber (1997) sobre LSTM; Cho et al. (2014) sobre GRU; Vaswani et al. (2017), "Attention Is All You Need".
- Documentación oficial de PyTorch, statsmodels, scipy y scikit-learn.

**Instalación sugerida (Python en Colab o entorno local):**

```bash
pip install torch statsmodels scikit-learn scipy pandas numpy matplotlib
pip install pmdarima giotto-tda   # opcionales
```

**Recomendaciones y puntos por decidir:**

- **Semana 4:** es la más cargada. Conviene un notebook previo (MLP y `DataLoader`) y tener el código de las redes casi listo, para que el tiempo de clase se use en parámetros y diagnóstico. Si el grupo va lento, la versión `TinyAttn` puede pasar a la semana 6.
- **Código sin ejecutar:** los fragmentos de este documento no se corrieron en el entorno donde lo preparé (no tenía PyTorch ni statsmodels). Corre cada notebook completo una vez antes de clase, sobre todo las salidas esperadas de ADF/KPSS, el periodograma y el signo de `ccf`.
- **Serie del Reto:** los ejemplos usan CO₂ de Mauna Loa para que corran sin descargas. Si la serie del Reto es multivariada o tiene otra frecuencia, hay que ajustar `seasonal_periods`, `period` y `s`.
- **Aprendizaje no supervisado:** PCA, clustering y Autoencoders del documento original no están en tu bloque. Conviene confirmar con el Profesor A que los cubre él, y si no, hay que decidir en qué semana entran.
- **Cómputo:** con series de cientos o pocos miles de puntos, todo corre en CPU en segundos o minutos; no hace falta GPU.
