# Semana 04: Redes Secuenciales y Transformers
### RNN, GRU, LSTM y Mecanismo de Auto-Atención
**MA3001B: Modelos y TDA • Profesor B**  
Tecnológico de Monterrey • 8vo Semestre

Note:
[00:00 - 00:05]
Bienvenida a la Semana 4 del módulo Modelos y TDA.
Esta sesión es de 180 minutos (3 horas) y cubre el salto de redes neuronales recurrentes clásicas hasta arquitecturas Transformer en series temporales.
Material editable en Markdown para Reveal.js.

---

## Agenda de la Sesión (180 min)

| Minutos | Bloque | Actividad y Enfoque |
| :--- | :--- | :--- |
| **00:00–00:20** | **Detonador** | Diagnóstico en equipo: MLP y RNN simple estancados |
| **00:20–00:55** | **Teoría 1** | RNN, gradientes desvanecientes, compuertas LSTM y GRU |
| **00:55–01:25** | **Paquete 1** | `nn.RNN`, `nn.LSTM`, `nn.GRU`, tensores y grad norm |
| **01:25–01:35** | **Descanso** | Pausa intermedia |
| **01:35–02:00** | **Teoría 2** | Auto-atención, $Q, K, V$, posicional y máscara causal |
| **02:00–02:30** | **Paquete 2** | `TransformerEncoder`, `MultiheadAttention` |
| **02:30–02:50** | **Reto** | Benchmark: redes vs. baseline ingenuo |
| **02:50–03:00** | **Cierre** | Discusión guiada y pregunta abierta para Semana 5 |

Note:
[00:05 - 00:10]
Explicar a los alumnos que es la sesión más densa del módulo.
Dividida en dos bloques de deep learning: recurrencia (RNN/LSTM/GRU) y atención (Transformers).

---

## Problema Detonador (00:00–00:20)

<div class="alert-box">
  <strong>El misterio de la pérdida estancada:</strong> Entrenamos un MLP y una RNN simple sobre ventanas largas ($L=100$) de una serie temporal continua. Ambos modelos se quedan atrapados en un MSE alto y sus predicciones convergen a una línea recta casi horizontal.
</div>

--

### Código del Problema Detonador

```python [1-6|8-13]
# Ventana larga L = 100
X, y = make_windows(serie_co2, L=100)

model_rnn = nn.RNN(input_size=1, hidden_size=32, batch_first=True)
head = nn.Linear(32, 1)
# Entrenamos por 50 épocas con Adam lr=1e-3...

# Resultado en evaluación:
preds = head(model_rnn(X_test)[0][:, -1, :])
# Predicciones: [365.1, 365.2, 365.1, 365.2, ...]
# Serie real: oscila entre 355 y 375 con tendencia alcista
# RMSE del baseline ingenuo (y[t-1]): 0.42
# RMSE de la RNN entrenada: 4.85
```

--

### Preguntas para los Equipos (15 min)

<div class="highlight-box">
  <ol>
    <li>¿Por qué la RNN simple no logra propagar la memoria a través de 100 pasos de tiempo?</li>
    <li>¿Qué ocurre matemáticamente con la derivada de la pérdida respecto a $W_h$ al aplicar la regla de la cadena a través de 100 pasos?</li>
    <li>¿Por qué predecir el valor del paso anterior ($y_{t-1}$) resulta en un error drásticamente menor que el modelo neuronal complejo?</li>
  </ol>
</div>

Note:
[00:10 - 00:20]
Dar 10 minutos a los equipos para formular hipótesis.
Pedir a dos equipos que compartan sus respuestas antes de entrar a la teoría.

---

## Objetivos de Aprendizaje

- Explicar formalmente el fenómeno del desvanecimiento y explosión del gradiente en BPTT. <!-- .element: class="fragment" -->
- Comparar los mecanismos de compuertas de **LSTM** (olvido, entrada, salida) y **GRU** (actualización, reinicio). <!-- .element: class="fragment" -->
- Comprender la auto-atención como un operador sin recurrencia que proyecta en $Q$, $K$, $V$. <!-- .element: class="fragment" -->
- Implementar modelos en PyTorch respetando las dimensiones de tensores `(batch, L, features)`. <!-- .element: class="fragment" -->
- Evaluar rigurosamente frente a un **baseline ingenuo** usando corte cronológico sin fuga de información. <!-- .element: class="fragment" -->

Note:
[00:20 - 00:25]
Resaltar que en series temporales un modelo sin baseline no tiene validez científica.

---

## Teoría 1: Redes Recurrentes (RNN)

Una RNN procesa una secuencia vector a vector manteniendo un estado oculto $h_t \in \mathbb{R}^d$ que resume el pasado observado.

$$
h_t = \tanh\left(W_x x_t + W_h h_{t-1} + b\right)
$$

$$
\hat{y}_t = W_y h_t + c
$$

--

### Esquema de Recurrencia Desplegada

<svg viewBox="0 0 700 180" style="width:100%; max-height:220px; margin: 10px auto; display:block;">
  <rect x="30" y="60" width="80" height="50" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="70" y="90" text-anchor="middle" font-family="Inter" font-size="14" fill="#073642">h_{t-1}</text>
  <path d="M 110 85 L 170 85" stroke="#CB4B16" stroke-width="2" marker-end="url(#arrow)"/>

  <rect x="180" y="60" width="80" height="50" rx="8" fill="#FDF6E3" stroke="#CB4B16" stroke-width="3"/>
  <text x="220" y="90" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="15" fill="#CB4B16">RNN (h_t)</text>
  <path d="M 220 150 L 220 115" stroke="#2AA198" stroke-width="2"/>
  <text x="220" y="170" text-anchor="middle" font-family="Fira Code" font-size="13" fill="#2AA198">x_t</text>
  <path d="M 220 60 L 220 25" stroke="#268BD2" stroke-width="2"/>
  <text x="220" y="18" text-anchor="middle" font-family="Fira Code" font-size="13" fill="#268BD2">\hat{y}_t</text>
  <path d="M 260 85 L 320 85" stroke="#CB4B16" stroke-width="2"/>

  <rect x="330" y="60" width="80" height="50" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="370" y="90" text-anchor="middle" font-family="Inter" font-size="14" fill="#073642">h_{t+1}</text>
  <path d="M 410 85 L 470 85" stroke="#93A1A1" stroke-width="2" stroke-dasharray="4"/>

  <rect x="480" y="60" width="80" height="50" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="520" y="90" text-anchor="middle" font-family="Inter" font-size="14" fill="#073642">h_T</text>
</svg>

- **Many-to-One:** Se utiliza únicamente el último vector de estado oculto $h_T$ para proyectar el pronóstico escalar $\hat{y}_{T+1}$.
- **Many-to-Many:** Se proyecta cada estado intermedio $h_t$ para generar una secuencia completa.

--

### El Desvanecimiento del Gradiente en BPTT

Al calcular el gradiente de la pérdida final $\mathcal{L}_T$ respecto al estado $h_k$ ($k \ll T$), aplicamos la regla de la cadena:

$$
\frac{\partial \mathcal{L}_T}{\partial h_k} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{j=k+1}^{T} \frac{\partial h_j}{\partial h_{j-1}}
$$

Donde la matriz jacobiana contiene:

$$
\frac{\partial h_j}{\partial h_{j-1}} = \operatorname{diag}\left(1 - \tanh^2(\dots)\right) W_h^\top
$$

- Si los valores singulares de $W_h$ son $< 1$, el producto de $(T - k)$ matrices decae exponencialmente a **cero**. <!-- .element: class="fragment" -->
- Si son $> 1$, el producto diverge exponencialmente: **explosión del gradiente** (mitigado con `clip_grad_norm_`). <!-- .element: class="fragment" -->

Note:
[00:35 - 00:45]
Destacar que con $T=100$, un factor de $0.9^{100} \approx 2.6 \times 10^{-5}$. La señal del inicio desaparece por completo.

---

## Solución: Arquitectura LSTM

Long Short-Term Memory introduce una "carretera lineal" de información: el **estado de celda** $c_t$, regulado por compuertas sigmoides $\sigma \in (0, 1)$.

$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Compuerta de Olvido)}
$$

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i), \quad \tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(Actualización de Celda)}
$$

$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t)
$$

--

### Anatomía de la Celda LSTM

<svg viewBox="0 0 620 220" style="width:100%; max-height:240px; margin: 10px auto; display:block;">
  <rect x="40" y="20" width="540" height="180" rx="12" fill="#FDF6E3" stroke="#2AA198" stroke-width="2"/>
  <text x="60" y="45" font-family="Crimson Text" font-weight="bold" font-size="16" fill="#073642">Celda LSTM</text>

  <!-- Canal c_t -->
  <line x1="20" y1="70" x2="600" y2="70" stroke="#B58900" stroke-width="4"/>
  <text x="25" y="60" font-family="Fira Code" font-size="12" fill="#B58900">c_{t-1}</text>
  <text x="580" y="60" font-family="Fira Code" font-size="12" fill="#B58900">c_t</text>

  <!-- Operaciones sobre c_t -->
  <circle cx="150" cy="70" r="14" fill="#EEE8D5" stroke="#CB4B16" stroke-width="2"/>
  <text x="150" y="75" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#CB4B16">×</text>

  <circle cx="310" cy="70" r="14" fill="#EEE8D5" stroke="#2AA198" stroke-width="2"/>
  <text x="310" y="75" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#2AA198">+</text>

  <!-- Compuertas -->
  <rect x="130" y="125" width="40" height="28" rx="4" fill="#EEE8D5" stroke="#CB4B16" stroke-width="1.5"/>
  <text x="150" y="144" text-anchor="middle" font-family="Fira Code" font-size="13" fill="#CB4B16">f_t</text>
  <line x1="150" y1="125" x2="150" y2="84" stroke="#CB4B16" stroke-width="2"/>

  <rect x="240" y="125" width="40" height="28" rx="4" fill="#EEE8D5" stroke="#2AA198" stroke-width="1.5"/>
  <text x="260" y="144" text-anchor="middle" font-family="Fira Code" font-size="13" fill="#2AA198">i_t</text>

  <rect x="310" y="125" width="40" height="28" rx="4" fill="#EEE8D5" stroke="#2AA198" stroke-width="1.5"/>
  <text x="330" y="144" text-anchor="middle" font-family="Fira Code" font-size="12" fill="#2AA198">\tilde{c}</text>

  <rect x="420" y="125" width="40" height="28" rx="4" fill="#EEE8D5" stroke="#6C71C4" stroke-width="1.5"/>
  <text x="440" y="144" text-anchor="middle" font-family="Fira Code" font-size="13" fill="#6C71C4">o_t</text>

  <!-- Salida h_t -->
  <line x1="20" y1="175" x2="600" y2="175" stroke="#268BD2" stroke-width="3"/>
  <text x="25" y="165" font-family="Fira Code" font-size="12" fill="#268BD2">h_{t-1}</text>
  <text x="580" y="165" font-family="Fira Code" font-size="12" fill="#268BD2">h_t</text>
</svg>

- Si $f_t = 1$, el gradiente fluye intacto hacia el pasado ($\frac{\partial c_t}{\partial c_{t-1}} = 1$): adición en lugar de multiplicación pura.

---

## Arquitectura GRU (Gated Recurrent Unit)

Simplifica la LSTM eliminando el estado de celda separado y fusionando compuertas en solo dos:

$$
z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(Compuerta de Actualización)}
$$

$$
r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(Compuerta de Reinicio)}
$$

$$
\tilde{h}_t = \tanh\left(W [r_t \odot h_{t-1}, x_t]\right)
$$

$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

--

### Comparativa: RNN vs. GRU vs. LSTM

| Característica | RNN Simple | GRU | LSTM |
| :--- | :---: | :---: | :---: |
| **Vectores de Estado** | $h_t$ | $h_t$ | $h_t$ y $c_t$ |
| **Compuertas** | 0 | 2 ($z, r$) | 3 ($f, i, o$) |
| **Parámetros Relativos** | **1×** | **3×** | **4×** |
| **Flujo del Gradiente** | Multiplicativo | Aditivo parcial | Aditivo directo en celda |
| **Caso de Uso Típico** | Ventanas muy cortas | Datos moderados | Series largas y ricas |

Note:
[00:50 - 00:55]
Verificar con los alumnos la fórmula de parámetros:
Para input=1 y hidden=32:
RNN: 32*(1+32) + 32 = 1,088 + 32 = 1,120
GRU: 3 * 1,120 = 3,360
LSTM: 4 * 1,120 = 4,480

---

## Paquetes en PyTorch: Recurrencia

```python [1-4|6-10|12-14]
# Dimensiones esperadas con batch_first=True:
# Entrada x: (batch_size, seq_len, input_size)
# Salida out: (batch_size, seq_len, hidden_size)
# Estado h_n: (num_layers, batch_size, hidden_size)

cell = nn.LSTM(
    input_size=1,       # 1 variable por paso
    hidden_size=32,     # Dimensión del estado
    num_layers=1,       # Capas apiladas
    batch_first=True    # Obligatorio para (N, L, C)
)

out, (h_n, c_n) = cell(x)
pred = head(out[:, -1, :])   # Tomamos el último paso temporal
```

--

### Errores Comunes de Implementación

<div class="alert-box">
  <ul>
    <li><strong>Olvidar <code>batch_first=True</code>:</strong> Por defecto en PyTorch es <code>False</code>, esperando <code>(seq_len, batch, input_size)</code>. Si se pasa <code>(batch, L, 1)</code>, ¡el modelo confunde el batch con el tiempo!</li>
    <li><strong>Ajustar el escalador con toda la serie:</strong> <code>StandardScaler().fit(serie_completa)</code> introduce fuga del futuro (data leakage). Se debe ajustar <em>exclusivamente</em> con el conjunto de entrenamiento.</li>
    <li><strong>No aplicar recorte de gradiente:</strong> Las RNN pueden explotar en gradientes sin previo aviso; utilizar siempre <code>nn.utils.clip_grad_norm_(model.parameters(), 1.0)</code>.</li>
  </ul>
</div>

Note:
[01:10 - 01:25]
Mostrar en vivo qué sucede con el MSE si batch_first no coincide.

---

## Teoría 2: Mecanismo de Auto-Atención

Los **Transformers** eliminan la recurrencia paso a paso y procesan todos los $L$ pasos en paralelo. Cada paso genera tres vectores mediante proyecciones lineales:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

--

### Interpretación de la Matriz de Pesos $A$

$$
A = \operatorname{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{L \times L}
$$

- La fila $i$ indica cuánto atiende el paso $i$ a cada uno de los pasos $j \in \{1, \dots, L\}$. <!-- .element: class="fragment" -->
- Al inspeccionar la última fila $A[L-1, :]$, observamos directamente qué rezagos del pasado influyen en el pronóstico actual. <!-- .element: class="fragment" -->
- El factor de escala $\frac{1}{\sqrt{d_k}}$ evita que los productos punto crezcan demasiado empujando la función softmax a zonas de gradiente casi nulo. <!-- .element: class="fragment" -->

--

### Codificación Posicional y Máscara Causal

Dado que el producto punto matricial es invariante ante permutaciones de filas, el Transformer requiere:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

- **Codificación Posicional:** Inyecta la noción de orden temporal sumando $X + PE$. <!-- .element: class="fragment" -->
- **Máscara Causal:** En decodificadores o tareas autorregresivas, anula posiciones futuras con $-\infty$ para asegurar que el paso $t$ no atienda a $t+k$. <!-- .element: class="fragment" -->

Note:
[01:45 - 02:00]
En nuestro caso para pronóstico many-to-one sobre una ventana fija, la ventana completa ya pertenece al pasado, por lo que el encoder estándar es causal respecto al horizonte futuro.

---

## Transformer en PyTorch

```python [1-6|8-12]
class TSTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.pos = PosEnc(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=64, dropout=0.1,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.enc(self.pos(self.inp(x)))
        return self.head(h[:, -1, :])   # Pronóstico con el último token
```

--

### Inspección de Pesos con TinyAttn

```python [1-8|10-13]
class TinyAttn(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.inp, self.pos = nn.Linear(1, d_model), PosEnc(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm, self.head = nn.LayerNorm(d_model), nn.Linear(d_model, 1)
    def forward(self, x):
        h = self.pos(self.inp(x))
        a, w = self.attn(h, h, h, need_weights=True)
        self.w = w.detach()  # Pesos de atención guardados (N, L, L)
        return self.head(self.norm(h + a)[:, -1, :])

# Extraer el perfil medio de atención del último paso:
attn_profile = model.w[:, -1, :].mean(dim=0).cpu().numpy()
```

Note:
[02:15 - 02:30]
Este vector `attn_profile` es fundamental: se guarda para contrastarlo con la función de autocorrelación parcial (PACF) en la Semana 6.

---

## Actividad del Reto (02:30–02:50)

<div class="highlight-box">
  <strong>Misión de laboratorio:</strong> Abrir el notebook <code>notebooks/N1_RedesSecuenciales_Transformers.ipynb</code> y ejecutarlo con la serie de práctica (CO₂) o la del Reto.
</div>

- Entrenar los 4 modelos: **RNN Simple**, **GRU**, **LSTM** y **Transformer**. <!-- .element: class="fragment" -->
- Construir la tabla comparativa de métricas en prueba frente al **Baseline Ingenuo** ($y_t = y_{t-1}$). <!-- .element: class="fragment" -->
- Graficar las pérdidas de entrenamiento y la curva de atención `attn_profile`. <!-- .element: class="fragment" -->

--

### Rúbrica Rápida del Entregable (Semana 4)

| Criterio | Excelente (100%) | Aceptable (70%) | Insuficiente (<50%) |
| :--- | :--- | :--- | :--- |
| **Validación** | Corte cronológico estricto sin fuga de escalador | Corte cronológico pero escalador ajustado globalmente | Split aleatorio `train_test_split` |
| **Baseline** | Incluye baseline ingenuo y calcula RMSE/MAE | Compara modelos sin baseline | No reporta métricas cuantitativas |
| **Atención** | Grafica e interpreta pesos respecto a rezagos | Solo genera la gráfica sin explicación | No implementa extracción de pesos |

---

## Cierre y Pregunta Abierta (02:50–03:00)

<div class="def-box">
  <strong>Hallazgo central de hoy:</strong> En series no estacionarias con tendencia, los modelos neuronales sufren para extrapolar valores fuera del rango observado, y el baseline ingenuo suele ser muy competitivo o incluso superior.
</div>

- **Pregunta para la Semana 05:** ¿Por qué nuestras redes neuronales se aplanan al intentar predecir el futuro lejano en una serie creciente, y cómo podemos transformar matemáticamente la serie para eliminar la tendencia antes de entrenar?

Note:
[02:55 - 03:00]
Recoger bitácoras breves por equipo y cerrar la sesión puntual a los 180 min.
Enlace directo al notebook: notebooks/html/N1_RedesSecuenciales_Transformers.html
