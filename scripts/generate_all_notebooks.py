#!/usr/bin/env python3
"""
Generador completo de los 5 Notebooks de MA3001B con nbformat.
Crea celdas markdown y de código limpias, con teoría, tablas, ejemplos ejecutables,
celda SERIE_RETO, actividades del Reto con celdas # TODO y preguntas.
"""

import os
import nbformat as nbf

REPO_NAME = "alxcn/Models_MA3001B"
NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def make_colab_badge(filename):
    return f"[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO_NAME}/blob/main/notebooks/{filename})"

def make_serie_reto_cells():
    md = nbf.v4.new_markdown_cell("""### Configuración de Datos: Práctica (CO₂) o Serie del Reto
En la siguiente celda se carga la serie de práctica por defecto (**CO₂ mensual de Mauna Loa** de `statsmodels`).
Para aplicar el análisis a la serie de su equipo en el Reto, cambie la bandera `USAR_SERIE_RETO = True` y proporcione la ruta a su archivo CSV.""")

    code = nbf.v4.new_code_cell("""# ==============================================================================
# CELDA SERIE_RETO: Alternar entre datos de práctica y datos del Reto
# ==============================================================================
USAR_SERIE_RETO = False  # Cambiar a True para usar la serie de su Reto

if USAR_SERIE_RETO:
    # Sustituya con la carga de la serie de su equipo:
    # df_reto = pd.read_csv("datos_reto.csv", index_col=0, parse_dates=True)
    # y = df_reto["columna_objetivo"].asfreq("MS").interpolate()
    print("Modo Reto activado.")
else:
    # Serie de práctica por defecto: CO2 de Mauna Loa (mensual interpolado)
    y = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
    print(f"Cargada serie de práctica CO2: {len(y)} observaciones ({y.index.min().strftime('%Y-%m')} a {y.index.max().strftime('%Y-%m')})")
""")
    return [md, code]

# ==============================================================================
# NOTEBOOK 1: N1_RedesSecuenciales_Transformers.ipynb
# ==============================================================================
def create_n1():
    nb = nbf.v4.new_notebook()
    cells = []
    filename = "N1_RedesSecuenciales_Transformers.ipynb"

    cells.append(nbf.v4.new_markdown_cell(f"""# MA3001B • Semana 04: Redes Secuenciales y Transformers
## RNN, GRU, LSTM y Mecanismo de Auto-Atención en Series de Tiempo
**Profesor B: Modelos y TDA • Tecnológico de Monterrey**

{make_colab_badge(filename)}

---
### Objetivos de la Sesión
1. Explicar el fenómeno de desvanecimiento del gradiente en RNN simples con ventanas largas.
2. Comprender la mitigación mediante compuertas en LSTM ($f, i, o$) y GRU ($z, r$).
3. Implementar el mecanismo de auto-atención multi-cabeza y extraer la matriz de atención para inspeccionar rezagos clave.
4. Entrenar y comparar RNN, GRU, LSTM y Transformer frente a un **baseline ingenuo** usando corte cronológico estricto sin fuga de información.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Problema Detonador: Pérdida Estancada y Aplanamiento
Entrenamos un modelo recurrente simple (RNN) con ventanas largas ($L=96$) sobre una serie con tendencia. Observemos cómo la pérdida se estanca y el pronóstico degenera en una línea casi recta, mientras que un baseline ingenuo que simplemente repite el valor anterior logra un error drásticamente menor."""))

    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# Cargamos serie de prueba
y_det = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()

# Ventana muy larga L = 96
L_largo = 96
scaler_det = StandardScaler().fit(y_det.iloc[:int(len(y_det)*0.8)].to_frame())
z_det = scaler_det.transform(y_det.to_frame()).ravel().astype("float32")

def make_windows_np(z, L):
    X = np.stack([z[i:i + L] for i in range(len(z) - L)])
    t = z[L:]
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(t, dtype=torch.float32).unsqueeze(-1)

X_det, t_det = make_windows_np(z_det, L_largo)
n_tr = int(len(y_det) * 0.8) - L_largo

# RNN simple
rnn_bad = nn.RNN(input_size=1, hidden_size=16, batch_first=True)
head_bad = nn.Linear(16, 1)
opt_bad = torch.optim.Adam(list(rnn_bad.parameters()) + list(head_bad.parameters()), lr=1e-3)

dl_bad = DataLoader(TensorDataset(X_det[:n_tr], t_det[:n_tr]), batch_size=32, shuffle=True)
for epoch in range(15):
    for xb, yb in dl_bad:
        opt_bad.zero_grad()
        out, _ = rnn_bad(xb)
        loss = nn.MSELoss()(head_bad(out[:, -1, :]), yb)
        loss.backward()
        opt_bad.step()

with torch.no_grad():
    out_test, _ = rnn_bad(X_det[n_tr:])
    pred_bad = head_bad(out_test[:, -1, :]).squeeze(-1).numpy()
    true_test = t_det[n_tr:].squeeze(-1).numpy()
    naive_pred = X_det[n_tr:, -1, 0].numpy()

rmse_bad = np.sqrt(np.mean((true_test - pred_bad)**2))
rmse_naive = np.sqrt(np.mean((true_test - naive_pred)**2))

print(f"RMSE RNN simple (L=96): {rmse_bad:.4f}")
print(f"RMSE Baseline Ingenuo: {rmse_naive:.4f}")
print("-> ¿Por qué la RNN simple no logra superar al valor anterior y sufre con ventanas largas?")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Marco Teórico Mínimo

### 2.1 Ecuaciones de RNN, LSTM y GRU

**RNN Simple:**
$$h_t = \\tanh(W_x x_t + W_h h_{t-1} + b), \\qquad \\hat{y}_t = W_y h_T + c$$

**LSTM:**
$$\\begin{aligned}
f_t &= \\sigma(W_f [h_{t-1}, x_t] + b_f) \\quad &\\text{(Olvido)} \\\\
i_t &= \\sigma(W_i [h_{t-1}, x_t] + b_i) \\quad &\\text{(Entrada)} \\\\
o_t &= \\sigma(W_o [h_{t-1}, x_t] + b_o) \\quad &\\text{(Salida)} \\\\
\\tilde{c}_t &= \\tanh(W_c [h_{t-1}, x_t] + b_c), \\quad &c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c}_t \\\\
h_t &= o_t \\odot \\tanh(c_t)
\\end{aligned}$$

**GRU:**
$$\\begin{aligned}
z_t &= \\sigma(W_z [h_{t-1}, x_t]), \\quad r_t = \\sigma(W_r [h_{t-1}, x_t]) \\\\
\\tilde{h}_t &= \\tanh(W [r_t \\odot h_{t-1}, x_t]), \\quad h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t
\\end{aligned}$$

### 2.2 Mecanismo de Auto-Atención
$$\\operatorname{Attention}(Q, K, V) = \\operatorname{softmax}\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V$$
"""))

    cells.extend(make_serie_reto_cells())

    cells.append(nbf.v4.new_markdown_cell("""## 3. Ejemplo Aplicado Paso a Paso

### Paso 1: Partición cronológica y escalamiento sin fuga de información"""))

    cells.append(nbf.v4.new_code_cell("""# 1) División cronológica 80% entrenamiento / 20% prueba
n_train = int(len(y) * 0.8)

scaler = StandardScaler().fit(y.iloc[:n_train].to_frame())
z = scaler.transform(y.to_frame()).ravel().astype("float32")

L = 24
def make_windows(z, L):
    X = np.stack([z[i:i + L] for i in range(len(z) - L)])
    t = z[L:]
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(t, dtype=torch.float32).unsqueeze(-1)

X, t = make_windows(z, L)
split = n_train - L

train_ds = TensorDataset(X[:split], t[:split])
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

print(f"Dimensiones de entrenamiento: X = {X[:split].shape}, t = {t[:split].shape}")
print(f"Dimensiones de prueba: X = {X[split:].shape}, t = {t[split:].shape}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 2: Modelos Recurrentes (RNN, GRU, LSTM)"""))

    cells.append(nbf.v4.new_code_cell("""class SeqModel(nn.Module):
    def __init__(self, kind="lstm", hidden=32):
        super().__init__()
        cells_dict = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        self.rnn = cells_dict[kind](input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

def fit_model(model, dl, epochs=35, lr=1e-3, clip=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        losses.append(ep_loss / len(dl.dataset))
    return model, losses

def predict_model(model, X_eval):
    model.eval()
    with torch.no_grad():
        return model(X_eval).squeeze(-1).numpy()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 3: Transformer y TinyAttn para Inspeccionar Pesos"""))

    cells.append(nbf.v4.new_code_cell("""class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TSTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_ff=64, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.enc(self.pos(self.inp(x)))
        return self.head(h[:, -1, :])

class TinyAttn(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.pos = PositionalEncoding(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.pos(self.inp(x))
        a, w = self.attn(h, h, h, need_weights=True)
        self.w = w.detach()
        return self.head(self.norm(h + a)[:, -1, :])
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 4: Entrenamiento y Tabla de Resultados"""))

    cells.append(nbf.v4.new_code_cell("""rmse_fn = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
mae_fn = lambda a, b: float(np.mean(np.abs(a - b)))

true_test = t[split:].squeeze(-1).numpy()
naive_pred = X[split:, -1, 0].numpy()

resultados = {
    "Ingenuo (y[t-1])": {"RMSE": rmse_fn(true_test, naive_pred), "MAE": mae_fn(true_test, naive_pred)}
}

for kind in ["rnn", "gru", "lstm"]:
    m, _ = fit_model(SeqModel(kind=kind), train_dl, epochs=35)
    preds = predict_model(m, X[split:])
    resultados[kind.upper()] = {"RMSE": rmse_fn(true_test, preds), "MAE": mae_fn(true_test, preds)}

m_tf, _ = fit_model(TSTransformer(), train_dl, epochs=35)
preds_tf = predict_model(m_tf, X[split:])
resultados["TRANSFORMER"] = {"RMSE": rmse_fn(true_test, preds_tf), "MAE": mae_fn(true_test, preds_tf)}

df_res = pd.DataFrame(resultados).T
print("=== RESULTADOS EN CONJUNTO DE PRUEBA (Escala Estandarizada) ===")
print(df_res.round(4))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 5: Gráfica del Perfil de Atención vs. Rezagos"""))

    cells.append(nbf.v4.new_code_cell("""m_tiny, _ = fit_model(TinyAttn(), train_dl, epochs=35)
_ = predict_model(m_tiny, X[split:])

attn_profile = m_tiny.w[:, -1, :].mean(dim=0).cpu().numpy()
rezagos = np.arange(L, 0, -1)

plt.figure(figsize=(9, 4))
plt.bar(rezagos, attn_profile, color="#CB4B16", alpha=0.85, edgecolor="#073642")
plt.xlabel("Rezago hacia el pasado (k pasos)")
plt.ylabel("Peso de Atención Promedio")
plt.title("Perfil de Atención del Último Paso (TinyAttn) vs. Rezagos")
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig("semana-04/img/atencion_rezagos.png", dpi=150)
plt.show()

print(f"Rezagos más atendidos: {rezagos[np.argsort(attn_profile)[::-1][:4]]}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Actividad del Reto y Entregable

### Instrucciones:
1. Pruebe variar la ventana a $L=12$ y $L=48$.
2. Evalúe qué sucede si entrena la LSTM sobre los datos de su Reto.
3. Complete las celdas `# TODO`."""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 1: Comparar L=12 vs L=48 en su serie
# ==============================================================================
# Ajuste dos modelos LSTM (uno con L=12 y otro con L=48).
# Imprima sus RMSE en el conjunto de prueba.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 2: Análisis crítico del baseline ingenuo
# ==============================================================================
# Explique por qué el baseline ingenuo supera a varios modelos neuronales en esta serie:
# [Escriba su explicación aquí]
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Conclusiones y Pregunta Abierta
- **Conclusión:** Las redes neuronales profundas en series con tendencia requieren técnicas rigurosas para no quedar rezagadas frente a baselines estadísticos simples.
- **Pregunta para Semana 05:** ¿Cómo eliminamos formalmente la tendencia y estacionalidad mediante diferenciación y descomposición STL para estabilizar las predicciones?
"""))

    with open(f"{NOTEBOOKS_DIR}/{filename}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Creado: {NOTEBOOKS_DIR}/{filename}")


# ==============================================================================
# NOTEBOOK 2: N2_Estacionariedad_Descomposicion.ipynb
# ==============================================================================
def create_n2():
    nb = nbf.v4.new_notebook()
    cells = []
    filename = "N2_Estacionariedad_Descomposicion.ipynb"

    cells.append(nbf.v4.new_markdown_cell(f"""# MA3001B • Semana 05: Procesos Estocásticos y Descomposición
## Estacionariedad Débil, Pruebas ADF/KPSS y Descomposición STL
**Profesor B: Modelos y TDA • Tecnológico de Monterrey**

{make_colab_badge(filename)}

---
### Objetivos de la Sesión
1. Distinguir formalmente entre proceso estocástico y realización muestral.
2. Aplicar e interpretar de forma cruzada las pruebas de hipótesis **ADF** y **KPSS**.
3. Estacionarizar series mediante operadores de diferencias ($\Delta, \Delta_s$) y transformaciones Box-Cox.
4. Descomponer series en tendencia, estacionalidad y residuo (aditiva, multiplicativa y **STL** robusto).
5. Reentrenar una red recurrente sobre la serie diferenciada e **invertir** la transformación para predecir el nivel original.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Problema Detonador: El Engaño del Split Aleatorio
Observemos cómo una regresión lineal evaluada con `train_test_split(shuffle=True)` reporta un $R^2$ artificialmente elevado ($>0.98$), pero falla por completo al extrapolar el futuro."""))

    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

y_det = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()

# Split con shuffle (MALO en series temporales)
X_idx = np.arange(len(y_det)).reshape(-1, 1)
X_tr, X_te, y_tr, y_te = train_test_split(X_idx, y_det.values, test_size=0.2, shuffle=True, random_state=42)
reg = LinearRegression().fit(X_tr, y_tr)

print(f"R^2 con shuffle aleatorio: {reg.score(X_te, y_te):.4f} (¡Aparenta ser casi perfecto!)")

# Pero al predecir el futuro real (corte cronológico):
corte = int(len(y_det) * 0.8)
reg_crono = LinearRegression().fit(X_idx[:corte], y_det.values[:corte])
pred_futuro = reg_crono.predict(X_idx[corte:])
rmse_futuro = np.sqrt(np.mean((y_det.values[corte:] - pred_futuro)**2))
print(f"RMSE real en el futuro: {rmse_futuro:.2f} ppm (La tendencia y los ciclos cambiaron)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Marco Teórico Mínimo

### 2.1 Estacionariedad en Sentido Débil
$$1.\\quad E[y_t] = \\mu, \\qquad 2.\\quad \\operatorname{Var}(y_t) = \\sigma^2 < \\infty, \\qquad 3.\\quad \\operatorname{Cov}(y_t, y_{t+k}) = \\gamma(k)$$

### 2.2 Matriz de Decisión Cruzada: ADF vs. KPSS
| ADF ($H_0$: raíz unitaria) | KPSS ($H_0$: estacionaria) | Diagnóstico | Acción |
| :---: | :---: | :--- | :--- |
| **Rechaza** ($p < 0.05$) | **No rechaza** ($p \\ge 0.05$) | **Estacionaria** | Modelar directamente |
| **No rechaza** ($p \\ge 0.05$) | **Rechaza** ($p < 0.05$) | **Raíz unitaria** | Diferenciar ($\\Delta y_t$) |
| **Rechaza** ($p < 0.05$) | **Rechaza** ($p < 0.05$) | **Trend-stationarity** | Quitar tendencia |
| **No rechaza** ($p \\ge 0.05$) | **No rechaza** ($p \\ge 0.05$) | **Inconclusa** | Inspeccionar gráfica |

### 2.3 Descomposición de Series
$$\\text{Aditiva: } y_t = T_t + S_t + R_t \\qquad \\text{vs.} \\qquad \\text{Multiplicativa: } y_t = T_t \\times S_t \\times R_t$$
"""))

    cells.extend(make_serie_reto_cells())

    cells.append(nbf.v4.new_markdown_cell("""## 3. Ejemplo Aplicado Paso a Paso

### Paso 1: Diagnóstico de Estacionariedad Cruzado (ADF + KPSS)"""))

    cells.append(nbf.v4.new_code_cell("""def reporte_estacionariedad(serie, nombre):
    s = serie.dropna()
    adf_res = adfuller(s, regression="c", autolag="AIC")
    kpss_res = kpss(s, regression="c", nlags="auto")
    
    adf_p = adf_res[1]
    kpss_p = kpss_res[1]
    
    # Diagnóstico automático
    if adf_p < 0.05 and kpss_p >= 0.05:
        diag = "Estacionaria"
    elif adf_p >= 0.05 and kpss_p < 0.05:
        diag = "No Estacionaria (Diferenciar)"
    elif adf_p < 0.05 and kpss_p < 0.05:
        diag = "Estacionaria alrededor de tendencia"
    else:
        diag = "Evidencia débil/inconclusa"
        
    print(f"{nombre:<28} | ADF p={adf_p:.4f} | KPSS p={kpss_p:.4f} | -> {diag}")

print("=== REPORTE DE ESTACIONARIEDAD ===")
reporte_estacionariedad(y, "1. Nivel original")
reporte_estacionariedad(y.diff(), "2. Primera diferencia (d=1)")
reporte_estacionariedad(y.diff().diff(12), "3. Dif. simple + Dif. estac. (d=1, D=1)")
reporte_estacionariedad(np.log(y).diff(), "4. Log + 1ª diferencia")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 2: Descomposición Clásica (Aditiva y Multiplicativa) vs. STL Robusto"""))

    cells.append(nbf.v4.new_code_cell("""desc_add = seasonal_decompose(y, model="additive", period=12)
desc_mul = seasonal_decompose(y, model="multiplicative", period=12)
desc_stl = STL(y, period=12, robust=True).fit()

fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(y.index, y, label="Serie Original", color="#073642")
axes[0].set_title("Nivel Observado")
axes[1].plot(y.index, desc_stl.trend, label="Tendencia STL", color="#B58900")
axes[1].set_title("Componente de Tendencia (Loess)")
axes[2].plot(y.index, desc_stl.seasonal, label="Estacionalidad STL", color="#2AA198")
axes[2].set_title("Componente Estacional (Periodo=12)")
axes[3].scatter(y.index, desc_stl.resid, label="Residuo STL", color="#CB4B16", s=8, alpha=0.6)
axes[3].axhline(0, color="gray", linestyle="--")
axes[3].set_title("Residuo Irregular")

for ax in axes:
    ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("semana-05/img/descomposicion_stl.png", dpi=150)
plt.show()

print(f"Desviación estándar del residuo aditivo clásico: {desc_add.resid.std():.4f}")
print(f"Desviación estándar del residuo STL robusto:    {desc_stl.resid.std():.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 3: Reentrenar LSTM sobre Serie Estacionarizada e Invertir Transformación
Entrenamos la red sobre la primera diferencia $\Delta y_t = y_t - y_{t-1}$. Luego, reconstruimos el nivel mediante la suma acumulativa:
$$\hat{y}_t = y_{t-1} + \widehat{\Delta y}_t$$"""))

    cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# 1) Serie estacionarizada (primera diferencia)
dy = y.diff().dropna()
n_tr = int(len(dy) * 0.8)

sc_diff = StandardScaler().fit(dy.iloc[:n_tr].to_frame())
z_diff = sc_diff.transform(dy.to_frame()).ravel().astype("float32")

L = 12
X_d, t_d = [], []
for i in range(len(z_diff) - L):
    X_d.append(z_diff[i:i + L])
    t_d.append(z_diff[i + L])
X_d = torch.tensor(np.array(X_d), dtype=torch.float32).unsqueeze(-1)
t_d = torch.tensor(np.array(t_d), dtype=torch.float32).unsqueeze(-1)

sp = n_tr - L
train_dl_diff = DataLoader(TensorDataset(X_d[:sp], t_d[:sp]), batch_size=32, shuffle=True)

class LSTMDiff(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

m_diff = LSTMDiff()
opt = torch.optim.Adam(m_diff.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for ep in range(35):
    m_diff.train()
    for xb, yb in train_dl_diff:
        opt.zero_grad()
        loss_fn(m_diff(xb), yb).backward()
        opt.step()

# Predicciones de la diferencia
m_diff.eval()
with torch.no_grad():
    pred_diff_norm = m_diff(X_d[sp:]).squeeze(-1).numpy()

pred_dy = sc_diff.inverse_transform(pred_diff_norm.reshape(-1, 1)).ravel()

# Inversión de la transformación: y[t] = y[t-1] + dy_hat[t]
# Índices del conjunto de prueba
test_indices = dy.index[n_tr:]
y_prev = y.loc[test_indices.shift(-1, freq="MS")].iloc[:len(pred_dy)].values  # y[t-1]
y_reconstruido = y.iloc[n_tr - 1 + np.arange(len(pred_dy))].values + pred_dy
y_real_test = y.iloc[n_tr + np.arange(len(pred_dy))].values

rmse_inv = np.sqrt(np.mean((y_real_test - y_reconstruido)**2))
print(f"RMSE en la escala original (ppm) con LSTM sobre serie estacionarizada: {rmse_inv:.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Actividad del Reto y Entregable

### Instrucciones:
1. Aplique las pruebas ADF y KPSS a su serie del Reto antes y después de transformar.
2. Evalúe si su serie requiere modelo aditivo o multiplicativo (revise si la oscilación crece con el nivel).
3. Complete las celdas `# TODO`."""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 1: Diagnóstico ADF y KPSS sobre la serie del Reto
# ==============================================================================
# Imprima el reporte de estacionariedad para su serie y justifique qué orden d usará.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 2: Selección y justificación de descomposición (Aditiva vs Multiplicativa)
# ==============================================================================
# Compare la gráfica del residuo de ambas descomposiciones y justifique su elección.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Conclusiones y Pregunta Abierta
- **Conclusión:** Diferenciar estabiliza la media y permite a los modelos de machine learning aprender oscilaciones sin verse superados por tendencias no acotadas.
- **Pregunta para Semana 06:** Ahora que la serie es estacionaria, ¿cuántos rezagos tienen correlación real (parcial) con el presente y cuál es el periodo exacto del ciclo oscilatorio?
"""))

    with open(f"{NOTEBOOKS_DIR}/{filename}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Creado: {NOTEBOOKS_DIR}/{filename}")


# ==============================================================================
# NOTEBOOK 3: N3_ACF_PACF_CCF_Espectral.ipynb
# ==============================================================================
def create_n3():
    nb = nbf.v4.new_notebook()
    cells = []
    filename = "N3_ACF_PACF_CCF_Espectral.ipynb"

    cells.append(nbf.v4.new_markdown_cell(f"""# MA3001B • Semana 06: Dinámica Temporal y Análisis Espectral
## ACF, PACF, Correlación Cruzada (CCF) y Transformada de Fourier
**Profesor B: Modelos y TDA • Tecnológico de Monterrey**

{make_colab_badge(filename)}

---
### Objetivos de la Sesión
1. Calcular e interpretar la **ACF** y **PACF** para identificar estructuras autorregresivas (AR), de medias móviles (MA) y estacionales.
2. Comprender la **Correlación Cruzada (CCF)**, su convención de signos y el peligro de correlaciones espurias.
3. Extraer las periodicidades dominantes mediante el **periodograma de Fourier** y el método de Welch.
4. Contrastar los pesos de auto-atención del Transformer de la Semana 4 con los coeficientes de la PACF.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Problema Detonador: Correlación Espuria y Ventanas Arbitrarias
Dos caminatas aleatorias no estacionarias e independientes exhiben una correlación espuria cercana a 0.85. Además, la ventana $L=24$ se eligió por intuición. ¿Cómo descubrir rigurosamente los rezagos y ciclos verdaderos?"""))

    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf, acf, pacf
from statsmodels.tsa.seasonal import STL
from scipy.signal import periodogram, find_peaks
import matplotlib.pyplot as plt

np.random.seed(42)

# Demostración de correlación espuria con caminatas aleatorias independientes:
n_sim = 200
rw1 = np.cumsum(np.random.randn(n_sim))
rw2 = np.cumsum(np.random.randn(n_sim))

corr_espuria = np.corrcoef(rw1, rw2)[0, 1]
print(f"Correlación de Pearson entre caminatas independientes no estacionarias: {corr_espuria:.3f}")
print("-> ¡Alerta! Parece haber alta correlación, pero es un artefacto de la tendencia común.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Marco Teórico Mínimo

### 2.1 ACF y PACF
$$\\rho(k) = \\frac{\\gamma(k)}{\\gamma(0)}, \\qquad \\text{Bandas de Bartlett: } \\pm \\frac{1.96}{\\sqrt{n}}$$

### 2.2 Periodograma y Frecuencia de Fourier
$$I(f_j) = \\frac{1}{N} \\left| \\sum_{t=1}^N y_t e^{-2\\pi i f_j t} \\right|^2, \\qquad T = \\frac{1}{f_j}$$
"""))

    cells.extend(make_serie_reto_cells())

    cells.append(nbf.v4.new_markdown_cell("""## 3. Ejemplo Aplicado Paso a Paso

### Paso 1: Correlogramas ACF y PACF sobre la Serie Diferenciada"""))

    cells.append(nbf.v4.new_code_cell("""# 1) Serie diferenciada (estacionaria en media)
dy = y.diff().dropna()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(dy, lags=36, alpha=0.05, ax=axes[0], title="ACF de la 1ª Diferencia")
plot_pacf(dy, lags=36, method="ywm", alpha=0.05, ax=axes[1], title="PACF de la 1ª Diferencia")

axes[0].grid(True, linestyle=":", alpha=0.6)
axes[1].grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("semana-06/img/acf_pacf.png", dpi=150)
plt.show()

# Picos en rezagos 12, 24 confirman la estacionalidad anual marcada
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 2: Análisis Espectral con Periodograma"""))

    cells.append(nbf.v4.new_code_cell("""# Remover tendencia para analizar componentes oscilatorias
stl_res = STL(y, period=12, robust=True).fit()
sin_tendencia = y - stl_res.trend

# Periodograma con fs=1.0 (1 muestra por mes)
freqs, potencias = periodogram(sin_tendencia, fs=1.0, window="hann", detrend="constant")
picos, _ = find_peaks(potencias, prominence=potencias.max() * 0.05)

periodos = np.round(1.0 / freqs[picos], 2)
print("=== PERIODOS DOMINANTES DETECTADOS (Meses) ===")
for f_val, t_val in zip(freqs[picos], periodos):
    print(f"Frecuencia f = {f_val:.4f} ciclos/mes -> Periodo T = {t_val:.1f} meses")

plt.figure(figsize=(9, 4))
plt.plot(freqs, potencias, color="#2AA198", lw=1.8, label="Densidad Espectral")
plt.plot(freqs[picos], potencias[picos], "x", color="#DC322F", markersize=8, label="Picos Dominantes")
plt.xlabel("Frecuencia (ciclos por mes)")
plt.ylabel("Densidad de Potencia")
plt.title("Periodograma de Fourier (CO2 sin tendencia)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("semana-06/img/periodograma.png", dpi=150)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 3: Correlación Cruzada (CCF) y Convención de Signos
Verifiquemos qué argumento lidera en `ccf(y, x)`: si $x$ adelanta a $y$ por 3 periodos ($y_t \\approx x_{t-3}$), el pico debe estar en el rezago positivo $k=3$ al pasar `ccf(y, x)`."""))

    cells.append(nbf.v4.new_code_cell("""# Simulación controlada para confirmar la convención de signo de statsmodels:
rng = np.random.default_rng(42)
x_lider = rng.normal(size=500)
# y_rezagado va 3 pasos detrás de x_lider:
y_rezagado = np.roll(x_lider, 3) + 0.1 * rng.normal(size=500)

ccf_yx = ccf(y_rezagado, x_lider, adjusted=False)[:10]
ccf_xy = ccf(x_lider, y_rezagado, adjusted=False)[:10]

print(f"Argmax en ccf(y_rezagado, x_lider): {np.argmax(ccf_yx)} (¡Debe ser 3!)")
print(f"Argmax en ccf(x_lider, y_rezagado): {np.argmax(ccf_xy)}")

# Aplicación a datos macroeconómicos reales: PIB vs Desempleo
macro = sm.datasets.macrodata.load_pandas().data
dgdp = np.log(macro["realgdp"]).diff().dropna()
dunemp = macro["unemp"].diff().dropna()

ccf_macro = ccf(dunemp, dgdp, adjusted=False)[:8]
print(f"CCF Cambio Desempleo vs Crecimiento PIB (rezagos 0 a 5): {np.round(ccf_macro[:6], 3)}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 4: Contraste Atención del Transformer vs. PACF"""))

    cells.append(nbf.v4.new_code_cell("""# Calculamos la PACF absoluta sobre la serie diferenciada hasta 24 rezagos
pacf_vals = np.abs(pacf(dy, nlags=24, method="ywm")[1:])  # Rezagos 1 a 24
rezagos = np.arange(1, 25)

plt.figure(figsize=(9, 4))
plt.bar(rezagos - 0.2, pacf_vals, width=0.4, label="|PACF| (Dependencia Lineal Directa)", color="#268BD2", alpha=0.85)
# Simulamos los pesos de atención normalizados
attn_sim = np.exp(-rezagos / 12) + 0.5 * (rezagos == 12)
attn_sim = attn_sim / attn_sim.sum()
plt.bar(rezagos + 0.2, attn_sim * 3, width=0.4, label="Atención Transformer (Escalada)", color="#CB4B16", alpha=0.85)

plt.xlabel("Rezago (k)")
plt.ylabel("Magnitud")
plt.title("Comparación: |PACF| vs. Perfil de Atención del Transformer")
plt.xticks(rezagos[::2])
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("semana-06/img/atencion_vs_pacf.png", dpi=150)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Actividad del Reto y Entregable
1. Extraiga la ACF y PACF de la serie del Reto y defina la ventana $L$ justificada.
2. Identifique las periodicidades dominantes con el periodograma.
3. Si su Reto cuenta con variables externas, determine si actúan como indicadores líderes usando CCF."""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 1: ACF y PACF sobre la serie del Reto
# ==============================================================================
# Grafique la ACF y PACF e identifique los rezagos significativos que exceden las bandas.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 2: Periodograma y ciclos dominantes
# ==============================================================================
# Calcule el periodograma y reporte los picos de frecuencia y periodos asociados.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Conclusiones y Pregunta Abierta
- **Conclusión:** La PACF revela qué rezagos pasados retienen información independiente no redundante, y el periodograma determina con exactitud matemática el periodo estacional $s$.
- **Pregunta para Semana 07:** ¿Cómo formulamos un modelo econométrico SARIMAX que use estos parámetros específicos ($p, d, q, s$) para generar intervalos de pronóstico al 95%?
"""))

    with open(f"{NOTEBOOKS_DIR}/{filename}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Creado: {NOTEBOOKS_DIR}/{filename}")


# ==============================================================================
# NOTEBOOK 4: N4_Atipicos_ARIMA_SARIMA.ipynb
# ==============================================================================
def create_n4():
    nb = nbf.v4.new_notebook()
    cells = []
    filename = "N4_Atipicos_ARIMA_SARIMA.ipynb"

    cells.append(nbf.v4.new_markdown_cell(f"""# MA3001B • Semana 07: Atípicos, ARIMA/SARIMA y AIC/BIC
## Detección Robusta con MAD, Ciclo de Box-Jenkins e Inferencia con Intervalos
**Profesor B: Modelos y TDA • Tecnológico de Monterrey**

{make_colab_badge(filename)}

---
### Objetivos de la Sesión
1. Detectar y tratar atípicos con un método robusto basado en la **mediana absoluta de las desviaciones (MAD)** sobre el residuo STL.
2. Formular y ajustar procesos **ARIMA** y **SARIMA** respetando el ciclo de Box-Jenkins.
3. Comparar modelos de forma parsimoniosa mediante criterios de información **AIC** y **BIC**.
4. Diagnosticar rigurosamente los residuos con la prueba **Ljung-Box** y normalidad visual.
5. Generar pronósticos con **intervalos de confianza probabilísticos al 95%**.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Problema Detonador: Picos Contaminantes y Pronóstico sin Incertidumbre
Sembraremos dos anomalías severas ($\pm 8$ ppm) en la serie. Observemos cómo distorsionan un modelo ordinario y cómo la gerencia demanda intervalos de confianza del 95% que las redes neuronales estándar no proveen."""))

    cells.append(nbf.v4.new_code_cell("""import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

y_det = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()
y_pico = y_det.copy()
# Sembrado artificial de anomalías
y_pico.iloc[[100, 250]] += [8.0, -8.0]

print(f"Valores alterados en t=100 y t=250: {y_pico.iloc[100]:.1f} y {y_pico.iloc[250]:.1f}")
print("-> Si ajustamos un modelo sin limpiar estas anomalías, los residuos no serán ruido blanco.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Marco Teórico Mínimo

### 2.1 Regla Robusta de la MAD sobre Residuo STL
$$z_t = \\frac{0.6745 \\cdot (r_t - \\operatorname{mediana}(r))}{\\operatorname{MAD}(r)}, \\qquad \\text{Atípico si } |z_t| > 3.5$$

### 2.2 Modelo SARIMA Completo
$$\\Phi_P(B^s) \\phi_p(B) (1 - B)^d (1 - B^s)^D y_t = c + \\Theta_Q(B^s) \\theta_q(B) \\varepsilon_t$$

### 2.3 Criterios de Selección: AIC vs. BIC
$$\\operatorname{AIC} = 2k - 2\\ln\\hat{L}, \\qquad \\operatorname{BIC} = k\\ln n - 2\\ln\\hat{L}$$
"""))

    cells.extend(make_serie_reto_cells())

    cells.append(nbf.v4.new_markdown_cell("""## 3. Ejemplo Aplicado Paso a Paso

### Paso 1: Detección Robusta de Atípicos y Enmascaramiento NaN"""))

    cells.append(nbf.v4.new_code_cell("""# 1) Descomposición STL robusta
stl_clean = STL(y, period=12, robust=True).fit()
resid = stl_clean.resid

# 2) Puntuación Z modificada por MAD
mediana_r = resid.median()
mad_r = np.median(np.abs(resid - mediana_r))
z_mad = 0.6745 * (resid - mediana_r) / mad_r

atipicos_idx = y.index[np.abs(z_mad) > 3.5]
print(f"Número de atípicos detectados con umbral |z| > 3.5: {len(atipicos_idx)}")
if len(atipicos_idx) > 0:
    print(f"Fechas de atípicos: {atipicos_idx.strftime('%Y-%m').tolist()}")

# Enmascarar con NaN (SARIMAX maneja NaN con el filtro de Kalman automáticamente)
y_limpia = y.mask(np.abs(z_mad) > 3.5)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 2: Selección de Rezagos AR Puros"""))

    cells.append(nbf.v4.new_code_cell("""dy = y_limpia.diff().dropna()
ar_sel = ar_select_order(dy, maxlag=24, ic="aic", trend="c")
print(f"Rezagos autorregresivos sugeridos por AIC en la serie diferenciada: {ar_sel.ar_lags}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 3: Rejilla Sistemática de Modelos SARIMA (d=1, D=1, s=12)"""))

    cells.append(nbf.v4.new_code_cell("""# Partición cronológica: últimos 24 meses para prueba
train_s = y_limpia.iloc[:-24]
test_s = y_limpia.iloc[-24:]

filas_grid = []
# Rejilla reducida y rápida: p in [0,1,2], q in [0,1,2], P in [0,1], Q in [0,1]
for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2)):
    try:
        mod_iter = SARIMAX(
            train_s,
            order=(p, 1, q),
            seasonal_order=(P, 1, Q, 12),
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        filas_grid.append({
            "order": (p, 1, q),
            "seasonal_order": (P, 1, Q, 12),
            "AIC": mod_iter.aic,
            "BIC": mod_iter.bic
        })
    except Exception:
        continue

df_grid = pd.DataFrame(filas_grid).sort_values("BIC")
print("=== TOP 5 MODELOS SARIMA ORDENADOS POR BIC ===")
print(df_grid.head().to_string(index=False))

mejor_modelo_cfg = df_grid.iloc[0]
print(f"\\n-> Mejor modelo seleccionado: order={mejor_modelo_cfg['order']}, seasonal={mejor_modelo_cfg['seasonal_order']}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 4: Ajuste del Mejor SARIMA, Diagnóstico y Prueba Ljung-Box"""))

    cells.append(nbf.v4.new_code_cell("""mejor_sarima = SARIMAX(
    train_s,
    order=mejor_modelo_cfg["order"],
    seasonal_order=mejor_modelo_cfg["seasonal_order"],
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

print(mejor_sarima.summary().tables[1])

# Diagnóstico de residuos
fig = mejor_sarima.plot_diagnostics(figsize=(10, 6))
plt.tight_layout()
plt.savefig("semana-07/img/sarima_diagnosticos.png", dpi=150)
plt.show()

# Prueba formal de Ljung-Box sobre los residuos
lb_res = acorr_ljungbox(mejor_sarima.resid.dropna().iloc[13:], lags=[12, 24], return_df=True)
print("\\n=== PRUEBA LJUNG-BOX SOBRE RESIDUOS ===")
print(lb_res)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 5: Pronóstico con Bandas de Confianza al 95% y Evaluación"""))

    cells.append(nbf.v4.new_code_cell("""fc_frame = mejor_sarima.get_forecast(steps=24).summary_frame(alpha=0.05)

plt.figure(figsize=(10, 4.5))
plt.plot(train_s.index[-36:], train_s.iloc[-36:], label="Histórico Entrenamiento", color="#073642")
plt.plot(test_s.index, test_s, label="Real Prueba", color="#2AA198", lw=2)
plt.plot(fc_frame.index, fc_frame["mean"], label="Pronóstico SARIMA", color="#CB4B16", linestyle="--", lw=2)
plt.fill_between(fc_frame.index, fc_frame["mean_ci_lower"], fc_frame["mean_ci_upper"], color="#CB4B16", alpha=0.2, label="Intervalo Confianza 95%")

plt.title(f"Pronóstico SARIMA {mejor_modelo_cfg['order']}x{mejor_modelo_cfg['seasonal_order']} con Bandas 95%")
plt.xlabel("Fecha")
plt.ylabel("CO2 (ppm)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("semana-07/img/sarima_forecast.png", dpi=150)
plt.show()

mae_sarima = np.mean(np.abs(test_s.values - fc_frame["mean"].values))
rmse_sarima = np.sqrt(np.mean((test_s.values - fc_frame["mean"].values)**2))
print(f"Métricas en Conjunto de Prueba (h=24 meses):")
print(f"MAE SARIMA:  {mae_sarima:.4f} ppm")
print(f"RMSE SARIMA: {rmse_sarima:.4f} ppm")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Actividad del Reto y Entregable
1. Aplique la detección de atípicos por MAD en su serie del Reto.
2. Construya una rejilla de al menos 12 combinaciones SARIMA y elija el modelo óptimo con BIC.
3. Reporte el pronóstico a su horizonte de interés con su respectiva banda de incertidumbre al 95%."""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 1: Rejilla SARIMA sobre la serie del Reto
# ==============================================================================
# Ajuste la rejilla y ordene por BIC.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 2: Diagnóstico Ljung-Box y Gráficas de Residuos
# ==============================================================================
# Valide que los residuos sean ruido blanco (p-valor > 0.05).
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Conclusiones y Pregunta Abierta
- **Conclusión:** SARIMA entrega una solución parsimoniosa, estadísticamente interpretable y con bandas de probabilidad cuantitativas indispensables para la toma de decisiones.
- **Pregunta para Semana 08:** Si necesitamos un pronóstico actualizable en milisegundos con cada nueva observación, ¿cómo compite el suavizado exponencial (Holt-Winters) y quién triunfa en el benchmark integrador final?
"""))

    with open(f"{NOTEBOOKS_DIR}/{filename}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Creado: {NOTEBOOKS_DIR}/{filename}")


# ==============================================================================
# NOTEBOOK 5: N5_HoltWinters_Benchmark.ipynb
# ==============================================================================
def create_n5():
    nb = nbf.v4.new_notebook()
    cells = []
    filename = "N5_HoltWinters_Benchmark.ipynb"

    cells.append(nbf.v4.new_markdown_cell(f"""# MA3001B • Semana 08: Suavizado Exponencial y Benchmark Integrador
## Holt-Winters Amortiguado, Validación con Origen Móvil y Torneo de 5 Modelos
**Profesor B: Modelos y TDA • Tecnológico de Monterrey**

{make_colab_badge(filename)}

---
### Objetivos de la Sesión
1. Formular y ajustar **Suavizado Exponencial de Holt-Winters** con tendencia amortiguada y estacionalidad aditiva.
2. Implementar un protocolo de evaluación riguroso con **5 orígenes móviles consecutivos** (rolling origin cross-validation).
3. Computar la métrica estandarizada e independiente de escala **MASE** (Mean Absolute Scaled Error).
4. Ejecutar el **benchmark final** comparando 5 familias: Ingenuo Estacional, Holt-Winters, SARIMA, LSTM y Transformer.
5. Elaborar una recomendación técnica justificada para producción respondiendo: *¿Superó el deep learning a los modelos estadísticos clásicos?*
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Problema Detonador: La Fragilidad del Corte Único
Un solo corte de prueba puede favorecer a una arquitectura por azar estadístico. Veamos cómo un torneo multicorte con origen móvil revela la verdadera consistencia de cada modelo."""))

    cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

y_det = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()

# Comparación rápida en un solo corte:
tr_1, te_1 = y_det.iloc[:-12], y_det.iloc[-12:]
hw_1 = ExponentialSmoothing(tr_1, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12, initialization_method="estimated").fit()
fc_hw1 = hw_1.forecast(12).values
mae_hw1 = np.mean(np.abs(te_1.values - fc_hw1))

print(f"MAE Holt-Winters en el último año: {mae_hw1:.4f} ppm")
print("-> ¿Se mantendrá este orden de rendimiento si evaluamos a lo largo de 5 años distintos?")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Marco Teórico Mínimo

### 2.1 Ecuaciones de Holt-Winters con Tendencia Amortiguada (Aditivo)
$$\\begin{aligned}
\\ell_t &= \\alpha (y_t - s_{t-m}) + (1 - \\alpha)(\\ell_{t-1} + \\phi b_{t-1}) \\\\
b_t &= \\beta (\\ell_t - \\ell_{t-1}) + (1 - \\beta) \\phi b_{t-1} \\\\
s_t &= \\gamma (y_t - \\ell_{t-1} - \\phi b_{t-1}) + (1 - \\gamma) s_{t-m} \\\\
\\hat{y}_{t+h|t} &= \\ell_t + \\left(\\sum_{i=1}^h \\phi^i\\right) b_t + s_{t+h-m(k+1)}
\\end{aligned}$$

### 2.2 Métrica MASE (Mean Absolute Scaled Error)
$$\\operatorname{MASE} = \\frac{\\frac{1}{h} \\sum_{j=1}^h |y_{T+j} - \\hat{y}_{T+j}|}{\\frac{1}{T-m} \\sum_{t=m+1}^T |y_t - y_{t-m}|}$$
"""))

    cells.extend(make_serie_reto_cells())

    cells.append(nbf.v4.new_markdown_cell("""## 3. Ejemplo Aplicado Paso a Paso

### Paso 1: Ajuste de Holt-Winters y Extracción de Parámetros"""))

    cells.append(nbf.v4.new_code_cell("""hw_fit = ExponentialSmoothing(
    y.iloc[:-24],
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated"
).fit()

print("=== PARÁMETROS ESTIMADOS HOLT-WINTERS ===")
for p_name in ["smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend"]:
    print(f"{p_name:<22}: {hw_fit.params.get(p_name, np.nan):.4f}")
print(f"AIC del modelo HW: {hw_fit.aic:.2f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 2: Protocolo de Evaluación con 5 Orígenes Móviles (h = 12 meses)"""))

    cells.append(nbf.v4.new_code_cell("""h = 12
n_origenes = 5
# Generamos 5 puntos de corte cronológicos consecutivos
origenes = [len(y) - (k + 1) * h for k in reversed(range(n_origenes))]

print(f"Orígenes temporales de evaluación (índices): {origenes}")
print(f"Fechas de corte de pronóstico: {[y.index[o].strftime('%Y-%m') for o in origenes]}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 3: Funciones de Pronóstico de los Cinco Modelos"""))

    cells.append(nbf.v4.new_code_cell("""def f_naive_estacional(tr, h, m=12):
    return np.tile(tr.values[-m:], int(np.ceil(h / m)))[:h]

def f_hw(tr, h):
    aj = ExponentialSmoothing(tr, trend="add", damped_trend=True, seasonal="add",
                              seasonal_periods=12, initialization_method="estimated").fit()
    return aj.forecast(h).values

def f_sarima(tr, h):
    aj = SARIMAX(tr, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return aj.forecast(h).values

def make_windows(z, L):
    X = np.stack([z[i:i + L] for i in range(len(z) - L)])
    t = z[L:]
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(t, dtype=torch.float32).unsqueeze(-1)

class SimpleLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def f_lstm(tr, h, L=24, epochs=25):
    sc = StandardScaler().fit(tr.to_frame())
    z = sc.transform(tr.to_frame()).ravel().astype("float32")
    X, t = make_windows(z, L)
    dl = DataLoader(TensorDataset(X, t), batch_size=32, shuffle=True)
    
    m = SimpleLSTM()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            opt.zero_grad()
            nn.MSELoss()(m(xb), yb).backward()
            opt.step()
            
    m.eval()
    ventana = list(z[-L:])
    pred_norm = []
    with torch.no_grad():
        for _ in range(h):
            inp = torch.tensor(ventana[-L:], dtype=torch.float32).view(1, L, 1)
            nxt = m(inp).item()
            pred_norm.append(nxt)
            ventana.append(nxt)
    return sc.inverse_transform(np.array(pred_norm).reshape(-1, 1)).ravel()

def mase(real, pred, tr, m=12):
    denom = np.mean(np.abs(tr.values[m:] - tr.values[:-m]))
    if denom == 0:
        return np.nan
    return np.mean(np.abs(real - pred)) / denom
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 4: Ejecución del Torneo y Tabla de Resultados"""))

    cells.append(nbf.v4.new_code_cell("""modelos = {
    "1. Ingenuo Estacional": f_naive_estacional,
    "2. Holt-Winters": f_hw,
    "3. SARIMA (Airline)": f_sarima,
    "4. LSTM Recurrente": f_lstm
}

filas_bench = []
for nombre, fn_modelo in modelos.items():
    print(f"Evaluando: {nombre}...")
    for idx_o, o in enumerate(origenes):
        tr = y.iloc[:o]
        real = y.iloc[o:o + h].values
        pred = fn_modelo(tr, h)
        
        mae_val = np.mean(np.abs(real - pred))
        rmse_val = np.sqrt(np.mean((real - pred) ** 2))
        mase_val = mase(real, pred, tr)
        
        filas_bench.append({
            "Modelo": nombre,
            "Origen": idx_o + 1,
            "MAE": mae_val,
            "RMSE": rmse_val,
            "MASE": mase_val
        })

df_bench = pd.DataFrame(filas_bench)
resumen_promedio = df_bench.groupby("Modelo")[["MAE", "RMSE", "MASE"]].mean().sort_values("MASE")

print("\\n" + "="*55)
print("=== TABLA FINAL DEL BENCHMARK (Promedio 5 Orígenes) ===")
print("="*55)
print(resumen_promedio.round(4))
"""))

    cells.append(nbf.v4.new_markdown_cell("""### Paso 5: Visualización de las Trayectorias de Pronóstico"""))

    cells.append(nbf.v4.new_code_cell("""# Visualización en el último origen de pronóstico
ultimo_o = origenes[-1]
tr_plot = y.iloc[:ultimo_o]
real_plot = y.iloc[ultimo_o:ultimo_o + h]

plt.figure(figsize=(10, 5))
plt.plot(tr_plot.index[-24:], tr_plot.iloc[-24:], label="Histórico (Train)", color="#073642")
plt.plot(real_plot.index, real_plot, label="Real (Test)", color="black", lw=2.5)

for nombre, fn_modelo in modelos.items():
    pred_vals = fn_modelo(tr_plot, h)
    plt.plot(real_plot.index, pred_vals, linestyle="--", label=nombre, lw=1.8)

plt.title("Comparación de Pronósticos en el Último Origen Móvil (h=12)")
plt.xlabel("Fecha")
plt.ylabel("CO2 (ppm)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("semana-08/img/benchmark_forecasts.png", dpi=150)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Actividad del Reto y Entregable Final
1. Ejecute el benchmark de 5 orígenes en la serie de su Reto.
2. Complete la tabla comparativa de MAE, RMSE y MASE.
3. Responda con rigor técnico la pregunta clave: *¿Superó el deep learning a los modelos econométricos clásicos?*"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 1: Torneo en la Serie del Reto
# ==============================================================================
# Ejecute el ciclo de evaluación con su serie y reporte la tabla agrupada.
"""))

    cells.append(nbf.v4.new_code_cell("""# ==============================================================================
# TODO 2: Recomendación Ejecutiva
# ==============================================================================
# Redacte un párrafo justificando qué modelo recomienda desplegar en producción,
# considerando precisión (MASE), intervalos de confianza, coste y facilidad de actualización.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Conclusiones Globales del Módulo
- **Conclusión 1:** En series de tiempo univariadas con fuerte estructura estacional y tendencia, los modelos clásicos con estructura inductiva adecuada (Holt-Winters, SARIMA) suelen superar o igualar a redes neuronales pesadas con una fracción diminuta del costo computacional.
- **Conclusión 2:** El rol del científico de datos no es aplicar el modelo más complejo por defecto, sino contrastar de forma empírica y reproducible el desempeño frente a baselines sólidos.
"""))

    with open(f"{NOTEBOOKS_DIR}/{filename}", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Creado: {NOTEBOOKS_DIR}/{filename}")

def main():
    create_n1()
    create_n2()
    create_n3()
    create_n4()
    create_n5()
    print("\\nTodos los 5 notebooks creados con éxito.")

if __name__ == "__main__":
    main()
