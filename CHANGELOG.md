# Registro de Cambios y Correcciones Técnicas (CHANGELOG)

**Curso:** MA3001B - Desarrollo de Proyectos de Ingeniería de Matemáticas  
**Módulo:** Modelos y TDA (Profesor B)  
**Profesor:** Alejandro Ucan-Puc • Tecnológico de Monterrey  
**Fecha:** Octubre 2026  

---

## 1. Correcciones de Código y Ajustes de API sobre `content/propuesta.md`

Durante la implementación y ejecución automatizada de los 5 notebooks (`N1` a `N5`) y las diapositivas Reveal.js, se detectaron y corrigieron los siguientes puntos técnicos:

### 1.1 Convención de Signos en Correlación Cruzada (`statsmodels.tsa.stattools.ccf`)
- **Problema en el documento base:**  
  En la Semana 6 de `propuesta.md` se proponía comprobar empíricamente qué orden de argumentos pasa a `ccf(x, y)` para detectar si una variable lidera a otra.
- **Resultado de la verificación:**  
  En una simulación controlada donde $a_t \sim N(0, 1)$ y $b_t = a_{t-3} + 0.1 \varepsilon_t$ (es decir, $a$ adelanta a $b$ por 3 periodos):
  - `np.argmax(ccf(a, b, adjusted=False)[:10])` arrojó $4$ (ruido aleatorio).
  - `np.argmax(ccf(b, a, adjusted=False)[:10])` arrojó exactamente **$3$**.
- **Ajuste aplicado:**  
  En `semana-06/slides.md` y `N3_ACF_PACF_CCF_Espectral.ipynb`, se documentó explícitamente que la función `ccf(y, x)` calcula $E[y_{t+k} x_t]$. Para probar si una variable exógena $x$ es un indicador líder de la serie objetivo $y$, la llamada correcta debe pasar la serie objetivo como primer argumento: `ccf(serie_objetivo, variable_lider)`.

### 1.2 Formas y Tipos de Tensores en PyTorch (`torch.float32`)
- **Problema:**  
  `make_windows` producía arreglos numpy que al convertirse con `torch.tensor(X)` heredaban el tipo `float64` por defecto de pandas/numpy. Las capas `nn.Linear`, `nn.RNN`, `nn.LSTM` y `nn.TransformerEncoderLayer` en PyTorch operan por defecto con `float32`, generando excepciones de tipo: `RuntimeError: expected scalar type Float but found Double`.
- **Ajuste aplicado:**  
  Se aseguró la conversión explícita `torch.tensor(X, dtype=torch.float32).unsqueeze(-1)` y `torch.tensor(t, dtype=torch.float32).unsqueeze(-1)` en todos los notebooks (`N1`, `N2`, `N5`) y en las diapositivas.

### 1.3 Claves de Parámetros en `ExponentialSmoothing` (Semana 8)
- **Problema:**  
  En versiones modernas de `statsmodels` (0.14+), el diccionario `hw.params` utiliza claves descriptivas:
  - `smoothing_level` ($\alpha$)
  - `smoothing_trend` ($\beta$)
  - `smoothing_seasonal` ($\gamma$)
  - `damping_trend` ($\phi$)
- **Ajuste aplicado:**  
  Se actualizaron las llamadas de impresión y extracción en `N5` y `semana-08/slides.md` utilizando estas claves estandarizadas.

### 1.4 Manejo de Advertencias en Prueba KPSS (`InterpolationWarning`)
- **Problema:**  
  La función `kpss` emite advertencias cuando el estadístico calculado cae fuera de los valores críticos tabulados (p-valores menores a 0.01 o mayores a 0.10).
- **Ajuste aplicado:**  
  Se incorporó filtrado de advertencias controlado en los notebooks y se clarificó en el texto que los p-valores extremos reportados por `kpss` quedan truncados a los límites de la tabla.

### 1.5 Manejo de Valores Faltantes (`NaN`) en Detección de Atípicos con SARIMAX (Semana 7)
- **Problema:**  
  Al sustituir atípicos con `np.nan`, las opciones por defecto de SARIMAX `enforce_stationarity=True` y `enforce_invertibility=True` pueden generar advertencias de optimización numérica en muestras pequeñas.
- **Ajuste aplicado:**  
  Se configuró `enforce_stationarity=False` y `enforce_invertibility=False` al ajustar la rejilla de modelos, permitiendo que el filtro de Kalman estime los estados no observados sin detener la optimización.

### 1.6 Optimización de Cómputo para CPU (< 5 minutos)
- **Ajuste aplicado:**  
  Para asegurar que los notebooks corran en CPU en cualquier entorno estándar (y en Google Colab) en menos de 2 minutos, se calibraron las épocas de entrenamiento a 25–40 épocas y se fijaron 5 orígenes en la validación móvil de la Semana 8.

### 1.7 Corrección en Serialización de Celdas de Notebooks (`nb.cells`)
- **Problema detectado:**  
  En el script generador `generate_all_notebooks.py`, las celdas se acumulaban en la lista `cells` pero no se asignaban al atributo `nb.cells` del objeto notebook antes de escribir con `nbf.write()`, lo que provocó que los archivos `.ipynb` iniciales tuvieran únicamente 72 bytes con lista de celdas vacía `[]`.
- **Ajuste aplicado:**  
  Se añadió la asignación explícita `nb.cells = cells` en las 5 funciones constructoras (`create_n1` a `create_n5`). Los archivos `.ipynb` ahora pesan entre 12 KB y 16 KB con contenido completo.

### 1.8 Desacoplamiento de Rutas de Guardado (`plt.savefig`)
- **Problema detectado:**  
  Al ejecutarse con `nbconvert`, el directorio de trabajo del kernel es `notebooks/`. Llamadas relativas a `semana-NN/img/` fallaban con `FileNotFoundError`.
- **Ajuste aplicado:**  
  Se retiraron las llamadas de guardado a disco dentro de las celdas del notebook, apoyándose en la visualización interactiva `plt.show()` que genera los gráficos embebidos directamente en el HTML.

### 1.9 Detección de Protocolo Local `file://` en Diapositivas
- **Problema detectado:**  
  Si un usuario abre `semana-NN/index.html` haciendo doble clic desde el explorador de archivos (`file://`), los navegadores bloquean la lectura de `slides.md` por restricciones de seguridad (CORS), mostrando una presentación en blanco.
- **Ajuste aplicado:**  
  Se integró un script de detección de protocolo en la portada y en las 5 presentaciones que muestra un banner flotante con las instrucciones exactas para iniciar el servidor local `python3 -m http.server 8000`.

---

## 2. Registro de Archivos y Recursos Generados

| Archivo / Carpeta | Descripción | Estado de Verificación |
| :--- | :--- | :---: |
| `index.html` | Portada principal adaptada con tema Solarized, tarjetas de semanas 04–08, notebooks N1–N5 y ecosistema | **Ejecutado y Verificado** |
| `assets/reveal/` | Copia local completa de Reveal.js 5.2.1 con plugins Markdown, Highlight, Notes, Math, Search, Zoom | **Local y Verificado** |
| `assets/reveal/katex/` | Instalación local completa de KaTeX para renderizado de fórmulas sin conexión a internet | **Local y Verificado** |
| `assets/css/theme-ucan.css` | Hoja de estilos académica Solarized compatible con el sitio de referencia MA2007B | **Verificado** |
| `semana-04/` a `semana-08/` | Carpetas de sesiones con `index.html` (Reveal.js) y `slides.md` (editables en Markdown) | **5/5 Creados y Verificados** |
| `notebooks/N1` a `N5` | 5 Notebooks en formato `.ipynb` con estructura ABP, celda `SERIE_RETO` y celdas `# TODO` | **5/5 Ejecutados sin errores** |
| `notebooks/html/N1` a `N5` | 5 Exportaciones HTML completas generadas vía `nbconvert --execute` | **5/5 Generados con éxito** |
| `scripts/build_notebooks.sh` | Script bash de automatización para ejecución y exportación a HTML | **Ejecutado con éxito (Exit 0)** |
| `scripts/new_session.py` | Script interactivo para generar nuevas semanas y actualizar la portada | **Probado y Verificado** |
| `requirements.txt` | Versiones exactas y reproducibles del entorno Python 3.11 | **Generado y Congelado** |
