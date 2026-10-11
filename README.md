# MA3001B: Modelos y TDA • Redes Secuenciales y Series de Tiempo

**Desarrollo de Proyectos de Ingeniería de Matemáticas (8vo Semestre)**  
*Profesor B: Alejandro Ucan-Puc* • Tecnológico de Monterrey  
Sitio web del curso con diapositivas interactivas en Reveal.js (editables en Markdown) y tutoriales de laboratorio en Python exportados a HTML.

---

## 1. Estructura del Repositorio

```text
/
├── index.html                      # Portada académica estilo MA2007B (Solarized)
├── assets/
│   ├── css/theme-ucan.css          # Estilos personalizados Solarized para Reveal.js y portada
│   ├── js/                         # Scripts compartidos
│   ├── img/                        # Recursos gráficos
│   └── reveal/                     # Copia local de Reveal.js 5.2.1, KaTeX y plugins
├── semana-04/ … semana-08/
│   ├── index.html                  # Visor Reveal.js configurado con plugins
│   ├── slides.md                   # AQUÍ se edita el contenido de las diapositivas
│   └── img/                        # Gráficas generadas para la presentación
├── notebooks/
│   ├── N1_RedesSecuenciales_Transformers.ipynb      (Semana 4)
│   ├── N2_Estacionariedad_Descomposicion.ipynb      (Semana 5)
│   ├── N3_ACF_PACF_CCF_Espectral.ipynb              (Semana 6)
│   ├── N4_Atipicos_ARIMA_SARIMA.ipynb               (Semana 7)
│   ├── N5_HoltWinters_Benchmark.ipynb               (Semana 8)
│   └── html/                                        # Versiones ejecutadas y exportadas (.html)
├── content/
│   └── propuesta.md                # Fuente de verdad pedagógica y técnica
├── scripts/
│   ├── build_notebooks.sh          # Automatización: ejecuta y exporta notebooks a HTML
│   ├── generate_all_notebooks.py   # Generador programático de notebooks
│   └── new_session.py              # Generador de nuevas sesiones (semana-NN)
├── requirements.txt                # Dependencias fijas reproducibles
├── CHANGELOG.md                    # Registro de correcciones y validaciones de código
└── README.md                       # Guía de uso, edición y despliegue
```

---

## 2. Instalación y Entorno Local

Se recomienda utilizar Python 3.10 o 3.11 en un entorno virtual (`venv`):

```bash
# Clonar el repositorio
git clone https://github.com/alxcn/Models_MA3001B.git
cd Models_MA3001B

# Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias exactas
pip install -r requirements.txt
```

---

## 3. Cómo Ejecutar Localmente

> [!IMPORTANT]
> El plugin Markdown de Reveal.js realiza peticiones locales `fetch` para cargar los archivos `slides.md`. Por políticas de seguridad del navegador (CORS), **abrir `index.html` con doble clic (`file:///...`) no cargará las diapositivas**. Se requiere un servidor HTTP local.

Inicie el servidor HTTP integrado de Python:

```bash
python -m http.server 8000
```

Abra su navegador en:
- **Portada:** [http://localhost:8000/](http://localhost:8000/)
- **Semana 04:** [http://localhost:8000/semana-04/](http://localhost:8000/semana-04/)
- **Semana 05:** [http://localhost:8000/semana-05/](http://localhost:8000/semana-05/)
- **Semana 06:** [http://localhost:8000/semana-06/](http://localhost:8000/semana-06/)
- **Semana 07:** [http://localhost:8000/semana-07/](http://localhost:8000/semana-07/)
- **Semana 08:** [http://localhost:8000/semana-08/](http://localhost:8000/semana-08/)
- **Tutoriales HTML:** [http://localhost:8000/notebooks/html/N1_RedesSecuenciales_Transformers.html](http://localhost:8000/notebooks/html/N1_RedesSecuenciales_Transformers.html)

---

## 4. Cómo Editar las Diapositivas

Para modificar el contenido de cualquier presentación, **únicamente edite el archivo `slides.md`** dentro de la carpeta correspondiente (e.g. `semana-04/slides.md`). No necesita editar los archivos `.html`.

### Convenciones de Sintaxis Reveal.js:
- **Diapositiva horizontal:** Tres guiones con saltos de línea:
  ```markdown
  ---
  ```
- **Diapositiva vertical (profundización):** Dos guiones con saltos de línea:
  ```markdown
  --
  ```
- **Notas del orador (Speaker Notes con cronómetro):**
  ```markdown
  Note:
  [00:15] Tiempo estimado. Puntos clave a enfatizar.
  ```
- **Fórmulas matemáticas en KaTeX:**
  - En línea: `$E[y_t] = \mu$`
  - En bloque: `$$\operatorname{MASE} = \frac{\dots}{\dots}$$`
- **Animaciones paso a paso (Fragmentos):**
  ```markdown
  - Idea a revelar paso a paso <!-- .element: class="fragment" -->
  ```
- **Impresión a PDF:** Añada `?print-pdf` a la URL (e.g. `http://localhost:8000/semana-04/?print-pdf`) y utilice la opción *Imprimir a PDF* de su navegador.

---

## 5. Cómo Ejecutar y Exportar los Notebooks

Para volver a ejecutar los notebooks de principio a fin y regenerar las versiones HTML en `notebooks/html/`:

```bash
./scripts/build_notebooks.sh
```

El script utiliza `jupyter nbconvert --to html --execute` con un límite de tiempo de 300 segundos por celda. Cada notebook está optimizado para completarse en menos de 2 minutos en CPU.

### Uso con la Serie del Reto
En cada uno de los 5 notebooks existe la celda `SERIE_RETO`:
```python
USAR_SERIE_RETO = False  # Cambiar a True para cargar la serie de su equipo
```
Al cambiar a `True`, los equipos pueden cargar su propio conjunto de datos CSV manteniendo intacto el pipeline de análisis.

---

## 6. Automatización: Crear Nuevas Sesiones

Para inicializar una nueva sesión de clase y añadir automáticamente la tarjeta correspondiente a la portada:

```bash
python scripts/new_session.py 09 "Inferencia Causal y Grafos en Series de Tiempo"
```

Esto generará la carpeta `semana-09/` con su `index.html` y una plantilla de `slides.md`, y registrará el enlace en la portada `index.html`.

---

## 7. Publicación en GitHub Pages

El repositorio está estructurado para publicarse directamente desde la raíz (`/`) de la rama `main`:

1. Realice commit y push de sus cambios:
   ```bash
   git add .
   git commit -m "Actualización de diapositivas y notebooks"
   git push origin main
   ```
2. En GitHub, ingrese a **Settings > Pages**.
3. En **Build and deployment > Source**, seleccione **Deploy from a branch**.
4. Elija la rama `main` y la carpeta `/ (root)`. Guarde los cambios.
5. Su sitio quedará publicado en:
   `https://alxcn.github.io/Models_MA3001B/`

---

## 8. Licencia y Créditos

Desarrollado para el Tecnológico de Monterrey por el **Prof. Alejandro Ucan-Puc**.  
Material bajo licencia académica.
