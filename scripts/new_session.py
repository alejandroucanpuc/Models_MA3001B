#!/usr/bin/env python3
"""
Script de automatización para crear una nueva sesión de clase en Reveal.js y actualizar la portada.
Uso:
    python scripts/new_session.py <NN> "<Título de la sesión>"

Ejemplo:
    python scripts/new_session.py 09 "Inferencia Causal en Series Temporales"
"""

import sys
import os
import re

SESSION_INDEX_TEMPLATE = """<!doctype html>
<html lang="es">
<head>
\t<meta charset="utf-8">
\t<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">

\t<title>MA3001B - Semana {num}: {title}</title>

\t<link rel="stylesheet" href="../assets/reveal/dist/reset.css">
\t<link rel="stylesheet" href="../assets/reveal/dist/reveal.css">
\t<link rel="stylesheet" href="../assets/reveal/dist/theme/solarized.css">
\t<link rel="stylesheet" href="../assets/reveal/plugin/highlight/monokai.css">
\t<link rel="stylesheet" href="../assets/reveal/katex/katex.min.css">
\t<link rel="stylesheet" href="../assets/css/theme-ucan.css">

\t<!-- Font Awesome -->
\t<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
\t<div class="reveal">
\t\t<div class="slides">
\t\t\t<section data-markdown="slides.md"
\t\t\t         data-separator="^\\n---\\n"
\t\t\t         data-separator-vertical="^\\n--\\n"
\t\t\t         data-separator-notes="^Note:">
\t\t\t</section>
\t\t</div>
\t</div>

\t<!-- Barra de navegación fija / botón volver -->
\t<div style="position: fixed; bottom: 12px; left: 16px; z-index: 99; font-family: 'Inter', sans-serif;">
\t\t<a href="../index.html" style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; background: rgba(238, 232, 213, 0.85); backdrop-filter: blur(4px); border: 1px solid rgba(181, 137, 0, 0.3); border-radius: 6px; font-size: 12px; font-weight: 600; color: #073642; text-decoration: none; box-shadow: 0 2px 5px rgba(0,0,0,0.06); transition: all 0.2s ease;">
\t\t\t<i class="fa-solid fa-arrow-left" style="color: #CB4B16;"></i> Portada MA3001B
\t\t</a>
\t</div>

\t<script src="../assets/reveal/dist/reveal.js"></script>
\t<script src="../assets/reveal/plugin/markdown/markdown.js"></script>
\t<script src="../assets/reveal/plugin/highlight/highlight.js"></script>
\t<script src="../assets/reveal/plugin/notes/notes.js"></script>
\t<script src="../assets/reveal/plugin/math/math.js"></script>
\t<script src="../assets/reveal/plugin/search/search.js"></script>
\t<script src="../assets/reveal/plugin/zoom/zoom.js"></script>
\t<script>
\t\tReveal.initialize({
\t\t\thash: true,
\t\t\tslideNumber: 'c/t',
\t\t\tcontrols: true,
\t\t\tprogress: true,
\t\t\tcenter: false,
\t\t\ttransition: 'slide',
\t\t\tkatex: {
\t\t\t\tlocal: '../assets/reveal/katex'
\t\t\t},
\t\t\tplugins: [
\t\t\t\tRevealMarkdown,
\t\t\t\tRevealHighlight,
\t\t\t\tRevealNotes,
\t\t\t\tRevealMath.KaTeX,
\t\t\t\tRevealSearch,
\t\t\t\tRevealZoom
\t\t\t]
\t\t});
\t</script>
</body>
</html>
"""

SLIDES_TEMPLATE = """# Semana {num}: {title}
### MA3001B: Modelos y TDA • Profesor B
**Tecnológico de Monterrey**

Note:
[00:00] Bienvenida a la sesión {num}.
Duración estimada: 180 min (cronograma de 3 horas).

---

## Agenda de la Sesión (180 min)

| Minutos | Bloque | Actividad |
| --- | --- | --- |
| 0–25 | Detonador | Diagnóstico inicial y preguntas al grupo |
| 25–75 | Teoría Básica | Fundamentos matemáticos y conceptuales |
| 75–85 | Descanso | Pausa |
| 85–120 | Paquete en Vivo | Comandos, parámetros clave y errores típicos |
| 120–170 | Actividad del Reto | Aplicación práctica sobre la serie del Reto |
| 170–180 | Cierre | Conclusiones y boleto de salida |

Note:
[00:05] Presentar la agenda y los tiempos de entrega.

---

## Problema Detonador

<div class="alert-box">
  <strong>Situación a diagnosticar:</strong> Plantea aquí el código roto o la gráfica incomprensible.
</div>

```python
# Código inicial con problema
import numpy as np
# TODO: Mostrar comportamiento inesperado
```

¿Por qué ocurre este comportamiento y qué supuesto se está violando?

Note:
[00:15] Trabajar en equipos de 3 o 4 alumnos durante 15 minutos proponiendo hipótesis.

---

## Objetivos de Aprendizaje

- Objetivo 1: Comprender los conceptos clave de la sesión {num}. <!-- .element: class="fragment" -->
- Objetivo 2: Implementar los algoritmos en Python usando el ecosistema científico. <!-- .element: class="fragment" -->
- Objetivo 3: Evaluar los resultados en la serie del Reto con métricas cuantitativas. <!-- .element: class="fragment" -->

Note:
[00:25] Revisar alineación con el entregable del Reto.

---

## Teoría Básica

### Concepto Principal

Una definición concisa y formal:

$$
E[y_t] = \\mu, \\qquad \\text{{Var}}(y_t) = \\sigma^2
$$

- Primera propiedad relevante. <!-- .element: class="fragment" -->
- Segunda propiedad relevante. <!-- .element: class="fragment" -->

Note:
[00:45] Enfatizar la interpretación intuitiva de la fórmula.

---

## Paquetes y Comandos

| Comando | Parámetros clave | Qué devuelve / Efecto |
| --- | --- | --- |
| `funcion_ejemplo(x)` | `param=True` | Objeto con resultados |

Note:
[01:30] Demostración en vivo en Jupyter.

---

## Actividad del Reto

<div class="highlight-box">
  <strong>Entregable de la semana {num}:</strong> Aplicar la metodología a la serie del Reto.
</div>

- Replicar el análisis del notebook correspondiente.
- Responder a las preguntas de diagnóstico.
- Generar reporte de métricas y gráficas.

Note:
[02:15] Monitorear el progreso por equipo.

---

## Cierre y Pregunta Abierta

- Conclusión principal de la sesión {num}.
- **Pregunta para la siguiente semana:** ¿Cómo se conecta este resultado con los siguientes modelos?

Note:
[02:55] Boleto de salida y despedida.
"""

def main():
    if len(sys.argv) < 3:
        print("Uso: python scripts/new_session.py <NN> \"<Título>\"")
        sys.exit(1)

    raw_num = sys.argv[1].strip()
    num = f"{int(raw_num):02d}"
    title = sys.argv[2].strip()

    session_dir = f"semana-{num}"
    os.makedirs(session_dir, exist_ok=True)

    index_path = os.path.join(session_dir, "index.html")
    slides_path = os.path.join(session_dir, "slides.md")

    if not os.path.exists(index_path):
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(SESSION_INDEX_TEMPLATE.format(num=num, title=title))
        print(f"Creado: {index_path}")
    else:
        print(f"Ya existe: {index_path}")

    if not os.path.exists(slides_path):
        with open(slides_path, "w", encoding="utf-8") as f:
            f.write(SLIDES_TEMPLATE.format(num=num, title=title))
        print(f"Creado: {slides_path}")
    else:
        print(f"Ya existe: {slides_path}")

    # Actualizar index.html principal si existe
    home_path = "index.html"
    if os.path.exists(home_path):
        with open(home_path, "r", encoding="utf-8") as f:
            content = f.read()

        card_marker = f'semana-{num}/index.html'
        if card_marker not in content:
            new_card = f"""
                <!-- Semana {num} -->
                <div class="card-solar p-5 rounded-xl flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between mb-3">
                            <span class="w-8 h-8 bg-solar-orange/15 text-solar-orange rounded flex items-center justify-center font-mono font-bold text-xs">{num}</span>
                            <span class="text-[10px] font-mono text-solar-base01">Sesión</span>
                        </div>
                        <h3 class="text-lg font-serif font-bold text-solar-base02 mb-1">{title}</h3>
                        <p class="text-solar-base01 text-xs mb-4">Material y diapositivas de la Semana {num}.</p>
                    </div>
                    <a href="semana-{num}/index.html" class="p-2 bg-solar-base2 rounded text-solar-base02 hover:bg-solar-orange hover:text-white transition-colors text-xs font-medium flex justify-between items-center">
                        <span>Ver Presentación</span><i class="fas fa-arrow-right text-[10px]"></i>
                    </a>
                </div>
"""
            # Insertar antes del cierre de grid en la sección sesiones
            insertion_point = content.find('<!-- /grid-sesiones -->')
            if insertion_point != -1:
                content = content[:insertion_point] + new_card + "\n                " + content[insertion_point:]
                with open(home_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Tarjeta de semana {num} agregada a {home_path}")

    print(f"Sesión {num} lista en ./{session_dir}/")

if __name__ == "__main__":
    main()
