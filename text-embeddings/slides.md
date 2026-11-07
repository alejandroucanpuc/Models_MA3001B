# Text Embeddings in Python
### Math · Pipelines · Metrics · Cost
**MA3001B: Modelos y TDA • Special Topic / Deep Learning NLP**  
Prof. Alejandro Ucan-Puc • Tecnológico de Monterrey

<div style="margin-top: 25px;">
  <span class="badge badge-orange">Deep Learning & NLP</span>
  <span class="badge badge-teal">Information Retrieval</span>
  <span class="badge badge-amber">Vector Search</span>
</div>

<span class="time-pill"><i class="fa-solid fa-clock"></i> 120 min lecture + 60 min hands-on lab</span>

Note:
[00:00 - 00:05]
Welcome to the Text Embeddings masterclass.
Timing breakdown for the 180 min session:
- 00:00 - 00:10 Motivation & Problem
- 00:10 - 00:40 The Mathematical Foundation
- 00:40 - 00:55 Models, Dimensions & Quantization
- 00:55 - 01:20 End-to-End Pipelines & ANN
- 01:20 - 01:40 Evaluation Metrics (Recall@k, MRR, nDCG)
- 01:40 - 01:55 Cost Modeling & Storage
- 01:55 - 02:00 Synthesis & Takeaways
- 02:00 - 03:00 Hands-on Lab: Your Own Dataset

---

## Agenda de la Sesión (180 min)

| Bloque | Tema | Duración | Enfoque Principal |
| :---: | :--- | :---: | :--- |
| **1** | **Why embeddings?** | 10 min | Falla de keyword search, intuición geométrica |
| **2** | **The math** | 30 min | Espacio métrico, pooling, esfera $S^{d-1}$, InfoNCE, MRL |
| **3** | **Models & Quantization** | 15 min | Selección de modelos, dimensiones, compresión (int8, binary) |
| **4** | **Pipelines & Search** | 25 min | Chunking, tokenización, búsqueda exacta y ANN (HNSW, IVF) |
| **5** | **Evaluation Metrics** | 20 min | Recall@k, Hit@k, MRR, nDCG@k y construcción de test sets |
| **6** | **Cost & Engineering** | 15 min | Indexación, queries, memoria RAM, self-host vs. API |
| **7** | **Wrap-up** | 5 min | 5 principios de diseño en producción |
| **Lab** | **Hands-on Lab** | 60 min | Pipeline reproducible sobre dataset propio o benchmark |

Note:
[00:05 - 00:10]
Present the roadmap. Emphasize that in modern AI engineering, retrieval quality determines 80% of downstream RAG performance, and embeddings are the core mathematical primitive.

---

# 1. Why Embeddings?
<span class="badge badge-orange">Block 1 • 10 min</span>

> "You shall know a word by the company it keeps."  
> — John Rupert Firth (1957)

Note:
[00:10 - 00:12]
Begin with the fundamental limitation of lexical (string-matching) search.

---

## Keyword Search Fails on Meaning

Tres consultas reales buscando asistencia vehicular en carretera:

<div class="alert-box">
  <ul>
    <li><code>"how to fix a flat tire"</code></li>
    <li><code>"changing a punctured tyre"</code></li>
    <li><code>"roadside wheel repair"</code></li>
  </ul>
</div>

- **Falla léxica:** El solapamiento exacto de palabras es prácticamente **cero** (ortografía británica vs. americana, sinónimos, estructura). <!-- .element: class="fragment" -->
- **Identidad semántica:** Describen con exactitud la **misma intención del usuario**. <!-- .element: class="fragment" -->
- **BM25 / TF-IDF:** Privilegia coincidencia superficial de tokens, ignorando conceptos equivalentes. <!-- .element: class="fragment" -->

<div class="highlight-box">
  <strong>Principio fundamental de los embeddings:</strong>
  $$\mathrm{dist}(e_1, e_2) \to 0 \iff \mathrm{semantics}(t_1) \approx \mathrm{semantics}(t_2)$$
</div>

Note:
[00:12 - 00:15]
Ask students for real-world examples: medical terminology (myocardial infarction vs. heart attack), legal contracts, multilingual search.

--

## Where Embeddings Power Systems

<div class="container">
  <div class="col">
    <h4><i class="fa-solid fa-magnifying-glass"></i> Retrieval & RAG</h4>
    <p>Búsqueda semántica sobre millones de chunks. Provee contexto fáctico a los LLMs para anular alucinaciones.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4><i class="fa-solid fa-layer-group"></i> Discovery & Cluster</h4>
    <p>Agrupamiento no supervisado (HDBSCAN, UMAP) para descubrir temas emergentes en tickets y bitácoras.</p>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-tags"></i> Few-Shot Classifier</h4>
    <p>Sonda lineal: clasificador logístico sobre embeddings congelados. Alta exactitud con 5 a 10 ejemplos.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4><i class="fa-solid fa-bullseye"></i> RecSys & Anomalías</h4>
    <p>Espacios duales usuario-ítem, filtrado por similitud angular y detección de valores atípicos espaciales.</p>
  </div>
</div>

Note:
[00:15 - 00:18]
Stress that embeddings are not just for RAG; they are the general representation layer for all unstructured data.

--

## Learning Goals

Al concluir esta sesión de 180 minutos, serás capaz de:

1. **Formalizar un modelo de embedding** como la composición $f_\theta = \mathrm{norm}\circ \mathrm{pool}\circ \mathrm{Enc}_\theta \circ \tau$ en la hipersfera $S^{d-1}$. <!-- .element: class="fragment" -->
2. **Demostrar la identidad geométrica** que unifica Similitud Coseno, Producto Punto y Distancia Euclidiana sobre vectores normalizados. <!-- .element: class="fragment" -->
3. **Diseñar un pipeline de producción:** seleccionar chunking, tratar asimetría query/doc y escalar con índices ANN. <!-- .element: class="fragment" -->
4. **Evaluar cuantitativamente la recuperación** usando Recall@k, MRR y nDCG sobre un test set con Ground Truth. <!-- .element: class="fragment" -->
5. **Calcular la economía de infraestructura:** proyectar costos de tokens de indexación, consultas mensuales y consumo de RAM. <!-- .element: class="fragment" -->

Note:
[00:18 - 00:20]
Review the learning contract. The session balances rigorous linear algebra with real production engineering numbers.

---

# 2. The Math
<span class="badge badge-teal">Block 2 • 30 min</span>

> La geometría de representaciones densas: hipersferas, proyecciones y optimización contrastiva.

Note:
[00:20 - 00:25]
Transition into the mathematical formulation. We define the model formally, examine the pooling operations, and analyze the geometry of the unit sphere.

---

## Formal Definition

<div class="def-box">
  <strong>Definición (Modelo de Embedding):</strong> Sea $\Sigma$ un alfabeto finito y $\Sigma^* = \bigcup_{k=0}^\infty \Sigma^k$ el conjunto de cadenas de texto finitas. Un <strong>modelo de embedding</strong> parametrizado por $\theta$ es una función medible:
  $$f_\theta : \Sigma^* \longrightarrow \mathbb{R}^d$$
  que mapea cualquier cadena de texto arbitraria a un espacio vectorial continuo de dimensión fija $d \in \mathbb{N}$, entrenada de modo que la <strong>afinidad semántica</strong> entre textos induzca <strong>proximidad geométrica</strong> en $\mathbb{R}^d$.
</div>

- **Representación densa:** Cada coordenada en $\mathbb{R}^d$ es un número real continuo y denso ($d \in [384, 3072]$).
- **Sin ejes monosemánticos:** Los conceptos no habitan en una dimensión única; son patrones distribuidos en el espacio latente.

Note:
[00:25 - 00:28]
Highlight the contrast with bag-of-words. In dense vectors, dimensionality is small and information is distributed across all dimensions.

--

## Anatomy of $f_\theta$

Un modelo moderno descompone su cómputo en cuatro etapas sucesivas:

<div class="def-box">
  $$f_\theta = \mathrm{norm} \circ \mathrm{pool} \circ \mathrm{Enc}_\theta \circ \tau$$
</div>

<svg viewBox="0 0 760 135" style="width:100%; max-height:140px; margin: 15px auto; display:block;">
  <!-- Text Input -->
  <rect x="5" y="42" width="105" height="50" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="57" y="64" text-anchor="middle" font-family="Crimson Text" font-weight="bold" font-size="13" fill="#073642">Texto x</text>
  <text x="57" y="80" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">x ∈ Σ*</text>
  <path d="M 110 67 L 140 67" stroke="#CB4B16" stroke-width="2"/>

  <!-- Tokenizer -->
  <rect x="145" y="42" width="120" height="50" rx="8" fill="#FDF6E3" stroke="#CB4B16" stroke-width="2"/>
  <text x="205" y="64" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#CB4B16">Tokenizador τ</text>
  <text x="205" y="80" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">(t₁,...,tₙ)</text>
  <path d="M 265 67 L 295 67" stroke="#CB4B16" stroke-width="2"/>

  <!-- Encoder -->
  <rect x="300" y="38" width="150" height="58" rx="8" fill="#FDF6E3" stroke="#2AA198" stroke-width="2.5"/>
  <text x="375" y="62" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#2AA198">Transformer Enc_θ</text>
  <text x="375" y="80" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#073642">H ∈ ℝ^(n×d)</text>
  <path d="M 450 67 L 480 67" stroke="#2AA198" stroke-width="2"/>

  <!-- Pooling -->
  <rect x="485" y="42" width="115" height="50" rx="8" fill="#FDF6E3" stroke="#6C71C4" stroke-width="2"/>
  <text x="542" y="64" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#6C71C4">Pooling</text>
  <text x="542" y="80" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">z ∈ ℝ^d</text>
  <path d="M 600 67 L 630 67" stroke="#6C71C4" stroke-width="2"/>

  <!-- Normalization -->
  <rect x="635" y="42" width="120" height="50" rx="8" fill="#EEE8D5" stroke="#859900" stroke-width="2.5"/>
  <text x="695" y="64" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#859900">Norm-L2</text>
  <text x="695" y="80" text-anchor="middle" font-family="Fira Code" font-weight="bold" font-size="11" fill="#859900">e ∈ S^(d-1)</text>
</svg>

Note:
[00:28 - 00:30]
Show the pipeline. Each stage serves a specific mathematical transformation.

--

## The Four Stages of $f_\theta$

<div class="container">
  <div class="col">
    <h4>1. Tokenizador $\tau$</h4>
    <p>Segmenta el texto en subpalabras discretas (BPE o WordPiece):</p>
    $$x \mapsto (t_1, \dots, t_n)$$
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4>2. Transformer Encoder $\mathrm{Enc}_\theta$</h4>
    <p>Genera representaciones contextualizadas por token:</p>
    $$H = [h_1, \dots, h_n]^\top \in \mathbb{R}^{n \times d}$$
  </div>
  <div class="col">
    <h4>3. Operador de Pooling</h4>
    <p>Colapsa los $n$ vectores de tokens en un único vector estático:</p>
    $$z = \mathrm{pool}(H) \in \mathbb{R}^d$$
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4>4. Normalización $L_2$</h4>
    <p>Proyecta radialmente sobre la hipersfera unitaria:</p>
    $$e = z / \|z\|_2 \in S^{d-1}$$
  </div>
</div>

Note:
[00:30 - 00:32]
Point out that the transformer outputs $n$ vectors (one per token). Pooling is essential to convert variable-length tokens into a single fixed vector.

--

## Pooling & Normalization

### 1. Mean Pooling (Promedio Ponderado)
Colapsa la secuencia ponderando con la máscara de atención $\mathbb{I}(t_i \neq \text{pad})$:

<div class="def-box">
  $$z = \frac{\sum_{i=1}^n m_i h_i}{\sum_{i=1}^n m_i} \in \mathbb{R}^d, \qquad m_i \in \{0, 1\}$$
</div>

*(Nota: En Sentence-BERT se demostró que mean-pooling supera al token `[CLS]` por más de 12 puntos en tareas STS).*

### 2. Normalización $L_2$
<div class="def-box">
  $$e = \mathrm{norm}(z) = \frac{z}{\|z\|_2} = \frac{z}{\sqrt{\sum_{j=1}^d z_j^2}} \in S^{d-1}$$
</div>

Todos los embeddings normalizados habitan en la **hipersfera unitaria** $S^{d-1} = \{x \in \mathbb{R}^d : \|x\|_2 = 1\}$.

Note:
[00:32 - 00:35]
Emphasize why normalization is non-negotiable. Without normalization, vector lengths corrupt distance rankings.

--

## Toy Example in $\mathbb{R}^2$

Consideremos tres vectores sobre la circunferencia unitaria $S^1 \subset \mathbb{R}^2$:

$$u = (1, 0), \qquad v = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right), \qquad w = (0, 1)$$

<div class="container">
  <div class="col" style="flex: 1;">
    <svg viewBox="0 0 260 190" style="width:100%; max-height:190px; display:block; margin:auto;">
      <!-- Grid axes -->
      <line x1="30" y1="160" x2="230" y2="160" stroke="#93A1A1" stroke-width="1.5"/>
      <line x1="40" y1="170" x2="40" y2="20" stroke="#93A1A1" stroke-width="1.5"/>
      <text x="225" y="155" font-family="Fira Code" font-size="11" fill="#586E75">x₁</text>
      <text x="45" y="25" font-family="Fira Code" font-size="11" fill="#586E75">x₂</text>

      <!-- Unit arc -->
      <path d="M 160 160 A 120 120 0 0 0 40 40" fill="none" stroke="#B58900" stroke-width="2" stroke-dasharray="3"/>

      <!-- Vector u -->
      <line x1="40" y1="160" x2="160" y2="160" stroke="#CB4B16" stroke-width="3"/>
      <circle cx="160" cy="160" r="4" fill="#CB4B16"/>
      <text x="165" y="175" font-family="Fira Code" font-weight="bold" font-size="11" fill="#CB4B16">u (1,0)</text>

      <!-- Vector v -->
      <line x1="40" y1="160" x2="125" y2="75" stroke="#2AA198" stroke-width="3"/>
      <circle cx="125" cy="75" r="4" fill="#2AA198"/>
      <text x="132" y="72" font-family="Fira Code" font-weight="bold" font-size="11" fill="#2AA198">v (0.71, 0.71)</text>

      <!-- Vector w -->
      <line x1="40" y1="160" x2="40" y2="40" stroke="#6C71C4" stroke-width="3"/>
      <circle cx="40" cy="40" r="4" fill="#6C71C4"/>
      <text x="46" y="38" font-family="Fira Code" font-weight="bold" font-size="11" fill="#6C71C4">w (0,1)</text>
    </svg>
  </div>
  <div class="col" style="flex: 1.2;">
    <div class="ex-box">
      <strong>Métricas angulares y productos punto:</strong>
      <ul>
        <li>$\cos(u, v) = \frac{1}{\sqrt{2}} \approx 0.707 \quad (\theta = 45^\circ)$</li>
        <li>$\cos(v, w) = \frac{1}{\sqrt{2}} \approx 0.707 \quad (\theta = 45^\circ)$</li>
        <li>$\cos(u, w) = 0.000 \quad (\theta = 90^\circ, \text{ortogonales})$</li>
      </ul>
    </div>
  </div>
</div>

Note:
[00:35 - 00:38]
Note on curse of dimensionality: in 1536 dimensions, the angle between two random vectors is tightly concentrated around 90 degrees ($\cos \approx 0$).

--

## High Dimensions: $S^{d-1}$ in Production

En espacios reales de embedding, la dimensionalidad es mucho mayor: $d \in [384, 3072]$.

<div class="def-box">
  <strong>El fenómeno de la cáscara esférica:</strong> En altas dimensiones, casi todo el volumen de la esfera se concentra en una delgada capa superficial externa.
</div>

<div class="highlight-box">
  <ul>
    <li>Dos vectores aleatorios independientes en $\mathbb{R}^{1536}$ tienen casi siempre un ángulo $\theta \approx 90^\circ$ ($\cos \approx 0$).</li>
    <li>Esta <strong>ortogonalidad casi segura</strong> permite que millones de conceptos distintos coexistan sin colisionar en el espacio latente.</li>
    <li>El modelo contrastivo se enfoca en agrupar pares conceptuales en conos hiperdimensionales muy estrechos.</li>
  </ul>
</div>

Note:
[00:38 - 00:40]
Explain why high dimension is a feature, not a bug: it gives capacity to separate millions of distinct thoughts.

---

## Similarity Measures in Vector Space

Sean $u, v \in \mathbb{R}^d$. Definimos tres operadores estándar:

<div class="container">
  <div class="col">
    <h4>Similitud Coseno</h4>
    $$\cos(u, v) = \frac{\langle u, v \rangle}{\|u\|_2 \, \|v\|_2}$$
    <p style="font-size:0.8em;">Invariante a la longitud de los vectores. Mide exclusivamente la alineación angular.</p>
  </div>
  <div class="col">
    <h4>Producto Punto</h4>
    $$\langle u, v \rangle = \sum_{i=1}^d u_i v_i = u^\top v$$
    <p style="font-size:0.8em;">Operación BLAS de nivel 1. Máxima eficiencia en aceleradores GPU/CPU.</p>
  </div>
  <div class="col">
    <h4>Distancia Euclidiana ($L_2$)</h4>
    $$\|u - v\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
    <p style="font-size:0.8em;">Métrica formal en espacios normados. Satisface la desigualdad del triángulo.</p>
  </div>
</div>

Note:
[00:40 - 00:42]
Ask students: "Which one should we use in our database index?" Lead them to the key identity on the next slide.

--

## The Fundamental Identity on $S^{d-1}$

<div class="def-box">
  <strong>Teorema (Equivalencia de Rango en la Hipersfera):</strong> Si los vectores están normalizados en norma $L_2$ ($\|u\|_2 = \|v\|_2 = 1$):
  $$\|u - v\|_2^2 = \|u\|_2^2 - 2\langle u, v \rangle + \|v\|_2^2 = \mathbf{2 - 2\langle u, v \rangle} = \mathbf{2 - 2\cos(u, v)}$$
</div>

<div class="highlight-box">
  <strong>Implicación computacional directa:</strong>
  $$\arg\max_v \cos(u, v) \equiv \arg\max_v \langle u, v \rangle \equiv \arg\min_v \|u - v\|_2$$
  ¡Los tres criterios generan exactamente el <strong>mismo ranking top-$k$</strong>!
</div>

- El **producto punto** $u^\top v$ omite las divisiones de las normas $\implies$ **es el más veloz**. <!-- .element: class="fragment" -->
- <span style="color:var(--accent-red); font-weight:bold;">¡Advertencia!</span> Si los vectores **no están normalizados**, los tres ordenamientos discrepan drásticamente. <!-- .element: class="fragment" -->

Note:
[00:42 - 00:44]
Walk through the proof line by line. It is foundational for index design in FAISS and pgvector.

--

## Metric Properties: Similarity vs. Distance

| Medida | Rango | ¿Es Métrica Formal? | Propiedades Matemáticas |
| :--- | :---: | :---: | :--- |
| **Similitud Coseno** | $[-1, 1]$ | **NO** | No es distancia (viola $\mathrm{dist}(x,x)=0$, no cumple desigualdad triangular) |
| **Distancia Angular** | $[0, 1]$ | **SÍ** | $d_{\text{ang}}(u, v) = \frac{\arccos(\cos(u, v))}{\pi}$. Métrica geodésica sobre la esfera. |
| **Distancia Euclidiana ($L_2$)** | $[0, \infty)$ | **SÍ** | Cumple no negatividad, identidad de indiscernibles, simetría y subaditividad. |
| **Distancia Manhattan ($L_1$)** | $[0, \infty)$ | **SÍ** | Menos sensible a dimensiones anómalas extremas. |

<div class="alert-box">
  <strong>Precaución de diseño:</strong> Los índices basados en árboles espaciales (como KD-Trees o VP-Trees) exigen estrictamente la desigualdad del triángulo para podar ramas. La similitud coseno en crudo no puede podar sin transformar a distancia angular o $L_2$.
</div>

Note:
[00:44 - 00:46]
Highlight why KD-trees fail with cosine similarity unless converted to metric distance.

---

## Contrastive Learning: The InfoNCE Loss

Los modelos de embedding se entrenan típicamente mediante **aprendizaje contrastivo** usando la función de pérdida **InfoNCE**:

<div class="def-box">
  Dada una consulta $q$, un pasaje positivo $d^+$ y $B-1$ negativos $\{d_j^-\}$:
  $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\left(\cos(q, d^+) / \tau\right)}{\exp\left(\cos(q, d^+) / \tau\right) + \sum_{j=1}^{B-1} \exp\left(\cos(q, d_j^-) / \tau\right)}$$
  donde $\tau > 0$ es la temperatura que modula la agudeza de la distribución softmax.
</div>

<svg viewBox="0 0 680 110" style="width:100%; max-height:115px; margin: 10px auto; display:block;">
  <!-- Anchor -->
  <circle cx="150" cy="55" r="26" fill="#EEE8D5" stroke="#CB4B16" stroke-width="2.5"/>
  <text x="150" y="53" text-anchor="middle" font-family="Crimson Text" font-weight="bold" font-size="13" fill="#073642">Consulta q</text>
  <text x="150" y="66" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#CB4B16">Ancla</text>

  <!-- Pull Force to Positive -->
  <path d="M 180 55 L 290 55" stroke="#859900" stroke-width="2.5"/>
  <text x="235" y="47" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="11" fill="#859900">Atracción</text>

  <!-- Positive Doc -->
  <circle cx="320" cy="55" r="24" fill="#FDF6E3" stroke="#859900" stroke-width="2.5"/>
  <text x="320" y="53" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#859900">Doc d⁺</text>
  <text x="320" y="66" text-anchor="middle" font-family="Fira Code" font-size="8" fill="#586E75">Relevante</text>

  <!-- Repel Forces to Negatives -->
  <path d="M 170 35 L 460 20" stroke="#DC322F" stroke-width="1.5" stroke-dasharray="3"/>
  <path d="M 170 65 L 460 60" stroke="#DC322F" stroke-width="1.5" stroke-dasharray="3"/>
  <path d="M 165 75 L 460 95" stroke="#DC322F" stroke-width="1.5" stroke-dasharray="3"/>
  <text x="340" y="98" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="11" fill="#DC322F">Repulsión de Negativos</text>

  <!-- Negatives -->
  <circle cx="490" cy="20" r="14" fill="#EEE8D5" stroke="#DC322F" stroke-width="1.5"/>
  <text x="490" y="24" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#DC322F">d₁⁻</text>
  <circle cx="490" cy="58" r="14" fill="#EEE8D5" stroke="#DC322F" stroke-width="1.5"/>
  <text x="490" y="62" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#DC322F">d₂⁻</text>
  <circle cx="490" cy="95" r="14" fill="#EEE8D5" stroke="#DC322F" stroke-width="1.5"/>
  <text x="490" y="99" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#DC322F">d₃⁻</text>
</svg>

Note:
[00:46 - 00:48]
Explain temperature $\tau$. A lower temperature acts like hard mining, forcing high gradient penalties on top errors.

--

## In-Batch & Hard Negatives

<div class="container">
  <div class="col">
    <h4>1. In-Batch Negatives</h4>
    <p>Todos los demás documentos dentro del mismo mini-batch actúan como ejemplos negativos automáticamente:</p>
    <ul>
      <li>Cero costo de cómputo adicional.</li>
      <li>Requiere batches grandes ($B \ge 512$) para proveer contraste suficiente.</li>
    </ul>
  </div>
  <div class="col">
    <h4>2. Hard Negatives</h4>
    <p>Ejemplos difíciles minados con BM25 que comparten léxico pero discrepan en significado:</p>
    <ul>
      <li>Obligan al modelo a discriminar sutilezas semánticas reales.</li>
      <li>Esenciales para evitar que el bi-encoder dependa de palabras clave triviales.</li>
    </ul>
  </div>
</div>

Note:
[00:48 - 00:50]
Mining hard negatives is the secret behind state-of-the-art embedding models like BGE and Cohere-v3.

--

## Matryoshka Representation Learning (MRL)

¿Podemos reducir la dimensión del vector **sin reentrenar** el modelo?

<div class="def-box">
  <strong>Pérdida Matryoshka:</strong> Se optimiza simultáneamente la pérdida contrastiva sobre múltiples prefijos anidados $d_1 < d_2 < \dots < d_m$:
  $$\mathcal{L}_{\text{MRL}} = \sum_{k=1}^m \lambda_k \, \mathcal{L}\left(e_{1:d_k}\right)$$
</div>

<div class="highlight-box">
  <strong>Truncamiento y Renormalización en Inferencia:</strong>
  Para un vector $e \in \mathbb{R}^{1536}$, si deseamos utilizar solo $d' = 256$:
  $$e' = \frac{e_{1:d'}}{\|e_{1:d'}\|_2} \in S^{d'-1}$$
</div>

- **Compresión 6x sin reentrenar:** Los primeros $d'$ coeficientes contienen la mayor densidad de señal discriminativa.
- Soportado de forma nativa en OpenAI `text-embedding-3`, Nomic Embed y BGE.

Note:
[00:50 - 00:52]
Analogy: Russian nesting dolls (Matryoshka). The first dimensions carry global semantic cluster info.

--

## Misconceptions (Part 1)

<div class="container">
  <div class="col">
    <div class="alert-box">
      <strong>1. Dimensión ≠ Atributo</strong>
      <p style="font-size:0.85em;">No busques la "coordenada 42" para saber si el texto es formal. Las representaciones son densas y distribuidas geométricamente sobre todo el vector.</p>
    </div>
  </div>
  <div class="col">
    <div class="alert-box">
      <strong>2. Puntuación ≠ Probabilidad</strong>
      <p style="font-size:0.85em;">Un coseno de 0.85 <strong>no significa</strong> 85% de probabilidad de acierto. La distribución de scores depende de la temperatura y la dimensionalidad.</p>
    </div>
  </div>
</div>

Note:
[00:52 - 00:54]
Never set arbitrary thresholds like `score > 0.8` without calibrating on actual validation data.

--

## Misconceptions (Part 2)

<div class="container">
  <div class="col">
    <div class="alert-box">
      <strong>3. Modelos Incompatibles</strong>
      <p style="font-size:0.85em;">Un score de 0.80 en BGE no equivale a un 0.80 en OpenAI. Sus geometrías espaciales y rotaciones son completamente incompatibles.</p>
    </div>
  </div>
  <div class="col">
    <div class="alert-box">
      <strong>4. Coseno Alto ≠ Verdad Factual</strong>
      <p style="font-size:0.85em;">Una alucinación falsa que comparta la estructura semántica de la consulta arrojará un coseno extremadamente elevado. Los embeddings miden afinidad, no veracidad.</p>
    </div>
  </div>
</div>

Note:
[00:54 - 00:55]
These 4 mistakes cost teams months of debugging in production systems.

---

# 3. Models, Dimensions & Quantization
<span class="badge badge-amber">Block 3 • 15 min</span>

> El equilibrio entre precisión de recuperación, capacidad de memoria y latencia de inferencia.

Note:
[00:55 - 00:58]
Now we translate theory into hardware and engineering constraints: picking models, storage footprint, and quantization.

---

## Choosing an Embedding Model

| Eje de Decisión | Pregunta Crítica | Trade-offs Principales |
| :--- | :--- | :--- |
| **Calidad de Dominio** | ¿Funciona en mi idioma y jerga técnica? | Modelos en inglés vs. multilingües (BGE-M3); términos biomédicos o legales. |
| **Costos Operativos** | ¿API administrada o auto-hospedaje en GPU? | API: pago por token, cero ops. Self-host: costo fijo de GPU, privacidad total. |
| **Soberanía del Dato** | ¿Pueden salir los textos a la nube pública? | Regulaciones GDPR, HIPAA, banca privada exigen modelos locales en VPC. |
| **Ventana Contexto** | ¿Cuántos tokens por chunk puede procesar? | De 512 tokens (BERT) hasta 8,192 o 32,768 tokens (Voyage, Jina). |
| **Dimensión $d$** | ¿Cuál es la huella de memoria resultante? | De 384 a 3072 dimensiones. Determina el costo en RAM del índice vectorial. |

<div class="highlight-box">
  <strong>Familias líderes:</strong> OpenAI (<code>text-embedding-3</code>), Cohere (<code>embed-v3</code>), Voyage AI, y open-source BAAI (<code>bge-small-en-v1.5</code>, <code>bge-m3</code>).
</div>

Note:
[00:58 - 01:02]
Never trust MTEB leaderboards blindly! MTEB tests academic tasks. Always evaluate candidate models on your domain.

--

## Dimension, Storage & Memory Scaling

En precisión completa (`float32`), cada número de punto flotante ocupa **4 bytes**.  
El espacio en memoria consumido por un único vector es:

$$\text{Memoria por vector} = 4 \times d \text{ bytes}$$

| Dimensión $d$ | Bytes / Vector | KB / Vector | Vectores por 1 GB RAM | Huella de $10^6$ Vectores |
| :---: | :---: | :---: | :---: | :---: |
| **384** (BGE-small) | 1,536 B | **1.5 KB** | ~699,000 | **1.5 GB** |
| **768** (BERT-base) | 3,072 B | **3.0 KB** | ~349,000 | **3.0 GB** |
| **1024** (BGE-large) | 4,096 B | **4.0 KB** | ~262,000 | **4.0 GB** |
| **1536** (OpenAI small) | 6,144 B | **6.0 KB** | ~174,000 | **6.0 GB** |
| **3072** (OpenAI large) | 12,288 B | **12.0 KB** | ~87,000 | **12.0 GB** |

*(Nota: Los grafos HNSW requieren un 20% a 50% de sobrecarga adicional en RAM para aristas).*

Note:
[01:02 - 01:06]
Walk through the numbers: 10 million vectors at 1536d requires over 60 GB of RAM just for raw floats!

--

## Vector Quantization

Podemos comprimir las representaciones vectoriales para reducir el consumo de RAM hasta un **97%**:

| Formato | Bytes / Dimensión | Factor Reducción | Pérdida de Recall Típica | Caso de Uso |
| :--- | :---: | :---: | :---: | :--- |
| **float32** | 4 | **1×** (Línea base) | 0% | Máxima precisión requerida |
| **float16 / bf16** | 2 | **2×** | $< 0.1\%$ | Aceleración nativa en GPU |
| **int8 (Scalar)** | 1 | **4×** | $\approx 1\% - 2\%$ | Producción estándar en vector DBs |
| **Binary (1-bit)** | $1/8$ (1 bit) | **32×** | $\approx 5\% - 10\%$ | Búsqueda masiva ultra-rápida (Hamming) |

<div class="def-box">
  <strong>Patrón Two-Stage (Coarse-to-Fine Search):</strong>
  <ol>
    <li><strong>Etapa 1 (Filtro grueso):</strong> Buscar en el índice binario o int8 a velocidad de hardware para extraer top-$k$ candidatos ($k=100$).</li>
    <li><strong>Etapa 2 (Refinamiento):</strong> Reordenar los 100 candidatos calculando el producto punto exacto en float32 sobre los vectores leídos de disco.</li>
  </ol>
</div>

Note:
[01:06 - 01:10]
Binary quantization uses bitwise XOR and POPCNT instructions: up to 40x faster retrieval than float operations!

---

# 4. Pipelines & Search
<span class="badge badge-teal">Block 4 • 25 min</span>

> Desde documentos extensos hasta consultas vectoriales aproximadas en microsegundos.

Note:
[01:10 - 01:12]
We now build the end-to-end pipeline: document ingestion, chunking trade-offs, code implementation, and indexing algorithms.

---

## The End-to-End Search Pipeline

Una falla en cualquiera de los eslabones degrada irreversiblemente la precisión final:

<svg viewBox="0 0 760 140" style="width:100%; max-height:140px; margin: 15px auto; display:block;">
  <!-- Corpus -->
  <rect x="5" y="45" width="95" height="46" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="52" y="66" text-anchor="middle" font-family="Crimson Text" font-weight="bold" font-size="12" fill="#073642">Documentos</text>
  <text x="52" y="80" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#657B83">PDF / HTML</text>
  <path d="M 100 68 L 125 68" stroke="#CB4B16" stroke-width="2"/>

  <!-- Chunking -->
  <rect x="130" y="45" width="105" height="46" rx="8" fill="#FDF6E3" stroke="#CB4B16" stroke-width="2"/>
  <text x="182" y="66" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="11" fill="#CB4B16">Chunking</text>
  <text x="182" y="80" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#657B83">400 tokens</text>
  <path d="M 235 68 L 260 68" stroke="#CB4B16" stroke-width="2"/>

  <!-- Embedding Model -->
  <rect x="265" y="45" width="115" height="46" rx="8" fill="#FDF6E3" stroke="#2AA198" stroke-width="2"/>
  <text x="322" y="66" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="11" fill="#2AA198">Embedding</text>
  <text x="322" y="80" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#657B83">f_θ(chunk)</text>
  <path d="M 380 68 L 405 68" stroke="#2AA198" stroke-width="2"/>

  <!-- Vector DB -->
  <rect x="410" y="45" width="105" height="46" rx="8" fill="#FDF6E3" stroke="#6C71C4" stroke-width="2"/>
  <text x="462" y="66" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="11" fill="#6C71C4">Vector DB</text>
  <text x="462" y="80" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#657B83">HNSW / IVF</text>
  <path d="M 515 68 L 540 68" stroke="#6C71C4" stroke-width="2"/>

  <!-- ANN Search -->
  <rect x="545" y="45" width="100" height="46" rx="8" fill="#FDF6E3" stroke="#268BD2" stroke-width="2"/>
  <text x="595" y="66" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="11" fill="#268BD2">Búsqueda</text>
  <text x="595" y="80" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#657B83">Top-k scores</text>
  <path d="M 645 68 L 670 68" stroke="#268BD2" stroke-width="2"/>

  <!-- LLM / RAG -->
  <rect x="675" y="45" width="80" height="46" rx="8" fill="#EEE8D5" stroke="#859900" stroke-width="2.5"/>
  <text x="715" y="66" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#859900">LLM</text>
  <text x="715" y="80" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#657B83">Generación</text>

  <!-- Query Flow from top -->
  <path d="M 322 10 L 322 42" stroke="#CB4B16" stroke-width="2" stroke-dasharray="3"/>
  <text x="322" y="8" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="10" fill="#CB4B16">Consulta q en vivo</text>
</svg>

<div class="alert-box">
  <strong>Regla de oro:</strong> Una mala partición en chunks o un modelo desalineado arruina el contexto del LLM antes de que este genere una sola palabra.
</div>

Note:
[01:12 - 01:15]
Emphasize the cascade failure: if the chunk doesn't contain the answer, no LLM can generate the right factual response.

--

## Chunking Strategies

| Estrategia | Calidad Semántica | Costo Cómputo | Cuándo Utilizarla |
| :--- | :---: | :---: | :--- |
| **Tamaño Fijo (+ Solapamiento)** | Media | Mínimo | Prototipado rápido, texto continuo sin estructura (transcripciones). |
| **Recursiva Jerárquica** | Buena | Bajo | **Opción predeterminada**. Divide por párrafos `\n\n`, luego oraciones `.` |
| **Basada en Estructura** | Excelente | Bajo | Documentación técnica, código, artículos Markdown, tablas HTML. |
| **Chunking Semántico** | Óptima | Alto | Textos críticos donde la frontera entre ideas evalúa distancias vectoriales. |

<div class="highlight-box">
  <strong>Configuración base recomendada para iniciar:</strong>
  <ul>
    <li>Tamaño de chunk: <strong>~350 a 500 tokens</strong></li>
    <li>Solapamiento (Overlap): <strong>~50 tokens</strong> (10% a 15% para no cortar oraciones a la mitad)</li>
  </ul>
</div>

Note:
[01:15 - 01:18]
Explain why overlap is necessary: it prevents named entities and concepts from being sliced right on the boundary between chunks.

--

## The Chunk Size Trade-off

<div class="container">
  <div class="col">
    <h4><i class="fa-solid fa-arrows-to-circle"></i> Chunks Grandes (&gt; 1000 tokens)</h4>
    <ul>
      <li>El vector se convierte en el <strong>promedio diluido</strong> de muchos temas distintos.</li>
      <li>Pierde resolución para preguntas específicas y puntuales.</li>
    </ul>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-compress"></i> Chunks Pequeños (&lt; 100 tokens)</h4>
    <ul>
      <li>Pierde el contexto circundante (antecedentes, pronombres, sujeto).</li>
      <li>Multiplica exponencialmente la cantidad de vectores en el índice.</li>
    </ul>
  </div>
</div>

<div class="alert-box">
  <strong>Límite de truncamiento silencioso:</strong> Si un chunk excede el límite del tokenizador (e.g. 512 tokens), muchas librerías descartan el texto sobrante <em>sin emitir advertencias</em>.
</div>

Note:
[01:18 - 01:21]
Stress the silent truncation bug! It is one of the most common bugs in production retrieval systems.

---

## Minimal Semantic Search: Indexing Corpus

Carga del modelo y codificación normalizada del corpus en lote:

```python [1-5|7-12|14-15]
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Cargar el modelo pre-entrenado
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# 2. Corpus de documentos a indexar
docs = [
    "RAG grounds generative LLMs with factual external documents.",
    "Cosine similarity measures the angle between normalized vectors.",
    "Bikes require regular maintenance for chains and tire pressure."
]

# 3. Embeddings normalizados directamente en S^(d-1)
D = model.encode(docs, normalize_embeddings=True)  # Matriz (N, d)
```

Note:
[01:21 - 01:23]
Explain that `normalize_embeddings=True` projects all rows directly onto the unit hypersphere.

--

## Minimal Semantic Search: Vector Query

Búsqueda por producto punto y extracción de los top-$k$ resultados:

```python [1-6|8-10]
def search(query: str, k: int = 2):
    # Codificar y normalizar la consulta
    q = model.encode([query], normalize_embeddings=True)[0]  # (d,)
    scores = D @ q                                           # Producto punto = Coseno
    top_indices = np.argsort(-scores)[:k]
    return [(docs[i], float(scores[i])) for i in top_indices]

# Probar búsqueda semántica
resultados = search("How do you evaluate distance between embeddings?")
print(resultados[0])
```

- La multiplicación matriz-vector `D @ q` evalúa todas las similitudes en paralelo. <!-- .element: class="fragment" -->
- El ordenamiento de NumPy `argsort` extrae los mejores índices en microsegundos. <!-- .element: class="fragment" -->

Note:
[01:23 - 01:25]
Walk students through the matrix multiplication `D @ q`. Because rows of $D$ and vector $q$ have unit norm, the simple matrix-vector product computes all cosine similarities in parallel.

--

## Scaling Up: ANN Indexes

Para $N = 10^7$ vectores, la búsqueda exacta por fuerza bruta toma $O(N \cdot d)$ operaciones por consulta (inviable a escala en tiempo real).

<div class="container">
  <div class="col">
    <h4>HNSW</h4>
    <p>Grafos multicapa de mundo pequeño. Navegación en $O(\log N)$. Alto recall (&gt;98%), pero consume más memoria RAM.</p>
  </div>
  <div class="col">
    <h4>IVF (Inverted File)</h4>
    <p>Clustering con K-Means. Particiona en celdas Voronoi; solo busca en las celdas más cercanas (`nprobe`). Menos RAM.</p>
  </div>
  <div class="col">
    <h4>PQ (Product Quant)</h4>
    <p>Descompone vectores en sub-vectores y los cuantiza con codebooks. Compresión agresiva para índices masivos.</p>
  </div>
</div>

Note:
[01:25 - 01:27]
Briefly explain HNSW: top layers have long-range edges for fast routing, bottom layers have dense local edges for precision.

--

## Vector Database Ecosystem

<div class="container">
  <div class="col">
    <h4>Librerías Embebidas</h4>
    <ul>
      <li><strong>FAISS (Meta):</strong> Estándar de investigación, optimizado para GPU.</li>
      <li><strong>ScaNN (Google):</strong> Alta eficiencia con cuantización anisotrópica.</li>
      <li><strong>USearch:</strong> Motor en C++ liviano para aplicaciones embebidas.</li>
    </ul>
  </div>
  <div class="col">
    <h4>Bases de Datos & Extensiones</h4>
    <ul>
      <li><strong>Dedicadas:</strong> Qdrant, Pinecone, Milvus, Weaviate, Chroma.</li>
      <li><strong>Relacionales:</strong> <code>pgvector</code> en PostgreSQL (ideal para unificar metadata y vectores).</li>
    </ul>
  </div>
</div>

<div class="highlight-box">
  <strong>Regla práctica:</strong> Comienza con <code>pgvector</code> si ya utilizas PostgreSQL; migra a Qdrant/Pinecone si superas 5M de vectores con SLA estricto de latencia.
</div>

Note:
[01:27 - 01:30]
Recommend pgvector for starting out: having metadata and embeddings in the same database avoids data sync headaches.

--

## Advanced Retrieval Architectures

<div class="container">
  <div class="col">
    <h4>1. Búsqueda Híbrida (Dense + Sparse)</h4>
    <p>Combina vectores densos con <strong>BM25</strong> léxico mediante <em>Reciprocal Rank Fusion (RRF)</em>:</p>
    $$RRF(d) = \sum_{m} \frac{1}{60 + \text{rank}_m(d)}$$
    <p style="font-size:0.8em;">Garantiza recuperar palabras clave exactas (IDs, nombres propios) y sinónimos conceptuales.</p>
  </div>
  <div class="col">
    <h4>2. Reordenamiento con Cross-Encoder</h4>
    <p>El bi-encoder extrae top-50 candidatos. Luego, un <strong>Cross-Encoder</strong> evalúa el par completo <code>[Query, Doc]</code> con atención cruzada:</p>
    $$\text{Score} = \mathrm{CrossEncoder}(q, d_i)$$
    <p style="font-size:0.8em;">Recupera interacciones finas entre términos que se pierden en la compresión del bi-encoder.</p>
  </div>
</div>

Note:
[01:30 - 01:32]
Bi-encoder + Cross-encoder is the industry-standard architecture at Google, Netflix, and Amazon.

--

## Query vs. Document Asymmetry

- **Consulta:** Corta, interrogativa, incompleta (e.g. *"síntomas gripe"*).
- **Documento:** Largo, expositivo, estructurado (e.g. *"El virus de la influenza provoca..."*).

<div class="def-box">
  <strong>Prefijos Asimétricos:</strong> Modelos modernos requieren prefijos explícitos para condicionar el vector al rol adecuado:
  <ul>
    <li><strong>BGE:</strong> Requiere <code>"Represent this sentence for searching relevant passages: "</code> en queries.</li>
    <li><strong>Cohere / E5:</strong> Parámetros explícitos <code>input_type="search_query"</code> vs. <code>input_type="search_document"</code>.</li>
  </ul>
</div>

<div class="alert-box">
  <strong>¡Regla inviolable!</strong> Ambos deben procesarse con el <strong>mismo modelo</strong>. Si cambias de modelo en producción, <em>debes re-indexar la totalidad de tus documentos</em>.
</div>

Note:
[01:32 - 01:34]
Explain asymmetric modeling: queries and documents occupy slightly different sub-spaces.

---

## Six Silent Production Pitfalls

<div class="alert-box">
  <ol>
    <li><strong>Modelos distintos en query e index:</strong> Ruido aleatorio por geometrías incompatibles.</li>
    <li><strong>Vectores obsoletos:</strong> Modificar documentos sin re-calcular sus chunks vectoriales.</li>
    <li><strong>Truncamiento silencioso:</strong> Chunks que exceden 512 tokens pierden contenido sin avisar.</li>
    <li><strong>Métrica incorrecta:</strong> Usar distancia $L_2$ o producto punto sobre vectores no normalizados.</li>
    <li><strong>Indexar documentos completos:</strong> Generar un solo vector para un PDF de 20 páginas.</li>
    <li><strong>Elegir modelos solo por MTEB:</strong> Confiar ciegamente en tablas académicas sin validar en el propio corpus.</li>
  </ol>
</div>

Note:
[01:34 - 01:35]
Every student building a production system will encounter at least one of these traps.

---

# 5. Metrics & Evaluation
<span class="badge badge-orange">Block 5 • 20 min</span>

> "If you cannot measure it, you cannot improve it."  
> Validación cuantitativa de sistemas de recuperación de información.

Note:
[01:35 - 01:38]
Transition to evaluation. We need Information Retrieval (IR) metrics to score our pipeline.

---

## Two Kinds of "Metrics"

<div class="container">
  <div class="col">
    <h4>1. Métrica Geométrica (Vectorial)</h4>
    <p>Calcula la relación espacial entre <strong>dos vectores individuales</strong>:</p>
    <ul>
      <li>Similitud Coseno</li>
      <li>Producto Punto</li>
      <li>Distancia Euclidiana ($L_2$)</li>
    </ul>
    <p style="font-size:0.8em; color:var(--accent-teal);">Se usa para calcular scores y ordenar candidatos en tiempo de ejecución.</p>
  </div>
  <div class="col">
    <h4>2. Métrica de Evaluación (Sistema)</h4>
    <p>Evalúa el rendimiento global del <strong>sistema completo</strong> sobre un conjunto de prueba:</p>
    <ul>
      <li>Recall@k / Hit@k</li>
      <li>Mean Reciprocal Rank (MRR)</li>
      <li>nDCG@k</li>
    </ul>
    <p style="font-size:0.8em; color:var(--accent-orange);">Requiere un Test Set con respuestas correctas conocidas (Ground Truth).</p>
  </div>
</div>

Note:
[01:38 - 01:40]
Make the distinction crystal clear: cosine is a ranking score, while Recall@k is an evaluation KPI.

--

## Building a Ground-Truth Test Set

Para evaluar un pipeline necesitas un conjunto de evaluación anotado:

<div class="highlight-box">
  <strong>Composición de un Test Set representativo:</strong>
  <ul>
    <li>Entre <strong>30 y 100 consultas realistas</strong> formuladas por usuarios finales.</li>
    <li>Para cada consulta $q$, una lista de IDs de los chunks genuinamente relevantes: $\mathcal{R}_q = \{\text{id}_1, \text{id}_2, \dots\}$.</li>
  </ul>
</div>

<div class="alert-box">
  <strong>¡Principio de partición estricta!</strong> Mantén el test set aislado. Jamás lo utilices para calibrar tamaños de chunk ni para fine-tuning.
</div>

Note:
[01:40 - 01:42]
Emphasize test set hygiene. Without a clean, held-out test set, all tuning is just overfitting.

--

## Sources for Test Set Queries

<div class="container">
  <div class="col">
    <h4>1. Logs Reales de Usuarios</h4>
    <p>Consultas y clics históricos en buscadores o sistemas de soporte previos. Máxima fidelidad con la intención real.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4>2. Juicio de Expertos</h4>
    <p>Anotación manual de pasajes pertinentes por especialistas de dominio. El estándar de oro en calidad.</p>
  </div>
  <div class="col">
    <h4>3. Generación Sintética con LLMs</h4>
    <p>Instruir a un LLM para formular preguntas a partir de chunks individuales:</p>
    <ul>
      <li>Rápido y económico para prototipos.</li>
      <li><strong>Obligatorio:</strong> Revisión humana aleatoria para depurar preguntas ambiguas.</li>
    </ul>
  </div>
</div>

Note:
[01:42 - 01:44]
Explain how to use LLMs to bootstrap an initial test set, followed by human curation.

---

## Recall@k & Hit@k

Sea $Q$ el conjunto de consultas, $\mathcal{R}_q$ los chunks relevantes y $\mathcal{T}_q^{(k)}$ la lista de los top-$k$ devueltos:

<div class="def-box">
  <h4>Recall@k</h4>
  $$\mathrm{Recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathcal{R}_q \cap \mathcal{T}_q^{(k)}|}{|\mathcal{R}_q|}$$
  <p style="font-size:0.82em;">Mide la proporción promedio de documentos relevantes capturados en los primeros $k$ lugares.</p>
</div>

<div class="def-box">
  <h4>Hit@k (Success@k)</h4>
  $$\mathrm{Hit@}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(|\mathcal{R}_q \cap \mathcal{T}_q^{(k)}| > 0\right)$$
  <p style="font-size:0.82em;">Vale 1 si <strong>al menos un</strong> documento relevante entra en el top-$k$; 0 en caso contrario.</p>
</div>

Note:
[01:44 - 01:46]
Explain the intuition: In RAG, if $k=5$, Recall@5 tells us if the context fed to the LLM contains the facts needed.

--

## Mean Reciprocal Rank (MRR)

En tareas donde existe una respuesta principal o el primer resultado es el más crítico:

<div class="def-box">
  $$\mathrm{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\mathrm{rank}_q}$$
  donde $\mathrm{rank}_q$ es la posición del <strong>primer documento relevante</strong> devuelto. Si ninguno entra en top-$k \implies 0$.
</div>

### Escala de penalización hiperbólica:
- En posición 1: aporte = $1/1 = \mathbf{1.00}$
- En posición 2: aporte = $1/2 = \mathbf{0.50}$
- En posición 5: aporte = $1/5 = \mathbf{0.20}$

Note:
[01:46 - 01:48]
MRR is the standard metric for question answering and customer support FAQ search.

--

## nDCG@k: Graded Relevance

Cuando los documentos tienen distintos niveles de relevancia (0 = irrelevante, 1 = parcial, 2 = relevante, 3 = perfecto):

<div class="def-box">
  <h4>DCG@k & nDCG@k</h4>
  $$\mathrm{DCG@}k = \sum_{i=1}^k \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i + 1)}, \qquad \mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k} \in [0, 1]$$
  donde $\mathrm{IDCG@}k$ es el DCG obtenido con el ordenamiento ideal perfecto.
</div>

- Recompensa ubicar los documentos más valiosos en las primeras posiciones.
- El factor $\log_2(i+1)$ descuenta la relevancia conforme baja en la lista.

Note:
[01:48 - 01:50]
Explain the logarithmic discount: users care far more about the difference between rank 1 and 2 than between 11 and 12.

--

## Implementing Recall@k & MRR in Python

Funciones puras en NumPy para evaluar recuperación:

```python [1-5|7-12]
import numpy as np

def recall_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """Calcula la fracción de documentos relevantes en el top-k."""
    top_k_set = set(ranked_ids[:k])
    return len(top_k_set & relevant_ids) / len(relevant_ids)

def mrr(ranked_ids: list, relevant_ids: set, k: int = 50) -> float:
    """Calcula el Reciprocal Rank del primer resultado relevante."""
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0
```

Note:
[01:50 - 01:52]
Pure functions without external dependencies: fast and vectorized.

--

## Implementing nDCG@k in Python

```python [1-6|8-11]
def ndcg_at_k(ranked_ids: list, rel_dict: dict, k: int) -> float:
    """Calcula nDCG@k para puntuaciones de relevancia graduada."""
    gains = [2**rel_dict.get(doc_id, 0) - 1 for doc_id in ranked_ids[:k]]
    discounts = [np.log2(i + 1) for i in range(1, len(gains) + 1)]
    dcg = sum(g / d for g, d in zip(gains, discounts))
    
    # IDCG: ganancia ideal ordenada de forma descendente
    ideal_gains = sorted([2**r - 1 for r in rel_dict.values()], reverse=True)[:k]
    idcg = sum(g / d for g, d in zip(ideal_gains, discounts[:len(ideal_gains)]))
    return dcg / idcg if idcg > 0 else 0.0
```

- Soporta diccionarios de relevancia graduada `{doc_id: rating}`.
- Maneja casos de borde ($idcg = 0$) retornando 0.0 de forma segura.

Note:
[01:52 - 01:55]
Students will use these exact functions during the hands-on lab.

---

# 6. Costs & Cloud Economics
<span class="badge badge-amber">Block 6 • 15 min</span>

> Dimensión de infraestructura vectorial y predicción del costo total de propiedad (TCO).

Note:
[01:55 - 01:57]
Now we address the business side: estimating index cost, monthly search API expense, and RAM hardware sizing.

---

## Architectural Cost Drivers

El costo total de propiedad (TCO) se descompone en cinco partidas:

| Componente | Qué lo Determina | Frecuencia |
| :--- | :--- | :--- |
| **Indexación** | Tokens totales $\times$ Precio por millón del modelo | Único al crear el índice (+ re-indexaciones) |
| **Consultas (Queries)** | Cantidad de consultas $\times$ Tokens por query $\times$ Tarifa | Recurrente mensual |
| **Memoria RAM** | $N$ vectores $\times d$ dimensiones $\times$ bytes por float | Fijo por capacidad de servidor |
| **Hosting Vector DB** | Tipo de réplicas, CPU, redundancia multi-AZ | Recurrente mensual |
| **Auto-hospedaje** | Horas de GPU (e.g. A10G/T4) $+$ horas de DevOps | Costo de infraestructura fija |

Note:
[01:57 - 02:00]
Emphasize re-indexing: whenever you upgrade the embedding model, you must pay the indexing cost all over again.

--

## The Core Formulas

Sean $N$ chunks, $\bar{T}$ tokens/chunk, $Q$ queries/mes, $\bar{T}_q$ tokens/query y $p$ precio por millón de tokens:

<div class="def-box">
  <h4>1. Costo de Indexación (One-off)</h4>
  $$C_{\text{index}} = \left(\frac{N \times \bar{T}}{10^6}\right) \times p \qquad (\text{USD})$$
</div>

<div class="def-box">
  <h4>2. Costo de Consultas (Mensual)</h4>
  $$C_{\text{query/mes}} = \left(\frac{Q \times \bar{T}_q}{10^6}\right) \times p \qquad (\text{USD/mes})$$
</div>

<div class="def-box">
  <h4>3. Consumo de Memoria RAM</h4>
  $$\text{RAM} \approx N \times d \times b \times 1.3 \quad \text{bytes} \quad (b=4\text{ en float32}, b=1\text{ en int8})$$
</div>

Note:
[02:00 - 02:02]
Walk through the units: checking that dimensional analysis yields dollars and bytes correctly.

--

## Worked Example: Indexing Costs

Base de conocimiento empresarial con:
- $N = 10,000,000$ chunks ($10^7$)
- Longitud promedio por chunk $\bar{T} = 200$ tokens
- Modelo: OpenAI `text-embedding-3-small` ($p = \$0.02$ por millón de tokens)

<div class="ex-box">
  <h4>Cálculo del Costo de Indexación:</h4>
  $$C_{\text{index}} = \frac{10^7 \times 200}{10^6} \times \$0.02 = 2,000 \times \$0.02 = \mathbf{\$40.00\text{ USD}}$$
</div>

- ¡Indexar 2,000 millones de tokens cuesta únicamente **$40 USD**! <!-- .element: class="fragment" -->
- El gasto en tokens de embedding suele ser la menor de las preocupaciones presupuestarias. <!-- .element: class="fragment" -->

Note:
[02:02 - 02:04]
Point out that token cost is rarely the issue; RAM and hosting is the true cost center.

--

## Worked Example: RAM Footprint

Dimensión completa $d = 1536$ con $N = 10^7$ vectores:

<div class="ex-box">
  <h4>1. Almacenamiento en Precisión Completa (float32):</h4>
  $$\text{Storage} = 10^7 \times 1536 \times 4\text{ bytes} \approx \mathbf{61.44\text{ GB RAM}}$$
</div>

<div class="ex-box">
  <h4>2. Con Matryoshka ($d=512$) + Cuantización int8:</h4>
  $$\text{Storage} = 10^7 \times 512 \times 1\text{ byte} \approx \mathbf{5.12\text{ GB RAM}}$$
</div>

- **Reducción del 91.7%** en la huella de memoria en servidor. <!-- .element: class="fragment" -->
- Permite alojar el índice completo en una sola instancia pequeña sin sharding. <!-- .element: class="fragment" -->

Note:
[02:04 - 02:06]
A 61 GB index requires large cluster instances; 5 GB fits on a modest single machine.

--

## Strategic Cost Optimization Levers

<div class="container">
  <div class="col">
    <h4>1. Reducción y Cuantización</h4>
    <p>Trunca con Matryoshka a $d=512$ y aplica int8. Ahorra 90% de RAM con $<1.5\%$ pérdida de recall.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4>2. Limpieza de Chunks</h4>
    <p>Elimina avisos legales, encabezados repetitivos y cookies antes de indexar. Menos chunks $=$ menos memoria.</p>
  </div>
  <div class="col">
    <h4>3. Caché de Embeddings</h4>
    <p>Guarda en Redis los vectores con clave <code>hash(texto + modelo)</code>. Jamás re-pagues por embedder el mismo texto.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4>4. Indexación Incremental</h4>
    <p>Calcula hashes por documento. Si un artículo cambia, re-indexa únicamente sus chunks afectados.</p>
  </div>
</div>

Note:
[02:06 - 02:08]
Always hash content with SHA-256! In web crawling, 20-30% of chunks are identical boilerplate.

--

## Self-Host vs. Managed API

<div class="container">
  <div class="col">
    <h4><i class="fa-solid fa-cloud"></i> APIs Administradas</h4>
    <ul>
      <li><strong>Ventajas:</strong> Cero mantenimiento de infraestructura, escalabilidad elástica instantánea.</li>
      <li><strong>Desventajas:</strong> Privacidad del dato (viaja a terceros), límites de tasa (rate limits).</li>
      <li><strong>Recomendado:</strong> $N < 50\text{M}$ chunks, tráfico variable, equipos ágiles sin DevOps.</li>
    </ul>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-server"></i> Auto-Hospedaje (Self-Host)</h4>
    <ul>
      <li><strong>Ventajas:</strong> Privacidad absoluta en VPC interna, costo marginal cero por query a plena capacidad.</li>
      <li><strong>Desventajas:</strong> Costo fijo de GPU ($>\$700$/mes por instancia A10G) $+$ mantenimiento.</li>
      <li><strong>Recomendado:</strong> Tráfico masivo diario, datos médicos/financieros regulados.</li>
    </ul>
  </div>
</div>

Note:
[02:08 - 02:09]
Stress the hidden cost of self-hosting: maintaining a Kubernetes cluster with GPU drivers is not free in engineering hours.

--

## The Economic Decision Rule

<div class="def-box">
  $$\text{Regla de decisión:} \quad \text{Si } C_{\text{API}}(\text{mes}) > \text{Costo GPU}(\text{mes}) + \text{Salario DevOps / Soporte} \implies \text{Auto-hospedar}$$
</div>

<div class="highlight-box">
  <ul>
    <li>Para menos de <strong>1 millón de consultas al mes</strong>, una API administrada casi siempre resulta más económica.</li>
    <li>Para más de <strong>50 millones de consultas al mes</strong> con GPUs altamente aprovechadas, el auto-hospedaje es significativamente más rentable.</li>
  </ul>
</div>

Note:
[02:09 - 02:10]
Review the economic threshold. Help students calculate their break-even point.

---

# 7. Wrap-up & Key Principles
<span class="badge badge-teal">Block 7 • 5 min</span>

> Los 5 principios de diseño en producción para recuperación vectorial.

Note:
[02:10 - 02:15]
Synthesize the key theoretical and practical lessons before sending students to the lab.

---

## The 5 Golden Takeaways (1 to 3)

<div class="highlight-box">
  <ol>
    <li><strong>Geometría en la Hipersfera $S^{d-1}$</strong><br>
    Al normalizar en norma $L_2$, similitud coseno, producto punto y distancia Euclidiana dan el <em>mismo ranking</em>. Usa producto punto por velocidad.</li>
    <br>
    <li><strong>El Chunking supera al Modelo</strong><br>
    Una partición limpia (recursiva, ~400 tokens, 10% overlap) tiene mayor impacto en Recall que saltar de un modelo de 768d a uno de 3072d.</li>
    <br>
    <li><strong>La Dimensión se Puede Optimizar</strong><br>
    Con Matryoshka (MRL) y cuantización int8, puedes recortar hasta el 90% de la RAM del índice sin reentrenar y con pérdida mínima.</li>
  </ol>
</div>

Note:
[02:15 - 02:17]
Review principles 1-3.

--

## The 5 Golden Takeaways (4 & 5)

<div class="highlight-box">
  <ol start="4">
    <li><strong>Mide en tu Propio Test Set</strong><br>
    Ignora las tablas generales de MTEB. Construye un test set de 30 a 50 pares (consulta, chunks relevantes) de tu dominio y evalúa Recall@k, MRR y nDCG.</li>
    <br>
    <li><strong>Calcula el Costo Antes de Indexar</strong><br>
    El costo en tokens de indexación suele ser bajo; el costo real reside en la memoria RAM del índice vectorial y los servidores de búsqueda.</li>
  </ol>
</div>

Note:
[02:17 - 02:20]
Pause for questions before explaining the hands-on lab instructions.

---

# Lab: Your Own Dataset
<span class="badge badge-orange">Hands-on Lab • 60 min</span>

> Construye, indexa, busca, evalúa y presupuesta un motor semántico completo sobre datos reales.

Note:
[02:20 - 02:25]
Explain the lab mechanics. Students can use their own CSV or the provided fallback dataset.

---

## Lab Plan & Deliverable

| Paso | Actividad | Tiempo | Objetivo Práctico |
| :---: | :--- | :---: | :--- |
| **0** | **Setup & Datos** | 10 min | Instalar dependencias, cargar CSV y aislar columna de texto. |
| **1** | **Tokens & Costos** | 10 min | Perfilado con `tiktoken` y proyección de costo de tokens y RAM. |
| **2** | **Embedding & Search** | 15 min | Codificar con `bge-small-en-v1.5`, indexar en NumPy y buscar. |
| **3** | **Test Set & Métricas** | 15 min | Crear consultas con ground truth y calcular Recall@5 y MRR. |
| **4** | **Experimentos** | 10 min | Truncamiento MRL o int8, y redactar el memo de 5 líneas. |

<div class="highlight-box">
  <strong>Entregable (Memo de 5 líneas):</strong> Reporte cuantitativo con tamaño del corpus ($N$, tokens), configuración, métricas (Recall@5, MRR), presupuesto y recomendación técnica.
</div>

Note:
[02:25 - 02:30]
Reiterate the deliverable. It should be concise and ready to show to an engineering director.

--

## Step 0: Virtual Environment & Packages

Instala las dependencias en tu entorno virtual:

```bash
# Crear y activar entorno virtual
python -m venv .venv && source .venv/bin/activate

# Instalar librerías de embeddings y evaluación
pip install pandas numpy tiktoken sentence-transformers scikit-learn
```

- `sentence-transformers`: Modelos preentrenados de HuggingFace. <!-- .element: class="fragment" -->
- `tiktoken`: Tokenizador de alta velocidad en Rust para conteo exacto de tokens. <!-- .element: class="fragment" -->

Note:
[02:30 - 02:32]
Ensure everyone has packages installed.

--

## Step 0: Data Ingestion in Pandas

Carga de datos y preparación de la lista de textos:

```python [1-5|7-9]
import pandas as pd

# Cargar dataset (debe contener una columna de texto)
df = pd.read_csv("my_data.csv")
text_col = "text"  # ajusta al nombre de tu columna

texts = df[text_col].dropna().astype(str).tolist()
print(f"Total documentos cargados: {len(texts):,}")
print(f"Muestra inicial: {texts[0][:150]}...")
```

Note:
[02:32 - 02:35]
Check that students have a valid text column with non-empty strings.

--

## Step 1: Token Counting with TikToken

Perfilado de tokens antes de calcular embeddings:

```python [1-5|7-11]
import tiktoken
import numpy as np

# Tokenizador de referencia cl100k_base
tokenizer = tiktoken.get_encoding("cl100k_base")
token_counts = [len(tokenizer.encode(t)) for t in texts]

print(f"Tokens totales: {sum(token_counts):,}")
print(f"Media tokens/doc: {np.mean(token_counts):.1f}")
print(f"Percentil 95: {np.percentile(token_counts, 95):.1f}")
```

Note:
[02:35 - 02:37]
Emphasize checking the 95th percentile against the model's token limit.

--

## Step 1: Sizing & Cost Estimation Function

```python [1-7|9-13]
def estimate_infra(n_docs, tokens, price_per_M=0.02, dim=384):
    return {
        "indexing_cost_usd": (tokens / 1e6) * price_per_M,
        "ram_float32_mb": (n_docs * dim * 4) / (1024**2),
        "ram_int8_mb": (n_docs * dim * 1) / (1024**2),
    }

presupuesto = estimate_infra(len(texts), sum(token_counts), dim=384)
for k, v in presupuesto.items():
    print(f"  {k}: {v:,.4f}")
```

<div class="ex-box">
  <strong>Tarea:</strong> Escala el cálculo al tamaño total previsto para tu sistema en producción (e.g. 1 millón de registros).
</div>

Note:
[02:37 - 02:40]
Help students project numbers to production scale.

--

## Step 2: Local Batch Encoding (BGE)

Codificación en lote con normalización unitaria:

```python [1-5|7-10]
import time
from sentence_transformers import SentenceTransformer

# Modelo compacto de 384 dimensiones
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

t0 = time.time()
E = model.encode(texts, batch_size=64, normalize_embeddings=True)
elapsed = time.time() - t0
print(f"Matriz E calculada: {E.shape} en {elapsed:.2f}s ({len(texts)/elapsed:.1f} doc/s)")
```

Note:
[02:40 - 02:44]
`normalize_embeddings=True` guarantees vectors are on $S^{d-1}$.

--

## Step 2: Vector Search & Query Testing

```python [1-6|8-11]
def semantic_search(query: str, k: int = 5):
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = E @ q_vec  # Producto punto sobre vectores unitarios
    top_indices = np.argsort(-scores)[:k]
    return [(idx, texts[idx], float(scores[idx])) for idx in top_indices]

# Probar consulta interactiva
resultados = semantic_search("¿Cuál es el procedimiento de soporte?")
for rank, (idx, txt, score) in enumerate(resultados[:3], 1):
    print(f"[{rank}] Score: {score:.4f} | {txt[:90]}...")
```

Note:
[02:44 - 02:48]
Have each team run 3 exploratory queries. Ask: "Do the results make conceptual sense?"

--

## Step 3: Ground-Truth Test Set

Diseña de 15 a 30 pares `(consulta, IDs relevantes)`:

```python [1-6]
test_set = [
    ("cómo solicitar reembolso o cancelación", [12, 45]),
    ("requisitos técnicos de hardware", [8]),
    ("política de privacidad y protección de datos", [33, 34, 102]),
]
```

- Incluye preguntas con **sinónimos** que no compartan palabras con el documento. <!-- .element: class="fragment" -->
- Incluye preguntas con terminología técnica y preguntas coloquiales. <!-- .element: class="fragment" -->

Note:
[02:48 - 02:51]
Encourage students to create realistic queries.

--

## Step 3: Calculating Recall@k & MRR

```python [1-8|10-14]
def evaluate_retrieval(search_fn, tests, k: int = 5):
    recalls, rrs = [], []
    for query, relevant_indices in tests:
        rel_set = set(relevant_indices)
        top_k = [item[0] for item in search_fn(query, k=k)]
        recalls.append(len(set(top_k) & rel_set) / len(rel_set))
        rr = next((1.0 / r for r, idx in enumerate(top_k, 1) if idx in rel_set), 0.0)
        rrs.append(rr)
    return {f"Recall@{k}": float(np.mean(recalls)), "MRR": float(np.mean(rrs))}

metricas_base = evaluate_retrieval(semantic_search, test_set, k=5)
print("Evaluación inicial:", metricas_base)
```

Note:
[02:51 - 02:55]
Ensure students record their baseline metrics before running optimization experiments.

--

## Step 4: Experiments A & B (Compression)

<div class="container">
  <div class="col">
    <h4>Experimento A: Matryoshka</h4>
    <p>Trunca a $d=128$ y re-normaliza:</p>
    <pre><code class="language-python">E_sub = E[:, :128]
E_trunc = E_sub / np.linalg.norm(
    E_sub, axis=1, keepdims=True
)
# Evalúa variación en Recall@5</code></pre>
  </div>
  <div class="col">
    <h4>Experimento B: Cuantización int8</h4>
    <p>Escala a enteros con signo:</p>
    <pre><code class="language-python">scale = 127.0 / np.max(np.abs(E))
E_int8 = np.round(E * scale).astype(
    np.int8
)
# Compara memoria vs. pérdida</code></pre>
  </div>
</div>

Note:
[02:55 - 02:57]
Students test compression trade-offs.

--

## Step 4: Experiments C & D (Architecture)

<div class="container">
  <div class="col">
    <h4>Experimento C: Chunk Size</h4>
    <p>Re-particiona el texto con distintos tamaños (150 vs 500 tokens) y compara si los chunks breves mejoran la precisión de respuestas específicas.</p>
  </div>
  <div class="col">
    <h4>Experimento D: Búsqueda Híbrida</h4>
    <p>Combina los puntajes de BM25 léxico con los scores densos usando Reciprocal Rank Fusion (RRF) y compara el Recall@5 obtenido.</p>
  </div>
</div>

Note:
[02:57 - 03:00]
Circulate around the room assisting teams with their comparisons.

--

## Deliverable: Executive Cost Memo

Redacta un memo de **5 líneas** listo para presentar a liderazgo técnico:

<div class="ex-box">
  <h4>Plantilla del Memo Ejecutivo (5 Líneas):</h4>
  <ol>
    <li><strong>Perfil del Corpus:</strong> $N = \dots$ chunks, promedio de $\bar{T} = \dots$ tokens por chunk (Total: $\dots$ tokens).</li>
    <li><strong>Mejor Configuración:</strong> Modelo [Nombre], dimensión $d = \dots$, chunking [Estrategia].</li>
    <li><strong>Métricas Validadas:</strong> $\mathrm{Recall@}5 = \dots$, $\mathrm{MRR} = \dots$ sobre $M = \dots$ consultas.</li>
    <li><strong>Presupuesto Operativo:</strong> Indexación de $\$\dots$ USD y gasto mensual de $\$\dots$ USD.</li>
    <li><strong>Recomendación Técnica:</strong> [API Administrada / Self-Host en VPC] porque [Justificación].</li>
  </ol>
</div>

Note:
[03:00]
Wrap up the lab. Announce teams that achieved the highest Recall@5.

---

## Questions & References

<div class="highlight-box">
  <h4>Recursos y Documentación Recomendada:</h4>
  <ul>
    <li><strong>Tutorial de referencia:</strong> <em>Text Embeddings in Python: Math, Pipelines, Metrics, Cost</em> (<a href="https://myengineeringpath.dev/genai-engineer/embeddings/" target="_blank">myengineeringpath.dev</a>).</li>
    <li><strong>Leaderboard MTEB:</strong> Massive Text Embedding Benchmark (<a href="https://huggingface.co/spaces/mteb/leaderboard" target="_blank">HuggingFace Spaces</a>).</li>
    <li><strong>Sentence-Transformers:</strong> Documentación oficial (<a href="https://sbert.net" target="_blank">sbert.net</a>).</li>
    <li><strong>Matryoshka Representation Learning:</strong> Kusupati et al. (NeurIPS 2022).</li>
  </ul>
</div>

<p style="text-align: center; margin-top: 30px; font-family: 'Crimson Text', serif; font-size: 1.3em; color: var(--heading-color);">
  ¡Gracias por su participación! ¿Preguntas o reflexiones finales?
</p>
