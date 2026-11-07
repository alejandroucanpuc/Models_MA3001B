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

Consider three user queries seeking roadside assistance:

<div class="alert-box">
  <ul>
    <li><code>"how to fix a flat tire"</code></li>
    <li><code>"changing a punctured tyre"</code></li>
    <li><code>"roadside wheel repair"</code></li>
  </ul>
</div>

- **Lexical reality:** Exact word overlap between queries is virtually **zero** (spelling differences, synonyms, phrasing). <!-- .element: class="fragment" -->
- **Semantic reality:** They represent the **exact same user intent**. <!-- .element: class="fragment" -->
- **BM25 / TF-IDF:** Will rank documents with matching tokens higher, completely missing synonymous knowledge. <!-- .element: class="fragment" -->

<div class="highlight-box">
  <strong>Core principle of embeddings:</strong> Map natural language into a continuous vector space where <em>equivalent meaning implies geometric proximity</em>:
  $$\mathrm{dist}(e_1, e_2) \to 0 \iff \mathrm{semantics}(t_1) \approx \mathrm{semantics}(t_2)$$
</div>

Note:
[00:12 - 00:15]
Ask students for real-world examples: medical terminology (myocardial infarction vs. heart attack), legal contracts, multilingual search (English query finding Spanish documents).

--

## Where Embeddings Power Systems

<div class="container">
  <div class="col">
    <h4><i class="fa-solid fa-magnifying-glass"></i> Retrieval & RAG</h4>
    <p>Semantic search over millions of chunks. Grounds generative LLMs with factual context to stop hallucinations.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4><i class="fa-solid fa-layer-group"></i> Discovery & Deduplication</h4>
    <p>Clustering (HDBSCAN, UMAP) for discovering emerging themes in logs, support tickets, and news feeds.</p>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-tags"></i> Few-Shot Classification</h4>
    <p>Linear probe: feed frozen embeddings into Logistic Regression. High accuracy with 5–10 examples per class.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4><i class="fa-solid fa-bullseye"></i> RecSys & Anomaly Detection</h4>
    <p>User-item dual embedding spaces, cosine similarity for candidate generation, vector distance outliers.</p>
  </div>
</div>

Note:
[00:15 - 00:18]
Stress that embeddings are not just for RAG; they are the general representation layer for all unstructured data.

--

## Learning Goals

By the end of this 180-minute session, you will be able to:

1. **Formalize an embedding model** as a composition $f_\theta = \mathrm{norm}\circ \mathrm{pool}\circ \mathrm{Enc}_\theta \circ \tau$ living on $S^{d-1}$. <!-- .element: class="fragment" -->
2. **Prove the geometric identity** that unifies Cosine similarity, Dot product, and Euclidean distance on normalized hyperspheres. <!-- .element: class="fragment" -->
3. **Design a production pipeline:** select chunking strategy, handle query/document asymmetry, and scale with ANN indexes. <!-- .element: class="fragment" -->
4. **Quantitatively evaluate retrieval** using Recall@k, MRR, and nDCG on a curated ground-truth test set. <!-- .element: class="fragment" -->
5. **Calculate infrastructure economics:** accurately forecast indexing cost, monthly query costs, and RAM storage footprints. <!-- .element: class="fragment" -->

Note:
[00:18 - 00:20]
Review the learning contract. The session balances rigorous linear algebra with real production engineering numbers.

---

# 2. The Math
<span class="badge badge-teal">Block 2 • 30 min</span>

> The geometry of dense representations: hyperspheres, projections, and contrastive optimization.

Note:
[00:20 - 00:25]
Transition into the mathematical formulation. We define the model formally, examine the pooling operations, and analyze the geometry of the unit sphere.

---

## Formal Definition

<div class="def-box">
  <strong>Definición (Modelo de Embedding):</strong> Sea $\Sigma$ un alfabeto finito y $\Sigma^* = \bigcup_{k=0}^\infty \Sigma^k$ el conjunto de todas las cadenas de texto finitas. Un <strong>modelo de embedding</strong> parametrizado por $\theta$ es una función medible:
  $$f_\theta : \Sigma^* \longrightarrow \mathbb{R}^d$$
  que mapea cualquier cadena de texto arbitraria a un espacio vectorial de dimensión fija $d \in \mathbb{N}$, entrenada de modo que la <strong>afinidad semántica</strong> entre textos induzca <strong>proximidad geométrica</strong> en $\mathbb{R}^d$.
</div>

- **Representación densa:** A diferencia de one-hot o TF-IDF ($d \approx 10^5$, 99.9% ceros), cada coordenada en $\mathbb{R}^d$ es un número real continuo y denso.
- **Sin ejes monosemánticos:** No existe un eje que represente "realeza" o "tiempo"; los conceptos son patrones distribuidos sobre el espacio latente.

Note:
[00:25 - 00:28]
Highlight the contrast with bag-of-words. In dense vectors, dimensionality is small ($d \in [384, 3072]$) and information is distributed across all dimensions.

--

## Anatomy of $f_\theta$

Un modelo moderno de embedding descompone su cómputo en cuatro etapas sucesivas:

<div class="def-box">
  $$f_\theta = \mathrm{norm} \circ \mathrm{pool} \circ \mathrm{Enc}_\theta \circ \tau$$
</div>

<svg viewBox="0 0 760 160" style="width:100%; max-height:190px; margin: 15px auto; display:block;">
  <!-- Text Input -->
  <rect x="10" y="55" width="105" height="50" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="62" y="77" text-anchor="middle" font-family="Crimson Text" font-weight="bold" font-size="14" fill="#073642">Texto x</text>
  <text x="62" y="94" text-anchor="middle" font-family="Fira Code" font-size="11" fill="#657B83">x ∈ Σ*</text>
  <path d="M 115 80 L 145 80" stroke="#CB4B16" stroke-width="2"/>

  <!-- Tokenizer -->
  <rect x="150" y="55" width="120" height="50" rx="8" fill="#FDF6E3" stroke="#CB4B16" stroke-width="2"/>
  <text x="210" y="77" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#CB4B16">Tokenizador τ</text>
  <text x="210" y="94" text-anchor="middle" font-family="Fira Code" font-size="11" fill="#657B83">(t₁,...,tₙ)</text>
  <path d="M 270 80 L 300 80" stroke="#CB4B16" stroke-width="2"/>

  <!-- Encoder -->
  <rect x="305" y="50" width="145" height="60" rx="8" fill="#FDF6E3" stroke="#2AA198" stroke-width="2.5"/>
  <text x="377" y="75" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#2AA198">Transformer Enc_θ</text>
  <text x="377" y="94" text-anchor="middle" font-family="Fira Code" font-size="11" fill="#073642">H ∈ ℝ^(n×d)</text>
  <path d="M 450 80 L 480 80" stroke="#2AA198" stroke-width="2"/>

  <!-- Pooling -->
  <rect x="485" y="55" width="115" height="50" rx="8" fill="#FDF6E3" stroke="#6C71C4" stroke-width="2"/>
  <text x="542" y="77" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#6C71C4">Pooling</text>
  <text x="542" y="94" text-anchor="middle" font-family="Fira Code" font-size="11" fill="#657B83">z ∈ ℝ^d</text>
  <path d="M 600 80 L 630 80" stroke="#6C71C4" stroke-width="2"/>

  <!-- Normalization -->
  <rect x="635" y="55" width="115" height="50" rx="8" fill="#EEE8D5" stroke="#859900" stroke-width="2.5"/>
  <text x="692" y="77" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#859900">Norm-L2</text>
  <text x="692" y="94" text-anchor="middle" font-family="Fira Code" font-weight="bold" font-size="12" fill="#859900">e ∈ S^(d-1)</text>
</svg>

- $\tau$: Segmenta en subpalabras (BPE o WordPiece): $x \mapsto (t_1, \dots, t_n)$.
- $\mathrm{Enc}_\theta$: Matriz contextualizada $H = [h_1, \dots, h_n]^\top \in \mathbb{R}^{n \times d}$.
- $\mathrm{pool}$: Colapsa $n$ vectores de tokens en un único vector estático $z \in \mathbb{R}^d$.
- $\mathrm{norm}$: Proyecta $z$ radialmente sobre la hipersfera unitaria $S^{d-1}$.

Note:
[00:28 - 00:32]
Point out that the transformer outputs $n$ vectors (one per token). Pooling is essential to convert variable-length tokens into a single fixed vector.

--

## Pooling & Normalization

### 1. Mean Pooling (Promedio Aritmético)
Colapsa la secuencia ponderando con la máscara de atención $\mathbb{I}(t_i \neq \text{pad})$:

<div class="def-box">
  $$z = \frac{\sum_{i=1}^n m_i h_i}{\sum_{i=1}^n m_i} \in \mathbb{R}^d, \qquad m_i \in \{0, 1\}$$
</div>

*(Nota: En BERT antiguo se usaba el token `[CLS]`, pero estudios empíricos en Sentence-BERT demostraron que mean-pooling supera a `[CLS]` por más de 12 puntos en STS).*

### 2. Normalización L2
<div class="def-box">
  $$e = \mathrm{norm}(z) = \frac{z}{\|z\|_2} = \frac{z}{\sqrt{\sum_{j=1}^d z_j^2}} \in S^{d-1}$$
</div>

Todos los embeddings normalizados habitan en la **superficie de la hipersfera unitaria** $S^{d-1} = \{x \in \mathbb{R}^d : \|x\|_2 = 1\}$.

Note:
[00:32 - 00:35]
Emphasize why normalization is non-negotiable. Without normalization, vector lengths corrupt distance rankings.

--

## Toy Example in $\mathbb{R}^2$: The Geometry of Meaning

Consider three vectors on the unit circle $S^1 \subset \mathbb{R}^2$:

$$u = (1, 0), \qquad v = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right), \qquad w = (0, 1)$$

<div class="container">
  <div class="col" style="flex: 1.1;">
    <svg viewBox="0 0 280 230" style="width:100%; display:block; margin:auto;">
      <!-- Grid axes -->
      <line x1="40" y1="190" x2="250" y2="190" stroke="#93A1A1" stroke-width="1.5"/>
      <line x1="50" y1="200" x2="50" y2="20" stroke="#93A1A1" stroke-width="1.5"/>
      <text x="245" y="185" font-family="Fira Code" font-size="11" fill="#586E75">x₁</text>
      <text x="55" y="25" font-family="Fira Code" font-size="11" fill="#586E75">x₂</text>

      <!-- Unit arc -->
      <path d="M 190 190 A 140 140 0 0 0 50 50" fill="none" stroke="#B58900" stroke-width="2" stroke-dasharray="3"/>

      <!-- Vector u -->
      <line x1="50" y1="190" x2="190" y2="190" stroke="#CB4B16" stroke-width="3"/>
      <circle cx="190" cy="190" r="4" fill="#CB4B16"/>
      <text x="195" y="205" font-family="Fira Code" font-weight="bold" font-size="12" fill="#CB4B16">u (1,0)</text>

      <!-- Vector v -->
      <line x1="50" y1="190" x2="149" y2="91" stroke="#2AA198" stroke-width="3"/>
      <circle cx="149" cy="91" r="4" fill="#2AA198"/>
      <text x="156" y="90" font-family="Fira Code" font-weight="bold" font-size="12" fill="#2AA198">v (0.71, 0.71)</text>

      <!-- Vector w -->
      <line x1="50" y1="190" x2="50" y2="50" stroke="#6C71C4" stroke-width="3"/>
      <circle cx="50" cy="50" r="4" fill="#6C71C4"/>
      <text x="56" y="45" font-family="Fira Code" font-weight="bold" font-size="12" fill="#6C71C4">w (0,1)</text>

      <!-- Angle labels -->
      <path d="M 90 190 A 40 40 0 0 0 78 162" fill="none" stroke="#2AA198" stroke-width="1.5"/>
      <text x="96" y="175" font-family="Inter" font-size="10" fill="#2AA198">45°</text>
      <path d="M 78 162 A 40 40 0 0 0 50 150" fill="none" stroke="#6C71C4" stroke-width="1.5"/>
      <text x="64" y="145" font-family="Inter" font-size="10" fill="#6C71C4">45°</text>
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
    <p style="font-size: 0.85em; color: var(--heading-color);">
      En producción, $d \in [384, 3072]$. El volumen de una esfera en alta dimensión se concentra en la cáscara externa: casi todos los pares aleatorios son <em>casi ortogonales</em>.
    </p>
  </div>
</div>

Note:
[00:35 - 00:38]
Note on curse of dimensionality: in 1536 dimensions, the angle between two random vectors is tightly concentrated around 90 degrees ($\cos \approx 0$).

---

## Similarity Measures in Vector Space

Sean $u, v \in \mathbb{R}^d$. Definimos tres operadores estándar:

<div class="container">
  <div class="col">
    <h4>Similitud Coseno</h4>
    $$\cos(u, v) = \frac{\langle u, v \rangle}{\|u\|_2 \, \|v\|_2}$$
    <p style="font-size:0.8em;">Invariante a la escala de los vectores. Mide exclusivamente la alineación angular.</p>
  </div>
  <div class="col">
    <h4>Producto Punto</h4>
    $$\langle u, v \rangle = \sum_{i=1}^d u_i v_i = u^\top v$$
    <p style="font-size:0.8em;">Operación fundamental BLAS de nivel 1. Máxima eficiencia en GPUs/CPUs.</p>
  </div>
  <div class="col">
    <h4>Distancia Euclidiana ($L_2$)</h4>
    $$\|u - v\|_2 = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
    <p style="font-size:0.8em;">Métrica canónica en espacios normados. Cumple desigualdad del triángulo.</p>
  </div>
</div>

Note:
[00:38 - 00:40]
Ask students: "Which one should we use in our database index?" Lead them to the key identity on the next slide.

--

## The Fundamental Identity on $S^{d-1}$

<div class="def-box">
  <strong>Teorema (Equivalencia de Rango en la Hipersfera):</strong> Si los vectores están normalizados en norma $L_2$, es decir, $\|u\|_2 = \|v\|_2 = 1$:
  $$\begin{aligned}
  \|u - v\|_2^2 &= \langle u - v, u - v \rangle \\
  &= \|u\|_2^2 - 2\langle u, v \rangle + \|v\|_2^2 \\
  &= 1 - 2\langle u, v \rangle + 1 \\
  &= \mathbf{2 - 2\langle u, v \rangle} = \mathbf{2 - 2\cos(u, v)}
  \end{aligned}$$
</div>

<div class="highlight-box">
  <strong>Implicación computacional directa:</strong>
  $$\arg\max_v \cos(u, v) \equiv \arg\max_v \langle u, v \rangle \equiv \arg\min_v \|u - v\|_2$$
  ¡Los tres criterios producen exactamente el <strong>mismo ranking top-$k$</strong>!
</div>

- El **producto punto** $u^\top v$ omite las divisiones y raíces de las normas $\implies$ **es el más veloz**. <!-- .element: class="fragment" -->
- <span style="color:var(--accent-red); font-weight:bold;">¡Advertencia!</span> Si los vectores **no están normalizados**, los tres ordenamientos discrepan drásticamente. <!-- .element: class="fragment" -->

Note:
[00:40 - 00:43]
Walk through the proof line by line. It is simple, beautiful, and foundational for index design in FAISS/pgvector.

--

## Metric Properties: Similarity vs. Distance

No toda función de puntuación es una distancia formal en topología:

| Medida | Rango | ¿Es Métrica Formal? | Propiedades Matemáticas |
| :--- | :---: | :---: | :--- |
| **Similitud Coseno** | $[-1, 1]$ | **NO** | No es distancia (viola $\mathrm{dist}(x,x)=0$, no cumple desigualdad triangular) |
| **Distancia Angular** | $[0, 1]$ | **SÍ** | $d_{\text{ang}}(u, v) = \frac{\arccos(\cos(u, v))}{\pi}$. Métrica geodésica sobre la esfera. |
| **Distancia Euclidiana ($L_2$)** | $[0, \infty)$ | **SÍ** | Cumple no negatividad, identidad de indiscernibles, simetría y subaditividad. |
| **Distancia Manhattan ($L_1$)** | $[0, \infty)$ | **SÍ** | $\sum |u_i - v_i|$. Menos sensible a dimensiones anómalas extremas. |

<div class="alert-box">
  <strong>Precaución de diseño:</strong> Los índices basados en árboles espaciales (como KD-Trees o VP-Trees) exigen estrictamente la desigualdad del triángulo para podar ramas. La similitud coseno en crudo no puede usarse para podar sin transformar a distancia angular o $L_2$.
</div>

Note:
[00:43 - 00:45]
Highlight why KD-trees fail with cosine similarity unless converted to metric distance.

---

## Contrastive Learning: How $\theta$ is Trained

Los modelos de embedding se entrenan típicamente mediante **aprendizaje contrastivo** usando la pérdida **InfoNCE** (Information Noise-Contrastive Estimation):

<div class="def-box">
  Dada una consulta ancla $q$, un pasaje positivo relevante $d^+$ y un conjunto de pasajes negativos $\{d_1^-, \dots, d_{B-1}^-\}$ en un mini-batch de tamaño $B$:
  $$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\left(\cos(q, d^+) / \tau\right)}{\exp\left(\cos(q, d^+) / \tau\right) + \sum_{j=1}^{B-1} \exp\left(\cos(q, d_j^-) / \tau\right)}$$
  donde $\tau > 0$ es un hiperparámetro de temperatura que calibra la agudeza de la distribución softmax.
</div>

<svg viewBox="0 0 680 130" style="width:100%; max-height:150px; margin: 10px auto; display:block;">
  <!-- Anchor -->
  <circle cx="160" cy="65" r="30" fill="#EEE8D5" stroke="#CB4B16" stroke-width="2.5"/>
  <text x="160" y="62" text-anchor="middle" font-family="Crimson Text" font-weight="bold" font-size="14" fill="#073642">Consulta q</text>
  <text x="160" y="77" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#CB4B16">Ancla</text>

  <!-- Pull Force to Positive -->
  <path d="M 195 65 L 300 65" stroke="#859900" stroke-width="3"/>
  <text x="250" y="55" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#859900">Atracción</text>

  <!-- Positive Doc -->
  <circle cx="335" cy="65" r="28" fill="#FDF6E3" stroke="#859900" stroke-width="2.5"/>
  <text x="335" y="62" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#859900">Doc d⁺</text>
  <text x="335" y="77" text-anchor="middle" font-family="Fira Code" font-size="9" fill="#586E75">Relevante</text>

  <!-- Repel Forces to Negatives -->
  <path d="M 180 40 L 460 25" stroke="#DC322F" stroke-width="1.8" stroke-dasharray="4"/>
  <path d="M 180 85 L 460 75" stroke="#DC322F" stroke-width="1.8" stroke-dasharray="4"/>
  <path d="M 175 95 L 460 115" stroke="#DC322F" stroke-width="1.8" stroke-dasharray="4"/>
  <text x="340" y="115" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#DC322F">Repulsión (In-batch + Hard Negatives)</text>

  <!-- Negatives -->
  <circle cx="490" cy="25" r="16" fill="#EEE8D5" stroke="#DC322F" stroke-width="1.5"/>
  <text x="490" y="29" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#DC322F">d₁⁻</text>

  <circle cx="490" cy="70" r="16" fill="#EEE8D5" stroke="#DC322F" stroke-width="1.5"/>
  <text x="490" y="74" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#DC322F">d₂⁻</text>

  <circle cx="490" cy="115" r="16" fill="#EEE8D5" stroke="#DC322F" stroke-width="1.5"/>
  <text x="490" y="119" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#DC322F">d₃⁻</text>
</svg>

- **In-batch negatives:** Todos los demás documentos del batch actúan como negativos gratuitos sin cómputo extra.
- **Hard negatives:** Negativos léxicamente idénticos pero semánticamente distintos (minados con BM25) para obligar al modelo a discriminar sutilezas.

Note:
[00:45 - 00:48]
Explain temperature $\tau$. A lower temperature acts like hard mining, forcing high gradient penalties on top errors.

--

## Matryoshka Representation Learning (MRL)

¿Podemos reducir la dimensión del vector **sin reentrenar** el modelo?

<div class="def-box">
  <strong>Pérdida Matryoshka:</strong> Se entrena el modelo optimizando simultáneamente la pérdida contrastiva sobre múltiples prefijos anidados $d_1 < d_2 < \dots < d_m$:
  $$\mathcal{L}_{\text{MRL}} = \sum_{k=1}^m \lambda_k \, \mathcal{L}\left(e_{1:d_k}\right)$$
</div>

<div class="highlight-box">
  <strong>Truncamiento y Renormalización en Inferencia:</strong>
  Para un vector $e \in \mathbb{R}^{1536}$, si deseamos utilizar solo $d' = 256$:
  $$e' = \frac{e_{1:d'}}{\|e_{1:d'}\|_2} \in S^{d'-1}$$
</div>

- **Compresión 6x sin nuevo entrenamiento:** Los primeros $d'$ coeficientes contienen la mayor densidad de señal discriminativa.
- Soportado de forma nativa en OpenAI `text-embedding-3`, Nomic Embed, y modelos modernos de Cohere y BAAI.

Note:
[00:48 - 00:50]
Analogy: Russian nesting dolls (Matryoshka). The first dimensions carry global semantic cluster info, while later dimensions capture fine-grained distinctions.

--

## Four Dangerous Misconceptions

<div class="container">
  <div class="col">
    <div class="alert-box">
      <strong>1. Dimensión ≠ Atributo</strong>
      <p style="font-size:0.8em;">No intentes buscar la "coordenada 42" para saber si el texto es formal. Las representaciones son densas y distribuidas geométricamente.</p>
    </div>
    <div class="alert-box">
      <strong>2. Puntuación ≠ Probabilidad</strong>
      <p style="font-size:0.8em;">Un coseno de 0.85 <strong>no significa</strong> 85% de probabilidad de acierto. La distribución de scores depende de la temperatura y la dimensionalidad.</p>
    </div>
  </div>
  <div class="col">
    <div class="alert-box">
      <strong>3. No compares modelos distintos</strong>
      <p style="font-size:0.8em;">Un 0.80 en BGE no equivale a un 0.80 en OpenAI. Sus geometrías espaciales son completamente incompatibles.</p>
    </div>
    <div class="alert-box">
      <strong>4. Coseno alto ≠ Verdad factual</strong>
      <p style="font-size:0.8em;">Una alucinación falsa que comparta la estructura semántica de la consulta arrojará un coseno extremadamente elevado.</p>
    </div>
  </div>
</div>

Note:
[00:50 - 00:55]
These 4 mistakes cost teams months of debugging. Emphasize never setting an arbitrary score threshold like `score > 0.8` without calibrating on actual validation data.

---

# 3. Models, Dimensions & Quantization
<span class="badge badge-amber">Block 3 • 15 min</span>

> The balance between retrieval accuracy, memory capacity, and inference latency.

Note:
[00:55 - 00:58]
Now we translate theory into hardware and engineering constraints: picking models, storage footprint, and quantization.

---

## Choosing an Embedding Model

| Eje de Decisión | Pregunta Crítica | Trade-offs Principales |
| :--- | :--- | :--- |
| **Calidad de Dominio** | ¿Funciona en mi idioma y jerga técnica? | Modelos en inglés vs. multilingües (BGE-M3); términos biomédicos o legales. |
| **Costos Operativos** | ¿API administrada o auto-hospedaje en GPU? | API: pago por token, cero ops. Self-host: costo fijo de GPU, privacidad total. |
| **Soberanía del Dato** | ¿Pueden salir los textos a la nube pública? | Regulaciones GDPR, HIPAA, banca privada exigen modelos locales en VPC privada. |
| **Ventana de Contexto** | ¿Cuántos tokens por chunk puede procesar? | Rango típico de 512 tokens (BERT) hasta 8,192 o 32,768 tokens (Voyage, Jina). |
| **Dimensión $d$** | ¿Cuál es la huella de memoria resultante? | De 384 a 3072 dimensiones. Determina el costo en RAM del índice vectorial. |

<div class="highlight-box">
  <strong>Familias líderes actuales:</strong>
  <ul>
    <li><strong>APIs propietarias:</strong> OpenAI (<code>text-embedding-3-small/large</code>), Cohere (<code>embed-v3</code>), Voyage AI.</li>
    <li><strong>Open Source (Self-host):</strong> BAAI (<code>bge-small-en-v1.5</code>, <code>bge-m3</code>), GTE-Qwen2, Nomic-Embed.</li>
  </ul>
</div>

Note:
[00:58 - 01:02]
Never trust MTEB leaderboards blindly! MTEB tests academic tasks. Always evaluate 2-3 candidate models on your domain's actual queries and corpus.

--

## Dimension, Storage & Memory Scaling

En precisión completa (`float32`), cada número de punto flotante ocupa **4 bytes**.  
El espacio en memoria consumido por un único vector es:

$$\text{Memoria por vector} = 4 \times d \text{ bytes}$$

| Dimensión $d$ | Bytes / Vector | KB / Vector | Vectores por 1 GB RAM | Huella de $10^6$ Vectores |
| :---: | :---: | :---: | :---: | :---: |
| **384** (MiniLM, BGE-small) | 1,536 B | **1.5 KB** | ~699,000 | **1.5 GB** |
| **768** (BERT-base, GTE-base) | 3,072 B | **3.0 KB** | ~349,000 | **3.0 GB** |
| **1024** (BGE-large, Cohere) | 4,096 B | **4.0 KB** | ~262,000 | **4.0 GB** |
| **1536** (OpenAI text-emb-3) | 6,144 B | **6.0 KB** | ~174,000 | **6.0 GB** |
| **3072** (OpenAI emb-3-large) | 12,288 B | **12.0 KB** | ~87,000 | **12.0 GB** |

*(Nota: Los grafos HNSW requieren un 20% a 50% de sobrecarga adicional en RAM para almacenar las conexiones de aristas).*

Note:
[01:02 - 01:06]
Walk through the numbers: 10 million vectors at 1536d requires over 60 GB of RAM just for raw floats! This leads naturally to quantization.

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
    <li><strong>Etapa 1 (Filtro grueso):</strong> Buscar en el índice binario o int8 a velocidad de hardware para extraer los top-$k$ candidatos (e.g. $k=100$).</li>
    <li><strong>Etapa 2 (Refinamiento):</strong> Reordenar los 100 candidatos calculando el producto punto exacto en float32 sobre los vectores originales leídos de disco o memoria.</li>
  </ol>
</div>

Note:
[01:06 - 01:10]
Binary quantization uses bitwise XOR and POPCNT instructions: up to 40x faster retrieval than float operations!

---

# 4. Pipelines & Search
<span class="badge badge-teal">Block 4 • 25 min</span>

> From raw multi-page documents to microsecond approximate vector lookups.

Note:
[01:10 - 01:12]
We now build the end-to-end pipeline: document ingestion, chunking trade-offs, code implementation, and indexing algorithms.

---

## The End-to-End Search Pipeline

Una falla en cualquiera de los eslabones degrada irreversiblemente la precisión final:

<svg viewBox="0 0 760 180" style="width:100%; max-height:210px; margin: 10px auto; display:block;">
  <!-- Corpus -->
  <rect x="5" y="65" width="95" height="50" rx="8" fill="#EEE8D5" stroke="#B58900" stroke-width="2"/>
  <text x="52" y="87" text-anchor="middle" font-family="Crimson Text" font-weight="bold" font-size="13" fill="#073642">Documentos</text>
  <text x="52" y="103" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">PDF / HTML</text>
  <path d="M 100 90 L 125 90" stroke="#CB4B16" stroke-width="2"/>

  <!-- Chunking -->
  <rect x="130" y="65" width="105" height="50" rx="8" fill="#FDF6E3" stroke="#CB4B16" stroke-width="2"/>
  <text x="182" y="87" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#CB4B16">Chunking</text>
  <text x="182" y="103" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">400 tokens</text>
  <path d="M 235 90 L 260 90" stroke="#CB4B16" stroke-width="2"/>

  <!-- Embedding Model -->
  <rect x="265" y="65" width="115" height="50" rx="8" fill="#FDF6E3" stroke="#2AA198" stroke-width="2"/>
  <text x="322" y="87" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#2AA198">Embedding</text>
  <text x="322" y="103" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">f_θ(chunk)</text>
  <path d="M 380 90 L 405 90" stroke="#2AA198" stroke-width="2"/>

  <!-- Vector DB -->
  <rect x="410" y="65" width="105" height="50" rx="8" fill="#FDF6E3" stroke="#6C71C4" stroke-width="2"/>
  <text x="462" y="87" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#6C71C4">Índice Vectorial</text>
  <text x="462" y="103" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">HNSW / IVF</text>
  <path d="M 515 90 L 540 90" stroke="#6C71C4" stroke-width="2"/>

  <!-- ANN Search -->
  <rect x="545" y="65" width="100" height="50" rx="8" fill="#FDF6E3" stroke="#268BD2" stroke-width="2"/>
  <text x="595" y="87" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="12" fill="#268BD2">Búsqueda</text>
  <text x="595" y="103" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">Top-k scores</text>
  <path d="M 645 90 L 670 90" stroke="#268BD2" stroke-width="2"/>

  <!-- LLM / RAG -->
  <rect x="675" y="65" width="80" height="50" rx="8" fill="#EEE8D5" stroke="#859900" stroke-width="2.5"/>
  <text x="715" y="87" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="13" fill="#859900">LLM</text>
  <text x="715" y="103" text-anchor="middle" font-family="Fira Code" font-size="10" fill="#657B83">Generación</text>

  <!-- Query Flow from top -->
  <path d="M 322 15 L 322 55" stroke="#CB4B16" stroke-width="2" stroke-dasharray="3"/>
  <text x="322" y="10" text-anchor="middle" font-family="Inter" font-weight="bold" font-size="11" fill="#CB4B16">Consulta q en vivo</text>
</svg>

<div class="alert-box">
  <strong>Regla de oro:</strong> Una mala partición en chunks o un modelo desalineado arruina el contexto del LLM antes de que este genere una sola palabra.
</div>

Note:
[01:12 - 01:15]
Emphasize the cascade failure: if the chunk doesn't contain the answer, no LLM in the world can generate the right factual response.

--

## Chunking Strategies: Partitioning the Corpus

| Estrategia | Calidad Semántica | Costo Cómputo | Cuándo Utilizarla |
| :--- | :---: | :---: | :--- |
| **Tamaño Fijo (+ Solapamiento)** | Media | Mínimo | Prototipado rápido, texto continuo sin estructura (transcripciones). |
| **Recursiva Jerárquica** | Buena | Bajo | **Opción predeterminada**. Divide por párrafos `\n\n`, luego oraciones `.` |
| **Basada en Estructura** | Excelente | Bajo | Documentación técnica, código, artículos Markdown, tablas HTML. |
| **Chunking Semántico** | Óptima | Alto | Textos críticos donde la frontera entre ideas requiere evaluar distancias vectoriales entre oraciones consecutivas. |

<div class="highlight-box">
  <strong>Configuración base recomendada para iniciar:</strong>
  <ul>
    <li>Tamaño de chunk: <strong>~350 a 500 tokens</strong></li>
    <li>Solapamiento (Overlap): <strong>~50 tokens</strong> (10% a 15% para no cortar oraciones a la mitad)</li>
  </ul>
</div>

Note:
[01:15 - 01:18]
Explain why overlap is necessary: it prevents named entities and concepts from being sliced right on the boundary between chunk $i$ and chunk $i+1$.

--

## The Chunk Size Trade-off

<div class="container">
  <div class="col">
    <h4><i class="fa-solid fa-arrows-to-circle"></i> Chunks Demasiado Grandes (&gt; 1000 tokens)</h4>
    <ul>
      <li>El vector se convierte en el <strong>promedio diluido</strong> de muchos temas distintos.</li>
      <li>Pierde resolución para preguntas específicas y puntuales.</li>
      <li>Ocupa más espacio en la ventana de contexto del LLM.</li>
    </ul>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-compress"></i> Chunks Demasiado Pequeños (&lt; 100 tokens)</h4>
    <ul>
      <li>Pierde el contexto circundante (antecedentes, pronombres, sujeto).</li>
      <li>Multiplica exponencialmente la cantidad de vectores en el índice.</li>
      <li>Aumenta el costo de memoria y fragmenta la información.</li>
    </ul>
  </div>
</div>

<div class="alert-box">
  <strong>Límite de truncamiento silencioso:</strong> Si un chunk excede el límite del tokenizador (e.g. 512 tokens), la mayoría de librerías descartan el texto sobrante <em>sin emitir advertencias de error</em>. ¡La mitad de tu documento queda invisible!
</div>

Note:
[01:18 - 01:21]
Stress the silent truncation bug! It is one of the most common bugs in production retrieval systems.

---

## Minimal Semantic Search in Python

Implementación completa en memoria en solo 14 líneas de código:

```python [1-4|6-8|10-14]
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
# Embeddings normalizados directamente en la hipersfera S^(d-1)
D = model.encode(docs, normalize_embeddings=True)  # Matriz de forma (N, d)

# 3. Función de búsqueda vectorial exacta
def search(query: str, k: int = 2):
    q = model.encode([query], normalize_embeddings=True)[0]  # Vector (d,)
    scores = D @ q                                           # Producto punto = Coseno
    top_indices = np.argsort(-scores)[:k]
    return [(docs[i], float(scores[i])) for i in top_indices]

# Probar búsqueda semántica
print(search("How do you evaluate distance between embeddings?"))
```

Note:
[01:21 - 01:25]
Walk students through the matrix multiplication `D @ q`. Because rows of $D$ and vector $q$ have unit norm, the simple matrix-vector product computes all cosine similarities in parallel.

--

## Scaling Up: Approximate Nearest Neighbors (ANN)

Para $N = 10^7$ vectores, la búsqueda exacta por fuerza bruta toma $O(N \cdot d)$ operaciones por consulta (inviable a escala en tiempo real).

<div class="container">
  <div class="col">
    <h4><i class="fa-solid fa-diagram-project"></i> HNSW (Hierarchical Navigable Small World)</h4>
    <p>Grafos multicapa estilo skip-list. Navegación ultra-rápida ($O(\log N)$). Alto recall (&gt;98%), pero consume más memoria RAM.</p>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-boxes-stacked"></i> IVF (Inverted File Index)</h4>
    <p>Clustering con K-Means. Particiona el espacio en celdas Voronoi; solo busca en las celdas más cercanas (`nprobe`). Menos RAM requerida.</p>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-compress"></i> PQ (Product Quantization)</h4>
    <p>Descompone vectores en sub-vectores y los cuantiza con codebooks. Compresión agresiva de memoria para índices masivos.</p>
  </div>
</div>

<div class="highlight-box">
  <strong>Ecosistema de almacenamiento vectorial:</strong>
  <ul>
    <li><strong>Librerías embebidas:</strong> FAISS (Meta), ScaNN (Google), USearch.</li>
    <li><strong>Bases de datos dedicadas:</strong> Qdrant, Pinecone, Milvus, Chroma, Weaviate.</li>
    <li><strong>Extensiones relacionales:</strong> <code>pgvector</code> (PostgreSQL).</li>
  </ul>
</div>

Note:
[01:25 - 01:28]
Briefly explain HNSW: top layers have long-range edges for fast routing, bottom layers have dense local edges for precision.

--

## Advanced Retrieval Architectures

<div class="container">
  <div class="col">
    <h4>1. Búsqueda Híbrida (Dense + Sparse)</h4>
    <p>Combina la búsqueda vectorial densa con <strong>BM25</strong> léxico. Fusión mediante <em>Reciprocal Rank Fusion (RRF)</em>:</p>
    $$RRF(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
    <p style="font-size:0.8em;">Garantiza encontrar palabras clave exactas (códigos de error, IDs) y sinónimos conceptuales.</p>
  </div>
  <div class="col">
    <h4>2. Reordenamiento con Cross-Encoder</h4>
    <p>La búsqueda vectorial extrae rápidamente los top-50 candidatos. Luego, un <strong>Cross-Encoder</strong> evalúa el par <code>[Query, Doc]</code> con atención cruzada total:</p>
    $$\text{Score} = \mathrm{CrossEncoder}(q, d_i)$$
    <p style="font-size:0.8em;">Recupera interacciones finas entre términos que el bi-encoder pierde por compresión.</p>
  </div>
</div>

Note:
[01:28 - 01:31]
Bi-encoder (embeddings) for fast candidate retrieval + Cross-encoder for precise reranking is the industry-standard architecture at Google, Netflix, and Amazon.

--

## Query vs. Document Embeddings (Asymmetry)

En sistemas de búsqueda, las consultas y los documentos tienen naturalezas distintas:
- **Consulta:** Corta, interrogativa, incompleta (e.g. *"síntomas gripe"*).
- **Documento:** Largo, expositivo, estructurado (e.g. *"El virus de la influenza provoca..."*).

<div class="def-box">
  <strong>Prefijos Asimétricos:</strong> Muchos modelos modernos exigen prefijos obligatorios para condicionar el vector al rol adecuado:
  <ul>
    <li><strong>BGE:</strong> Requiere <code>"Represent this sentence for searching relevant passages: "</code> en las consultas.</li>
    <li><strong>Cohere / E5:</strong> Parámetros explícitos <code>input_type="search_query"</code> vs. <code>input_type="search_document"</code>.</li>
  </ul>
</div>

<div class="alert-box">
  <strong>¡Regla inviolable!</strong> Ambos deben procesarse siempre con el <strong>mismo modelo</strong>. Si cambias de modelo en producción, <em>debes re-indexar la totalidad de tu base de documentos</em>.
</div>

Note:
[01:31 - 01:33]
Explain asymmetric modeling: queries and documents occupy slightly different sub-spaces, bridged by instruction prompts.

---

## Six Silent Production Pitfalls

<div class="alert-box">
  <ol>
    <li><strong>Modelos distintos en query e index:</strong> Usar un modelo para indexar y otro más rápido para responder consultas. El producto punto será ruido aleatorio.</li>
    <li><strong>Vectores huérfanos / obsoletos:</strong> Modificar documentos en el backend sin disparar la re-indexación de sus chunks vectoriales asociados.</li>
    <li><strong>Truncamiento silencioso:</strong> Enviar chunks de 1000 tokens a un modelo con límite de 512 tokens sin supervisar los logs del tokenizador.</li>
    <li><strong>Métrica incorrecta:</strong> Usar distancia $L_2$ o producto punto sobre vectores que <em>no</em> fueron normalizados en $L_2$.</li>
    <li><strong>Indexar documentos completos:</strong> Generar un solo vector para un PDF de 20 páginas. Toda la información se convierte en un promedio difuso.</li>
    <li><strong>Elegir modelos por leaderboards:</strong> Confiar ciegamente en el ranking MTEB general sin validar el rendimiento en el propio corpus de producción.</li>
  </ol>
</div>

Note:
[01:33 - 01:35]
Reiterate this slide before moving to metrics. Every student who builds a production system will encounter at least one of these traps.

---

# 5. Metrics & Evaluation
<span class="badge badge-orange">Block 5 • 20 min</span>

> "If you cannot measure it, you cannot improve it."  
> Quantitative validation of information retrieval systems.

Note:
[01:35 - 01:38]
Transition to evaluation. How do we know if our retrieval pipeline actually works? We need Information Retrieval (IR) metrics.

---

## Two Fundamentally Different "Metrics"

Es vital no confundir los dos niveles de métricas en un sistema de recuperación:

<div class="container">
  <div class="col">
    <h4>1. Métrica Geométrica (Vectorial)</h4>
    <p>Calcula la relación espacial local entre <strong>dos vectores individuales</strong>:</p>
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
      <li>Normalized Discounted Cumulative Gain (nDCG@k)</li>
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

### Fuentes de obtención:
1. **Logs históricos reales:** Consultas y clics de usuarios en sistemas de búsqueda previos.
2. **Juicio de expertos de dominio:** Anotación manual de documentos pertinentes por especialistas.
3. **Generación sintética con LLMs (Revisada):** Utilizar un LLM para formular preguntas a partir de chunks, y **verificar humanamente** una muestra aleatoria para eliminar falsos positivos.

<div class="alert-box">
  <strong>¡Principio de partición estricta!</strong> Mantén el test set aislado. Jamás lo utilices para calibrar tamaños de chunk ni para fine-tuning.
</div>

Note:
[01:40 - 01:43]
Emphasize test set hygiene. Without a clean, held-out test set, all tuning is just overfitting to personal intuition.

---

## Recall@k and Hit@k

Dado un conjunto de consultas $Q$. Para cada consulta $q$, sea $\mathcal{R}_q$ el conjunto de chunks relevantes y $\mathcal{T}_q^{(k)}$ la lista de los top-$k$ resultados recuperados por el sistema:

<div class="def-box">
  <h4>Recall@k</h4>
  $$\mathrm{Recall@}k = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathcal{R}_q \cap \mathcal{T}_q^{(k)}|}{|\mathcal{R}_q|}$$
  <p style="font-size:0.85em;">Mide la proporción promedio de documentos relevantes que lograron entrar en los top-$k$ puestos.</p>
</div>

<div class="def-box">
  <h4>Hit@k (Success@k)</h4>
  $$\mathrm{Hit@}k = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left(|\mathcal{R}_q \cap \mathcal{T}_q^{(k)}| > 0\right)$$
  <p style="font-size:0.85em;">Vale 1 si <strong>al menos un</strong> documento relevante aparece en el top-$k$; 0 en caso contrario.</p>
</div>

Note:
[01:43 - 01:46]
Explain the intuition: In RAG, if $k=5$, Recall@5 tells us if the context fed to the LLM actually contains the facts needed to answer.

--

## Mean Reciprocal Rank (MRR)

En tareas donde existe una única respuesta correcta (o donde el primer resultado es el más crítico):

<div class="def-box">
  $$\mathrm{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\mathrm{rank}_q}$$
  donde $\mathrm{rank}_q$ denota la posición ordinal del <strong>primer documento relevante</strong> devuelto para la consulta $q$.
  $$\text{Si ningún documento relevante entra en el top-}k \implies \frac{1}{\mathrm{rank}_q} = 0$$
</div>

### Ejemplo de cálculo:
- Si el documento relevante aparece en posición 1: aporte = $1/1 = \mathbf{1.0}$
- Si aparece en posición 2: aporte = $1/2 = \mathbf{0.50}$
- Si aparece en posición 5: aporte = $1/5 = \mathbf{0.20}$
- Penaliza de forma hiperbólica a los sistemas que entierran la respuesta relevante.

Note:
[01:46 - 01:49]
MRR is the standard metric for question answering and customer support FAQ search.

--

## nDCG@k: Graded Relevance

Cuando los documentos tienen distintos niveles de relevancia (e.g. 0 = irrelevante, 1 = marginal, 2 = relevante, 3 = respuesta perfecta):

<div class="def-box">
  <h4>Discounted Cumulative Gain (DCG@k)</h4>
  $$\mathrm{DCG@}k = \sum_{i=1}^k \frac{2^{\mathrm{rel}_i} - 1}{\log_2(i + 1)}$$
</div>

<div class="def-box">
  <h4>Normalized DCG@k (nDCG@k)</h4>
  $$\mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k} \in [0, 1]$$
  donde $\mathrm{IDCG@}k$ (Ideal DCG) es el DCG obtenido ordenando los documentos por relevancia perfecta de mayor a menor.
</div>

- Recompensa situar los documentos más valiosos en las primeras posiciones.
- El factor de descuento $\log_2(i+1)$ modela el decaimiento de la atención del usuario conforme baja en la lista.

Note:
[01:49 - 01:52]
Explain the logarithmic discount: users care far more about the difference between rank 1 and 2 than between rank 11 and 12.

--

## Python Implementation of IR Metrics

Funciones puras y reproducibles en NumPy:

```python [1-5|7-12|14-22]
import numpy as np

def recall_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    """Calcula la fracción de documentos relevantes capturados en el top-k."""
    top_k_set = set(ranked_ids[:k])
    return len(top_k_set & relevant_ids) / len(relevant_ids)

def mrr(ranked_ids: list, relevant_ids: set, k: int = 50) -> float:
    """Calcula el Reciprocal Rank del primer resultado relevante."""
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(ranked_ids: list, rel_dict: dict, k: int) -> float:
    """Calcula nDCG@k para puntuaciones de relevancia graduada."""
    gains = [2**rel_dict.get(doc_id, 0) - 1 for doc_id in ranked_ids[:k]]
    discounts = [np.log2(i + 1) for i in range(1, len(gains) + 1)]
    dcg = sum(g / d for g, d in zip(gains, discounts))
    
    ideal_gains = sorted([2**r - 1 for r in rel_dict.values()], reverse=True)[:k]
    idcg = sum(g / d for g, d in zip(ideal_gains, discounts[:len(ideal_gains)]))
    return dcg / idcg if idcg > 0 else 0.0
```

Note:
[01:52 - 01:55]
Students will use these exact functions during the hands-on lab to score their pipelines.

---

# 6. Costs & Cloud Economics
<span class="badge badge-amber">Block 6 • 15 min</span>

> "Good engineering is doing for ten cents what any fool can do for a dollar."  
> Sizing vector infrastructure and forecasting operational expense.

Note:
[01:55 - 01:57]
Now we address the business side: estimating index cost, monthly search API expense, and RAM hardware sizing.

---

## Architectural Cost Drivers

El costo total de propiedad (TCO) de una infraestructura de embeddings se divide en cinco partidas:

| Componente | Qué lo Determina | Frecuencia |
| :--- | :--- | :--- |
| **Costo de Indexación** | Tokens totales $\times$ Precio por millón del modelo | Único al crear el índice (+ re-indexaciones) |
| **Costo de Consultas (Queries)** | Cantidad de consultas $\times$ Tokens por consulta $\times$ Tarifa | Recurrente mensual |
| **Memoria RAM del Índice** | $N$ vectores $\times d$ dimensiones $\times$ bytes por float | Fijo por capacidad de servidor |
| **Hosting de Vector DB** | Tipo de réplicas, CPU, redundancia multi-AZ | Recurrente mensual |
| **Auto-hospedaje (Self-Host)** | Horas de GPU (e.g. A10G/T4) $+$ tiempo de ingeniería | Costo de infraestructura y DevOps |

Note:
[01:57 - 02:00]
Emphasize re-indexing: whenever you upgrade the embedding model, you must pay the indexing cost all over again for the whole corpus.

--

## The Core Formulas

Sean:
- $N$: Número total de chunks en el corpus
- $\bar{T}$: Promedio de tokens por chunk
- $Q$: Consultas mensuales de usuarios
- $\bar{T}_q$: Promedio de tokens por consulta
- $p$: Precio en USD por cada $1,000,000$ tokens del proveedor

<div class="def-box">
  $$C_{\text{index}} = \left(\frac{N \times \bar{T}}{10^6}\right) \times p \qquad (\text{USD, one-off})$$
</div>

<div class="def-box">
  $$C_{\text{query/mes}} = \left(\frac{Q \times \bar{T}_q}{10^6}\right) \times p \qquad (\text{USD/mes})$$
</div>

<div class="def-box">
  $$\text{RAM Requerida} \approx N \times d \times b \times 1.3 \quad \text{bytes} \quad (b=4\text{ para float32}, b=1\text{ para int8})$$
  *(El factor $1.3$ considera el overhead típico de índices HNSW).*
</div>

Note:
[02:00 - 02:03]
Walk through the units: checking that dimensional analysis yields dollars and bytes correctly.

--

## Worked Real-World Example

Supongamos una base de conocimiento empresarial con:
- $N = 10,000,000$ de chunks ($10^7$)
- Longitud promedio por chunk $\bar{T} = 200$ tokens
- Modelo: OpenAI `text-embedding-3-small` ($p \approx \$0.02$ por millón de tokens)
- Dimensión $d = 1536$

<div class="ex-box">
  <h4>1. Costo de Indexación:</h4>
  $$C_{\text{index}} = \frac{10^7 \times 200}{10^6} \times \$0.02 = 2,000 \times \$0.02 = \mathbf{\$40.00\text{ USD}}$$
  <p style="font-size:0.85em;">¡Sorprendentemente económico! Indexar 2,000 millones de tokens cuesta solo $40.</p>
</div>

<div class="ex-box">
  <h4>2. Huella de Almacenamiento en Memoria:</h4>
  $$\text{Storage (float32)} = 10^7 \times 1536 \times 4\text{ bytes} \approx \mathbf{61.44\text{ GB RAM}}$$
  Con truncamiento **Matryoshka a $d=512$** y cuantización **int8**:
  $$\text{Storage (int8, 512d)} = 10^7 \times 512 \times 1\text{ byte} \approx \mathbf{5.12\text{ GB RAM}}$$
  ¡Reducción de **61.4 GB a 5.1 GB** (más de 91% de ahorro en hardware de servidor)!
</div>

Note:
[02:03 - 02:06]
Highlight the takeaway: API token cost is rarely the bottleneck; RAM and vector DB server hosting is where the actual recurring bills accumulate!

--

## Strategic Cost Optimization Levers

<div class="container">
  <div class="col">
    <h4>1. Reducción Dimensional & Cuantización</h4>
    <p>Aprovecha Matryoshka para podar a $d=512$ y aplica int8. Reduce la factura de RAM hasta en un 90% con menos de 1.5% de pérdida de recall.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4>2. Limpieza y Deduplicación</h4>
    <p>Elimina avisos legales, encabezados repetitivos y cookies antes de chunkear. Menos chunks $=$ menor almacenamiento.</p>
  </div>
  <div class="col">
    <h4>3. Caché de Embeddings por Hash SHA-256</h4>
    <p>Almacena en Redis o Postgres los embeddings calculados con clave <code>hash(texto + modelo_id)</code>. Nunca re-pagues por embedder el mismo texto.</p>
    <hr style="border:0; border-top:1px solid rgba(181,137,0,0.2);">
    <h4>4. Re-indexación Incremental</h4>
    <p>Calcula hashes por documento. Si un artículo cambia, borra y re-calcula únicamente sus chunks correspondientes.</p>
  </div>
</div>

Note:
[02:06 - 02:08]
Always hash content with SHA-256! In web crawling or documentation repos, 20-30% of chunks are identical boilerplate.

--

## Self-Host vs. Managed API: The Economic Break-Even

<div class="container">
  <div class="col">
    <h4><i class="fa-solid fa-cloud"></i> APIs Administradas (OpenAI, Cohere)</h4>
    <ul>
      <li><strong>Ventajas:</strong> Cero mantenimiento de infraestructura, escalabilidad elástica instantánea, modelos en el estado del arte.</li>
      <li><strong>Desventajas:</strong> Privacidad y cumplimiento normativo (los datos viajan a terceros), límites de tasa (rate limits).</li>
      <li><strong>Ideal para:</strong> $N < 50\text{M}$ chunks, tráfico variable, equipos ágiles sin DevOps dedicado.</li>
    </ul>
  </div>
  <div class="col">
    <h4><i class="fa-solid fa-server"></i> Auto-Hospedaje (BGE, GTE en vLLM/Triton)</h4>
    <ul>
      <li><strong>Ventajas:</strong> Privacidad absoluta en VPC interna, latencia local constante, costo marginal cero por query a plena capacidad.</li>
      <li><strong>Desventajas:</strong> Costo fijo de GPU (e.g. A10G $\approx \$1.00$/h $=$ \$720/mes), monitoreo, parches y disponibilidad.</li>
      <li><strong>Ideal para:</strong> Millones de queries al día, datos sensibles regulados (banca, salud, defensa).</li>
    </ul>
  </div>
</div>

<div class="def-box">
  $$\text{Regla de decisión:} \quad \text{Si } C_{\text{API}}(\text{mes}) > \text{Costo GPU}(\text{mes}) + \text{Salario DevOps / Soporte} \implies \text{Auto-hospedar}$$
</div>

Note:
[02:08 - 02:10]
Stress the hidden cost of self-hosting: maintaining a Kubernetes cluster with GPU drivers is not free in engineering hours.

---

# 7. Wrap-up & Key Principles
<span class="badge badge-teal">Block 7 • 5 min</span>

> The 5 golden engineering laws of vector retrieval.

Note:
[02:10 - 02:15]
Synthesize the key theoretical and practical lessons before sending students to the lab.

---

## The 5 Golden Takeaways

<div class="highlight-box">
  <ol>
    <li><strong>Geometría en la Hipersfera $S^{d-1}$</strong><br>
    Al normalizar los embeddings en norma $L_2$, la similitud coseno, el producto punto y la distancia Euclidiana producen exactamente el <em>mismo ordenamiento</em>. Usa siempre producto punto por eficiencia.</li>
    <br>
    <li><strong>El Chunking supera al Modelo</strong><br>
    Una estrategia de partición limpia (recursiva, ~400 tokens, 10% overlap) tiene mayor impacto en el Recall final que saltar de un modelo de 768d a uno de 3072d.</li>
    <br>
    <li><strong>La Dimensión se Puede Optimizar</strong><br>
    Gracias a Matryoshka Representation Learning (MRL) y cuantización int8, puedes recortar hasta el 90% de la RAM del índice sin reentrenar y con pérdida mínima.</li>
    <br>
    <li><strong>Mide en tu Propio Test Set</strong><br>
    Ignora las tablas generales de MTEB. Construye un test set de 30 a 50 pares (consulta, chunk relevante) de tu propio dominio y monitorea Recall@k, MRR y nDCG.</li>
    <br>
    <li><strong>Calcula el Costo Antes de Indexar</strong><br>
    El costo de los tokens de indexación suele ser bajo; el verdadero costo reside en la memoria RAM del índice vectorial y los clusters HNSW.</li>
  </ol>
</div>

Note:
[02:15 - 02:20]
Pause for questions before explaining the hands-on lab instructions.

---

# Lab: Your Own Dataset
<span class="badge badge-orange">Hands-on Lab • 60 min</span>

> Construye, indexa, busca, evalúa y presupuesta un motor semántico completo sobre datos reales.

Note:
[02:20 - 02:25]
Explain the lab mechanics. Students can use their own CSV (support tickets, legal clauses, course syllabi, product reviews) or the provided fallback dataset.

---

## Lab Plan & Deliverable

| Paso | Actividad | Tiempo | Objetivo Práctico |
| :---: | :--- | :---: | :--- |
| **0** | **Setup & Carga de Datos** | 10 min | Instalar dependencias, cargar CSV propio y aislar la columna de texto. |
| **1** | **Conteo de Tokens & Presupuesto** | 10 min | Calcular distribución de tokens con `tiktoken` y proyectar costo e indexación. |
| **2** | **Embedding & Búsqueda Semántica** | 15 min | Codificar con `bge-small-en-v1.5`, indexar en NumPy y verificar queries. |
| **3** | **Construcción de Test Set & Métricas** | 15 min | Crear 15-30 consultas con ground truth y calcular Recall@5 y MRR. |
| **4** | **Experimentos & Cost Memo** | 10 min | Ejecutar truncamiento MRL o int8, y redactar el memo ejecutivo de 5 líneas. |

<div class="highlight-box">
  <strong>Entregable del Lab (Memo de 5 líneas):</strong>
  Cada equipo entregará un reporte cuantitativo con tamaño del corpus ($N$, tokens), mejor configuración, métricas alcanzadas (Recall@5, MRR), presupuesto proyectado y recomendación técnica (API vs. Self-host).
</div>

Note:
[02:25 - 02:30]
Reiterate the deliverable. It should be concise, professional, and ready to show to an engineering director.

--

## Step 0: Setup & Data Ingestion

Prepara tu entorno virtual de Python e instala los paquetes necesarios:

```bash
# Crear y activar entorno virtual
python -m venv .venv && source .venv/bin/activate

# Instalar librerías de embeddings y evaluación
pip install pandas numpy tiktoken sentence-transformers scikit-learn
```

Carga tu archivo de datos en memoria:

```python
import pandas as pd

# Cargar dataset (debe contener al menos una columna con texto no trivial)
df = pd.read_csv("my_data.csv")
text_col = "text"  # ajusta al nombre de tu columna

texts = df[text_col].dropna().astype(str).tolist()
print(f"Total documentos cargados: {len(texts):,}")
print(f"Muestra del primer documento:\n{texts[0][:200]}...")
```

Note:
[02:30 - 02:35]
Ensure everyone has pandas and sentence-transformers installed. If GPU is unavailable, bge-small runs quickly on CPU.

--

## Step 1: Token Profiling & Cost Estimation

```python [1-4|6-14|16-20]
import tiktoken

# Usamos el tokenizador estándar cl100k_base como referencia universal
tokenizer = tiktoken.get_encoding("cl100k_base")
token_counts = [len(tokenizer.encode(t)) for t in texts]

def estimate_infrastructure(n_tokens: int, n_vectors: int, price_per_M: float, 
                            dim: int, bytes_per_dim: int = 4):
    """Estima costos de tokens de embedding y consumo de RAM."""
    return {
        "total_tokens": n_tokens,
        "avg_tokens_per_doc": round(n_tokens / n_vectors, 1),
        "indexing_cost_usd": round((n_tokens / 1e6) * price_per_M, 4),
        "ram_footprint_mb": round((n_vectors * dim * bytes_per_dim) / (1024**2), 2),
    }

total_tokens = sum(token_counts)
presupuesto = estimate_infrastructure(
    n_tokens=total_tokens, n_vectors=len(texts),
    price_per_M=0.02, dim=1536, bytes_per_dim=4
)
print("Estimación preliminar de indexación:")
for k, v in presupuesto.items():
    print(f"  {k}: {v}")
```

<div class="ex-box">
  <strong>Tarea:</strong> Escala el cálculo al tamaño total previsto para tu sistema en producción (e.g. 1 millón de registros).
</div>

Note:
[02:35 - 02:40]
Help students adjust the parameters for their specific corpus size.

--

## Step 2: Local Vector Encoding & Search

Genera embeddings de alta calidad en tu máquina local de forma gratuita:

```python [1-6|8-12|14-19]
import time
import numpy as np
from sentence_transformers import SentenceTransformer

# Modelo compacto de alto rendimiento (384 dimensiones)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Generando embeddings normalizados en lote...")
t0 = time.time()
E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
elapsed = time.time() - t0
print(f"Matriz de embeddings calculada: {E.shape} en {elapsed:.2f}s ({len(texts)/elapsed:.1f} doc/s)")

def semantic_search(query: str, k: int = 5):
    """Codifica la query y ejecuta producto punto contra todo el índice."""
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = E @ q_vec  # Producto punto sobre vectores unitarios
    top_indices = np.argsort(-scores)[:k]
    return [(idx, texts[idx], float(scores[idx])) for idx in top_indices]

# Prueba manual con 3 consultas distintas
resultados = semantic_search("¿Cuál es el procedimiento de atención o soporte?")
for rank, (idx, txt, score) in enumerate(resultados, 1):
    print(f"[{rank}] Score: {score:.4f} | ID: {idx} | {txt[:120]}...")
```

Note:
[02:40 - 02:48]
Have each team run 3 exploratory queries. Ask: "Do the results make conceptual sense? Is there any false positive?"

--

## Step 3: Ground-Truth Test Set & Quantitative Evaluation

```python [1-6|8-18|20-22]
# Diseña de 15 a 30 pares: (consulta formulada por ti, lista de índices de filas relevantes)
test_set = [
    ("cómo solicitar reembolso o cancelación", [12, 45]),
    ("requisitos técnicos de hardware", [8]),
    ("política de privacidad y protección de datos", [33, 34, 102]),
]

def evaluate_retrieval(search_fn, tests, k: int = 5):
    recalls, rrs = [], []
    for query, relevant_indices in tests:
        rel_set = set(relevant_indices)
        top_k_indices = [item[0] for item in search_fn(query, k=k)]
        
        # Recall@k
        recalls.append(len(set(top_k_indices) & rel_set) / len(rel_set))
        # MRR
        rr = next((1.0 / rank for rank, idx in enumerate(top_k_indices, 1) if idx in rel_set), 0.0)
        rrs.append(rr)
        
    return {f"Recall@{k}": float(np.mean(recalls)), f"MRR@{k}": float(np.mean(rrs))}

metricas_base = evaluate_retrieval(semantic_search, test_set, k=5)
print("Resultados de la evaluación cuantitativa inicial:", metricas_base)
```

Note:
[02:48 - 02:55]
Students create their ground truth. Encourage them to include tricky queries with synonyms or colloquial terms.

--

## Step 4: Optimization Experiments (Choose Two)

Elige **dos** de los siguientes experimentos y registra el impacto en Recall@5, MRR, latencia y memoria:

<div class="container">
  <div class="col">
    <h4>Experimento A: Matryoshka Truncation</h4>
    <p>Trunca la matriz a dimensión reducida y re-normaliza:</p>
    <pre><code class="language-python">E_sub = E[:, :128]
E_trunc = E_sub / np.linalg.norm(E_sub, axis=1, keepdims=True)
# Evalúa el impacto de pasar de 384d a 128d</code></pre>
  </div>
  <div class="col">
    <h4>Experimento B: Cuantización int8</h4>
    <p>Simula cuantización de punto fijo:</p>
    <pre><code class="language-python"># Cuantización escalar a int8
scale = 127.0 / np.max(np.abs(E))
E_int8 = np.round(E * scale).astype(np.int8)
# Compara memoria y variación de ranking</code></pre>
  </div>
</div>

<div class="container" style="margin-top:15px;">
  <div class="col">
    <h4>Experimento C: Chunk Size (150 vs 500)</h4>
    <p>Re-particiona el texto con distintos tamaños y compara si las respuestas cortas capturan mejor la relevancia.</p>
  </div>
  <div class="col">
    <h4>Experimento D: Búsqueda Híbrida con BM25</h4>
    <p>Combina los puntajes de BM25 con los scores vectoriales usando Reciprocal Rank Fusion (RRF).</p>
  </div>
</div>

Note:
[02:55 - 03:00]
Circulate around the room assisting teams with their comparisons.

--

## Entregable: Executive Cost & Technical Memo

Redacta un memo de **5 líneas** listo para presentar a liderazgo técnico:

<div class="ex-box">
  <h4>Plantilla del Memo Ejecutivo (5 Líneas):</h4>
  <ol>
    <li><strong>Perfil del Corpus:</strong> $N = \dots$ chunks, promedio de $\bar{T} = \dots$ tokens por chunk (Total: $\dots$ tokens).</li>
    <li><strong>Mejor Configuración Evaluada:</strong> Modelo [Nombre], dimensión $d = \dots$, chunking [Estrategia].</li>
    <li><strong>Métricas Validadas:</strong> $\mathrm{Recall@}5 = \dots$, $\mathrm{MRR} = \dots$ sobre $M = \dots$ consultas de prueba.</li>
    <li><strong>Presupuesto Operativo:</strong> Costo one-off de indexación de $\$\dots$ USD y gasto mensual estimado de $\$\dots$ USD.</li>
    <li><strong>Recomendación Arquitectónica:</strong> [API Administrada / Self-Host en VPC] porque [Justificación de costo, latencia o soberanía].</li>
  </ol>
</div>

Note:
[03:00]
Wrap up the lab. Announce teams that achieved the highest Recall@5 and commend well-reasoned cost justifications.

---

## Questions & References

<div class="highlight-box">
  <h4>Recursos y Documentación Recomendada:</h4>
  <ul>
    <li><strong>Tutorial de referencia:</strong> <em>Text Embeddings in Python: Math, Pipelines, Metrics, Cost</em> (<a href="https://myengineeringpath.dev/genai-engineer/embeddings/" target="_blank">myengineeringpath.dev</a>).</li>
    <li><strong>Leaderboard MTEB:</strong> Massive Text Embedding Benchmark (<a href="https://huggingface.co/spaces/mteb/leaderboard" target="_blank">HuggingFace Spaces</a>).</li>
    <li><strong>Sentence-Transformers:</strong> Documentación oficial y repositorio (<a href="https://sbert.net" target="_blank">sbert.net</a>).</li>
    <li><strong>Matryoshka Representation Learning:</strong> Kusupati et al. (NeurIPS 2022).</li>
  </ul>
</div>

<p style="text-align: center; margin-top: 30px; font-family: 'Crimson Text', serif; font-size: 1.3em; color: var(--heading-color);">
  ¡Gracias por su participación! ¿Preguntas o reflexiones finales?
</p>

