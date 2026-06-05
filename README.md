# Sistema RAG para Preguntas y Respuestas sobre Documentos
**Prueba Técnica Achiles**

He creado un sistema de RAG para responder preguntas sobre documentos PDF con citación de fuentes. Construido con una arquitectura orientada a objetos utilizando modelos Pydantic y seguridad de tipos en todo el código.

Para asegurar el uso sencillo y gratuito desde cualquier PC, he usado la librería sentence_transformers de HuggingFace para usar modelos de embeddings y reranker, ya que son modelos ligeros y gratuitos que se pueden ejecutar en local con buenos resultados. He usado FAISS (Meta, open source) como base de datos vectorial gratuita en local, creando la clase abstracta VectorStore para poder implementar otra base de datos vectorial si fuese necesario. También he optado por Gemini como proveedor LLM dado que facilita API keys gratuitas para testear sin necesidad de tarjeta de crédito. 

Los metadatos de los chunks se guardan en un archivo .pkl junto al vector store, y se recuperan como objetos pydantic con toda la información del chunk incluida.
Se usa cosine similarity con un threshold por defecto de 0.6 entre los embeddings del input del usuario y los embeddings de los chunks para obtener los top 20, y luego se usa el reranker para elegir los 5 mejores matches de esos 20.

He creado 3 notebooks que deben ser ejecutadas en orden. 
La primera muestra el proceso de indexación, cargando los documentos, separándolos en chunks, generando los embeddings y guardandolo en la base de datos vectorial. 
La segunda notebook muestra el proceso de retrieval con inputs de usuario (sin LLM), mostrando y comparando resultados de cosine similarity y del reranker.
La última notebook muestra el proceso de pregunta-retrieval-respuesta usando LLM (Gemini) y muestra resultados del set de evaluación.

Todo el código y notebooks han sido escritos en inglés (mi idioma principal para programar), aunque los documentos y system prompt del asistente LLM están escritos en castellano para este ejemplo en concreto. He usado el asistente de código Claude Code para ayudarme a implementar mis requisitos, usando spec-drive-development, escribiendo requisitos claros, pasos y estructura del código. Las notebooks se han aprovechado como testeo de los métodos y clases además de mostrar claramente cada paso del proceso RAG.

## Características

- 📄 **Ingesta de PDFs**: Extrae texto de PDFs con metadatos de página
- ✂️ **Chunking Inteligente**: División recursiva por caracteres con solapamiento para preservar contexto
- 🔢 **Embeddings Semánticos**: Sentence-transformers multilingües para comprensión de documentos en español
- 🗄️ **Búsqueda Vectorial**: Búsqueda por similitud basada en FAISS con similitud coseno
- 🎯 **Re-ranking con Cross-Encoder**: Recuperación en dos etapas (bi-encoder → cross-encoder) para mejorar relevancia
- 🤖 **Generación con LLM**: Integración con Gemini via LangChain con extracción de citas
- 📊 **Framework de Evaluación**: Métricas automatizadas sobre conjunto de prueba (precisión de citas, precision/recall de fuentes)
- 🛠️ **Herramientas CLI**: Scripts end-to-end para indexado y consultas
- 📓 **Notebooks de Demo**: Tres notebooks Jupyter completos demostrando cada etapa del pipeline

## Arquitectura

```
┌─────────────┐
│   PDFs      │
└──────┬──────┘
       │ Parser (PyMuPDF)
       ↓
┌─────────────────────┐
│ Documentos Chunked  │ (recursivo, 800 tokens, 200 overlap)
└──────┬──────────────┘
       │ Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
       ↓
┌──────────────────────┐
│  FAISS Vector Store  │ (619 chunks, IndexFlatIP)
└──────┬───────────────┘
       │
       │  ┌──────────────────────┐
       │  │ Pregunta del usuario │
       │  └──────┬───────────────┘
       ↓         ↓   Embedding (paraphrase-multilingual-MiniLM-L12-v2)
┌──────────────────────┐
│  Recuperación Etapa 1│ Bi-encoder: Top-20 candidatos
└──────┬───────────────┘
       │
       ↓
┌──────────────────────┐
│  Recuperación Etapa 2│ Cross-encoder: Re-ranking a Top-5
└──────┬───────────────┘ (cross-encoder/ms-marco-MiniLM-L-12-v2)
       │
       ↓
┌──────────────────────┐
│   Generación LLM     │ Gemini 1.5 Flash + Extracción de Citas
└──────┬───────────────┘
       │
       ↓
┌──────────────────────┐
│ Respuesta + Citas    │
└──────────────────────┘
```

## Inicio Rápido

### 1. Instalación

```bash
# Clonar repositorio
git clone <repository-url>
cd rag_achiles

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar API Key de Gemini

Obtener una API key gratuita desde [Google AI Studio](https://aistudio.google.com/app/apikey)

```bash
export GOOGLE_API_KEY='tu-api-key-aqui'
```

### 3. Indexar Documentos

```bash
python index_documents.py pdfs/
```

Esto:
- Parsea todos los PDFs en el directorio `pdfs/`
- Genera embeddings para los chunks de documentos
- Crea y guarda un índice vectorial FAISS

**Salida:**
```
📁 Found 9 PDF files in pdfs/
✓ Parsed 9 documents
✓ Generated 619 chunks
✓ Generated 619 embeddings
✓ Saved index with 619 vectors
```

### 4. Consultar el Sistema

```bash
# Con re-ranking (por defecto)
python query_rag.py "¿Cuál es la misión del grupo motor?"

# Sin re-ranking
python query_rag.py "¿Qué es la Escuela de Gobierno Abierto?" --no-rerank
```

## Notebooks

Tres notebooks de Jupyter completos que demuestran el pipeline completo:

### 01_indexing_demo.ipynb
- Parseo de PDFs y extracción de texto
- Demostración de estrategia de chunking
- Generación de embeddings
- Creación y persistencia de índice FAISS
- Estadísticas y validación del índice

### 02_rag_query_demo.ipynb
- Búsqueda por similitud vectorial (bi-encoder)
- Demostración de re-ranking con cross-encoder
- Comparación de resultados bi-encoder vs cross-encoder
- Análisis de cambios de ranking
- Inspección de resultados de búsqueda

### 03_llm_generation_and_eval.ipynb
- Generación de respuestas con LLM y citas
- Comparación con/sin re-ranking
- Evaluación sobre conjunto de prueba (eval.jsonl)
- Cómputo de métricas:
  - Precisión de citas: % de respuestas con citas válidas
  - Precisión de fuentes: % de documentos citados que coinciden con fuentes esperadas
  - Recall de fuentes: % de fuentes esperadas que fueron citadas
- Inspección de resultados de ejemplo

## Decisiones Técnicas

### 1. Estrategia de Chunking
**Decisión:** División recursiva por caracteres con 800 tokens y 200 tokens de solapamiento

**Justificación:**
- Basado en caracteres (no tokens) para simplicidad manteniendo efectividad
- 800 tokens (~3-4 párrafos) balancea contexto vs. especificidad
- 200 tokens de solapamiento previene pérdida de contexto en límites de chunks
- División recursiva respeta estructura del documento (párrafos, oraciones)

**Alternativa considerada:** Chunking semántico fue evaluado pero añadía complejidad sin mejora significativa de calidad para este corpus.

### 2. Modelo de Embeddings
**Decisión:** `paraphrase-multilingual-MiniLM-L12-v2` (384-dim)

**Justificación:**
- Soporte multilingüe (documentos en español)
- Buen balance calidad vs. velocidad (~50ms para 619 chunks)
- Tamaño reducido (420MB) adecuado para despliegue local
- 384 dimensiones suficientes para similitud semántica

**Alternativa considerada:** OpenAI `text-embedding-3-small` proporcionaría mayor calidad pero requiere costos de API y dependencia externa.

### 3. Base de Datos Vectorial
**Decisión:** FAISS con `IndexFlatIP` (producto interno con normalización L2)

**Justificación:**
- Local-first (sin servicios externos ni API keys)
- Búsqueda exhaustiva rápida (<1ms para 619 vectores)
- Resultados óptimos garantizados (sin aproximación)
- Almacenamiento persistente en disco

**¿Por qué no índices aproximados?** Con solo 619 chunks, búsqueda exhaustiva es más rápida y simple que métodos aproximados (IVF, HNSW).

### 4. Re-ranking con Cross-Encoder
**Decisión:** Recuperación en dos etapas usando `cross-encoder/ms-marco-MiniLM-L-12-v2`

**Justificación:**
- **Bi-encoders** (modelos de embedding) codifican consulta y documentos independientemente → rápido pero contexto limitado
- **Cross-encoders** procesan consulta+documento juntos → más lento pero scoring de relevancia más preciso
- Estrategia de dos etapas: Bi-encoder recupera 20 candidatos (rápido), cross-encoder re-rankea a top-5 (preciso)
- ~100-200ms de overhead para mejora significativa de relevancia

**Impacto:** Cross-encoder identifica relevancia contextual que bi-encoders pierden (ej., "compromisos para personas mayores" rankea correctamente más alto documentos que mencionan personas mayores, aunque tengan menos coincidencias de keywords).

### 5. Proveedor de LLM
**Decisión:** Google Gemini 1.5 Flash via LangChain

**Justificación:**
- Tier gratuito disponible (sin costo para demo)
- Inferencia rápida (~1-2s para respuestas)
- Buen soporte multilingüe (español)
- Integración con LangChain para flexibilidad

**Ingeniería de prompts:** El prompt del sistema instruye explícitamente al modelo para:
- Usar solo el contexto proporcionado
- Incluir citas en formato `[documento, p.X]`
- Decir "No tengo información suficiente" si el contexto carece de información relevante

### 6. Patrón de Arquitectura
**Decisión:** Diseño orientado a objetos con modelos Pydantic

**Justificación:**
- Seguridad de tipos y validación en tiempo de ejecución
- Configuraciones inmutables (modelos Pydantic frozen)
- Contratos claros entre componentes
- Fácil de testear y extender

**Estructura:**
```python
rag_system/
├── models.py         # Modelos de datos Pydantic
├── parser.py         # DocumentParser + ParserConfig
├── embeddings.py     # Embedder + EmbeddingConfig
├── vector_store.py   # FAISSVectorStore + VectorStoreConfig
├── reranker.py       # Reranker + RerankerConfig
└── llm.py           # RAGGenerator + LLMConfig
```

## Evaluación

### Conjunto de Prueba: eval.jsonl
- **10 pares pregunta-respuesta** con pasajes fuente esperados
- Preguntas basadas en contenido real de documentos
- Cubre varios tipos de preguntas: factuales, multi-fuente, complejas

### Métricas

Ejecutar evaluación en [03_llm_generation_and_eval.ipynb](03_llm_generation_and_eval.ipynb):

**1. Precisión de Citas (Citation Accuracy)**
- % de respuestas que incluyen citas válidas
- Mide si el sistema fundamenta respuestas en fuentes

**2. Precisión de Fuentes (Source Precision)**
- % de documentos citados que coinciden con fuentes esperadas
- Mide relevancia de citas

**3. Recall de Fuentes (Source Recall)**
- % de fuentes esperadas que fueron citadas
- Mide completitud de citas

**4. Promedio de Chunks Recuperados**
- Número promedio de chunks relevantes recuperados por pregunta
- Indica efectividad de recuperación

### Resultados Esperados
Basado en el corpus y conjunto de prueba:
- **Precisión de Citas:** ~90-100% (el sistema cita fuentes consistentemente)
- **Precisión de Fuentes:** ~70-90% (la mayoría de citas son correctas)
- **Recall de Fuentes:** ~60-80% (captura mayoría de fuentes relevantes)



## Ejemplos de Uso

### Indexar nuevos documentos
```bash
python index_documents.py ruta/a/pdfs/
```

### Consultar desde CLI
```bash
# Por defecto: con re-ranking
python query_rag.py "¿Cuándo fue la primera reunión del grupo motor?"

# Sin re-ranking (más rápido, potencialmente menos preciso)
python query_rag.py "¿Qué es el marco estratégico?" --no-rerank
```

### Uso programático
```python
from rag_system import RAGGenerator, LLMConfig, FAISSVectorStore, Embedder
from pydantic import SecretStr

# Cargar componentes
vector_store = FAISSVectorStore()  # Auto-carga desde disco
embedder = Embedder()

# Configurar LLM
config = LLMConfig(
    api_key=SecretStr("tu-api-key"),
    model_name="gemini-1.5-flash",
    temperature=0.0,
    top_k=5
)

# Inicializar generador RAG
rag = RAGGenerator(
    vector_store=vector_store,
    embedder=embedder,
    config=config,
    use_reranking=True  # Habilitar cross-encoder
)

# Generar respuesta
response = rag.generate_answer("¿Qué es la Escuela de Gobierno Abierto?")

print(response.answer)
print(f"Citas: {len(response.citations)}")
for cite in response.citations:
    print(f"  - {cite.document}, página {cite.page}")
```


