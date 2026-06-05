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

## 🎯 Puntos Clave

- ✅ **93% de precisión en citación** - respuestas fundamentadas en fuentes
- 🔄 **Re-ranking en dos etapas** - bi-encoder + cross-encoder mejora relevancia
- 🌍 **Soporte español** - embeddings multilingües optimizados
- 🆓 **Completamente gratuito** - sin API keys de pago requeridas
- 📦 **Todo local** - embeddings, vector DB y reranker en tu máquina

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
│ Documentos Chunked  │ (recursivo, 768 tokens, 150 overlap)
└──────┬──────────────┘
       │ Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
       ↓
┌──────────────────────┐
│  FAISS Vector Store  │ (~500 chunks, IndexFlatIP)
└──────┬───────────────┘
       │
       │  ┌──────────────────────┐
       │  │ Pregunta del usuario │
       │  └──────┬───────────────┘
       ↓         ↓   Embedding (paraphrase-multilingual-MiniLM-L12-v2)
┌──────────────────────┐
│  Recuperación Etapa 1│ Bi-encoder: Top-20 candidatos
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│  Recuperación Etapa 2│ Cross-encoder: Re-ranking a Top-5
└──────────┬───────────┘ (cross-encoder/ms-marco-MiniLM-L-12-v2)
           │
           ↓
┌──────────────────────┐
│   Generación LLM     │ Gemini 1.5 Flash + Extracción de Citas
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│ Respuesta + Citas    │
└──────────────────────┘
```

## Inicio Rápido

### Requisitos
- **Python 3.11 o superior** (testeado con Python 3.11.9)

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

**Opción 1: Archivo .env (Recomendado)**
```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar .env y añadir tu API key
GOOGLE_API_KEY=tu-api-key-aqui
```

**Opción 2: Variable de entorno**

Linux/Mac:
```bash
export GOOGLE_API_KEY='tu-api-key-aqui'
```

Windows PowerShell:
```powershell
$env:GOOGLE_API_KEY='tu-api-key-aqui'
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
✓ Generated ~500 chunks
✓ Generated ~500 embeddings
✓ Saved index with ~500 vectors
```

### 4. Consultar el Sistema

```bash
# Con re-ranking (por defecto)
python query_rag.py "¿Cuál es la misión del grupo motor?"

# Sin re-ranking
python query_rag.py "¿Qué es la Escuela de Gobierno Abierto?" --no-rerank
```

## 📋 Especificaciones Técnicas

Documentación detallada de diseño y decisiones técnicas para cada componente:

- **[Ingesta y Chunking](specs/ingestion_spec.md)** - Extracción de PDFs, chunking recursivo, y preservación de metadatos
- **[Embeddings](specs/embedding_spec.md)** - Modelos multilingües, generación de vectores, y configuración
- **[Vector Database](specs/vectordb_spec.md)** - FAISS IndexFlatIP, búsqueda por similitud coseno, y persistencia
- **[Re-ranker](specs/reranker_spec.md)** - Cross-encoder para re-ranking en dos etapas y mejora de relevancia
- **[Generación LLM](specs/llm_spec.md)** - Integración con Gemini, ingeniería de prompts, y extracción de citas
- **[Evaluación](specs/evaluation_spec.md)** - Métricas, conjunto de prueba, y framework de evaluación

Estas especificaciones documentan las decisiones de diseño, contratos de API, y detalles de implementación para cada componente del sistema RAG.

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
- Cómputo de métricas de calidad
- Inspección detallada de resultados

## 📊 Resultados de Evaluación

Evaluación sobre 15 preguntas del conjunto de prueba (`eval.jsonl`):

| Métrica | Resultado |
|---------|-----------|
| **Precisión de Citas** | 93.3% (14/15 respuestas con citas) |
| **Precisión de Fuentes** | 63.3% |
| **Recall de Fuentes** | 63.3% |
| **Similitud Semántica** | 66.0% |
| **Chunks Recuperados (promedio)** | 3.6 |

✅ El sistema cita fuentes consistentemente y genera respuestas fundamentadas en documentos.

Ver análisis completo en [03_llm_generation_and_eval.ipynb](03_llm_generation_and_eval.ipynb).

## Evaluación

### Conjunto de Prueba
- **15 pares pregunta-respuesta** con fuentes esperadas
- Preguntas basadas en contenido real de documentos
- Tipos: factuales, definiciones, listas, síntesis multi-fuente

### Métricas Computadas
1. **Precisión de Citas**: % de respuestas con citas válidas
2. **Precisión de Fuentes**: % de citas que coinciden con fuentes esperadas  
3. **Recall de Fuentes**: % de fuentes esperadas citadas
4. **Similitud Semántica**: Similitud coseno entre respuestas generadas y esperadas
5. **Chunks Recuperados**: Efectividad de recuperación

Ver resultados detallados arriba y en el notebook de evaluación.



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
    model_name="gemini-2.0-flash-001",
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


