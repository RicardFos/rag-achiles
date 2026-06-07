# Sistema RAG para Preguntas y Respuestas sobre Documentos
**Prueba Técnica Achilles**

He creado un sistema de RAG para responder preguntas sobre documentos PDF con citación de fuentes. Construido con una arquitectura orientada a objetos utilizando modelos Pydantic y seguridad de tipos en todo el código. Me he limitado a los requisitos y alcance del challenge técnico, debido al poco tiempo disponible para prepararlo. 

Para asegurar el uso sencillo y gratuito desde cualquier PC, he usado la librería sentence_transformers de HuggingFace para usar modelos de embeddings y reranker, ya que son modelos ligeros y gratuitos que se pueden ejecutar en local con buenos resultados. He usado FAISS (Meta, open source) como base de datos vectorial gratuita en local, creando la clase abstracta VectorStore para poder implementar otra base de datos vectorial si fuese necesario. También he optado por Gemini como proveedor LLM dado que facilita API keys gratuitas para testear sin necesidad de tarjeta de crédito. 

Como documentos PDF de prueba, he usado un conjunto de [documentos oficiales del Ayuntamiento de Madrid](https://www.madrid.es/portales/munimadrid/es/Inicio/Cultura-ocio-y-deporte/Publicaciones/Reuniones-del-grupo-motor-del-IV-Plan-de-Gobierno-Abierto-del-Ayuntamiento-de-Madrid/?vgnextfmt=default&vgnextoid=968e725075c7a910VgnVCM200000f921e388RCRD&vgnextchannel=f9e2f073808fe410VgnVCM2000000c205a0aRCRD), dado que son públicos, en formato PDF parseable (no hace falta OCR) y trata temas específicos que una LLM difícilmente pueda responder sin ayuda de un RAG.

Los documentos son parseados, limpiados y chunkeados de forma recursiva siguiendo una jerarquía de párrafo->linea->palábra->letra. Los metadatos de los chunks se guardan en un archivo .pkl junto al vector store, y se recuperan como objetos pydantic con toda la información del chunk incluida.
Se usa cosine similarity con un threshold por defecto de 0.6 entre los embeddings del input del usuario y los embeddings de los chunks para obtener los top 20, y luego se usa el reranker para elegir los 7 mejores matches de esos 20.

He creado 3 notebooks que deben ser ejecutadas en orden. 
La primera muestra el proceso de indexación, cargando los documentos, separándolos en chunks, generando los embeddings y guardándolo en la base de datos vectorial. 
La segunda notebook muestra el proceso de retrieval con inputs de usuario (sin LLM), mostrando y comparando resultados de cosine similarity y del reranker.
La última notebook muestra el proceso de pregunta-retrieval-respuesta usando LLM (Gemini) y muestra resultados del set de evaluación.

Todo el código y notebooks han sido escritos en inglés (mi idioma principal para programar), aunque los documentos y system prompt del asistente LLM están escritos en castellano para este ejemplo en concreto. He usado el asistente de código Claude Code para ayudarme a implementar mis requisitos, usando spec-driven-development, escribiendo requisitos claros, pasos y estructura del código. Las notebooks se han aprovechado como testeo de los métodos y clases además de mostrar claramente cada paso del proceso RAG.

Para futuras iteraciones y mejoras, se podría implementar un backend local con FastAPI y un pequeño front end para evitar cargar las librerias, base de datos y modelos de embeddings + rerank. Se podría añadir también un sistema de retrieval híbrido, usando keywords además de similitud semántica, y opciones como recuperar los X chunks anteriores y posteriores a los elegidos para ampliar la información.

También se podría ampliar el alcance del asistente LLM con más agentes y memoria de contexto para convertirlo en un chat real. También se debería añadir un sistema de logging para un sistema en producción, y mantener una base de datos sobre los procesos de RAG y su estado.

Aunque el RAG es una herramienta muy útil, requiere mucho trabajo de "fine tuning", evaluación y mantenimiento en caso de actualización y ampliación de documentos. Con modelos y sistemas multi-agentes modernos, existen alternativas como RAG agénticos que centran el proceso de retrieval en las decisiones de las propias LLM, por ejemplo, dándo al agente retriever un resúmen e índice del documento para que decida qué documento y qué páginas cargar de forma autónoma, aunque es un proceso potencialmente más lento y caro (por uso de tokens) que un RAG vectorial tradicional.


## Arquitectura

```
┌─────────────┐
│   PDFs      │
└──────┬──────┘
       │ Parser (PyPDF2)
       ↓
┌─────────────────────┐
│ Documentos Chunked  │ (recursivo, 512 tokens, 50 overlap)
└──────┬──────────────┘
       │ Embeddings (paraphrase-multilingual-MiniLM-L12-v2)
       ↓
┌──────────────────────┐
│  FAISS Vector Store  │ (~620 chunks, IndexFlatIP)
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
│  Recuperación Etapa 2│ Cross-encoder: Re-ranking a Top-7
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


# instalar uv para instalación de dependencias rápida
pip install uv 
# Instalar dependencias
uv pip install -r requirements.txt
```

### 2. Configurar API Key de Gemini

Obtener una API key gratuita desde [Google AI Studio](https://aistudio.google.com/app/apikey)


```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar .env y añadir tu API key
GOOGLE_API_KEY='tu-api-key-aqui'
```

### 3. Indexar Documentos

```bash
python index_documents.py ./pdfs
```

- Parsea todos los PDFs en el directorio `pdfs/`
- Genera embeddings para los chunks de documentos
- Crea y guarda un índice vectorial FAISS

**Salida:**
```
📁 Found 9 PDF files in pdfs/
✓ Parsed 9 documents
✓ Generated ~620 chunks
✓ Generated ~620 embeddings
✓ Saved index with ~620 vectors
```

### 4. Consultar el Sistema

```bash
python query_rag.py "¿Qué es la Escuela de Gobierno Abierto y cuál es su objetivo?"
```

## 📋 Especificaciones Técnicas

Documentación detallada de diseño y decisiones técnicas para cada componente:

- **[Ingesta y Chunking](specs/ingestion_spec.md)** - Extracción de PDFs, chunking recursivo, y preservación de metadatos
- **[Embeddings](specs/embedding_spec.md)** - Modelos multilingües, generación de vectores, y configuración
- **[Vector Database](specs/vectordb_spec.md)** - FAISS IndexFlatIP, búsqueda por similitud coseno, y persistencia
- **[Re-ranker](specs/reranker_spec.md)** - Cross-encoder para re-ranking en dos etapas y mejora de relevancia
- **[Generación LLM](specs/llm_spec.md)** - Integración con Gemini, ingeniería de prompts, y extracción de citas
- **[Evaluación](specs/evaluation_spec.md)** - Métricas, conjunto de prueba, y framework de evaluación

Estas especificaciones documentan las decisiones de diseño, contratos de API, y detalles de implementación para cada componente del sistema RAG. Han sido generadas con mis instrucciones precisas de requisitos de desarrollos y ampliadas durante el avance del proyecto. Están escritas en inglés para facilitar el uso de agentes de código como Claude Code.

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
- Evaluación sobre conjunto de prueba (eval.jsonl)
- Cómputo de métricas de calidad
- Inspección detallada de resultados

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

### 📊 Resultados

Evaluación sobre 15 preguntas del conjunto de prueba (`eval.jsonl`):

| Métrica | Resultado |
|---------|-----------|
| **Precisión de Citas** | 93.3% (14/15 respuestas con citas) |
| **Precisión de Fuentes** | 76.7% |
| **Recall de Fuentes** | 80.0% |
| **Similitud Semántica** | 71.8% |
| **Chunks Recuperados (promedio)** | 4.5 |

✅ **Tasa de éxito: 87%** (13/15 preguntas respondidas correctamente)

El sistema cita fuentes consistentemente y genera respuestas fundamentadas en documentos. Las 2 preguntas que no se responden correctamente representan casos límite realistas:

- **Pregunta 5** (995 personas): Dificultad para distinguir entre múltiples consultas públicas similares mencionadas en diferentes documentos
- **Pregunta 9** (THIVIC): Término poco frecuente cuya definición no aparece consistentemente en los top-20 candidatos de recuperación

Estos casos representan áreas de mejora futura y demuestran la evaluación honesta del sistema en escenarios del mundo real.

Ver análisis completo en [03_llm_generation_and_eval.ipynb](03_llm_generation_and_eval.ipynb).

