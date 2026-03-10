# 📄 Analizador de Contratos SaaS

Sistema multiagente para el análisis automático de contratos SaaS. Extrae **Derechos**, **Obligaciones** y **Prohibiciones** de documentos PDF usando una arquitectura de agentes especializados que se comunican mediante el protocolo **A2A (Agent-to-Agent)**.

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend (Gradio)                   │
│                     localhost:7860                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Agente Orquestador (Google ADK)            │
│              LiteLLM → OpenRouter                       │
│              Modelo: gemini-2.5-flash-lite              │
└──────────────┬──────────────────────┬───────────────────┘
               │ A2A                  │ A2A
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  Agente Almacenador  │  │      Agente Analizador        │
│  localhost:8001      │  │      localhost:8002           │
│  Google ADK          │  │      CrewAI (3 sub-agentes)   │
│  Modelo: llama-4     │  │      Modelo: grok-4.1-fast    │
└──────────┬───────────┘  └──────────────┬────────────────┘
           │                             │ (solo lectura)
           ▼                             ▼
┌─────────────────────────────────────────────────────────┐
│                   Qdrant (Docker)                       │
│                   localhost:6333                        │
│   Colección documentos │ Colección análisis             │
│   Embeddings: all-MiniLM-L6-v2 (384 dimensiones)       │
└─────────────────────────────────────────────────────────┘
```

### Responsabilidades por agente

| Agente | Puerto | Framework | Modelo | Rol |
|---|---|---|---|---|
| **Orquestador** | — | Google ADK | gemini-2.5-flash-lite | Enruta mensajes al agente correcto |
| **Almacenador** | 8001 | Google ADK | llama-4-maverick | Extrae texto PDF, fragmenta semánticamente y almacena en Qdrant |
| **Analizador** | 8002 | CrewAI | grok-4.1-fast | Recupera chunks de Qdrant y analiza derechos, obligaciones y prohibiciones |
| **Frontend** | 7860 | Gradio | — | Interfaz web para el usuario |

---

## 📁 Estructura del proyecto

```
saas-contract-analyzer/
├── .venv/                        # Entorno virtual Python
├── almacenador_agent/
│   ├── agent.py                  # Configuración del agente y Runner ADK
│   ├── agent_executor.py         # Executor A2A — lógica de operaciones
│   ├── qdrant_storage.py         # Escritura/lectura en Qdrant
│   ├── tools_agent.py            # PDFProcessor, chunking semántico, ResponseFormatter
│   └── main.py                   # Servidor A2A en puerto 8001
├── analisador_agent/
│   ├── agent.py                  # Crew de 3 agentes CrewAI
│   ├── agent_executor.py         # Executor A2A — orquesta CrewAI + Qdrant
│   ├── qdrant_retriever.py       # Solo lectura desde Qdrant
│   └── main.py                   # Servidor A2A en puerto 8002
├── orquestador_agent/
│   └── orquestador/
│       └── agent.py              # LlmAgent con sub-agentes remotos A2A
├── Frontend/
│   └── gradio_app.py             # Interfaz Gradio en puerto 7860
├── Test/
│   ├── conftest.py               # Ajuste de sys.path para pytest
│   ├── test_storage.py           # 80 tests unitarios
│   ├── test_integracion_agentes.py # 17 tests de integración A2A
│   └── test_casos_extremos.py    # 18 tests de casos extremos
├── requirements.txt
├── metrics.csv                   # Métricas de rendimiento por operación
└── .env                          # Variables de entorno (no subir a Git)
```

---

## ⚙️ Requisitos previos

- Python 3.10 o superior
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y corriendo
- Cuenta en [OpenRouter](https://openrouter.ai/) con API key activa

---

## 🚀 Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/saas-contract-analyzer.git
cd saas-contract-analyzer
```

### 2. Crear y activar el entorno virtual

**Crear el entorno virtual en la raíz del proyecto:**

```bash
python3 -m venv .venv
```

**Activar en Windows (CMD):**

```cmd
.venv\Scripts\activate.bat
```

**Activar en Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Activar en macOS / Linux:**

```bash
source .venv/bin/activate
```

Sabrás que está activo porque el prompt mostrará `(.venv)` al inicio.

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto para cada agente (ver sección [Variables de entorno](#-variables-de-entorno)).

### 5. Iniciar Qdrant con Docker

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

> El flag `-v` agrega persistencia. Sin él, los datos se pierden al detener el contenedor.

---

## 🔑 Variables de entorno

Cada agente lee su propio archivo `.env`. Crea uno en cada carpeta correspondiente.

### `orquestador_agent/.env`

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

### `almacenador_agent/.env`

```env
OPENROUTER_API_KEY=sk-or-v1-...

QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=contratos-saas
```

### `analisador_agent/.env`

```env
OPENROUTER_API_KEY=sk-or-v1-...

QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=contratos-saas
```

> **Nota:** Los tres agentes usan la misma instancia de Qdrant y la misma colección. El agente almacenador escribe; el analizador solo lee.

---

## ▶️ Uso

Cada agente se inicia en una terminal separada con el entorno virtual activo.

### Terminal 1 — Agente Almacenador

```bash
cd almacenador_agent
python main.py
```

Servidor disponible en `http://localhost:8001`

Agent Card visible en la ruta `http://localhost:8001/.well-known/agent-card.json`

### Terminal 2 — Agente Analizador

```bash
cd analisador_agent
python main.py
```

Servidor disponible en `http://localhost:8002`

Agent Card visible en la ruta `http://localhost:8002/.well-known/agent-card.json`

### Terminal 3 — Frontend Gradio

```bash
cd Frontend
python gradio_app.py
```

Interfaz disponible en `http://localhost:7860`

> **Orden importante:** Inicia primero el almacenador y el analizador antes del frontend, ya que el orquestador intenta conectarse a ambos al arrancar.

---

## 💬 Ejemplos de uso

Una vez que la interfaz esté abierta en `http://localhost:7860`:

**1. Ver instrucciones:**
```
hola
```

**2. Almacenar un contrato PDF:**
```
Adjunta el PDF → escribe:
Almacena el documento con el nombre contrato_servicios.
```

**3. Analizar el contrato almacenado:**
```
Analiza el documento llamado contrato_servicios.pdf
```

**4. Ver el análisis guardado:**
```
Ver el análisis del documento contrato_servicios.pdf
```

**5. Consultar documentos almacenados:**
```
¿Cuántos documentos están almacenados?
```

**6. Consultar análisis almacenados:**
```
¿Cuántos análisis están almacenados?
```

**7. Ver qué documentos tienen análisis:**
```
¿Qué documentos tienen análisis?
```

---

## 📦 Dependencias principales

| Paquete | Versión mínima | Usado en |
|---|---|---|
| `google-adk` | — | Orquestador, Almacenador, Frontend |
| `google-adk[a2a]` | — | Orquestador, Almacenador |
| `google-adk[extensions]` | — | Frontend |
| `crewai[tools]` | — | Analizador |
| `litellm` | — | Orquestador, Almacenador |
| `python-a2a` | — | Almacenador, Analizador |
| `qdrant-client` | — | Almacenador, Analizador |
| `sentence-transformers` | — | Almacenador |
| `nltk` | — | Almacenador |
| `PyPDF2` | — | Almacenador |
| `gradio` | — | Frontend |
| `uvicorn` | — | Almacenador, Analizador |
| `python-dotenv` | — | Todos |

---

## 🧪 Tests

El proyecto incluye **115 tests** que cubren módulos unitarios, flujos de integración entre agentes y manejo de casos extremos.

### Configuración previa

**1. Instalar pytest** (con el entorno virtual activo):
```bash
pip install pytest
```

**2. Crear el archivo `Test/conftest.py`** en la carpeta `Test/`:
```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

> Este archivo es obligatorio. Sin él, Python no encuentra los módulos del proyecto y los tests fallan con `ModuleNotFoundError`.

### Ejecutar los tests

```bash
# Tests unitarios
pytest Test/test_funcionalidades.py -v

# Tests de integración
pytest Test/test_integracion_agentes.py -v

# Tests de casos extremos
pytest Test/test_casos_extremos.py -v

# Todos los tests a la vez
pytest Test/ -v
```

### Resultado esperado

```
115 passed, 3 warnings in ~70s
```

> Los tests usan mocks — no requieren Qdrant corriendo ni conexión a internet.

---

## 📊 Métricas

El sistema registra automáticamente cada operación en `metrics.csv` en la raíz del proyecto:

| Campo | Descripción |
|---|---|
| `timestamp` | Fecha y hora de la operación |
| `agente` | Agente que ejecutó la operación |
| `operacion` | Tipo de operación realizada |
| `documento` | Nombre o ID del documento procesado |
| `tiempo_s` | Tiempo de respuesta en segundos |
| `status` | `success`, `error`, `not_found`, etc. |

---

## 📄 Licencia

Este proyecto está bajo la licencia incluida en el archivo [LICENSE](LICENSE).