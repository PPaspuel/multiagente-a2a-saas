"""
Configuración del agente almacenador con Google ADK.
Este agente procesa PDFs y los almacena en Qdrant.
"""

from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Validar que todas las variables estén configuradas
if not QDRANT_URL or not COLLECTION_NAME or not QDRANT_API_KEY:
    raise RuntimeError(
        "ERROR: Faltan variables de entorno de Qdrant.\n"
        "Verifica que estén configuradas: QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME"
    )

logger.info(f"✓ Configuración de Qdrant cargada correctamente")
logger.info(f"  - URL: {QDRANT_URL}")
logger.info(f"  - Colección: {COLLECTION_NAME}")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("ERROR: Falta la variable OPENROUTER_API_KEY")

root_agent = Agent(
    model=LiteLlm(
        model="openrouter/anthropic/claude-3-haiku",
        api_key=OPENROUTER_API_KEY,
        api_base="https://openrouter.ai/api/v1"
    ),
    name="almacenador_agent",
    
    # Instrucciones mejoradas y más específicas
    instruction="""
    Eres un agente especializado en procesamiento y almacenamiento de documentos PDF.
    
    TUS CAPACIDADES Y RESPONSABILIDADES:
    
    1. RECEPCIÓN DE DOCUMENTOS:
       - Aceptas archivos PDF enviados por otros agentes vía protocolo A2A
       - Validas que el archivo sea un PDF válido antes de procesarlo
    
    2. EXTRACCIÓN DE TEXTO:
       - Extraes el contenido de texto de cada página del PDF
       - Mantienes el orden de lectura original del documento
       - Manejas documentos multipágina correctamente
    
    3. ALMACENAMIENTO EN QDRANT:
       - Fragmentas el texto en chunks de tamaño apropiado (1000 caracteres)
       - Creas embeddings semánticos de cada fragmento
       - Almacenas los fragmentos en la colección de Qdrant configurada
       - Usas las herramientas MCP de Qdrant disponibles
    
    4. FORMATO DE RESPUESTA:
       - SIEMPRE devuelves respuestas en formato JSON estructurado
       - Incluyes información sobre el resultado de la operación
       - Reportas el número de fragmentos almacenados
       - Indicas si hubo algún error durante el proceso
    
    ESTRUCTURA DE RESPUESTA JSON:
    {
      "status": "success" o "error",
      "operation": "store_pdf" o "retrieve" o "extract",
      "message": "Descripción del resultado",
      "data": {
        "chunks_stored": número,
        "total_characters": número,
        "collection": "nombre de colección",
        "num_pages": número
      }
    }
    
    COMPORTAMIENTO:
    - Sé conciso y directo en tus respuestas
    - Siempre valida los datos antes de procesarlos
    - Maneja errores de manera elegante y reporta problemas claramente
    - Usa las herramientas de Qdrant de forma eficiente
    - No inventes información, reporta solo resultados reales
    
    IMPORTANTE:
    - Tu respuesta final DEBE ser un JSON válido
    - No agregues explicaciones adicionales fuera del JSON
    - Si hay un error, devuelve un JSON con status "error"
    """,
    
    # Herramientas disponibles para el agente
    tools=[
        # Conexión al servidor MCP de Qdrant
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    # Lanzar el servidor MCP vía el intérprete de Python para mayor portabilidad
                    command=sys.executable,
                    args=["-m", "mcp_server_qdrant.main", "--transport", "stdio"],
                    env={**os.environ,
                         "QDRANT_URL": QDRANT_URL,
                         "QDRANT_API_KEY": QDRANT_API_KEY,
                         "COLLECTION_NAME": COLLECTION_NAME,
                    }
                ),
                timeout=60,  # Aumentado a 60s para dar más tiempo en entornos lentos
            ),
        )
    ],
)

logger.info(f"✓ Agente '{root_agent.name}' configurado correctamente")

# Servicio de sesiones (maneja las conversaciones)
session_service = InMemorySessionService()
# Servicio de memoria (mantiene el contexto)
memory_service = InMemoryMemoryService()
# Servicio de artefactos (maneja archivos y recursos)
artifact_service = InMemoryArtifactService()

almacenador_agent_runner = Runner(
    agent=root_agent,
    app_name="almacenador_agent_app",
    session_service=session_service,
    memory_service=memory_service,
    artifact_service=artifact_service
)

logger.info(f"✓ Runner del agente configurado correctamente")
logger.info(f"  - App: {almacenador_agent_runner.app_name}")
logger.info(f"  - Servicios: Session, Memory, Artifact")

def get_agent_info() -> dict:
    """
    Retorna información sobre el agente configurado.
    Útil para debugging y verificación.
    """
    return {
        "agent_name": root_agent.name,
        "model": root_agent.model.__class__.__name__,
        "qdrant_collection": COLLECTION_NAME,
        "tools_count": len(root_agent.tools),
        "app_name": almacenador_agent_runner.app_name
    }


if __name__ == "__main__":
    # Prueba de configuración cuando se ejecuta directamente
    info = get_agent_info()
    print("\n" + "="*50)
    print("CONFIGURACIÓN DEL AGENTE ALMACENADOR")
    print("="*50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")