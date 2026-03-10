"""
Configuración del agente almacenador con Google ADK.
Este agente procesa PDFs y los almacena en Qdrant.
"""

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("ERROR: Falta la variable OPENROUTER_API_KEY")

root_agent = Agent(
    model=LiteLlm(
        model="openrouter/meta-llama/llama-4-maverick",
        api_key=OPENROUTER_API_KEY,
        api_base="https://openrouter.ai/api/v1"
    ),
    name="almacenador_agent",
    
    # Instrucciones mejoradas y más específicas
    instruction="""
    Eres un agente especializado en procesamiento y almacenamiento de documentos PDF.
    
    ROL:
    Recibes el resultado de operaciones ya ejecutadas por el executor
    (extracción de texto, chunking semántico, almacenamiento en Qdrant)
    y tu única responsabilidad es presentar ese resultado al usuario
    en formato HTML estructurado.

    FORMATO DE RESPUESTA:
    - SIEMPRE devuelves respuestas en formato HTML válido
    - No agregues texto ni explicaciones fuera del HTML
    - Si la operación fue exitosa, reporta: nombre del documento,
    fragmentos almacenados, total de caracteres, colección y document_id
    - Si hubo un error, devuelve un HTML con estado "error" y descripción clara
    - No inventes datos — reporta únicamente los resultados recibidos
    """,
    tools=[],
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
        "tools_count": len(root_agent.tools),
        "app_name": almacenador_agent_runner.app_name
    }


if __name__ == "__main__":
    info = get_agent_info()
    print("\n" + "="*50)
    print("CONFIGURACIÓN DEL AGENTE ALMACENADOR")
    print("="*50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")