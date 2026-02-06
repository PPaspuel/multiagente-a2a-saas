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

    4. FORMATO DE RESPUESTA:
       - SIEMPRE devuelves respuestas en formato HTML estructurado
       - Incluyes información sobre el resultado de la operación
       - Reportas el número de fragmentos almacenados
       - Indicas si hubo algún error durante el proceso
    
    ESTRUCTURA DE RESPUESTA HTML:
    <div class="response">
      <h3>Resultado de la operación</h3>
      <p><strong>Estado:</strong> success o error</p>
      <p><strong>Mensaje:</strong> Descripción del resultado</p>
      <ul>
        <li><strong>Fragmentos almacenados:</strong> número</li>
        <li><strong>Total de caracteres:</strong> número</li>
        <li><strong>Colección:</strong> nombre de colección</li>
        <li><strong>Páginas procesadas:</strong> número</li>
      </ul>
    </div>
    
    COMPORTAMIENTO:
    - Sé conciso y directo en tus respuestas
    - Siempre valida los datos antes de procesarlos
    - Maneja errores de manera elegante y reporta problemas claramente
    - Usa las herramientas de Qdrant de forma eficiente
    - No inventes información, reporta solo resultados reales
    
    IMPORTANTE:
    - Tu respuesta final DEBE ser un HTML válido
    - No agregues explicaciones adicionales fuera del HTML
    - Si hay un error, devuelve un HTML con estado "error"
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