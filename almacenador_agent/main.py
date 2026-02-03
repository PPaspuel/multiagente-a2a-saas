"""
Servidor principal del agente almacenador usando protocolo A2A.
Punto de entrada de la aplicación.
"""

import logging
import sys
import os
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv
from almacenador_agent.agent_executor import AlmacenadorAgentExecutor

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_agent_card(public_url=None):
    """
    Crea la tarjeta de presentación del agente para el protocolo A2A.
    
    La AgentCard describe las capacidades y habilidades del agente,
    permitiendo que otros agentes sepan cómo interactuar con él.
    
    Args:
        public_url: URL pública del agente (ej: http://localhost:8001)
        
    Returns:
        AgentCard: Tarjeta de configuración del agente
    """
    try:
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        skill_extract = AgentSkill(
            id="extract_text_from_pdf",
            name="Extracción de Texto desde PDF",
            description=(
                "Extrae el contenido de texto completo de archivos PDF. "
                "Lee página por página y devuelve el texto en orden de lectura. "
            ),
            tags=["pdf", "extracción", "texto", "procesamiento"],
            examples=[
                "Extrae el texto de este contrato PDF",
                "Lee este documento PDF y dame su contenido"
            ],
        )

        skill_store = AgentSkill(
            id="store_pdf_in_qdrant",
            name="Almacenamiento Vectorial de PDFs",
            description=(
                "Almacena el contenido de documentos PDF en una base de datos "
                "vectorial Qdrant. Fragmenta automáticamente el texto, crea "
                "embeddings semánticos y los almacena para búsqueda posterior."
            ),
            tags=["almacenamiento", "qdrant", "vectorial"],
            examples=[
                "Almacena este contrato en la base de datos",
                "Guarda este PDF para búsqueda semántica",
                "Indexa este documento en Qdrant"
            ],
        )
        
        skill_json = AgentSkill(
            id="json_structured_response",
            name="Respuestas JSON Estructuradas",
            description=(
                "Devuelve todas las respuestas en formato JSON estructurado, "
                "facilitando la integración con otros sistemas y agentes. "
                "Incluye información detallada sobre el resultado de las operaciones."
            ),
            tags=["json", "api", "estructurado", "integración"],
            examples=[
                "Dame el resultado en formato JSON",
                "Responde con un JSON estructurado"
            ],
        )
        
        # CREAR LA AGENT CARD
        agent_card = AgentCard(
            name="almacenador_agent",
            description=(
                "Agente especializado en el procesamiento y almacenamiento de documentos PDF. "
                "Extrae texto de PDFs, lo fragmenta inteligentemente, y lo almacena en "
                "una base de datos vectorial Qdrant para permitir búsquedas semánticas. "
            ),
            url=public_url,  # ⭐ USAR LA URL PÚBLICA
            version='2.0.0',
            default_input_modes=[
                'text/plain',           # Texto plano
                'application/pdf'       # Archivos PDF
            ],
            default_output_modes=[
                'application/json'      # Respuestas JSON
            ],
            capabilities=capabilities,
            skills=[skill_extract, skill_store, skill_json],
        )
        
        return agent_card
        
    except Exception as e:
        logger.error(f'Error creando AgentCard: {e}')
        raise


def main():
    """
    Función principal que inicia el servidor del agente.
    """
    try:
        # Obtener configuración del servidor
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 8001))
        
        # ⭐ IMPORTANTE: Obtener la URL pública
        # Si está en localhost, usar localhost
        # Si está en una red, usar la IP local o dominio público
        public_url = os.getenv('PUBLIC_URL', f'http://localhost:{port}')
        
        logger.info(f"🔧 Configuración del servidor:")
        
        # Crear la tarjeta del agente CON LA URL PÚBLICA
        agent_card = create_agent_card(public_url=public_url)
        agent_executor = AlmacenadorAgentExecutor()
        
        # Configurar el manejador de peticiones
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
        )
        
        # Crear la aplicación Starlette
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )

        # Iniciar el servidor
        logger.info(f"🚀 Iniciando el servidor almacenador_agent")
        logger.info(f"📍 Servidor escuchando en: http://{host}:{port}")
        logger.info(f"🌐 URL pública: {public_url}")
        logger.info(f"📋 Agent Card: {public_url}/.well-known/agent-card.json")
        
        uvicorn.run(server.build(), host=host, port=port)
        
    except ValueError as e:
        logger.error(f'❌ Error de valor: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'❌ Se produjo un error durante el inicio del servidor: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()