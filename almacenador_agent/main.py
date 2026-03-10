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
        
        skill_semantic_chunking = AgentSkill(
            id="semantic_chunking",
            name="Fragmentación Semántica de Texto",
            description=(
                "Divide el texto extraído de un PDF en fragmentos semánticamente "
                "coherentes usando similitud coseno entre oraciones consecutivas "
                "(modelo all-MiniLM-L6-v2). El umbral de similitud es configurable: "
                "valores altos generan chunks más pequeños y homogéneos, valores bajos "
                "generan chunks más amplios. Filtra automáticamente títulos, "
                "numeraciones y fragmentos sin contenido informativo relevante."
            ),
            tags=["chunking", "semántico", "embeddings", "nlp", "sentence-transformers"],
            examples=[
                "Fragmenta este texto en chunks semánticos",
                "Divide el contrato en secciones coherentes para indexarlo",
            ],
        )

        skill_analysis_storage = AgentSkill(
            id="store_and_retrieve_analysis",
            name="Almacenamiento y Recuperación de Análisis",
            description=(
                "Almacena y recupera análisis de contratos asociados a documentos "
                "previamente indexados. Permite guardar el resultado de un análisis "
                "legal vinculado a un document_id, consultarlo posteriormente y "
                "listar qué documentos tienen análisis disponibles."
            ),
            tags=["análisis", "recuperación", "contratos", "qdrant", "historial"],
            examples=[
                "Almacena el análisis del contrato con ID a97c3cb5-...",
                "Dame el análisis del documento a97c3cb5-...",
                "Qué documentos tienen análisis almacenados?",
                "Ver el análisis del documento contrato.pdf"
            ],
        )

        skill_deduplication = AgentSkill(
            id="document_deduplication",
            name="Deduplicación de Documentos",
            description=(
                "Antes de almacenar un documento, verifica si ya existe en la "
                "colección Qdrant mediante su huella digital (hash SHA-256). "
                "Si el documento ya fue indexado, elimina los chunks anteriores "
                "y los reemplaza con los nuevos, evitando duplicados sin perder "
                "trazabilidad del documento original."
            ),
            tags=["deduplicación", "sha256", "hash", "actualización", "qdrant"],
            examples=[
                "Actualiza el documento contrato.pdf que ya fue almacenado",
                "Guarda este PDF aunque ya exista una versión anterior",
            ],
        )
        
        # CREAR LA AGENT CARD
        agent_card = AgentCard(
            name="almacenador_agent",
            description=(
                "Agente especializado en el procesamiento y almacenamiento de documentos PDF. "
                "Extrae texto de PDFs, lo fragmenta semánticamente con all-MiniLM-L6-v2, "
                "lo almacena en Qdrant con deduplicación por SHA-256, y gestiona el "
                "almacenamiento y recuperación de análisis de contratos."
            ),
            url=public_url,
            version='2.0.0',
            default_input_modes=[
                'text/plain',           # Texto plano
                'application/pdf'       # Archivos PDF
            ],
            default_output_modes=[
                'text/html'             # Respuestas HTML para visualización
            ],
            capabilities=capabilities,
            skills=[
                skill_extract,
                skill_store,
                skill_semantic_chunking,
                skill_analysis_storage,
                skill_deduplication,
            ],
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