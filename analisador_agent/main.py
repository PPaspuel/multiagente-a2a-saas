"""
Servidor principal del agente analizador de contratos usando protocolo A2A.
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
from analisador_agent.agent_executor import ContractAnalyzerExecutor

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
        public_url: URL pública del agente (ej: http://localhost:8002)
        
    Returns:
        AgentCard: Tarjeta de configuración del agente
    """
    try:
        # Definir capacidades del agente
        capabilities = AgentCapabilities(
            streaming=True,  # Requiere streaming
            push_notifications=True  # Requiere notificaciones push
        )
        
        # Habilidad 1: Identificación de Derechos
        skill_derechos = AgentSkill(
            id="identify_rights",
            name="Identificación de Derechos Contractuales",
            description=(
                "Analiza contratos legales e identifica todos los derechos otorgados a las partes. "
                "Incluye derechos de pago, uso de propiedad intelectual, rescisión, auditoría, "
                "recepción de servicios, y cualquier otra prerrogativa contractual."
            ),
            tags=["derechos", "contratos", "análisis legal", "prerrogativas"],
            examples=[
                "¿Qué derechos tiene el cliente en este contrato?",
                "Identifica todos los derechos de rescisión",
                "Lista los derechos de propiedad intelectual en este acuerdo"
            ],
        )
        
        # Habilidad 2: Identificación de Obligaciones
        skill_obligaciones = AgentSkill(
            id="identify_obligations",
            name="Identificación de Obligaciones Contractuales",
            description=(
                "Extrae y clasifica todas las obligaciones y deberes que las partes deben cumplir "
                "según el contrato. Incluye obligaciones de pago, entrega, confidencialidad, "
                "cumplimiento de plazos, estándares de calidad, y cualquier otro compromiso contractual."
            ),
            tags=["obligaciones", "deberes", "contratos", "compromisos"],
            examples=[
                "¿Cuáles son las obligaciones del proveedor?",
                "Lista todas las obligaciones de pago",
                "Identifica los plazos y obligaciones de entrega"
            ],
        )
        
        # Habilidad 3: Identificación de Prohibiciones
        skill_prohibiciones = AgentSkill(
            id="identify_prohibitions",
            name="Identificación de Prohibiciones y Restricciones",
            description=(
                "Detecta todas las prohibiciones, limitaciones y restricciones impuestas a las partes. "
                "Incluye cláusulas de no competencia, restricciones de divulgación, limitaciones de uso, "
                "restricciones territoriales, y cualquier otra prohibición contractual."
            ),
            tags=["prohibiciones", "restricciones", "limitaciones", "cláusulas"],
            examples=[
                "¿Qué está prohibido en este contrato?",
                "Identifica las cláusulas de no competencia",
                "Lista todas las restricciones de confidencialidad"
            ],
        )
        
        # Habilidad 4: Análisis Integral
        skill_analisis_integral = AgentSkill(
            id="comprehensive_contract_analysis",
            name="Análisis Integral de Contratos",
            description=(
                "Realiza un análisis completo del contrato identificando simultáneamente "
                "derechos, obligaciones y prohibiciones. Clasifica cada elemento por criticidad "
                "(ALTA/MEDIA/BAJA) y proporciona referencias exactas a las cláusulas del contrato. "
                "Devuelve resultados en formato JSON estructurado para fácil integración."
            ),
            tags=["análisis completo", "json", "criticidad", "estructurado"],
            examples=[
                "Analiza este contrato completamente",
                "Dame un análisis integral de derechos, obligaciones y prohibiciones",
                "Extrae todos los elementos legales críticos de este acuerdo"
            ],
        )
        
        # Crear la Agent Card
        agent_card = AgentCard(
            name="Contract Analyzer Agent",
            description=(
                "Agente especializado en análisis legal de contratos utilizando CrewAI. "
                "Identifica y extrae de forma precisa Derechos, Obligaciones y Prohibiciones "
                "presentes en documentos contractuales. Proporciona análisis estructurado en JSON "
                "con clasificación de criticidad y referencias exactas a cláusulas. "
                "Ideal para revisión legal automatizada, due diligence, y gestión contractual."
            ),
            url=public_url,
            version='1.0.0',
            default_input_modes=[
                'application/pdf',      # Archivos PDF de contratos
                'text/plain'            # Texto plano de contratos
            ],
            default_output_modes=[
                'application/json'      # Respuestas JSON estructuradas
            ],
            capabilities=capabilities,
            skills=[
                skill_derechos,
                skill_obligaciones,
                skill_prohibiciones,
                skill_analisis_integral
            ],
        )
        
        logger.info("✅ Agent Card creada exitosamente")
        return agent_card
        
    except Exception as e:
        logger.error(f'❌ Error creando AgentCard: {e}')
        raise


def main():
    """
    Función principal que inicia el servidor del agente.
    """
    try:
        # Obtener configuración del servidor
        host = os.getenv('HOST', '0.0.0.0')
        port = int(os.getenv('PORT', 8002))
        
        # Obtener URL pública
        # Si está en producción, usar dominio público
        # Si está en localhost, usar localhost
        public_url = os.getenv('PUBLIC_URL', f'http://localhost:{port}')
        
        logger.info("=" * 60)
        logger.info("INICIANDO AGENTE ANALIZADOR DE CONTRATOS")
        logger.info("=" * 60)
        
        # Crear la tarjeta del agente
        agent_card = create_agent_card(public_url=public_url)
        
        # Crear el ejecutor del agente
        agent_executor = ContractAnalyzerExecutor()
        
        # Configurar el manejador de peticiones
        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor,
            task_store=InMemoryTaskStore(),
        )
        
        # Crear la aplicación Starlette con A2A
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )
        
        # Información útil para el usuario
        logger.info("🚀 SERVIDOR LISTO")
        logger.info(f"📋 Agent Card disponible en: {public_url}/.well-known/agent-card.json")
        logger.info("=" * 60)
        
        # Iniciar el servidor
        uvicorn.run(server.build(), host=host, port=port)
        
    except ValueError as e:
        logger.error(f'❌ Error de configuración: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'❌ Error durante el inicio del servidor: {e}')
        logger.error(f'   Detalles: {str(e)}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()