import logging
import os
import sys
import click
import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

# Importar tu agente y ejecutor personalizados
from analisador_agent.agent import ContractAgent
from analisador_agent.agent_executor import ContractAgentExecutor


# Cargar variables de entorno desde .env
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Excepción para cuando falta la clave API."""
    pass


@click.command()
@click.option('--host', 'host', default='localhost', help='Host del servidor')
@click.option('--port', 'port', default=8002, help='Puerto del servidor')
def main(host, port):
    """
    Inicia el servidor del Agente de Análisis de Contratos SaaS.
    
    Este agente se especializa en identificar y clasificar cláusulas
    en contratos SaaS (derechos, obligaciones, prohibiciones).
    """
    
    try:
        # Verificar que existe la clave API de Google
        if not os.getenv('GOOGLE_API_KEY'):
            raise MissingAPIKeyError(
                'GOOGLE_API_KEY no está configurada. '
                'Por favor, configúrala en tu archivo .env'
            )
        
        logger.info("✓ Clave API de Google configurada correctamente")

        # Definir las capacidades del agente
        capabilities = AgentCapabilities(
            streaming=True,  # Soporta respuestas en streaming
            push_notifications=True  # Puede enviar notificaciones
        )
        
        # Definir la habilidad principal del agente
        skill = AgentSkill(
            id='analyze_saas_contracts',
            name='Análisis de Contratos SaaS',
            description=(
                'Identifica y clasifica cláusulas en contratos SaaS: '
                'derechos del cliente, obligaciones contractuales, '
                'y prohibiciones o restricciones'
            ),
            tags=[
                'contratos', 
                'SaaS', 
                'análisis legal', 
                'cláusulas',
                'derechos',
                'obligaciones'
            ],
            examples=[
                'Analiza este contrato de servicio en la nube',
                'Identifica las obligaciones del cliente en este contrato',
                '¿Qué derechos tengo según este acuerdo SaaS?',
                'Lista las prohibiciones en este contrato de software'
            ],
        )
        
        # Crear la tarjeta del agente (su "presentación")
        agent_card = AgentCard(
            name='Agente de Análisis de Contratos SaaS',
            description=(
                'Agente especializado en análisis de contratos de Software as a Service. '
                'Identifica automáticamente derechos, obligaciones y prohibiciones '
                'en documentos contractuales.'
            ),
            url=f'http://{host}:{port}',
            version='1.0.0',
            default_input_modes=ContractAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=ContractAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )
        
        logger.info(f"✓ Tarjeta del agente creada: {agent_card.name}")

        # Configurar los componentes del servidor
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )
        
        # Crear el manejador de peticiones
        request_handler = DefaultRequestHandler(
            agent_executor=ContractAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )
        
        logger.info("✓ Manejador de peticiones configurado")
        
        # Crear la aplicación del servidor
        server = A2AStarletteApplication(
            agent_card=agent_card, 
            http_handler=request_handler
        )
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║  🤖 Servidor del Agente de Contratos SaaS Iniciado           ║
╟──────────────────────────────────────────────────────────────╢
║  📍 URL: http://{host}:{port}/                          
║  📋 Tarjeta: http://{host}:{port}/.well-known/agent-card.json
║  ⚡ Capacidades: Streaming, Notificaciones Push              ║
║  🎯 Especialidad: Análisis de Contratos SaaS                 ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        # Iniciar el servidor
        uvicorn.run(server.build(), host=host, port=port, log_level="info")

    except MissingAPIKeyError as e:
        logger.error(f'❌ Error de configuración: {e}')
        logger.error('💡 Solución: Crea un archivo .env con: GOOGLE_API_KEY=tu_clave_aqui')
        sys.exit(1)
        
    except Exception as e:
        logger.error(f'❌ Error inesperado durante el inicio: {e}', exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()