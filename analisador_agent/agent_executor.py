import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError

# Importa tu agente personalizado
from analisador_agent.agent import ContractAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContractAgentExecutor(AgentExecutor):
    """Ejecutor para el agente de análisis de contratos SaaS."""

    def __init__(self):
        self.agent = ContractAgent()
        logger.info("ContractAgentExecutor inicializado correctamente")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Ejecuta el procesamiento de una solicitud de análisis de contrato.
        
        Args:
            context: Contexto de la petición con información del mensaje
            event_queue: Cola de eventos para enviar actualizaciones
        """
        
        # Validar la petición
        error = self._validate_request(context)
        if error:
            logger.error("Petición inválida")
            raise ServerError(error=InvalidParamsError())

        # Extraer la consulta del usuario
        query = context.get_user_input()
        logger.info(f"Procesando consulta: {query[:100]}...")
        
        # Obtener o crear la tarea
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
            logger.info(f"Nueva tarea creada: {task.id}")
        
        # Crear actualizador de tareas
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Procesar la consulta con el agente (streaming)
            async for item in self.agent.stream(query, task.context_id):
                
                is_task_complete = item['is_task_complete']
                require_user_input = item['require_user_input']
                content = item['content']

                # Estado: Trabajando (el agente está procesando)
                if not is_task_complete and not require_user_input:
                    logger.info(f"Estado: Trabajando - {content}")
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                    )
                
                # Estado: Requiere entrada del usuario
                elif require_user_input:
                    logger.info(f"Estado: Requiere entrada - {content}")
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            content,
                            task.context_id,
                            task.id,
                        ),
                        final=True,
                    )
                    break
                
                # Estado: Completado
                else:
                    logger.info("Estado: Completado")
                    # Agregar el resultado como artefacto
                    await updater.add_artifact(
                        [Part(root=TextPart(text=content))],
                        name='contract_analysis_result',
                    )
                    # Marcar tarea como completada
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(f'Error durante el streaming: {e}', exc_info=True)
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """
        Valida si la petición es correcta.
        
        Returns:
            False si es válida, True si hay error
        """
        # Aquí puedes agregar validaciones personalizadas
        # Por ejemplo, verificar que el mensaje no esté vacío
        
        user_input = context.get_user_input()
        if not user_input or len(user_input.strip()) == 0:
            logger.warning("Petición vacía recibida")
            return True
        
        return False

    async def cancel(
        self, 
        context: RequestContext, 
        event_queue: EventQueue
    ) -> None:
        """
        Cancela una tarea en ejecución.
        Actualmente no soportado.
        """
        logger.warning("Intento de cancelación - operación no soportada")
        raise ServerError(error=UnsupportedOperationError())