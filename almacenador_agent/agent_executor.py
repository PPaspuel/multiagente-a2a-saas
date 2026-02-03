"""
Ejecutor del agente almacenador para el protocolo A2A.
VERSIÓN MEJORADA: Almacenamiento directo a Qdrant desplegada en Docker.
"""

import logging
import base64
from typing import Optional, List
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    InternalError, 
    TextPart, 
    UnsupportedOperationError,
    FilePart,
    Part,
    FileWithBytes,
    FileWithUri,
    TaskState
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from a2a.server.tasks import TaskUpdater
from almacenador_agent.agent import root_agent, almacenador_agent_runner
from almacenador_agent.tools_agent import (
    PDFProcessor,
    ResponseFormatter,
    validate_pdf_content,
    get_pdf_metadata
)
# ⭐ NUEVO: Importar el gestor de almacenamiento directo
from almacenador_agent.qdrant_storage import storage_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlmacenadorAgentExecutor(AgentExecutor):
    """
    Ejecutor del agente almacenador.
    
    VERSIÓN MEJORADA:
    - Almacenamiento DIRECTO a Qdrant (sin MCP problemático)
    - Procesamiento más rápido y confiable
    - Mejor manejo de errores
    
    Flujo:
    1. Recibe requests del protocolo A2A
    2. Extrae archivos PDF y texto
    3. Procesa PDFs (extracción + fragmentación)
    4. Almacena DIRECTAMENTE en Qdrant
    5. Devuelve respuestas en formato JSON
    """
    
    def __init__(self):
        self.agent = root_agent
        self.runner = almacenador_agent_runner
        self.pdf_processor = PDFProcessor()
        self.response_formatter = ResponseFormatter()
        logger.info("✅ AlmacenadorAgentExecutor inicializado")
        
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Ejecuta el agente para procesar la solicitud.
        
        CAMBIO PRINCIPAL:
        - Ya NO usa el agente con herramientas MCP
        - Ahora almacena DIRECTAMENTE en Qdrant
        """
        
        logger.info(f"🚀 Iniciando ejecución del agente almacenador")
        logger.info(f"📦 Contexto recibido: task_id={context.task_id}, context_id={context.context_id}")
        
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # ==========================================
            # PASO 0: Actualizar estado de la tarea
            # ==========================================
            if not context.current_task:
                await updater.submit()
            await updater.start_work()
            
            # ==========================================
            # PASO 1: EXTRAER INPUT DEL USUARIO
            # ==========================================
            user_text = ""
            user_parts = []
            
            if hasattr(context, 'message') and context.message:
                message = context.message
                logger.info(f"📨 Message received")
                
                if hasattr(message, 'parts') and message.parts:
                    user_parts = message.parts
                    
                    # Extraer texto de las partes
                    text_content = []
                    for part in user_parts:
                        if isinstance(part, Part):
                            root = getattr(part, 'root', None)
                            if isinstance(root, TextPart):
                                text_content.append(root.text)
                    
                    user_text = " ".join(text_content) if text_content else ""
                    
            logger.info(f"📝 Texto extraído: {user_text[:100] if user_text else 'Sin texto'}")
            logger.info(f"📦 Número de partes: {len(user_parts)}")
            
            # ==========================================
            # PASO 2: PROCESAR ARCHIVOS PDF
            # ==========================================
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="📄 Procesando archivos PDF..."))
                ])
            )
            
            pdf_text = await self._process_pdf_files(user_parts)
            
            if not pdf_text:
                # Si no hay PDF, informar al usuario
                error_msg = "❌ No se recibió ningún archivo PDF para procesar."
                await updater.update_status(
                    TaskState.failed,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=error_msg))
                    ])
                )
                raise ValueError(error_msg)
            
            logger.info(f"✅ PDF procesado: {len(pdf_text)} caracteres")
            
            # ==========================================
            # PASO 3: FRAGMENTAR TEXTO
            # ⭐ NUEVO: Fragmentación directa sin pasar por el agente
            # ==========================================
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="✂️ Fragmentando texto en chunks..."))
                ])
            )
            
            chunks = self.pdf_processor.chunk_text(
                text=pdf_text,
                chunk_size=1000,
                overlap=200
            )
            
            logger.info(f"✂️ Texto fragmentado en {len(chunks)} chunks")
            
            # ==========================================
            # PASO 4: ALMACENAR DIRECTAMENTE EN QDRANT
            # ⭐ CAMBIO PRINCIPAL: Ya no usa MCP, almacena directo
            # ==========================================
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text=f"💾 Almacenando {len(chunks)} fragmentos en Qdrant..."))
                ])
            )
            
            # Almacenar usando el gestor directo
            storage_result = storage_manager.store_chunks(
                chunks=chunks,
                metadata={
                    "task_id": context.task_id,
                    "origen": "a2a_upload",
                    "user_query": user_text[:200] if user_text else "Sin consulta"
                }
            )
            
            # ==========================================
            # PASO 5: PREPARAR Y ENVIAR RESPUESTA JSON
            # ==========================================
            if storage_result["status"] == "success":
                logger.info(f"✅ {storage_result['chunks_stored']} fragmentos almacenados exitosamente")
                
                # Crear respuesta JSON exitosa
                json_response = self.response_formatter.format_storage_response(
                    num_chunks=storage_result["chunks_stored"],
                    total_characters=len(pdf_text),
                    collection_name=storage_result["collection"]
                )
                
                # Actualizar estado
                await updater.update_status(
                    TaskState.working,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text="✅ Almacenamiento completado exitosamente"))
                    ])
                )
                
            else:
                # Error en almacenamiento
                logger.error(f"❌ Error en almacenamiento: {storage_result.get('message')}")
                json_response = self.response_formatter.format_error_response(
                    operation="store_pdf",
                    error_message=storage_result.get("message", "Error desconocido en almacenamiento"),
                    error_type="StorageError"
                )
                
                await updater.update_status(
                    TaskState.working,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text="⚠️ Error durante el almacenamiento"))
                    ])
                )
            
            # ==========================================
            # PASO 6: ENVIAR RESPUESTA Y COMPLETAR
            # ==========================================
            logger.info(f"📤 Enviando respuesta JSON...")
            
            # Enviar respuesta como artefacto
            await updater.add_artifact([
                Part(root=TextPart(text=json_response))
            ])
            
            # Completar la tarea
            await updater.complete()
            
            # Enviar al event queue
            await event_queue.enqueue_event(new_agent_text_message(json_response))
            
            logger.info("✅ Ejecución completada exitosamente")
            
        except Exception as e:
            logger.error(f'❌ Error durante la ejecución: {str(e)}', exc_info=True)
            
            # Crear respuesta de error en JSON
            error_response = self.response_formatter.format_error_response(
                operation="execute",
                error_message=str(e),
                error_type=type(e).__name__
            )
            
            # Actualizar estado de la tarea como fallida
            try:
                await updater.fail(
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=error_response))
                    ])
                )
            except:
                # Si falla el updater, enviar directamente al event queue
                await event_queue.enqueue_event(new_agent_text_message(error_response))
            
            raise ServerError(error=InternalError()) from e
    
    
    async def _process_pdf_files(self, user_parts: List[Part]) -> Optional[str]:
        """
        Procesa archivos PDF de la solicitud.
        
        Args:
            user_parts: Lista de partes del mensaje del usuario
            
        Returns:
            str: Texto extraído del PDF o None si no hay PDF
        """
        pdf_texts = []
        
        for part in user_parts:
            if isinstance(part, Part):
                root = getattr(part, 'root', None)
                
                # Verificar si es un archivo
                if isinstance(root, FilePart):
                    file_obj = getattr(root, 'file', None)
                    
                    if file_obj:
                        file_name = ""
                        file_content = None
                        
                        # Manejar FileWithUri
                        if isinstance(file_obj, FileWithUri):
                            file_name = getattr(file_obj, 'uri', 'archivo.pdf').split('/')[-1]
                            logger.warning(f"⚠️ FileWithUri detectado: {file_name}. Necesita implementación de descarga.")
                            continue
                        
                        # Manejar FileWithBytes
                        elif isinstance(file_obj, FileWithBytes):
                            file_name = getattr(file_obj, 'filename', 'archivo.pdf')
                            file_bytes = getattr(file_obj, 'bytes', None)
                            
                            if file_bytes:
                                # Decodificar si es base64 o string
                                if isinstance(file_bytes, str):
                                    try:
                                        file_content = base64.b64decode(file_bytes)
                                    except:
                                        file_content = file_bytes.encode('utf-8')
                                else:
                                    file_content = file_bytes
                        
                        # Verificar que sea PDF
                        if file_name.lower().endswith('.pdf') and file_content:
                            try:
                                # Validar PDF
                                if not validate_pdf_content(file_content):
                                    logger.warning(f"⚠️ El archivo '{file_name}' no es un PDF válido")
                                    continue
                                
                                # Obtener metadatos
                                metadata = get_pdf_metadata(file_content)
                                logger.info(f"📊 Metadatos del PDF: {metadata}")
                                
                                # Extraer texto
                                text = self.pdf_processor.extract_text_from_pdf(file_content)
                                
                                if text and text.strip():
                                    pdf_texts.append({
                                        'filename': file_name,
                                        'text': text,
                                        'metadata': metadata
                                    })
                                    logger.info(f"✅ Texto extraído de '{file_name}': {len(text)} caracteres")
                                else:
                                    logger.warning(f"⚠️ No se pudo extraer texto de '{file_name}'")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error procesando PDF '{file_name}': {str(e)}")
                                raise ValueError(f"Error al procesar PDF: {str(e)}")
        
        # Combinar textos de todos los PDFs
        if pdf_texts:
            combined_text = "\n\n".join([
                f"=== ARCHIVO: {item['filename']} ===\n{item['text']}"
                for item in pdf_texts
            ])
            return combined_text
        
        return None
    
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Maneja la cancelación de una solicitud.
        """
        logger.warning("⚠️ Cancelación solicitada")
        
        try:
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await updater.cancel()
            
            cancel_msg = "La operación ha sido cancelada por el usuario."
            await event_queue.enqueue_event(new_agent_text_message(
                self.response_formatter.format_error_response(
                    operation="cancel",
                    error_message=cancel_msg
                )
            ))
            
        except Exception as e:
            logger.error(f"❌ Error al cancelar: {str(e)}")
            raise ServerError(error=UnsupportedOperationError(
                details=f"Cancelación fallida: {str(e)}"
            ))