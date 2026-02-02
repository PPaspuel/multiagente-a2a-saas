"""
Ejecutor del agente almacenador para el protocolo A2A.
Maneja la recepción de PDFs, procesamiento y respuesta en JSON.
"""

import logging
import uuid
import json
import base64
from typing import Optional, List
from google.genai import types
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext  # ⭐ IMPORTANTE
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
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2a.server.tasks import TaskUpdater  # ⭐ IMPORTANTE: Necesario para el contexto
from almacenador_agent.agent import root_agent, almacenador_agent_runner
from almacenador_agent.tools_agent import (
    PDFProcessor,
    ResponseFormatter,
    validate_pdf_content,
    get_pdf_metadata
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlmacenadorAgentExecutor(AgentExecutor):
    """
    Ejecutor del agente almacenador.
    
    Este ejecutor:
    1. Recibe requests del protocolo A2A
    2. Extrae archivos PDF y texto de la solicitud
    3. Procesa los PDFs usando las herramientas
    4. Ejecuta el agente con el contenido procesado
    5. Devuelve respuestas en formato JSON
    """
    
    def __init__(self):
        self.agent = root_agent
        self.runner = almacenador_agent_runner
        self.pdf_processor = PDFProcessor()
        self.response_formatter = ResponseFormatter()
        
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Ejecuta el agente para procesar la solicitud.
        
        Flujo de ejecución:
        1. Extraer input del usuario (texto y archivos)
        2. Procesar archivos PDF si existen
        3. Construir el prompt para el agente
        4. Ejecutar el agente con Qdrant
        5. Devolver respuesta JSON
        """
        
        logger.info(f"🚀 Iniciando ejecución del agente almacenador")
        logger.info(f"📦 Contexto recibido: task_id={context.task_id}, context_id={context.context_id}")
        
        # ⭐ CORRECCIÓN: Crear TaskUpdater ANTES de procesar
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
            # ⭐ CORRECCIÓN: Acceder correctamente al mensaje
            # ==========================================
            user_text = ""
            user_parts = []
            
            # Verificar estructura del contexto
            logger.debug(f"📋 Context structure: {dir(context)}")
            
            if hasattr(context, 'message') and context.message:
                message = context.message
                logger.info(f"📨 Message received: {message}")
                
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
                    Part(root=TextPart(text="Procesando archivos PDF..."))
                ])
            )
            
            pdf_text = await self._process_pdf_files(user_parts)
            
            # ==========================================
            # PASO 3: CONSTRUIR PROMPT PARA EL AGENTE
            # ==========================================
            if pdf_text:
                await updater.update_status(
                    TaskState.working,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=f"PDF procesado ({len(pdf_text)} caracteres). Ejecutando agente..."))
                    ])
                )
                
                agent_prompt = self._build_pdf_storage_prompt(
                    user_query=user_text if user_text else "Almacenar PDF",
                    pdf_text=pdf_text
                )
            else:
                # Si no hay PDF, informar al usuario
                if user_text:
                    agent_prompt = user_text
                else:
                    error_msg = "No se recibió texto ni archivos PDF para procesar."
                    await updater.update_status(
                        TaskState.failed,
                        message=updater.new_agent_message([
                            Part(root=TextPart(text=error_msg))
                        ])
                    )
                    raise ValueError(error_msg)
            
            logger.info(f"🤖 Ejecutando agente con prompt de {len(agent_prompt)} caracteres")
            
            # ==========================================
            # PASO 4: EJECUTAR EL AGENTE
            # ==========================================
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="Ejecutando agente de almacenamiento..."))
                ])
            )
            
            final_response = await self._run_agent(agent_prompt)
            
            # ==========================================
            # PASO 5: VALIDAR Y ENVIAR RESPUESTA JSON
            # ==========================================
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="Formateando respuesta..."))
                ])
            )
            
            json_response = self._ensure_json_response(final_response, pdf_text)
            
            # ==========================================
            # PASO 6: ENVIAR RESPUESTA Y COMPLETAR TAREA
            # ==========================================
            logger.info(f"✅ Respuesta generada: {json_response[:200]}...")
            
            # Enviar respuesta como artefacto
            await updater.add_artifact([
                Part(root=TextPart(text=json_response))
            ])
            
            # Completar la tarea exitosamente
            await updater.complete()
            
            # También enviar la respuesta al event queue
            await event_queue.enqueue_event(new_agent_text_message(json_response))
            
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
                            # Para FileWithUri, necesitaríamos descargar el archivo
                            # Por simplicidad, lo omitimos por ahora
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
                                        # Intentar decodificar como base64
                                        file_content = base64.b64decode(file_bytes)
                                    except:
                                        # Si falla, usar como bytes directamente
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
    
    
    def _build_pdf_storage_prompt(self, user_query: str, pdf_text: str) -> str:
        """
        Construye el prompt para que el agente almacene el PDF en Qdrant.
        
        Args:
            user_query: Consulta original del usuario
            pdf_text: Texto extraído del PDF
            
        Returns:
            str: Prompt completo para el agente
        """
        collection_name = "contratos-saas"
        
        try:
            import os
            collection_name = os.getenv('COLLECTION_NAME', 'contratos-saas')
        except:
            pass
        
        prompt = f"""
TAREA: Almacenar el siguiente documento PDF en la base de datos vectorial Qdrant.

CONSULTA DEL USUARIO:
{user_query}

CONTENIDO DEL PDF EXTRAÍDO:
{pdf_text[:5000]}...

INSTRUCCIONES:
1. Fragmenta este texto en chunks de aproximadamente 1000 caracteres con overlap de 200
2. Usa las herramientas de Qdrant para almacenar cada fragmento con sus embeddings
3. Asegúrate de almacenar en la colección: {collection_name}
4. Devuelve una respuesta en formato JSON con la siguiente estructura:
{{
  "status": "success",
  "operation": "store_pdf",
  "message": "Descripción del resultado",
  "data": {{
    "chunks_stored": número_de_chunks,
    "total_characters": número_total_de_caracteres,
    "collection": "nombre_de_colección"
  }}
}}

TEXTO COMPLETO A PROCESAR:
{pdf_text}

IMPORTANTE: Devuelve SOLO el JSON de respuesta, sin texto adicional.
"""
        return prompt
    
    
    async def _run_agent(self, prompt: str) -> str:
        """
        Ejecuta el agente con el prompt dado.
        
        Args:
            prompt: Texto del prompt para el agente
            
        Returns:
            str: Respuesta final del agente
        """
        # Crear contenido para el agente
        user_content = types.Content(
            role='user',
            parts=[types.Part(text=prompt)]
        )
        
        # Crear IDs únicos
        session_id = str(uuid.uuid4())
        user_id = "almacenador_agent_user"
        
        # Crear/obtener sesión
        session = await self._upsert_session(user_id=user_id, session_id=session_id)
        logger.info(f"🔗 Sesión: {session_id}")
        
        # Ejecutar el agente
        final_text = ""
        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        final_text += part.text
        
        if not final_text:
            final_text = "El agente no devolvió respuesta."
            
        logger.info(f"📨 Respuesta del agente: {len(final_text)} caracteres")
        return final_text
    
    
    def _ensure_json_response(self, agent_response: str, pdf_text: Optional[str]) -> str:
        """
        Asegura que la respuesta esté en formato JSON válido.
        
        Args:
            agent_response: Respuesta del agente
            pdf_text: Texto del PDF (para estadísticas)
            
        Returns:
            str: JSON válido
        """
        try:
            # Intentar extraer JSON de la respuesta
            json_start = agent_response.find('{')
            json_end = agent_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = agent_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validar estructura
                if isinstance(parsed, dict) and 'status' in parsed:
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
            
            # Si no hay JSON válido, crear uno
            logger.warning("⚠️ Respuesta no es JSON válido, creando estructura")
            
            # Calcular estadísticas
            num_chunks = 0
            if pdf_text:
                chunks = self.pdf_processor.chunk_text(pdf_text)
                num_chunks = len(chunks) if chunks else 0
            
            total_chars = len(pdf_text) if pdf_text else 0
            
            return self.response_formatter.format_success_response(
                operation="store_pdf",
                data={
                    "chunks_stored": num_chunks,
                    "total_characters": total_chars,
                    "agent_response": agent_response[:500] if agent_response else "No response"
                },
                message="Operación completada (respuesta del agente procesada)"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error al parsear JSON: {str(e)}")
            return self.response_formatter.format_error_response(
                operation="parse_response",
                error_message=f"No se pudo generar JSON válido: {str(e)}"
            )
        except Exception as e:
            logger.error(f"❌ Error inesperado: {str(e)}")
            return self.response_formatter.format_error_response(
                operation="ensure_json_response",
                error_message=f"Error inesperado: {str(e)}"
            )
    
    
    async def _upsert_session(self, user_id: str, session_id: str):
        """
        Crea o recupera una sesión existente.
        
        Args:
            user_id: ID del usuario
            session_id: ID de la sesión
            
        Returns:
            Session: Objeto de sesión
        """
        try:
            # Intentar obtener sesión existente
            session = await self.runner.session_service.get_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=session_id
            )
            
            if session is None:
                logger.info(f"🆕 Creando nueva sesión: {session_id}")
                session = await self.runner.session_service.create_session(
                    app_name=self.runner.app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
            else:
                logger.info(f"🔁 Sesión existente recuperada: {session_id}")
            
            if session is None:
                raise RuntimeError(f"No se pudo obtener o crear la sesión: {session_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"❌ Error en _upsert_session: {str(e)}")
            raise
    
    
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
            # Crear updater para la tarea
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await updater.cancel()
            
            # Enviar mensaje de cancelación
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