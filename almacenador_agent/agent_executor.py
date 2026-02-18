"""
Ejecutor del agente almacenador para el protocolo A2A.
VERSIÓN MEJORADA:
- Deduplicación automática de documentos
- Almacenamiento de análisis vinculados
- Recuperación de análisis almacenados
- FIX: Removido uso de updater.fail() que no existe
- NUEVO: Extracción de nombre personalizado del usuario
"""

import logging
import base64
import json
import re
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
from almacenador_agent.qdrant_storage import storage_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlmacenadorAgentExecutor(AgentExecutor):
    """
    Ejecutor del agente almacenador.
    
    VERSIÓN MEJORADA:
    - Deduplicación automática de documentos
    - Almacenamiento directo a Qdrant 
    - Almacenamiento de análisis vinculados a documentos
    - Recuperación de análisis almacenados
    - Extracción de nombre personalizado del usuario
    
    Flujo de operaciones:
    1. ALMACENAR PDF: Recibe PDF, detecta duplicados, almacena/actualiza
    2. ALMACENAR ANÁLISIS: Recibe análisis en texto y lo vincula al documento
    3. RECUPERAR ANÁLISIS: Busca y muestra análisis almacenados
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
        
        TIPOS DE OPERACIÓN:
        1. Almacenar PDF (con deduplicación automática)
        2. Almacenar análisis (requiere document_id o referencia)
        3. Recuperar análisis (por document_id o búsqueda general)
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
            # PASO 2: DETERMINAR TIPO DE OPERACIÓN
            # ==========================================
            operation_type = self._detect_operation_type(user_text, user_parts)
            logger.info(f"🎯 Operación detectada: {operation_type}")
            
            # Ejecutar operación correspondiente
            if operation_type == "store_pdf":
                await self._handle_store_pdf(updater, event_queue, user_parts, user_text, context)
            
            elif operation_type == "store_analysis":
                await self._handle_store_analysis(updater, event_queue, user_text, context)
            
            elif operation_type == "retrieve_analysis":
                await self._handle_retrieve_analysis(updater, event_queue, user_text)
            
            else:
                # Operación no reconocida
                error_msg = (
                    "❌ No se pudo determinar la operación solicitada.\n\n"
                    "Operaciones disponibles:\n"
                    "1. 📄 Almacenar PDF: Envía un archivo PDF\n"
                    "2. 💾 Almacenar análisis: Envía texto con 'almacena el análisis' + document_id + análisis\n"
                    "3. 🔍 Recuperar análisis: Envía 'recupera el análisis' o 'muestra el análisis'"
                )
                
                # FIX: Usar update_status en lugar de fail
                await updater.update_status(
                    TaskState.failed,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=error_msg))
                    ])
                )
                await event_queue.enqueue_event(new_agent_text_message(error_msg))
            
            logger.info("✅ Ejecución completada exitosamente")
            
        except Exception as e:
            logger.error(f'❌ Error durante la ejecución: {str(e)}', exc_info=True)
            
            # Crear respuesta de error en JSON
            error_response = self.response_formatter.format_error_response(
                operation="execute",
                error_message=str(e),
                error_type=type(e).__name__
            )
            
            # FIX: Enviar error directamente sin usar updater.fail()
            try:
                await updater.update_status(
                    TaskState.failed,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=error_response))
                    ])
                )
            except Exception as update_error:
                logger.error(f"Error actualizando estado: {update_error}")
            
            # Enviar al event queue
            await event_queue.enqueue_event(new_agent_text_message(error_response))
            
            raise ServerError(error=InternalError()) from e
    
    
    def _detect_operation_type(self, user_text: str, user_parts: List[Part]) -> str:
        """
        Detecta el tipo de operación solicitada por el usuario.
        
        Returns:
            str: "store_pdf", "store_analysis", "retrieve_analysis", "unknown"
        """
        user_text_lower = user_text.lower()
        
        # Verificar si hay archivos PDF
        has_pdf = any(
            isinstance(getattr(part, 'root', None), FilePart)
            for part in user_parts
        )
        
        # Palabras clave para almacenar análisis
        store_analysis_keywords = [
            "almacena el análisis",
            "guarda el análisis",
            "almacenar análisis",
            "guardar análisis",
            "almacena análisis",
            "guarda análisis"
        ]
        
        # Palabras clave para recuperar análisis
        retrieve_analysis_keywords = [
            "recupera el análisis",
            "muestra el análisis",
            "ver el análisis",
            "obtener análisis",
            "mostrar análisis",
            "ver análisis"
        ]
        
        # Decisión de operación
        if has_pdf:
            return "store_pdf"
        
        elif any(keyword in user_text_lower for keyword in store_analysis_keywords):
            return "store_analysis"
        
        elif any(keyword in user_text_lower for keyword in retrieve_analysis_keywords):
            return "retrieve_analysis"
        
        else:
            return "unknown"
    
    
    def _extract_custom_filename(self, user_text: str) -> Optional[str]:
        """
        Extrae el nombre personalizado que el usuario quiere darle al archivo.
        
        Busca patrones como:
        - "almacena con el nombre X"
        - "guarda con el nombre X"
        - "almacena como X"
        - "guarda como X"
        - "nombre: X"
        - "llamado X"
        - "denominado X"
        
        Args:
            user_text: Texto del mensaje del usuario
            
        Returns:
            str: Nombre personalizado extraído (sin extensión) o None
        """
        if not user_text:
            return None
        
        # Patrones para extraer el nombre
        patterns = [
            r'(?:con el nombre|con nombre)\s+["\']?([^"\'.\n]+?)["\']?(?:\.|$|\n)',
            r'(?:almacena|guarda|guardar|almacenar)\s+como\s+["\']?([^"\'.\n]+?)["\']?(?:\.|$|\n)',
            r'(?:llamado|denominado|titulado)\s+["\']?([^"\'.\n]+?)["\']?(?:\.|$|\n)',
            r'nombre:\s*["\']?([^"\'.\n]+?)["\']?(?:\.|$|\n)',
            r'nombre\s+["\']?([^"\'.\n]+?)["\']?(?:\.|$|\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                custom_name = match.group(1).strip()
                # Limpiar el nombre (eliminar caracteres no válidos)
                custom_name = re.sub(r'[<>:"/\\|?*]', '', custom_name)
                
                if custom_name:
                    logger.info(f"📝 Nombre personalizado detectado: '{custom_name}'")
                    # Asegurarnos de que tenga extensión .pdf
                    if not custom_name.lower().endswith('.pdf'):
                        custom_name = f"{custom_name}.pdf"
                    return custom_name
        
        logger.info("📝 No se detectó nombre personalizado, se usará el nombre original del archivo")
        return None
    
    
    async def _handle_store_pdf(
        self,
        updater: TaskUpdater,
        event_queue: EventQueue,
        user_parts: List[Part],
        user_text: str,
        context: RequestContext
    ):
        """
        Maneja el almacenamiento de archivos PDF.
        MEJORADO: Detecta duplicados automáticamente y extrae nombre personalizado.
        """
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([
                Part(root=TextPart(text="📄 Procesando PDF..."))
            ])
        )
        
        # Extraer nombre personalizado del usuario
        custom_filename = self._extract_custom_filename(user_text)
        
        # Procesar archivos PDF
        pdf_result = await self._process_pdf_files(user_parts, custom_filename)
        
        if not pdf_result:
            error_msg = "❌ No se pudo procesar el archivo PDF"
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message([
                    Part(root=TextPart(text=error_msg))
                ])
            )
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        # Fragmentar el texto
        chunks = self.pdf_processor.chunk_text(pdf_result['text'])
        
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([
                Part(root=TextPart(text=f"💾 Almacenando en Qdrant como '{pdf_result['filename']}'..."))
            ])
        )
        
        # Almacenar en Qdrant (con el nombre correcto)
        storage_result = storage_manager.store_chunks(
            chunks=chunks,
            metadata={
                "source": "a2a_protocol",
                "task_id": context.task_id,
                "num_pages": pdf_result['metadata'].get('num_pages', 0)
            },
            full_content=pdf_result['text'],
            filename=pdf_result['filename']  # Usar el nombre (ya sea personalizado o original)
        )
        
        # Preparar respuesta
        if storage_result["status"] == "success":
            response_data = {
                "status": "success",
                "operation": "store_pdf",
                "message": storage_result.get("message", "PDF almacenado exitosamente"),
                "data": {
                    "filename": pdf_result['filename'],
                    "chunks_stored": storage_result["chunks_stored"],
                    "total_characters": len(pdf_result['text']),
                    "collection": storage_result["collection"],
                    "document_id": storage_result.get("document_id"),
                    "was_updated": storage_result.get("was_updated", False),
                    "num_pages": pdf_result['metadata'].get('num_pages', 0)
                }
            }
            
            # Renderizar HTML
            html_response = self.response_formatter.render_storage_response_html(response_data)
            json_response = json.dumps(response_data, indent=2, ensure_ascii=False)
            
            await event_queue.enqueue_event(new_agent_text_message(html_response))
            await updater.add_artifact([Part(root=TextPart(text=json_response))])
            await updater.complete()
            
        else:
            error_msg = f"❌ Error almacenando PDF: {storage_result.get('message')}"
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message([
                    Part(root=TextPart(text=error_msg))
                ])
            )
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
    
    
    async def _handle_store_analysis(
        self,
        updater: TaskUpdater,
        event_queue: EventQueue,
        user_text: str,
        context: RequestContext
    ):
        """
        Maneja el almacenamiento de análisis vinculado a un documento.
        
        Formato esperado:
        "Almacena el análisis: <document_id> <contenido del análisis>"
        """
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([
                Part(root=TextPart(text="💾 Almacenando análisis..."))
            ])
        )
        
        # Extraer document_id y contenido del análisis
        doc_id_pattern = r'([a-f0-9\-]{36})'
        match = re.search(doc_id_pattern, user_text)
        
        if not match:
            error_msg = (
                "❌ No se encontró un document_id válido en el mensaje.\n"
                "Formato esperado: 'Almacena el análisis: <document_id> <contenido>'"
            )
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message([
                    Part(root=TextPart(text=error_msg))
                ])
            )
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        document_id = match.group(1)
        
        # Extraer el contenido del análisis (todo después del UUID)
        analysis_start = match.end()
        analysis_content = user_text[analysis_start:].strip()
        
        if not analysis_content:
            error_msg = "❌ El contenido del análisis está vacío"
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message([
                    Part(root=TextPart(text=error_msg))
                ])
            )
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        logger.info(f"📝 Almacenando análisis para documento: {document_id}")
        logger.info(f"📝 Longitud del análisis: {len(analysis_content)} caracteres")
        
        # Almacenar el análisis
        storage_result = storage_manager.store_analysis(
            document_id=document_id,
            analysis_content=analysis_content,
            analysis_type="general",
            metadata={
                "task_id": context.task_id,
                "origen": "a2a_analysis"
            }
        )
        
        # Preparar respuesta
        if storage_result["status"] == "success":
            html_response = f"""
            <h3>✅ Análisis almacenado exitosamente</h3>
            
            <p><b>Documento ID:</b> {document_id}</p>
            <p><b>Análisis ID:</b> {storage_result['analysis_id']}</p>
            <p><b>Tipo:</b> {storage_result['analysis_type']}</p>
            <p><b>Longitud:</b> {len(analysis_content)} caracteres</p>
            """
            
            json_response = json.dumps(storage_result, indent=2)
            
        else:
            html_response = f"❌ Error almacenando análisis: {storage_result.get('message')}"
            json_response = json.dumps(storage_result, indent=2)
        
        await event_queue.enqueue_event(new_agent_text_message(html_response))
        await updater.add_artifact([Part(root=TextPart(text=json_response))])
        await updater.complete()
    
    
    async def _handle_retrieve_analysis(
        self,
        updater: TaskUpdater,
        event_queue: EventQueue,
        user_text: str
    ):
        """
        Maneja la recuperación de análisis almacenados.
        
        Formatos aceptados:
        - "Recupera el análisis"
        - "Muestra el análisis del documento <document_id>"
        - "Ver todos los análisis"
        """
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([
                Part(root=TextPart(text="🔍 Buscando análisis almacenados..."))
            ])
        )
        
        # Buscar document_id en el texto
        doc_id_pattern = r'([a-f0-9\-]{36})'
        match = re.search(doc_id_pattern, user_text)
        document_id = match.group(1) if match else None
        
        # Recuperar análisis
        if document_id:
            logger.info(f"🔍 Buscando análisis para documento: {document_id}")
            analysis_list = storage_manager.retrieve_analysis(document_id=document_id)
        else:
            logger.info(f"🔍 Buscando todos los análisis")
            analysis_list = storage_manager.retrieve_analysis(limit=20)
        
        # Preparar respuesta
        if not analysis_list:
            html_response = """
            <h3>📭 No se encontraron análisis</h3>
            <p>No hay análisis almacenados que coincidan con tu búsqueda.</p>
            """
        else:
            # Generar HTML con los análisis
            html_parts = [f"<h3>📊 Análisis encontrados ({len(analysis_list)})</h3>"]
            
            for i, analysis in enumerate(analysis_list, 1):
                html_parts.append(f"""
                    <h3>Análisis #{i}</h3>
                    <p><b>Documento ID:</b> {analysis['document_id']}</p>
                    <p><b>Tipo:</b> {analysis['analysis_type']}</p>
                    <p><b>Fecha:</b> {analysis['created_at']}</p>
                    
                    <h3>Contenido:</h3>
                        {analysis['analysis_content']}
                """)
            
            html_response = "\n".join(html_parts)
        
        # Crear JSON response
        json_response = json.dumps({
            "status": "success",
            "operation": "retrieve_analysis",
            "count": len(analysis_list),
            "analysis": analysis_list
        }, indent=2, ensure_ascii=False)
        
        await event_queue.enqueue_event(new_agent_text_message(html_response))
        await updater.add_artifact([Part(root=TextPart(text=json_response))])
        await updater.complete()
    
    
    async def _process_pdf_files(
        self, 
        user_parts: List[Part],
        custom_filename: Optional[str] = None
    ) -> Optional[dict]:
        """
        Procesa archivos PDF de la solicitud.
        MEJORADO: Acepta un nombre personalizado del usuario.
        
        Args:
            user_parts: Partes del mensaje del usuario
            custom_filename: Nombre personalizado proporcionado por el usuario
        
        Returns:
            dict: {'text': str, 'filename': str, 'metadata': dict} o None
        """
        for part in user_parts:
            if isinstance(part, Part):
                root = getattr(part, 'root', None)
                
                if isinstance(root, FilePart):
                    file_obj = getattr(root, 'file', None)
                    
                    if file_obj:
                        file_name = ""
                        file_content = None
                        
                        if isinstance(file_obj, FileWithUri):
                            file_name = getattr(file_obj, 'uri', 'archivo.pdf').split('/')[-1]
                            logger.warning(f"⚠️ FileWithUri detectado: {file_name}. Necesita implementación de descarga.")
                            continue
                        
                        elif isinstance(file_obj, FileWithBytes):
                            # Obtener nombre original del archivo
                            original_filename = getattr(file_obj, 'filename', 'archivo.pdf')
                            file_bytes = getattr(file_obj, 'bytes', None)
                            
                            # DECISIÓN: ¿Usar nombre personalizado o nombre original?
                            if custom_filename:
                                # El usuario especificó un nombre personalizado
                                file_name = custom_filename
                                logger.info(f"📝 Usando nombre personalizado: '{file_name}'")
                            else:
                                # Usar nombre original del archivo
                                file_name = original_filename
                                logger.info(f"📝 Usando nombre original: '{file_name}'")
                            
                            if file_bytes:
                                if isinstance(file_bytes, str):
                                    try:
                                        file_content = base64.b64decode(file_bytes)
                                    except:
                                        file_content = file_bytes.encode('utf-8')
                                else:
                                    file_content = file_bytes
                        
                        if file_name.lower().endswith('.pdf') and file_content:
                            try:
                                if not validate_pdf_content(file_content):
                                    logger.warning(f"⚠️ El archivo '{file_name}' no es un PDF válido")
                                    continue
                                
                                metadata = get_pdf_metadata(file_content)
                                logger.info(f"📊 Metadatos del PDF: {metadata}")
                                
                                text = self.pdf_processor.extract_text_from_pdf(file_content)
                                
                                if text and text.strip():
                                    logger.info(f"✅ Texto extraído de '{file_name}': {len(text)} caracteres")
                                    return {
                                        'filename': file_name,
                                        'text': text,
                                        'metadata': metadata
                                    }
                                else:
                                    logger.warning(f"⚠️ No se pudo extraer texto de '{file_name}'")
                                    
                            except Exception as e:
                                logger.error(f"❌ Error procesando PDF '{file_name}': {str(e)}")
                                raise ValueError(f"Error al procesar PDF: {str(e)}")
        
        return None
    
    
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """Maneja la cancelación de una solicitud."""
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