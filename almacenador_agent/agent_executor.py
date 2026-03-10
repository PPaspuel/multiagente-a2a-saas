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
import os
import re
import time                    
import csv                    
from datetime import datetime  
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

def _save_metric(agente, operacion, documento, elapsed, status):
    file_exists = os.path.exists("metrics.csv")
    with open("metrics.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp","agente","operacion","documento","tiempo_s","status"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            agente, operacion, documento, f"{elapsed:.2f}", status
        ])

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
                    
                    # Recopilar todas las parts de texto
                    text_parts = []
                    for part in user_parts:
                        if isinstance(part, Part):
                            root = getattr(part, 'root', None)
                            if isinstance(root, TextPart):
                                text_parts.append(root.text)
                    
                    # La instrucción real es la ÚLTIMA part de texto.
                    # Las anteriores son historial inyectado por el protocolo A2A.
                    # Filtramos parts de contexto para quedarnos solo con la instrucción actual.
                    actual_instruction_parts = [
                        t for t in text_parts
                        if not t.startswith("For context:")
                        and not (t.startswith("[") and ("] called tool" in t or "] said:" in t or "] `" in t))
                    ]
                    
                    # Tomar la última instrucción real
                    user_text = actual_instruction_parts[-1] if actual_instruction_parts else ""
                        
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

            elif operation_type == "get_stats":
                await self._handle_get_stats(updater, event_queue, user_text)
            
            elif operation_type == "get_analyzed_docs":  
                await self._handle_get_analyzed_docs(updater, event_queue)

            else:
                # Operación no reconocida
                error_msg = (
                    "❌ No se pudo determinar la operación solicitada.\n\n"
                    "Operaciones disponibles:\n"
                    "1. 📄 Almacenar PDF: Envía un archivo PDF\n"
                    "2. 💾 Almacenar análisis: Envía texto con 'almacena el análisis' + document_id + análisis\n"
                    "3. 🔍 Recuperar análisis: Envía 'recupera el análisis' o 'muestra el análisis'\n"
                    "4. 📊 Obtener documentos analizados: Envía 'qué documentos han sido analizados'\n"
                    "5. 📊 Obtener estadísticas: Envía 'cuantos documentos' o 'cuantos análisis'\n"
                    "\nVUELVE A INTENTAR CON UN FORMATO VÁLIDO SI EL PROBLEMA PERSISTE."
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
        
        # Palabras clave para estadísticas
        stats_keywords = [
            "cuantos documentos",
            "cuántos documentos",
            "cuantos análisis",
            "cuántos análisis",
            "cuantos archivos",
            "cuántos archivos",
            "estadísticas",
            "estadisticas",
            "que hay almacenado",
            "qué hay almacenado"
        ]

        # Palabras clave para documentos analizados
        analyzed_docs_keywords = [
            "documentos analizados",
            "documento analizado",
            "tienen análisis",
            "tienen analisis",
            "tiene análisis",
            "tiene analisis",
            "han sido analizados",
            "ya fue analizado",
            "cuáles tienen análisis",
            "cuales tienen analisis",
            "que documentos han",  
            "qué documentos han",
            "documentos tienen analisis",
            "documentos tienen análisis",
            "con análisis",
            "con analisis"
        ]


        # Decisión de operación
        if has_pdf:
            return "store_pdf"
        
        elif any(keyword in user_text_lower for keyword in store_analysis_keywords):
            return "store_analysis"
        
        elif any(keyword in user_text_lower for keyword in retrieve_analysis_keywords):
            return "retrieve_analysis"
        
        elif any(keyword in user_text_lower for keyword in stats_keywords):
            return "get_stats"
        
        elif any(keyword in user_text_lower for keyword in analyzed_docs_keywords):
            return "get_analyzed_docs"
        
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
            r'(?:con el nombre|con nombre)\s+(?:de\s+)?["\']?([^"\'.\n]+?)["\']?(?:\.|$|\n)',
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
        start_time = time.time()  # Calcular tiempo y guardar métrica
        # Extraer nombre personalizado del usuario
        custom_filename = self._extract_custom_filename(user_text)

        if not custom_filename:
            error_msg = (
                "⚠️ *Nombre de documento requerido*\n\n"
                "Para almacenar un PDF debes especificar un nombre para identificarlo.\n\n"
                "📌 *Ejemplo de uso correcto:*\n"
                "  _'Almacena el documento con el nombre contrato_2024.'_\n\n"
                "Por favor, reenvía tu solicitud incluyendo un nombre para el documento.\n"
                "No olvides colocar el punto final después del nombre para asegurar que se detecte correctamente.\n"
                "NOTA: Recuerda que si el documento ya fue almacenado previamente, solo se actualizará su contenido sin crear duplicados,\n"
                "pero tomará el nombre que le hayas dado en esta solicitud para futuras referencias."
            )
            logger.warning("⚠️ El usuario no proporcionó un nombre para el documento")
            
            # Primero notifica el error al cliente vía event_queue
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            
            # Luego cierra la tarea en estado "failed" correctamente
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message([
                    Part(root=TextPart(text=error_msg))
                ])
            )
            return
        
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
        
        # Fragmentar el texto con chunking semántico
        # Agrupa oraciones por coherencia temática en lugar de cortar por caracteres
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([
                Part(root=TextPart(text="🧠 Analizando estructura semántica del documento..."))
            ])
        )
        
        try:
            chunks = self.pdf_processor.semantic_chunking(
                pdf_result['text'],
                similarity_threshold=0.5  # Ajustar según el dominio: más alto = chunks más pequeños
            )
        except ImportError:
            # Fallback a chunking por caracteres si las dependencias no están instaladas
            logger.warning("⚠️ Usando chunking por caracteres como fallback")
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

            # Calcular tiempo y guardar métrica:
            _save_metric("almacenador", "store_pdf", pdf_result['filename'], 
                        time.time() - start_time, "success") 
            await updater.complete()
            
        else:
            error_msg = f"❌ Error almacenando PDF: {storage_result.get('message')}"
            await updater.update_status(
                TaskState.failed,
                message=updater.new_agent_message([
                    Part(root=TextPart(text=error_msg))
                ])
            )

            # Calcular tiempo y guardar métrica:
            _save_metric("almacenador", "store_pdf", pdf_result.get('filename','-'),
                        time.time() - start_time, "error")   
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
        start_time = time.time() 
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
        filename = storage_manager.get_filename_by_document_id(document_id)
        storage_result = storage_manager.store_analysis(
            document_id=document_id,
            analysis_content=analysis_content,
            analysis_type="general",
            filename=filename,
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
        
        # Calcular tiempo y guardar métrica:
        _save_metric("almacenador", "store_analysis", document_id,
                    time.time() - start_time, 
                    "success" if storage_result["status"] == "success" else "error")
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
        start_time = time.time() 
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
        
        # NUEVO: Si no hay UUID, intentar resolver por nombre de archivo
        if not document_id:
            filename_pattern = r'[\w\-]+\.pdf'
            filename_match = re.search(filename_pattern, user_text, re.IGNORECASE)
            if filename_match:
                filename = filename_match.group(0)
                logger.info(f"🔍 Resolviendo document_id por nombre: {filename}")
                document_id = storage_manager.get_document_id_by_filename(filename)

        # Recuperar análisis
        if document_id:
            logger.info(f"🔍 Buscando análisis para documento: {document_id}")
            analysis_list = storage_manager.retrieve_analysis(document_id=document_id)
        else:
            logger.info(f"🔍 Buscando todos los análisis")
            analysis_list = storage_manager.retrieve_analysis(limit=20)
        
        # Preparar respuesta
        if not analysis_list:
            text_response = (
                "📭 No se encontraron análisis\n"
                "\nNo hay análisis almacenados que coincidan con tu búsqueda."
            )
            json_response = json.dumps({
                "status": "success",
                "operation": "retrieve_analysis",
                "count": 0,
                "analysis": []
            }, indent=2, ensure_ascii=False)
        else:
            # Generar respuesta en texto plano estructurado
            text_parts = [f"✅ Se encontraron {len(analysis_list)} análisis\n"]

            for i, analysis in enumerate(analysis_list, 1):
                text_parts.append(
                    f"\n📊 Análisis #{i}\n"
                    f"\n{'─'*40}\n"
                    f"\n📄 Documento:  {analysis.get('filename', 'No disponible')}\n"
                    f"\n🆔 ID:         {analysis.get('document_id', 'N/A')}\n"
                    f"\n🏷️  Tipo:       {analysis.get('analysis_type', 'N/A')}\n"
                    f"\n📅 Fecha:      {analysis.get('created_at', 'N/A')}\n"
                    f"\n📝 Contenido:\n{analysis.get('analysis_content', '')}\n"
                    f"\n{'─'*40}"
                )

            text_response = "\n".join(text_parts)
            json_response = json.dumps({
                "status": "success",
                "operation": "retrieve_analysis",
                "count": len(analysis_list),
                "analysis": analysis_list
            }, indent=2, ensure_ascii=False)

        # Siempre se ejecuta, sin importar si hay análisis o no
        await event_queue.enqueue_event(new_agent_text_message(text_response))
        await updater.add_artifact([Part(root=TextPart(text=json_response))])
        
        # Calcular tiempo y guardar métrica:
        _save_metric("almacenador", "retrieve_analysis", 
                    document_id if document_id else "todos",
                    time.time() - start_time, "success")
        await updater.complete()
    
    async def _handle_get_stats(
    self,
    updater: TaskUpdater,
    event_queue: EventQueue,
    user_text: str
    ):
        start_time = time.time() 
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([
                Part(root=TextPart(text="📊 Consultando estadísticas..."))
            ])
        )

        stats = storage_manager.get_stats()

        if stats["status"] == "error":
            await event_queue.enqueue_event(new_agent_text_message(
                f"❌ Error obteniendo estadísticas: {stats['message']}"
            ))
            await updater.complete()
            return

        user_text_lower = user_text.lower()
        solo_analisis = any(k in user_text_lower for k in ["análisis", "analisis"])
        solo_documentos = any(k in user_text_lower for k in ["documento", "archivo"])

        if solo_analisis and not solo_documentos:
            # El usuario preguntó solo por análisis
            text_response = (
                f"🔍 Análisis guardados\n"
                f"\n{'─'*40}\n"
                f"\n📊 Total de análisis almacenados: {stats['analysis']['total']}\n"
                f"\n{'─'*40}"
            )

        elif solo_documentos and not solo_analisis:
            # El usuario preguntó solo por documentos
            doc_list = ""
            for i, doc in enumerate(stats["documents"]["list"], 1):
                doc_list += (
                    f"\n{i}. {doc['filename']}\n"
                    f"   ID: {doc['document_id']}\n"
                    f"   Almacenado: {doc['stored_at']}\n"
                )
            text_response = (
                f"📄 Documentos almacenados\n"
                f"\n{'─'*40}\n"
                f"\n📄 Total de documentos únicos: {stats['documents']['total']}\n"
                f"\n{'─'*40}\n\n"
                f"📁 Lista de documentos:\n"
                f"{doc_list if doc_list else '  (ninguno)'}"
            )

        else:
            # El usuario preguntó por todo o no fue específico
            doc_list = ""
            for i, doc in enumerate(stats["documents"]["list"], 1):
                doc_list += (
                    f"\n{i}. {doc['filename']}\n"
                    f"   ID: {doc['document_id']}\n"
                    f"   Almacenado: {doc['stored_at']}\n"
                )
            text_response = (
                f"📊 Estadísticas del almacenamiento\n"
                f"\n{'─'*40}\n"
                f"\n📄 Documentos únicos:  {stats['documents']['total']}\n"
                f"\n🔍 Análisis guardados: {stats['analysis']['total']}\n"
                f"\n🧩 Chunks totales:     {stats['chunks']['total']}\n"
                f"\n{'─'*40}\n\n"
                f"📁 Documentos almacenados:\n"
                f"{doc_list if doc_list else '  (ninguno)'}"
            )

        await event_queue.enqueue_event(new_agent_text_message(text_response))
        await updater.add_artifact([Part(root=TextPart(text=json.dumps(stats, indent=2, ensure_ascii=False)))])
        
        # Calcular tiempo y guardar métrica:
        _save_metric("almacenador", "get_stats", "-",
                    time.time() - start_time, "success")
        await updater.complete()


    async def _handle_get_analyzed_docs(self, updater, event_queue):
        
        start_time = time.time() 
        result = storage_manager.get_analyzed_documents()

        if result["status"] == "error":
            await event_queue.enqueue_event(new_agent_text_message(
                f"❌ Error: {result['message']}"
            ))
            await updater.complete()
            return

        if result["total"] == 0:
            text_response = "📭 Ningún documento tiene análisis almacenado aún."
        else:
            doc_list = ""
            for i, doc in enumerate(result["documents"], 1):
                doc_list += (
                    f"\n{i}. {doc['filename']}\n"
                    f"   ID: {doc['document_id']}\n"
                    f"   Análisis guardados: {doc['total_analyses']}\n"
                )
            text_response = (
                f"📊 Documentos con análisis almacenados\n"
                f"\n{'─'*40}\n"
                f"\n✅ Total: {result['total']} documento(s) analizado(s)\n"
                f"\n{'─'*40}\n"
                f"{doc_list}"
            )

        await event_queue.enqueue_event(new_agent_text_message(text_response))
        
        # Calcular tiempo y guardar métrica:
        _save_metric("almacenador", "get_analyzed_docs", "-",
                    time.time() - start_time, "success")
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