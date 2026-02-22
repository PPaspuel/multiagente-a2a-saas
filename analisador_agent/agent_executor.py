"""
Ejecutor del agente analizador de contratos para el protocolo A2A.
Conecta el agente CrewAI con el servidor A2A.

FLUJOS SOPORTADOS:
- Flujo 1 (original): PDF adjunto en el mensaje → extrae texto → CrewAI → HTML
- Flujo 2 (nuevo):    nombre o UUID en el texto → recupera de Qdrant → CrewAI → HTML
"""

import logging
import base64
import re
from typing import Optional, List, Dict
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

# Importar el agente CrewAI
from analisador_agent.agent import analyze_contract

# Módulo de recuperación desde Qdrant (solo lectura, para Flujo 2)
from analisador_agent.qdrant_retriever import QdrantRetriever

# Herramientas para procesamiento de PDF
import io
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Clase helper para procesar archivos PDF. Sin cambios respecto al original."""

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"--- PÁGINA {page_num} ---\n{text}")
            full_text = "\n\n".join(text_parts)
            logger.info(f"✅ Texto extraído: {len(full_text)} caracteres de {len(reader.pages)} páginas")
            return full_text
        except Exception as e:
            logger.error(f"❌ Error extrayendo texto del PDF: {str(e)}")
            raise ValueError(f"No se pudo extraer texto del PDF: {str(e)}")

    @staticmethod
    def validate_pdf(pdf_bytes: bytes) -> bool:
        try:
            if pdf_bytes[:4] != b'%PDF':
                return False
            pdf_file = io.BytesIO(pdf_bytes)
            PdfReader(pdf_file)
            return True
        except Exception:
            return False


class ContractAnalyzerExecutor(AgentExecutor):
    """
    Ejecutor del agente analizador de contratos.

    Flujo 1 — PDF adjunto (comportamiento original intacto):
        Recibe PDF → extrae texto con PyPDF2 → CrewAI → HTML

    Flujo 2 — Nombre o UUID en el texto (nuevo):
        Recibe "Analiza el documento X" → busca en Qdrant
        → reconstruye texto desde chunks → CrewAI → HTML

    La detección es automática: si hay PDF adjunto usa Flujo 1,
    si no hay PDF pero hay un identificador en el texto usa Flujo 2.
    """

    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.qdrant = QdrantRetriever()
        logger.info("✅ ContractAnalyzerExecutor inicializado")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:

        logger.info(f"🚀 Iniciando ejecución del agente analizador")
        logger.info(f"📦 Contexto: task_id={context.task_id}, context_id={context.context_id}")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        try:
            # ── PASO 0: Inicializar tarea ──────────────────────────────────────
            if not context.current_task:
                await updater.submit()
            await updater.start_work()

            # ── PASO 1: Extraer input del usuario ─────────────────────────────
            user_text = ""
            user_parts = []

            if hasattr(context, 'message') and context.message:
                message = context.message
                logger.info(f"📨 Mensaje recibido")
                if hasattr(message, 'parts') and message.parts:
                    user_parts = message.parts
                    text_content = []
                    for part in user_parts:
                        if isinstance(part, Part):
                            root = getattr(part, 'root', None)
                            if isinstance(root, TextPart):
                                text_content.append(root.text)
                    user_text = " ".join(text_content) if text_content else ""

            logger.info(f"📝 Texto del usuario: {user_text[:100] if user_text else 'Sin texto'}")
            logger.info(f"📦 Número de partes: {len(user_parts)}")

            # ── PASO 2: Detectar flujo y obtener texto del contrato ───────────
            has_pdf = self._has_pdf_attachment(user_parts)

            if has_pdf:
                # ════════════════════════════════════════════════════
                # FLUJO 1: PDF adjunto — comportamiento original
                # ════════════════════════════════════════════════════
                logger.info("📄 Flujo 1 activado: PDF adjunto detectado")

                await updater.update_status(
                    TaskState.working,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text="📄 Procesando contrato PDF..."))
                    ])
                )

                pdf_text = await self._extract_pdf_text(user_parts)

                if not pdf_text:
                    error_msg = "❌ No se recibió ningún archivo PDF de contrato para analizar."
                    await updater.update_status(
                        TaskState.failed,
                        message=updater.new_agent_message([
                            Part(root=TextPart(text=error_msg))
                        ])
                    )
                    raise ValueError(error_msg)

                logger.info(f"✅ PDF procesado: {len(pdf_text)} caracteres")
                contract_text = pdf_text
                source_info = "PDF adjunto"

            else:
                # ════════════════════════════════════════════════════
                # FLUJO 2: Recuperar desde Qdrant por nombre o UUID
                # ════════════════════════════════════════════════════
                logger.info("🗄️ Flujo 2 activado: sin PDF adjunto, buscando en Qdrant")

                doc_query = self._extract_document_query(user_text)

                if not doc_query:
                    # Sin PDF ni identificador: mostrar documentos disponibles
                    available = self.qdrant.list_documents()
                    if available:
                        await event_queue.enqueue_event(
                            new_agent_text_message(self._render_available_documents(available))
                        )
                        await updater.complete()
                    else:
                        error_msg = (
                            "❌ No se recibió un PDF adjunto ni se especificó un documento.\n"
                            "Opciones:\n"
                            "1. Adjunta un PDF directamente en el mensaje\n"
                            "2. Indica el nombre o ID de un documento ya almacenado"
                        )
                        await updater.update_status(
                            TaskState.failed,
                            message=updater.new_agent_message([
                                Part(root=TextPart(text=error_msg))
                            ])
                        )
                    return

                await updater.update_status(
                    TaskState.working,
                    message=updater.new_agent_message([
                        Part(root=TextPart(
                            text=f"🗄️ Buscando '{doc_query}' en la base de conocimiento..."
                        ))
                    ])
                )

                retrieval = self.qdrant.get_document(doc_query)

                if retrieval["status"] == "not_found":
                    available = self.qdrant.list_documents()
                    await event_queue.enqueue_event(
                        new_agent_text_message(self._render_not_found(doc_query, available))
                    )
                    await updater.complete()
                    return

                elif retrieval["status"] == "ambiguous":
                    await event_queue.enqueue_event(
                        new_agent_text_message(self._render_ambiguous(retrieval))
                    )
                    await updater.complete()
                    return

                elif retrieval["status"] == "error":
                    error_msg = f"❌ Error al recuperar el documento desde Qdrant: {retrieval['message']}"
                    await updater.update_status(
                        TaskState.failed,
                        message=updater.new_agent_message([
                            Part(root=TextPart(text=error_msg))
                        ])
                    )
                    return

                filename = retrieval["filename"]
                num_chunks = retrieval["num_chunks"]
                contract_text = retrieval["content"]
                source_info = f"Qdrant — '{filename}' ({num_chunks} chunks)"
                logger.info(f"✅ Documento recuperado: {source_info}")

                await updater.update_status(
                    TaskState.working,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=(
                            f"✅ Documento encontrado: '{filename}' "
                            f"({num_chunks} fragmentos recuperados)\n"
                            "🔍 Iniciando análisis legal con CrewAI..."
                        )))
                    ])
                )

            # ── PASO 3: Ejecutar análisis con CrewAI (igual en ambos flujos) ──
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="🔍 Analizando derechos, obligaciones y prohibiciones..."))
                ])
            )

            logger.info(f"⚙️ Iniciando análisis con CrewAI — fuente: {source_info}")
            analysis_result = analyze_contract(contract_text)

            logger.info(f"✅ Análisis completado")
            logger.info(f"📊 Resultado: {analysis_result[:200]}...")

            # ── PASO 4: Enviar respuesta ───────────────────────────────────────
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="✅ Análisis completado exitosamente"))
                ])
            )

            await updater.add_artifact([
                Part(root=TextPart(text=analysis_result))
            ])
            await updater.complete()
            await event_queue.enqueue_event(new_agent_text_message(analysis_result))

            logger.info("✅ Ejecución completada exitosamente")

        except Exception as e:
            logger.error(f'❌ Error durante la ejecución: {str(e)}', exc_info=True)

            error_html = f"""
<h3>❌ Error en el Análisis</h3>
<p><b>Operación:</b> Análisis de Contrato</p>
<p><b>Error:</b> {str(e)}</p>
<p><b>Tipo:</b> {type(e).__name__}</p>
"""
            try:
                await updater.fail(
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=error_html))
                    ])
                )
            except Exception:
                await event_queue.enqueue_event(new_agent_text_message(error_html))

            raise ServerError(error=InternalError()) from e

    # ── Métodos de detección ───────────────────────────────────────────────────

    def _has_pdf_attachment(self, user_parts: List[Part]) -> bool:
        """Retorna True si el mensaje contiene al menos un archivo PDF adjunto."""
        for part in user_parts:
            if isinstance(part, Part):
                root = getattr(part, 'root', None)
                if isinstance(root, FilePart):
                    file_obj = getattr(root, 'file', None)
                    if file_obj:
                        name = (
                            getattr(file_obj, 'filename', '') or
                            getattr(file_obj, 'uri', '') or
                            getattr(file_obj, 'name', '') or
                            ''
                        ).lower()
                        mime = getattr(file_obj, 'mime_type', '') or ''
                        if name.endswith('.pdf') or mime == 'application/pdf':
                            return True
        return False

    def _extract_document_query(self, user_text: str) -> Optional[str]:
        """
        Extrae el identificador del documento desde el texto del usuario.
        Orden de detección:
        1. UUID completo (document_id exacto)
        2. Nombre de archivo con extensión .pdf
        3. Texto entre comillas simples o dobles
        4. Nombre tras palabras clave de análisis
        """
        if not user_text:
            return None

        # 1. UUID (document_id)
        m = re.search(
            r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',
            user_text, re.IGNORECASE
        )
        if m:
            return m.group(0)

        # 2. Nombre con extensión .pdf
        m = re.search(r'[\w\-\.]+\.pdf', user_text, re.IGNORECASE)
        if m:
            return m.group(0)

        # 3. Nombre entre comillas
        m = re.search(r'["\']([^"\']{3,})["\']', user_text)
        if m:
            return m.group(1)

        # 4. Nombre tras palabras clave de análisis
        m = re.search(
            r'(?:analiza(?:r)?|revisa(?:r)?|examina(?:r)?|procesa(?:r)?)'
            r'\s+(?:el\s+)?(?:documento|contrato|archivo)?\s*'
            r'["\']?([A-Za-z0-9_\-\s]{3,50})["\']?',
            user_text, re.IGNORECASE
        )
        if m:
            candidate = m.group(1).strip()
            generic = {"este", "el", "la", "un", "una", "contrato", "documento", "archivo", "pdf"}
            if candidate.lower() not in generic:
                return candidate

        return None

    # ── Métodos de renderizado HTML ────────────────────────────────────────────

    def _render_available_documents(self, documents: List[Dict]) -> str:
        items = "".join([
            f"<li><b>{d['filename']}</b><br>"
            f"<b>ID:</b> <code>{d['document_id']}</code><br>"
            f"<b>Almacenado:</b> {d['stored_at'][:10]}<br>"
            f"<b>Chunks:</b> {d['num_chunks']}</li>"
            for d in documents
        ])
        return (
            "<h3>📋 Documentos disponibles para análisis</h3>"
            "<p>Indica el nombre o ID del documento que deseas analizar:</p>"
            f"<ul>{items}</ul>"
            "<p><b>Ejemplo:</b> \"Analiza el documento contrato_servicios.pdf\"</p>"
        )

    def _render_not_found(self, query: str, available: List[Dict]) -> str:
        if available:
            items = "".join([
                f"<li><b>{d['filename']}</b> — <code>{d['document_id'][:8]}...</code></li>"
                for d in available[:5]
            ])
            available_html = f"<h3>📋 Documentos disponibles:</h3><ul>{items}</ul>"
        else:
            available_html = "<p>No hay documentos almacenados en la base de conocimiento.</p>"
        return (
            f"<h3>🔍 Documento no encontrado: '{query}'</h3>"
            f"{available_html}"
            "<p>Verifica el nombre o usa el <b>document_id</b> completo.</p>"
        )

    def _render_ambiguous(self, retrieval: Dict) -> str:
        items = "".join([
            f"<li><b>{m['filename']}</b><br><code>{m['document_id']}</code></li>"
            for m in retrieval.get("matches", [])
        ])
        return (
            "<h3>⚠️ Nombre ambiguo — múltiples documentos encontrados</h3>"
            f"<p>{retrieval['message']}</p>"
            f"<ul>{items}</ul>"
            "<p>Usa el <b>document_id</b> completo para identificar el documento exacto.</p>"
        )

    # ── Método original de extracción de PDF (sin cambios) ────────────────────

    async def _extract_pdf_text(self, user_parts: List[Part]) -> Optional[str]:
        """
        Extrae texto de archivos PDF en la solicitud.
        Método original sin cambios.
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
                            logger.warning(f"⚠️ FileWithUri detectado: {file_name}. Se requiere descarga.")
                            continue

                        elif isinstance(file_obj, FileWithBytes):
                            file_name = getattr(file_obj, 'filename', 'contrato.pdf')
                            file_bytes = getattr(file_obj, 'bytes', None)

                            if file_bytes:
                                if isinstance(file_bytes, str):
                                    try:
                                        file_content = base64.b64decode(file_bytes)
                                    except Exception:
                                        file_content = file_bytes.encode('utf-8')
                                else:
                                    file_content = file_bytes

                        if file_name.lower().endswith('.pdf') and file_content:
                            try:
                                if not self.pdf_processor.validate_pdf(file_content):
                                    logger.warning(f"⚠️ '{file_name}' no es un PDF válido")
                                    continue

                                text = self.pdf_processor.extract_text_from_pdf(file_content)

                                if text and text.strip():
                                    logger.info(f"✅ Texto extraído de '{file_name}': {len(text)} caracteres")
                                    return text
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

            cancel_html = """
<h3>⚠️ Operación Cancelada</h3>
<p><b>Operación:</b> Análisis de Contrato</p>
<p><b>Mensaje:</b> La operación ha sido cancelada por el usuario.</p>
"""
            await event_queue.enqueue_event(new_agent_text_message(cancel_html))

        except Exception as e:
            logger.error(f"❌ Error al cancelar: {str(e)}")
            raise ServerError(error=UnsupportedOperationError(
                details=f"Cancelación fallida: {str(e)}"
            ))