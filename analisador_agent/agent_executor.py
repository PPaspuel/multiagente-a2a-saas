"""
Ejecutor del agente analizador de contratos para el protocolo A2A.
Conecta el agente CrewAI con el servidor A2A.

FLUJO DE PROCESO:
- Flujo: Obtener el nombre o UUID en el texto → recupera de Qdrant → CrewAI → HTML
"""

import logging
import re
import os
import time                    
import csv                    
from datetime import datetime
import json
from pathlib import Path 
from typing import Optional, List, Dict
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    InternalError,
    TextPart,
    UnsupportedOperationError,
    Part,
    TaskState
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from a2a.server.tasks import TaskUpdater

# Importar el agente CrewAI
from analisador_agent.agent import analyze_contract

# Módulo de recuperación desde Qdrant (solo lectura, para Flujo 2)
from analisador_agent.qdrant_retriever import QdrantRetriever


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


def _save_chunks_to_json(retrieval: dict) -> str:
    """
    Guarda los chunks filtrados en JSON local.
    Retorna la ruta del archivo guardado.
    """
    output = {
        "filename":            retrieval["filename"],
        "document_id":         retrieval["document_id"],
        "stored_at":           retrieval["stored_at"],
        "total_chunks_raw":    retrieval["total_chunks_raw"],
        "num_chunks_filtered": retrieval["num_chunks"],
        "chunks": [
            c.strip()
            for c in retrieval["content"].split("\n\n")
            if c.strip()
        ]
    }

    # Nombre del archivo: chunks_<filename>_<fecha>.json
    safe_name = retrieval["filename"].replace(".pdf", "").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"chunks_{safe_name}_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return str(output_path)


class ContractAnalyzerExecutor(AgentExecutor):
    """
    Ejecutor del agente analizador de contratos.

    Flujo — Nombre o UUID en el texto:
        Recibe "Analiza el documento X" → busca en Qdrant
        → reconstruye texto desde chunks → CrewAI → HTML

    La detección es automática: si hay PDF adjunto usa Flujo 1,
    si no hay PDF pero hay un identificador en el texto usa Flujo 2.
    """

    def __init__(self):
        self.qdrant = QdrantRetriever()
        logger.info("✅ ContractAnalyzerExecutor inicializado")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        start_time = time.time() 

        logger.info(f"🚀 Iniciando ejecución del agente analizador")
        logger.info(f"📦 Contexto: task_id={context.task_id}, context_id={context.context_id}")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        try:
            # PASO 0: Inicializar tarea
            if not context.current_task:
                await updater.submit()
            await updater.start_work()

            # PASO 1: Extraer input del usuario
            user_text = ""
            user_parts = []

            if hasattr(context, 'message') and context.message:
                message = context.message
                logger.info(f"📨 Mensaje recibido")
                if hasattr(message, 'parts') and message.parts:
                    user_parts = message.parts
                    
                    # Recopilar todas las parts de texto
                    text_parts = []
                    for part in user_parts:
                        if isinstance(part, Part):
                            root = getattr(part, 'root', None)
                            if isinstance(root, TextPart):
                                text_parts.append(root.text)
                    
                    # Filtrar historial inyectado por A2A y quedarse solo con la instrucción actual
                    actual_instruction_parts = [
                        t for t in text_parts
                        if not t.startswith("For context:")
                        and not (t.startswith("[") and ("] called tool" in t or "] said:" in t or "] `" in t))
                    ]
                    
                    # Tomar la última instrucción real
                    user_text = actual_instruction_parts[-1] if actual_instruction_parts else ""

            logger.info(f"📝 Texto del usuario: {user_text[:100] if user_text else 'Sin texto'}")
            logger.info(f"📦 Número de partes: {len(user_parts)}")

            # PASO 2: Detectar flujo y obtener texto del contrato
            logger.info("🗄️ Buscando documento en Qdrant")

            doc_query = self._extract_document_query(user_text)

            if not doc_query:
                # Mostrar documentos disponibles
                available = self.qdrant.list_documents()
                if available:
                    await event_queue.enqueue_event(
                        new_agent_text_message(self._render_available_documents(available))
                    )
                    await updater.complete()
                    _save_metric("analizador", "no_doc_query", "-",
                        time.time() - start_time, "no_documento")  
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
                    _save_metric("analizador", "no_doc_query", "-",
                        time.time() - start_time, "error")         
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
                _save_metric("analizador", "analyze_contract", doc_query,
                    time.time() - start_time, "not_found")         
                return

            elif retrieval["status"] == "ambiguous":
                await event_queue.enqueue_event(
                    new_agent_text_message(self._render_ambiguous(retrieval))
                )
                await updater.complete()
                _save_metric("analizador", "analyze_contract", doc_query,
                    time.time() - start_time, "ambiguous")        
                return

            elif retrieval["status"] == "error":
                error_msg = f"❌ Error al recuperar el documento desde Qdrant: {retrieval['message']}"
                await updater.update_status(
                    TaskState.failed,
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=error_msg))
                    ])
                )
                _save_metric("analizador", "analyze_contract", doc_query,
                    time.time() - start_time, "error")             
                return

            filename = retrieval["filename"]
            num_chunks = retrieval["num_chunks"]
            contract_text = retrieval["content"]
            document_id = retrieval.get("document_id", doc_query) 
            source_info = f"Qdrant — '{filename}' ({num_chunks} chunks)"
            logger.info(f"✅ Documento recuperado: {source_info}")

            # ✅ AGREGAR AQUÍ — una sola línea
            chunks_file = _save_chunks_to_json(retrieval)
            logger.info(f"💾 Chunks filtrados guardados en: {chunks_file}")

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

            # PASO 3: Ejecutar análisis con CrewAI (igual en ambos flujos)
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

            store_notice = (
                f"<p>💾 Si deseas almacenar este análisis, copia la siguiente instrucción "
                f"junto con el análisis apartir de los Derechos y pégala en la bandeja de entrada:<br>"
                f"<b>Almacena el análisis del documento {document_id}:</b></p>"
            )
            final_result = store_notice + analysis_result

            # PASO 4: Enviar respuesta
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="✅ Análisis completado exitosamente"))
                ])
            )

            await updater.add_artifact([
                Part(root=TextPart(text=final_result))
            ])
            
            # ← FALTA ESTO:
            _save_metric("analizador", "analyze_contract", filename,
                        time.time() - start_time, "success")

            await updater.complete()
            await event_queue.enqueue_event(new_agent_text_message(final_result))

            logger.info("✅ Ejecución completada exitosamente")

        except Exception as e:
            logger.error(f'❌ Error durante la ejecución: {str(e)}', exc_info=True)

            # ← FALTA ESTO:
            _save_metric("analizador", "analyze_contract", "-",
                        time.time() - start_time, "error")

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

    # Métodos de detección y renderizado de resultados
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
            r'(?:llamado|denominado|nombrado|con nombre)?\s*'
            r'["\']?([A-Za-z0-9_\-\.]{3,50})["\']?',
            user_text, re.IGNORECASE
        )
        if m:
            candidate = m.group(1).strip()
            generic = {"este", "el", "la", "un", "una", "contrato", "documento", "archivo", "pdf"}
            if candidate.lower() not in generic:
                return candidate

        return None

    # Métodos de renderizado HTML 

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
                f"<li><b>{d['filename']}</b> — <code>{d['document_id']}</code></li>"
                for d in available[:5]
            ])
            available_html = f"<h3>📋 Documentos disponibles:</h3><ul>{items}</ul>"
        else:
            available_html = "<p>No hay documentos almacenados en la base de conocimiento.</p>"
        return (
            f"<h3>🔍 Documento no encontrado: '{query}'</h3>"
            f"{available_html}"
            f"<p>Verifica el nombre o usa el <b>document_id</b> completo.</p>"
            f"<p>Recuerda colocar la extensión .pdf junto con el nombre del documento.</p>"
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