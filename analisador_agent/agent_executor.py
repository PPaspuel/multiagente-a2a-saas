"""
Ejecutor del agente analizador de contratos para el protocolo A2A.
Conecta el agente CrewAI con el servidor A2A.
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

# Importar el agente CrewAI
from analisador_agent.agent import analyze_contract

# Herramientas para procesamiento de PDF
import io
from PyPDF2 import PdfReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Clase helper para procesar archivos PDF.
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """
        Extrae texto de un archivo PDF.
        
        Args:
            pdf_bytes: Contenido del PDF en bytes
            
        Returns:
            str: Texto extraído del PDF
        """
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
        """
        Valida que los bytes sean un PDF válido.
        
        Args:
            pdf_bytes: Contenido a validar
            
        Returns:
            bool: True si es un PDF válido
        """
        try:
            # Verificar signature de PDF
            if pdf_bytes[:4] != b'%PDF':
                return False
            
            # Intentar leer con PyPDF2
            pdf_file = io.BytesIO(pdf_bytes)
            PdfReader(pdf_file)
            return True
            
        except:
            return False


class ContractAnalyzerExecutor(AgentExecutor):
    """
    Ejecutor del agente analizador de contratos.
    
    Flujo de ejecución:
    1. Recibe requests del protocolo A2A
    2. Extrae archivos PDF adjuntos
    3. Extrae texto del PDF usando PyPDF2
    4. Ejecuta el análisis con CrewAI
    5. Devuelve resultados en formato HTML estructurado
    """
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        logger.info("✅ ContractAnalyzerExecutor inicializado")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Ejecuta el agente para analizar el contrato.
        
        Args:
            context: Contexto de la petición con el PDF
            event_queue: Cola para enviar eventos y actualizaciones
        """
        
        logger.info(f"🚀 Iniciando ejecución del agente analizador")
        logger.info(f"📦 Contexto: task_id={context.task_id}, context_id={context.context_id}")
        
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        
        try:
            # ==========================================
            # PASO 0: Inicializar tarea
            # ==========================================
            if not context.current_task:
                await updater.submit()
            await updater.start_work()
            
            # ==========================================
            # PASO 1: Extraer input del usuario
            # ==========================================
            user_text = ""
            user_parts = []
            
            if hasattr(context, 'message') and context.message:
                message = context.message
                logger.info(f"📨 Mensaje recibido")
                
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
            
            logger.info(f"📝 Texto del usuario: {user_text[:100] if user_text else 'Sin texto'}")
            logger.info(f"📦 Número de partes: {len(user_parts)}")
            
            # ==========================================
            # PASO 2: Procesar archivo PDF
            # ==========================================
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
            
            # ==========================================
            # PASO 3: Ejecutar análisis con CrewAI
            # ==========================================
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="🔍 Analizando derechos, obligaciones y prohibiciones..."))
                ])
            )
            
            logger.info("⚙️ Iniciando análisis con CrewAI...")
            analysis_result = analyze_contract(pdf_text)
            
            logger.info(f"✅ Análisis completado")
            logger.info(f"📊 Resultado: {analysis_result[:200]}...")
            
            # ==========================================
            # PASO 4: Preparar respuesta HTML
            # ==========================================
            html_result = analysis_result  # El resultado ya viene en HTML desde CrewAI
            
            # ==========================================
            # PASO 5: Enviar respuesta
            # ==========================================
            await updater.update_status(
                TaskState.working,
                message=updater.new_agent_message([
                    Part(root=TextPart(text="✅ Análisis completado exitosamente"))
                ])
            )
            
            # Enviar resultado HTML como artefacto
            await updater.add_artifact([
                Part(root=TextPart(text=html_result))
            ])
            
            # Completar tarea
            await updater.complete()
            
            # Enviar al event queue
            await event_queue.enqueue_event(new_agent_text_message(html_result))
            
            logger.info("✅ Ejecución completada exitosamente")
            
        except Exception as e:
            logger.error(f'❌ Error durante la ejecución: {str(e)}', exc_info=True)
            
            # Crear respuesta de error en HTML
            error_html = f"""
<h3>❌ Error en el Análisis</h3>
<p><b>Operación:</b> Análisis de Contrato</p>
<p><b>Error:</b> {str(e)}</p>
<p><b>Tipo:</b> {type(e).__name__}</p>
"""
            
            # Actualizar estado como fallido
            try:
                await updater.fail(
                    message=updater.new_agent_message([
                        Part(root=TextPart(text=error_html))
                    ])
                )
            except:
                # Si falla el updater, enviar directamente
                await event_queue.enqueue_event(new_agent_text_message(error_html))
            
            raise ServerError(error=InternalError()) from e
    
    async def _extract_pdf_text(self, user_parts: List[Part]) -> Optional[str]:
        """
        Extrae texto de archivos PDF en la solicitud.
        
        Args:
            user_parts: Lista de partes del mensaje
            
        Returns:
            str: Texto extraído del PDF o None
        """
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
                            logger.warning(f"⚠️ FileWithUri detectado: {file_name}. Se requiere descarga.")
                            continue
                        
                        # Manejar FileWithBytes
                        elif isinstance(file_obj, FileWithBytes):
                            file_name = getattr(file_obj, 'filename', 'contrato.pdf')
                            file_bytes = getattr(file_obj, 'bytes', None)
                            
                            if file_bytes:
                                # Decodificar si es base64
                                if isinstance(file_bytes, str):
                                    try:
                                        file_content = base64.b64decode(file_bytes)
                                    except:
                                        file_content = file_bytes.encode('utf-8')
                                else:
                                    file_content = file_bytes
                        
                        # Verificar que sea PDF y extraer texto
                        if file_name.lower().endswith('.pdf') and file_content:
                            try:
                                # Validar PDF
                                if not self.pdf_processor.validate_pdf(file_content):
                                    logger.warning(f"⚠️ '{file_name}' no es un PDF válido")
                                    continue
                                
                                # Extraer texto
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
        """
        Maneja la cancelación de una solicitud.
        """
        logger.warning("⚠️ Cancelación solicitada")
        
        try:
            updater = TaskUpdater(event_queue, context.task_id, context.context_id)
            await updater.cancel()
            
            cancel_html = """
<h3>⚠️ Operación Cancelada</h3>
<p><b>Operación:</b> Análisis de Contrato</p>
<p><b>Mensaje:</b> La operación ha sido cancelada por el usuario.</p>
"""
            
            await event_queue.enqueue_event(
                new_agent_text_message(cancel_html)
            )
            
        except Exception as e:
            logger.error(f"❌ Error al cancelar: {str(e)}")
            raise ServerError(error=UnsupportedOperationError(
                details=f"Cancelación fallida: {str(e)}"
            ))