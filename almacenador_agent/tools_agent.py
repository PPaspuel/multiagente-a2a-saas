"""
Herramientas personalizadas para el agente almacenador.
VERSIÓN MEJORADA:
- Soporte para análisis en ResponseFormatter
- Nuevas funciones de formateo HTML
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import PyPDF2
import io

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Clase para procesar archivos PDF y extraer su contenido de texto.
    """
    
    @staticmethod
    def extract_text_from_pdf(pdf_content: bytes) -> str:
        """
        Extrae todo el texto de un archivo PDF.
        
        Args:
            pdf_content: Contenido del PDF en bytes
            
        Returns:
            str: Texto extraído del PDF, página por página
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_pages = []
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    text_pages.append(f"--- Página {page_num + 1} ---\n{text}")
                    
            full_text = "\n\n".join(text_pages)
            
            logger.info(f"✓ PDF procesado exitosamente: {len(pdf_reader.pages)} páginas")
            return full_text
            
        except Exception as e:
            logger.error(f"✗ Error al extraer texto del PDF: {str(e)}")
            raise ValueError(f"No se pudo procesar el PDF: {str(e)}")
    
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Divide el texto en fragmentos (chunks) para almacenar en Qdrant.
        
        Args:
            text: Texto completo a fragmentar
            chunk_size: Tamaño máximo de cada fragmento en caracteres
            overlap: Cantidad de caracteres que se solapan entre fragmentos
            
        Returns:
            List[str]: Lista de fragmentos de texto
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start = end - overlap
            
        logger.info(f"✓ Texto fragmentado en {len(chunks)} chunks")
        return chunks


class ResponseFormatter:
    """
    Clase para formatear respuestas en JSON y HTML estructurado.
    """
    
    @staticmethod
    def format_success_response(
        operation: str,
        data: Dict[str, Any],
        message: str = "Operación exitosa"
    ) -> str:
        """
        Crea una respuesta JSON exitosa.
        
        Args:
            operation: Tipo de operación realizada
            data: Datos relevantes de la operación
            message: Mensaje descriptivo
            
        Returns:
            str: JSON formateado como string
        """
        response = {
            "status": "success",
            "operation": operation,
            "message": message,
            "data": data,
            "timestamp": None
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    
    @staticmethod
    def format_error_response(
        operation: str,
        error_message: str,
        error_type: str = "ProcessingError"
    ) -> str:
        """
        Crea una respuesta JSON de error.
        
        Args:
            operation: Tipo de operación que falló
            error_message: Descripción del error
            error_type: Tipo de error
            
        Returns:
            str: JSON formateado como string
        """
        response = {
            "status": "error",
            "operation": operation,
            "error": {
                "type": error_type,
                "message": error_message
            }
        }
        
        return json.dumps(response, ensure_ascii=False, indent=2)
    
    
    @staticmethod
    def format_storage_response(
        num_chunks: int,
        total_characters: int,
        collection_name: str,
        document_id: Optional[str] = None,
        was_updated: bool = False
    ) -> str:
        """
        Formatea la respuesta después de almacenar en Qdrant.
        
        Args:
            num_chunks: Número de fragmentos almacenados
            total_characters: Total de caracteres procesados
            collection_name: Nombre de la colección en Qdrant
            document_id: ID del documento almacenado
            was_updated: Si fue una actualización de documento existente
            
        Returns:
            str: JSON con información del almacenamiento
        """
        data = {
            "chunks_stored": num_chunks,
            "total_characters": total_characters,
            "collection": collection_name,
            "document_id": document_id,
            "was_updated": was_updated,
            "storage_location": "qdrant_local"
        }
        
        action = "actualizado" if was_updated else "almacenado"
        message = f"PDF {action} exitosamente en {num_chunks} fragmentos"
        
        return ResponseFormatter.format_success_response(
            operation="store_pdf",
            data=data,
            message=message
        )
    
    
    @staticmethod
    def render_storage_response_html(response: dict) -> str:
        """
        Renderiza la respuesta de almacenamiento en HTML.
        
        Args:
            response: Diccionario con la respuesta
            
        Returns:
            str: HTML formateado
        """
        was_updated = response.get("data", {}).get("was_updated", False)
        icon = "🔄" if was_updated else "📄"
        action = "Actualización" if was_updated else "Almacenamiento"
        
        return f"""
            <h3>{icon} Reporte de {action.lower()}</h3>

            <p><b>Estado:</b> {response.get("status")}</p>
            <p><b>Operación:</b> {response.get("operation")}</p>

            <p>{response.get("message")}</p>

            <h3>📊 Detalles del proceso</h3>
            <ul>
                <li><b>Fragmentos almacenados:</b> {response["data"]["chunks_stored"]}</li>
                <li><b>Total de caracteres:</b> {response["data"]["total_characters"]}</li>
                <li><b>Colección:</b> {response["data"]["collection"]}</li>
                <li><b>Tipo:</b> {"Actualización de documento existente" if was_updated else "Nuevo documento"}</li>
                <li>
                    <b>Document ID:</b>
                    <code>{response["data"].get("document_id", "N/A")}</code>
                </li>
            </ul>
        """
    
    
    @staticmethod
    def render_analysis_response_html(
        analysis_list: List[Dict[str, Any]],
        document_id: Optional[str] = None
    ) -> str:
        """
        Renderiza una lista de análisis en HTML.
        
        Args:
            analysis_list: Lista de análisis recuperados
            document_id: ID del documento (opcional)
            
        Returns:
            str: HTML formateado
        """
        if not analysis_list:
            return """
            <h3>📭 No se encontraron análisis</h3>
            <p>No hay análisis almacenados para este documento.</p>
            """
        
        html_parts = [
            f"<h3>📊 Análisis encontrados ({len(analysis_list)})</h3>"
        ]
        
        if document_id:
            html_parts.append(f"<p><b>Document ID:</b> {document_id[:16]}...</p>")
        
        for i, analysis in enumerate(analysis_list, 1):
            html_parts.append(f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 8px; background: #fafafa;">
                <h4>🔍 Análisis #{i}</h4>
                <p><b>ID:</b> {analysis['analysis_id'][:16]}...</p>
                <p><b>Documento:</b> {analysis['document_id'][:16]}...</p>
                <p><b>Tipo:</b> {analysis['analysis_type']}</p>
                <p><b>Fecha:</b> {analysis['created_at']}</p>
                
                <h5>📝 Contenido:</h5>
                <div style="background: white; padding: 12px; border-radius: 4px; border-left: 3px solid #4CAF50;">
                    {analysis['analysis_content']}
                </div>
            </div>
            """)
        
        return "\n".join(html_parts)


# FUNCIONES DE UTILIDAD
def validate_pdf_content(content: bytes) -> bool:
    """
    Valida que el contenido sea un PDF válido.
    
    Args:
        content: Bytes del archivo a validar
        
    Returns:
        bool: True si es un PDF válido
    """
    try:
        pdf_signature = b'%PDF'
        return content.startswith(pdf_signature)
    except Exception:
        return False


def get_pdf_metadata(pdf_content: bytes) -> Dict[str, Any]:
    """
    Extrae metadatos del PDF.
    
    Args:
        pdf_content: Contenido del PDF en bytes
        
    Returns:
        Dict con metadatos del PDF
    """
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        metadata = {
            "num_pages": len(pdf_reader.pages),
            "has_text": False
        }
        
        if len(pdf_reader.pages) > 0:
            first_page_text = pdf_reader.pages[0].extract_text()
            metadata["has_text"] = bool(first_page_text.strip())
        
        if pdf_reader.metadata:
            metadata.update({
                "title": pdf_reader.metadata.get('/Title', 'N/A'),
                "author": pdf_reader.metadata.get('/Author', 'N/A'),
                "subject": pdf_reader.metadata.get('/Subject', 'N/A'),
            })
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error al extraer metadatos: {str(e)}")
        return {"num_pages": 0, "has_text": False, "error": str(e)}


def extract_document_id_from_text(text: str) -> Optional[str]:
    """
    Extrae un UUID (document_id) del texto usando expresiones regulares.
    
    Args:
        text: Texto donde buscar el UUID
        
    Returns:
        str: UUID encontrado o None
    """
    import re
    uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
    match = re.search(uuid_pattern, text, re.IGNORECASE)
    return match.group(0) if match else None