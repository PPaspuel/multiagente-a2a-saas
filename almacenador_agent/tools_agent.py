"""
Herramientas personalizadas para el agente almacenador.
Estas funciones permiten procesar archivos PDF y estructurar respuestas.
"""

import json
import logging
from typing import Dict, List, Any
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
            
        Ejemplo:
            >>> pdf_bytes = open('documento.pdf', 'rb').read()
            >>> texto = PDFProcessor.extract_text_from_pdf(pdf_bytes)
        """
        try:
            # Crear un objeto similar a un archivo desde los bytes
            pdf_file = io.BytesIO(pdf_content)
            
            # Leer el PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Lista para almacenar el texto de cada página
            text_pages = []
            
            # Extraer texto de cada página
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():  # Solo agregar si hay texto
                    text_pages.append(f"--- Página {page_num + 1} ---\n{text}")
                    
            # Unir todo el texto
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
            
        Explicación:
            - Los chunks permiten búsquedas más precisas en bases vectoriales
            - El overlap asegura que no se pierda contexto en los límites
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calcular el final del chunk
            end = start + chunk_size
            
            # Extraer el fragmento
            chunk = text[start:end]
            
            # Agregar a la lista si no está vacío
            if chunk.strip():
                chunks.append(chunk)
            
            # Mover el inicio con overlap
            start = end - overlap
            
        logger.info(f"✓ Texto fragmentado en {len(chunks)} chunks")
        return chunks


class ResponseFormatter:
    """
    Clase para formatear respuestas en JSON estructurado.
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
            operation: Tipo de operación realizada ('store', 'retrieve', 'extract')
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
            "timestamp": None  # Se puede agregar timestamp si se necesita
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
        collection_name: str
    ) -> str:
        """
        Formatea la respuesta después de almacenar en Qdrant.
        
        Args:
            num_chunks: Número de fragmentos almacenados
            total_characters: Total de caracteres procesados
            collection_name: Nombre de la colección en Qdrant
            
        Returns:
            str: HTML con información del almacenamiento
        """
        data = {
            "chunks_stored": num_chunks,
            "total_characters": total_characters,
            "collection": collection_name,
            "storage_location": "qdrant_remote"
        }
        
        return ResponseFormatter.format_success_response(
            operation="store_pdf",
            data=data,
            message=f"PDF almacenado exitosamente en {num_chunks} fragmentos"
        )
    
    def render_storage_response_html(self, response: dict) -> str:
        return f"""
            <h3>📄 Reporte de almacenamiento</h3>

            <p><b>Estado:</b> {response.get("status")}</p>
            <p><b>Operación:</b> {response.get("operation")}</p>

            <p>{response.get("message")}</p>

            <h3>📊 Detalles del proceso</h3>
            <ul>
                <li><b>Fragmentos almacenados:</b> {response["data"]["chunks_stored"]}</li>
                <li><b>Total de caracteres:</b> {response["data"]["total_characters"]}</li>
                <li><b>Colección:</b> {response["data"]["collection"]}</li>
            </ul>
        """

# FUNCIONES DE UTILIDAD
def validate_pdf_content(content: bytes) -> bool:
    """
    Valida que el contenido sea un PDF válido.
    
    Args:
        content: Bytes del archivo a validar
        
    Returns:
        bool: True si es un PDF válido, False en caso contrario
    """
    try:
        # Los PDFs comienzan con "%PDF"
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
        
        # Verificar si tiene texto extraíble
        if len(pdf_reader.pages) > 0:
            first_page_text = pdf_reader.pages[0].extract_text()
            metadata["has_text"] = bool(first_page_text.strip())
        
        # Intentar obtener metadatos del documento
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