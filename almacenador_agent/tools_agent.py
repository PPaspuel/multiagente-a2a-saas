"""
Herramientas personalizadas para el agente almacenador.
VERSIÓN MEJORADA:
- Soporte para análisis en ResponseFormatter
- Nuevas funciones de formateo HTML
"""

import json
import logging
import re
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
    def semantic_chunking(text: str, similarity_threshold: float = 0.5) -> List[str]:
        """
        Divide el texto en fragmentos semánticamente coherentes usando
        sentence-transformers y similitud coseno entre oraciones consecutivas.

        Los chunks se forman agrupando oraciones mientras su similitud semántica
        supera el umbral. Cuando la similitud cae, se abre un nuevo chunk.
        También filtra fragmentos que son solo títulos, numeraciones o texto vacío.

        Args:
            text: Texto completo a fragmentar (idealmente texto extraído de PDF).
            similarity_threshold: Umbral de similitud coseno entre [0, 1].
                - Valores altos (ej: 0.7-0.9): chunks más pequeños y temáticamente homogéneos.
                - Valores bajos (ej: 0.3-0.5): chunks más grandes con más variedad temática.
                Se recomienda comenzar con 0.5 y ajustar según el dominio del documento.

        Returns:
            List[str]: Lista de fragmentos semánticamente coherentes y limpios.
                    Los chunks vacíos, títulos cortos y numeraciones se descartan.

        Raises:
            ImportError: Si sentence-transformers o nltk no están instalados.
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            import nltk
        except ImportError as e:
            logger.error(
                "❌ Dependencias faltantes para chunking semántico. "
                "Instala con: pip install sentence-transformers nltk"
            )
            raise ImportError(
                "sentence-transformers y nltk son requeridos para semantic_chunking. "
                f"Error original: {e}"
            )

        # Descargar tokenizador de oraciones si no está disponible
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("📥 Descargando recursos NLTK (punkt_tab)...")
            nltk.download('punkt_tab', quiet=True)

        # Cargar modelo ligero y eficiente para generar embeddings
        logger.info("🤖 Cargando modelo de embeddings semánticos (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Tokenizar el texto en oraciones individuales
        sentences = nltk.sent_tokenize(text)

        if not sentences:
            logger.warning("⚠️ No se encontraron oraciones en el texto")
            return []

        if len(sentences) == 1:
            return [sentences[0].strip()] if sentences[0].strip() else []

        # Generar embeddings para todas las oraciones en un solo paso (eficiente)
        logger.info(f"🔢 Generando embeddings para {len(sentences)} oraciones...")
        embeddings = model.encode(sentences, show_progress_bar=False)

        # Calcular similitud coseno entre oraciones consecutivas
        similarities = [
            util.cos_sim(embeddings[i], embeddings[i + 1]).item()
            for i in range(len(embeddings) - 1)
        ]

        # Agrupar oraciones en chunks según el umbral de similitud
        # Cuando la similitud cae por debajo del umbral → ruptura semántica → nuevo chunk
        raw_chunks = []
        current_chunk = [sentences[0]]

        for i, score in enumerate(similarities):
            if score < similarity_threshold:
                raw_chunks.append(" ".join(current_chunk))
                current_chunk = []
            current_chunk.append(sentences[i + 1])

        if current_chunk:
            raw_chunks.append(" ".join(current_chunk))

        # Limpieza y filtrado de chunks 
        # Elimina prefijos de numeración al inicio: "1.", "I.", "2.1.", "III." etc.
        prefix_pattern = re.compile(
            r"^\s*(?:(?:[IVXLCDMivxlcdm]+|\d+)\.(?:\d+\.)*\s*)"
        )

        def is_title_or_noise(t: str, min_length: int = 15) -> bool:
            """
            Retorna True si el fragmento es probablemente ruido o un título sin
            contenido informativo suficiente para almacenar en la base vectorial.
            """
            t = t.strip()
            if not t:
                return True
            if len(t) < min_length:
                return True
            alpha_chars = sum(c.isalpha() for c in t)
            if alpha_chars == 0:
                return True  # Solo símbolos/números
            upper_chars = sum(c.isupper() for c in t)
            if upper_chars / len(t) > 0.75:
                return True  # Probable encabezado en mayúsculas
            if re.fullmatch(
                r"^\s*(?:SECTION|ARTICLE|CLAUSE|CAPÍTULO|SECCIÓN|ARTÍCULO)\s+"
                r"(?:[IVXLCDM]+|\d+)\s*$",
                t, re.IGNORECASE
            ):
                return True
            return False

        filtered_chunks = []
        for chunk in raw_chunks:
            cleaned = prefix_pattern.sub("", chunk, count=1).strip()
            if is_title_or_noise(cleaned):
                logger.debug(f"🗑️ Chunk descartado (ruido/título): '{cleaned[:60]}'")
                continue
            filtered_chunks.append(cleaned)

        logger.info(
            f"✓ Chunking semántico: {len(raw_chunks)} chunks crudos → "
            f"{len(filtered_chunks)} chunks válidos almacenables"
        )
        return filtered_chunks


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
                <li><b>Nombre del documento:</b> {response["data"].get("filename", "N/A")}</li>
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
            <div>
                <h3>🔍 Análisis #{i}</h3>
                <p><b>ID:</b> {analysis['analysis_id'][:16]}...</p>
                <p><b>Documento:</b> {analysis['document_id'][:16]}...</p>
                <p><b>Tipo:</b> {analysis['analysis_type']}</p>
                <p><b>Fecha:</b> {analysis['created_at']}</p>
                
                <h3>📝 Contenido:</h3>
                <div>
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