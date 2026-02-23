"""
Módulo de recuperación de chunks desde Qdrant para el agente analizador.

Este módulo es de SOLO LECTURA — no escribe ni modifica nada en Qdrant.
Su única función es recuperar chunks almacenados por el agente almacenador
y reconstruir el texto del documento para su análisis.

Busca documentos por:
  - document_id (UUID exacto)
  - filename    (nombre del archivo, búsqueda parcial)
"""

import logging
import os
import re
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class QdrantRetriever:
    """
    Cliente de solo lectura para recuperar chunks desde Qdrant.

    Se conecta a la misma instancia de Qdrant que usa el agente almacenador,
    usando las mismas variables de entorno (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME).

    Uso típico:
        retriever = QdrantRetriever()
        result = retriever.get_document_by_name("contrato_servicios")
        if result["status"] == "success":
            texto = result["content"]
    """

    def __init__(self):
        """
        Inicializa la conexión a Qdrant en modo lectura.
        Usa las mismas variables de entorno que el agente almacenador.
        """
        try:
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
            self.collection_name = os.getenv("COLLECTION_NAME", "contratos_saas")

            if not self.collection_name:
                raise ValueError("La variable de entorno COLLECTION_NAME no está definida.")

            self.client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                timeout=10
            )

            # Verificar que la colección existe
            if not self.client.collection_exists(self.collection_name):
                raise RuntimeError(
                    f"La colección '{self.collection_name}' no existe en Qdrant. "
                    "Asegúrate de que el agente almacenador haya procesado al menos un documento."
                )

            self.available = True
            logger.info(
                f"✅ QdrantRetriever conectado — colección: '{self.collection_name}'"
            )

        except Exception as e:
            logger.error(f"❌ QdrantRetriever no pudo conectarse a Qdrant: {e}")
            self.client = None
            self.available = False

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL: entrada unificada para el agente analizador
    # ──────────────────────────────────────────────────────────────────────────

    def get_document(self, query: str) -> Dict[str, Any]:
        """
        Punto de entrada principal. Detecta automáticamente si `query` es
        un UUID (document_id) o un nombre de archivo y ejecuta la búsqueda
        correspondiente.

        Args:
            query: document_id (UUID) o nombre/parte del nombre del archivo.

        Returns:
            Dict con:
              - status:       "success" | "not_found" | "error"
              - content:      texto reconstruido del documento (si status == "success")
              - document_id:  UUID del documento encontrado
              - filename:     nombre del archivo
              - num_chunks:   número de chunks recuperados
              - message:      descripción del resultado (útil para mensajes de error)
        """
        if not self.available:
            return {
                "status": "error",
                "message": "Qdrant no está disponible. Verifica que Docker esté corriendo.",
                "content": None
            }

        # Detectar si es un UUID
        uuid_pattern = re.compile(
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
            re.IGNORECASE
        )
        if uuid_pattern.match(query.strip()):
            logger.info(f"🔍 Buscando por document_id: {query.strip()}")
            return self.get_document_by_id(query.strip())
        else:
            logger.info(f"🔍 Buscando por nombre de archivo: '{query}'")
            return self.get_document_by_name(query.strip())

    # ──────────────────────────────────────────────────────────────────────────
    # BÚSQUEDA POR document_id
    # ──────────────────────────────────────────────────────────────────────────

    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """
        Recupera todos los chunks de un documento dado su document_id.
        Los chunks se devuelven ordenados por chunk_index para preservar
        el orden original del documento.

        Args:
            document_id: UUID exacto del documento almacenado.

        Returns:
            Dict con el texto reconstruido y metadatos del documento.
        """
        if not self.available:
            return {"status": "error", "message": "Qdrant no disponible", "content": None}

        try:
            # Paginar para recuperar todos los chunks (documentos grandes pueden tener muchos)
            all_points = []
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    ),
                    limit=100,          # Recuperar de a 100 chunks por vez
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # No necesitamos los vectores, solo el payload
                )

                points, next_offset = scroll_result
                all_points.extend(points)

                if next_offset is None:
                    break
                offset = next_offset

            if not all_points:
                return {
                    "status": "not_found",
                    "message": f"No se encontró ningún documento con ID: {document_id}",
                    "content": None,
                    "document_id": document_id
                }

            return self._build_result_from_points(all_points, document_id)

        except Exception as e:
            logger.error(f"❌ Error buscando documento por ID '{document_id}': {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error al buscar en Qdrant: {str(e)}",
                "content": None
            }

    # ──────────────────────────────────────────────────────────────────────────
    # BÚSQUEDA POR NOMBRE DE ARCHIVO
    # ──────────────────────────────────────────────────────────────────────────

    def get_document_by_name(self, filename_query: str) -> Dict[str, Any]:
        """
        Recupera un documento buscando por nombre de archivo (búsqueda parcial).

        Si hay varios documentos que coinciden con el nombre, se retorna el más
        reciente (ordenado por stored_at). Si hay ambigüedad, se informa al
        agente para que el usuario aclare.

        Args:
            filename_query: Nombre o parte del nombre del archivo (sin importar
                            mayúsculas/minúsculas, con o sin extensión .pdf).

        Returns:
            Dict con el texto reconstruido y metadatos, o lista de coincidencias
            si hay ambigüedad.
        """
        if not self.available:
            return {"status": "error", "message": "Qdrant no disponible", "content": None}

        try:
            # Normalizar la consulta: quitar extensión para búsqueda más flexible
            query_clean = filename_query.lower().replace(".pdf", "").strip()

            # Recuperar todos los chunks y filtrar en Python por nombre
            # (Qdrant no soporta búsqueda parcial de string en scroll directamente)
            all_points = []
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = scroll_result
                all_points.extend(points)

                if next_offset is None:
                    break
                offset = next_offset

            if not all_points:
                return {
                    "status": "not_found",
                    "message": "No hay documentos almacenados en Qdrant.",
                    "content": None
                }

            # Agrupar puntos por document_id y filtrar por nombre
            doc_groups: Dict[str, List] = {}
            for point in all_points:
                payload = point.payload or {}
                stored_filename = (payload.get("filename") or "").lower().replace(".pdf", "")

                # Coincidencia parcial: el nombre buscado está contenido en el nombre guardado
                if query_clean in stored_filename or stored_filename in query_clean:
                    doc_id = payload.get("document_id")
                    if doc_id:
                        if doc_id not in doc_groups:
                            doc_groups[doc_id] = []
                        doc_groups[doc_id].append(point)

            if not doc_groups:
                return {
                    "status": "not_found",
                    "message": (
                        f"No se encontró ningún documento con el nombre '{filename_query}'. "
                        f"Verifica el nombre o usa el document_id directamente."
                    ),
                    "content": None,
                    "available_hint": "Usa list_documents() para ver los documentos disponibles."
                }

            # Si hay varios documentos coincidentes, informar al agente
            if len(doc_groups) > 1:
                matches = []
                for doc_id, points in doc_groups.items():
                    sample_payload = points[0].payload or {}
                    matches.append({
                        "document_id": doc_id,
                        "filename": sample_payload.get("filename", "desconocido"),
                        "stored_at": sample_payload.get("stored_at", "N/A"),
                        "num_chunks": len(points)
                    })

                # Ordenar por fecha más reciente
                matches.sort(key=lambda x: x["stored_at"], reverse=True)

                return {
                    "status": "ambiguous",
                    "message": (
                        f"Se encontraron {len(matches)} documentos con nombre similar a "
                        f"'{filename_query}'. Por favor especifica el document_id."
                    ),
                    "matches": matches,
                    "content": None
                }

            # Un solo documento encontrado: reconstruir y devolver
            doc_id, points = list(doc_groups.items())[0]
            logger.info(
                f"✅ Documento encontrado: '{points[0].payload.get('filename')}' "
                f"({len(points)} chunks)"
            )
            return self._build_result_from_points(points, doc_id)

        except Exception as e:
            logger.error(f"❌ Error buscando por nombre '{filename_query}': {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error al buscar en Qdrant: {str(e)}",
                "content": None
            }

    # ──────────────────────────────────────────────────────────────────────────
    # LISTAR DOCUMENTOS DISPONIBLES
    # ──────────────────────────────────────────────────────────────────────────

    def list_documents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Devuelve un listado de los documentos almacenados en Qdrant.
        Útil para que el agente informe al usuario qué documentos puede analizar.

        Args:
            limit: Número máximo de documentos a listar.

        Returns:
            Lista de dicts con document_id, filename, stored_at, num_chunks.
        """
        if not self.available:
            return []

        try:
            all_points = []
            offset = None

            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                points, next_offset = scroll_result
                all_points.extend(points)
                if next_offset is None:
                    break
                offset = next_offset

            # Agrupar por document_id para contar chunks
            doc_index: Dict[str, Dict] = {}
            for point in all_points:
                payload = point.payload or {}
                doc_id = payload.get("document_id")
                if not doc_id:
                    continue
                if doc_id not in doc_index:
                    doc_index[doc_id] = {
                        "document_id": doc_id,
                        "filename": payload.get("filename", "desconocido"),
                        "stored_at": payload.get("stored_at", "N/A"),
                        "num_chunks": 0
                    }
                doc_index[doc_id]["num_chunks"] += 1

            # Ordenar por fecha más reciente y limitar
            documents = sorted(
                doc_index.values(),
                key=lambda x: x["stored_at"],
                reverse=True
            )[:limit]

            logger.info(f"📋 Documentos disponibles en Qdrant: {len(documents)}")
            return documents

        except Exception as e:
            logger.error(f"❌ Error listando documentos: {e}", exc_info=True)
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # MÉTODO INTERNO: reconstruir texto desde los puntos recuperados
    # ──────────────────────────────────────────────────────────────────────────

    def _build_result_from_points(
        self,
        points: List,
        document_id: str
    ) -> Dict[str, Any]:
        """
        Toma una lista de puntos de Qdrant, los ordena por chunk_index,
        concatena el contenido y devuelve el resultado estructurado.

        Args:
            points: Lista de ScoredPoint o Record de Qdrant.
            document_id: ID del documento para incluir en el resultado.

        Returns:
            Dict con el texto completo reconstruido y metadatos.
        """
        # Ordenar chunks por su posición original en el documento
        sorted_points = sorted(
            points,
            key=lambda p: p.payload.get("chunk_index", 0) if p.payload else 0
        )

        # Filtro de chunks relevantes
        content_parts = []
        discarded = 0

        for point in sorted_points:
            payload = point.payload or {}
            chunk_text = payload.get("contenido", "").strip()

            # Criterio 1: descartar vacíos
            if not chunk_text:
                discarded += 1
                continue

            # Criterio 2: descartar chunks con 2 palabras o menos
            words = chunk_text.split()
            if len(words) <= 2:
                discarded += 1
                continue

            # Criterio 3: descartar si no contiene ninguna letra
            if not any(c.isalpha() for c in chunk_text):
                discarded += 1
                continue

            # Criterio 4: descartar si más del 70% son mayúsculas (título)
            alpha_chars = [c for c in chunk_text if c.isalpha()]
            if alpha_chars and sum(c.isupper() for c in alpha_chars) / len(alpha_chars) > 0.70:
                discarded += 1
                continue

            content_parts.append(chunk_text)
        logger.info(
            f"📊 Chunks: {len(sorted_points)} total → "
            f"{len(content_parts)} relevantes, {discarded} descartados"
        )

        full_content = "\n\n".join(content_parts)

        # Extraer metadatos del primer chunk (todos comparten los mismos)
        first_payload = sorted_points[0].payload or {}

        return {
            "status": "success",
            "document_id": document_id,
            "filename": first_payload.get("filename", "desconocido"),
            "stored_at": first_payload.get("stored_at", "N/A"),
            "num_chunks": len(content_parts),
            "total_chunks_raw": len(sorted_points),
            "total_characters": len(full_content),
            "content": full_content,
            "message": (
                f"Documento '{first_payload.get('filename')}' recuperado exitosamente "
                f"({len(sorted_points)} chunks, {len(full_content)} caracteres)."
            )
        }