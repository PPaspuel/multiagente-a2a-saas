"""
Almacenamiento directo a Qdrant.
VERSIÓN MEJORADA:
- Deduplicación de documentos por hash
- Almacenamiento de análisis vinculados a documentos
- Recuperación de análisis almacenados
"""

import logging
import uuid
import hashlib
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)


class QdrantStorageManager:
    """
    Gestor de almacenamiento directo a Qdrant.
    Compatible con Qdrant ejecutándose en Docker local.
    
    NUEVAS CAPACIDADES:
    - Deduplicación automática por hash de documento
    - Almacenamiento de análisis vinculados
    - Recuperación de análisis por documento
    """
    
    def __init__(self):
        """Inicializa la conexión a Qdrant local (Docker) y crea las colecciones si no existen."""
        try:
            # Leer configuración desde variables de entorno
            qdrant_host = os.getenv("QDRANT_HOST")
            qdrant_port = int(os.getenv("QDRANT_PORT"))
            self.collection_name = os.getenv("COLLECTION_NAME")
            
            logger.info(f"🔌 Conectando a Qdrant en {qdrant_host}:{qdrant_port}")
            
            # Crear cliente de Qdrant para conexión local (Docker)
            self.client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                timeout=10
            )
            
            # Verificar conexión
            try:
                collections = self.client.get_collections()
                logger.info(f"✅ Conectado a Qdrant - {len(collections.collections)} colecciones existentes")
            except Exception as e:
                logger.error(f"❌ No se pudo conectar a Qdrant: {e}")
                logger.error("🐳 Asegúrate de que Docker esté corriendo:")
                logger.error("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
                raise
            
            # Crear colección de documentos si no existe
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"📦 Creando colección '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=384,  # Dimensión real de all-MiniLM-L6-v2
                        distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"✅ Colección '{self.collection_name}' creada exitosamente")
            else:
                logger.info(f"✅ Usando colección existente '{self.collection_name}'")
            
            # Crear colección para análisis
            self.analysis_collection = f"{self.collection_name}_analysis"
            if not self.client.collection_exists(self.analysis_collection):
                logger.info(f"📦 Creando colección de análisis '{self.analysis_collection}'...")
                self.client.create_collection(
                    collection_name=self.analysis_collection,
                    vectors_config=models.VectorParams(
                        size=384,  # Dimensión real de all-MiniLM-L6-v2
                        distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"✅ Colección de análisis creada exitosamente")
            
            self.available = True
            
            # Inicializar modelo de embeddings para vectorización real
            # Se carga una sola vez y se reutiliza en todas las operaciones
            logger.info("🤖 Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self._embedding_size = 384  # Dimensión real del modelo all-MiniLM-L6-v2
                logger.info("✅ Modelo de embeddings cargado correctamente")
            except ImportError:
                logger.error(
                    "❌ sentence-transformers no instalado. "
                    "Instala con: pip install sentence-transformers"
                )
                self._embedding_model = None
                self._embedding_size = 768  # Fallback

            logger.info("✅ QdrantStorageManager inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error conectando a Qdrant: {e}", exc_info=True)
            logger.warning("=" * 70)
            logger.warning("⚠️  QDRANT NO DISPONIBLE - Trabajando en MODO SIN ALMACENAMIENTO")
            logger.warning("=" * 70)
            logger.warning("🐳 Para iniciar Qdrant con Docker:")
            logger.warning("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
            logger.warning("")
            logger.warning("🔍 O verifica que el contenedor esté corriendo:")
            logger.warning("   docker ps | grep qdrant")
            logger.warning("=" * 70)
            self.client = None
            self.available = False
    
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Genera un embedding real para el texto dado usando el modelo cargado.

        Si el modelo no está disponible (por falta de dependencias), devuelve
        un vector de ceros como fallback para no bloquear el flujo.

        Args:
            text: Texto a vectorizar

        Returns:
            List[float]: Vector de embeddings de dimensión 384
        """
        if self._embedding_model is None:
            logger.warning("⚠️ Modelo de embeddings no disponible. Usando vector de fallback.")
            return [0.0] * 384

        try:
            vector = self._embedding_model.encode(text, show_progress_bar=False)
            return vector.tolist()
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            return [0.0] * 384

    
    def _calculate_document_hash(self, content: str) -> str:
        """
        Calcula un hash único para el contenido del documento.
        
        Args:
            content: Contenido completo del documento
            
        Returns:
            str: Hash SHA-256 del documento
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    
    def _check_document_exists(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """
        Verifica si un documento con el mismo hash ya existe.
        
        Args:
            doc_hash: Hash del documento a verificar
            
        Returns:
            Dict con información del documento si existe, None en caso contrario
        """
        if not self.available:
            return None
        
        try:
            # Buscar documentos con el mismo hash
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_hash",
                            match=MatchValue(value=doc_hash)
                        )
                    ]
                ),
                limit=1
            )
            
            if search_result[0]:  # Si hay resultados
                point = search_result[0][0]
                return {
                    "exists": True,
                    "document_id": point.payload.get("document_id"),
                    "stored_at": point.payload.get("stored_at"),
                    "filename": point.payload.get("filename"),
                    "num_chunks": point.payload.get("total_chunks")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error verificando existencia del documento: {e}")
            return None
    
    
    def store_chunks(
        self, 
        chunks: List[str], 
        metadata: Optional[Dict[str, Any]] = None,
        full_content: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Almacena fragmentos de texto en Qdrant con vectorización.
        MEJORADO: Detecta duplicados y actualiza en lugar de duplicar.
        
        Args:
            chunks: Lista de fragmentos de texto a almacenar
            metadata: Metadatos adicionales para cada fragmento
            full_content: Contenido completo para calcular hash (opcional)
            filename: Nombre del archivo (opcional)
            
        Returns:
            Dict con información del resultado
        """
        if not self.available:
            logger.error("❌ Qdrant no está disponible")
            return {
                "status": "error",
                "message": "Qdrant no disponible. Inicia Docker: docker run -p 6333:6333 qdrant/qdrant",
                "chunks_stored": 0
            }
        
        try:
            # Generar ID único para este documento
            document_id = str(uuid.uuid4())
            
            # Calcular hash si se proporciona contenido completo
            doc_hash = None
            existing_doc = None
            was_updated = False
            
            if full_content:
                doc_hash = self._calculate_document_hash(full_content)
                logger.info(f"🔍 Hash del documento: {doc_hash[:16]}...")
                
                # Verificar si ya existe
                existing_doc = self._check_document_exists(doc_hash)
                
                if existing_doc:
                    logger.info(f"⚠️  Documento duplicado detectado!")
                    logger.info(f"   - Archivo: {existing_doc.get('filename')}")
                    logger.info(f"   - Almacenado: {existing_doc.get('stored_at')}")
                    logger.info(f"   - Chunks: {existing_doc.get('num_chunks')}")
                    logger.info(f"🔄 Actualizando documento existente...")
                    
                    document_id = existing_doc['document_id']
                    was_updated = True
                    
                    # Eliminar chunks antiguos del documento
                    self._delete_document_chunks(document_id)
            
            logger.info(f"💾 Preparando {len(chunks)} fragmentos para almacenamiento...")
            
            points = []
            base_metadata = metadata or {}
            timestamp = datetime.utcnow().isoformat()
            
            for idx, chunk in enumerate(chunks):
                # Generar ID único para cada punto
                point_id = str(uuid.uuid4())
                
                # Generar embedding real del chunk para búsqueda semántica
                chunk_vector = self._get_embedding(chunk)
                
                # Crear punto con payload completo
                point = models.PointStruct(
                    id=point_id,
                    vector=chunk_vector,
                    payload={
                        "contenido": chunk,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "chunk_length": len(chunk),
                        "document_id": document_id,
                        "document_hash": doc_hash,
                        "filename": filename or "unknown.pdf",
                        "stored_at": timestamp,
                        "updated_at": timestamp if was_updated else None,
                        **base_metadata
                    }
                )
                points.append(point)
            
            # Almacenar todos los puntos en Qdrant
            logger.info(f"📤 Subiendo {len(points)} puntos a Qdrant...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            action = "actualizados" if was_updated else "almacenados"
            logger.info(f"✅ {len(chunks)} fragmentos {action} exitosamente en '{self.collection_name}'")
            
            return {
                "status": "success",
                "chunks_stored": len(chunks),
                "collection": self.collection_name,
                "document_id": document_id,
                "document_hash": doc_hash,
                "was_updated": was_updated,
                "point_ids": [p.id for p in points],
                "existing_doc_info": existing_doc if existing_doc else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error almacenando fragmentos en Qdrant: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error de almacenamiento: {str(e)}",
                "chunks_stored": 0
            }
    
    
    def _delete_document_chunks(self, document_id: str) -> bool:
        """
        Elimina todos los chunks de un documento específico.
        
        Args:
            document_id: ID del documento a eliminar
            
        Returns:
            bool: True si se eliminó correctamente
        """
        try:
            # Obtener todos los puntos del documento
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
                limit=1000
            )
            
            point_ids = [point.id for point in scroll_result[0]]
            
            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    )
                )
                logger.info(f"🗑️  Eliminados {len(point_ids)} chunks antiguos del documento")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando chunks del documento: {e}")
            return False
    
    
    def store_analysis(
        self,
        document_id: str,
        analysis_content: str,
        analysis_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Almacena un análisis vinculado a un documento.
        
        Args:
            document_id: ID del documento original
            analysis_content: Contenido del análisis
            analysis_type: Tipo de análisis (general, summary, key_points, etc.)
            metadata: Metadatos adicionales
            
        Returns:
            Dict con información del resultado
        """
        if not self.available:
            return {
                "status": "error",
                "message": "Qdrant no disponible",
                "analysis_stored": False
            }
        
        try:
            analysis_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()
            
            # Generar embedding real del contenido del análisis
            analysis_vector = self._get_embedding(analysis_content)
            
            base_metadata = metadata or {}
            
            point = models.PointStruct(
                id=analysis_id,
                vector=analysis_vector,
                payload={
                    "analysis_content": analysis_content,
                    "document_id": document_id,
                    "analysis_type": analysis_type,
                    "created_at": timestamp,
                    "content_length": len(analysis_content),
                    **base_metadata
                }
            )
            
            # Almacenar en la colección de análisis
            self.client.upsert(
                collection_name=self.analysis_collection,
                points=[point]
            )
            
            logger.info(f"✅ Análisis '{analysis_type}' almacenado para documento {document_id[:8]}...")
            
            return {
                "status": "success",
                "analysis_stored": True,
                "analysis_id": analysis_id,
                "document_id": document_id,
                "analysis_type": analysis_type,
                "collection": self.analysis_collection
            }
            
        except Exception as e:
            logger.error(f"❌ Error almacenando análisis: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error almacenando análisis: {str(e)}",
                "analysis_stored": False
            }
    
    
    def retrieve_analysis(
        self,
        document_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recupera análisis almacenados.
        
        Args:
            document_id: ID del documento (opcional)
            analysis_type: Tipo de análisis a recuperar (opcional)
            limit: Número máximo de resultados
            
        Returns:
            Lista de análisis encontrados
        """
        if not self.available:
            logger.warning("⚠️ Recuperación no disponible - Qdrant no conectado")
            return []
        
        try:
            # Construir filtros
            filters = []
            
            if document_id:
                filters.append(
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                )
            
            if analysis_type:
                filters.append(
                    FieldCondition(
                        key="analysis_type",
                        match=MatchValue(value=analysis_type)
                    )
                )
            
            # Realizar búsqueda
            if filters:
                scroll_result = self.client.scroll(
                    collection_name=self.analysis_collection,
                    scroll_filter=Filter(must=filters),
                    limit=limit
                )
            else:
                scroll_result = self.client.scroll(
                    collection_name=self.analysis_collection,
                    limit=limit
                )
            
            results = []
            for point in scroll_result[0]:
                results.append({
                    "analysis_id": point.id,
                    "document_id": point.payload.get("document_id"),
                    "analysis_type": point.payload.get("analysis_type"),
                    "analysis_content": point.payload.get("analysis_content"),
                    "created_at": point.payload.get("created_at"),
                    "metadata": {
                        k: v for k, v in point.payload.items()
                        if k not in ["analysis_content", "document_id", "analysis_type", "created_at"]
                    }
                })
            
            logger.info(f"🔍 Encontrados {len(results)} análisis")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error recuperando análisis: {e}", exc_info=True)
            return []
    
    
    def get_document_with_analysis(self, document_id: str) -> Dict[str, Any]:
        """
        Recupera un documento junto con todos sus análisis.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Dict con documento y análisis
        """
        if not self.available:
            return {
                "status": "error",
                "message": "Qdrant no disponible"
            }
        
        try:
            # Obtener chunks del documento
            doc_chunks = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000
            )
            
            # Obtener análisis del documento
            analysis_list = self.retrieve_analysis(document_id=document_id)
            
            if not doc_chunks[0]:
                return {
                    "status": "error",
                    "message": f"Documento {document_id} no encontrado"
                }
            
            # Reconstruir documento
            chunks_data = sorted(
                [(p.payload.get("chunk_index"), p.payload.get("contenido")) 
                 for p in doc_chunks[0]],
                key=lambda x: x[0]
            )
            
            full_content = "\n".join([chunk[1] for chunk in chunks_data])
            
            first_chunk = doc_chunks[0][0]
            
            return {
                "status": "success",
                "document": {
                    "document_id": document_id,
                    "filename": first_chunk.payload.get("filename"),
                    "content": full_content,
                    "num_chunks": len(doc_chunks[0]),
                    "stored_at": first_chunk.payload.get("stored_at"),
                    "document_hash": first_chunk.payload.get("document_hash")
                },
                "analysis": analysis_list,
                "analysis_count": len(analysis_list)
            }
            
        except Exception as e:
            logger.error(f"❌ Error recuperando documento con análisis: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error: {str(e)}"
            }
    
    
    def search(
        self, 
        query: str, 
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Busca fragmentos similares en Qdrant.
        
        Args:
            query: Texto de búsqueda
            limit: Número máximo de resultados
            score_threshold: Umbral mínimo de similitud (0-1)
            
        Returns:
            Lista de resultados ordenados por similitud
        """
        if not self.available:
            logger.warning("⚠️ Búsqueda no disponible - Qdrant no conectado")
            return []
        
        try:
            # Generar embedding real de la consulta para búsqueda semántica genuina
            query_vector = self._get_embedding(query)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.info(f"🔍 Encontrados {len(results)} resultados para la búsqueda")
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "contenido": hit.payload.get("contenido", ""),
                    "document_id": hit.payload.get("document_id"),
                    "filename": hit.payload.get("filename"),
                    "metadata": {
                        k: v for k, v in hit.payload.items() 
                        if k not in ["contenido", "document_id", "filename"]
                    }
                }
                for hit in results
            ]
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}", exc_info=True)
            return []
    
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre las colecciones actuales.
        
        Returns:
            Dict con información de las colecciones
        """
        if not self.available:
            return {
                "available": False,
                "message": "Qdrant no disponible"
            }
        
        try:
            doc_info = self.client.get_collection(self.collection_name)
            analysis_info = self.client.get_collection(self.analysis_collection)
            
            return {
                "available": True,
                "documents_collection": {
                    "name": self.collection_name,
                    "vectors_count": doc_info.vectors_count,
                    "points_count": doc_info.points_count,
                    "status": doc_info.status
                },
                "analysis_collection": {
                    "name": self.analysis_collection,
                    "vectors_count": analysis_info.vectors_count,
                    "points_count": analysis_info.points_count,
                    "status": analysis_info.status
                }
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de colecciones: {e}")
            return {
                "available": True,
                "error": str(e)
            }


# Instancia global del gestor de almacenamiento
storage_manager = QdrantStorageManager()