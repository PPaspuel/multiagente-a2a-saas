"""
Almacenamiento directo a Qdrant sin MCP.
VERSIÓN: Compatible con Qdrant local en Docker
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class QdrantStorageManager:
    """
    Gestor de almacenamiento directo a Qdrant.
    Compatible con Qdrant ejecutándose en Docker local.
    """
    
    def __init__(self):
        """Inicializa la conexión a Qdrant local (Docker) y crea la colección si no existe."""
        try:
            # Leer configuración desde variables de entorno
            # Para Qdrant en Docker local, usa localhost:6333
            qdrant_host = os.getenv("QDRANT_HOST")
            qdrant_port = int(os.getenv("QDRANT_PORT"))
            self.collection_name = os.getenv("COLLECTION_NAME")
            
            logger.info(f"🔌 Conectando a Qdrant en {qdrant_host}:{qdrant_port}")
            
            # Crear cliente de Qdrant para conexión local (Docker)
            self.client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                timeout=10  # Timeout de 10 segundos
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
            
            # Crear colección si no existe
            if not self.client.collection_exists(self.collection_name):
                logger.info(f"📦 Creando colección '{self.collection_name}'...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=768,  # Tamaño del vector (compatible con muchos modelos de embeddings)
                        distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"✅ Colección '{self.collection_name}' creada exitosamente")
            else:
                logger.info(f"✅ Usando colección existente '{self.collection_name}'")
            
            self.available = True
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
    
    
    def store_chunks(
        self, 
        chunks: List[str], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Almacena fragmentos de texto en Qdrant con vectorización.
        
        Args:
            chunks: Lista de fragmentos de texto a almacenar
            metadata: Metadatos adicionales para cada fragmento
            
        Returns:
            Dict con información del resultado:
                - status: "success" o "error"
                - chunks_stored: número de chunks almacenados
                - collection: nombre de la colección
                - point_ids: lista de IDs generados (solo si success)
                - message: mensaje de error (solo si error)
        """
        if not self.available:
            logger.error("❌ Qdrant no está disponible")
            return {
                "status": "error",
                "message": "Qdrant no disponible. Inicia Docker: docker run -p 6333:6333 qdrant/qdrant",
                "chunks_stored": 0
            }
        
        try:
            logger.info(f"💾 Preparando {len(chunks)} fragmentos para almacenamiento...")
            
            points = []
            base_metadata = metadata or {}
            
            for idx, chunk in enumerate(chunks):
                # Generar ID único para cada punto
                point_id = str(uuid.uuid4())
                
                # IMPORTANTE: En producción, aquí deberías generar embeddings reales
                # Ejemplo con sentence-transformers:
                # from sentence_transformers import SentenceTransformer
                # model = SentenceTransformer('all-MiniLM-L6-v2')
                # vector = model.encode(chunk).tolist()
                
                # Vector dummy para demostración (REEMPLAZAR en producción)
                vector_dummy = [0.1] * 768
                
                # Crear punto con payload completo
                point = models.PointStruct(
                    id=point_id,
                    vector=vector_dummy,
                    payload={
                        "contenido": chunk,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "chunk_length": len(chunk),
                        **base_metadata  # Agregar metadatos adicionales
                    }
                )
                points.append(point)
            
            # Almacenar todos los puntos en Qdrant
            logger.info(f"📤 Subiendo {len(points)} puntos a Qdrant...")
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"✅ {len(chunks)} fragmentos almacenados exitosamente en '{self.collection_name}'")
            
            return {
                "status": "success",
                "chunks_stored": len(chunks),
                "collection": self.collection_name,
                "point_ids": [p.id for p in points]
            }
            
        except Exception as e:
            logger.error(f"❌ Error almacenando fragmentos en Qdrant: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error de almacenamiento: {str(e)}",
                "chunks_stored": 0
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
            # IMPORTANTE: Generar embedding real del query en producción
            query_vector = [0.1] * 768
            
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
                    "metadata": {
                        k: v for k, v in hit.payload.items() 
                        if k != "contenido"
                    }
                }
                for hit in results
            ]
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}", exc_info=True)
            return []
    
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre la colección actual.
        
        Returns:
            Dict con información de la colección
        """
        if not self.available:
            return {
                "available": False,
                "message": "Qdrant no disponible"
            }
        
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "available": True,
                "collection_name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error obteniendo info de colección: {e}")
            return {
                "available": True,
                "error": str(e)
            }


# Instancia global del gestor de almacenamiento
storage_manager = QdrantStorageManager()