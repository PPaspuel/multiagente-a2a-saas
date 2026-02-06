"""
Almacenamiento de respuestas/resultados de análisis.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResponseStorageHandler:
    """
    Maneja el almacenamiento de respuestas de análisis.
    """
    
    @staticmethod
    def should_store_response(user_text: str) -> bool:
        """
        Determina si el usuario quiere almacenar una respuesta/análisis.
        
        Args:
            user_text: Texto del usuario (ej: "almacena esta respuesta")
            
        Returns:
            bool: True si debe almacenar una respuesta
        """
        if not user_text:
            return False
        
        storage_keywords = [
            "almacena esta respuesta",
            "guarda este análisis",
            "almacena este resultado",
            "guarda la respuesta",
            "store this response",
            "save this analysis",
            "almacena esto"
        ]
        
        user_text_lower = user_text.lower().strip()
        
        # Verificar comandos explícitos de almacenamiento de respuestas
        for keyword in storage_keywords:
            if keyword in user_text_lower:
                return True
        
        # Si no hay PDFs pero hay texto del usuario, preguntar
        return len(user_text) > 20 and "pdf" not in user_text_lower
    
    @staticmethod
    def extract_response_text(user_parts, user_text: str) -> Optional[Dict[str, Any]]:
        """
        Extrae texto de respuesta para almacenar.
        
        Args:
            user_parts: Partes del mensaje A2A
            user_text: Texto extraído del usuario
            
        Returns:
            Dict con la respuesta a almacenar o None
        """
        # Caso 1: Texto directo del usuario (si es largo, probablemente es una respuesta)
        if len(user_text) > 100:  # Umbral para considerar que es una respuesta
            return {
                "text": user_text,
                "source": "user_input",
                "type": "response",
                "metadata": {
                    "extracted_at": datetime.now().isoformat(),
                    "is_response": True
                }
            }
        
        # Caso 2: Buscar en partes del mensaje
        response_parts = []
        for part in user_parts:
            if hasattr(part, 'root') and hasattr(part.root, 'text'):
                part_text = part.root.text
                if part_text and len(part_text.strip()) > 50:  # Texto significativo
                    response_parts.append(part_text)
        
        if response_parts:
            combined_text = "\n".join(response_parts)
            return {
                "text": combined_text,
                "source": "message_parts",
                "type": "response",
                "metadata": {
                    "extracted_at": datetime.now().isoformat(),
                    "is_response": True,
                    "num_parts": len(response_parts)
                }
            }
        
        return None
    
    @staticmethod
    def prepare_response_for_storage(response_data: Dict[str, Any], context) -> Dict[str, Any]:
        """
        Prepara los metadatos para almacenar una respuesta.
        
        Args:
            response_data: Datos de la respuesta extraída
            context: Contexto de la solicitud
            
        Returns:
            Dict con metadatos enriquecidos
        """
        base_metadata = {
            "task_id": context.task_id if hasattr(context, 'task_id') else "unknown",
            "origen": "response_storage",
            "content_type": "response",
            "storage_timestamp": datetime.now().isoformat(),
            "is_analysis_result": True,
            "original_length": len(response_data["text"])
        }
        
        # Combinar con metadatos existentes
        if "metadata" in response_data:
            base_metadata.update(response_data["metadata"])
        
        return base_metadata