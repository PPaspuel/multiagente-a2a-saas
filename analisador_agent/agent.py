import os
from collections.abc import AsyncIterable
from typing import Any, Literal
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

memory = MemorySaver()


@tool
def extract_contract_clauses(contract_text: str, clause_type: str = "all"):
    """Extrae cláusulas específicas de un contrato SaaS.

    Args:
        contract_text: El texto del contrato a analizar.
        clause_type: Tipo de cláusula a extraer: 'derechos', 'obligaciones', 
                'prohibiciones', o 'all' para todas.

    Returns:
        Un diccionario con las cláusulas encontradas organizadas por tipo.
    """
    # Esta es una función simplificada. En producción, aquí procesarías 
    # el texto del contrato con técnicas de NLP más avanzadas
    
    result = {
        "derechos": [],
        "obligaciones": [],
        "prohibiciones": [],
        "metadata": {
            "total_palabras": len(contract_text.split()),
            "tipo_analisis": clause_type
        }
    }
    
    # Palabras clave comunes en contratos SaaS
    keywords = {
        "derechos": ["derecho a", "podrá", "tiene derecho", "puede", "permitido"],
        "obligaciones": ["debe", "deberá", "obligado a", "compromete a", "responsable de"],
        "prohibiciones": ["no podrá", "prohibido", "no está permitido", "no debe", "restricción"]
    }
    
    # Análisis básico por líneas
    lines = contract_text.split('.')
    
    for line in lines:
        line_lower = line.lower().strip()
        if not line_lower:
            continue
            
        # Buscar derechos
        if clause_type in ["all", "derechos"]:
            for keyword in keywords["derechos"]:
                if keyword in line_lower:
                    result["derechos"].append(line.strip())
                    break
        
        # Buscar obligaciones
        if clause_type in ["all", "obligaciones"]:
            for keyword in keywords["obligaciones"]:
                if keyword in line_lower:
                    result["obligaciones"].append(line.strip())
                    break
        
        # Buscar prohibiciones
        if clause_type in ["all", "prohibiciones"]:
            for keyword in keywords["prohibiciones"]:
                if keyword in line_lower:
                    result["prohibiciones"].append(line.strip())
                    break
    
    return result


class ResponseFormat(BaseModel):
    """Formato de respuesta del agente."""
    
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str
    clausulas_encontradas: dict = {}


class ContractAgent:
    """Agente especializado en análisis de contratos SaaS."""

    SYSTEM_INSTRUCTION = (
        'Eres un asistente especializado en el análisis de contratos SaaS. '
        'Tu propósito es identificar y clasificar cláusulas contractuales en tres categorías: '
        '1. DERECHOS: lo que el cliente puede hacer o recibir '
        '2. OBLIGACIONES: lo que el cliente debe hacer o cumplir '
        '3. PROHIBICIONES: lo que el cliente no puede hacer '
        ''
        'Usa la herramienta "extract_contract_clauses" para analizar el texto del contrato. '
        'Presenta los resultados de manera clara y organizada. '
        ''
        'Si el usuario pregunta sobre temas que no sean análisis de contratos SaaS, '
        'indica amablemente que solo puedes ayudar con análisis contractual.'
    )

    FORMAT_INSTRUCTION = (
        'Establece el estado como "input_required" si necesitas más información del usuario. '
        'Establece el estado como "error" si hay un error al procesar. '
        'Establece el estado como "completed" cuando hayas completado el análisis. '
        'Incluye las cláusulas encontradas en el campo clausulas_encontradas.'
    )

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        model_source = os.getenv('model_source', 'google')
        
        if model_source == 'google':
            self.model = ChatGoogleGenerativeAI(
                model='gemini-2.0-flash-exp',
                temperature=0.1  # Baja temperatura para mayor precisión
            )
        else:
            # Configuración alternativa para otros modelos
            raise NotImplementedError("Solo Google Gemini está configurado")
        
        self.tools = [extract_contract_clauses]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        """Procesa una consulta y genera respuestas en streaming."""
        
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        # Streaming del proceso
        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            
            # Cuando el agente está usando herramientas
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Analizando el contrato y extrayendo cláusulas...',
                }
            
            # Cuando procesa resultados de herramientas
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Clasificando cláusulas encontradas...',
                }

        # Respuesta final
        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        """Obtiene la respuesta estructurada final del agente."""
        
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        
        if structured_response and isinstance(structured_response, ResponseFormat):
            
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            
            if structured_response.status == 'completed':
                # Formatear la respuesta con las cláusulas
                response_content = structured_response.message
                
                if structured_response.clausulas_encontradas:
                    clausulas = structured_response.clausulas_encontradas
                    response_content += "\n\n📋 RESUMEN DEL ANÁLISIS:\n"
                    
                    if 'derechos' in clausulas and clausulas['derechos']:
                        response_content += f"\n✅ DERECHOS ({len(clausulas['derechos'])} encontrados)\n"
                    
                    if 'obligaciones' in clausulas and clausulas['obligaciones']:
                        response_content += f"📌 OBLIGACIONES ({len(clausulas['obligaciones'])} encontradas)\n"
                    
                    if 'prohibiciones' in clausulas and clausulas['prohibiciones']:
                        response_content += f"⛔ PROHIBICIONES ({len(clausulas['prohibiciones'])} encontradas)\n"
                
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': response_content,
                }

        # Respuesta por defecto en caso de error
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'No se pudo procesar la solicitud. '
                'Por favor, proporciona el texto del contrato a analizar.'
            ),
        }