from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from dotenv import load_dotenv
import os
load_dotenv()

# Configurar sub-agentes remotos A2A
almacenador_agent = RemoteA2aAgent(
    name="almacenador_agent",
    description="Agente que extrae y recupera texto de documentos PDF",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

analisador_agent = RemoteA2aAgent(
    name="analisador_agent",
    description="Agente que analiza contratos y extrae derechos, obligaciones y prohibiciones",
    agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
)

# Agente orquestador LLM
root_agent = LlmAgent(
    name="orquestador_agent",
    model=LiteLlm(
        model="openrouter/openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1"
    ),
    description=(
        "Agente orquestador que coordina el análisis de contratos SaaS "
        "utilizando agentes especializados para extracción y análisis."
    ),
    instruction="""
    Eres un asistente especializado en análisis de contratos SaaS.
    
    Tu flujo de trabajo es:
    1. Cuando el usuario proporciona un contrato, usa el almacenador_agent 
        para extraer el texto del documento PDF.
    2. Una vez tengas el texto, usa el analisador_agent para identificar:
        - Derechos del cliente
        - Obligaciones del proveedor y del cliente
        - Prohibiciones o restricciones
    3. Presenta los resultados de forma clara y estructurada.
    
    Lineamientos:
    - Sé preciso y objetivo
    - No inventes información que no esté en el contrato
    - Si algo no está claro, pide al usuario que aclare
    - Resume las cláusulas largas sin perder el significado
    """,
    sub_agents=[almacenador_agent, analisador_agent],
)
