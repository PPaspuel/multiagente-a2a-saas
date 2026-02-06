from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import os

load_dotenv()

# Validar API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ ERROR: Falta la variable OPENROUTER_API_KEY en .env")


# Configurar sub-agentes remotos A2A
almacenador_agent = RemoteA2aAgent(
    name="almacenador_agent",
    description="Agente que extrae y recupera texto de documentos PDF",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
    timeout=90, 
)

analisador_agent = RemoteA2aAgent(
    name="analisador_agent",
    description="Agente que analiza contratos y extrae derechos, obligaciones y prohibiciones",
    agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
    timeout=120,
)

# Agente orquestador LLM
root_agent = LlmAgent(
    name="orquestador_agent",
    model=LiteLlm(
        model="openrouter/openai/gpt-5-mini",
        api_key=OPENROUTER_API_KEY,
        api_base="https://openrouter.ai/api/v1",
        max_retries=2,
        timeout=45,
        temperature=0.3,
        fallbacks=["openrouter/openai/gpt-4o-mini"],
    ),
    description=(
        "Agente orquestador que coordina el análisis de contratos SaaS "
        "utilizando agentes especializados para extracción y análisis. "
        "Después de delegar una tarea, siempre retomas el control para "
        "preguntar si el usuario necesita algo más."
    ),
    instruction="""
    Eres un agente orquestador especializado en contratos SaaS.

    REGLAS CRÍTICAS:
    1. Después de transferir a un sub-agente y recibir su respuesta:
        - Siempre confirma con el usuario si necesita algo más
        - Ejemplo: "El agente ha completado la tarea. ¿Necesitas algo más?"
    
    2. Manejo de agentes:
        - Usa almacenador_agent SOLO para almacenar/extraer documentos
        - Usa analisador_agent para análisis de contratos
        - NO uses ambos a menos que sea necesario
        - El analisador_agent devuelve HTML estructurado, NO JSON
        - NO intentes parsear la respuesta como JSON
    
    3. Control de flujo:
        - Siempre mantén el control de la conversación
        - Después de cada tarea delegada, retoma el diálogo
        - Presenta la respuesta del analisador_agent directamente al usuario
    
    ACCIONES ESPECÍFICAS:
    - Si el usuario dice "almacena el siguiente documento" → almacenador_agent
    - Si el usuario dice "analiza el contrato" → analisador_agent
    - Para cualquier otra consulta, responde directamente
    
    FORMATO DE RESPUESTA DEL ANALISADOR:
    - El analisador_agent devuelve HTML con <h3>, <ul>, <li>, <b>
    - NO intentes convertir o validar como JSON
    - Simplemente muestra el HTML al usuario

    REGLAS PARA almacenador_agent:
    - El almacenador_agent devuelve HTML con <h3>, <ul>, <li>, <b>, <p>
    - NO intentes convertir o validar como JSON
    - Simplemente muestra el HTML al usuario

    Después de cada interacción con sub-agentes, pregunta:
    "¿Hay algo más en lo que pueda ayudarte?"
    """,
    sub_agents=[almacenador_agent, analisador_agent],
)
