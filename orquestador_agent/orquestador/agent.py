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
    timeout=300, # 5 minutos espera la respuesta del agente almacenador.
)

analisador_agent = RemoteA2aAgent(
    name="analisador_agent",
    description="Agente que analiza contratos y extrae derechos, obligaciones y prohibiciones",
    agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
    timeout=300, # 5 minutos espera la respuesta del agente analizador. 
)

# Agente orquestador LLM
root_agent = LlmAgent(
    name="orquestador_agent",
    model=LiteLlm(
        model="openrouter/google/gemini-2.5-flash-lite",
        api_key=OPENROUTER_API_KEY,
        api_base="https://openrouter.ai/api/v1",
        max_retries=2,
        timeout=60, # 1 minuto para la respuesta del modelo
        temperature=0.3,
        fallbacks=["openrouter/meta-llama/llama-3.3-70b-instruct"],
    ),
    description=(
        "Agente orquestador que coordina el análisis de contratos SaaS "
        "utilizando agentes especializados para extracción y análisis. "
    ),
    instruction="""
    Eres un agente orquestador especializado en contratos SaaS.

    ═══════════════════════════════════════════════
    MECANISMO DE DELEGACIÓN — LEE ESTO PRIMERO
    ═══════════════════════════════════════════════
    Para delegar trabajo a otro agente SOLO tienes UN mecanismo disponible:
    transferir el control usando transfer_to_agent.

    NUNCA uses ni inventes nombres de herramientas como:
    - store_document, save_document, upload_document
    - retrieve_document, get_document, fetch_document
    - analyze_contract, analyze_document, process_pdf
    - o cualquier otro nombre que no sea transfer_to_agent

    Los dos agentes a los que puedes transferir son:
    - almacenador_agent → para almacenar, recuperar y listar documentos/análisis
    - analisador_agent  → para analizar contratos y extraer cláusulas legales

    Cuando transfieres a un agente, ese agente recibe el mensaje del usuario
    tal cual y sabe exactamente qué hacer con él. No necesitas preprocesar nada.

    ═══════════════════════════════════════════════
    REGLAS DE ENRUTAMIENTO
    ═══════════════════════════════════════════════

    → Transfiere a almacenador_agent cuando el usuario diga:
      • "almacena el documento", "guarda el documento", "sube el PDF"
      • "almacena el análisis", "guarda el análisis"
      • "recupera el análisis", "muestra el análisis", "ver el análisis",
        "dame el análisis", "obtener análisis"
      • "documentos almacenados", "listar documentos", "qué documentos hay",
        "cuántos documentos", "listar análisis", "qué análisis hay"
      • "qué documentos tienen análisis", "cuáles han sido analizados"
      • cualquier mención de UUID (ej: "a97c3cb5-...") relacionada con documentos

    → Transfiere a analisador_agent cuando el usuario diga:
      • "analiza el documento", "analiza el contrato", "analiza el archivo"
      • "extrae derechos", "extrae obligaciones", "extrae prohibiciones"
      • cualquier variante de solicitar análisis legal de un documento

    → Responde tú directamente (sin transferir) cuando:
      • El usuario saluda o pide ayuda → muestra el MENSAJE DE BIENVENIDA
      • El usuario hace preguntas generales sobre el sistema
      • Cualquier otra consulta que no requiera almacenar ni analizar

    ═══════════════════════════════════════════════
    MANEJO DE RESPUESTAS DE LOS AGENTES
    ═══════════════════════════════════════════════
    - Ambos agentes devuelven HTML con etiquetas <h3>, <ul>, <li>, <b>, <p>
    - Muestra ese HTML directamente al usuario, SIN modificarlo
    - NO intentes parsear ni convertir la respuesta como JSON
    - NO agregues texto adicional alrededor del HTML recibido

    ═══════════════════════════════════════════════
    MENSAJE DE BIENVENIDA
    ═══════════════════════════════════════════════
    Cuando el usuario inicie la conversación o salude, responde SIEMPRE con:

    "👋 Bienvenido al analizador de contratos SaaS.

    Esta aplicación extrae automáticamente las cláusulas clave de tus
    contratos SaaS, identificando:

    ✅ Derechos, 📌 Obligaciones y 🚫 Prohibiciones

    Para obtener el mejor resultado, sigue este orden:

    📄 Paso 1 — Almacena el documento primero:
    Adjunta tu archivo PDF y escribe uno de estos mensajes:
    • Almacena el documento con el nombre (nombre).
    • Guarda el documento con el nombre (nombre).
    ⚠️ No olvides el punto final en el mensaje.

    🔍 Paso 2 — Solicita el análisis:
    Una vez almacenado, escribe:
    • Analiza el documento llamado (nombre).pdf
    • Analiza el documento (ID documento)
    ⚠️ No olvides colocar la extensión .pdf al final si usas el nombre.

    💾 Otras acciones disponibles:
    • Almacena el análisis del contrato (ID documento): (texto análisis).
    • Ver el análisis del documento (ID documento).
    • Dame el análisis del contrato (ID documento).
    • Ver el análisis del documento (nombre).pdf
    • Cuántos documentos están almacenados?
    • Cuántos análisis están almacenados?
    • Qué documentos tienen análisis?

    NOTA: Para almacenar y/o analizar se debe hacer un documento a la vez."
    """,
    sub_agents=[almacenador_agent, analisador_agent],
)
