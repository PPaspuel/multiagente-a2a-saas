"""
Configuración del agente analizador de contratos con CrewAI.
Este agente identifica Derechos, Obligaciones y Prohibiciones en documentos PDF.
"""

from crewai import Agent, Task, Crew, LLM
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Validar API Key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("ERROR: Falta la variable OPENROUTER_API_KEY")

# ============================================
# CONFIGURACIÓN DEL MODELO LLM
# ============================================
#from langchain_openai import ChatOpenAI

llm = LLM(
    model="openrouter/anthropic/claude-3-haiku",
    api_key=OPENROUTER_API_KEY,
    api_base="https://openrouter.ai/api/v1",
    temperature=0.3,  # Baja temperatura para análisis preciso
)

logger.info(f"✅ Modelo LLM configurado")


# ============================================
# AGENTE EXPERTO EN ANÁLISIS LEGAL
# ============================================
contract_analyzer_agent = Agent(
    role='Analista Legal de Contratos',
    goal='Identificar y extraer de forma precisa todos los Derechos, Obligaciones y Prohibiciones presentes en contratos legales',
    backstory="""Eres un experto abogado especializado en análisis contractual con más de 15 años de experiencia.
    Tu especialidad es revisar contratos y extraer las cláusulas críticas que definen:
    
    - DERECHOS: Prerrogativas, facultades y beneficios que el contrato otorga a las partes
    - OBLIGACIONES: Compromisos, deberes y responsabilidades que las partes deben cumplir
    - PROHIBICIONES: Restricciones, limitaciones y acciones vedadas para las partes
    
    Tienes un ojo entrenado para identificar cláusulas implícitas y explícitas, 
    entender lenguaje legal complejo, y presentar la información de manera clara y estructurada.
    Siempre citas la sección exacta del contrato de donde extraes cada elemento.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


# ============================================
# AGENTE EXTRACTOR DE TEXTO PDF
# ============================================
pdf_extractor_agent = Agent(
    role='Extractor de Contenido PDF',
    goal='Extraer y preparar el texto completo de documentos PDF para su análisis',
    backstory="""Eres un especialista en procesamiento de documentos digitales.
    Tu trabajo es asegurar que el texto de los PDFs se extraiga correctamente,
    preservando la estructura, numeración de cláusulas, y formato relevante.
    Detectas y limpias errores de OCR cuando es necesario.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


# ============================================
# AGENTE FORMATEADOR DE RESPUESTAS JSON
# ============================================
json_formatter_agent = Agent(
    role='Estructurador de Datos JSON',
    goal='Convertir el análisis legal en un formato JSON perfectamente estructurado y válido',
    backstory="""Eres un experto en estructuración de datos.
    Tomas información legal compleja y la transformas en JSON limpio, válido y bien organizado.
    Nunca incluyes texto adicional fuera del JSON, y siempre validas que la estructura sea correcta.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


# ============================================
# FUNCIONES HELPER PARA CREAR TAREAS
# ============================================

def create_extraction_task(pdf_content: str) -> Task:
    """
    Crea la tarea de extracción de texto del PDF.
    
    Args:
        pdf_content: Contenido del PDF ya extraído
        
    Returns:
        Task: Tarea de preparación del texto
    """
    return Task(
        description=f"""Prepara el siguiente contenido de contrato para análisis:
        
        {pdf_content[:2000]}... (contenido completo disponible)
        
        Asegúrate de que el texto esté limpio y bien estructurado.
        Identifica la numeración de secciones y cláusulas si existe.
        """,
        expected_output="Texto del contrato limpio y estructurado, listo para análisis legal",
        agent=pdf_extractor_agent
    )


def create_analysis_task(pdf_content: str) -> Task:
    """
    Crea la tarea principal de análisis del contrato.
    
    Args:
        pdf_content: Contenido del contrato a analizar
        
    Returns:
        Task: Tarea de análisis legal
    """
    return Task(
        description=f"""Analiza exhaustivamente el siguiente contrato e identifica:

CONTRATO:
{pdf_content}

INSTRUCCIONES CRÍTICAS:

1. DERECHOS - Identifica todos los derechos otorgados a cada parte:
   - Derecho a recibir pagos
   - Derecho a usar propiedad intelectual
   - Derecho a rescindir el contrato
   - Derecho a auditar
   - Derecho a recibir servicios
   - Cualquier otra prerrogativa otorgada

2. OBLIGACIONES - Identifica todas las obligaciones de cada parte:
   - Obligaciones de pago
   - Obligaciones de entrega
   - Obligaciones de confidencialidad
   - Obligaciones de cumplimiento de plazos
   - Obligaciones de calidad
   - Cualquier otro deber contractual

3. PROHIBICIONES - Identifica todas las restricciones y limitaciones:
   - Prohibición de competencia
   - Prohibición de divulgación
   - Prohibición de transferencia
   - Limitaciones de uso
   - Restricciones territoriales
   - Cualquier otra prohibición

FORMATO DE SALIDA REQUERIDO:
Para cada elemento identificado, proporciona:
- Categoría: DERECHO / OBLIGACIÓN / PROHIBICIÓN
- Parte afectada: ¿A quién aplica?
- Descripción: Explicación clara del elemento
- Referencia: Sección o cláusula del contrato donde aparece
- Criticidad: ALTA / MEDIA / BAJA
""",
        expected_output="""Listado completo de:
        - Derechos identificados con sus partes, descripciones y referencias
        - Obligaciones identificadas con sus partes, descripciones y referencias  
        - Prohibiciones identificadas con sus partes, descripciones y referencias
        Todo organizado de manera clara y citando las cláusulas específicas.""",
        agent=contract_analyzer_agent
    )


def create_formatting_task() -> Task:
    """
    Crea la tarea de formateo JSON de los resultados.
    
    Returns:
        Task: Tarea de formateo
    """
    return Task(
        description="""Convierte el análisis legal anterior en un JSON perfectamente estructurado.

ESTRUCTURA JSON REQUERIDA:
{
  "status": "success",
  "operation": "analyze_contract",
  "data": {
    "derechos": [
      {
        "parte": "Nombre de la parte",
        "descripcion": "Descripción del derecho",
        "referencia": "Cláusula X.Y",
        "criticidad": "ALTA/MEDIA/BAJA"
      }
    ],
    "obligaciones": [
      {
        "parte": "Nombre de la parte",
        "descripcion": "Descripción de la obligación",
        "referencia": "Cláusula X.Y",
        "criticidad": "ALTA/MEDIA/BAJA"
      }
    ],
    "prohibiciones": [
      {
        "parte": "Nombre de la parte",
        "descripcion": "Descripción de la prohibición",
        "referencia": "Cláusula X.Y",
        "criticidad": "ALTA/MEDIA/BAJA"
      }
    ],
    "resumen": {
      "total_derechos": 0,
      "total_obligaciones": 0,
      "total_prohibiciones": 0,
      "elementos_criticos": 0
    }
  }
}

REGLAS ESTRICTAS:
1. Devuelve SOLO el JSON, sin texto adicional
2. No uses comillas triples ni markdown
3. Asegura que el JSON sea válido
4. Incluye todos los elementos identificados en el análisis previo
""",
        expected_output="JSON válido y bien estructurado con el análisis completo del contrato",
        agent=json_formatter_agent
    )


# ============================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================

def analyze_contract(pdf_content: str) -> str:
    """
    Analiza un contrato PDF y extrae derechos, obligaciones y prohibiciones.
    
    Args:
        pdf_content: Contenido de texto del PDF del contrato
        
    Returns:
        str: Resultado del análisis en formato JSON
    """
    try:
        logger.info("🔍 Iniciando análisis de contrato con CrewAI...")
        
        # Crear las tareas en secuencia
        extraction_task = create_extraction_task(pdf_content)
        analysis_task = create_analysis_task(pdf_content)
        formatting_task = create_formatting_task()
        
        # Crear el Crew con los agentes y tareas
        contract_crew = Crew(
            agents=[
                pdf_extractor_agent,
                contract_analyzer_agent,
                json_formatter_agent
            ],
            tasks=[
                extraction_task,
                analysis_task,
                formatting_task
            ],
            verbose=True
        )
        
        # Ejecutar el análisis
        logger.info("⚙️ Ejecutando Crew de análisis...")
        result = contract_crew.kickoff()
        
        logger.info("✅ Análisis completado exitosamente")
        
        # Extraer el resultado como string
        if hasattr(result, 'raw'):
            return result.raw
        elif isinstance(result, str):
            return result
        else:
            return str(result)
            
    except Exception as e:
        logger.error(f"❌ Error durante el análisis: {str(e)}", exc_info=True)
        
        # Devolver JSON de error
        error_json = {
            "status": "error",
            "operation": "analyze_contract",
            "message": f"Error durante el análisis: {str(e)}",
            "data": None
        }
        
        import json
        return json.dumps(error_json, indent=2, ensure_ascii=False)


# ============================================
# INFORMACIÓN DEL AGENTE
# ============================================

def get_agent_info() -> dict:
    """
    Retorna información sobre el agente configurado.
    Útil para debugging y verificación.
    """
    return {
        "agent_name": "contract_analyzer_agent",
        "framework": "CrewAI",
        "agents_count": 3,
        "agents": [
            contract_analyzer_agent.role,
            pdf_extractor_agent.role,
            json_formatter_agent.role
        ],
        "capabilities": [
            "Extracción de derechos",
            "Extracción de obligaciones",
            "Extracción de prohibiciones",
            "Análisis de criticidad",
            "Formato JSON estructurado"
        ]
    }


# ============================================
# TESTING (si se ejecuta directamente)
# ============================================

if __name__ == "__main__":
    info = get_agent_info()
    print("\n" + "="*60)
    print("CONFIGURACIÓN DEL AGENTE ANALIZADOR DE CONTRATOS")
    print("="*60)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    # Test con un contrato de ejemplo
    test_contract = """
    CONTRATO DE PRESTACIÓN DE SERVICIOS
    
    CLÁUSULA 1: El PRESTADOR se obliga a entregar los servicios pactados antes del 31 de diciembre de 2025.
    
    CLÁUSULA 2: El CLIENTE tiene derecho a rescindir este contrato con 30 días de aviso previo.
    
    CLÁUSULA 3: Queda prohibido al PRESTADOR divulgar información confidencial del CLIENTE.
    
    CLÁUSULA 4: El CLIENTE se obliga a pagar $10,000 USD mensuales.
    """
    
    print("🧪 Ejecutando análisis de prueba...")
    result = analyze_contract(test_contract)
    print("\n📊 RESULTADO:")
    print(result)