"""
eval_analizador.py
==================
Script de evaluación G-Eval para el agente analizador de contratos.

Métricas evaluadas:
    - Faithfulness        → La respuesta está fundamentada en los chunks del contrato
    - Answer Relevancy    → La respuesta responde lo que se pidió
    - Contextual Precision → Los chunks recuperados de Qdrant son los correctos
    - Contextual Recall   → Se recuperaron todos los chunks necesarios

LLM juez: OpenRouter vía LiteLLM
Framework: DeepEval

Instalación previa:
    pip install deepeval litellm

Uso:
    python eval_analizador.py
"""

import os
import json
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from litellm import completion

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ Falta OPENROUTER_API_KEY en el archivo .env")

# Cargar casos de prueba desde archivos HTML en eval_data
GEVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ANALISIS_DIR = os.path.join(os.path.dirname(GEVAL_DIR), "Analisis")

def load_html(filename: str) -> str:
    """
    Carga el contenido HTML desde la carpeta Analisis.
    """

    filepath = os.path.join(ANALISIS_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ No se encontró el archivo: {filepath}\n"
            f"   Asegúrate de guardar el HTML del análisis en eval_data/{filename}"
        )
    
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


CHUNKS_DIR = os.path.join(os.path.dirname(GEVAL_DIR), "Chunks")
def load_context(filename: str) -> list[str]:
    filepath = os.path.join(CHUNKS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"❌ No se encontró: {filepath}\n"
            f"   Ejecuta primero el script de generación de chunks"
        )
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Modelo juez — OpenRouter vía LiteLLM
#    DeepEval necesita un modelo que implemente DeepEvalBaseLLM.
#    Usamos gemini-2.5-flash-lite (el mismo del orquestador) como juez.
# ──────────────────────────────────────────────────────────────────────────────

class OpenRouterJudge(DeepEvalBaseLLM):
    """
    Adaptador que conecta DeepEval con OpenRouter usando LiteLLM.
    Actúa como LLM juez para todas las métricas G-Eval.
    """

    def __init__(self, model: str = "openrouter/openai/gpt-4o-mini"):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_key=OPENROUTER_API_KEY,
            api_base="https://openrouter.ai/api/v1",
            temperature=0.0,  # Temperatura 0 para evaluación determinista
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        # DeepEval puede llamar versión async — reutilizamos la síncrona
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model


judge = OpenRouterJudge()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Métricas G-Eval
#    threshold: puntaje mínimo para considerar el test como pasado (0.0 - 1.0)
# ──────────────────────────────────────────────────────────────────────────────

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,   # Incluir justificación del puntaje
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)

contextual_recall_metric = ContextualRecallMetric(
    threshold=0.7,
    model=judge,
    include_reason=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Casos de prueba
#
#    Estructura de LLMTestCase para métricas RAG:
#      input              → pregunta / instrucción del usuario
#      actual_output      → respuesta real del agente analizador (HTML)
#      expected_output    → respuesta ideal esperada
#      retrieval_context  → chunks recuperados de Qdrant (contexto)
#
#    CÓMO OBTENER LOS VALORES REALES:
#      1. Ejecuta el sistema normalmente con un contrato real
#      2. Copia la respuesta HTML del agente analizador en actual_output
#      3. Copia los chunks recuperados de Qdrant en retrieval_context
#      4. Define en expected_output qué debería haber respondido idealmente
# ──────────────────────────────────────────────────────────────────────────────

caso_1 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado Adobe.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("Adobe-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Adobe has the right to terminate the Agreement for convenience at any time with immediate effect upon written notice (Section 5.4, HIGH)
    - Adobe has the right to audit redacted background check reports of Provider's personnel assigned to Adobe (Section 3.5, MEDIUM)

    Obligations identified:
    - Provider must maintain specified insurance coverage, naming Adobe as an additional insured (Section 10.10, HIGH)
    - Provider must defend, indemnify, and hold Adobe harmless from third-party claims arising from the Provider's performance or breach of the Agreement (Section 9.1(A), HIGH)

    Prohibitions identified:
    - Adobe cannot remove, obscure, or alter any copyright notice, trademark, or other proprietary right appearing in the Software (Section 6.3(B), MEDIUM)
    - Provider cannot withhold any Taxes from Adobe's compensation; Provider must pay all such taxes itself (Section 4.2, HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("Adobe-context.json"),
)

caso_2 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado AnthropicB.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("AnthropicB-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Customer retains all rights, title, and interest in and to its Prompts and Outputs (Section D, HIGH)
    - Anthropic has the right to update the Agreement, including the AUP, with changes effective 30 days after posting (Section N.1, MEDIUM)

    Obligations identified:
    - Customer must implement and maintain commercially reasonable security controls to ensure only authorized Users access the AI Services (Section E.1(a), HIGH)
    - Anthropic must defend and indemnify Customer from third-party claims that Customer's authorized use of the AI Services violates third-party IP rights (Section L.1, HIGH)

    Prohibitions identified:
    - Customer cannot attempt to reverse engineer the AI Services (Section G(ii), HIGH)
    - Customer cannot use the AI Services to build a competing product or service, including to train competing AI models (Section G(ii), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("AnthropicB-context.json"),
)

caso_3 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado Atlassian.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("Atlassian-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Atlassian has the right to verify that Customer owns or controls a domain specified for a Cloud Product's operation (Section 3.3, MEDIUM)
    - Customer has the right to terminate a Product subscription within 30 days of the initial Order and receive a refund (Section 10.3, HIGH)

    Obligations identified:
    - Atlassian must maintain an information security program with measures designed to protect Customer Data (Section 4.2, HIGH)
    - Customer must pay all applicable fees and taxes, which are non-refundable except as provided in the Agreement (Section 10.1(e) & 10.2(a), HIGH)

    Prohibitions identified:
    - Customer cannot rent, lease, sell, distribute, or sublicense the Products to a third party (Section 2.2(a), HIGH)
    - Customer cannot use the Products to develop a similar or competing product or service (Section 2.2(d), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("Atlassian-context.json"),
)

caso_4 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado BlazeMeter.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("BlazeMeter-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Perforce has the right to suspend service without refund if Customer interferes with the integrity of the Service or uses it to cause harm (Section 9, Saas Listing, HIGH)
    - Customer has the right to terminate the subscription if the Service Availability falls below the "Major" threshold for three consecutive months (Section 9, SaaS Listing, LOW)

    Obligations identified:
    - Customer must pay all fees, and any past due balances accrue interest (Section 12(a), HIGH)
    - Perforce warrants the Subscription Services will materially conform to the Documentation and will not contain Malicious Code (Section 13(a), HIGH)

    Prohibitions identified:
    - Customer cannot reverse engineer, disassemble, or decompile the Subscription Services (Section 9(b)(i), HIGH)
    - Customer cannot use the Subscription Service for any activity that constitutes a criminal offense or violates any law (Section 9(c), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("BlazeMeter-context.json"),
)

caso_5 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado Canva.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("Canva-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Customer owns all right, title, and interest in and to its Customer Material (Section 4.1, LOW)
    - Canva has the right to modify, remove, add, or enhance features of the Service at its sole discretion (Section 2.1, MEDIUM)

    Obligations identified:
    - Customer must pay all Subscription Fees as set forth in the Order Forms (Section 5.1, HIGH)
    - Canva must defend Customer from third-party claims that the Service infringes any patent, copyright, or trade secret (Section 9.1, HIGH)

    Prohibitions identified:
    - Customer cannot rent, lease, sell, distribute, or sublicense the Service to any third party other than Licensed Users (Section 2.5(i), HIGH)
    - Customer cannot copy, replicate, decompile, reverse-engineer, or modify the Service (Section 2.5(ii), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("Canva-context.json"),
)

caso_6 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado Figma.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("Figma-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Customer retains all right, title, and interest in and to its Customer Content (Section 2.7, HIGH)
    - Figma has the right to collect and analyze Usage Data in de-identified and aggregated form to maintain, improve, and enhance its products (Section 2.6, LOW)

    Obligations identified:
    - Figma must implement and maintain the security requirements set forth in Exhibit B (Section 1.2, HIGH)
    - Customer is responsible for maintaining the confidentiality of its Authorized Users' usernames and passwords (Section 2.3(c), HIGH)

    Prohibitions identified:
    - Customer cannot reverse engineer, decompile, or disassemble the Figma Platform (Section 2.1(i), HIGH)
    - Customer cannot provide, sell, resell, transfer, or sublicense the Figma Platform to others (Section 2.1(ii), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("Figma-context.json"),
)

caso_7 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado GitHub.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("GitHub-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Customer may assign each Subscription License to one individual End User for use on any number of devices (Section 1.3, HIGH)
    - GitHub has the right to charge 2% monthly interest on past due amounts (Section 8.2, MEDIUM)

    Obligations identified:
    - Customer is responsible for its End Users' compliance with the Agreement and all activities of its End Users (Section 1.4, HIGH)
    - Customer must pay all fees, which are non-refundable except as stated in the Agreement (Section 8.1, HIGH)

    Prohibitions identified:
    - Customer cannot reverse engineer, decompile, or disassemble any Product (Section 1.12(a), HIGH)
    - Customer cannot sell, rent, lease, sublicense, distribute, or lend any Products to others (Section 1.12(e), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("GitHub-context.json"),
)

caso_8 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado HubSpot.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("HubSpot-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - HubSpot has the right to increase Subscription Fees at renewal up to its then-current list price, with 30 days' notice (Section 3.2, LOW)
    - Customer owns and retains all rights to its Customer Materials and Customer Data (Section 5.1, HIGH)

    Obligations identified:
    - Customer must pay all fees, which are non-cancellable and non-refundable except as provided in the Agreement (Section 3.5, HIGH)
    - HubSpot must maintain commercially appropriate administrative, physical, and technical safeguards to protect Personal Data as described in the DPA (Section 5.4, HIGH)

    Prohibitions identified:
    - Customer cannot reverse engineer, decompile, disassemble, or otherwise attempt to discover the source code of the Subscription Services (Section 2.9(i), HIGH)
    - Customer cannot copy, rent, lease, sell, distribute, or create derivative works based on HubSpot Content (Section 6.1, HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("HubSpot-context.json"),
)

caso_9 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado OpenAI.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("OpenAI-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Customer retains all ownership rights in Input and owns all Output, to the extent permitted by law (Section 4.1, HIGH)
    - OpenAI may update the Services periodically, with notice if an update materially reduces functionality (Section 2.3, MEDIUM)

    Obligations identified:
    - Customer is responsible for all Input and must have all rights, licenses, and permissions required to provide it (Section 4.3, HIGH)
    - OpenAI must indemnify and defend Customer against third-party claims that the Services infringe third-party IP rights (Section 13.1, HIGH)

    Prohibitions identified:
    - Customer cannot reverse engineer any aspect of the Services or the systems used to provide them (Section 3.3(d), HIGH)
    - Customer cannot use Output to develop AI models that compete with OpenAI's products and services, except for a Permitted Exception (Section 3.3(e), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("OpenAI-context.json"),
)

caso_10 = LLMTestCase(
    #Esto fue lo que el usuario pidió
    input="Analiza el documento llamado OracleCloud.pdf",
    #Esto fue lo que el agente respondió
    actual_output=load_html("OracleCloud-analysis.html"),

    #Esto es lo que yo esperaba que respondiera
    expected_output="""
    Rights identified:
    - Oracle may suspend access to the Services if it believes there is a significant threat to the functionality, security, integrity, or availability of the Services (Section 9.3, HIGH)
    - Oracle retains all ownership and IP rights in the Services, derivative works thereof, and anything developed or delivered by or on behalf of Oracle (Section 3.1, HIGH)

    Obligations identified:
    - Customer is responsible for obtaining all rights related to Your Content required by Oracle to perform the Services (Section 3.3, HIGH)
    - Oracle must defend and indemnify Customer against third-party claims that Material furnished by Oracle infringes the third party's IP rights (Section 8.1, HIGH)

    Prohibitions identified:
    - Customer may not use the Services to perform cyber currency or crypto currency mining (Section 1.3(d), HIGH)
    - Customer may not modify, make derivative works of, disassemble, decompile, or reverse engineer any part of the Services (Section 3.4(a), HIGH)
    """,

    #Este fue el contexto que tenía disponible el agente (chunks recuperados de Qdrant)
    retrieval_context=load_context("OracleCloud-context.json"),
)
# ──────────────────────────────────────────────────────────────────────────────
# 4. Ejecutar evaluación
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n" + "="*60)
    print("  G-EVAL — Agente Analizador de Contratos")
    print("  Framework: DeepEval | Juez: OpenRouter")
    print("="*60)

    test_cases = [
        (caso_1,"Contrato de Adobe para suscripción a software como servicio (SaaS)."),
        (caso_2,"Acuerdo de Anthropic para servicios de IA ofrecidos como SaaS."),
        (caso_3,"Acuerdo de Atlassian para productos en la nube (SaaS)."),
        (caso_4,"Acuerdo de Perforce para plataforma SaaS de pruebas de rendimiento (BlazeMeter)."),
        (caso_5,"Contrato de Canva para su servicio de diseño SaaS."),
        (caso_6,"Acuerdo de Figma para su plataforma SaaS de diseño colaborativo."),
        (caso_7,"Acuerdo de GitHub para productos en línea (SaaS)."),
        (caso_8,"Términos de servicio de HubSpot para su plataforma SaaS."),
        (caso_9,"Acuerdo de OpenAI para servicios de IA (SaaS)."),
        (caso_10,"Acuerdo de Oracle para servicios en la nube (SaaS)."),

    ]

    metrics = [
        faithfulness_metric,
        answer_relevancy_metric,
        contextual_precision_metric,
        contextual_recall_metric,
    ]

    results_summary = []

    for test_case, description in test_cases:
        print(f"\n🧪 {description}")
        print("-" * 50)

        case_results = {}

        for metric in metrics:
            try:
                metric.measure(test_case)
                score  = round(metric.score, 3)
                passed = metric.is_successful()
                reason = getattr(metric, "reason", "Sin justificación")

                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status} | {metric.__class__.__name__:<28} | Score: {score:.3f}")
                if not passed:
                    print(f"         Razón: {reason}")

                case_results[metric.__class__.__name__] = {
                    "score": score,
                    "passed": passed,
                    "reason": reason,
                }

            except Exception as e:
                print(f"  ⚠️  ERROR en {metric.__class__.__name__}: {e}")
                case_results[metric.__class__.__name__] = {
                    "score": None,
                    "passed": False,
                    "reason": str(e),
                }

        results_summary.append({
            "description": description,
            "metrics": case_results,
        })

    # Guardar resultados en JSON
    output_path = "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Resultados guardados en: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_evaluation()