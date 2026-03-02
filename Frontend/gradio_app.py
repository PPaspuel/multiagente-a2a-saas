import gradio as gr
import sys
import os
import warnings
import uuid as _uuid

# ── Silenciar warnings ────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="google.adk")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ── Ajustar sys.path ──────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FRONTEND_DIR)

for _p in (PROJECT_ROOT, FRONTEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from orquestador_agent.orquestador.agent import root_agent

# ──────────────────────────────────────────
# 1. Configuración del Runner de ADK
# ──────────────────────────────────────────
APP_NAME   = "saas_contract_analyzer"
USER_ID    = "gradio_user"

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

async def create_new_session() -> str:
    """
    Crea una sesión nueva por cada mensaje.
    Con RemoteA2aAgent, al terminar una delegación la sesión queda con
    estado interno contaminado. Si se reutiliza, el LlmAgent no genera
    respuesta nueva. Sesión fresca por turno = contexto siempre limpio.
    El historial visible lo gestiona ChatInterface en el frontend.
    """
    session_id = str(_uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    return session_id


# ──────────────────────────────────────────
# 2. Manejo de PDF
# ──────────────────────────────────────────
_uploaded_pdf_path: str = ""

MAX_PDF_SIZE_MB = 50
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024 

def handle_pdf_upload(file) -> str:
    global _uploaded_pdf_path
    if file is None:
        _uploaded_pdf_path = ""
        return ""

    file_size = os.path.getsize(file)
    if file_size > MAX_PDF_SIZE_BYTES:
        _uploaded_pdf_path = ""
        return (
            f"❌ **Archivo rechazado:** el PDF pesa {file_size / (1024 * 1024):.1f} MB, "
            f"el límite es {MAX_PDF_SIZE_MB} MB. "
            f"Por favor, comprime o divide el documento."
        )

    _uploaded_pdf_path = file
    filename = os.path.basename(file)
    return (
        f"✅ PDF cargado: **{filename}** ({file_size / (1024 * 1024):.1f} MB)\n"
        f"Ahora escribe en el chat, por ejemplo:\n"
        f"*Almacena el documento con el nombre {os.path.splitext(filename)[0]}.*"
    )

# ──────────────────────────────────────────
# 3. Función principal del agente
# ──────────────────────────────────────────

async def agent_response_with_pdf(message: str, history: list):
    """
    Llama al Runner de ADK e itera TODOS los eventos para capturar
    la respuesta real.

    PROBLEMA ORIGINAL:
        Con RemoteA2aAgent el runner emite múltiples eventos:
        submitted → working → (eventos A2A intermedios) → texto final
        
        Hacer `break` en el primer `is_final_response()` hace que el código
        salga en un evento de estado vacío antes de llegar al texto real.

    SOLUCIÓN:
        Iterar todos los eventos sin break y recolectar todos los textos
        no vacíos. Al final usar el último texto con sustancia (> 20 chars).
    """
    global _uploaded_pdf_path

    # Validación de mensaje vacío 
    if not message or not message.strip():
        yield "⚠️ Por favor escribe un mensaje antes de enviar."
        return

    session_id = await create_new_session()
    
    parts = []

    # Adjuntar PDF si hay uno pendiente
    if _uploaded_pdf_path and os.path.exists(_uploaded_pdf_path):
        try:
            with open(_uploaded_pdf_path, "rb") as f:
                pdf_bytes = f.read()
            filename = os.path.basename(_uploaded_pdf_path)
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type="application/pdf",
                        data=pdf_bytes,
                    )
                )
            )
            parts.append(types.Part(text=f"[Archivo PDF adjunto: {filename}]"))
            _uploaded_pdf_path = ""  # limpiar tras adjuntar
        except Exception as exc:
            parts.append(types.Part(text=f"[Error al leer PDF: {exc}]"))

    parts.append(types.Part(text=message))
    adk_content = types.Content(role="user", parts=parts)

    # ── Iterar TODOS los eventos, sin break ────
    collected_texts = []
    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=adk_content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text and part.text.strip():
                        collected_texts.append(part.text.strip())

    except Exception as e:
        yield f"❌ Error al contactar el agente: {str(e)}"
        return

    # Tomar el último texto con más de 20 caracteres (descarta estados cortos)
    final_response = "⚠️ El agente no devolvió una respuesta."
    for text in reversed(collected_texts):
        if len(text) > 20:
            final_response = text
            break

    # Si todos son cortos, tomar el último de todos
    if final_response == "⚠️ El agente no devolvió una respuesta." and collected_texts:
        final_response = collected_texts[-1]

    # Streaming palabra a palabra (no corta tags HTML)
    response_so_far = ""
    for word in final_response.split(" "):
        response_so_far += word + " "
        yield response_so_far.rstrip()


# ──────────────────────────────────────────
# 4. Construcción de la UI
# ──────────────────────────────────────────
with gr.Blocks(title="Analizador de Contratos SaaS", fill_height=True) as demo:

    gr.Markdown("""
    # 📄 Analizador de Contratos SaaS
    *Powered by Google ADK · Escribe **hola** para ver las instrucciones*
    """)

    # Zona de subida de PDF
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📎 Adjuntar PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            pdf_status = gr.Markdown("")

    pdf_input.change(
        fn=handle_pdf_upload,
        inputs=[pdf_input],
        outputs=[pdf_status],
    )

    gr.Markdown("---")

    # Chat principal
    chat = gr.ChatInterface(
        fn=agent_response_with_pdf,
        flagging_mode="manual",
        flagging_options=["👍 Útil", "👎 Incorrecto", "⚠️ Inapropiado"],
        save_history=True,
        examples=[
            "hola",
            "¿Cuantos documentos están almacenados?",
            "¿Cuantos análisis están almacenados?",
        ],
    )

# ──────────────────────────────────────────
# 5. Punto de entrada
# ──────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
    )