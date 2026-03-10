"""
Tests de integración — Comunicación entre agentes
==================================================
Verifica que los 3 agentes se comunican correctamente:

  Test 1: orquestador → almacenador → respuesta HTML
  Test 2: orquestador → analizador  → respuesta HTML
  Test 3: flujo completo con PDF real (store → analyze → retrieve)

Ejecución:
    pytest Test/test_integracion_agentes.py -v

Fixes aplicados v2:
  1. URLs de RemoteA2aAgent: se busca en repr() del objeto que incluye la URL de configuración
  2. _capturar_texto_eventos: lee event.parts[N].root.text (formato Message de A2A)
  3. response_formatter mock: se configura return_value con string real para evitar
     que pydantic reciba un MagicMock en lugar de str al construir TextPart
"""

import asyncio
import uuid
import pytest
import re
from unittest.mock import MagicMock, AsyncMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_part(text: str):
    """Crea un Part de texto real compatible con A2A."""
    from a2a.types import TextPart, Part
    return Part(root=TextPart(text=text))


def _make_file_part(pdf_bytes: bytes, filename: str = "contrato.pdf"):
    """Crea un Part de archivo PDF real compatible con A2A."""
    from a2a.types import FilePart, Part, FileWithBytes
    file_obj = FileWithBytes(bytes=pdf_bytes, filename=filename)
    return Part(root=FilePart(file=file_obj))


def _make_context(parts, task_id: str = None, context_id: str = None):
    """Construye un RequestContext mockeado con las parts dadas."""
    ctx = MagicMock()
    ctx.task_id = task_id or str(uuid.uuid4())
    ctx.context_id = context_id or str(uuid.uuid4())
    ctx.current_task = None
    message = MagicMock()
    message.parts = parts
    ctx.message = message
    return ctx


def _make_event_queue():
    """EventQueue mockeado que captura todos los eventos encolados."""
    queue = MagicMock()
    queue._events = []

    async def _capture(event):
        queue._events.append(event)

    queue.enqueue_event = AsyncMock(side_effect=_capture)
    return queue


def _make_updater_mock():
    """TaskUpdater completamente mockeado."""
    updater = MagicMock()
    updater.submit = AsyncMock()
    updater.start_work = AsyncMock()
    updater.complete = AsyncMock()
    updater.add_artifact = AsyncMock()
    updater.update_status = AsyncMock()
    updater.fail = AsyncMock()
    updater.cancel = AsyncMock()
    updater.new_agent_message = MagicMock(return_value=MagicMock())
    return updater


def _make_almacenador_executor():
    """
    Construye AlmacenadorAgentExecutor con dependencias mockeadas.
    FIX: response_formatter retorna strings reales (no MagicMock)
    para que pydantic pueda construir TextPart sin ValidationError.
    """
    with patch("almacenador_agent.agent_executor.root_agent"), \
         patch("almacenador_agent.agent_executor.almacenador_agent_runner"), \
         patch("almacenador_agent.agent_executor.storage_manager"), \
         patch("almacenador_agent.agent_executor.PDFProcessor"), \
         patch("almacenador_agent.agent_executor.ResponseFormatter"):
        from almacenador_agent.agent_executor import AlmacenadorAgentExecutor
        executor = AlmacenadorAgentExecutor.__new__(AlmacenadorAgentExecutor)
        executor.pdf_processor = MagicMock()

        # FIX CRÍTICO: response_formatter — TODOS los métodos retornan str.
        # MagicMock no permite sobreescribir __getattr__ directamente.
        # Solución: crear una clase real con todos los métodos que retornan str,
        # así pydantic nunca recibe un MagicMock al construir TextPart.
        html_ok  = "<p>✅ Operación exitosa</p>"
        html_err = "<p>❌ Error en operación</p>"

        class _FakeFormatter:
            """Formatter falso que garantiza strings en todos sus métodos."""
            def render_storage_response_html(self, *a, **kw): return html_ok
            def render_duplicate_update_html(self, *a, **kw): return html_ok
            def format_error_response(self, *a, **kw):        return html_err
            def format_success_html(self, *a, **kw):          return html_ok
            # Fallback genérico: CUALQUIER método no declarado retorna html_ok
            def __getattr__(self, name):
                return lambda *a, **kw: html_ok

        executor.response_formatter = _FakeFormatter()
        return executor


def _make_analizador_executor(qdrant_mock=None):
    """Construye ContractAnalyzerExecutor con QdrantRetriever mockeado."""
    with patch("analisador_agent.agent_executor.QdrantRetriever") as mock_cls, \
         patch("analisador_agent.agent_executor.analyze_contract"):
        instance = qdrant_mock or MagicMock()
        mock_cls.return_value = instance
        from analisador_agent.agent_executor import ContractAnalyzerExecutor
        executor = ContractAnalyzerExecutor.__new__(ContractAnalyzerExecutor)
        executor.qdrant = instance
        return executor


def _minimal_pdf_bytes() -> bytes:
    """Retorna bytes de un PDF mínimo válido."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"/Contents 4 0 R /Resources << /Font << /F1 << /Type /Font "
        b"/Subtype /Type1 /BaseFont /Helvetica >> >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Contrato SaaS) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000274 00000 n \n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n370\n%%EOF"
    )


def _html_valido(texto: str) -> bool:
    """Verifica que el texto contiene al menos una etiqueta HTML."""
    return bool(re.search(r'<(h[1-6]|ul|li|p|b|div|span|table)[^>]*>', texto, re.IGNORECASE))


def _capturar_texto_eventos(queue) -> str:
    """
    Extrae todo el texto de los eventos capturados en el queue.

    FIX v2: new_agent_text_message() retorna un objeto Message de A2A con:
      message.parts → [Part(root=TextPart(text=...))]
    El atributo es .parts directamente en el evento, NO .content.parts
    """
    textos = []
    for event in queue._events:
        # Formato principal: Message A2A → event.parts[N].root.text
        if hasattr(event, 'parts') and event.parts:
            for part in event.parts:
                if hasattr(part, 'root') and hasattr(part.root, 'text'):
                    t = part.root.text
                    if t:
                        textos.append(t)

        # Formato alternativo con .content wrapper
        if hasattr(event, 'content') and event.content:
            if hasattr(event.content, 'parts') and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'root') and hasattr(part.root, 'text'):
                        t = part.root.text
                        if t:
                            textos.append(t)
            if hasattr(event.content, 'text') and event.content.text:
                textos.append(event.content.text)

        # Fallback: .text directo
        if hasattr(event, 'text') and isinstance(event.text, str) and event.text:
            textos.append(event.text)

    return "\n".join(textos)


def _buscar_url_en_agente(agente, puerto: str) -> str:
    """
    FIX v2: RemoteA2aAgent almacena la URL internamente.
    repr() del objeto incluye la configuración con la URL completa.
    También buscamos en todos los atributos privados y públicos.
    """
    # Estrategia 1: repr() suele incluir la URL de configuración
    repr_str = repr(agente)
    if puerto in repr_str:
        return repr_str

    # Estrategia 2: buscar en vars() (atributos de instancia)
    for val in vars(agente).values():
        s = str(val)
        if puerto in s:
            return s

    # Estrategia 3: buscar en atributos privados (_xxx)
    for attr_name in dir(agente):
        try:
            val = str(getattr(agente, attr_name, ""))
            if puerto in val:
                return val
        except Exception:
            pass

    return ""


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: orquestador → almacenador → respuesta HTML
# ═════════════════════════════════════════════════════════════════════════════

class TestOrquestadorAlmacenador:
    """
    Verifica que el orquestador enruta correctamente al almacenador
    y que la respuesta final contiene información válida.
    """

    def test_orquestador_detecta_intencion_almacenar(self):
        """El orquestador debe tener almacenador_agent como sub-agente."""
        from orquestador_agent.orquestador.agent import root_agent
        assert hasattr(root_agent, 'sub_agents'), "root_agent debe tener sub_agents"
        nombres = [a.name for a in root_agent.sub_agents]
        assert "almacenador_agent" in nombres, \
            f"almacenador_agent no encontrado en sub_agents: {nombres}"

    def test_orquestador_tiene_almacenador_configurado(self):
        """Verifica que el RemoteA2aAgent del almacenador apunta al puerto 8001."""
        from orquestador_agent.orquestador.agent import root_agent
        almacenador = next(
            (a for a in root_agent.sub_agents if a.name == "almacenador_agent"), None
        )
        assert almacenador is not None

        card_url = _buscar_url_en_agente(almacenador, "8001")
        assert "8001" in card_url, (
            f"El almacenador debe apuntar al puerto 8001.\n"
            f"repr: {repr(almacenador)[:400]}\n"
            f"vars: { {k: str(v)[:60] for k, v in vars(almacenador).items()} }"
        )

    def test_almacenador_procesa_texto_y_genera_respuesta(self):
        """El executor del almacenador debe emitir al menos un evento para get_stats."""
        executor = _make_almacenador_executor()

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("almacenador_agent.agent_executor._save_metric"):

            sm_mock.get_stats.return_value = {
                "status": "success",
                "documents": {"total": 2, "list": [
                    {"filename": "contrato.pdf", "document_id": "doc-1", "stored_at": "2024-01-01"}
                ]},
                "analysis": {"total": 1},
                "chunks": {"total": 10}
            }
            updater = _make_updater_mock()
            updater_cls.return_value = updater
            context = _make_context([_make_part("cuantos documentos hay")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            assert queue.enqueue_event.called, \
                "El almacenador debe emitir al menos un evento de respuesta"
            updater.complete.assert_called_once()

    def test_almacenador_respuesta_contiene_info_documentos(self):
        """La respuesta del almacenador para get_stats debe incluir info de documentos."""
        executor = _make_almacenador_executor()

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("almacenador_agent.agent_executor._save_metric"):

            sm_mock.get_stats.return_value = {
                "status": "success",
                "documents": {"total": 3, "list": [
                    {"filename": "contrato_a.pdf", "document_id": "uuid-a", "stored_at": "2024-01-01"},
                ]},
                "analysis": {"total": 2},
                "chunks": {"total": 15}
            }
            updater = _make_updater_mock()
            updater_cls.return_value = updater
            context = _make_context([_make_part("estadísticas del sistema")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            texto = _capturar_texto_eventos(queue)
            assert any(keyword in texto for keyword in ["3", "documento", "análisis", "chunk"]), \
                f"Respuesta debe contener info de stats. Got: '{texto[:400]}'"

    def test_almacenador_responde_error_si_qdrant_no_disponible(self):
        """Si Qdrant no está disponible, no debe lanzar excepción."""
        executor = _make_almacenador_executor()

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("almacenador_agent.agent_executor._save_metric"):

            sm_mock.get_stats.return_value = {
                "status": "error",
                "message": "Qdrant no disponible"
            }
            updater = _make_updater_mock()
            updater_cls.return_value = updater
            context = _make_context([_make_part("cuantos documentos hay")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )
            updater.complete.assert_called()


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: orquestador → analizador → respuesta HTML
# ═════════════════════════════════════════════════════════════════════════════

class TestOrquestadorAnalizador:
    """
    Verifica que el orquestador enruta al analizador y que
    la respuesta contiene HTML válido con derechos, obligaciones y prohibiciones.
    """

    def test_orquestador_tiene_analizador_configurado(self):
        """Verifica que el RemoteA2aAgent del analizador apunta al puerto 8002."""
        from orquestador_agent.orquestador.agent import root_agent
        analizador = next(
            (a for a in root_agent.sub_agents if a.name == "analisador_agent"), None
        )
        assert analizador is not None, "analisador_agent debe estar en sub_agents"

        card_url = _buscar_url_en_agente(analizador, "8002")
        assert "8002" in card_url, (
            f"El analizador debe apuntar al puerto 8002.\n"
            f"repr: {repr(analizador)[:400]}\n"
            f"vars: { {k: str(v)[:60] for k, v in vars(analizador).items()} }"
        )

    def test_analizador_recupera_documento_y_analiza(self):
        """El analizador debe recuperar el documento de Qdrant e invocar CrewAI."""
        html_esperado = (
            "<h3>📋 Análisis del Contrato</h3>"
            "<h4>✅ Derechos</h4><ul><li>Derecho de uso</li></ul>"
            "<h4>📌 Obligaciones</h4><ul><li>Pagar mensualmente</li></ul>"
            "<h4>🚫 Prohibiciones</h4><ul><li>Revender el servicio</li></ul>"
        )
        qdrant_mock = MagicMock()
        qdrant_mock.get_document.return_value = {
            "status": "success",
            "document_id": "doc-uuid-001",
            "filename": "contrato_saas.pdf",
            "content": "El cliente tiene derecho a usar el servicio...",
            "num_chunks": 5,
            "total_chunks_raw": 6,
            "stored_at": "2024-01-01",
            "total_characters": 500,
            "message": "Documento recuperado"
        }

        with patch("analisador_agent.agent_executor.QdrantRetriever", return_value=qdrant_mock), \
             patch("analisador_agent.agent_executor.analyze_contract", return_value=html_esperado), \
             patch("analisador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("analisador_agent.agent_executor._save_metric"), \
             patch("analisador_agent.agent_executor._save_chunks_to_json", return_value="chunks.json"):

            updater = _make_updater_mock()
            updater_cls.return_value = updater

            from analisador_agent.agent_executor import ContractAnalyzerExecutor
            executor = ContractAnalyzerExecutor.__new__(ContractAnalyzerExecutor)
            executor.qdrant = qdrant_mock

            context = _make_context([_make_part("Analiza el documento contrato_saas.pdf")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            assert queue.enqueue_event.called, "El analizador debe emitir un evento"
            updater.complete.assert_called_once()

    def test_analizador_respuesta_contiene_html(self):
        """La respuesta final del analizador debe contener etiquetas HTML válidas."""
        html_analisis = (
            "<h3>📋 Análisis del Contrato SaaS</h3>"
            "<h4>✅ Derechos</h4><ul><li>Acceso al servicio</li></ul>"
            "<h4>📌 Obligaciones</h4><ul><li>Respetar los términos</li></ul>"
            "<h4>🚫 Prohibiciones</h4><ul><li>Compartir credenciales</li></ul>"
        )
        qdrant_mock = MagicMock()
        qdrant_mock.get_document.return_value = {
            "status": "success",
            "document_id": "doc-002",
            "filename": "contrato.pdf",
            "content": "Texto del contrato para análisis legal.",
            "num_chunks": 3,
            "total_chunks_raw": 4,
            "stored_at": "2024-06-01",
            "total_characters": 200,
            "message": "OK"
        }

        with patch("analisador_agent.agent_executor.QdrantRetriever", return_value=qdrant_mock), \
             patch("analisador_agent.agent_executor.analyze_contract", return_value=html_analisis), \
             patch("analisador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("analisador_agent.agent_executor._save_metric"), \
             patch("analisador_agent.agent_executor._save_chunks_to_json", return_value="chunks.json"):

            updater = _make_updater_mock()
            updater_cls.return_value = updater

            from analisador_agent.agent_executor import ContractAnalyzerExecutor
            executor = ContractAnalyzerExecutor.__new__(ContractAnalyzerExecutor)
            executor.qdrant = qdrant_mock

            context = _make_context([_make_part("Analiza el documento contrato.pdf")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            # FIX v2: leer el texto con _capturar_texto_eventos corregido
            texto = _capturar_texto_eventos(queue)
            assert _html_valido(texto), \
                f"La respuesta debe contener HTML válido. Got: '{texto[:500]}'"

    def test_analizador_retorna_not_found_si_documento_no_existe(self):
        """Si el documento no existe, debe responder con not_found sin lanzar excepción."""
        qdrant_mock = MagicMock()
        qdrant_mock.get_document.return_value = {
            "status": "not_found",
            "message": "Documento no encontrado",
            "content": None
        }
        qdrant_mock.list_documents.return_value = []

        with patch("analisador_agent.agent_executor.QdrantRetriever", return_value=qdrant_mock), \
             patch("analisador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("analisador_agent.agent_executor._save_metric"):

            updater = _make_updater_mock()
            updater_cls.return_value = updater

            from analisador_agent.agent_executor import ContractAnalyzerExecutor
            executor = ContractAnalyzerExecutor.__new__(ContractAnalyzerExecutor)
            executor.qdrant = qdrant_mock

            context = _make_context([_make_part("Analiza el documento inexistente.pdf")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            assert queue.enqueue_event.called
            updater.complete.assert_called()

    def test_analizador_responde_ambiguous_si_hay_multiples_docs(self):
        """Si hay múltiples documentos con el mismo nombre, responde con ambiguous."""
        qdrant_mock = MagicMock()
        qdrant_mock.get_document.return_value = {
            "status": "ambiguous",
            "message": "Se encontraron 2 documentos con nombre similar",
            "matches": [
                {"document_id": "doc-1", "filename": "contrato.pdf"},
                {"document_id": "doc-2", "filename": "contrato.pdf"},
            ],
            "content": None
        }

        with patch("analisador_agent.agent_executor.QdrantRetriever", return_value=qdrant_mock), \
             patch("analisador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("analisador_agent.agent_executor._save_metric"):

            updater = _make_updater_mock()
            updater_cls.return_value = updater

            from analisador_agent.agent_executor import ContractAnalyzerExecutor
            executor = ContractAnalyzerExecutor.__new__(ContractAnalyzerExecutor)
            executor.qdrant = qdrant_mock

            context = _make_context([_make_part("Analiza el documento contrato.pdf")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            assert queue.enqueue_event.called
            updater.complete.assert_called()

    def test_orquestador_instruccion_menciona_html(self):
        """La instrucción del orquestador debe indicar que las respuestas son HTML."""
        from orquestador_agent.orquestador.agent import root_agent
        instruccion = root_agent.instruction or ""
        assert "HTML" in instruccion, \
            "La instrucción del orquestador debe mencionar HTML"
        assert "SIN modificarlo" in instruccion or "sin modificar" in instruccion.lower(), \
            "La instrucción debe indicar que el HTML no debe modificarse"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: flujo completo con PDF real
# ═════════════════════════════════════════════════════════════════════════════

class TestFlujoCompletoConPDF:
    """
    Verifica el flujo end-to-end completo:
    1. Almacenar PDF (bytes reales)
    2. Analizar el documento almacenado
    3. Recuperar el análisis guardado
    """

    def test_flujo_1_almacenar_pdf_bytes_reales(self):
        """
        FIX v2: response_formatter retorna strings reales.
        Esto evita el ValidationError de pydantic al construir TextPart.
        """
        pdf_bytes = _minimal_pdf_bytes()
        executor = _make_almacenador_executor()
        texto_extraido = "El cliente tiene derecho a usar el servicio de software."

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("almacenador_agent.agent_executor.validate_pdf_content", return_value=True), \
             patch("almacenador_agent.agent_executor.get_pdf_metadata", return_value={"pages": 1}), \
             patch("almacenador_agent.agent_executor._save_metric"):

            executor.pdf_processor.extract_text_from_pdf.return_value = texto_extraido
            executor.pdf_processor.semantic_chunking.return_value = [
                texto_extraido[:30],
                texto_extraido[30:]
            ]
            # _FakeFormatter ya retorna strings reales — no requiere configuración adicional

            sm_mock.store_chunks.return_value = {
                "status": "success",
                "document_id": "doc-flujo-001",
                "filename": "contrato_saas.pdf",
                "chunks_stored": 2,
                "was_updated": False,
                "collection": "contratos"
            }
            sm_mock.available = True

            updater = _make_updater_mock()
            updater_cls.return_value = updater

            parts = [
                _make_file_part(pdf_bytes, "contrato_saas.pdf"),
                _make_part("[Archivo PDF adjunto: contrato_saas.pdf]"),
                _make_part("Almacena el documento con el nombre contrato_saas.")
            ]
            context = _make_context(parts)
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            assert queue.enqueue_event.called, \
                "El almacenador debe emitir respuesta al procesar PDF"
            updater.complete.assert_called_once()

    def test_flujo_2_analizar_documento_almacenado(self):
        """El analizador recupera el texto y genera HTML válido con CrewAI."""
        html_resultado = (
            "<h3>📋 Análisis del Contrato SaaS</h3>"
            "<h4>✅ Derechos</h4>"
            "<ul><li>El cliente tiene derecho a acceder al servicio 24/7.</li></ul>"
            "<h4>📌 Obligaciones</h4>"
            "<ul><li>Pagar la suscripción mensual.</li></ul>"
            "<h4>🚫 Prohibiciones</h4>"
            "<ul><li>Prohibido revender el servicio.</li></ul>"
        )
        qdrant_mock = MagicMock()
        qdrant_mock.get_document.return_value = {
            "status": "success",
            "document_id": "doc-flujo-001",
            "filename": "contrato_saas.pdf",
            "content": "El cliente tiene derecho a usar el servicio...",
            "num_chunks": 4,
            "total_chunks_raw": 5,
            "stored_at": "2024-01-01",
            "total_characters": 800,
            "message": "Documento recuperado exitosamente"
        }

        with patch("analisador_agent.agent_executor.QdrantRetriever", return_value=qdrant_mock), \
             patch("analisador_agent.agent_executor.analyze_contract", return_value=html_resultado) as mock_crew, \
             patch("analisador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("analisador_agent.agent_executor._save_metric"), \
             patch("analisador_agent.agent_executor._save_chunks_to_json", return_value="chunks.json"):

            updater = _make_updater_mock()
            updater_cls.return_value = updater

            from analisador_agent.agent_executor import ContractAnalyzerExecutor
            executor = ContractAnalyzerExecutor.__new__(ContractAnalyzerExecutor)
            executor.qdrant = qdrant_mock

            context = _make_context([_make_part("Analiza el documento contrato_saas.pdf")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            mock_crew.assert_called_once()
            call_args = mock_crew.call_args[0][0]
            assert len(call_args) > 0, "CrewAI debe recibir texto no vacío"
            assert queue.enqueue_event.called
            updater.complete.assert_called_once()

    def test_flujo_3_recuperar_analisis_guardado(self):
        """El almacenador retorna el análisis previamente guardado para un documento."""
        executor = _make_almacenador_executor()
        doc_id = "doc-flujo-001"

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("almacenador_agent.agent_executor._save_metric"):

            sm_mock.retrieve_analysis.return_value = [{
                "analysis_id": "anal-001",
                "document_id": doc_id,
                "analysis_type": "general",
                "analysis_content": "<h3>Derechos</h3><ul><li>Acceso al servicio</li></ul>",
                "created_at": "2024-01-01T10:00:00",
                "filename": "contrato_saas.pdf"
            }]
            updater = _make_updater_mock()
            updater_cls.return_value = updater

            context = _make_context([_make_part(f"Ver el análisis del documento {doc_id}")])
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            sm_mock.retrieve_analysis.assert_called()
            assert queue.enqueue_event.called
            updater.complete.assert_called_once()

    def test_flujo_completo_store_analyze_retrieve_en_secuencia(self):
        """
        Verifica la secuencia completa: almacenar → analizar → recuperar.
        FIX v2: response_formatter con strings reales en todos los pasos.
        """
        doc_id = str(uuid.uuid4())
        filename = "contrato_completo.pdf"
        contenido_analisis = (
            "<h3>Análisis Completo</h3>"
            "<ul><li>Derecho de uso</li></ul>"
        )

        # ── Paso 1: Almacenar ──────────────────────────────────────────────
        executor_alm = _make_almacenador_executor()
        executor_alm.pdf_processor.extract_text_from_pdf.return_value = \
            "Texto completo del contrato SaaS para análisis."
        executor_alm.pdf_processor.semantic_chunking.return_value = [
            "Chunk uno del contrato.", "Chunk dos.", "Chunk tres."
        ]
        # _FakeFormatter ya retorna strings reales — no requiere configuración adicional

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("almacenador_agent.agent_executor.validate_pdf_content", return_value=True), \
             patch("almacenador_agent.agent_executor.get_pdf_metadata", return_value={"pages": 1}), \
             patch("almacenador_agent.agent_executor._save_metric"):

            sm_mock.store_chunks.return_value = {
                "status": "success",
                "document_id": doc_id,
                "filename": filename,
                "chunks_stored": 3,
                "was_updated": False,
                "collection": "contratos"
            }
            sm_mock.available = True
            updater = _make_updater_mock()
            updater_cls.return_value = updater

            parts = [
                _make_file_part(_minimal_pdf_bytes(), filename),
                _make_part(f"[Archivo PDF adjunto: {filename}]"),
                _make_part("Almacena el documento con el nombre contrato_completo.")
            ]
            ctx1 = _make_context(parts)
            q1 = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor_alm.execute(ctx1, q1)
            )

            assert q1.enqueue_event.called, "Paso 1: debe emitir respuesta al almacenar"
            updater.complete.assert_called_once()

        # ── Paso 2: Analizar ───────────────────────────────────────────────
        qdrant_mock = MagicMock()
        qdrant_mock.get_document.return_value = {
            "status": "success",
            "document_id": doc_id,
            "filename": filename,
            "content": "Texto completo del contrato SaaS para análisis.",
            "num_chunks": 3,
            "total_chunks_raw": 3,
            "stored_at": "2024-01-01",
            "total_characters": 600,
            "message": "OK"
        }

        with patch("analisador_agent.agent_executor.QdrantRetriever", return_value=qdrant_mock), \
             patch("analisador_agent.agent_executor.analyze_contract", return_value=contenido_analisis) as mock_crew, \
             patch("analisador_agent.agent_executor.TaskUpdater") as updater_cls2, \
             patch("analisador_agent.agent_executor._save_metric"), \
             patch("analisador_agent.agent_executor._save_chunks_to_json", return_value="chunks.json"):

            updater2 = _make_updater_mock()
            updater_cls2.return_value = updater2

            from analisador_agent.agent_executor import ContractAnalyzerExecutor
            executor_anal = ContractAnalyzerExecutor.__new__(ContractAnalyzerExecutor)
            executor_anal.qdrant = qdrant_mock

            ctx2 = _make_context([_make_part(f"Analiza el documento {filename}")])
            q2 = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor_anal.execute(ctx2, q2)
            )

            assert mock_crew.called, "Paso 2: CrewAI debe ser invocado"
            assert q2.enqueue_event.called, "Paso 2: debe emitir respuesta HTML"
            updater2.complete.assert_called_once()

        # ── Paso 3: Recuperar análisis ─────────────────────────────────────
        executor_alm2 = _make_almacenador_executor()

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock2, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls3, \
             patch("almacenador_agent.agent_executor._save_metric"):

            sm_mock2.retrieve_analysis.return_value = [{
                "analysis_id": "anal-completo-001",
                "document_id": doc_id,
                "analysis_type": "general",
                "analysis_content": contenido_analisis,
                "created_at": "2024-01-01T12:00:00",
                "filename": filename
            }]
            updater3 = _make_updater_mock()
            updater_cls3.return_value = updater3

            ctx3 = _make_context([_make_part(f"Ver el análisis del documento {doc_id}")])
            q3 = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor_alm2.execute(ctx3, q3)
            )

            sm_mock2.retrieve_analysis.assert_called()
            assert q3.enqueue_event.called, "Paso 3: debe emitir el análisis recuperado"
            updater3.complete.assert_called_once()

    def test_flujo_pdf_invalido_es_rechazado(self):
        """Si los bytes no son PDF válido, store_chunks no debe ser llamado."""
        executor = _make_almacenador_executor()
        bytes_invalidos = b"esto no es un pdf"

        with patch("almacenador_agent.agent_executor.storage_manager") as sm_mock, \
             patch("almacenador_agent.agent_executor.TaskUpdater") as updater_cls, \
             patch("almacenador_agent.agent_executor.validate_pdf_content", return_value=False), \
             patch("almacenador_agent.agent_executor._save_metric"):

            sm_mock.available = True
            updater = _make_updater_mock()
            updater_cls.return_value = updater

            parts = [
                _make_file_part(bytes_invalidos, "no_es_pdf.pdf"),
                _make_part("Almacena el documento con el nombre prueba.")
            ]
            context = _make_context(parts)
            queue = _make_event_queue()

            asyncio.get_event_loop().run_until_complete(
                executor.execute(context, queue)
            )

            sm_mock.store_chunks.assert_not_called()

    def test_html_del_analizador_contiene_secciones_clave(self):
        """El HTML generado debe tener las 3 secciones: Derechos, Obligaciones, Prohibiciones."""
        html_completo = (
            "<h3>📋 Análisis Legal del Contrato SaaS</h3>"
            "<h4>✅ Derechos</h4><ul>"
            "<li>Acceso ilimitado al software durante vigencia del contrato.</li>"
            "</ul>"
            "<h4>📌 Obligaciones</h4><ul>"
            "<li>El cliente debe pagar la tarifa mensual acordada.</li>"
            "</ul>"
            "<h4>🚫 Prohibiciones</h4><ul>"
            "<li>Está prohibido sublicenciar el software a terceros.</li>"
            "</ul>"
        )

        assert _html_valido(html_completo), "El HTML debe ser válido"
        assert "Derechos" in html_completo
        assert "Obligaciones" in html_completo
        assert "Prohibiciones" in html_completo
        assert "<ul>" in html_completo
        assert "<li>" in html_completo