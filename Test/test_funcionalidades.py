"""
Tests unitarios — Agente Almacenador / Analizador
==================================================
Cubre los 3 módulos solicitados:

  1. qdrant_storage.py   → almacenar, recuperar, listar
  2. qdrant_retriever.py → búsqueda por nombre y UUID
  3. _extract_document_query() del analizador (agent_executor.py)

Ejecución:
    pytest test_almacenador_agent.py -v

Requisitos:
    pip install pytest qdrant-client python-dotenv

Notas técnicas:
  - SentenceTransformer se importa dentro del __init__ con try/except,
    NO es atributo global del módulo. Se usa __new__ para evitar el __init__.
  - QdrantRetriever también se construye con __new__ para evitar conexión real.
  - Los patrones regex de _extract_custom_filename requieren punto final.
"""

import hashlib
import uuid
import re
import json
import pytest
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS COMPARTIDOS
# ─────────────────────────────────────────────────────────────────────────────

def _make_point(payload: dict, point_id: str = None):
    """Crea un punto Qdrant mockeado con payload dado."""
    point = MagicMock()
    point.id = point_id or str(uuid.uuid4())
    point.payload = payload
    return point


def _scroll_result(payloads: list, next_offset=None):
    """Simula la respuesta de client.scroll()."""
    return ([_make_point(p) for p in payloads], next_offset)


def _make_storage_manager(collection_name="contratos"):
    """
    Construye QdrantStorageManager con __new__ sin pasar por __init__.
    Evita conexión real a Qdrant y carga de SentenceTransformer.
    """
    from almacenador_agent.qdrant_storage import QdrantStorageManager

    manager = QdrantStorageManager.__new__(QdrantStorageManager)
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    client.collection_exists.return_value = True

    manager.client = client
    manager.collection_name = collection_name
    manager.analysis_collection = f"{collection_name}_analysis"
    manager.available = True
    manager._embedding_size = 384

    embedding_model = MagicMock()
    embedding_model.encode.return_value = [0.1] * 384
    manager._embedding_model = embedding_model

    return manager


def _make_retriever(collection_name="contratos"):
    """
    Construye QdrantRetriever con __new__ sin pasar por __init__.
    Evita conexión real a Qdrant.
    """
    from analisador_agent.qdrant_retriever import QdrantRetriever

    retriever = QdrantRetriever.__new__(QdrantRetriever)
    client = MagicMock()
    client.collection_exists.return_value = True

    retriever.client = client
    retriever.collection_name = collection_name
    retriever.available = True

    return retriever


def _make_executor():
    """Construye AlmacenadorAgentExecutor mockeando todas sus dependencias."""
    with patch("almacenador_agent.agent_executor.root_agent"), \
         patch("almacenador_agent.agent_executor.almacenador_agent_runner"), \
         patch("almacenador_agent.agent_executor.storage_manager"), \
         patch("almacenador_agent.agent_executor.PDFProcessor"), \
         patch("almacenador_agent.agent_executor.ResponseFormatter"):
        from almacenador_agent.agent_executor import AlmacenadorAgentExecutor
        executor = AlmacenadorAgentExecutor.__new__(AlmacenadorAgentExecutor)
        executor.pdf_processor = MagicMock()
        executor.response_formatter = MagicMock()
        return executor


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 1: qdrant_storage.py — almacenar, recuperar, listar
# ═════════════════════════════════════════════════════════════════════════════

class TestAlmacenar:
    """
    Tests para store_chunks: almacenamiento de chunks en Qdrant.
    Verifica nuevo documento, deduplicación y manejo de errores.
    """

    def test_retorna_error_si_qdrant_no_disponible(self):
        manager = _make_storage_manager()
        manager.available = False
        result = manager.store_chunks(["chunk"])
        assert result["status"] == "error"
        assert result["chunks_stored"] == 0

    def test_almacena_chunks_y_retorna_success(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager.client.scroll.return_value = ([], None)
        manager.client.upsert.return_value = None

        result = manager.store_chunks(["chunk uno", "chunk dos"], filename="contrato.pdf")

        assert result["status"] == "success"
        assert result["chunks_stored"] == 2
        assert result["collection"] == "contratos"

    def test_genera_document_id_con_formato_uuid(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager.client.scroll.return_value = ([], None)
        manager.client.upsert.return_value = None

        result = manager.store_chunks(["chunk"], filename="doc.pdf")
        assert re.match(
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
            result["document_id"]
        )

    def test_detecta_duplicado_y_actualiza_en_lugar_de_duplicar(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager._check_document_exists = MagicMock(return_value={
            "exists": True,
            "document_id": "uuid-existente",
            "stored_at": "2024-01-01",
            "filename": "contrato.pdf",
            "num_chunks": 3
        })
        manager._delete_document_chunks = MagicMock(return_value=True)
        manager.client.upsert.return_value = None

        result = manager.store_chunks(
            chunks=["nuevo chunk"],
            full_content="contenido completo para hash",
            filename="contrato.pdf"
        )

        assert result["status"] == "success"
        assert result["was_updated"] is True
        assert result["document_id"] == "uuid-existente"
        manager._delete_document_chunks.assert_called_once_with("uuid-existente")

    def test_nuevo_documento_was_updated_es_false(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager.client.scroll.return_value = ([], None)
        manager.client.upsert.return_value = None

        result = manager.store_chunks(["chunk"], filename="nuevo.pdf")
        assert result["was_updated"] is False

    def test_payload_tiene_campos_obligatorios(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager.client.scroll.return_value = ([], None)
        captured = []
        manager.client.upsert.side_effect = lambda collection_name, points: captured.extend(points)

        manager.store_chunks(["texto del chunk"], filename="prueba.pdf")

        payload = captured[0].payload
        for campo in ["contenido", "document_id", "filename", "stored_at", "chunk_index", "total_chunks"]:
            assert campo in payload, f"Campo '{campo}' faltante en payload"

    def test_retorna_error_si_upsert_falla(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager.client.scroll.return_value = ([], None)
        manager.client.upsert.side_effect = Exception("timeout de conexión")

        result = manager.store_chunks(["chunk"])
        assert result["status"] == "error"
        assert result["chunks_stored"] == 0

    def test_hash_sha256_es_determinista(self):
        manager = _make_storage_manager()
        texto = "Contenido de prueba para hash"
        h1 = manager._calculate_document_hash(texto)
        h2 = manager._calculate_document_hash(texto)
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_distinto_para_textos_diferentes(self):
        manager = _make_storage_manager()
        assert manager._calculate_document_hash("A") != manager._calculate_document_hash("B")


class TestRecuperar:
    """
    Tests para retrieve_analysis y store_analysis:
    almacenamiento y recuperación de análisis vinculados a documentos.
    """

    def test_retrieve_retorna_lista_vacia_si_no_disponible(self):
        manager = _make_storage_manager()
        manager.available = False
        assert manager.retrieve_analysis() == []

    def test_retrieve_retorna_lista_vacia_si_no_hay_datos(self):
        manager = _make_storage_manager()
        manager.client.scroll.return_value = ([], None)
        assert manager.retrieve_analysis() == []

    def test_retrieve_por_document_id(self):
        manager = _make_storage_manager()
        doc_id = str(uuid.uuid4())
        manager.client.scroll.return_value = _scroll_result([{
            "document_id": doc_id,
            "analysis_type": "general",
            "analysis_content": "Contenido del análisis legal",
            "created_at": "2024-06-01T10:00:00",
            "filename": "contrato.pdf"
        }])

        result = manager.retrieve_analysis(document_id=doc_id)
        assert len(result) == 1
        assert result[0]["document_id"] == doc_id
        assert result[0]["analysis_type"] == "general"

    def test_retrieve_por_tipo_de_analisis(self):
        manager = _make_storage_manager()
        manager.client.scroll.return_value = _scroll_result([{
            "document_id": "doc-1",
            "analysis_type": "summary",
            "analysis_content": "Resumen ejecutivo",
            "created_at": "2024-06-01",
            "filename": "doc.pdf"
        }])

        result = manager.retrieve_analysis(analysis_type="summary")
        assert result[0]["analysis_type"] == "summary"

    def test_retrieve_resultado_tiene_claves_requeridas(self):
        manager = _make_storage_manager()
        manager.client.scroll.return_value = _scroll_result([{
            "document_id": "abc",
            "analysis_type": "general",
            "analysis_content": "Análisis",
            "created_at": "2024-01-01",
            "filename": "archivo.pdf"
        }])

        item = manager.retrieve_analysis()[0]
        for clave in ["analysis_id", "document_id", "analysis_type", "analysis_content", "created_at"]:
            assert clave in item

    def test_retrieve_retorna_vacio_si_scroll_falla(self):
        manager = _make_storage_manager()
        manager.client.scroll.side_effect = Exception("error de red")
        assert manager.retrieve_analysis() == []

    def test_store_analysis_retorna_error_si_no_disponible(self):
        manager = _make_storage_manager()
        manager.available = False
        result = manager.store_analysis("doc-id", "contenido")
        assert result["status"] == "error"

    def test_store_analysis_guarda_exitosamente(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager.client.upsert.return_value = None

        result = manager.store_analysis(
            document_id="doc-uuid-abc",
            analysis_content="Análisis legal del contrato",
            analysis_type="general"
        )

        assert result["status"] == "success"
        assert "analysis_id" in result
        assert result["document_id"] == "doc-uuid-abc"

    def test_store_analysis_retorna_error_si_upsert_falla(self):
        manager = _make_storage_manager()
        manager._get_embedding = MagicMock(return_value=[0.0] * 384)
        manager.client.upsert.side_effect = Exception("fallo al escribir")
        result = manager.store_analysis("doc-id", "contenido")
        assert result["status"] == "error"


class TestListar:
    """
    Tests para get_document_id_by_filename y get_filename_by_document_id:
    búsqueda y listado de documentos en storage.
    """

    def test_get_id_por_filename_retorna_none_si_no_existe(self):
        manager = _make_storage_manager()
        manager.client.scroll.return_value = ([], None)
        assert manager.get_document_id_by_filename("inexistente.pdf") is None

    def test_get_id_por_filename_retorna_id_correcto(self):
        manager = _make_storage_manager()
        manager.client.scroll.return_value = _scroll_result([{
            "document_id": "id-encontrado",
            "filename": "contrato.pdf"
        }])
        assert manager.get_document_id_by_filename("contrato.pdf") == "id-encontrado"

    def test_get_id_por_filename_retorna_none_si_falla(self):
        manager = _make_storage_manager()
        manager.client.scroll.side_effect = Exception("error")
        assert manager.get_document_id_by_filename("contrato.pdf") is None

    def test_get_filename_por_id_retorna_no_disponible_si_no_existe(self):
        manager = _make_storage_manager()
        manager.client.scroll.return_value = ([], None)
        assert manager.get_filename_by_document_id("uuid-x") == "No disponible"

    def test_get_filename_por_id_retorna_nombre_correcto(self):
        manager = _make_storage_manager()
        manager.client.scroll.return_value = _scroll_result([{"filename": "mi_contrato.pdf"}])
        assert manager.get_filename_by_document_id("uuid-real") == "mi_contrato.pdf"

    def test_get_filename_por_id_retorna_no_disponible_si_falla(self):
        manager = _make_storage_manager()
        manager.client.scroll.side_effect = Exception("fallo")
        assert manager.get_filename_by_document_id("uuid-x") == "No disponible"

    def test_get_stats_retorna_error_si_no_disponible(self):
        manager = _make_storage_manager()
        manager.available = False
        result = manager.get_stats()
        assert result["status"] == "error"

    def test_get_stats_estructura_correcta(self):
        manager = _make_storage_manager()
        # Simular un chunk con document_id
        manager.client.scroll.return_value = _scroll_result([{
            "document_id": "doc-1",
            "filename": "contrato.pdf",
            "stored_at": "2024-01-01"
        }])

        result = manager.get_stats()
        assert result["status"] == "success"
        assert "documents" in result
        assert "chunks" in result


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 2: qdrant_retriever.py — búsqueda por nombre y UUID
# ═════════════════════════════════════════════════════════════════════════════

class TestGetDocumentByUUID:
    """
    Tests para get_document_by_id y get_document():
    búsqueda de documentos por UUID exacto.
    """

    def test_retorna_error_si_no_disponible(self):
        retriever = _make_retriever()
        retriever.available = False
        result = retriever.get_document_by_id("cualquier-uuid")
        assert result["status"] == "error"
        assert result["content"] is None

    def test_retorna_not_found_si_no_hay_chunks(self):
        retriever = _make_retriever()
        retriever.client.scroll.return_value = ([], None)
        doc_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        result = retriever.get_document_by_id(doc_id)
        assert result["status"] == "not_found"
        assert result["document_id"] == doc_id

    def test_retorna_success_con_documento_valido(self):
        retriever = _make_retriever()
        chunks = [
            {"document_id": "doc-1", "filename": "contrato.pdf",
             "contenido": "Este es el primer párrafo del contrato legal.",
             "chunk_index": 0, "stored_at": "2024-01-01"},
            {"document_id": "doc-1", "filename": "contrato.pdf",
             "contenido": "Este es el segundo párrafo con más información relevante.",
             "chunk_index": 1, "stored_at": "2024-01-01"},
        ]
        retriever.client.scroll.return_value = _scroll_result(chunks)

        result = retriever.get_document_by_id("doc-1")
        assert result["status"] == "success"
        assert result["document_id"] == "doc-1"
        assert result["filename"] == "contrato.pdf"
        assert "content" in result
        assert len(result["content"]) > 0

    def test_reconstruye_contenido_en_orden_de_chunks(self):
        retriever = _make_retriever()
        # Entregar chunks en orden invertido para verificar que se reordena
        chunks = [
            {"document_id": "doc-1", "filename": "doc.pdf",
             "contenido": "Segundo párrafo del documento extenso aquí.",
             "chunk_index": 1, "stored_at": "2024-01-01"},
            {"document_id": "doc-1", "filename": "doc.pdf",
             "contenido": "Primer párrafo del documento extenso aquí.",
             "chunk_index": 0, "stored_at": "2024-01-01"},
        ]
        retriever.client.scroll.return_value = _scroll_result(chunks)

        result = retriever.get_document_by_id("doc-1")
        assert result["status"] == "success"
        # El primer párrafo debe aparecer antes que el segundo
        assert result["content"].index("Primer") < result["content"].index("Segundo")

    def test_retorna_error_si_scroll_falla(self):
        retriever = _make_retriever()
        retriever.client.scroll.side_effect = Exception("conexión perdida")
        result = retriever.get_document_by_id("doc-1")
        assert result["status"] == "error"

    def test_get_document_detecta_uuid_y_llama_by_id(self):
        retriever = _make_retriever()
        retriever.get_document_by_id = MagicMock(return_value={"status": "success"})
        retriever.get_document_by_name = MagicMock()

        uuid_val = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        retriever.get_document(uuid_val)

        retriever.get_document_by_id.assert_called_once_with(uuid_val)
        retriever.get_document_by_name.assert_not_called()

    def test_get_document_detecta_nombre_y_llama_by_name(self):
        retriever = _make_retriever()
        retriever.get_document_by_id = MagicMock()
        retriever.get_document_by_name = MagicMock(return_value={"status": "success"})

        retriever.get_document("contrato_servicios")

        retriever.get_document_by_name.assert_called_once_with("contrato_servicios")
        retriever.get_document_by_id.assert_not_called()

    def test_get_document_retorna_error_si_no_disponible(self):
        retriever = _make_retriever()
        retriever.available = False
        result = retriever.get_document("cualquier-cosa")
        assert result["status"] == "error"


class TestGetDocumentByName:
    """
    Tests para get_document_by_name:
    búsqueda de documentos por nombre de archivo (búsqueda parcial).
    """

    def test_retorna_error_si_no_disponible(self):
        retriever = _make_retriever()
        retriever.available = False
        result = retriever.get_document_by_name("contrato")
        assert result["status"] == "error"

    def test_retorna_not_found_si_no_hay_documentos(self):
        retriever = _make_retriever()
        retriever.client.scroll.return_value = ([], None)
        result = retriever.get_document_by_name("contrato")
        assert result["status"] == "not_found"

    def test_retorna_not_found_si_nombre_no_coincide(self):
        retriever = _make_retriever()
        # Hay documentos pero ninguno coincide con el nombre buscado
        retriever.client.scroll.return_value = _scroll_result([{
            "document_id": "doc-1",
            "filename": "otro_archivo.pdf",
            "contenido": "Contenido del otro archivo aquí.",
            "chunk_index": 0,
            "stored_at": "2024-01-01"
        }])
        result = retriever.get_document_by_name("contrato_buscado")
        assert result["status"] == "not_found"

    def test_retorna_success_con_nombre_exacto(self):
        retriever = _make_retriever()
        retriever.client.scroll.return_value = _scroll_result([{
            "document_id": "doc-xyz",
            "filename": "contrato_2024.pdf",
            "contenido": "Las partes acuerdan los siguientes términos y condiciones.",
            "chunk_index": 0,
            "stored_at": "2024-01-01"
        }])

        result = retriever.get_document_by_name("contrato_2024")
        assert result["status"] == "success"
        assert result["document_id"] == "doc-xyz"
        assert result["filename"] == "contrato_2024.pdf"

    def test_busqueda_sin_extension_pdf_funciona(self):
        """Buscar 'contrato' debe encontrar 'contrato.pdf'."""
        retriever = _make_retriever()
        retriever.client.scroll.return_value = _scroll_result([{
            "document_id": "doc-abc",
            "filename": "contrato.pdf",
            "contenido": "Contenido válido del contrato con suficientes palabras.",
            "chunk_index": 0,
            "stored_at": "2024-01-01"
        }])

        result = retriever.get_document_by_name("contrato")
        assert result["status"] == "success"

    def test_retorna_ambiguous_si_hay_multiples_coincidencias(self):
        retriever = _make_retriever()
        # Dos documentos con el mismo nombre base
        retriever.client.scroll.return_value = _scroll_result([
            {"document_id": "doc-1", "filename": "contrato.pdf",
             "contenido": "Primer contrato firmado.", "chunk_index": 0, "stored_at": "2024-01-01"},
            {"document_id": "doc-2", "filename": "contrato.pdf",
             "contenido": "Segundo contrato firmado.", "chunk_index": 0, "stored_at": "2024-06-01"},
        ])

        result = retriever.get_document_by_name("contrato")
        assert result["status"] == "ambiguous"
        assert "matches" in result
        assert len(result["matches"]) == 2

    def test_retorna_error_si_scroll_falla(self):
        retriever = _make_retriever()
        retriever.client.scroll.side_effect = Exception("fallo de red")
        result = retriever.get_document_by_name("contrato")
        assert result["status"] == "error"


class TestListDocuments:
    """
    Tests para list_documents: listado de documentos disponibles en Qdrant.
    """

    def test_retorna_lista_vacia_si_no_disponible(self):
        retriever = _make_retriever()
        retriever.available = False
        assert retriever.list_documents() == []

    def test_retorna_lista_vacia_si_no_hay_documentos(self):
        retriever = _make_retriever()
        retriever.client.scroll.return_value = ([], None)
        assert retriever.list_documents() == []

    def test_lista_documentos_unicos(self):
        retriever = _make_retriever()
        # 3 chunks del mismo documento
        retriever.client.scroll.return_value = _scroll_result([
            {"document_id": "doc-1", "filename": "contrato.pdf", "stored_at": "2024-01-01"},
            {"document_id": "doc-1", "filename": "contrato.pdf", "stored_at": "2024-01-01"},
            {"document_id": "doc-1", "filename": "contrato.pdf", "stored_at": "2024-01-01"},
        ])

        result = retriever.list_documents()
        # Debe retornar 1 documento único, no 3 chunks
        assert len(result) == 1
        assert result[0]["document_id"] == "doc-1"

    def test_cuenta_chunks_por_documento(self):
        retriever = _make_retriever()
        retriever.client.scroll.return_value = _scroll_result([
            {"document_id": "doc-1", "filename": "contrato.pdf", "stored_at": "2024-01-01"},
            {"document_id": "doc-1", "filename": "contrato.pdf", "stored_at": "2024-01-01"},
            {"document_id": "doc-2", "filename": "informe.pdf",  "stored_at": "2024-06-01"},
        ])

        result = retriever.list_documents()
        assert len(result) == 2
        doc1 = next(d for d in result if d["document_id"] == "doc-1")
        assert doc1["num_chunks"] == 2

    def test_ordena_por_fecha_mas_reciente(self):
        retriever = _make_retriever()
        retriever.client.scroll.return_value = _scroll_result([
            {"document_id": "doc-antiguo", "filename": "viejo.pdf",   "stored_at": "2023-01-01"},
            {"document_id": "doc-reciente", "filename": "nuevo.pdf",  "stored_at": "2024-12-01"},
        ])

        result = retriever.list_documents()
        assert result[0]["document_id"] == "doc-reciente"

    def test_respeta_limite_de_documentos(self):
        retriever = _make_retriever()
        # 5 documentos distintos
        payloads = [
            {"document_id": f"doc-{i}", "filename": f"archivo_{i}.pdf", "stored_at": f"2024-0{i+1}-01"}
            for i in range(5)
        ]
        retriever.client.scroll.return_value = _scroll_result(payloads)

        result = retriever.list_documents(limit=3)
        assert len(result) <= 3

    def test_resultado_tiene_campos_requeridos(self):
        retriever = _make_retriever()
        retriever.client.scroll.return_value = _scroll_result([
            {"document_id": "doc-1", "filename": "contrato.pdf", "stored_at": "2024-01-01"},
        ])

        result = retriever.list_documents()
        for campo in ["document_id", "filename", "stored_at", "num_chunks"]:
            assert campo in result[0]

    def test_retorna_lista_vacia_si_scroll_falla(self):
        retriever = _make_retriever()
        retriever.client.scroll.side_effect = Exception("timeout")
        assert retriever.list_documents() == []


class TestBuildResultFromPoints:
    """
    Tests para _build_result_from_points:
    reconstrucción del texto y filtrado de chunks irrelevantes.
    """

    def test_ordena_chunks_por_index(self):
        retriever = _make_retriever()
        points = [
            _make_point({"contenido": "Segundo párrafo válido aquí.", "chunk_index": 1,
                         "filename": "doc.pdf", "stored_at": "2024-01-01"}),
            _make_point({"contenido": "Primer párrafo válido aquí.", "chunk_index": 0,
                         "filename": "doc.pdf", "stored_at": "2024-01-01"}),
        ]
        result = retriever._build_result_from_points(points, "doc-1")
        assert result["content"].index("Primer") < result["content"].index("Segundo")

    def test_descarta_chunks_vacios(self):
        retriever = _make_retriever()
        points = [
            _make_point({"contenido": "", "chunk_index": 0,
                         "filename": "doc.pdf", "stored_at": "2024-01-01"}),
            _make_point({"contenido": "Este es un párrafo válido con contenido.", "chunk_index": 1,
                         "filename": "doc.pdf", "stored_at": "2024-01-01"}),
        ]
        result = retriever._build_result_from_points(points, "doc-1")
        assert result["num_chunks"] == 1

    def test_descarta_chunks_con_dos_palabras_o_menos(self):
        retriever = _make_retriever()
        points = [
            _make_point({"contenido": "Solo dos", "chunk_index": 0,
                         "filename": "doc.pdf", "stored_at": "2024-01-01"}),
            _make_point({"contenido": "Este chunk tiene suficientes palabras para ser válido.",
                         "chunk_index": 1, "filename": "doc.pdf", "stored_at": "2024-01-01"}),
        ]
        result = retriever._build_result_from_points(points, "doc-1")
        assert result["num_chunks"] == 1

    def test_descarta_chunks_sin_letras(self):
        retriever = _make_retriever()
        points = [
            _make_point({"contenido": "123 456 789 000", "chunk_index": 0,
                         "filename": "doc.pdf", "stored_at": "2024-01-01"}),
            _make_point({"contenido": "Texto válido con letras y significado aquí.",
                         "chunk_index": 1, "filename": "doc.pdf", "stored_at": "2024-01-01"}),
        ]
        result = retriever._build_result_from_points(points, "doc-1")
        assert result["num_chunks"] == 1

    def test_descarta_chunks_mayormente_en_mayusculas(self):
        retriever = _make_retriever()
        points = [
            _make_point({"contenido": "TÍTULO EN MAYÚSCULAS COMPLETO", "chunk_index": 0,
                         "filename": "doc.pdf", "stored_at": "2024-01-01"}),
            _make_point({"contenido": "Este es contenido normal en minúsculas válido.",
                         "chunk_index": 1, "filename": "doc.pdf", "stored_at": "2024-01-01"}),
        ]
        result = retriever._build_result_from_points(points, "doc-1")
        assert result["num_chunks"] == 1

    def test_retorna_estructura_completa(self):
        retriever = _make_retriever()
        points = [
            _make_point({"contenido": "Contenido válido del documento aquí.",
                         "chunk_index": 0, "filename": "contrato.pdf", "stored_at": "2024-01-01"}),
        ]
        result = retriever._build_result_from_points(points, "doc-xyz")
        for campo in ["status", "document_id", "filename", "content", "num_chunks",
                      "total_chunks_raw", "total_characters", "message"]:
            assert campo in result

    def test_status_es_success(self):
        retriever = _make_retriever()
        points = [
            _make_point({"contenido": "Texto completo y válido para el test.",
                         "chunk_index": 0, "filename": "doc.pdf", "stored_at": "2024-01-01"}),
        ]
        result = retriever._build_result_from_points(points, "doc-1")
        assert result["status"] == "success"


# ═════════════════════════════════════════════════════════════════════════════
# MÓDULO 3: _extract_document_query() del analizador (agent_executor.py)
# ═════════════════════════════════════════════════════════════════════════════

class TestExtractDocumentQuery:
    """
    Tests para la lógica de _extract_document_query tal como existe en
    _handle_retrieve_analysis del AlmacenadorAgentExecutor.

    Lógica:
      1. Busca UUID de 36 caracteres en el texto → document_id directo
      2. Si no hay UUID, busca patrón *.pdf y resuelve via storage_manager
      3. Si no hay nada, retorna None → recupera todos los análisis
    """

    @staticmethod
    def _extract(user_text: str, storage_mock=None):
        """Replica exactamente la lógica de _handle_retrieve_analysis."""
        doc_id_pattern = r'([a-f0-9\-]{36})'
        match = re.search(doc_id_pattern, user_text)
        document_id = match.group(1) if match else None

        if not document_id:
            filename_pattern = r'[\w\-]+\.pdf'
            filename_match = re.search(filename_pattern, user_text, re.IGNORECASE)
            if filename_match and storage_mock:
                filename = filename_match.group(0)
                document_id = storage_mock.get_document_id_by_filename(filename)

        return document_id

    # ── Tests: extracción por UUID ────────────────────────────────────────────

    def test_extrae_uuid_valido_del_texto(self):
        text = "Muestra el análisis del documento a97c3cb5-1234-4abc-89ef-000000000001"
        assert self._extract(text) == "a97c3cb5-1234-4abc-89ef-000000000001"

    def test_extrae_uuid_en_medio_de_texto_largo(self):
        uuid_val = "ffffffff-aaaa-4444-bbbb-cccccccccccc"
        text = f"El análisis del contrato {uuid_val} muestra cláusulas importantes."
        assert self._extract(text) == uuid_val

    def test_extrae_primer_uuid_si_hay_varios(self):
        uuid1 = "aaaaaaaa-1111-2222-3333-444444444444"
        uuid2 = "bbbbbbbb-5555-6666-7777-888888888888"
        text = f"Documento {uuid1} y también {uuid2}"
        assert self._extract(text) == uuid1

    def test_uuid_en_minusculas(self):
        uuid_val = "abcdef12-3456-7890-abcd-ef1234567890"
        assert self._extract(f"Ver análisis {uuid_val}") == uuid_val

    def test_no_detecta_cadena_corta_como_uuid(self):
        """Cadenas de menos de 36 chars no son UUIDs."""
        text = "El documento abcd-1234 tiene análisis"
        assert self._extract(text) is None

    # ── Tests: extracción por nombre de archivo ───────────────────────────────

    def test_extrae_filename_y_resuelve_via_storage(self):
        storage_mock = MagicMock()
        storage_mock.get_document_id_by_filename.return_value = "uuid-resuelto"
        result = self._extract("Ver el análisis de contrato_2024.pdf", storage_mock)
        assert result == "uuid-resuelto"
        storage_mock.get_document_id_by_filename.assert_called_once_with("contrato_2024.pdf")

    def test_uuid_tiene_prioridad_sobre_filename(self):
        """Si hay UUID y filename en el mismo texto, UUID gana."""
        storage_mock = MagicMock()
        uuid_val = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        text = f"Análisis del documento {uuid_val} guardado en contrato.pdf"
        result = self._extract(text, storage_mock)
        assert result == uuid_val
        storage_mock.get_document_id_by_filename.assert_not_called()

    def test_retorna_none_si_filename_no_existe_en_storage(self):
        storage_mock = MagicMock()
        storage_mock.get_document_id_by_filename.return_value = None
        result = self._extract("Ver análisis de doc_inexistente.pdf", storage_mock)
        assert result is None

    def test_sin_storage_no_puede_resolver_filename(self):
        """Sin storage_manager, el filename no se puede resolver."""
        result = self._extract("Ver el análisis de contrato.pdf", storage_mock=None)
        assert result is None

    # ── Tests: sin UUID ni filename ───────────────────────────────────────────

    def test_retorna_none_si_no_hay_uuid_ni_filename(self):
        assert self._extract("Ver todos los análisis disponibles") is None

    def test_retorna_none_con_texto_vacio(self):
        assert self._extract("") is None

    def test_retorna_none_con_solo_texto_generico(self):
        assert self._extract("Muéstrame el análisis del contrato reciente") is None


# ═════════════════════════════════════════════════════════════════════════════
# EXTRA: _detect_operation_type — detección de intención del usuario
# ═════════════════════════════════════════════════════════════════════════════

class TestDetectOperationType:
    """Tests para _detect_operation_type del AlmacenadorAgentExecutor."""

    def setup_method(self):
        self.executor = _make_executor()

    def _parts_con_pdf(self):
        from a2a.types import FilePart, Part
        part = MagicMock(spec=Part)
        part.root = MagicMock(spec=FilePart)
        return [part]

    def test_store_pdf_cuando_hay_archivo_adjunto(self):
        assert self.executor._detect_operation_type("texto", self._parts_con_pdf()) == "store_pdf"

    def test_store_analysis_almacena_el_analisis(self):
        assert self.executor._detect_operation_type("almacena el análisis", []) == "store_analysis"

    def test_store_analysis_guarda_el_analisis(self):
        assert self.executor._detect_operation_type("guarda el análisis", []) == "store_analysis"

    def test_retrieve_analysis_recupera(self):
        assert self.executor._detect_operation_type("recupera el análisis", []) == "retrieve_analysis"

    def test_retrieve_analysis_muestra(self):
        assert self.executor._detect_operation_type("muestra el análisis", []) == "retrieve_analysis"

    def test_retrieve_analysis_ver(self):
        assert self.executor._detect_operation_type("ver el análisis", []) == "retrieve_analysis"

    def test_get_stats_cuantos_documentos(self):
        assert self.executor._detect_operation_type("cuantos documentos hay", []) == "get_stats"

    def test_get_stats_estadisticas(self):
        assert self.executor._detect_operation_type("estadísticas del sistema", []) == "get_stats"

    def test_get_analyzed_docs_documentos_analizados(self):
        assert self.executor._detect_operation_type("documentos analizados", []) == "get_analyzed_docs"

    def test_get_analyzed_docs_tienen_analisis(self):
        assert self.executor._detect_operation_type("qué documentos tienen análisis", []) == "get_analyzed_docs"

    def test_unknown_sin_coincidencias(self):
        assert self.executor._detect_operation_type("hola, como estas?", []) == "unknown"

    def test_pdf_tiene_prioridad_sobre_keywords(self):
        parts = self._parts_con_pdf()
        assert self.executor._detect_operation_type("almacena el análisis", parts) == "store_pdf"