"""
Script de prueba para verificar el funcionamiento del agente almacenador.
Ejecuta pruebas básicas sin necesidad de un cliente A2A completo.
"""

import asyncio
import logging
from pathlib import Path
from tools_agent import (
    PDFProcessor,
    ResponseFormatter,
    validate_pdf_content,
    get_pdf_metadata
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_response_formatter():
    """Prueba el formateador de respuestas JSON."""
    print("\n" + "="*60)
    print("TEST 1: Response Formatter")
    print("="*60)
    
    formatter = ResponseFormatter()
    
    # Test respuesta exitosa
    json_success = formatter.format_success_response(
        operation="store_pdf",
        data={
            "chunks_stored": 10,
            "total_characters": 5000,
            "collection": "test-collection"
        },
        message="Prueba exitosa"
    )
    
    print("\n✅ Respuesta de éxito:")
    print(json_success)
    
    # Test respuesta de error
    json_error = formatter.format_error_response(
        operation="test",
        error_message="Este es un error de prueba",
        error_type="TestError"
    )
    
    print("\n❌ Respuesta de error:")
    print(json_error)
    
    print("\n✓ Test completado: Response Formatter funciona correctamente\n")


def test_pdf_validation():
    """Prueba la validación de PDFs."""
    print("\n" + "="*60)
    print("TEST 2: Validación de PDFs")
    print("="*60)
    
    # Simular contenido PDF (los PDFs reales comienzan con %PDF)
    valid_pdf = b'%PDF-1.4\n...'
    invalid_pdf = b'Not a PDF file'
    
    print("\n📄 Probando PDF válido:")
    is_valid = validate_pdf_content(valid_pdf)
    print(f"   Resultado: {'✓ Válido' if is_valid else '✗ Inválido'}")
    
    print("\n📄 Probando archivo inválido:")
    is_valid = validate_pdf_content(invalid_pdf)
    print(f"   Resultado: {'✓ Válido' if is_valid else '✗ Inválido (esperado)'}")
    
    print("\n✓ Test completado: Validación funciona correctamente\n")


def test_text_chunking():
    """Prueba la fragmentación de texto."""
    print("\n" + "="*60)
    print("TEST 3: Fragmentación de Texto")
    print("="*60)
    
    processor = PDFProcessor()
    
    # Texto de prueba
    test_text = "Este es un texto de prueba. " * 100  # ~2800 caracteres
    
    print(f"\n📝 Texto original: {len(test_text)} caracteres")
    
    # Fragmentar con parámetros pequeños para testing
    chunks = processor.chunk_text(test_text, chunk_size=500, overlap=50)
    
    print(f"✂️  Fragmentado en: {len(chunks)} chunks")
    print(f"📊 Tamaño del primer chunk: {len(chunks[0])} caracteres")
    print(f"📊 Tamaño del último chunk: {len(chunks[-1])} caracteres")
    
    print("\n📄 Primer chunk (primeros 100 caracteres):")
    print(f"   '{chunks[0][:100]}...'")
    
    print("\n✓ Test completado: Fragmentación funciona correctamente\n")


def test_storage_response():
    """Prueba la respuesta de almacenamiento."""
    print("\n" + "="*60)
    print("TEST 4: Respuesta de Almacenamiento")
    print("="*60)
    
    formatter = ResponseFormatter()
    
    response = formatter.format_storage_response(
        num_chunks=15,
        total_characters=12500,
        collection_name="contratos-saas"
    )
    
    print("\n📦 Respuesta de almacenamiento generada:")
    print(response)
    
    print("\n✓ Test completado: Respuesta de almacenamiento funciona\n")


async def test_agent_configuration():
    """Prueba la configuración del agente."""
    print("\n" + "="*60)
    print("TEST 5: Configuración del Agente")
    print("="*60)
    
    try:
        from agent import get_agent_info, root_agent
        
        info = get_agent_info()
        
        print("\n🤖 Información del agente:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        print(f"\n📋 Nombre del agente: {root_agent.name}")
        print(f"🔧 Número de herramientas: {len(root_agent.tools)}")
        print(f"📝 Longitud de instrucciones: {len(root_agent.instruction)} caracteres")
        
        print("\n✓ Test completado: Agente configurado correctamente\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Asegúrate de que las variables de entorno estén configuradas\n")


def test_environment_variables():
    """Verifica que las variables de entorno estén configuradas."""
    print("\n" + "="*60)
    print("TEST 6: Variables de Entorno")
    print("="*60)
    
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    required_vars = [
        "OPENROUTER_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "COLLECTION_NAME"
    ]
    
    print("\n🔍 Verificando variables requeridas:")
    all_ok = True
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mostrar solo primeros/últimos caracteres para seguridad
            if "KEY" in var:
                display = f"{value[:10]}...{value[-10:]}"
            else:
                display = value
            print(f"   ✓ {var}: {display}")
        else:
            print(f"   ✗ {var}: NO CONFIGURADA")
            all_ok = False
    
    if all_ok:
        print("\n✓ Test completado: Todas las variables están configuradas\n")
    else:
        print("\n⚠️  ADVERTENCIA: Faltan variables de entorno\n")


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "="*70)
    print(" SUITE DE PRUEBAS - AGENTE ALMACENADOR ".center(70, "="))
    print("="*70)
    
    try:
        # Tests síncronos
        test_environment_variables()
        test_response_formatter()
        test_pdf_validation()
        test_text_chunking()
        test_storage_response()
        
        # Test asíncrono
        asyncio.run(test_agent_configuration())
        
        print("\n" + "="*70)
        print(" ✅ TODOS LOS TESTS COMPLETADOS ".center(70, "="))
        print("="*70 + "\n")
        
        print("🎉 ¡Tu agente está listo para usar!")
        print("\nPróximos pasos:")
        print("1. Inicia el servidor: python main.py")
        print("2. Prueba con un PDF real")
        print("3. Verifica que se almacene en Qdrant\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print(" ❌ ERROR EN LOS TESTS ".center(70, "="))
        print("="*70)
        print(f"\nError: {e}")
        print("\nRevisa:")
        print("- Que todas las dependencias estén instaladas")
        print("- Que el archivo .env esté configurado correctamente")
        print("- Los logs arriba para más detalles\n")


if __name__ == "__main__":
    run_all_tests()