"""
Prueba simple para verificar que el servidor A2A está funcionando.
Compatible con todas las versiones de la librería A2A.
"""
import asyncio
import httpx


async def test_conexion():
    """Prueba básica de conectividad con el servidor."""
    
    base_url = 'http://localhost:8002'
    
    print("🔍 Verificando servidor A2A...")
    print(f"📍 URL base: {base_url}\n")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        
        # Paso 1: Verificar que el servidor responde
        try:
            print("1️⃣ Probando conectividad básica...")
            response = await client.get(f"{base_url}/.well-known/agent-card.json")
            
            if response.status_code == 200:
                print("   ✅ Servidor responde correctamente")
                print(f"   📄 Código HTTP: {response.status_code}")
                
                # Mostrar la tarjeta del agente
                card_data = response.json()
                print(f"\n   📋 Información del agente:")
                print(f"   • Nombre: {card_data.get('name')}")
                print(f"   • Descripción: {card_data.get('description')[:60]}...")
                print(f"   • Versión: {card_data.get('version')}")
                print(f"   • Protocolo: {card_data.get('protocolVersion')}")
                print(f"   • Transporte: {card_data.get('preferredTransport')}")
                
                # Mostrar capacidades
                capabilities = card_data.get('capabilities', {})
                print(f"\n   ⚡ Capacidades:")
                print(f"   • Streaming: {capabilities.get('streaming', False)}")
                print(f"   • Push Notifications: {capabilities.get('pushNotifications', False)}")
                
            else:
                print(f"   ❌ Error: código {response.status_code}")
                print(f"   📄 Respuesta: {response.text[:200]}")
                return False
                
        except httpx.ConnectError:
            print("   ❌ No se pudo conectar al servidor")
            print("\n   💡 Verifica que el servidor esté corriendo:")
            print("      python __main__.py --host localhost --port 8002")
            print("\n   🔍 O intenta manualmente:")
            print(f"      curl {base_url}/.well-known/agent-card.json")
            return False
        except Exception as e:
            print(f"   ❌ Error inesperado: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Paso 2: Probar con A2ACardResolver
        try:
            print("\n2️⃣ Probando A2ACardResolver...")
            from a2a.client import A2ACardResolver
            
            resolver = A2ACardResolver(
                httpx_client=client,
                base_url=base_url,
            )
            
            agent_card = await resolver.get_agent_card()
            
            print("   ✅ A2ACardResolver funcionó correctamente")
            print(f"   🤖 Agente: {agent_card.name}")
            
        except ImportError as e:
            print(f"   ⚠️  Error de importación: {e}")
            print("   💡 Verifica la instalación: pip install a2a-sdk")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Paso 3: Crear transporte JSON-RPC
        try:
            print("\n3️⃣ Creando transporte JSON-RPC...")
            from a2a.client.transports.jsonrpc import JSONRPCTransport
            
            transport = JSONRPCTransport(
                httpx_client=client,
                agent_card=agent_card,
            )
            
            print("   ✅ JSONRPCTransport creado correctamente")
            
        except ImportError as e:
            print(f"   ⚠️  Error de importación: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
        
        # Paso 4: Enviar mensaje de prueba
        try:
            print("\n4️⃣ Enviando mensaje de prueba...")
            from uuid import uuid4
            from a2a.types import MessageSendParams, SendMessageRequest
            
            mensaje = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {
                            'kind': 'text',
                            'text': '¿Puedes ayudarme a analizar un contrato?'
                        }
                    ],
                    'message_id': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**mensaje)
            )
            
            print("   ⏳ Enviando petición al agente...")
            response = await transport.send_message(request)
            
            print("   ✅ Mensaje enviado y recibido correctamente")
            
            # Verificar la respuesta
            if hasattr(response, 'root') and hasattr(response.root, 'result'):
                result = response.root.result
                print(f"   📨 Estado de la tarea: {result.state}")
                print(f"   🆔 Task ID: {result.id[:16]}...")
                
                # Mostrar respuesta del agente
                if result.messages:
                    print("\n   💬 Respuesta del agente:")
                    for msg in result.messages:
                        if hasattr(msg, 'parts'):
                            for part in msg.parts:
                                if hasattr(part.root, 'text'):
                                    texto = part.root.text
                                    # Mostrar solo los primeros 150 caracteres
                                    if len(texto) > 150:
                                        texto = texto[:150] + "..."
                                    print(f"      {texto}")
                
                if result.artifacts:
                    print(f"\n   📄 Artefactos generados: {len(result.artifacts)}")
            else:
                print("   ⚠️  Respuesta en formato inesperado")
                print(f"   📄 Respuesta: {response}")
            
        except Exception as e:
            print(f"   ❌ Error enviando mensaje: {e}")
            print(f"   🔍 Tipo de error: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n" + "="*70)
        print("✅ TODAS LAS PRUEBAS PASARON CORRECTAMENTE")
        print("="*70)
        print("\n💡 Siguiente paso: ejecuta el cliente completo")
        print("   python test.py")
        print("\n")
        return True


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║  🧪 Prueba Simple de Conectividad A2A                       ║
║  Verifica que tu servidor esté funcionando correctamente    ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    resultado = asyncio.run(test_conexion())
    
    if not resultado:
        print("\n❌ Algunas pruebas fallaron. Revisa los errores arriba.")
        print("\n📚 Checklist de solución de problemas:")
        print("   □ Servidor corriendo: python __main__.py --port 8002")
        print("   □ Puerto correcto: 8002")
        print("   □ GOOGLE_API_KEY configurada en .env")
        print("   □ Dependencias instaladas: pip install -r requirements.txt")
        exit(1)
    else:
        print("🎉 ¡Todo funciona correctamente!")
        exit(0)