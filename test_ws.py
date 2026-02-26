import asyncio
import websockets
import json
import sys
import time

async def test_chat():
    uri = "ws://localhost:8000/ws/chat"
    retries = 30
    print("=== INICIANDO PRUEBA DE WEBSOCKETS ===")
    
    for i in range(retries):
        try:
            print(f"\n[Intento {i+1}] Conectando a {uri}...")
            async with websockets.connect(uri) as websocket:
                print("✅ ¡Conectado con éxito al servidor!")
                request = {
                    "messages": [{"role": "user", "content": "Di exactamente 'Prueba de streaming exitosa' y nada más."}],
                    "temperature": 0.7,
                    "max_tokens": 50
                }
                print("Enviando petición de prueba al modelo AI...")
                await websocket.send(json.dumps(request))
                
                print("⬇️  Recibiendo stream en tiempo real: ⬇️")
                print("-" * 40)
                
                received_text = ""
                while True:
                    response = await websocket.recv()
                    if response == "[DONE]":
                        print("\n" + "-" * 40)
                        print("✅ [El stream ha finalizado correctamente]")
                        return True
                    print(response, end="", flush=True)
                    received_text += response
        except Exception as e:
            print(f"⚠️ Error de conexión: {e}. Reintentando en 2 segundos... (Asegúrate de que el servidor cargue OpenVINO)")
            time.sleep(2)
            
    print("❌ Se superaron los reintentos. No se pudo conectar.")
    return False

if __name__ == "__main__":
    success = asyncio.run(test_chat())
    sys.exit(0 if success else 1)
