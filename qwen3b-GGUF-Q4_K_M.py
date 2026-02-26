import os
import logging
import time
import threading
import queue
import asyncio
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Importación segura
try:
    import openvino_genai as ov_genai
except ImportError:
    print("⚠️ ADVERTENCIA: openvino_genai no está instalado.")
    ov_genai = None

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CLASE DEL MODELO (OPTIMIZADA) ---
class QwenGenAIModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.pipe = None
        return cls._instance
    
    def initialize(self):
        if self._initialized:
            return
        
        try:
            logger.info("🚀 Inicializando motor de IA...")
            
            model_path = Path(__file__).parent / "model_cache" / "qwen25_3b_gguf" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
            
            if not model_path.exists():
                raise FileNotFoundError(f"No se encontró el archivo GGUF en {model_path}")
            
            if ov_genai:
                # INTENTO 1: CARGAR EN GPU (Máxima Velocidad)
                try:
                    logger.info("⚡ Intentando cargar modelo en GPU (Intel iGPU/Arc)...")
                    # 'GPU' usa la gráfica integrada. Si tienes varias, usa 'GPU.0' o 'GPU.1'
                    self.pipe = ov_genai.LLMPipeline(str(model_path), "GPU")
                    logger.info("✅ ¡ÉXITO! Modelo cargado en GPU. El rendimiento será máximo.")
                except Exception as e_gpu:
                    logger.warning(f"⚠️ No se pudo cargar en GPU ({str(e_gpu)}). Cambiando a CPU.")
                    # FALLBACK: CARGAR EN CPU
                    self.pipe = ov_genai.LLMPipeline(str(model_path), "CPU")
                    logger.info("✅ Modelo cargado en CPU (Modo compatibilidad).")
            else:
                logger.error("❌ Librería openvino_genai no detectada.")

            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error crítico al inicializar el modelo: {str(e)}")

    def _build_qwen_prompt(self, messages: List[dict]) -> str:
        """Construcción manual eficiente del prompt ChatML"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def stream_response(self, messages: List[dict], **kwargs):
        """
        Generador asíncrono que produce tokens a medida que se crean.
        Usa threading para no bloquear el servidor FastAPI.
        """
        if not self.pipe:
            raise RuntimeError("Modelo no inicializado")

        # 1. Configuración de parámetros
        prompt = self._build_qwen_prompt(messages)
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = kwargs.get("max_tokens", 2048)
        config.temperature = kwargs.get("temperature", 0.7)
        config.do_sample = kwargs.get("do_sample", True)
        config.repetition_penalty = kwargs.get("repetition_penalty", 1.1)

        # 2. Cola para comunicación entre hilos (Thread-safe queue)
        token_queue = queue.Queue()
        
        # Evento para saber cuando termina la generación
        generation_finished = threading.Event()

        # 3. Función Callback del Streamer (Se ejecuta en C++)
        def streamer_callback(subword: str) -> bool:
            # Poner el fragmento de texto en la cola
            token_queue.put(subword)
            # Retornar False significa "no detenerse", seguir generando
            return False 

        # 4. Función wrapper para ejecutar en un hilo aparte
        def run_generation():
            try:
                self.pipe.generate(prompt, config, streamer_callback)
            except Exception as e:
                logger.error(f"Error en hilo de generación: {e}")
                token_queue.put(f"[ERROR: {str(e)}]")
            finally:
                generation_finished.set()

        # 5. Iniciar el hilo de generación
        t = threading.Thread(target=run_generation)
        t.start()

        # 6. Bucle de consumo (Yielding para FastAPI)
        while not generation_finished.is_set() or not token_queue.empty():
            try:
                # Esperamos un poco por un token (non-blocking effect)
                token = token_queue.get(timeout=0.05)
                yield token
            except queue.Empty:
                # Si la cola está vacía pero el hilo sigue vivo, esperamos
                continue

# Instancia global
qwen_model = QwenGenAIModel()

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚡ Iniciando API de Alto Rendimiento...")
    qwen_model.initialize()
    yield
    logger.info("🛑 Apagando...")

# --- APP ---
app = FastAPI(title="Qwen2.5 High-Performance API", lifespan=lifespan)

# --- Configuración CORS ---
# Permite que main.html haga peticiones a este servidor local sin errores de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Para desarrollo. En producción, especificar los dominios permitidos.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELOS ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7

# --- ENDPOINT OPTIMIZADO CON WEBSOCKETS ---
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint de Chat por WebSocket.
    Evita cualquier buffering HTTP del navegador y entrega los tokens en puro tiempo real.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 2048)
        temperature = data.get("temperature", 0.7)

        messages_dict = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
        
        for token in qwen_model.stream_response(
            messages_dict, 
            max_tokens=max_tokens, 
            temperature=temperature
        ):
            await websocket.send_text(token)
            # Un micro sleep asegura que FastAPI envíe el frame por red inmediatamente
            await asyncio.sleep(0.001) 

        # Enviar señal de que terminó (opcional, cerraremos la conexión desde el server o enviamos token especial)
        await websocket.send_text("[DONE]")
            
    except WebSocketDisconnect:
        logger.info("Cliente desconectado del WebSocket.")
    except Exception as e:
        logger.error(f"Error WebSocket: {e}")
        try:
            await websocket.send_text(f"[ERROR: {str(e)}]")
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    # Ajustes de Uvicorn para producción
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")