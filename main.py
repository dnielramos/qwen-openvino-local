import os
import logging
import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Importación condicional para evitar errores si no tienes la librería instalada al probar el código
try:
    import openvino_genai as ov_genai
except ImportError:
    # Esto es solo para que el código no falle si te falta la librería al copiarlo
    print("⚠️ ADVERTENCIA: openvino_genai no está instalado.") 
    ov_genai = None

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CLASE DEL MODELO ---
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
            logger.info("🚀 Inicializando Qwen2.5-3B-Instruct con OpenVINO GenAI + GGUF...")
            
            # Ajusta la ruta si es necesario
            model_path = Path(__file__).parent / "model_cache" / "qwen25_3b_gguf" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
            
            if not model_path.exists():
                raise FileNotFoundError(f"No se encontró el archivo GGUF en {model_path}")
            
            # Carga del modelo
            if ov_genai:
                self.pipe = ov_genai.LLMPipeline(str(model_path), "CPU")
                logger.info("✅ Modelo cargado correctamente en CPU")
            else:
                logger.error("❌ Librería openvino_genai no detectada.")

            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error crítico al inicializar el modelo: {str(e)}")

    def _build_qwen_prompt(self, messages: List[dict]) -> str:
        """
        Construye el prompt manualmente en formato ChatML para Qwen.
        Formato:
        <|im_start|>system
        You are...<|im_end|>
        <|im_start|>user
        Hola<|im_end|>
        <|im_start|>assistant
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Añadir el inicio del turno del asistente para que el modelo complete
        prompt += "<|im_start|>assistant\n"
        return prompt

    def generate_response(self, messages: List[dict], **kwargs):
        if not self.pipe:
            raise RuntimeError("El modelo no está inicializado o falló al cargar.")

        start_time = time.time()
        try:
            # 1. Construcción MANUAL del prompt (Solución al error)
            prompt = self._build_qwen_prompt(messages)
            
            # Logs para depuración (opcional, para ver qué entra al modelo)
            # logger.info(f"Prompt enviado al modelo:\n{prompt}")

            config = ov_genai.GenerationConfig()
            config.max_new_tokens = kwargs.get("max_tokens", 2048)
            config.temperature = kwargs.get("temperature", 0.7)
            config.do_sample = kwargs.get("do_sample", True)
            config.repetition_penalty = kwargs.get("repetition_penalty", 1.1)
            
            # 2. Generación usando el string plano
            response = self.pipe.generate(prompt, config)
            
            # 3. Limpieza (Qwen a veces devuelve el prompt o tokens especiales al final)
            # Quitamos el prompt original si la librería lo incluye en el output (depende de la versión)
            if response.startswith(prompt):
                response = response[len(prompt):]

            cleaned = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            
            tokens_used = len(cleaned.split()) 
            gen_time = time.time() - start_time
            
            return cleaned, tokens_used, gen_time
            
        except Exception as e:
            logger.error(f"❌ Error en generación: {str(e)}")
            raise e

# Instancia global del modelo
qwen_model = QwenGenAIModel()

# --- DEFINICIÓN DE LIFESPAN (REEMPLAZA A ON_EVENT) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lógica de ARRANQUE (Startup)
    logger.info("⚡ Ejecutando lógica de lifespan (arranque)...")
    qwen_model.initialize()
    yield
    # Lógica de APAGADO (Shutdown) - Aquí puedes liberar memoria si fuera necesario
    logger.info("🛑 Apagando aplicación...")

# --- CREACIÓN DE LA APP ---
app = FastAPI(
    title="Qwen2.5 API", 
    lifespan=lifespan  # <--- AQUÍ ES DONDE SE CONECTA AHORA
)

# --- MODELOS DE DATOS (PYDANTIC) ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7

# --- ENDPOINTS ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para chatear con el modelo.
    """
    try:
        # Convertimos los modelos Pydantic a lista de dicts para OpenVINO
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        response, tokens, time_taken = qwen_model.generate_response(
            messages_dict, 
            max_tokens=request.max_tokens, 
            temperature=request.temperature
        )
        
        return {
            "response": response, 
            "usage": {"tokens_approx": tokens}, 
            "metrics": {"time_sec": time_taken}
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Asegúrate de ejecutar esto en consola para ver los logs
    uvicorn.run(app, host="0.0.0.0", port=8000)