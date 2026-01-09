import os
import logging
import time
import threading
import queue
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

try:
    import openvino_genai as ov_genai
except ImportError:
    print("⚠️ ADVERTENCIA: openvino_genai no está instalado.")
    ov_genai = None

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
            logger.info("🚀 Inicializando motor nativo OpenVINO (Qwen 1.5B INT4)...")
            
            # AHORA APUNTAMOS A LA CARPETA DONDE HICISTE EL EXPORT
            # Asegúrate que esta carpeta existe junto a app.py
            model_path = Path(__file__).parent / "model_converted"
            
            if not model_path.exists():
                 raise FileNotFoundError(f"No se encontró la carpeta del modelo en: {model_path}. \nEjecuta: optimum-cli export openvino --model Qwen/Qwen2.5-1.5B-Instruct --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 1.0 model_converted")
            
            if ov_genai:
                # INTENTO GPU
                try:
                    logger.info("⚡ Cargando en GPU Intel (Nativo)...")
                    # Al pasar una carpeta, OpenVINO busca openvino_model.xml automáticamente
                    self.pipe = ov_genai.LLMPipeline(str(model_path), "GPU")
                    logger.info("✅ ¡ÉXITO! Modelo NATIVO cargado en GPU. Máxima velocidad posible.")
                except Exception as e_gpu:
                    logger.warning(f"⚠️ GPU falló ({e_gpu}). Usando CPU.")
                    self.pipe = ov_genai.LLMPipeline(str(model_path), "CPU")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error crítico: {str(e)}")

    def _build_qwen_prompt(self, messages: List[dict]) -> str:
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def stream_response(self, messages: List[dict], **kwargs):
        if not self.pipe:
            yield "Error: Modelo no cargado."
            return

        prompt = self._build_qwen_prompt(messages)
        
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = kwargs.get("max_tokens", 2048)
        config.temperature = kwargs.get("temperature", 0.7)
        config.do_sample = kwargs.get("do_sample", True)
        config.repetition_penalty = 1.1

        token_queue = queue.Queue()
        generation_finished = threading.Event()

        def streamer_callback(subword: str) -> bool:
            token_queue.put(subword)
            return False 

        def run_generation():
            try:
                self.pipe.generate(prompt, config, streamer_callback)
            except Exception as e:
                logger.error(f"Gen Error: {e}")
            finally:
                generation_finished.set()

        t = threading.Thread(target=run_generation)
        t.start()

        while not generation_finished.is_set() or not token_queue.empty():
            try:
                yield token_queue.get(timeout=0.02)
            except queue.Empty:
                continue

qwen_model = QwenGenAIModel()

@asynccontextmanager
async def lifespan(app: FastAPI):
    qwen_model.initialize()
    yield

app = FastAPI(lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7

@app.post("/chat")
async def chat_stream(request: ChatRequest):
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    return StreamingResponse(
        (token for token in qwen_model.stream_response(messages_dict, max_tokens=request.max_tokens, temperature=request.temperature)), 
        media_type="text/plain"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)