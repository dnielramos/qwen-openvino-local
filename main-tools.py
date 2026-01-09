import os
import logging
import time
import threading
import queue
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # <--- IMPORTANTE: IMPORTAR CORS
from pydantic import BaseModel, Field

# --- IMPORTACIÓN SEGURA ---
try:
    import openvino_genai as ov_genai
except ImportError:
    print("⚠️ ADVERTENCIA: openvino_genai no está instalado.")
    ov_genai = None

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- ESTRUCTURAS DE DATOS PARA TOOLS ---
class ToolFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

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
            
            # RUTA DEL MODELO CONVERTIDO
            model_path = Path(__file__).parent / "model_converted"
            
            if not model_path.exists():
                 raise FileNotFoundError(f"No se encontró la carpeta: {model_path}")
            
            if ov_genai:
                try:
                    logger.info("⚡ Intentando cargar en GPU Intel...")
                    self.pipe = ov_genai.LLMPipeline(str(model_path), "GPU")
                    logger.info("✅ ¡ÉXITO! Modelo cargado en GPU.")
                except Exception as e_gpu:
                    logger.warning(f"⚠️ Falló GPU ({e_gpu}). Usando CPU.")
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

    def _build_tool_prompt(self, messages: List[dict], tools: List[dict]) -> str:
        system_content = "You are a helpful assistant."
        if messages and messages[0]['role'] == 'system':
            system_content = messages[0]['content']
            messages = messages[1:]

        tools_json = json.dumps(tools, indent=2)
        
        augmented_system = (
            f"{system_content}\n\n"
            f"# AVAILABLE TOOLS\n"
            f"You have access to the following tools:\n"
            f"{tools_json}\n\n"
            f"# TOOL USE FORMAT\n"
            f"If you need to use a tool, you MUST respond ONLY with a JSON object in this format:\n"
            f'{{"name": "tool_name", "arguments": {{"arg_name": "value"}}}}\n\n'
            f"If you do not need a tool, respond normally in text."
        )

        prompt = f"<|im_start|>system\n{augmented_system}<|im_end|>\n"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def stream_response(self, messages: List[dict], **kwargs):
        if not self.pipe: yield "Error: Modelo no cargado."; return

        prompt = self._build_qwen_prompt(messages)
        config = self._get_config(**kwargs)
        
        token_queue = queue.Queue()
        generation_finished = threading.Event()

        def streamer_callback(subword: str) -> bool:
            token_queue.put(subword)
            return False 

        def run_generation():
            try: self.pipe.generate(prompt, config, streamer_callback)
            finally: generation_finished.set()

        t = threading.Thread(target=run_generation)
        t.start()

        while not generation_finished.is_set() or not token_queue.empty():
            try: yield token_queue.get(timeout=0.02)
            except queue.Empty: continue

    def generate_tool_response(self, messages: List[dict], tools: List[dict], **kwargs) -> dict:
        if not self.pipe: raise RuntimeError("Modelo no cargado")

        prompt = self._build_tool_prompt(messages, tools)
        config = self._get_config(**kwargs)
        
        response_text = self.pipe.generate(prompt, config)
        clean_response = response_text.strip()
        
        if clean_response.startswith("{") and "name" in clean_response:
            try:
                start = clean_response.find("{")
                end = clean_response.rfind("}") + 1
                json_str = clean_response[start:end]
                tool_call = json.loads(json_str)
                if "name" in tool_call and "arguments" in tool_call:
                    return {"type": "tool_call", "content": None, "tool_call": tool_call}
            except json.JSONDecodeError:
                logger.warning("JSON inválido.")
        
        return {"type": "message", "content": clean_response, "tool_call": None}

    def _get_config(self, **kwargs):
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = kwargs.get("max_tokens", 2048)
        config.temperature = kwargs.get("temperature", 0.6)
        config.do_sample = kwargs.get("do_sample", True)
        config.repetition_penalty = 1.1
        return config

# Instancia global
qwen_model = QwenGenAIModel()

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    qwen_model.initialize()
    yield

app = FastAPI(title="Qwen 2.5 API + Tools", lifespan=lifespan)

# --- CORRECCIÓN CORS (LO QUE TE FALTABA) ---
# Esto permite que tu HTML (localhost) hable con tu API (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite conexiones desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permite POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)
# -------------------------------------------

# --- MODELOS DE ENTRADA ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 2048
    temperature: float = 0.7

class ToolChatRequest(BaseModel):
    messages: List[Message]
    tools: List[Tool]
    max_tokens: int = 1024
    temperature: float = 0.1

# --- ENDPOINTS ---
@app.post("/chat")
async def chat_stream(request: ChatRequest):
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    return StreamingResponse(
        (token for token in qwen_model.stream_response(messages_dict, max_tokens=request.max_tokens, temperature=request.temperature)), 
        media_type="text/plain"
    )

@app.post("/tool_chat")
async def tool_chat_endpoint(request: ToolChatRequest):
    messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
    tools_dict = [t.model_dump() for t in request.tools]
    try:
        result = qwen_model.generate_tool_response(messages_dict, tools_dict, max_tokens=request.max_tokens, temperature=request.temperature)
        return result
    except Exception as e:
        logger.error(f"Error en tool_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Importante: host 0.0.0.0 permite acceso desde red local
    uvicorn.run(app, host="0.0.0.0", port=8000)