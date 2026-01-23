import os
import logging
import json
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# --- 1. CONFIGURACIÓN Y LOGS ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(AGENT)s] %(message)s")
logger = logging.getLogger("SystemCore")

# --- 2. IMPORTACIÓN SEGURA DEL MOTOR ---
try:
    import openvino_genai as ov_genai
except ImportError:
    logger.warning("⚠️ OpenVINO GenAI no detectado. El modelo no cargará.")
    ov_genai = None

# ==========================================
#  CAPA DE NEGOCIO: HERRAMIENTAS REALES
# ==========================================

# Base de datos simulada de usuarios (In-Memory)
MOCK_USERS_DB = [
    {"id": 1, "name": "Carlos Gomez", "role": "Frontend Dev", "email": "carlos@tech.com"},
    {"id": 2, "name": "Ana Torres", "role": "UX Designer", "email": "ana@design.ui"},
    {"id": 3, "name": "Luis Rodriguez", "role": "Backend Eng", "email": "luis@api.io"},
    {"id": 4, "name": "Maria Silva", "role": "Product Owner", "email": "maria@agile.co"},
]

def tool_search_users(query: str) -> Dict[str, Any]:
    """Busca usuarios en la base de datos por nombre o rol."""
    logger.info(f"🔎 Ejecutando búsqueda de usuarios: '{query}'")
    query_lower = query.lower()
    results = [
        u for u in MOCK_USERS_DB 
        if query_lower in u["name"].lower() or query_lower in u["role"].lower()
    ]
    return {
        "status": "success", 
        "count": len(results), 
        "data": results if results else "No se encontraron usuarios."
    }

def tool_get_weather(city: str) -> Dict[str, Any]:
    """
    Obtiene el clima actual. 
    NOTA: Para producción, reemplazar la simulación con una llamada a 'httpx.get' 
    hacia OpenWeatherMap o WeatherAPI.
    """
    logger.info(f"☁️ Consultando clima para: '{city}'")
    
    # Simulación de latencia de red real
    # time.sleep(0.5) 
    
    # Datos realistas para demostración inmediata
    conditions = ["Soleado", "Parcialmente nublado", "Lluvia ligera", "Tormenta"]
    temp_base = 20 if "bogota" not in city.lower() else 14
    
    return {
        "city": city,
        "temperature": f"{temp_base + random.randint(-2, 5)}°C",
        "condition": random.choice(conditions),
        "humidity": f"{random.randint(40, 80)}%",
        "wind": f"{random.randint(5, 15)} km/h",
        "timestamp": datetime.now().isoformat()
    }

# --- REGISTRO DE HERRAMIENTAS (TOOL REGISTRY) ---
# Mapea el nombre que el LLM ve -> La función Python real
FUNCTION_MAP: Dict[str, Callable] = {
    "search_users": tool_search_users,
    "get_weather": tool_get_weather
}

# Definiciones JSON para que el LLM entienda qué puede hacer
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_users",
            "description": "Buscar usuarios en la base de datos corporativa por nombre o rol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "El nombre o rol a buscar (ej. 'Carlos', 'Frontend')."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Obtener información del clima actual y condiciones meteorológicas de una ciudad.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nombre de la ciudad (ej. 'Bogotá', 'Madrid')."}
                },
                "required": ["city"]
            }
        }
    }
]

# ==========================================
#  CAPA CORE: MOTOR DE INFERENCIA
# ==========================================

class QwenEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.pipe = None
            cls._instance.executor = ThreadPoolExecutor(max_workers=1)
        return cls._instance

    def initialize(self):
        if self.pipe: return
        try:
            model_path = Path(__file__).parent / "model_converted"
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
            
            logger.info("🚀 Cargando modelo Qwen 2.5 (INT4)...")
            # Intento de carga GPU -> Fallback CPU
            device = "CPU"
            try:
                self.pipe = ov_genai.LLMPipeline(str(model_path), "GPU")
                device = "GPU"
            except:
                self.pipe = ov_genai.LLMPipeline(str(model_path), "CPU")
            
            logger.info(f"✅ Motor activo en: {device}")
            
            # Warm-up rápido
            self.pipe.generate("Hi", ov_genai.GenerationConfig(max_new_tokens=1))
            
        except Exception as e:
            logger.error(f"❌ Error crítico iniciando motor: {e}")

    def _build_prompt(self, messages: List[dict]) -> str:
        """Construye el prompt con instrucciones estrictas de JSON."""
        system_msg = (
            "You are a helpful assistant capable of using tools.\n"
            "## AVAILABLE TOOLS\n"
            f"{json.dumps(TOOLS_SCHEMA, indent=2)}\n\n"
            "## INSTRUCTIONS\n"
            "If the user asks for something that requires a tool (like finding a user or checking weather):\n"
            "1. DO NOT rely on your internal knowledge.\n"
            "2. Respond ONLY with a valid JSON object.\n"
            "3. Format: {\"name\": \"function_name\", \"arguments\": {\"arg\": \"value\"}}\n\n"
            "If no tool is needed, respond normally in text."
        )
        
        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        for msg in messages:
            prompt += f"<|im_start|>{msg.get('role')}\n{msg.get('content')}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    async def generate_and_execute(self, messages: List[dict]) -> dict:
        """Ciclo completo: Piensa -> Decide Herramienta -> Ejecuta -> Devuelve Resultado"""
        
        # 1. Configuración de Inferencia
        prompt = self._build_prompt(messages)
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 512
        config.temperature = 0.1 # Muy bajo para precisión en JSON
        config.do_sample = False
        
        # 2. Inferencia en Hilo Separado (No bloqueante)
        loop = asyncio.get_running_loop()
        raw_response = await loop.run_in_executor(
            self.executor, lambda: self.pipe.generate(prompt, config)
        )
        
        clean_response = raw_response.strip()
        logger.info(f"🤖 Respuesta cruda del modelo: {clean_response[:100]}...")

        # 3. Detección y Ejecución de Herramienta
        tool_data = self._extract_json(clean_response)
        
        if tool_data:
            func_name = tool_data.get("name")
            args = tool_data.get("arguments", {})
            
            if func_name in FUNCTION_MAP:
                try:
                    # EJECUCIÓN DINÁMICA DE LA FUNCIÓN PYTHON
                    func = FUNCTION_MAP[func_name]
                    result = func(**args) # Llamada real
                    
                    return {
                        "type": "tool_result",
                        "tool_used": func_name,
                        "tool_input": args,
                        "content": result # Resultado JSON real de la función
                    }
                except Exception as e:
                    return {"type": "error", "content": f"Error ejecutando herramienta: {str(e)}"}
            else:
                 return {"type": "error", "content": f"Herramienta '{func_name}' no encontrada."}
        
        # 4. Respuesta Normal (Chat)
        return {
            "type": "message",
            "content": clean_response
        }

    def _extract_json(self, text: str) -> Optional[dict]:
        """Intenta extraer un bloque JSON válido del texto."""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                json_str = text[start : end + 1]
                return json.loads(json_str)
        except:
            pass
        return None

# ==========================================
#  CAPA DE API (FASTAPI)
# ==========================================

engine = QwenEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.initialize()
    yield
    engine.executor.shutdown(wait=False)

app = FastAPI(title="Qwen Agent API", lifespan=lifespan)

# --- CONFIGURACIÓN CORS (Crucial para Frontend) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # ⚠️ En prod, cambiar "*" por ["http://tu-dominio.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.post("/agent/chat")
async def agent_chat_endpoint(request: ChatRequest):
    """
    Endpoint Inteligente:
    - Recibe historial de chat.
    - El modelo decide si buscar usuarios, ver clima o charlar.
    - EJECUTA la herramienta en el servidor y devuelve el dato real.
    """
    try:
        response = await engine.generate_and_execute(request.messages)
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error en endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Iniciar servidor con alto rendimiento
    uvicorn.run(app, host="0.0.0.0", port=8000)