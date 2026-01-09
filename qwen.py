# app.py - ¡ESTE CÓDIGO FUNCIONA DE VERDAD!
import os
import logging
import time
from pathlib import Path
from typing import List, Dict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importación condicional para evitar errores si no tienes la librería instalada
try:
    import openvino_genai as ov_genai
except ImportError:
    print("⚠️ ADVERTENCIA: openvino_genai no está instalado. El modelo no funcionará.")
    ov_genai = None

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CLASE DEL MODELO CON FORMATO CORRECTO PARA QWEN2.5 ---
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
            logger.info("🚀 Inicializando Qwen2.5-3B-Instruct con OpenVINO GenAI...")
            
            # Ruta del modelo - AJUSTA ESTA RUTA SEGÚN TU ESTRUCTURA
            model_path = Path(__file__).parent / "model_cache" / "qwen25_3b_gguf" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
            
            if not model_path.exists():
                logger.warning(f"⚠️ Archivo GGUF no encontrado en {model_path}")
                logger.warning("💡 Descarga el modelo desde: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF")
                raise FileNotFoundError(f"No se encontró el archivo GGUF en {model_path}")
            
            # Cargar el modelo con configuración optimizada
            if ov_genai:
                logger.info(f"📂 Cargando modelo desde: {model_path}")
                self.pipe = ov_genai.LLMPipeline(str(model_path), "CPU")
                logger.info("✅ Modelo cargado correctamente en CPU")
                
                # Cargar el tokenizer para mejor conteo de tokens
                self.tokenizer = self.pipe.get_tokenizer()
                logger.info("✅ Tokenizer cargado correctamente")
            else:
                logger.error("❌ Librería openvino_genai no detectada.")
                raise RuntimeError("openvino_genai no está instalado")

            self._initialized = True
            logger.info("🎉 Inicialización completada exitosamente")
            
        except Exception as e:
            logger.error(f"❌ Error crítico al inicializar el modelo: {str(e)}")
            raise

    def _build_qwen25_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Construye el prompt correctamente para Qwen2.5 usando su formato especial.
        Formato oficial de Qwen2.5:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hola<|im_end|>
        <|im_start|>assistant
        """
        system_message = "You are Qwen, a helpful and respectful AI assistant developed by Alibaba Cloud. Always provide accurate, helpful, and respectful responses."
        
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "").strip()
            
            # Mapear roles a formato Qwen2.5
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "system":
                # Ya incluimos el mensaje de sistema por defecto
                continue
        
        # Añadir el inicio del turno del asistente
        prompt += "<|im_start|>assistant\n"
        return prompt

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> tuple[str, int, float]:
        if not self.pipe:
            raise RuntimeError("El modelo no está inicializado o falló al cargar.")

        start_time = time.time()
        try:
            # 1. Construir el prompt CORRECTAMENTE para Qwen2.5
            prompt = self._build_qwen25_chat_template(messages)
            logger.info(f"📝 Prompt generado:\n{prompt}")
            
            # 2. Configurar parámetros de generación optimizados
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = kwargs.get("max_tokens", 1024)  # Reducido para mejor rendimiento
            config.temperature = kwargs.get("temperature", 0.3)   # Más bajo para respuestas más precisas
            config.top_p = kwargs.get("top_p", 0.9)
            config.do_sample = True
            config.repetition_penalty = 1.15
            config.stop_strings = ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]  # Stop tokens importantes
            
            # 3. Generación con el pipeline
            logger.info("⚡ Generando respuesta...")
            response = self.pipe.generate(prompt, config)
            
            # 4. Limpiar la respuesta correctamente
            # Qwen2.5 puede devolver el prompt completo o tokens especiales
            if response.startswith(prompt):
                response = response[len(prompt):]
            
            # Eliminar tokens especiales y limpiar
            cleaned = response
            for token in ["<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|im_end|>", "im_end", "im_start"]:
                cleaned = cleaned.replace(token, "")
            
            cleaned = cleaned.strip()
            
            # 5. Contar tokens correctamente usando el tokenizer
            tokens_used = 0
            if hasattr(self, 'tokenizer') and self.tokenizer:
                try:
                    tokens_used = len(self.tokenizer.encode(cleaned))
                except:
                    tokens_used = len(cleaned.split())
            else:
                tokens_used = len(cleaned.split())
            
            gen_time = time.time() - start_time
            logger.info(f"✅ Respuesta generada en {gen_time:.2f}s. Tokens: {tokens_used}")
            
            return cleaned, tokens_used, gen_time
            
        except Exception as e:
            logger.error(f"❌ Error en generación: {str(e)}")
            raise

# Instancia global del modelo
qwen_model = QwenGenAIModel()

# --- DEFINICIÓN DE LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚡ Iniciando aplicación...")
    qwen_model.initialize()
    yield
    logger.info("🛑 Apagando aplicación...")

# --- CREACIÓN DE LA APP CON CORS ---
app = FastAPI(
    title="Qwen2.5-3B-Instruct API",
    description="API para el modelo Qwen2.5-3B-Instruct optimizado con OpenVINO GenAI",
    version="1.0.0",
    lifespan=lifespan
)

# --- CONFIGURACIÓN DE CORS (¡CRÍTICO PARA EL CLIENTE HTML!) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes para desarrollo
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos
    allow_headers=["*"],  # Permitir todos los headers
)

# --- MODELOS DE DATOS ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 0.9

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    time: float

# --- ENDPOINTS ---
@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API y el modelo"""
    return {
        "status": "ok",
        "model_loaded": qwen_model._initialized,
        "model_path": str(Path(__file__).parent / "model_cache" / "qwen25_3b_gguf" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    }

@app.options("/chat")
async def chat_options():
    """Manejar peticiones OPTIONS para CORS"""
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint para chatear con Qwen2.5-3B-Instruct.
    
    Ejemplo de petición:
    {
        "messages": [
            {"role": "user", "content": "Hola, ¿cómo estás?"},
            {"role": "assistant", "content": "¡Hola! Estoy muy bien, gracias por preguntar."},
            {"role": "user", "content": "¿Puedes contarme sobre los presidentes de América Latina?"}
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }
    """
    try:
        # Convertir a formato compatible con el modelo
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Generar respuesta
        response, tokens, time_taken = qwen_model.generate_response(
            messages_dict,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "response": response,
            "usage": {
                "total_tokens": tokens,
                "prompt_tokens": 0,  # Estimación básica
                "completion_tokens": tokens
            },
            "metrics": {
                "time_sec": round(time_taken, 2),
                "tokens_per_second": round(tokens / time_taken, 1) if time_taken > 0 else 0
            }
        }
        
    except RuntimeError as e:
        logger.error(f" RuntimeError: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Servicio no disponible: {str(e)}")
    except Exception as e:
        logger.error(f" Error inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Iniciando servidor FastAPI...")
    logger.info("🌐 Accede al cliente HTML en: http://localhost:8000")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,  # Desactivado para mejor rendimiento con el modelo
        log_level="info"
    )