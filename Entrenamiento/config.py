# config.py
"""
Configuración para el modelo, base de datos y servidor.
Los valores se leen del archivo .env en la raíz del proyecto.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Configuración del modelo Llama.cpp (modelo local legacy)
    MODEL_PATH = "/home/nicolasrodrigeztorres04/.lmstudio/models/TheBloke/dolphin-2.6-mistral-7B-GGUF/dolphin-2.6-mistral-7b.Q4_K_S.gguf"
    N_CTX = 4096
    N_THREADS = 6

    # Embeddings
    EMBEDDING_MODEL_PATH = "all-MiniLM-L6-v2"

    # Base de datos vectorial
    VECTOR_DB_PATH = "vector_database"
    VECTOR_DIMENSION = 384

    # Servidor Flask
    HOST  = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT  = int(os.environ.get("FLASK_PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

    # OpenAI API — generación de requerimientos
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "REEMPLAZAR_API_KEY")
    OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
    OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", 60))
    REQUIREMENT_CHAT_MODEL = os.environ.get("REQUIREMENT_CHAT_MODEL", OPENAI_MODEL)
    REQUIREMENT_CHAT_TIMEOUT = int(os.environ.get("REQUIREMENT_CHAT_TIMEOUT", OPENAI_TIMEOUT))

    # Estimación de esfuerzo — proveedor activo: "openai" | "lmstudio"
    ESTIMATION_PROVIDER      = os.environ.get("ESTIMATION_PROVIDER", "openai")
    ESTIMATION_OPENAI_MODEL  = os.environ.get("ESTIMATION_OPENAI_MODEL", "gpt-4.1-nano")
    ESTIMATION_TIMEOUT       = int(os.environ.get("ESTIMATION_TIMEOUT", 120))
    ESTIMATION_MAX_RETRIES   = int(os.environ.get("ESTIMATION_MAX_RETRIES", 2))

    # Backend UniDev para callbacks internos
    BACKEND_INTERNAL_BASE_URL = os.environ.get("BACKEND_INTERNAL_BASE_URL", "http://localhost:8081")
    INTERNAL_SERVICE_TOKEN    = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
    BACKEND_INTERNAL_TIMEOUT  = int(os.environ.get("BACKEND_INTERNAL_TIMEOUT", 30))

    # LM Studio (solo aplica si ESTIMATION_PROVIDER=lmstudio)
    ESTIMATION_LMSTUDIO_BASE_URL = os.environ.get("ESTIMATION_LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
    ESTIMATION_LMSTUDIO_MODEL    = os.environ.get("ESTIMATION_LMSTUDIO_MODEL", "local-model")
