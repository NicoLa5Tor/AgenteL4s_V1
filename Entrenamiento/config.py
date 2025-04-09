# config.py
"""
Configuración para el modelo, base de datos y servidor
"""

class Config:
    # Configuración del modelo Llama.cpp
    MODEL_PATH = "/home/nicolasrodrigeztorres04/.lmstudio/models/TheBloke/dolphin-2.6-mistral-7B-GGUF/dolphin-2.6-mistral-7b.Q4_K_S.gguf"
    N_CTX = 4096
    N_THREADS = 6
    
    # Configuración de embeddings (modelo separado para embeddings)
    EMBEDDING_MODEL_PATH = "all-MiniLM-L6-v2"  # Modelo de embeddings de Sentence Transformers 
    
    # Configuración de la base de datos vectorial
    VECTOR_DB_PATH = "vector_database"
    VECTOR_DIMENSION = 384  # Dimensión para el modelo all-MiniLM-L6-v2
    
    # Configuración del servidor Flask
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = True