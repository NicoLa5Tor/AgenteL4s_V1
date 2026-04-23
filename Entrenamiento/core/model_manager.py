# model_manager.py
"""
Clase para gestionar el modelo de lenguaje
"""
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np

class ModelManager:
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.embedding_model = None
        
    def load_model(self):
        """Carga el modelo LLM usando llama-cpp-python"""
        print(f"Cargando modelo desde {self.config.MODEL_PATH}...")
        self.llm = Llama(
            model_path=self.config.MODEL_PATH,
            n_ctx=self.config.N_CTX,
            n_threads=self.config.N_THREADS
        )
        print("Modelo LLM cargado exitosamente")
        return self.llm
    
    def load_embedding_model(self):
        """Carga el modelo de embeddings usando sentence-transformers"""
        print(f"Cargando modelo de embeddings {self.config.EMBEDDING_MODEL_PATH}...")
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL_PATH)
        print("Modelo de embeddings cargado exitosamente")
        return self.embedding_model
        
    def generate_response(self, prompt, max_tokens=150, temperature=0.7):
        """Genera una respuesta usando el modelo LLM"""
        if self.llm is None:
            self.load_model()
            
        # Formateamos el prompt según el formato del modelo
        formatted_prompt = f"[INST] {prompt} [/INST]"
        
        # Generamos la respuesta
        response = self.llm(
            formatted_prompt, 
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extraemos el texto generado de la respuesta
        generated_text = response["choices"][0]["text"].strip()
        return generated_text
    
    def generate_embeddings(self, text):
        """Genera embeddings para un texto dado usando el modelo de embeddings"""
        if self.embedding_model is None:
            self.load_embedding_model()
            
        # Generamos el embedding
        embedding = self.embedding_model.encode(text)
        
        # Convertimos a numpy array para compatibilidad con FAISS
        return np.array(embedding)