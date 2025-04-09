# vector_database.py
"""
Clase para gestionar la base de datos vectorial
"""
import os
import numpy as np
import faiss
import pickle
import json

class VectorDatabase:
    def __init__(self, config):
        self.config = config
        self.vector_dimension = config.VECTOR_DIMENSION
        self.db_path = config.VECTOR_DB_PATH
        self.index = None
        self.documents = []
        self.initialize_db()
        
    def initialize_db(self):
        """Inicializa o carga la base de datos vectorial"""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        docs_path = os.path.join(self.db_path, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            # Cargar base de datos existente
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"Base de datos vectorial cargada con {len(self.documents)} documentos")
            except Exception as e:
                print(f"Error al cargar base de datos existente: {str(e)}")
                # Si hay error al cargar, crear nueva base de datos
                self.index = faiss.IndexFlatL2(self.vector_dimension)
                self.documents = []
                print("Se creó una nueva base de datos debido a un error al cargar la existente")
        else:
            # Crear nueva base de datos
            self.index = faiss.IndexFlatL2(self.vector_dimension)
            self.documents = []
            print("Nueva base de datos vectorial creada")
            
    def add_document(self, text, embedding, metadata=None):
        """Añade un documento y su embedding a la base de datos"""
        if metadata is None:
            metadata = {}
            
        doc_id = len(self.documents)
        document = {
            "id": doc_id,
            "text": text,
            "metadata": metadata,
            "embedding": embedding  # Guardar el embedding para posible reconstrucción del índice
        }
        
        self.documents.append(document)
        
        # Convertir embedding a formato adecuado para FAISS
        embedding_np = np.array([embedding]).astype('float32')
        self.index.add(embedding_np)
        
        return doc_id
        
    def search(self, query_embedding, top_k=5):
        """Busca los documentos más similares a un embedding de consulta"""
        if len(self.documents) == 0:
            return []  # No hay documentos para buscar
            
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                result = {
                    "document": self.documents[idx],
                    "distance": float(distances[0][i])
                }
                results.append(result)
                
        return results
    
    def save(self):
        """Guarda la base de datos vectorial en disco"""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        docs_path = os.path.join(self.db_path, "documents.pkl")
        
        try:
            # Guardar el índice FAISS
            if self.index is not None:
                faiss.write_index(self.index, index_path)
            
            # Guardar los documentos
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
                
            print(f"Base de datos guardada con {len(self.documents)} documentos")
        except Exception as e:
            print(f"Error al guardar la base de datos: {str(e)}")
        
    def load_documents_from_directory(self, directory, model_manager):
        """Carga documentos desde archivos de texto en un directorio y genera embeddings"""
        loaded_count = 0
        for filename in os.listdir(directory):
            if filename.endswith(".txt") or filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if filename.endswith(".txt"):
                            content = f.read()
                            metadata = {"filename": filename}
                        else:  # JSON
                            data = json.load(f)
                            content = data.get("text", "")
                            metadata = data.get("metadata", {})
                            metadata["filename"] = filename
                            
                        # Generar embedding para el documento
                        embedding = model_manager.generate_embeddings(content)
                        
                        # Añadir a la base de datos
                        self.add_document(content, embedding, metadata)
                        loaded_count += 1
                        
                except Exception as e:
                    print(f"Error al cargar {file_path}: {str(e)}")
                    
        print(f"Cargados {loaded_count} documentos del directorio {directory}")
        return loaded_count
        
    def clear_all(self):
        """Elimina todos los documentos de la base de datos"""
        # Limpiar la lista de documentos
        self.documents = []
        
        # Reiniciar el índice FAISS
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        
        # Eliminar archivos existentes si existen
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        docs_path = os.path.join(self.db_path, "documents.pkl")
        
        try:
            if os.path.exists(index_path):
                os.remove(index_path)
                print(f"Archivo eliminado: {index_path}")
            if os.path.exists(docs_path):
                os.remove(docs_path)
                print(f"Archivo eliminado: {docs_path}")
        except Exception as e:
            print(f"Error al eliminar archivos de base de datos: {str(e)}")
            
        # Guardar los cambios (crear archivos vacíos)
        try:
            # Crear una BD vacía
            faiss.write_index(self.index, index_path)
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            print("Archivos de base de datos vacíos creados")
        except Exception as e:
            print(f"Error al guardar la base de datos vacía: {str(e)}")
            
        print("Base de datos vectorial reiniciada. Todos los documentos eliminados.")