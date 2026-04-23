# main.py
"""
Script principal actualizado con soporte para PDFs
"""
import argparse
import os
from Entrenamiento.rag.pdf_utils import load_pdf_to_db

def main():
    parser = argparse.ArgumentParser(description="API de servicio LLM con base de datos vectorial")
    parser.add_argument("--serve", action="store_true", help="Iniciar el servidor API")
    parser.add_argument("--port", type=int, default=5000, help="Puerto para el servidor (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host para el servidor (default: 0.0.0.0)")
    parser.add_argument("--load_pdf", type=str, help="Ruta al archivo PDF para cargar")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Tamaño de cada fragmento (en caracteres)")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Superposición entre fragmentos")
    parser.add_argument("--no_debug", action="store_true", help="Desactivar modo debug de Flask")
    args = parser.parse_args()
    
    # Importar componentes
    from Entrenamiento.core.config import Config
    from Entrenamiento.core.model_manager import ModelManager
    from Entrenamiento.rag.vector_database import VectorDatabase
    from Entrenamiento.api.flask_service import FlaskService
    
    # Crear instancias
    config = Config()
    
    # Actualizar config con parámetros de línea de comandos
    if args.port:
        config.PORT = args.port
    if args.host:
        config.HOST = args.host
    if args.no_debug:
        config.DEBUG = False
    
    # Inicializar componentes
    model_manager = ModelManager(config)
    vector_db = VectorDatabase(config)

    # Cargar solo el modelo de embeddings (no se carga el LLM local)
    model_manager.load_embedding_model()

    # Cargar PDF si se especifica
    if args.load_pdf:
        if os.path.exists(args.load_pdf):
            print(f"Cargando PDF: {args.load_pdf}")
            load_pdf_to_db(
                args.load_pdf,
                model_manager,
                vector_db,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
        else:
            print(f"Error: El archivo PDF {args.load_pdf} no existe")
            return

    # Si se solicita iniciar el servidor
    if args.serve:
        print("Inicializando servicio API...")
        print(f"Modelo activo: {config.OPENAI_MODEL} (OpenAI API)")

        # Iniciar el servidor Flask (el LLM local no se carga)
        print(f"Iniciando servidor API en http://{config.HOST}:{config.PORT}")
        server = FlaskService(model_manager, vector_db, config)
        server.run()
        
if __name__ == "__main__":
    main()
