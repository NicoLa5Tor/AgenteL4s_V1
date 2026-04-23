# app.py
"""
API Flask para desplegar el servicio - Versión final completa con correcciones
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import tempfile
import base64
import json
import threading
from urllib import request as urllib_request
from urllib import error as urllib_error
from ..rag.pdf_utils import extract_text_from_pdf, chunk_text, load_pdf_to_db
from ..rag.estimation_examples_service import EstimationExamplesService
from ..services.requirements_service import RequirementsService
from ..services.estimation_service import EstimationService
from ..services.requirement_chat_service import RequirementChatService
from openai import APITimeoutError, APIError
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

class FlaskService:
    def __init__(self, model_manager, vector_db, config):
        self.app = Flask(__name__)
        CORS(self.app, origins=["http://localhost:8081"])
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        self.vector_db = vector_db
        self.config = config
        self.requirements_service = RequirementsService(config)
        self.estimation_examples_service = EstimationExamplesService(config, model_manager)
        self.estimation_service = EstimationService(config, self.estimation_examples_service)
        self.requirement_chat_service = RequirementChatService(config)

        # Definir rutas
        self.setup_routes()
        
    def setup_routes(self):
        @self.app.route('/api/generate', methods=['POST'])
        def generate():
            """
            Endpoint para generar texto con el modelo LLM
            """
            data = request.json
            prompt = data.get('prompt', '')
            max_tokens = data.get('max_tokens', 150)
            temperature = data.get('temperature', 0.7)
            
            if not prompt:
                return jsonify({"error": "Se requiere un prompt"}), 400
                
            try:
                response = self.model_manager.generate_response(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return jsonify({"response": response})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/api/vector/add', methods=['POST'])
        def add_to_db():
            """
            Endpoint para añadir un documento a la base de datos vectorial
            """
            data = request.json
            text = data.get('text', '')
            metadata = data.get('metadata', {})
            
            if not text:
                return jsonify({"error": "Se requiere texto para añadir a la base de datos"}), 400
                
            try:
                embedding = self.model_manager.generate_embeddings(text)
                doc_id = self.vector_db.add_document(text, embedding, metadata)
                self.vector_db.save()
                return jsonify({"doc_id": doc_id, "status": "success"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/api/vector/search', methods=['POST'])
        def search():
            """
            Endpoint para buscar documentos similares en la base de datos
            """
            data = request.json
            query = data.get('query', '')
            top_k = data.get('top_k', 5)
            
            if not query:
                return jsonify({"error": "Se requiere una consulta para buscar"}), 400
                
            try:
                query_embedding = self.model_manager.generate_embeddings(query)
                results = self.vector_db.search(query_embedding, top_k)
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "id": result["document"]["id"],
                        "text": result["document"]["text"],
                        "metadata": result["document"]["metadata"],
                        "similarity": 1.0 - result["distance"] / 2.0
                    })
                    
                return jsonify({"results": formatted_results})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/pdf/upload', methods=['POST'])
        def upload_pdf():
            """
            Endpoint para cargar un PDF a la base de datos
            
            Request:
            {
                "pdf_data": "base64_encoded_pdf_data",
                "filename": "documento.pdf",
                "chunk_size": 1000,
                "chunk_overlap": 200
            }
            
            También se puede enviar un archivo PDF usando form-data con el campo "pdf_file"
            """
            try:
                # Verificar si se envió un archivo como form-data
                if 'pdf_file' in request.files:
                    pdf_file = request.files['pdf_file']
                    
                    if pdf_file.filename == '':
                        return jsonify({"error": "No se seleccionó ningún archivo"}), 400
                    
                    if not pdf_file.filename.lower().endswith('.pdf'):
                        return jsonify({"error": "El archivo debe ser un PDF"}), 400
                    
                    # Guardar archivo en ubicación temporal
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    pdf_file.save(temp_file.name)
                    pdf_path = temp_file.name
                    filename = pdf_file.filename
                    
                # Si no hay archivo, buscar datos en formato JSON con base64
                else:
                    data = request.json
                    if not data or 'pdf_data' not in data:
                        return jsonify({"error": "No se proporcionaron datos del PDF"}), 400
                    
                    pdf_data = data.get('pdf_data', '')
                    filename = data.get('filename', 'documento.pdf')
                    
                    if not pdf_data:
                        return jsonify({"error": "Se requieren los datos del PDF en base64"}), 400
                    
                    # Decodificar PDF desde base64
                    try:
                        pdf_bytes = base64.b64decode(pdf_data)
                    except Exception:
                        return jsonify({"error": "Error al decodificar los datos base64"}), 400
                    
                    # Guardar PDF en archivo temporal
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_file.write(pdf_bytes)
                    temp_file.close()
                    pdf_path = temp_file.name
                
                # Obtener parámetros adicionales
                chunk_size = request.json.get('chunk_size', 1000) if request.is_json else 1000
                chunk_overlap = request.json.get('chunk_overlap', 200) if request.is_json else 200
                
                # Procesar el PDF
                chunks_added = load_pdf_to_db(
                    pdf_path, 
                    self.model_manager, 
                    self.vector_db, 
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap
                )
                
                # Eliminar el archivo temporal
                try:
                    os.unlink(pdf_path)
                except Exception:
                    pass
                
                return jsonify({
                    "status": "success",
                    "filename": filename,
                    "chunks_added": chunks_added,
                    "message": f"PDF procesado: {chunks_added} fragmentos añadidos a la base de datos"
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/query-pdf', methods=['POST'])
        def query_pdf():
            """
            Endpoint para consultar según información de PDFs - MODO ESTRICTO
            
            En esta versión, el modelo SIEMPRE responde basándose únicamente en los datos
            disponibles en los fragmentos recuperados, sin añadir conocimiento externo.
            
            Request:
            {
                "query": "Pregunta sobre el contenido del PDF",
                "top_k": 5,
                "max_tokens": 250,
                "temperature": 0.7,
                "source_filter": "nombre_del_pdf.pdf" (opcional)
            }
            """
            data = request.json
            query = data.get('query', '')
            top_k = data.get('top_k', 5)
            max_tokens = data.get('max_tokens', 250)
            temperature = data.get('temperature', 0.7)
            source_filter = data.get('source_filter', None)
            
            if not query:
                return jsonify({"error": "Se requiere una consulta"}), 400
                
            try:
                # Buscar documentos relevantes
                query_embedding = self.model_manager.generate_embeddings(query)
                results = self.vector_db.search(query_embedding, top_k)
                
                # Filtrar por fuente si es necesario
                if source_filter:
                    results = [r for r in results if source_filter.lower() in 
                              r["document"]["metadata"].get("source", "").lower()]
                
                # Si no hay resultados, informar
                if not results:
                    return jsonify({
                        "response": "No encontré información relevante para responder a esta pregunta en los documentos proporcionados.",
                        "sources": []
                    })
                
                # Construir contexto con los documentos encontrados
                context = ""
                for i, result in enumerate(results):
                    source = result["document"]["metadata"].get("source", "Desconocido")
                    context += f"\nDocumento {i+1} (Fuente: {source}):\n{result['document']['text']}\n\n"
                
                # Instruir al modelo a responder SOLO con la información proporcionada
                system_instruction = """Eres un asistente que responde preguntas basándose ÚNICAMENTE en la información proporcionada. 
Si la información proporcionada no es suficiente para responder la pregunta, debes indicar 
"No puedo responder a esta pregunta con la información proporcionada".
NO inventes información ni utilices conocimiento externo bajo ninguna circunstancia.
Cita las fuentes específicas de donde obtienes la información en tu respuesta.
Si te preguntan algo que no está relacionado con los documentos proporcionados, 
debes responder: "Solo puedo responder preguntas relacionadas con los documentos proporcionados"."""
                
                prompt = f"""{system_instruction}

Basándote ÚNICAMENTE en la siguiente información:

{context}

Responde a esta pregunta: {query}"""
                
                # Generar respuesta
                response = self.model_manager.generate_response(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Preparar fuentes para la respuesta
                sources = []
                for r in results:
                    source_info = {
                        "id": r["document"]["id"],
                        "text_preview": r["document"]["text"][:150] + "..." if len(r["document"]["text"]) > 150 else r["document"]["text"],
                        "metadata": r["document"]["metadata"],
                        "similarity": 1.0 - r["distance"] / 2.0
                    }
                    sources.append(source_info)
                
                return jsonify({
                    "response": response,
                    "sources": sources
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/query-simple', methods=['POST'])
        def query_simple():
            """
            Endpoint simplificado para consultar información (solo devuelve la respuesta)
            Sin fuentes ni información adicional, ideal para respuestas rápidas.
            
            Request:
            {
                "query": "Pregunta sobre el contenido del PDF",
                "max_tokens": 500,
                "temperature": 0.7,
                "source_filter": "nombre_del_pdf.pdf" (opcional)
            }
            """
            data = request.json
            query = data.get('query', '')
            top_k = data.get('top_k', 5)
            max_tokens = data.get('max_tokens', 500)  # Valor predeterminado más alto
            temperature = data.get('temperature', 0.7)
            source_filter = data.get('source_filter', None)
            
            if not query:
                return jsonify({"error": "Se requiere una consulta"}), 400
                
            try:
                # Buscar documentos relevantes
                query_embedding = self.model_manager.generate_embeddings(query)
                results = self.vector_db.search(query_embedding, top_k)
                
                # Filtrar por fuente si es necesario
                if source_filter:
                    results = [r for r in results if source_filter.lower() in 
                              r["document"]["metadata"].get("source", "").lower()]
                
                # Si no hay resultados, informar
                if not results:
                    return jsonify({
                        "response": "No encontré información relevante para responder a esta pregunta en los documentos proporcionados."
                    })
                
                # Construir contexto con los documentos encontrados
                context = ""
                for i, result in enumerate(results):
                    source = result["document"]["metadata"].get("source", "Desconocido")
                    context += f"\nDocumento {i+1}:\n{result['document']['text']}\n\n"
                
                # Instruir al modelo a responder de manera completa sin exceder tokens
                system_instruction = """Eres un asistente que responde preguntas basándose ÚNICAMENTE en la información proporcionada.
Si la información proporcionada no es suficiente para responder la pregunta, debes indicar 
"No puedo responder a esta pregunta con la información proporcionada".
NO inventes información ni utilices conocimiento externo.
Adapta tu respuesta para que sea completa pero concisa. 
No menciones las fuentes específicas en tu respuesta."""
                
                prompt = f"""{system_instruction}

Basándote ÚNICAMENTE en la siguiente información:

{context}

Responde a esta pregunta: {query}"""
                
                # Generar respuesta
                response = self.model_manager.generate_response(
                    prompt, 
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Devolver solo la respuesta sin metadatos adicionales
                return jsonify({"response": response})
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/pdf/info', methods=['GET'])
        def pdf_info():
            """
            Endpoint para obtener información sobre los PDFs cargados
            """
            try:
                # Recopilar información de los PDFs
                pdf_documents = {}
                
                for doc in self.vector_db.documents:
                    metadata = doc["metadata"]
                    source = metadata.get("source", "Unknown")
                    
                    # Solo considerar documentos de tipo PDF
                    if metadata.get("type") == "pdf" or source.lower().endswith('.pdf'):
                        if source in pdf_documents:
                            pdf_documents[source]["chunks"] += 1
                        else:
                            pdf_documents[source] = {
                                "chunks": 1,
                                "example": doc["text"][:100] + "..." if len(doc["text"]) > 100 else doc["text"]
                            }
                
                return jsonify({
                    "total_pdfs": len(pdf_documents),
                    "pdfs": pdf_documents
                })
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
                
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Endpoint para verificar el estado del servicio"""
            return jsonify({
                "status": "ok",
                "model_loaded": self.model_manager.llm is not None,
                "embedding_model_loaded": self.model_manager.embedding_model is not None,
                "documents_count": len(self.vector_db.documents)
            })
            
        # ENDPOINTS PARA BORRAR DATOS - CORREGIDOS
        @self.app.route('/api/data/clear', methods=['POST'])
        def clear_data():
            """
            Endpoint para borrar todos los datos de entrenamiento
            
            Request (opcional):
            {
                "confirm": true
            }
            """
            data = request.json or {}
            confirm = data.get('confirm', False)
            
            if not confirm:
                return jsonify({
                    "status": "warning",
                    "message": "Esta acción borrará TODOS los datos de entrenamiento. Para confirmar, envía confirm=true"
                }), 400
                
            try:
                # Usar el método clear_all de la clase VectorDatabase
                self.vector_db.clear_all()
                
                # Verificación adicional para asegurar que se haya limpiado
                if len(self.vector_db.documents) > 0:
                    self.vector_db.documents = []
                    self.vector_db.index = None
                    self.vector_db.initialize_db()
                
                # Guardar la BD vacía explícitamente
                self.vector_db.save()
                
                return jsonify({
                    "status": "success",
                    "message": "Todos los datos de entrenamiento han sido eliminados"
                })
                
            except Exception as e:
                return jsonify({
                    "error": f"Error al eliminar los datos: {str(e)}"
                }), 500
        
        @self.app.route('/api/data/clear-pdf', methods=['POST'])
        def clear_pdf():
            """
            Endpoint para borrar un PDF específico
            
            Request:
            {
                "pdf_name": "nombre_del_pdf.pdf"
            }
            """
            data = request.json
            if not data or 'pdf_name' not in data:
                return jsonify({
                    "error": "Se requiere especificar pdf_name"
                }), 400
                
            pdf_name = data.get('pdf_name', '')
            
            if not pdf_name:
                return jsonify({
                    "error": "El nombre del PDF no puede estar vacío"
                }), 400
                
            try:
                # Contar documentos antes de eliminar
                total_before = len(self.vector_db.documents)
                
                # Filtrar documentos para conservar solo los que NO son del PDF especificado
                filtered_documents = []
                documents_to_remove = []
                
                for doc in self.vector_db.documents:
                    source = doc["metadata"].get("source", "")
                    if pdf_name.lower() not in source.lower():
                        filtered_documents.append(doc)
                    else:
                        documents_to_remove.append(doc)
                
                if not documents_to_remove:
                    return jsonify({
                        "status": "warning",
                        "message": f"No se encontraron documentos para el PDF: {pdf_name}"
                    })
                
                # Reconstruir la base de datos vectorial
                removed_count = len(documents_to_remove)
                self.vector_db.documents = filtered_documents
                
                # Recrear el índice con los documentos filtrados
                self.vector_db.index = None
                self.vector_db.initialize_db()
                
                # Añadir los documentos conservados al nuevo índice
                for doc in filtered_documents:
                    if "embedding" in doc:
                        embedding_np = np.array([doc["embedding"]]).astype('float32')
                        self.vector_db.index.add(embedding_np)
                
                # Guardar la base de datos actualizada
                self.vector_db.save()
                
                return jsonify({
                    "status": "success",
                    "message": f"PDF eliminado: {pdf_name}",
                    "chunks_removed": removed_count,
                    "remaining_documents": len(filtered_documents)
                })
                
            except Exception as e:
                return jsonify({
                    "error": f"Error al eliminar el PDF: {str(e)}"
                }), 500
    
        @self.app.route('/estimate-effort', methods=['POST'])
        def estimate_effort():
            """
            Endpoint para estimar el esfuerzo en horas de una lista de requerimientos.

            Recibe la salida directa de /generate-requirements.

            Request:
            [
                {
                    "title": "string",
                    "description": "string",
                    "priority": "alta" | "media" | "baja",
                    "involvedUser": "string",
                    "hasExternalConnection": boolean,
                    "requiresVisualScreen": boolean,
                    "devNumber": integer
                }
            ]

            Response: objeto JSON con proyecto, módulos, resumen de horas,
                      riesgos, supuestos y advertencias del equipo.
            """
            data = request.json
            requirements = data
            context = None

            if isinstance(data, dict):
                requirements = data.get('requirements')
                context = data.get('context')

            if not isinstance(requirements, list) or len(requirements) == 0:
                return jsonify({
                    "error": "El body debe ser un array JSON no vacío de requerimientos."
                }), 400

            # Validación mínima de cada requerimiento recibido
            required_fields = {
                "title", "description", "priority",
                "involvedUser", "hasExternalConnection",
                "requiresVisualScreen", "devNumber",
            }
            for index, req in enumerate(requirements):
                if not isinstance(req, dict):
                    return jsonify({
                        "error": f"El requerimiento en la posición {index} no es un objeto JSON."
                    }), 400
                missing = required_fields - req.keys()
                if missing:
                    return jsonify({
                        "error": f"Faltan campos en el requerimiento {index}: {sorted(missing)}"
                    }), 400

            try:
                estimation = self.estimation_service.estimate(requirements, context)
                return jsonify(estimation)
            except APITimeoutError:
                return jsonify({
                    "error": "Timeout al conectar con el proveedor de IA. Intente nuevamente."
                }), 504
            except APIError as e:
                return jsonify({"error": f"Error del proveedor de IA: {str(e)}"}), 502
            except Exception as e:
                return jsonify({"error": f"Error al procesar la estimación: {str(e)}"}), 502

        @self.app.route('/estimate-effort-async', methods=['POST'])
        def estimate_effort_async():
            data = request.json or {}
            project_id = data.get('projectId')
            requirements = data.get('requirements')
            context = data.get('context')

            if project_id is None:
                return jsonify({"error": "Se requiere projectId"}), 400
            if not isinstance(requirements, list) or len(requirements) == 0:
                return jsonify({"error": "Se requiere un array no vacio de requirements"}), 400

            worker = threading.Thread(
                target=self._process_estimation_only,
                args=(project_id, requirements, context),
                daemon=True
            )
            worker.start()

            return jsonify({
                "accepted": True,
                "projectId": project_id,
                "message": "Reestimacion del proyecto iniciada."
            }), 202

        @self.app.route('/generate-requirements', methods=['POST'])
        def generate_requirements():
            """
            Endpoint para generar requerimientos funcionales a partir de una descripción de proyecto.

            Request:
            {
                "projectId": 1,
                "project_description": "Descripción del proyecto en lenguaje natural"
            }

            Response:
            [
                {
                    "title": "string",
                    "description": "string",
                    "priority": "alta" | "media" | "baja",
                    "involvedUser": "string",
                    "hasExternalConnection": boolean,
                    "requiresVisualScreen": boolean,
                    "devNumber": integer
                }
            ]
            """
            data = request.json or {}
            project_id = data.get('projectId')
            project_description = data.get('project_description', '').strip()
            context = data.get('context')

            if project_id is None:
                return jsonify({"error": "Se requiere projectId"}), 400
            if not project_description:
                return jsonify({"error": "Se requiere project_description"}), 400

            worker = threading.Thread(
                target=self._process_project_pipeline,
                args=(project_id, project_description, context),
                daemon=True
            )
            worker.start()

            return jsonify({
                "accepted": True,
                "projectId": project_id,
                "message": "Procesamiento de requerimientos y estimacion iniciado."
            }), 202

        @self.app.route('/requirements-chat', methods=['POST'])
        def requirements_chat():
            data = request.json or {}
            project_id = data.get('projectId')
            requirement_id = data.get('requirementId')
            current_requirement = data.get('currentRequirement')
            conversation = data.get('conversation')

            if project_id is None:
                return jsonify({"error": "Se requiere projectId"}), 400
            if requirement_id is None:
                return jsonify({"error": "Se requiere requirementId"}), 400
            if not isinstance(current_requirement, dict):
                return jsonify({"error": "Se requiere currentRequirement"}), 400
            if not isinstance(conversation, list) or len(conversation) == 0:
                return jsonify({"error": "Se requiere una conversacion no vacia"}), 400

            try:
                response = self.requirement_chat_service.reply(data)
                return jsonify(response)
            except APITimeoutError:
                return jsonify({"error": "Timeout al conectar con el proveedor de IA. Intente nuevamente."}), 504
            except APIError as e:
                return jsonify({"error": f"Error del proveedor de IA: {str(e)}"}), 502
            except Exception as e:
                return jsonify({"error": f"Error al procesar el chat del requerimiento: {str(e)}"}), 502

    def _process_project_pipeline(self, project_id, project_description, context=None):
        try:
            requirements = self.requirements_service.generate(project_id, project_description)
            if not isinstance(requirements, list) or len(requirements) == 0:
                self._post_failure(project_id, "REQUIREMENTS", "La IA no devolvio requerimientos para este proyecto.")
                return

            sanitized_requirements = [self._sanitize_requirement(req) for req in requirements]
            self._post_json(
                f"{self.config.BACKEND_INTERNAL_BASE_URL.rstrip('/')}/internal/estimations/{project_id}/requirements",
                sanitized_requirements
            )
        except APITimeoutError:
            self._post_failure(project_id, "REQUIREMENTS", "Timeout al conectar con OpenAI.")
        except APIError as e:
            self._post_failure(project_id, "REQUIREMENTS", f"Error del proveedor de IA: {str(e)}")
        except Exception as e:
            self.logger.exception("Error en pipeline async de projectId=%s", project_id)
            self._post_failure(project_id, "REQUIREMENTS", f"Error al procesar requerimientos: {str(e)}")
            return

        try:
            estimation = self.estimation_service.estimate(requirements, context)
            self._post_json(
                f"{self.config.BACKEND_INTERNAL_BASE_URL.rstrip('/')}/internal/estimations/{project_id}/effort",
                estimation
            )
            self.logger.info("Pipeline completado para projectId=%s", project_id)
        except APITimeoutError:
            self._post_failure(project_id, "ESTIMATION", "Timeout al generar la estimacion.")
        except APIError as e:
            self._post_failure(project_id, "ESTIMATION", f"Error del proveedor de IA durante la estimacion: {str(e)}")
        except Exception as e:
            self.logger.exception("Error en estimacion async de projectId=%s", project_id)
            self._post_failure(project_id, "ESTIMATION", f"Error al procesar la estimacion: {str(e)}")

    def _process_estimation_only(self, project_id, requirements, context=None):
        try:
            estimation = self.estimation_service.estimate(requirements, context)
            self._post_json(
                f"{self.config.BACKEND_INTERNAL_BASE_URL.rstrip('/')}/internal/estimations/{project_id}/effort",
                estimation
            )
            self.logger.info("Reestimacion completada para projectId=%s", project_id)
        except APITimeoutError:
            self._post_failure(project_id, "ESTIMATION", "Timeout al generar la estimacion.")
        except APIError as e:
            self._post_failure(project_id, "ESTIMATION", f"Error del proveedor de IA durante la estimacion: {str(e)}")
        except Exception as e:
            self.logger.exception("Error en reestimacion async de projectId=%s", project_id)
            self._post_failure(project_id, "ESTIMATION", f"Error al procesar la reestimacion: {str(e)}")

    def _sanitize_requirement(self, requirement):
        return {
            "title": requirement.get("title"),
            "description": requirement.get("description"),
            "priority": requirement.get("priority"),
            "involvedUser": requirement.get("involvedUser"),
            "hasExternalConnection": requirement.get("hasExternalConnection"),
            "requiresVisualScreen": requirement.get("requiresVisualScreen"),
            "devNumber": requirement.get("devNumber", 1) if requirement.get("devNumber") else 1,
        }

    def _post_failure(self, project_id, stage, error):
        try:
            self._post_json(
                f"{self.config.BACKEND_INTERNAL_BASE_URL.rstrip('/')}/internal/estimations/{project_id}/failure",
                {
                    "stage": stage,
                    "error": error[:4000]
                }
            )
        except Exception:
            self.logger.exception("No se pudo reportar fallo al backend para projectId=%s", project_id)

    def _post_json(self, url, payload):
        token = (self.config.INTERNAL_SERVICE_TOKEN or "").strip()
        if not token:
            raise RuntimeError("INTERNAL_SERVICE_TOKEN no configurado en AgenteL4s.")

        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-Internal-Service-Token": token,
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=self.config.BACKEND_INTERNAL_TIMEOUT) as response:
                status = response.getcode()
                if status >= 400:
                    raise RuntimeError(f"Callback al backend fallo con status {status}")
        except urllib_error.HTTPError as exc:
            raise RuntimeError(f"Callback al backend fallo con status {exc.code}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"No se pudo conectar con el backend: {exc.reason}") from exc

    def run(self):
        """Inicia el servidor Flask"""
        self.app.run(
            host=self.config.HOST,
            port=self.config.PORT,
            debug=self.config.DEBUG
        )
