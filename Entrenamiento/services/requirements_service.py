# requirements_service.py
"""
Servicio para generar requerimientos funcionales a partir de descripciones de proyectos
utilizando la API de OpenAI (gpt-4.1-nano).
"""
import json
import logging
from openai import OpenAI, APITimeoutError, APIError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un analizador de requerimientos de software. Tu tarea es recibir la descripción \
de un proyecto y dividirla en requerimientos funcionales independientes.

REGLAS OBLIGATORIAS:
1. La salida debe ser ÚNICAMENTE un JSON válido, sin texto extra, markdown ni comentarios.
2. Siempre retorna un array JSON, incluso si hay un solo requerimiento.
3. Cada requerimiento representa una funcionalidad independiente.
4. "title": corto y descriptivo.
5. "description": explica claramente la funcionalidad.
6. "priority": solo puede ser "alta", "media" o "baja".
7. "involvedUser": tipo de actor involucrado.
8. "hasExternalConnection": true si depende de APIs externas, correo, base de datos externa, \
pasarelas de pago, servicios cloud o terceros.
9. "requiresVisualScreen": true si requiere interfaz gráfica, formulario, dashboard, modal o pantalla.
10. "devNumber": número entero estimado de desarrolladores para implementar el requerimiento.
11. Si hay múltiples módulos, separar en múltiples objetos dentro del array.
12. Si falta información, inferir el valor más razonable según el contexto.
13. Formato exacto de cada objeto del array:
{
  "title": "string",
  "description": "string",
  "priority": "alta" | "media" | "baja",
  "involvedUser": "string",
  "hasExternalConnection": boolean,
  "requiresVisualScreen": boolean,
  "devNumber": integer
}"""

REQUIRED_FIELDS = {
    "title": str,
    "description": str,
    "priority": str,
    "involvedUser": str,
    "hasExternalConnection": bool,
    "requiresVisualScreen": bool,
    "devNumber": int,
}

VALID_PRIORITIES = {"alta", "media", "baja"}


class RequirementsService:
    def __init__(self, config):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.OPENAI_TIMEOUT,
        )
        self.model = config.OPENAI_MODEL

    def generate(self, project_id: int, project_description: str) -> list:
        """
        Genera requerimientos funcionales para un proyecto dado.

        Args:
            project_id: Identificador del proyecto.
            project_description: Descripción del proyecto en lenguaje natural.

        Returns:
            Lista de requerimientos funcionales como dicts.

        Raises:
            APITimeoutError: Si la llamada a OpenAI supera el timeout.
            APIError: Si la API de OpenAI retorna un error.
            json.JSONDecodeError: Si la respuesta no es JSON válido.
            ValueError: Si el JSON no cumple el esquema esperado.
        """
        logger.info(
            "generate-requirements request | projectId=%s | description_length=%d",
            project_id,
            len(project_description),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": project_description},
            ],
            temperature=0.3,
        )

        raw_content = response.choices[0].message.content.strip()

        logger.info(
            "generate-requirements response | projectId=%s | tokens_used=%d | raw_length=%d",
            project_id,
            response.usage.total_tokens if response.usage else -1,
            len(raw_content),
        )

        requirements = json.loads(raw_content)

        if not isinstance(requirements, list):
            raise ValueError("La respuesta del modelo no es un array JSON.")

        return self._validate_schema(requirements)

    def _validate_schema(self, requirements: list) -> list:
        """
        Valida que cada requerimiento tenga los campos obligatorios con tipos correctos.

        Raises:
            ValueError: Si algún requerimiento no cumple el esquema.
        """
        for index, req in enumerate(requirements):
            if not isinstance(req, dict):
                raise ValueError(
                    f"El requerimiento en la posición {index} no es un objeto JSON."
                )

            for field, expected_type in REQUIRED_FIELDS.items():
                if field not in req:
                    raise ValueError(
                        f"Campo '{field}' faltante en el requerimiento {index}."
                    )
                if not isinstance(req[field], expected_type):
                    raise ValueError(
                        f"Campo '{field}' en el requerimiento {index} debe ser de tipo "
                        f"{expected_type.__name__}, se recibió {type(req[field]).__name__}."
                    )

            if req["priority"] not in VALID_PRIORITIES:
                raise ValueError(
                    f"Campo 'priority' en el requerimiento {index} tiene valor inválido "
                    f"'{req['priority']}'. Valores permitidos: {VALID_PRIORITIES}."
                )

            if req["devNumber"] < 1:
                raise ValueError(
                    f"Campo 'devNumber' en el requerimiento {index} debe ser >= 1."
                )

        return requirements
