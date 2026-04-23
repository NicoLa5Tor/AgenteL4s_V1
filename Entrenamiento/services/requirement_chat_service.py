import json
import logging

from openai import OpenAI, APITimeoutError, APIError

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un analista funcional senior de UniDev. Tu trabajo es conversar en espanol con la empresa
sobre un requerimiento puntual y proponer una version mejorada sin salirte del contexto del proyecto.

REGLAS OBLIGATORIAS:
1. Responde UNICAMENTE con JSON valido.
2. No uses markdown, ni explicaciones fuera del JSON.
3. Tu respuesta siempre debe incluir:
{
  "reply": "respuesta conversacional corta y clara para la empresa",
  "suggestedRequirement": {
    "title": "string",
    "description": "string",
    "priority": "alta|media|baja",
    "involvedUser": "string",
    "hasExternalConnection": true,
    "requiresVisualScreen": true,
    "devNumber": 1
  }
}
4. "reply" debe explicar que entendiste, que cambiaste y cualquier supuesto importante.
5. "suggestedRequirement" siempre debe devolver una propuesta completa y consistente, aunque el usuario solo haya pedido un ajuste parcial.
6. Conserva el objetivo real del proyecto. No inventes modulos absurdos.
7. Si el usuario expresa duda o contradiccion, propone una opcion razonable y dilo en "reply".
8. Si hay ambiguedad, toma la decision mas util para producto y dejala explicita en "reply".
9. devNumber debe ser entero >= 1.
10. priority solo puede ser alta, media o baja.
11. Mantente enfocado en un solo requerimiento, pero usa el contexto del proyecto para no perder coherencia.
"""

VALID_PRIORITIES = {"alta", "media", "baja"}


class RequirementChatService:
    def __init__(self, config):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.REQUIREMENT_CHAT_TIMEOUT,
        )
        self.model = config.REQUIREMENT_CHAT_MODEL

    def reply(self, payload: dict) -> dict:
        logger.info(
            "requirements-chat request | projectId=%s | requirementId=%s | conversation_count=%d",
            payload.get("projectId"),
            payload.get("requirementId"),
            len(payload.get("conversation") or []),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0.3,
        )

        raw_content = response.choices[0].message.content.strip()
        result = json.loads(raw_content)
        self._validate(result)
        return result

    def _validate(self, result: dict) -> None:
        if not isinstance(result, dict):
            raise ValueError("La respuesta del asistente no es un objeto JSON.")

        reply = result.get("reply")
        suggestion = result.get("suggestedRequirement")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("La respuesta del asistente no incluye un campo 'reply' valido.")
        if not isinstance(suggestion, dict):
            raise ValueError("La respuesta del asistente no incluye un 'suggestedRequirement' valido.")

        required_fields = {
            "title": str,
            "description": str,
            "priority": str,
            "involvedUser": str,
            "hasExternalConnection": bool,
            "requiresVisualScreen": bool,
            "devNumber": int,
        }
        for field, expected_type in required_fields.items():
            if field not in suggestion:
                raise ValueError(f"Falta el campo '{field}' en suggestedRequirement.")
            if not isinstance(suggestion[field], expected_type):
                raise ValueError(
                    f"El campo '{field}' debe ser {expected_type.__name__} y llego {type(suggestion[field]).__name__}."
                )

        for field in ("title", "description", "involvedUser"):
            if not suggestion[field].strip():
                raise ValueError(f"El campo '{field}' no puede venir vacio en suggestedRequirement.")

        if suggestion["priority"] not in VALID_PRIORITIES:
            raise ValueError("La prioridad sugerida no es valida.")
        if suggestion["devNumber"] < 1:
            raise ValueError("devNumber debe ser mayor o igual a 1.")
