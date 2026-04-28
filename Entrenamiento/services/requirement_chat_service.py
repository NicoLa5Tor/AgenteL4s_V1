import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un analista funcional senior de UniDev. Ayudas a la empresa a refinar un requerimiento puntual \
de su proyecto de software. Conversas en español, eres conciso y orientado a producto.

REGLAS OBLIGATORIAS:
1. Responde UNICAMENTE con JSON valido. Sin texto fuera del JSON.
2. Tu respuesta siempre debe tener exactamente esta estructura:
{
  "reply": "respuesta conversacional clara para la empresa (1-3 párrafos máximo)",
  "suggestedRequirement": {
    "title": "string",
    "description": "string",
    "priority": "alta|media|baja",
    "involvedUser": "string",
    "hasExternalConnection": true|false,
    "requiresVisualScreen": true|false,
    "devNumber": 1
  }
}
3. "reply": explica qué entendiste, qué cambiaste y cualquier supuesto relevante. Sé directo.
4. "suggestedRequirement": siempre devuelve una propuesta completa y consistente, \
aunque el usuario solo haya pedido un ajuste parcial.
5. Conserva el objetivo real del proyecto. No inventes módulos absurdos.
6. Si el usuario expresa duda o contradicción, propone una opción razonable y explícala en "reply".
7. Si hay ambigüedad, toma la decisión más útil para el producto y déjala explícita en "reply".
8. devNumber debe ser entero >= 1.
9. priority solo puede ser: alta, media o baja.
10. Mantente enfocado en UN solo requerimiento, usando el contexto del proyecto para no perder coherencia.
11. No hagas preguntas innecesarias — si puedes inferir razonablemente, infiere y documéntalo.
"""

VALID_PRIORITIES = {"alta", "media", "baja"}


class RequirementChatService:
    def __init__(self, config):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=config.REQUIREMENT_CHAT_TIMEOUT,
        )
        self.model = config.REQUIREMENT_CHAT_MODEL
        self.max_retries = 2

    def reply(self, payload: dict) -> dict:
        project_id = payload.get("projectId")
        requirement_id = payload.get("requirementId")
        current_requirement = payload.get("currentRequirement", {})
        conversation = payload.get("conversation", [])
        # Back manda projectContext como objeto con múltiples campos
        project_context = payload.get("projectContext") or payload.get("projectDescription") or {}

        logger.info(
            "requirements-chat request | projectId=%s | requirementId=%s | conversation_count=%d",
            project_id,
            requirement_id,
            len(conversation),
        )

        # Contexto fijo para el sistema sobre el requerimiento actual y el proyecto
        context_block = {
            "projectId": project_id,
            "requirementId": requirement_id,
            "projectContext": project_context,
            "currentRequirement": current_requirement,
        }
        context_message = (
            "CONTEXTO DEL REQUERIMIENTO A REFINAR:\n"
            + json.dumps(context_block, ensure_ascii=False, indent=2)
        )

        # Construir historial de chat con roles user/assistant correctos
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": context_message},
        ]

        for turn in conversation:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")
            if not content:
                continue
            if role == "assistant":
                # Turnos previos del asistente: el content puede ser JSON o texto plano
                if isinstance(content, dict):
                    messages.append({"role": "assistant", "content": json.dumps(content, ensure_ascii=False)})
                else:
                    messages.append({"role": "assistant", "content": str(content)})
            else:
                # Turno del usuario
                messages.append({"role": "user", "content": str(content)})

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                )

                raw_content = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens if response.usage else -1

                logger.info(
                    "requirements-chat response | projectId=%s | attempt=%d/%d | tokens_used=%d",
                    project_id,
                    attempt,
                    self.max_retries,
                    tokens_used,
                )

                result = json.loads(raw_content)
                self._validate(result)
                return result

            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "requirements-chat invalid response | attempt=%d/%d | error=%s",
                    attempt,
                    self.max_retries,
                    str(exc),
                )
                if attempt < self.max_retries:
                    # Añadir feedback de reintento al hilo
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tu respuesta fue rechazada por el validador. Error: {str(exc)}. "
                            "Devuelve ÚNICAMENTE JSON válido con los campos 'reply' y 'suggestedRequirement' completos."
                        )
                    })

        raise ValueError(
            f"El chat de requerimiento no pudo generar una respuesta valida tras {self.max_retries} intentos: {last_error}"
        )

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
