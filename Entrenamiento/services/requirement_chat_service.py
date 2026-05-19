import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_CHAT = """Eres un analista funcional senior de UniDev. Ayudas a la empresa a refinar un requerimiento \
puntual de su proyecto de software. Conversas en español, eres conciso y orientado a producto.

REGLAS OBLIGATORIAS:
1. Responde UNICAMENTE con JSON valido. Sin texto fuera del JSON.
2. Tu respuesta debe tener exactamente esta estructura:
{
  "reply": "respuesta conversacional clara para la empresa (1-3 párrafos máximo)"
}
3. NO incluyas "suggestedRequirement" ni ningun otro campo. Solo "reply".
4. Conversa naturalmente: haz preguntas si necesitas aclarar, propón alternativas, explora el requerimiento.
5. Conserva el objetivo real del proyecto. No inventes módulos absurdos.
6. Mantente enfocado en UN solo requerimiento, usando el contexto del proyecto para no perder coherencia.
"""

SYSTEM_PROMPT_PROPOSAL = """Eres un analista funcional senior de UniDev. La empresa pidió explícitamente \
una propuesta estructurada para refinar su requerimiento, basada en la conversación anterior.

REGLAS OBLIGATORIAS:
1. Responde UNICAMENTE con JSON valido. Sin texto fuera del JSON.
2. Tu respuesta siempre debe tener exactamente esta estructura:
{
  "reply": "explica brevemente qué cambiaste y por qué (1-2 párrafos)",
  "suggestedRequirement": {
    "title": "string",
    "description": "string",
    "priority": "alta|media|baja",
    "involvedUser": "string",
    "hasExternalConnection": true|false,
    "requiresVisualScreen": true|false,
    "devNumber": 1,
    "estimatedHours": 1
  }
}
3. "reply": sintetiza los acuerdos de la conversación y explica los cambios propuestos.
4. "suggestedRequirement": propuesta completa y consistente basada en lo conversado.
5. Conserva el objetivo real del proyecto. No inventes módulos absurdos.
6. devNumber debe ser entero >= 1.
7. priority solo puede ser: alta, media o baja.
8. Mantente enfocado en UN solo requerimiento, usando el contexto del proyecto para no perder coherencia.
9. "estimatedHours": estima las horas de desarrollo para implementar ESTE requerimiento según la propuesta refinada.
   - Usa como referencia "estimatedHours" del currentRequirement si existe (es la estimación previa del pipeline).
   - Ajusta al alza si la propuesta agrega complejidad/pantallas/integraciones respecto al original.
   - Ajusta a la baja si la propuesta simplifica el alcance.
   - Debe ser entero >= 1. Rango típico: baja=4-16h, media=16-40h, alta=40-80h.
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
        request_proposal = bool(payload.get("requestProposal", False))
        current_requirement = payload.get("currentRequirement", {})
        conversation = payload.get("conversation", [])
        project_context = payload.get("projectContext") or payload.get("projectDescription") or {}

        logger.info(
            "requirements-chat request | projectId=%s | requirementId=%s | conversation_count=%d | requestProposal=%s",
            project_id,
            requirement_id,
            len(conversation),
            request_proposal,
        )

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

        system_prompt = SYSTEM_PROMPT_PROPOSAL if request_proposal else SYSTEM_PROMPT_CHAT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
        ]

        for turn in conversation:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")
            if not content:
                continue
            if role == "assistant":
                if isinstance(content, dict):
                    messages.append({"role": "assistant", "content": json.dumps(content, ensure_ascii=False)})
                else:
                    messages.append({"role": "assistant", "content": str(content)})
            else:
                messages.append({"role": "user", "content": str(content)})

        retry_hint = (
            "Devuelve ÚNICAMENTE JSON válido con los campos 'reply' y 'suggestedRequirement' completos."
            if request_proposal
            else "Devuelve ÚNICAMENTE JSON válido con el campo 'reply'."
        )

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
                self._validate(result, request_proposal)
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
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Tu respuesta fue rechazada por el validador. Error: {str(exc)}. "
                            + retry_hint
                        )
                    })

        raise ValueError(
            f"El chat de requerimiento no pudo generar una respuesta valida tras {self.max_retries} intentos: {last_error}"
        )

    def _validate(self, result: dict, request_proposal: bool) -> None:
        if not isinstance(result, dict):
            raise ValueError("La respuesta del asistente no es un objeto JSON.")

        reply = result.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("La respuesta del asistente no incluye un campo 'reply' valido.")

        if not request_proposal:
            return

        suggestion = result.get("suggestedRequirement")
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
            "estimatedHours": int,
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
        if suggestion["estimatedHours"] < 1:
            raise ValueError("estimatedHours debe ser mayor o igual a 1.")
