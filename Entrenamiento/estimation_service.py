# estimation_service.py
"""
Servicio de estimación de esfuerzo en horas para requerimientos funcionales.

Soporta dos proveedores de modelo de lenguaje:
  - "openai"   → GPT-4.1-nano via OpenAI API
  - "lmstudio" → modelo local via LM Studio (API compatible con OpenAI)

Para cambiar de proveedor basta con modificar ESTIMATION_PROVIDER en .env
o establecer la variable de entorno ESTIMATION_PROVIDER=lmstudio.
"""
import json
import logging
from openai import OpenAI, APITimeoutError, APIError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Actúa como un arquitecto de software senior y especialista en estimación \
de esfuerzo en proyectos de desarrollo empresarial.

CONTEXTO DE ENTRADA:
Recibirás una lista de requerimientos funcionales en formato JSON. Cada requerimiento incluye:
- title: nombre corto del requerimiento
- description: descripción detallada de la funcionalidad
- priority: "alta" | "media" | "baja"
- involvedUser: tipo de actor involucrado
- hasExternalConnection: true si depende de APIs externas, pasarelas de pago, servicios cloud, etc.
- requiresVisualScreen: true si requiere interfaz gráfica, formulario, dashboard o pantalla
- devNumber: número de desarrolladores junior asignados a ese requerimiento

PERFIL DEL EQUIPO:
El equipo está compuesto únicamente por desarrolladores junior. \
Ajusta las horas de cada requerimiento considerando:
- Mayor tiempo de análisis y diseño previo
- Mayor tiempo de implementación respecto a un perfil senior
- Mayor necesidad de pruebas y correcciones
- Curva de aprendizaje en integraciones externas o tecnologías nuevas

OBJETIVO:
Analizar los requerimientos, agruparlos en módulos funcionales coherentes y estimar \
las horas totales de esfuerzo por cada requerimiento individual, considerando:
- Complejidad funcional y número de flujos de negocio
- Pantallas, endpoints o procesos involucrados
- Integraciones externas y validaciones de negocio
- Volumen de datos y riesgos técnicos
- Cantidad de desarrolladores junior disponibles (devNumber)
- Pruebas requeridas y retrabajo esperado por ambigüedad

Usa criterio de proyectos reales con equipos junior en entornos empresariales.

==================================================
REGLAS ESTRICTAS DE RESPUESTA
==================================================

1. Responde ÚNICAMENTE con JSON válido. Sin texto fuera del JSON.
2. No uses markdown, no uses bloques de código, no uses comentarios.
3. No agregues campos fuera del schema definido.
4. Todos los valores de horas deben ser enteros positivos (>= 1).
5. Si un requerimiento es ambiguo: infiere una solución razonable, documenta la inferencia \
en "supuestos" y ajusta ligeramente al alza las horas estimadas.
6. La complejidad solo puede ser: "baja", "media", "alta" o "muy_alta".
7. Agrupa los requerimientos en módulos funcionales coherentes.
8. Cada módulo debe contener al menos un requerimiento en "requerimientos".
9. "total_horas_modulo" debe ser la suma exacta de "horas_estimadas" de sus requerimientos.
10. "total_horas_proyecto" debe ser la suma exacta de todos los "total_horas_modulo".
11. Si existen dependencias entre módulos, aumentar la complejidad del módulo dependiente.
12. Si detectas incertidumbre técnica o riesgos, agregarlos en "riesgos_detectados".
13. Usa "advertencias_equipo" para señalar cuellos de botella o sobrecargas del equipo junior.

==================================================
CRITERIOS DE COMPLEJIDAD
==================================================

- baja:     CRUD simple, pocas reglas de negocio, sin integraciones. ~8-20h por requerimiento.
- media:    Validaciones moderadas, reportes, dashboards o lógica intermedia. ~20-50h.
- alta:     Múltiples flujos, procesos transaccionales, seguridad o integraciones. ~50-100h.
- muy_alta: Alta concurrencia, múltiples terceros simultáneos, tecnología compleja. ~100h+.

Ajusta horas considerando:
- El campo "devNumber" indica cuántos juniors trabajan en ese requerimiento en paralelo; \
  a mayor devNumber, las horas individuales se distribuyen pero el esfuerzo total puede bajar.
- Si "hasExternalConnection" es true: sumar horas adicionales por integración, \
  manejo de errores y pruebas de conectividad.
- Si "requiresVisualScreen" es true: sumar horas adicionales por diseño, \
  maquetado y pruebas de UI.
- La prioridad "alta" implica mayor rigor en pruebas y revisión de código.

==================================================
SCHEMA DE SALIDA — OBLIGATORIO
==================================================

{
  "proyecto": {
    "nombre": "string",
    "tipo": "web | mobile | api | hibrido | otro",
    "complejidad_general": "baja | media | alta | muy_alta",
    "resumen": "string"
  },
  "modulos": [
    {
      "id": 1,
      "nombre": "string",
      "complejidad": "baja | media | alta | muy_alta",
      "razon_complejidad": "string",
      "requerimientos": [
        {
          "title": "string",
          "horas_estimadas": 0,
          "razon": "string"
        }
      ],
      "total_horas_modulo": 0,
      "requiere_integracion_externa": true,
      "integraciones": ["string"]
    }
  ],
  "total_horas_proyecto": 0,
  "riesgos_detectados": ["string"],
  "supuestos": ["string"],
  "advertencias_equipo": ["string"]
}"""

# ---------------------------------------------------------------------------
# Constantes de validación
# ---------------------------------------------------------------------------

_VALID_COMPLEXITY = {"baja", "media", "alta", "muy_alta"}


# ---------------------------------------------------------------------------
# Servicio
# ---------------------------------------------------------------------------

class EstimationService:
    """
    Genera estimaciones de esfuerzo en horas por requerimiento a partir de
    una lista de requerimientos funcionales producidos por RequirementsService.

    El proveedor activo se controla con ESTIMATION_PROVIDER en .env:
      - "openai"   → usa ESTIMATION_OPENAI_MODEL vía OpenAI API
      - "lmstudio" → usa ESTIMATION_LMSTUDIO_MODEL vía LM Studio local
    """

    def __init__(self, config):
        provider = config.ESTIMATION_PROVIDER.lower()

        if provider == "lmstudio":
            self.client = OpenAI(
                base_url=config.ESTIMATION_LMSTUDIO_BASE_URL,
                api_key="lm-studio",           # LM Studio no valida la key
                timeout=config.ESTIMATION_TIMEOUT,
            )
            self.model = config.ESTIMATION_LMSTUDIO_MODEL
            logger.info("EstimationService inicializado con LM Studio | model=%s", self.model)
        else:
            self.client = OpenAI(
                api_key=config.OPENAI_API_KEY,
                timeout=config.ESTIMATION_TIMEOUT,
            )
            self.model = config.ESTIMATION_OPENAI_MODEL
            logger.info("EstimationService inicializado con OpenAI | model=%s", self.model)

        self.provider = provider

    # ------------------------------------------------------------------
    # Método público
    # ------------------------------------------------------------------

    def estimate(self, requirements: list) -> dict:
        """
        Estima el esfuerzo en horas por requerimiento para una lista de entradas.

        Args:
            requirements: Lista de dicts con el formato de /generate-requirements.

        Returns:
            Dict con la estimación según el schema definido en el prompt.

        Raises:
            APITimeoutError: Si la llamada supera el timeout configurado.
            APIError: Si el proveedor retorna un error HTTP.
            json.JSONDecodeError: Si la respuesta no es JSON válido.
            ValueError: Si el JSON no cumple el schema esperado.
        """
        logger.info(
            "estimate-effort request | provider=%s | model=%s | requirements_count=%d",
            self.provider,
            self.model,
            len(requirements),
        )

        user_message = json.dumps(requirements, ensure_ascii=False, indent=2)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,   # baja temperatura → estimaciones más consistentes
        )

        raw_content = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens if response.usage else -1

        logger.info(
            "estimate-effort response | provider=%s | tokens_used=%d | raw_length=%d",
            self.provider,
            tokens_used,
            len(raw_content),
        )

        estimation = json.loads(raw_content)

        if not isinstance(estimation, dict):
            raise ValueError("La respuesta del modelo no es un objeto JSON.")

        return self._validate_schema(estimation)

    # ------------------------------------------------------------------
    # Validación del schema de salida
    # ------------------------------------------------------------------

    def _validate_schema(self, estimation: dict) -> dict:
        """
        Valida la estructura del JSON devuelto por el modelo.

        Raises:
            ValueError: Si falta algún campo obligatorio o los tipos no coinciden.
        """
        # ── proyecto ──────────────────────────────────────────────────
        proyecto = estimation.get("proyecto")
        if not isinstance(proyecto, dict):
            raise ValueError("Falta el campo 'proyecto' o no es un objeto.")

        for field in ("nombre", "tipo", "complejidad_general", "resumen"):
            if field not in proyecto:
                raise ValueError(f"Falta 'proyecto.{field}'.")

        if proyecto["complejidad_general"] not in _VALID_COMPLEXITY:
            raise ValueError(
                f"'proyecto.complejidad_general' inválido: '{proyecto['complejidad_general']}'."
            )

        # ── modulos ───────────────────────────────────────────────────
        modulos = estimation.get("modulos")
        if not isinstance(modulos, list) or len(modulos) == 0:
            raise ValueError("El campo 'modulos' debe ser un array no vacío.")

        for i, mod in enumerate(modulos):
            self._validate_module(mod, i)

        # ── total_horas_proyecto ──────────────────────────────────────
        total = estimation.get("total_horas_proyecto")
        if not isinstance(total, int) or total < 0:
            raise ValueError("'total_horas_proyecto' debe ser un entero >= 0.")

        # ── listas opcionales ─────────────────────────────────────────
        for field in ("riesgos_detectados", "supuestos", "advertencias_equipo"):
            value = estimation.get(field, [])
            if not isinstance(value, list):
                raise ValueError(f"El campo '{field}' debe ser un array.")
            estimation[field] = value   # garantizar que exista aunque esté vacío

        return estimation

    def _validate_module(self, mod: dict, index: int) -> None:
        """Valida un módulo individual dentro de 'modulos'."""
        if not isinstance(mod, dict):
            raise ValueError(f"El módulo en la posición {index} no es un objeto.")

        for field in ("id", "nombre", "complejidad", "razon_complejidad",
                      "requerimientos", "total_horas_modulo",
                      "requiere_integracion_externa", "integraciones"):
            if field not in mod:
                raise ValueError(f"Falta el campo '{field}' en el módulo {index}.")

        if mod["complejidad"] not in _VALID_COMPLEXITY:
            raise ValueError(
                f"'complejidad' inválida en módulo {index}: '{mod['complejidad']}'."
            )

        reqs = mod["requerimientos"]
        if not isinstance(reqs, list) or len(reqs) == 0:
            raise ValueError(
                f"'requerimientos' en módulo {index} debe ser un array no vacío."
            )

        for j, req in enumerate(reqs):
            self._validate_requirement_hours(req, index, j)

        if not isinstance(mod["total_horas_modulo"], int) or mod["total_horas_modulo"] < 0:
            raise ValueError(
                f"'total_horas_modulo' en módulo {index} debe ser un entero >= 0."
            )

        if not isinstance(mod["requiere_integracion_externa"], bool):
            raise ValueError(
                f"'requiere_integracion_externa' en módulo {index} debe ser boolean."
            )

        if not isinstance(mod["integraciones"], list):
            raise ValueError(f"'integraciones' en módulo {index} debe ser un array.")

    def _validate_requirement_hours(self, req: dict, mod_index: int, req_index: int) -> None:
        """Valida un objeto de requerimiento dentro de 'modulos[].requerimientos'."""
        if not isinstance(req, dict):
            raise ValueError(
                f"El requerimiento {req_index} del módulo {mod_index} no es un objeto."
            )

        for field in ("title", "horas_estimadas", "razon"):
            if field not in req:
                raise ValueError(
                    f"Falta '{field}' en el requerimiento {req_index} del módulo {mod_index}."
                )

        if not isinstance(req["horas_estimadas"], int) or req["horas_estimadas"] < 1:
            raise ValueError(
                f"'horas_estimadas' en requerimiento {req_index} del módulo {mod_index} "
                "debe ser un entero >= 1."
            )

        if not isinstance(req["title"], str) or not req["title"].strip():
            raise ValueError(
                f"'title' en requerimiento {req_index} del módulo {mod_index} "
                "debe ser un string no vacío."
            )
