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
import unicodedata
import re
from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt del sistema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Actúa como un arquitecto de software senior y especialista en estimación \
de esfuerzo en proyectos de desarrollo empresarial con equipos asistidos por IA.

CONTEXTO DE ENTRADA:
Recibirás un objeto JSON con:
- contexto_estimacion:
  - aiAssistedByDefault: boolean
  - activeLevels: lista de niveles activos a comparar
- ejemplos_relevantes: casos historicos parecidos recuperados por similitud semantica
- requirements: lista de requerimientos funcionales

Cada requerimiento incluye:
- title: nombre corto del requerimiento
- description: descripción detallada de la funcionalidad
- priority: "alta" | "media" | "baja"
- involvedUser: tipo de actor involucrado
- hasExternalConnection: true si depende de APIs externas, pasarelas de pago, servicios cloud, etc.
- requiresVisualScreen: true si requiere interfaz gráfica, formulario, dashboard o pantalla
- devNumber: número de desarrolladores asignados a ese requerimiento

PERFIL DEL EQUIPO:
Debes estimar escenarios por nivel. Los niveles activos vendrán en "activeLevels".

USO DE EJEMPLOS:
Si recibes "ejemplos_relevantes", úsalos como anclas de calibración para evitar inflar o subestimar horas.
No copies las horas ciegamente. Ajusta según diferencias reales de contexto, pero si el caso actual es muy parecido
a un ejemplo simple, mantén la estimación cerca de ese rango razonable.
Si un ejemplo relevante tiene alta coincidencia de tags y retrievalScore alto, trátalo como referencia prioritaria.
No respondas 12h, 16h o más para un requerimiento visual simple de mapa embebido con iframe si el ejemplo equivalente
indica que ese caso cae en un rango corto y no hay rutas, tracking, geolocalización activa, panel admin ni lógica adicional.

REGLA BASE OBLIGATORIA:
Asume que cualquier nivel usa IA para ayudar en el desarrollo. La IA acelera:
- scaffolding
- boilerplate
- documentación base
- tests iniciales
- maquetado repetitivo
- refactors simples y tareas repetitivas

La IA no reduce de forma agresiva:
- arquitectura
- integraciones complejas
- seguridad
- debugging difícil
- reglas de negocio críticas
- decisiones técnicas de alto impacto

Diferencia entre niveles:
- JUNIOR + IA: más lento, más validación y más retrabajo
- MIDDLE + IA: velocidad intermedia y menos retrabajo
- SENIOR + IA: más rápido y con mejor resolución de ambigüedad

OBJETIVO:
Analizar los requerimientos, agruparlos en módulos funcionales coherentes y estimar \
las horas totales de esfuerzo por cada requerimiento individual, considerando:
- Complejidad funcional y número de flujos de negocio
- Pantallas, endpoints o procesos involucrados
- Integraciones externas y validaciones de negocio
- Volumen de datos y riesgos técnicos
- Cantidad de desarrolladores disponibles (devNumber)
- Pruebas requeridas y retrabajo esperado por ambigüedad

Usa criterio de proyectos reales con equipos asistidos por IA en entornos empresariales.

==================================================
TAXONOMIA CANONICA DE MODULOS
==================================================

Cuando el proyecto corresponda claramente a un ecommerce/retail, debes usar esta taxonomía
canónica de módulos de forma estable y consistente entre corridas.
Un proyecto es ecommerce/retail si el brief menciona carrito, checkout, pagos, catálogo de productos o gestión de pedidos.

1. Catálogo y navegación
2. Carrito y checkout
3. Pagos y confirmaciones
4. Gestión de pedidos y estados
5. Panel administrativo
6. Integraciones externas

Reglas obligatorias para ecommerce/retail:
- Usa exactamente esos nombres de módulo.
- Asigna cada requerimiento a uno de esos módulos.
- No inventes módulos alternativos si el brief encaja en ecommerce/retail.
- Solo puedes fusionar dos módulos si uno de los dos, por sí solo, queda por debajo de 10 horas.
- Si fusionas módulos, conserva el nombre del módulo dominante y explica la fusión en "razon_complejidad".
- Mantén la taxonomía lo más estable posible entre corridas del mismo brief.

==================================================
REGLAS ESTRICTAS DE RESPUESTA
==================================================

1. Responde ÚNICAMENTE con JSON válido. Sin texto fuera del JSON, y sin dejar ningun campo vacio.
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
13. Usa "advertencias_equipo" para señalar cuellos de botella o sobrecargas del equipo.
14. Debes devolver escenarios en "estimaciones_por_nivel" para cada nivel activo recibido.
15. El bloque principal del proyecto, módulos y "total_horas_proyecto" debe representar un escenario base razonable bajo uso de IA.
16. Si hay ejemplos_relevantes de baja complejidad muy parecidos al requerimiento actual, evita inflar horas sin justificación concreta.
17. Si el requerimiento es esencialmente "mostrar un mapa embebido/iframe" sin tracking, rutas, panel admin ni lógica de negocio adicional, normalmente debe quedar cerca de 2-6 horas base.
18. Si el requerimiento es una landing o formulario simple y existe ejemplo relevante equivalente, mantén la estimación cerca del ejemplo salvo diferencia explícita.
19. CRÍTICO: El campo "title" de cada requerimiento en "modulos[].requerimientos" debe ser EXACTAMENTE igual \
al campo "title" del requerimiento de entrada correspondiente. No lo parafrasees, no lo abrevies, no lo \
reformules. Cópialo literalmente tal como aparece en "requirements".

==================================================
CRITERIOS DE COMPLEJIDAD
==================================================

- baja:     CRUD simple, pocas reglas de negocio, sin integraciones. ~6-16h por requerimiento.
- media:    Validaciones moderadas, reportes, dashboards o lógica intermedia. ~16-38h.
- alta:     Múltiples flujos, procesos transaccionales, seguridad o integraciones. ~38-82h.
- muy_alta: Alta concurrencia, múltiples terceros simultáneos, tecnología compleja. ~82h+.

Ajusta horas considerando:
- El campo "devNumber" indica cuántos desarrolladores trabajan en ese requerimiento en paralelo; \
  a mayor devNumber, las horas individuales se distribuyen pero el esfuerzo total puede bajar.
- Si "hasExternalConnection" es true: sumar horas adicionales por integración, \
  manejo de errores y pruebas de conectividad.
- Si "requiresVisualScreen" es true: sumar horas adicionales por diseño, \
  maquetado y pruebas de UI.
- La prioridad "alta" implica mayor rigor en pruebas y revisión de código.
- En todos los niveles asume apoyo de IA por defecto.
- Entre niveles, la estructura funcional debe mantenerse, pero el total de horas debe reflejar la diferencia de seniority.

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
          "title": "string — copia LITERAL del title de entrada",
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
  "advertencias_equipo": ["string"],
  "estimaciones_por_nivel": [
    {
      "codigo_nivel": "JUNIOR | MIDDLE | SENIOR | OTRO",
      "nombre_nivel": "string",
      "complejidad_general": "baja | media | alta | muy_alta",
      "total_horas_proyecto": 0,
      "nota_estimacion": "string"
    }
  ]
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

    def __init__(self, config, estimation_examples_service=None):
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
        self.max_retries = max(1, int(config.ESTIMATION_MAX_RETRIES))
        self.estimation_examples_service = estimation_examples_service

    # ------------------------------------------------------------------
    # Método público
    # ------------------------------------------------------------------

    def estimate(self, requirements: list, context=None) -> dict:
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

        examples = self._retrieve_examples(requirements)
        user_message = json.dumps(
            {
                "contexto_estimacion": self._sanitize_context(context),
                "ejemplos_relevantes": examples,
                "requirements": requirements,
            },
            ensure_ascii=False,
            indent=2,
        )

        last_error = None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for attempt in range(1, self.max_retries + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
            )

            raw_content = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else -1

            logger.info(
                "estimate-effort response | provider=%s | attempt=%d/%d | tokens_used=%d | raw_length=%d",
                self.provider,
                attempt,
                self.max_retries,
                tokens_used,
                len(raw_content),
            )

            try:
                estimation = json.loads(raw_content)

                if not isinstance(estimation, dict):
                    raise ValueError("La respuesta del modelo no es un objeto JSON.")

                return self._validate_schema(estimation, requirements)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "estimate-effort invalid response | attempt=%d/%d | error=%s",
                    attempt,
                    self.max_retries,
                    str(exc),
                )
                if attempt < self.max_retries:
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({"role": "user", "content": self._build_retry_feedback(exc, requirements)})

        raise ValueError(
            f"La respuesta de estimacion siguio incompleta despues de {self.max_retries} intentos: {last_error}"
        )

    def _retrieve_examples(self, requirements: list) -> list:
        if self.estimation_examples_service is None:
            return []
        try:
            return self.estimation_examples_service.retrieve(requirements)
        except Exception as exc:
            logger.warning("No se pudieron recuperar ejemplos relevantes para estimacion: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Validación del schema de salida
    # ------------------------------------------------------------------

    def _validate_schema(self, estimation: dict, source_requirements: list) -> dict:
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

        level_estimations = estimation.get("estimaciones_por_nivel")
        if not isinstance(level_estimations, list) or len(level_estimations) == 0:
            raise ValueError("El campo 'estimaciones_por_nivel' debe ser un array no vacío.")

        for index, level_estimation in enumerate(level_estimations):
            self._validate_level_estimation(level_estimation, index)

        # Valida cobertura y auto-corrige títulos (normalización case/whitespace)
        self._validate_and_fix_requirement_coverage(estimation, source_requirements)

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

        if not isinstance(req["razon"], str) or not req["razon"].strip():
            raise ValueError(
                f"'razon' en requerimiento {req_index} del módulo {mod_index} "
                "debe ser un string no vacío."
            )

    def _validate_level_estimation(self, level_estimation: dict, index: int) -> None:
        if not isinstance(level_estimation, dict):
            raise ValueError(f"La estimacion por nivel {index} no es un objeto.")

        for field in ("codigo_nivel", "nombre_nivel", "complejidad_general", "total_horas_proyecto", "nota_estimacion"):
            if field not in level_estimation:
                raise ValueError(f"Falta '{field}' en estimaciones_por_nivel[{index}].")

        if level_estimation["complejidad_general"] not in _VALID_COMPLEXITY:
            raise ValueError(
                f"'complejidad_general' inválida en estimaciones_por_nivel[{index}]: "
                f"'{level_estimation['complejidad_general']}'."
            )

        if not isinstance(level_estimation["total_horas_proyecto"], int) or level_estimation["total_horas_proyecto"] < 1:
            raise ValueError(f"'total_horas_proyecto' en estimaciones_por_nivel[{index}] debe ser un entero >= 1.")

        if not isinstance(level_estimation["codigo_nivel"], str) or not level_estimation["codigo_nivel"].strip():
            raise ValueError(f"'codigo_nivel' en estimaciones_por_nivel[{index}] debe ser un string no vacío.")

        if not isinstance(level_estimation["nombre_nivel"], str) or not level_estimation["nombre_nivel"].strip():
            raise ValueError(f"'nombre_nivel' en estimaciones_por_nivel[{index}] debe ser un string no vacío.")

    def _sanitize_context(self, context) -> dict:
        fallback_levels = [
            {"code": "JUNIOR", "displayName": "Junior"},
            {"code": "MIDDLE", "displayName": "Middle"},
            {"code": "SENIOR", "displayName": "Senior"},
        ]

        if not isinstance(context, dict):
            return {
                "aiAssistedByDefault": True,
                "activeLevels": fallback_levels,
            }

        active_levels = context.get("activeLevels")
        if not isinstance(active_levels, list) or len(active_levels) == 0:
            active_levels = fallback_levels

        return {
            "aiAssistedByDefault": True,
            "activeLevels": active_levels,
        }

    # ------------------------------------------------------------------
    # Normalización de títulos
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_title(title: str) -> str:
        """
        Normaliza un título para comparación tolerante:
        - Strip de espacios
        - Minúsculas
        - Elimina acentos/diacríticos
        - Colapsa espacios múltiples
        """
        s = title.strip().lower()
        # eliminar diacríticos
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        # colapsar espacios
        s = re.sub(r"\s+", " ", s)
        return s

    def _validate_and_fix_requirement_coverage(
        self, estimation: dict, source_requirements: list
    ) -> None:
        """
        Valida que cada requerimiento de entrada esté en la salida.
        Si el título coincide tras normalización (mayúsculas/acentos/espacios),
        lo corrige en el objeto en lugar de rechazar.
        Sólo falla si hay requerimientos realmente ausentes o extras.
        """
        # Construir mapa normalizado → título original
        expected: dict[str, str] = {}
        for index, requirement in enumerate(source_requirements):
            title = requirement.get("title") if isinstance(requirement, dict) else None
            if not isinstance(title, str) or not title.strip():
                raise ValueError(f"El requerimiento de entrada {index} no tiene un 'title' valido.")
            norm = self._normalize_title(title)
            expected[norm] = title.strip()   # clave normalizada → título exacto esperado

        # Recoger títulos devueltos y detectar duplicados
        returned: list[str] = []
        for module in estimation.get("modulos", []):
            for req in module.get("requerimientos", []):
                returned.append(req["title"].strip())

        returned_normalized = [self._normalize_title(t) for t in returned]
        duplicates_norm = [t for t in set(returned_normalized) if returned_normalized.count(t) > 1]
        if duplicates_norm:
            dup_originals = [returned[returned_normalized.index(n)] for n in duplicates_norm]
            raise ValueError(f"La salida repite requerimientos y eso no es valido: {sorted(dup_originals)}.")

        # Intentar mapear cada título devuelto al esperado (por normalización)
        norm_returned: dict[str, str] = {
            self._normalize_title(t): t for t in returned
        }

        missing = [
            original for norm, original in expected.items()
            if norm not in norm_returned
        ]
        extras = [
            ret for norm_ret, ret in norm_returned.items()
            if norm_ret not in expected
        ]

        if missing or extras:
            chunks = []
            if missing:
                chunks.append(f"faltan estos requerimientos: {missing}")
            if extras:
                chunks.append(f"sobran estos requerimientos: {extras}")
            raise ValueError("La salida no coincide exactamente con la entrada; " + "; ".join(chunks))

        # Auto-corregir títulos en la respuesta para que coincidan exactamente con la entrada
        for module in estimation.get("modulos", []):
            for req in module.get("requerimientos", []):
                norm_ret = self._normalize_title(req["title"])
                if norm_ret in expected:
                    req["title"] = expected[norm_ret]

    def _build_retry_feedback(self, error: Exception, requirements: list) -> str:
        expected_titles = [
            req.get("title")
            for req in requirements
            if isinstance(req, dict) and isinstance(req.get("title"), str)
        ]
        titles_json = json.dumps(expected_titles, ensure_ascii=False)
        return (
            "Tu respuesta anterior fue rechazada por el validador de esquema. "
            f"Error exacto: {str(error)}. "
            "Debes reenviar TODO el JSON completo desde cero. No dejes campos vacios ni parciales. "
            "CRÍTICO: el campo \"title\" de cada objeto en modulos[].requerimientos DEBE ser una "
            "copia LITERAL del title del requerimiento de entrada correspondiente — sin parafrasear, "
            "sin abreviar, sin cambiar mayúsculas ni tildes. "
            "Cada requerimiento debe aparecer exactamente una vez dentro de modulos[].requerimientos "
            "y cada uno debe traer obligatoriamente 'title', 'horas_estimadas' y 'razon'. "
            f"Lista EXACTA de títulos esperados (cópialos tal cual): {titles_json}."
        )
