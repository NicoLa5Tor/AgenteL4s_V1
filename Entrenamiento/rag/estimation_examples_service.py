import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EstimationExamplesService:
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.enabled = bool(getattr(config, "ESTIMATION_EXAMPLES_ENABLED", True))
        self.top_k = max(1, int(getattr(config, "ESTIMATION_EXAMPLES_TOP_K", 4)))
        self.examples_path = Path(getattr(config, "ESTIMATION_EXAMPLES_PATH", "estimation_examples.json"))
        self.examples: list[dict[str, Any]] = []
        self.embeddings: list[np.ndarray] = []
        self._load_examples()

    def retrieve(self, requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.enabled or not self.examples or not requirements:
            return []

        scored: dict[int, dict[str, float]] = {}
        last_query_tags: set[str] = set()
        for requirement in requirements:
            query_vector = self._embed_requirement(requirement)
            query_tags = self._infer_tags(requirement)
            last_query_tags = query_tags
            for index, example_vector in enumerate(self.embeddings):
                similarity = self._cosine_similarity(query_vector, example_vector)
                example_tags = set(self.examples[index].get("tags", []))
                tag_overlap = self._tag_overlap_score(query_tags, example_tags)
                magnitude_bonus = self._magnitude_bonus(query_tags, example_tags)
                intent_bonus = self._intent_bonus(query_tags, example_tags)
                penalty = self._mismatch_penalty(query_tags, example_tags)
                total_score = (similarity * 0.45) + (tag_overlap * 0.3) + magnitude_bonus + intent_bonus - penalty
                current = scored.get(index)
                if current is None or total_score > current["total"]:
                    scored[index] = {
                        "semantic": similarity,
                        "tags": tag_overlap,
                        "magnitude_bonus": magnitude_bonus,
                        "intent_bonus": intent_bonus,
                        "penalty": penalty,
                        "total": total_score,
                    }

        ordered_indexes = sorted(scored.keys(), key=lambda index: scored[index]["total"], reverse=True)[: self.top_k]
        selected = []
        for index in ordered_indexes:
            example = dict(self.examples[index])
            example["similarity"] = round(scored[index]["semantic"], 4)
            example["retrievalScore"] = round(scored[index]["total"], 4)
            example["matchedTags"] = sorted(set(example.get("tags", [])) & last_query_tags)
            selected.append(example)

        logger.info(
            "estimation-examples retrieval | requested=%d | returned=%d | selected=%s",
            len(requirements),
            len(selected),
            [
                {
                    "id": example.get("id"),
                    "score": example.get("retrievalScore"),
                    "tags": example.get("matchedTags"),
                }
                for example in selected
            ],
        )
        return selected

    def _load_examples(self) -> None:
        if not self.enabled:
            logger.info("EstimationExamplesService deshabilitado por configuracion")
            return

        if not self.examples_path.exists():
            logger.warning("No existe catalogo de ejemplos de estimacion: %s", self.examples_path)
            return

        try:
            payload = json.loads(self.examples_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.exception("No se pudo leer el catalogo de ejemplos de estimacion: %s", exc)
            return

        manifest_examples = self._resolve_examples_payload(payload)
        if not manifest_examples:
            logger.warning("El catalogo de ejemplos de estimacion no contiene entradas validas")
            return

        loaded_examples = []
        loaded_embeddings = []
        for index, example in enumerate(manifest_examples):
            if not isinstance(example, dict):
                continue
            requirement = example.get("requirement")
            estimation = example.get("estimation")
            if not isinstance(requirement, dict) or not isinstance(estimation, dict):
                logger.warning("Ejemplo de estimacion invalido en posicion %d", index)
                continue
            try:
                embedding = self._embed_requirement(requirement)
                loaded_examples.append(example)
                loaded_embeddings.append(embedding)
            except Exception as exc:
                logger.warning("No se pudo indexar ejemplo de estimacion %d: %s", index, exc)

        self.examples = loaded_examples
        self.embeddings = loaded_embeddings
        logger.info("EstimationExamplesService cargado con %d ejemplos", len(self.examples))

    def _resolve_examples_payload(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [example for example in payload if isinstance(example, dict)]

        if not isinstance(payload, dict):
            return []

        manifest_entries = payload.get("examples")
        if not isinstance(manifest_entries, list):
            return []

        resolved_examples = []
        base_dir = self.examples_path.parent
        for index, entry in enumerate(manifest_entries):
            if not isinstance(entry, dict):
                continue

            relative_path = entry.get("path")
            if not isinstance(relative_path, str) or not relative_path.strip():
                logger.warning("Entrada del manifest sin path valido en posicion %d", index)
                continue

            case_path = (base_dir / relative_path).resolve()
            try:
                raw_case = json.loads(case_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("No se pudo leer caso de estimacion %s: %s", case_path, exc)
                continue

            if not isinstance(raw_case, dict):
                logger.warning("Caso de estimacion invalido en %s", case_path)
                continue

            tags = raw_case.get("tags")
            if not isinstance(tags, list):
                tags = entry.get("tags") if isinstance(entry.get("tags"), list) else []
            raw_case["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
            raw_case.setdefault("id", entry.get("id"))
            raw_case.setdefault("source", str(case_path.relative_to(base_dir)))
            resolved_examples.append(raw_case)

        return resolved_examples

    def _embed_requirement(self, requirement: dict[str, Any]) -> np.ndarray:
        text = self._requirement_to_text(requirement)
        embedding = self.model_manager.generate_embeddings(text)
        vector = np.array(embedding, dtype=np.float32)
        if vector.ndim != 1:
            vector = vector.flatten()
        return vector

    def _requirement_to_text(self, requirement: dict[str, Any]) -> str:
        return (
            f"titulo: {requirement.get('title', '')}\n"
            f"descripcion: {requirement.get('description', '')}\n"
            f"prioridad: {requirement.get('priority', '')}\n"
            f"usuario: {requirement.get('involvedUser', '')}\n"
            f"integracion_externa: {requirement.get('hasExternalConnection', False)}\n"
            f"pantalla_visual: {requirement.get('requiresVisualScreen', False)}\n"
            f"devs: {requirement.get('devNumber', 1)}"
        )

    def _cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return float(np.dot(left, right) / (left_norm * right_norm))

    def _tag_overlap_score(self, query_tags: set[str], example_tags: set[str]) -> float:
        if not query_tags or not example_tags:
            return 0.0
        intersection = len(query_tags & example_tags)
        denominator = max(1, len(query_tags))
        return intersection / denominator

    def _magnitude_bonus(self, query_tags: set[str], example_tags: set[str]) -> float:
        magnitude_tags = {"low", "medium", "high", "very_high", "small", "large", "enterprise"}
        shared = (query_tags & example_tags) & magnitude_tags
        if "very_high" in shared:
            return 0.15
        if "high" in shared or "enterprise" in shared:
            return 0.1
        if "medium" in shared or "large" in shared:
            return 0.06
        if "low" in shared or "small" in shared:
            return 0.04
        return 0.0

    def _intent_bonus(self, query_tags: set[str], example_tags: set[str]) -> float:
        exact_pairs = [
            {"map_iframe"},
            {"landing_page"},
            {"contact_form"},
            {"admin_panel"},
            {"dashboard"},
            {"reports"},
            {"catalog"},
            {"booking"},
            {"authentication"},
            {"roles"},
            {"payments"},
            {"ecommerce"},
            {"marketplace"},
            {"file_upload"},
            {"documents"},
            {"cms"},
            {"lead_management"},
            {"mobile_app"},
            {"logistics"},
            {"erp"},
            {"multi_tenant"},
        ]
        bonus = 0.0
        for pair in exact_pairs:
            if pair <= query_tags and pair <= example_tags:
                bonus += 0.08
        if {"map_iframe", "small", "low"} <= query_tags and {"map_iframe", "small", "low"} <= example_tags:
            bonus += 0.12
        return bonus

    def _mismatch_penalty(self, query_tags: set[str], example_tags: set[str]) -> float:
        penalty = 0.0
        if "map_iframe" in query_tags and "map_iframe" not in example_tags and "geolocation" in example_tags:
            penalty += 0.15
        if "landing_page" in query_tags and "marketplace" in example_tags:
            penalty += 0.12
        if "small" in query_tags and {"enterprise", "very_high"} & example_tags:
            penalty += 0.12
        if "low" in query_tags and {"high", "very_high"} & example_tags:
            penalty += 0.1
        if "mobile_app" not in query_tags and "mobile_app" in example_tags:
            penalty += 0.08
        return penalty

    def _infer_tags(self, requirement: dict[str, Any]) -> set[str]:
        text = " ".join([
            str(requirement.get("title", "")),
            str(requirement.get("description", "")),
            str(requirement.get("involvedUser", "")),
            str(requirement.get("priority", "")),
        ]).lower()
        normalized_text = re.sub(r"[^a-z0-9áéíóúñü\s_-]", " ", text)

        tags: set[str] = set()
        if requirement.get("requiresVisualScreen"):
            tags.add("web")
        if requirement.get("hasExternalConnection"):
            tags.add("integration")

        dev_number = int(requirement.get("devNumber", 1) or 1)
        if dev_number >= 3:
            tags.update({"enterprise", "very_high"})
        elif dev_number == 2:
            tags.update({"large", "high"})
        else:
            tags.update({"small", "low"})

        keyword_map = {
            "landing_page": ["landing", "home", "brochure"],
            "corporate_site": ["sitio corporativo", "pagina corporativa", "institucional"],
            "contact_form": ["contacto", "formulario", "lead"],
            "lead_management": ["lead", "prospecto", "cotizacion", "solicitud", "cliente potencial"],
            "map_iframe": ["iframe", "mapa embebido"],
            "geolocation": ["geolocalizacion", "ubicacion", "mapa", "tracking", "ruta"],
            "admin_panel": ["admin", "panel", "backoffice", "dashboard", "gestion"],
            "cms": ["cms", "contenido", "blog", "noticia", "pagina editable", "editor"],
            "dashboard": ["dashboard", "metricas", "indicadores"],
            "reports": ["reporte", "reportes", "analitica"],
            "catalog": ["catalogo", "productos", "servicios"],
            "filters": ["filtro", "filtros", "busqueda"],
            "booking": ["reserva", "agenda", "cita", "booking", "calendario"],
            "calendar": ["calendario", "agenda"],
            "notifications": ["notificacion", "notificaciones", "alerta", "email", "correo"],
            "authentication": ["login", "autenticacion", "sesion", "acceso"],
            "roles": ["roles", "permisos", "autorizacion"],
            "file_upload": ["archivo", "adjunto", "subir", "cargar documento", "upload"],
            "documents": ["documento", "pdf", "contrato", "certificado"],
            "payments": ["pago", "pagos", "checkout", "billing", "suscripcion"],
            "ecommerce": ["tienda", "ecommerce", "carrito", "pedido"],
            "marketplace": ["marketplace", "vendedor", "comprador"],
            "chat": ["chat", "mensajeria", "mensaje"],
            "mobile_app": ["app movil", "movil", "mobile", "android", "ios"],
            "logistics": ["entrega", "repartidor", "logistica", "despacho", "ruta"],
            "inventory": ["inventario", "stock", "almacen"],
            "erp": ["erp", "operacion interna"],
            "education": ["curso", "estudiante", "docente", "evaluacion", "clase"],
            "video": ["video", "streaming", "reproduccion"],
            "multi_tenant": ["multi tenant", "multitenant", "tenant"],
            "automation": ["automatizacion", "workflow", "regla automatica"],
            "saas": ["saas", "suscripcion recurrente"],
        }

        for tag, keywords in keyword_map.items():
            if any(keyword in normalized_text for keyword in keywords):
                tags.add(tag)

        if "muy_alta" in normalized_text or "enterprise" in tags or {"multi_tenant", "real_time"} & tags:
            tags.update({"enterprise", "very_high"})
        elif {"payments", "marketplace", "mobile_app", "logistics", "erp"} & tags:
            tags.update({"large", "high"})
        elif {"dashboard", "admin_panel", "catalog", "booking", "notifications", "authentication", "cms", "file_upload", "documents", "lead_management"} & tags:
            tags.update({"medium"})
            tags.discard("low")

        return tags
