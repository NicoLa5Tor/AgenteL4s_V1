# Instalacion local de AgenteL4s_V1

## Que es

Servicio Flask para RAG sobre PDFs y flujos de IA de UniDev:

- generacion de requerimientos
- estimacion de esfuerzo
- chat de refinamiento de requerimientos

Corre por defecto en `http://localhost:5000`.

## Requisitos

- Python 3.10+
- pip
- virtualenv recomendado

## Archivo `.env`

El proyecto ya lee variables desde `AgenteL4s_V1/.env`. Una base razonable para local es:

```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=true

OPENAI_API_KEY=tu_api_key
OPENAI_MODEL=gpt-4.1-nano
OPENAI_TIMEOUT=60

REQUIREMENT_CHAT_MODEL=gpt-4.1-nano
REQUIREMENT_CHAT_TIMEOUT=60

ESTIMATION_PROVIDER=openai
ESTIMATION_OPENAI_MODEL=gpt-4.1-nano
ESTIMATION_TIMEOUT=120
ESTIMATION_MAX_RETRIES=2
ESTIMATION_EXAMPLES_ENABLED=true
ESTIMATION_EXAMPLES_TOP_K=4

BACKEND_INTERNAL_BASE_URL=http://localhost:8081
INTERNAL_SERVICE_TOKEN=token-interno-compartido
BACKEND_INTERNAL_TIMEOUT=30

ESTIMATION_LMSTUDIO_BASE_URL=http://localhost:1234/v1
ESTIMATION_LMSTUDIO_MODEL=local-model
```

## Notas importantes

- Si usas `ESTIMATION_PROVIDER=openai`, necesitas `OPENAI_API_KEY`.
- Si usas `ESTIMATION_PROVIDER=lmstudio`, debes tener LM Studio exponiendo API compatible con OpenAI.
- Para callbacks al backend, `INTERNAL_SERVICE_TOKEN` debe ser el mismo que usa `UniDev-BackEnd`.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar en local

```bash
python main.py --serve
```

Abrir:

- API: `http://localhost:5000`
- Health: `http://localhost:5000/api/health`

## Endpoints que usa UniDev

- `POST /generate-requirements`
- `POST /estimate-effort-async`
- `POST /requirements-chat`

## Dependencias externas esperadas

- OpenAI o LM Studio
- Backend UniDev en `http://localhost:8081` si vas a usar callbacks async

## Ajuste manual si quieres usar el endpoint `/api/generate`

El modelo local de `llama.cpp` tiene una ruta hardcodeada en `Entrenamiento/core/config.py`:

- `MODEL_PATH`

Si vas a usar ese endpoint con un modelo GGUF local, cambia esa ruta por una valida en tu maquina.

## Problemas comunes

- Si falla la estimacion async, revisa `INTERNAL_SERVICE_TOKEN`.
- Si falla `/api/generate`, normalmente falta un `MODEL_PATH` valido o dependencias de `llama-cpp-python`.
