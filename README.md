# API para Modelo LLM con Base de Datos Vectorial

## Desarrollado por Nicolás Rodríguez Torres

Este proyecto ha sido desarrollado con fines educativos y como herramienta de ayuda para desarrolladores. Su objetivo es facilitar el acceso a modelos de lenguaje locales y permitir consultas basadas en documentos propios, respetando la privacidad de los datos y ofreciendo una alternativa a los servicios en la nube.

Como desarrollador, mi intención es proporcionar una solución accesible que permita a otros profesionales implementar sistemas de Retrieval Augmented Generation (RAG) de manera sencilla y sin depender de APIs externas costosas.

## Descripción del Proyecto

Este servicio proporciona una API que permite utilizar un modelo de lenguaje local (LLM) junto con una base de datos vectorial para almacenar y consultar documentos PDF. El sistema está diseñado para permitir consultas al modelo basadas exclusivamente en la información contenida en los documentos cargados (Retrieval Augmented Generation).

## Requisitos

- Python 3.8 o superior
- Dependencias en `requirements.txt`

## Instalación

```bash
pip install -r requirements.txt
```

## Iniciar el Servidor

```bash
python main.py --serve
```

Por defecto, el servidor escucha en el puerto 5000. Puedes cambiar el puerto con el parámetro `--port`.

## API Endpoints

### Gestión de PDFs

#### 1. Cargar un PDF
```
POST /api/pdf/upload
```
**Parámetros**:
- Form-data: `pdf_file` - Archivo PDF
- O JSON: `pdf_data` (base64), `filename`, `chunk_size`, `chunk_overlap`

**Ejemplo**:
```bash
curl -X POST -F "pdf_file=@documento.pdf" http://localhost:5000/api/pdf/upload
```

#### 2. Obtener información de PDFs cargados
```
GET /api/pdf/info
```
**Ejemplo**:
```bash
curl -X GET http://localhost:5000/api/pdf/info
```

#### 3. Eliminar un PDF específico
```
POST /api/data/clear-pdf
```
**Parámetros**:
- `pdf_name`: Nombre del PDF a eliminar

**Ejemplo**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_name": "documento.pdf"
  }' \
  http://localhost:5000/api/data/clear-pdf
```

#### 4. Eliminar todos los datos
```
POST /api/data/clear
```
**Parámetros**:
- `confirm`: Debe ser `true` para confirmar

**Ejemplo**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "confirm": true
  }' \
  http://localhost:5000/api/data/clear
```

### Consultas y Generación de Texto

#### 5. Consulta simplificada (solo respuesta)
```
POST /api/query-simple
```
**Parámetros**:
- `query`: Pregunta sobre el contenido
- `max_tokens`: Longitud máxima (default: 500)
- `temperature`: Temperatura (default: 0.7)
- `source_filter`: Opcional, filtrar por PDF específico

**Ejemplo**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cuáles son los puntos principales del documento?",
    "max_tokens": 500
  }' \
  http://localhost:5000/api/query-simple
```

#### 6. Consulta completa (con fuentes)
```
POST /api/query-pdf
```
**Parámetros**:
- `query`: Pregunta sobre el contenido
- `top_k`: Número de fragmentos a recuperar (default: 5)
- `max_tokens`: Longitud máxima (default: 250)
- `temperature`: Temperatura (default: 0.7)
- `source_filter`: Opcional, filtrar por PDF específico

**Ejemplo**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué información contiene el documento?",
    "top_k": 5
  }' \
  http://localhost:5000/api/query-pdf
```

#### 7. Generación de texto (sin contexto de PDFs)
```
POST /api/generate
```
**Parámetros**:
- `prompt`: Texto para generar respuesta
- `max_tokens`: Longitud máxima (default: 150)
- `temperature`: Temperatura (default: 0.7)

**Ejemplo**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "¿Qué es una red neuronal?",
    "max_tokens": 150
  }' \
  http://localhost:5000/api/generate
```

### Base de Datos Vectorial

#### 8. Añadir documento a la BD
```
POST /api/vector/add
```
**Parámetros**:
- `text`: Texto del documento
- `metadata`: Metadatos (opcional)

**Ejemplo**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Este es un texto de ejemplo.",
    "metadata": {"source": "manual"}
  }' \
  http://localhost:5000/api/vector/add
```

#### 9. Buscar documentos similares
```
POST /api/vector/search
```
**Parámetros**:
- `query`: Texto para buscar
- `top_k`: Número de resultados (default: 5)

**Ejemplo**:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "texto de ejemplo",
    "top_k": 3
  }' \
  http://localhost:5000/api/vector/search
```

### Utilidades

#### 10. Verificar estado del servicio
```
GET /api/health
```
**Ejemplo**:
```bash
curl -X GET http://localhost:5000/api/health
```

## Características

1. **Modo estricto**: El modelo solo responde basándose en la información de los documentos cargados.
2. **Procesamiento de PDFs**: División en fragmentos con superposición para mejorar la recuperación.
3. **Base de datos vectorial**: Almacenamiento eficiente de documentos y búsqueda por similitud semántica.
4. **Configuración flexible**: Ajuste de parámetros como temperatura, tamaño de fragmentos, etc.

## Ejemplos de Uso

### Flujo de trabajo típico:

1. Cargar un PDF:
```bash
curl -X POST -F "pdf_file=@documento.pdf" http://localhost:5000/api/pdf/upload
```

2. Verificar que se cargó correctamente:
```bash
curl -X GET http://localhost:5000/api/pdf/info
```

3. Realizar una consulta sobre el contenido:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Cuáles son los puntos principales del documento?"
  }' \
  http://localhost:5000/api/query-simple
```

## Notas

- El modelo responde solo con información presente en los documentos cuando se usa `/api/query-pdf` o `/api/query-simple`.
- Para borrar todos los datos, es necesario confirmar con `"confirm": true`.
- El endpoint `/api/query-simple` es más rápido ya que no devuelve las fuentes.

## Sobre mí

Mi nombre es Nicolás Rodríguez Torres, desarrollador interesado en la aplicación de tecnologías de IA en entornos locales para democratizar el acceso a estas herramientas. Este proyecto forma parte de mi compromiso con la comunidad de desarrollo, proporcionando soluciones de código abierto que faciliten la innovación y el aprendizaje.

Si tienes preguntas o sugerencias sobre este proyecto, no dudes en contactarme.

## Licencia

Este proyecto se distribuye bajo licencia MIT, permitiendo su uso libre tanto para fines educativos como comerciales.