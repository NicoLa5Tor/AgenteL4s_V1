FROM python:3.11-slim

WORKDIR /app

# Build deps for llama-cpp-python y faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch CPU-only first to avoid pulling 400MB+ CUDA packages
RUN pip install --no-cache-dir --timeout 300 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "5000", "--no_debug"]
