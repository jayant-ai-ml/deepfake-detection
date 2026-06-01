FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    DEPLOYMENT_MODEL_DIR=/app/deployment_models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-production.txt .
RUN python -m pip install --no-cache-dir -r requirements-production.txt

COPY app_api ./app_api
COPY utils ./utils
COPY deployment_models ./deployment_models

CMD ["sh", "-c", "python -m uvicorn app_api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
