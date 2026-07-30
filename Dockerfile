<<<<<<< HEAD
FROM python:3.11-slim AS builder

LABEL org.opencontainers.image.title="Automation" \
      org.opencontainers.image.description="Production container for the Automation reinforcement-learning system" \
      org.opencontainers.image.source="https://github.com/Dj221981/Automation" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY config/ /app/config/

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/logs /app/checkpoints \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import importlib; importlib.import_module('src.config')" || exit 1

CMD ["python", "-m", "src"]
=======
# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM base AS runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY src src
COPY tests tests
COPY requirements-dev.txt requirements-dev.txt
CMD ["python", "-m", "pytest", "tests/test_neural_network.py", "-v"]
>>>>>>> origin/main
