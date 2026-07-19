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
