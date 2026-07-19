FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Automation" \
      org.opencontainers.image.description="Production container for the Automation platform" \
      org.opencontainers.image.source="https://github.com/Dj221981/Automation" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY config/ /app/config/

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('src') else 1)"

CMD ["python", "-m", "src"]
