FROM python:3.11-slim AS builder

WORKDIR /app

# portaudio + C headers for building PyAudio from source
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.frontend.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.frontend.txt

# --- runtime stage ---
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# non-root user
RUN useradd --create-home appuser
USER appuser

COPY --chown=appuser:appuser UI/ ./UI/

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV BACKEND_URL=http://backend:8000

EXPOSE 8501

CMD ["streamlit", "run", "UI/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
