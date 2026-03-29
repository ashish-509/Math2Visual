
# Multi-stage build for backend: builder and runtime stages

FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies and system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libc6-dev git pkg-config \
    ffmpeg texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-science \
    libcairo2-dev libpango1.0-dev libglib2.0-dev portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-science \
    libcairo2-dev libpango1.0-dev libglib2.0-dev portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# non-root user for security
RUN useradd --create-home appuser
USER appuser

COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser crawler/ ./crawler/
COPY --chown=appuser:appuser data/ ./data/

RUN mkdir -p outputs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
