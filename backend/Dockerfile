# Backend image. Build from the repo root so ml/ is included.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DJANGO_DEBUG=0 \
    DJANGO_SETTINGS_MODULE=config.settings

WORKDIR /app

# curl is used by the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY ml /app/ml
COPY backend /app/backend

# Run as a non-root user with a writable data folder.
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/data /app/backend/staticfiles \
    && chown -R app:app /app

USER app
WORKDIR /app/backend

# Build static files without the production secret; Compose adds it at runtime.
RUN DJANGO_DEBUG=1 python manage.py collectstatic --noinput

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health/ || exit 1

# Two workers keep memory low; each loads its own model copy.
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 120 --no-control-socket --access-logfile - --error-logfile -"]
