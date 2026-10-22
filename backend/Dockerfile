# Sift backend image.
# The build context is the repository root so the trained artifact in ml/
# can be copied in alongside the Django project.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DJANGO_DEBUG=0 \
    DJANGO_SETTINGS_MODULE=config.settings

WORKDIR /app

# Small runtime dependencies: curl is only used by the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY ml /app/ml
COPY backend /app/backend

# Non-root runtime user; the SQLite database lives in a writable volume.
RUN addgroup --system app \
    && adduser --system --ingroup app app \
    && mkdir -p /app/data /app/backend/staticfiles \
    && chown -R app:app /app

USER app
WORKDIR /app/backend

RUN python manage.py collectstatic --noinput

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health/ || exit 1

# Two workers keep memory modest: each worker loads its own copy of the model.
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 2 --timeout 120 --access-logfile - --error-logfile -"]
