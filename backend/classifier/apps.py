"""App configuration.  The model is loaded lazily on the first prediction."""

from __future__ import annotations

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ClassifierConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"
    verbose_name = "Spam classifier"

    def ready(self) -> None:
        # Importing services here validates that ml/ is importable without
        # paying the cost of loading a multi-megabyte joblib artifact at
        # startup.  Gunicorn workers load the model on their first request.
        from django.conf import settings

        from classifier import services

        logger.info("Sift model path: %s", settings.MODEL_PATH)
        logger.debug("Service module ready: %s", services.__name__)
