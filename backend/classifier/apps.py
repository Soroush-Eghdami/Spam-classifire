"""App setup. The model loads on the first prediction."""

from __future__ import annotations

import logging

from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ClassifierConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "classifier"
    verbose_name = "Spam classifier"

    def ready(self) -> None:
        # Check imports now, but wait to load the large model file.
        from django.conf import settings

        from classifier import services

        logger.info("Sift model path: %s", settings.MODEL_PATH)
        logger.debug("Service module ready: %s", services.__name__)
