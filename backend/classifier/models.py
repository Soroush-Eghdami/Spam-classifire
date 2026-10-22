"""Database model for prediction history."""

from __future__ import annotations

from django.db import models
from django.utils import timezone


class Prediction(models.Model):
    """One classified message.

    Only a short preview of the message is stored.  Full emails can contain
    personal data and are not needed to show history, so the API deliberately
    keeps just enough context for a human to recognise the row.
    """

    SOURCE_SINGLE = "single"
    SOURCE_BATCH = "batch"
    SOURCE_CHOICES = [
        (SOURCE_SINGLE, "Single message"),
        (SOURCE_BATCH, "Batch upload"),
    ]

    LABEL_SPAM = "spam"
    LABEL_HAM = "ham"
    LABEL_CHOICES = [
        (LABEL_SPAM, "Spam"),
        (LABEL_HAM, "Ham"),
    ]

    text_preview = models.CharField(max_length=280)
    text_length = models.PositiveIntegerField(default=0)
    label = models.CharField(max_length=8, choices=LABEL_CHOICES, db_index=True)
    spam_probability = models.FloatField()
    threshold = models.FloatField(default=0.5)
    top_words = models.JSONField(default=list, blank=True)
    model_version = models.CharField(max_length=32, default="unknown")
    source = models.CharField(max_length=16, choices=SOURCE_CHOICES, default=SOURCE_SINGLE, db_index=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        indexes = [
            models.Index(fields=["label", "-created_at"], name="pred_label_created_idx"),
            models.Index(fields=["source", "-created_at"], name="pred_source_created_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.label} ({self.spam_probability:.2f})"
