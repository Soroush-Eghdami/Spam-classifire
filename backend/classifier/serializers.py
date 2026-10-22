"""Serializers for requests, prediction history, and model metadata."""

from __future__ import annotations

from django.conf import settings
from rest_framework import serializers

from classifier.models import Prediction


class PredictRequestSerializer(serializers.Serializer):
    """Validate a single message before it ever reaches the model."""

    text = serializers.CharField(
        required=True,
        allow_blank=False,
        trim_whitespace=False,
        style={"base_template": "textarea.html"},
        help_text=f"Message body to classify (1-{settings.MAX_TEXT_CHARS} characters).",
    )

    def validate_text(self, value: str) -> str:
        if not value or not value.strip():
            raise serializers.ValidationError("Text must not be empty or whitespace only.")
        if len(value) > settings.MAX_TEXT_CHARS:
            raise serializers.ValidationError(
                f"Text is too long ({len(value):,} characters). "
                f"The limit is {settings.MAX_TEXT_CHARS:,} characters."
            )
        return value


class BatchUploadSerializer(serializers.Serializer):
    """Multipart form used by the batch endpoint (for schema generation)."""

    file = serializers.FileField(help_text="CSV file, maximum 5 MB and 500 rows.")
    text_column = serializers.CharField(
        required=False,
        allow_blank=True,
        help_text="Name of the column containing the message. Auto-detected if omitted.",
    )


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = [
            "id",
            "text_preview",
            "text_length",
            "label",
            "spam_probability",
            "threshold",
            "top_words",
            "model_version",
            "source",
            "created_at",
        ]
        read_only_fields = fields


class PredictionResultSerializer(serializers.Serializer):
    """Response contract for ``POST /api/predict/``."""

    id = serializers.IntegerField()
    label = serializers.ChoiceField(choices=["spam", "ham"])
    spam_probability = serializers.FloatField()
    ham_probability = serializers.FloatField()
    threshold = serializers.FloatField()
    confidence = serializers.FloatField()
    top_words = serializers.ListField(child=serializers.DictField())
    signals = serializers.DictField()
    model_version = serializers.CharField()
    text_length = serializers.IntegerField()
    created_at = serializers.DateTimeField()
