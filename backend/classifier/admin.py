"""Django admin view for prediction history."""

from django.contrib import admin

from classifier.models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "label",
        "spam_probability",
        "source",
        "model_version",
        "created_at",
    )
    list_filter = ("label", "source", "model_version", "created_at")
    search_fields = ("text_preview",)
    date_hierarchy = "created_at"
    readonly_fields = (
        "text_preview",
        "text_length",
        "label",
        "spam_probability",
        "threshold",
        "top_words",
        "model_version",
        "source",
        "created_at",
    )
