"""URL routes for the classifier API."""

from django.urls import path

from classifier.views import (
    BatchPredictView,
    HealthView,
    HistoryView,
    ModelInfoView,
    PredictView,
    StatsView,
)

app_name = "classifier"

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("predict/batch/", BatchPredictView.as_view(), name="predict-batch"),
    path("history/", HistoryView.as_view(), name="history"),
    path("model-info/", ModelInfoView.as_view(), name="model-info"),
    path("stats/", StatsView.as_view(), name="stats"),
    path("health/", HealthView.as_view(), name="health"),
]
