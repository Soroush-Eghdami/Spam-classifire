"""API endpoints for predictions, history, and health checks."""

from __future__ import annotations

import csv
import io
import logging
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import connection
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework import generics, status
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from classifier.models import Prediction
from classifier.serializers import (
    BatchUploadSerializer,
    PredictRequestSerializer,
    PredictionResultSerializer,
    PredictionSerializer,
)
from classifier.services import ModelNotReadyError, make_preview, model_service

logger = logging.getLogger(__name__)

TEXT_COLUMN_CANDIDATES = (
    "text",
    "email",
    "email_content",
    "message",
    "content",
    "body",
    "email_text",
)


def error_response(message: str, *, code: int, details: dict | None = None) -> Response:
    return Response(
        {
            "error": {
                "status": code,
                "message": message,
                "details": details or {},
            }
        },
        status=code,
    )


class PredictView(APIView):
    """Classify one message and save it to history."""

    parser_classes = [JSONParser, FormParser, MultiPartParser]
    throttle_scope = "predict"

    @extend_schema(
        request=PredictRequestSerializer,
        responses={200: PredictionResultSerializer},
        summary="Classify a single message",
        description="Returns the label, calibrated spam probability, decision threshold, and the words that drove the decision.",
    )
    def post(self, request):
        serializer = PredictRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        text = serializer.validated_data["text"]

        try:
            result = model_service.predict_one(text)
        except ModelNotReadyError as exc:
            return error_response(str(exc), code=status.HTTP_503_SERVICE_UNAVAILABLE)
        except ValueError as exc:
            return error_response(str(exc), code=status.HTTP_400_BAD_REQUEST)
        except Exception:  # noqa: BLE001
            logger.exception("Prediction failed")
            return error_response(
                "Prediction failed unexpectedly. The error has been logged.",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        prediction = Prediction.objects.create(
            text_preview=make_preview(text),
            text_length=len(text),
            label=result["label"],
            spam_probability=result["spam_probability"],
            threshold=result["threshold"],
            top_words=result.get("top_words", []),
            model_version=result.get("model_version", "unknown"),
            source=Prediction.SOURCE_SINGLE,
        )

        payload = {**result, "id": prediction.id, "created_at": prediction.created_at}
        return Response(PredictionResultSerializer(payload).data)


class BatchPredictView(APIView):
    """Classify every row in an uploaded CSV."""

    parser_classes = [MultiPartParser, FormParser]
    throttle_scope = "batch"

    @extend_schema(
        request=BatchUploadSerializer,
        responses={200: None},
        summary="Classify a CSV of messages",
        description=(
            "Upload a CSV (maximum 5 MB / 500 rows). The message column is "
            "auto-detected unless ``text_column`` is supplied. Empty rows are "
            "skipped and reported."
        ),
    )
    def post(self, request):
        upload = request.FILES.get("file")
        if upload is None:
            return error_response(
                "A CSV file is required in the 'file' field.",
                code=status.HTTP_400_BAD_REQUEST,
            )

        max_bytes = settings.MAX_BATCH_FILE_MB * 1024 * 1024
        if upload.size > max_bytes:
            return error_response(
                f"File is too large ({upload.size / (1024 * 1024):.1f} MB). "
                f"The limit is {settings.MAX_BATCH_FILE_MB} MB.",
                code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        if not upload.name.lower().endswith(".csv"):
            return error_response(
                "Only .csv files are supported.",
                code=status.HTTP_400_BAD_REQUEST,
            )

        raw = upload.read()
        decoded = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                decoded = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if decoded is None:
            return error_response(
                "Could not decode the CSV. Save it as UTF-8 or Latin-1.",
                code=status.HTTP_400_BAD_REQUEST,
            )
        decoded = decoded.lstrip("\ufeff")

        # Catch rows with unquoted commas before pandas truncates them.
        try:
            rows = list(csv.reader(io.StringIO(decoded)))
        except csv.Error as exc:
            return error_response(
                "Could not parse the CSV. Check the delimiter and quoting.",
                code=status.HTTP_400_BAD_REQUEST,
                details={"reason": str(exc)},
            )
        if not rows:
            return error_response(
                "The CSV is empty.",
                code=status.HTTP_400_BAD_REQUEST,
            )
        header = rows[0]
        for line_number, row in enumerate(rows[1:], start=2):
            if len(row) > len(header):
                return error_response(
                    f"Row {line_number} has more fields than the header. "
                    "Fields containing commas must be wrapped in double quotes.",
                    code=status.HTTP_400_BAD_REQUEST,
                    details={"row": line_number, "expected_fields": len(header)},
                )

        try:
            frame = pd.read_csv(
                io.StringIO(decoded),
                dtype=str,
                keep_default_na=False,
                index_col=False,
            )
        except pd.errors.ParserError as exc:
            return error_response(
                "Could not read the CSV. Check the delimiter and quoting.",
                code=status.HTTP_400_BAD_REQUEST,
                details={"reason": str(exc)},
            )
        if frame.empty:
            return error_response(
                "The CSV contains no data rows.",
                code=status.HTTP_400_BAD_REQUEST,
            )
        if len(frame) > settings.MAX_BATCH_ROWS:
            return error_response(
                f"Too many rows ({len(frame):,}). The limit is "
                f"{settings.MAX_BATCH_ROWS:,} rows per upload.",
                code=status.HTTP_400_BAD_REQUEST,
            )

        columns = [str(column).strip() for column in frame.columns]
        frame.columns = columns
        requested_column = str(request.data.get("text_column", "")).strip()
        if requested_column:
            if requested_column not in columns:
                return error_response(
                    f"Column '{requested_column}' was not found.",
                    code=status.HTTP_400_BAD_REQUEST,
                    details={"columns": columns},
                )
            text_column = requested_column
        elif "text" in columns:
            text_column = "text"
        else:
            text_column = next(
                (name for name in TEXT_COLUMN_CANDIDATES if name in columns),
                columns[0],
            )

        series = frame[text_column].fillna("").astype(str)
        non_empty = series.str.strip().str.len() > 0
        texts = series[non_empty].str.slice(0, settings.MAX_TEXT_CHARS).tolist()
        if not texts:
            return error_response(
                f"Column '{text_column}' contains no non-empty messages.",
                code=status.HTTP_400_BAD_REQUEST,
                details={"columns": columns},
            )

        try:
            predictions = model_service.predict_many(texts)
        except ModelNotReadyError as exc:
            return error_response(str(exc), code=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception:  # noqa: BLE001
            logger.exception("Batch prediction failed")
            return error_response(
                "Batch prediction failed unexpectedly.",
                code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        valid_positions = np.flatnonzero(non_empty.to_numpy())
        results = []
        to_create = []
        for row_index, prediction in zip(valid_positions, predictions):
            row_text = series.iloc[int(row_index)]
            result = {
                "row": int(row_index) + 1,
                "text_preview": make_preview(row_text, limit=160),
                "label": prediction["label"],
                "spam_probability": prediction["spam_probability"],
                "threshold": prediction["threshold"],
            }
            results.append(result)
            to_create.append(
                Prediction(
                    text_preview=make_preview(row_text),
                    text_length=len(row_text),
                    label=result["label"],
                    spam_probability=result["spam_probability"],
                    threshold=result["threshold"],
                    top_words=[],
                    model_version=prediction.get("model_version", "unknown"),
                    source=Prediction.SOURCE_BATCH,
                )
            )

        Prediction.objects.bulk_create(to_create, batch_size=200)

        spam_count = sum(1 for result in results if result["label"] == "spam")
        return Response(
            {
                "count": len(results),
                "skipped_empty": int((~non_empty).sum()),
                "spam_count": spam_count,
                "ham_count": len(results) - spam_count,
                "threshold": results[0]["threshold"] if results else 0.5,
                "model_version": predictions[0].get("model_version", "unknown") if predictions else "unknown",
                "text_column": text_column,
                "columns": columns,
                "results": results,
            }
        )


class HistoryView(generics.ListAPIView):
    """Paginated history with label, source, and text filters."""

    serializer_class = PredictionSerializer

    @extend_schema(
        parameters=[
            OpenApiParameter("label", str, enum=["spam", "ham"], description="Filter by label"),
            OpenApiParameter("source", str, enum=["single", "batch"], description="Filter by origin"),
            OpenApiParameter("search", str, description="Case-insensitive preview search"),
        ]
    )
    def get(self, request, *args, **kwargs):
        return super().get(request, *args, **kwargs)

    def get_queryset(self):
        queryset = Prediction.objects.all()
        label = self.request.query_params.get("label")
        source = self.request.query_params.get("source")
        search = self.request.query_params.get("search")

        if label:
            if label not in {Prediction.LABEL_SPAM, Prediction.LABEL_HAM}:
                raise ValidationError("label must be 'spam' or 'ham'")
            queryset = queryset.filter(label=label)
        if source:
            if source not in {Prediction.SOURCE_SINGLE, Prediction.SOURCE_BATCH}:
                raise ValidationError("source must be 'single' or 'batch'")
            queryset = queryset.filter(source=source)
        if search:
            queryset = queryset.filter(text_preview__icontains=search.strip()[:100])
        return queryset


class ModelInfoView(APIView):
    """Return model version, metrics, and training details."""

    @extend_schema(responses={200: dict})
    def get(self, request):
        info = model_service.model_info()
        info.setdefault("version", "unknown")
        return Response(info)


class StatsView(APIView):
    """Return counters for the dashboard."""

    @extend_schema(responses={200: dict})
    def get(self, request):
        queryset = Prediction.objects.all()
        total = queryset.count()
        spam = queryset.filter(label=Prediction.LABEL_SPAM).count()
        since = timezone.now() - timedelta(hours=24)
        recent = queryset.filter(created_at__gte=since)
        average = queryset.aggregate(value=Avg("spam_probability"))["value"]

        week_ago = timezone.now() - timedelta(days=7)
        trend_rows = (
            queryset.filter(created_at__gte=week_ago)
            .annotate(day=TruncDate("created_at"))
            .values("day")
            .annotate(count=Count("id"), spam=Count("id", filter=Q(label=Prediction.LABEL_SPAM)))
            .order_by("day")
        )

        return Response(
            {
                "total": total,
                "spam": spam,
                "ham": total - spam,
                "spam_rate": round(spam / total, 4) if total else 0.0,
                "last_24h": recent.count(),
                "last_24h_spam": recent.filter(label=Prediction.LABEL_SPAM).count(),
                "average_spam_probability": round(float(average), 4) if average is not None else 0.0,
                "trend": [
                    {
                        "date": row["day"].isoformat() if row["day"] else None,
                        "count": row["count"],
                        "spam": row["spam"],
                    }
                    for row in trend_rows
                ],
                "sources": {
                    row["source"]: row["count"]
                    for row in queryset.values("source").annotate(count=Count("id"))
                },
            }
        )


class HealthView(APIView):
    """Health check used by Docker and the frontend."""

    @extend_schema(responses={200: dict, 503: dict})
    def get(self, request):
        started = time.perf_counter()
        database_ok = True
        try:
            connection.ensure_connection()
        except Exception:  # noqa: BLE001
            database_ok = False

        try:
            model_service._ensure_loaded()  # noqa: SLF001
            model_ok = True
            model_error = None
        except ModelNotReadyError as exc:
            model_ok = False
            model_error = str(exc)

        status_payload = model_service.status()
        payload = {
            "status": "ok" if (database_ok and model_ok) else "degraded",
            "database": database_ok,
            "model_loaded": model_ok,
            "model_version": status_payload.get("model_version"),
            "threshold": status_payload.get("threshold"),
            "api_version": "2.1.0",
            "response_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        if model_error:
            payload["error"] = model_error

        return Response(
            payload,
            status=status.HTTP_200_OK if (database_ok and model_ok) else status.HTTP_503_SERVICE_UNAVAILABLE,
        )
