"""Shared, consistent error responses for the API."""

from __future__ import annotations

from typing import Any

from rest_framework.views import exception_handler


def _message_from_data(data: Any) -> str:
    if isinstance(data, dict):
        detail = data.get("detail")
        if detail:
            return str(detail)
        first_key = next(iter(data), None)
        if first_key is not None:
            value = data[first_key]
            if isinstance(value, (list, tuple)) and value:
                return f"{first_key}: {value[0]}"
            return f"{first_key}: {value}"
    if isinstance(data, (list, tuple)) and data:
        return str(data[0])
    return "Request failed."


def api_exception_handler(exc: Exception, context: dict[str, Any]):
    """Wrap DRF errors as ``{"error": {...}}`` while preserving details.

    A single error shape means the frontend needs one code path for every
    failure: validation, throttling, not found, and server errors.
    """

    response = exception_handler(exc, context)
    if response is None:
        return None

    data = response.data
    response.data = {
        "error": {
            "status": response.status_code,
            "message": _message_from_data(data),
            "details": data,
        }
    }
    return response
