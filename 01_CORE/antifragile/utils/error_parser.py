# antifragile_framework/utils/error_parser.py

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

# Provider SDKs are imported safely to allow the system to run even if some are not installed.
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google.api_core import exceptions as google_exceptions

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

log = logging.getLogger(__name__)


class ErrorCategory(Enum):
    FATAL = auto()
    TRANSIENT = auto()
    CONTENT_POLICY = auto()
    MODEL_ISSUE = auto()
    UNKNOWN = auto()


@dataclass
class ErrorDetails:
    category: ErrorCategory
    is_retriable: bool
    provider: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retry_after_seconds: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorParser:
    def __init__(self):
        self._build_exception_map()
        self.model_error_keywords = [
            "model not found",
            "model not available",
            "model does not support",
            "invalid model",
            "model does not exist",
            "model is deprecated",
            "context_length_exceeded",
        ]

    def _build_exception_map(self):
        self.EXCEPTION_MAP = {}
        if OPENAI_AVAILABLE:
            self.EXCEPTION_MAP.update(
                {
                    openai.AuthenticationError: ErrorCategory.FATAL,
                    openai.PermissionDeniedError: ErrorCategory.FATAL,
                    # Note: NotFoundError is now handled explicitly below
                    openai.RateLimitError: ErrorCategory.TRANSIENT,
                    openai.APIConnectionError: ErrorCategory.TRANSIENT,
                    openai.APITimeoutError: ErrorCategory.TRANSIENT,
                    openai.InternalServerError: ErrorCategory.TRANSIENT,
                }
            )
        # ... (rest of the map is the same)
        if ANTHROPIC_AVAILABLE:
            self.EXCEPTION_MAP.update(
                {
                    anthropic.AuthenticationError: ErrorCategory.FATAL,
                    anthropic.PermissionDeniedError: ErrorCategory.FATAL,
                    anthropic.NotFoundError: ErrorCategory.FATAL,
                    anthropic.RateLimitError: ErrorCategory.TRANSIENT,
                    anthropic.APIConnectionError: ErrorCategory.TRANSIENT,
                    anthropic.APITimeoutError: ErrorCategory.TRANSIENT,
                    anthropic.InternalServerError: ErrorCategory.TRANSIENT,
                }
            )
        if GOOGLE_AVAILABLE:
            self.EXCEPTION_MAP.update(
                {
                    google_exceptions.PermissionDenied: ErrorCategory.FATAL,
                    google_exceptions.Unauthenticated: ErrorCategory.FATAL,
                    google_exceptions.NotFound: ErrorCategory.FATAL,
                    google_exceptions.ResourceExhausted: ErrorCategory.TRANSIENT,
                    google_exceptions.ServiceUnavailable: ErrorCategory.TRANSIENT,
                    google_exceptions.DeadlineExceeded: ErrorCategory.TRANSIENT,
                    google_exceptions.InternalServerError: ErrorCategory.TRANSIENT,
                }
            )

    def _is_model_error_by_message(self, message: Optional[str]) -> bool:
        """Checks if an error message string indicates a model-specific issue."""
        if not message:
            return False
        lower_message = message.lower()
        return any(keyword in lower_message for keyword in self.model_error_keywords)

    def classify_error(self, exception: Exception, provider_name: str) -> ErrorDetails:
        error_message = str(exception)

        # ==============================================================================
        # FINAL FIX: Add a high-priority, explicit check for the most common model error.
        # This is more robust than relying on string matching alone.
        # ==============================================================================
        if OPENAI_AVAILABLE and isinstance(exception, openai.NotFoundError):
            return ErrorDetails(
                category=ErrorCategory.MODEL_ISSUE,
                is_retriable=False,
                provider=provider_name,
                error_message=error_message,
            )

        if OPENAI_AVAILABLE and isinstance(exception, openai.BadRequestError):
            error_details = self._extract_openai_error_details(exception)
            if error_details.get("error_code") == "content_policy_violation":
                return ErrorDetails(
                    category=ErrorCategory.CONTENT_POLICY,
                    is_retriable=True,
                    provider=provider_name,
                    **error_details
                )
            # Check for model issue even in bad requests, as context length errors are often 400s
            if self._is_model_error_by_message(error_details.get("error_message", "")):
                return ErrorDetails(
                    category=ErrorCategory.MODEL_ISSUE,
                    is_retriable=False,
                    provider=provider_name,
                    **error_details
                )
            return ErrorDetails(
                category=ErrorCategory.FATAL,
                is_retriable=False,
                provider=provider_name,
                **error_details
            )

        if ANTHROPIC_AVAILABLE and isinstance(exception, anthropic.BadRequestError):
            error_details = self._extract_anthropic_error_details(exception)
            error_type = error_details.get("error_code", "")
            if (
                error_type == "invalid_request_error"
                and "safety" in error_details.get("error_message", "").lower()
            ):
                return ErrorDetails(
                    category=ErrorCategory.CONTENT_POLICY,
                    is_retriable=True,
                    provider=provider_name,
                    **error_details
                )
            if self._is_model_error_by_message(error_details.get("error_message", "")):
                return ErrorDetails(
                    category=ErrorCategory.MODEL_ISSUE,
                    is_retriable=False,
                    provider=provider_name,
                    **error_details
                )
            return ErrorDetails(
                category=ErrorCategory.FATAL,
                is_retriable=False,
                provider=provider_name,
                **error_details
            )

        for exc_type in type(exception).__mro__:
            if exc_type in self.EXCEPTION_MAP:
                if self.EXCEPTION_MAP[
                    exc_type
                ] == ErrorCategory.FATAL and self._is_model_error_by_message(
                    error_message
                ):
                    return ErrorDetails(
                        category=ErrorCategory.MODEL_ISSUE,
                        is_retriable=False,
                        provider=provider_name,
                        error_message=error_message,
                    )
                category = self.EXCEPTION_MAP[exc_type]
                retry_after = (
                    self._extract_retry_after(exception)
                    if category == ErrorCategory.TRANSIENT
                    else None
                )
                return ErrorDetails(
                    category=category,
                    is_retriable=(category == ErrorCategory.TRANSIENT),
                    provider=provider_name,
                    retry_after_seconds=retry_after,
                    error_message=error_message,
                )

        return ErrorDetails(
            category=ErrorCategory.UNKNOWN,
            is_retriable=False,
            provider=provider_name,
            error_message=error_message,
        )

    def _extract_openai_error_details(
        self, exception: openai.APIError
    ) -> Dict[str, Any]:
        try:
            body = getattr(exception, "body", {}) or {}
            error = body.get("error", {}) or {}
            return {
                "error_code": error.get("code"),
                "error_message": error.get("message"),
                "metadata": {
                    "type": error.get("type"),
                    "param": error.get("param"),
                },
            }
        except Exception:
            return {"error_message": str(exception)}

    def _extract_anthropic_error_details(
        self, exception: anthropic.APIError
    ) -> Dict[str, Any]:
        try:
            if hasattr(exception, "response"):
                error_data = exception.response.json()
                if isinstance(error_data, dict) and "error" in error_data:
                    err = error_data["error"]
                    return {
                        "error_code": err.get("type"),
                        "error_message": err.get("message"),
                    }
        except Exception:
            pass
        return {"error_message": str(exception)}

    def _extract_retry_after(self, exception: Exception) -> Optional[int]:
        try:
            if hasattr(exception, "response") and hasattr(
                exception.response, "headers"
            ):
                retry_after_str = exception.response.headers.get("retry-after")
                if retry_after_str:
                    return int(retry_after_str)
        except (ValueError, AttributeError):
            pass
        return None
