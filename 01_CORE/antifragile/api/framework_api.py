# antifragile_framework/api/framework_api.py

# ==============================================================================
# CRITICAL: LOAD ENVIRONMENT VARIABLES FIRST
# This must be the very first action to ensure all subsequent imports and
# global variables have access to the values in the .env file.
# ==============================================================================
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import uvicorn
from antifragile_framework.config.config_loader import load_provider_profiles
from antifragile_framework.core.exceptions import AllProvidersFailedError
from antifragile_framework.core.failover_engine import FailoverEngine
from antifragile_framework.core.online_learning_subscriber import (
    OnlineLearningSubscriber,
)
from antifragile_framework.core.provider_ranking_engine import (
    ProviderRankingEngine,
)
from antifragile_framework.providers.api_abstraction_layer import (
    ChatMessage,
    CompletionResponse,
)
from antifragile_framework.providers.provider_registry import (
    get_default_provider_registry,
)
from antifragile_framework.resilience.bias_ledger import BiasLedger
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from telemetry import event_topics
from telemetry.core_logger import UniversalEventSchema, core_logger
from telemetry.event_bus import EventBus

load_dotenv()

# ==============================================================================
# Standard Imports
# ==============================================================================

# ==============================================================================
# REFACTOR: Import the new registry function
# ==============================================================================


def get_api_keys_from_env(
    env_var_name: str, default: str = "YOUR_KEY_HERE"
) -> List[str]:
    keys_str = os.getenv(env_var_name, default)
    return [key.strip() for key in keys_str.split(",") if key.strip()]


DEFAULT_PROVIDER_CONFIGS = {
    "openai": {
        "api_keys": get_api_keys_from_env("OPENAI_API_KEY"),
        "resource_config": {},
        "circuit_breaker_config": {},
    },
    "google_gemini": {
        "api_keys": get_api_keys_from_env("GEMINI_API_KEY"),
        "resource_config": {},
        "circuit_breaker_config": {},
    },
    "anthropic": {
        "api_keys": get_api_keys_from_env("ANTHROPIC_API_KEY"),
        "resource_config": {},
        "circuit_breaker_config": {},
    },
}


class ChatCompletionRequest(BaseModel):
    model_priority_map: Dict[str, List[str]] = Field(
        ...,
        description=(
            "A dictionary mapping provider names to a prioritized list of "
            "their models. The order of providers serves as a fallback if "
            "no dynamic ranking is available."
        ),
        examples=[
            {
                "openai": ["gpt-4o", "gpt-4-turbo"],
                "google_gemini": ["gemini-1.5-flash-latest"],
                "anthropic": ["claude-3-5-sonnet-20240620"],
            }
        ],
    )
    messages: List[ChatMessage]
    preferred_provider: Optional[str] = Field(
        None,
        description=(
            "Optional: The user's preferred provider to attempt first. "
            "If specified, the system will prioritize this provider before "
            "falling back to dynamic ranking."
        ),
        examples=["openai", "anthropic"],
    )
    max_estimated_cost_usd: Optional[float] = Field(
        None,
        description=(
            "Optional: The maximum estimated cost (in USD) for this "
            "single API call. If a model's estimated cost exceeds this cap, "
            "it will be skipped."
        ),
        examples=[0.01, 0.05],
    )


class ErrorDetail(BaseModel):
    detail: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan, initializing all core components.
    """
    core_logger.log_event(
        event_type="api.startup.begin",
        event_topic="api.lifecycle",
        payload={"message": "Initializing Adaptive Mind components..."},
        severity="INFO"
    )

    is_perf_mode = os.getenv("PERFORMANCE_TEST_MODE", "False").lower() == "true"
    if is_perf_mode:
        print("[OK] Lifespan: PERFORMANCE_TEST_MODE is 'True'. Init mocks.")
        core_logger.log_event(
            event_type="api.startup.mode",
            event_topic="api.lifecycle",
            payload={"message": "PERFORMANCE_TEST_MODE is active."},
            severity="WARNING"
        )
    else:
        print("[INFO] Lifespan: PERFORMANCE_TEST_MODE not set. Init production.")

    try:
        provider_profiles = load_provider_profiles()
    except (FileNotFoundError, ValueError) as e:
        core_logger.log_event(
            event_type="api.startup.failure",
            event_topic="api.errors",
            payload={"error": f"Failed to load provider profiles: {e}"},
            severity="CRITICAL"
        )
        sys.exit(f"CRITICAL ERROR: Invalid provider profiles. Error: {e}")

    # ==============================================================================
    # REFACTOR: Create the provider registry
    # ==============================================================================
    provider_registry = get_default_provider_registry()

    event_bus = EventBus()
    ranking_engine = ProviderRankingEngine()
    learning_subscriber = OnlineLearningSubscriber(ranking_engine)
    bias_ledger = BiasLedger(event_bus=event_bus, provider_profiles=provider_profiles)

    event_bus.subscribe(
        event_topics.LEARNING_FEEDBACK_PUBLISHED,
        learning_subscriber.handle_event,
    )
    core_logger.log_event(
        event_type="api.startup.wiring",
        event_topic="system.setup",
        payload={"message": "Online learning subscriber connected to event bus."},
        severity="INFO"
    )

    # ==============================================================================
    # REFACTOR: Inject the provider registry into the FailoverEngine
    # ==============================================================================
    failover_engine = FailoverEngine(
        provider_configs=DEFAULT_PROVIDER_CONFIGS,
        provider_registry=provider_registry,
        event_bus=event_bus,
        bias_ledger=bias_ledger,
        provider_ranking_engine=ranking_engine,
        provider_profiles=provider_profiles,  # ← This line added
    )

    app.state.failover_engine = failover_engine
    app.state.ranking_engine = ranking_engine

    core_logger.log_event(
        event_type="api.startup.end",
        event_topic="api.lifecycle",
        payload={"message": "Adaptive Mind API initialization complete."},
        severity="INFO"
    )
    yield
    core_logger.log_event(
        event_type="api.shutdown",
        event_topic="api.lifecycle",
        payload={"message": "Adaptive Mind API shutting down."},
        severity="INFO"
    )
    event_bus.shutdown()


app = FastAPI(
    title="Adaptive Mind Resilience Framework API",
    version="1.9.0",  # Version Bump
    description="Exposes the core functionality of Adaptive Mind framework",
    lifespan=lifespan,
)


@app.exception_handler(AllProvidersFailedError)
async def all_providers_failed_exception_handler(
    request: Request, exc: AllProvidersFailedError
):
    request_id = getattr(request.state, "request_id", "N/A")
    core_logger.log_event(
        event_type=event_topics.API_SERVICE_UNAVAILABLE,
        event_topic="api.errors",
        payload={
            "request_id": request_id,
            "error": str(exc),
            "client_host": (request.client.host if request.client else "N/A"),
        },
        severity="ERROR"
    )
    return JSONResponse(
        status_code=503,
        content={"detail": "All underlying AI providers are currently unavailable."},
        headers={"Retry-After": "60"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "N/A")
    core_logger.log_event(
        event_type=event_topics.API_UNHANDLED_ERROR,
        event_topic="api.errors",
        payload={
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error_details": str(exc),
            "client_host": (request.client.host if request.client else "N/A"),
        },
        severity="CRITICAL"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred."},
    )


@app.middleware("http")
async def logging_and_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    core_logger.log_event(
        event_type=event_topics.API_REQUEST_START,
        event_topic="api.requests",
        payload={
            "request_id": request_id,
            "client_host": (request.client.host if request.client else "N/A"),
            "method": request.method,
            "path": request.url.path,
        },
        severity="INFO"
    )
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    core_logger.log_event(
        event_type=event_topics.API_REQUEST_END,
        event_topic="api.requests",
        payload={
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time_ms": round(process_time, 2),
        },
        severity="INFO"
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", tags=["Monitoring"])
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/v1/chat/completions",
    response_model=CompletionResponse,
    responses={503: {"model": ErrorDetail}, 500: {"model": ErrorDetail}},
    tags=["Core Functionality"],
)
async def chat_completions(request: Request, body: ChatCompletionRequest):
    failover_engine: FailoverEngine = request.app.state.failover_engine
    completion_response = await failover_engine.execute_request(
        model_priority_map=body.model_priority_map,
        messages=body.messages,
        request_id=request.state.request_id,
        preferred_provider=body.preferred_provider,
        max_estimated_cost_usd=body.max_estimated_cost_usd,
    )
    return completion_response


@app.get("/v1/learning/rankings", tags=["Learning Engine"])
async def get_provider_rankings(request: Request):
    """
    Returns the current real-time performance scores and rankings of providers.
    """
    ranking_engine: ProviderRankingEngine = request.app.state.ranking_engine
    return {
        "ranked_providers": ranking_engine.get_ranked_providers(),
        "provider_scores": ranking_engine.get_provider_scores(),
    }


if __name__ == "__main__":
    print("Starting Adaptive Mind API server with Uvicorn...")
    uvicorn.run(
        "antifragile_framework.api.framework_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
