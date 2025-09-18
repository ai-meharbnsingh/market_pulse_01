# antifragile_framework/core/schemas.py

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class RequestContext:
    """Manages the state of a single request's lifecycle for auditing."""

    # Non-default fields must come first.
    initial_messages: List[
        Dict[str, Any]
        # Changed to Dict[str, Any] to avoid circular dependency with ChatMessage
    ]
    final_messages: List[Dict[str, Any]]  # Changed to Dict[str, Any]

    # Default fields follow.
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lifecycle_events: List[Dict[str, Any]] = field(default_factory=list)
    api_call_count: int = 0
    mitigation_attempted: bool = False
    # NEW FIELDS FOR USER-INTENT FIRST ARCHITECTURE
    preferred_provider: Optional[str] = None
    max_estimated_cost_usd: Optional[float] = None
    cost_cap_enforced: bool = (
        False  # NEW: Track if cost cap was enforced for this request
    )
    cost_cap_skip_reason: Optional[str] = None  # NEW: Reason for cost cap enforcement


class ProviderPerformanceAnalysis(BaseModel):
    """
    Represents an aggregated analysis of a specific provider/model's performance
    over a given time period, derived from BiasLedger data.
    """

    model_config = ConfigDict(
        protected_namespaces=()
    )  # ADDED: Fix for pydantic warnings

    provider_name: str = Field(
        ...,
        description="Name of the AI provider (e.g., 'openai', 'anthropic').",
    )
    model_name: str = Field(
        ...,
        description="Name of the specific model (e.g., 'gpt-4o', 'claude-3-opus-20240229').",
    )

    total_requests: int = Field(
        0,
        ge=0,
        description="Total number of requests made to this provider/model.",
    )
    successful_requests: int = Field(
        0, ge=0, description="Number of successful requests."
    )

    success_rate: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Percentage of successful requests (0.0 to 1.0).",
    )
    avg_latency_ms: float = Field(
        0.0,
        ge=0.0,
        description="Average latency in milliseconds for successful requests.",
    )

    error_distribution: Dict[str, int] = Field(
        {}, description="Count of different error types encountered."
    )

    mitigation_attempted_count: int = Field(
        0,
        ge=0,
        description="Number of requests where mitigation was attempted.",
    )
    mitigation_successful_count: int = Field(
        0,
        ge=0,
        description="Number of requests where mitigation was successful.",
    )
    mitigation_success_rate: float = Field(
        0.0, ge=0.0, le=1.0, description="Success rate of mitigation attempts."
    )

    failover_occurred_count: int = Field(
        0,
        ge=0,
        description="Number of requests that led to a failover from this provider/model.",
    )
    failover_rate: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Rate at which requests from this provider/model led to failovers.",
    )

    circuit_breaker_tripped_count: int = Field(
        0,
        ge=0,
        description="Number of times the circuit breaker for this provider/model tripped.",
    )
    circuit_breaker_trip_rate: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Rate at which the circuit breaker for this provider/model tripped.",
    )

    avg_resilience_score: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Average resilience score for requests handled by this provider/model.",
    )

    analysis_period_start: datetime = Field(
        ..., description="Start timestamp (UTC) of the analysis period."
    )
    analysis_period_end: datetime = Field(
        ..., description="End timestamp (UTC) of the analysis period."
    )

    # Optional field for unique identifier if needed for finer granularity (e.g., API key ID)
    api_key_id: Optional[str] = Field(
        None,
        description="The specific API key ID this analysis pertains to, if applicable.",
    )


class UniversalEventSchema(BaseModel):
    """
    A standardized Pydantic schema for all telemetry events logged by the framework.
    This ensures consistency for ingestion into the TimeSeriesDB.
    """

    model_config = ConfigDict(
        protected_namespaces=()
    )  # ADDED: Fix for pydantic warnings
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this event.",
    )
    event_type: str = Field(
        ...,
        description="A categorized name for the event (e.g., 'api.call.success', 'framework.failover').",
    )
    event_topic: str = Field(
        ...,
        description="The broad topic/stream this event belongs to (e.g., 'api.call', 'bias.ledger', 'system.health').",
    )
    event_source: str = Field(
        ...,
        description="The component or module that emitted this event (e.g., 'FailoverEngine', 'BiasLedger').",
    )
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of when the event occurred.",
    )
    severity: str = Field(
        "INFO",
        description="Severity level of the event (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="A JSON-serializable dictionary containing event-specific data.",
    )
    parent_event_id: Optional[str] = Field(
        None,
        description="Optional: ID of a parent event for causality tracking.",
    )
