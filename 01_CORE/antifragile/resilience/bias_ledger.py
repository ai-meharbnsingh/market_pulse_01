# antifragile_framework/resilience/bias_ledger.py


import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# FIXED: Add proper path setup for telemetry imports
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent  # Go up to project root
TELEMETRY_PATH = PROJECT_ROOT / "01_Framework_Core" / "telemetry"

sys.path.insert(0, str(TELEMETRY_PATH))

# Now import telemetry modules with the correct path
try:
    from telemetry import event_topics
    from telemetry.core_logger import UniversalEventSchema, core_logger
    from telemetry.event_bus import EventBus
except ImportError as e:
    # Fallback for testing environments
    import warnings

    warnings.warn(f"Could not import telemetry modules: {e}")


    class MockEventTopics:
        BIAS_LOG_ENTRY_CREATED = "bias.log_entry.created"


    event_topics = MockEventTopics()


    class MockLogger:
        def log_event(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass


    core_logger = MockLogger()


    class MockEventBus:
        def publish(self, *args, **kwargs):
            pass


    EventBus = MockEventBus


    class MockEventSchema:
        def __init__(self, *args, **kwargs):
            pass


    UniversalEventSchema = MockEventSchema
from pydantic import BaseModel, Field
# Continue with your other imports (config, provider schemas, etc.)
from antifragile_framework.config.schemas import ProviderProfiles
from antifragile_framework.core.schemas import RequestContext
from antifragile_framework.providers.api_abstraction_layer import CompletionResponse

log = logging.getLogger(__name__)


# --- Helper Functions ---
def _generate_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _truncate_text(text: Optional[str], max_length: int) -> Optional[str]:
    if not isinstance(text, str):
        return None
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


# --- Pydantic Schema for the Audit Entry ---
class BiasLedgerEntry(BaseModel):
    request_id: str = Field(
        ..., description="Unique identifier for the entire request lifecycle."
    )
    timestamp_utc: str = Field(
        ..., description="ISO 8601 timestamp of when the entry was logged."
    )
    schema_version: int = Field(
        4,
        description="Schema version of the BiasLedgerEntry to handle future evolutions. Incremented for user-intent fields.",
    )

    # Input/Output Data
    initial_prompt_hash: str = Field(
        ...,
        description="SHA-256 hash of the original user prompt for integrity checks.",
    )
    initial_prompt_preview: str = Field(
        ..., description="A truncated preview of the original user prompt."
    )
    final_prompt_preview: Optional[str] = Field(
        None,
        description="A truncated preview of the final prompt if different from initial.",
    )
    final_response_preview: Optional[str] = Field(
        None, description="A truncated preview of the successful response."
    )

    # Outcome & Performance Metrics
    outcome: str = Field(
        ...,
        description="Final status (e.g., 'SUCCESS', 'FAILURE', 'MITIGATED_SUCCESS').",
    )
    total_latency_ms: float = Field(
        ..., description="Total time taken for the request in milliseconds."
    )
    total_api_calls: int = Field(
        ..., description="Total number of API calls made during the lifecycle."
    )
    final_provider: Optional[str] = Field(
        None, description="The name of the provider that ultimately succeeded."
    )
    final_model: Optional[str] = Field(
        None, description="The name of the model that ultimately succeeded."
    )

    # Resilience & Mitigation
    resilience_events: List[Dict[str, Any]] = Field(
        ..., description="A chronological list of all resilience actions."
    )
    mitigation_attempted: bool = Field(
        False, description="Whether a content policy mitigation was attempted."
    )
    mitigation_succeeded: Optional[bool] = Field(
        None, description="If mitigation was attempted, whether it succeeded."
    )
    resilience_score: Optional[float] = Field(
        None,
        description="Score from 1.0 (perfect) down to 0.0 (total failure).",
    )

    # NEW: Decision Context and Failover Reasons
    preferred_provider_requested: Optional[str] = Field(
        None,
        description="The preferred provider specified by the user in the request, if any.",
    )
    initial_selection_mode: str = Field(
        ...,
        description="Indicates how the initial provider selection was made (e.g., 'PREFERENCE_DRIVEN', 'VALUE_DRIVEN').",
    )
    failover_reason: Optional[str] = Field(
        None,
        description="Specific reason for a failover or for skipping a preferred provider/model (e.g., 'PREFERRED_PROVIDER_UNHEALTHY', 'ALL_PREFERRED_MODELS_EXCEED_COST_CAP_INITIAL_CHECK', 'API_ERROR', 'CIRCUIT_OPEN').",
    )
    cost_cap_enforced: bool = Field(
        False,
        description="True if a cost cap was active and influenced provider/model selection for this request (e.g., by skipping options).",
    )
    cost_cap_skip_reason: Optional[str] = Field(
        None,
        description="Specific reason for skipping a model/provider due to cost cap (e.g., 'MODEL_TOO_EXPENSIVE').",
    )

    # Cost & Usage Data
    input_tokens: Optional[int] = Field(
        None, description="Number of input tokens for the successful call."
    )
    output_tokens: Optional[int] = Field(
        None, description="Number of output tokens for the successful call."
    )
    estimated_cost_usd: Optional[Decimal] = Field(
        None, description="Estimated cost of the successful call in USD."
    )

    class Config:
        json_encoders = {Decimal: lambda v: str(v)}  # Serialize Decimal to string


class BiasLedger:
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        provider_profiles: Optional[ProviderProfiles] = None,
        prompt_preview_len: int = 512,
        response_preview_len: int = 1024,
    ):
        self.event_bus = event_bus
        self.provider_profiles = provider_profiles
        self.logger = core_logger
        self.prompt_preview_len = prompt_preview_len
        self.response_preview_len = response_preview_len

    def _calculate_estimated_cost(
        self, provider: str, model: str, in_tokens: int, out_tokens: int
    ) -> Optional[Decimal]:
        """Calculates the estimated cost with precision, handling missing profiles."""
        if not self.provider_profiles:
            return None

        try:
            provider_data = self.provider_profiles.profiles.get(provider, {})
            cost_profile: Optional[CostProfile] = provider_data.get(model)

            if not cost_profile:
                cost_profile = provider_data.get("_default")
                if cost_profile:
                    log.warning(
                        f"Cost profile for '{provider}/{model}' not found. Using provider default."
                    )
                else:
                    log.warning(
                        f"No cost profile or default found for provider '{provider}'. Cannot calculate cost."
                    )
                    return None

            input_cost = (Decimal(in_tokens) * cost_profile.input_cpm) / Decimal(
                "1000000"
            )
            output_cost = (Decimal(out_tokens) * cost_profile.output_cpm) / Decimal(
                "1000000"
            )

            total_cost = (input_cost + output_cost).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )
            return total_cost

        except Exception as e:
            log.error(
                f"Error calculating cost for {provider}/{model}: {e}",
                exc_info=True,
            )
            return None

    def log_request_lifecycle(
        self,
        context: RequestContext,
        initial_selection_mode: str,  # NEW PARAM
        final_response: Optional[CompletionResponse] = None,
        final_error: Optional[Exception] = None,
        resilience_score: Optional[float] = None,
        failover_reason: Optional[str] = None,  # NEW PARAM
        cost_cap_enforced: bool = False,  # NEW PARAM
        cost_cap_skip_reason: Optional[str] = None,  # NEW PARAM
    ) -> Optional[BiasLedgerEntry]:
        initial_user_content = next(
            (
                msg.content
                for msg in reversed(context.initial_messages)
                if msg.role == "user"
            ),
            "",
        )
        outcome = "FAILURE"
        final_prompt_content = initial_user_content
        input_tokens, output_tokens, estimated_cost_usd = None, None, None

        if final_response and final_response.success:
            outcome = "MITIGATED_SUCCESS" if context.mitigation_attempted else "SUCCESS"
            final_prompt_content = next(
                (
                    msg.content
                    for msg in reversed(context.final_messages)
                    if msg.role == "user"
                ),
                "",
            )

        try:
            timestamp_utc = datetime.now(timezone.utc).isoformat()
            total_latency_ms = round(
                (datetime.now(timezone.utc) - context.start_time).total_seconds()
                * 1000,
                2,
            )

            final_provider = (
                final_response.metadata.get("provider_name")
                if final_response and final_response.metadata
                else None
            )
            final_model = final_response.model_used if final_response else None

            if final_response and final_response.usage:
                input_tokens = final_response.usage.input_tokens
                output_tokens = final_response.usage.output_tokens
                if final_provider and final_model:
                    estimated_cost_usd = self._calculate_estimated_cost(
                        final_provider,
                        final_model,
                        input_tokens,
                        output_tokens,
                    )

            entry = BiasLedgerEntry(
                request_id=context.request_id,
                timestamp_utc=timestamp_utc,
                initial_prompt_hash=_generate_hash(initial_user_content),
                initial_prompt_preview=_truncate_text(
                    initial_user_content, self.prompt_preview_len
                ),
                final_prompt_preview=(
                    _truncate_text(final_prompt_content, self.prompt_preview_len)
                    if final_prompt_content != initial_user_content
                    else None
                ),
                final_response_preview=_truncate_text(
                    getattr(final_response, "content", None),
                    self.response_preview_len,
                ),
                outcome=outcome,
                total_latency_ms=total_latency_ms,
                total_api_calls=context.api_call_count,
                final_provider=final_provider,
                final_model=final_model,
                resilience_events=context.lifecycle_events or [],
                mitigation_attempted=context.mitigation_attempted,
                mitigation_succeeded=(
                    (outcome == "MITIGATED_SUCCESS")
                    if context.mitigation_attempted
                    else None
                ),
                resilience_score=resilience_score,
                # NEW FIELDS FOR DECISION CONTEXT
                preferred_provider_requested=context.preferred_provider,  # Pulled from RequestContext
                initial_selection_mode=initial_selection_mode,
                failover_reason=failover_reason,
                cost_cap_enforced=cost_cap_enforced,
                cost_cap_skip_reason=cost_cap_skip_reason,
                # Existing Cost & Usage Data
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimated_cost_usd,
                schema_version=4,  # UPDATED SCHEMA VERSION
            )

            if self.event_bus:
                payload = json.loads(
                    entry.model_dump_json()
                )  # Use model_dump_json for Decimal serialization
                event_schema = UniversalEventSchema(
                    event_type=event_topics.BIAS_LOG_ENTRY_CREATED,
                    event_topic="bias.ledger",
                    event_source=self.__class__.__name__,
                    timestamp_utc=entry.timestamp_utc,
                    severity="INFO",
                    payload=payload,
                )
                self.logger.log(event_schema)
                self.event_bus.publish(
                    event_type=event_topics.BIAS_LOG_ENTRY_CREATED,
                    payload=payload,
                )

            return entry

        except Exception as e:
            import traceback

            error_details = {
                "error": f"Failed to create BiasLedgerEntry: {e}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "request_id": context.request_id,
            }
            self.logger.log(
                UniversalEventSchema(
                    event_type="BIAS_LEDGER_FAILURE",
                    event_topic="bias.ledger",
                    event_source=self.__class__.__name__,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    severity="CRITICAL",
                    payload=error_details,
                )
            )
            return None