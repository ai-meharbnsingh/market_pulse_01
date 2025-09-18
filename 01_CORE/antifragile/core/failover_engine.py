# antifragile_framework/core/failover_engine.py

import json
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

from antifragile_framework.config.config_loader import load_resilience_config
from antifragile_framework.config.schemas import CostProfile, ProviderProfiles
from antifragile_framework.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitBreakerState,
)
from antifragile_framework.core.resource_guard import ResourceGuard
from antifragile_framework.providers.api_abstraction_layer import (
    ChatMessage,
    CompletionResponse,
    LLMProvider,
)

# ==============================================================================
# REFACTOR 1: Import the ProviderRegistry instead of individual providers
# ==============================================================================
from antifragile_framework.providers.provider_registry import (
    ProviderRegistry,
    get_default_provider_registry,
)
from antifragile_framework.resilience.bias_ledger import BiasLedger
from antifragile_framework.resilience.prompt_rewriter import PromptRewriter
from antifragile_framework.utils.error_parser import ErrorCategory, ErrorParser
from telemetry import event_topics
from telemetry.core_logger import UniversalEventSchema, core_logger
from telemetry.event_bus import EventBus

from .exceptions import (
    AllProviderKeysFailedError,
    AllProvidersFailedError,
    ContentPolicyError,
    NoResourcesAvailableError,
    RewriteFailedError,
)
from .provider_ranking_engine import ProviderRankingEngine
from .schemas import RequestContext

log = logging.getLogger(__name__)

# A conservative estimate for cost capping if user doesn't specify max_tokens
DEFAULT_MAX_OUTPUT_TOKEN_ESTIMATE = 2048


class FailoverEngine:
    def __init__(
        self,
        provider_configs: Dict[str, Dict[str, Any]],
        # ==============================================================================
        # REFACTOR 1: Accept a ProviderRegistry in the constructor
        # ==============================================================================
        provider_registry: Optional[ProviderRegistry] = None,
        event_bus: Optional[EventBus] = None,
        prompt_rewriter: Optional[PromptRewriter] = None,
        bias_ledger: Optional[BiasLedger] = None,
        provider_ranking_engine: Optional[ProviderRankingEngine] = None,
        provider_profiles: Optional[ProviderProfiles] = None,
        config_path: Optional[str] = None,
    ):

        # Use the provided registry or create a default one
        self.provider_registry = provider_registry or get_default_provider_registry()

        self.providers: Dict[str, LLMProvider] = {}
        self.guards: Dict[str, ResourceGuard] = {}
        self.circuit_breakers = CircuitBreakerRegistry()
        self.error_parser = ErrorParser()
        self.event_bus = event_bus
        self.prompt_rewriter = prompt_rewriter
        self.bias_ledger = bias_ledger
        self.provider_ranking_engine = provider_ranking_engine
        self.logger = core_logger
        self.provider_profiles = provider_profiles

        self.resilience_config = load_resilience_config(config_path=config_path)
        self.resilience_score_penalties: Dict[str, float] = {}
        self._load_and_validate_penalties()

        for name, config in provider_configs.items():
            # ==============================================================================
            # REFACTOR 1: Use the registry to get the provider class dynamically
            # ==============================================================================
            try:
                provider_class = self.provider_registry.get_provider_class(name)
            except KeyError:
                log.warning(f"Provider '{name}' not found in registry. Skipping.")
                continue

            api_keys = config.get("api_keys")
            if not api_keys:
                log.warning(f"No API keys for provider '{name}'. Skipping.")
                continue

            provider_instance_config = config.copy()
            if "api_key" not in provider_instance_config:
                provider_instance_config["api_key"] = api_keys[0]

            self.providers[name] = provider_class(provider_instance_config)
            self.guards[name] = ResourceGuard(
                provider_name=name,
                api_keys=api_keys,
                resource_config=config.get("resource_config", {}),
                event_bus=self.event_bus,
            )
            self.circuit_breakers.get_breaker(
                name, **config.get("circuit_breaker_config", {})
            )
            log.info(f"Initialized provider '{name}' with {len(api_keys)} resources.")

    # ... (methods _load_and_validate_penalties through _is_provider_healthy_for_dynamic_selection remain the same) ...
    def _load_and_validate_penalties(self):
        penalties_config = self.resilience_config.get("resilience_score_penalties", {})
        expected_penalties = [
            "base_successful_penalty",
            "mitigated_success_penalty",
            "api_call_failure_penalty",
            "api_key_rotation_penalty",
            "model_failover_penalty",
            "provider_failover_penalty",
            "circuit_tripped_penalty",
            "all_providers_failed_penalty",
        ]
        for penalty_key in expected_penalties:
            penalty_value = penalties_config.get(penalty_key)
            if not isinstance(penalty_value, (int, float)):
                raise ValueError(
                    f"Resilience score penalty '{penalty_key}' must be a number. Found: {penalty_value}"
                )
            if not (0.0 <= penalty_value <= 1.0):
                raise ValueError(
                    f"Resilience score penalty '{penalty_key}' must be between 0.0 and 1.0. Found: {penalty_value}"
                )
            self.resilience_score_penalties[penalty_key] = float(penalty_value)
        log.info("Resilience score penalties loaded and validated successfully.")

    def _record_lifecycle_event(
            self,
            context: RequestContext,
            event_name: str,
            severity: str,
            payload_data: Dict[str, Any],
    ):
        timestamp = datetime.now(timezone.utc).isoformat()
        context.lifecycle_events.append(
            {"timestamp": timestamp, "event_name": event_name, **payload_data}
        )

        # Map event types to appropriate topics
        event_topic = self._get_event_topic(event_name)

        event_schema = UniversalEventSchema(
            event_type=event_name,
            event_topic=event_topic,  # ADD THIS LINE
            event_source=self.__class__.__name__,
            timestamp_utc=timestamp,
            severity=severity,
            payload={"request_id": context.request_id, **payload_data},
        )

        # Fix the logger call - use log_event instead of log
        self.logger.log_event(
            event_type=event_name,
            event_topic=event_topic,
            payload={"request_id": context.request_id, **payload_data},
            severity=severity
        )

        if self.event_bus:
            self.event_bus.publish(
                event_type=event_name, payload=event_schema.model_dump()
            )

    def _get_event_topic(self, event_type: str) -> str:
        """Map event types to appropriate topics"""
        topic_mapping = {
            "api_key.rotation": "resilience.failover",
            "model.failover": "resilience.failover",
            "provider.failover": "resilience.failover",
            "circuit.tripped": "system.health",
            "model.skipped.cost_cap": "cost.management",
            "prompt.humanization.attempt": "content.policy",
            "prompt.humanization.success": "content.policy",
            "prompt.humanization.failure": "content.policy",
        }
        return topic_mapping.get(event_type, "system.general")

    def _calculate_resilience_score(
        self, context: RequestContext, final_outcome: str
    ) -> float:
        score = 1.0
        if final_outcome == "MITIGATED_SUCCESS":
            score -= self.resilience_score_penalties.get(
                "mitigated_success_penalty", 0.0
            )
        for event in context.lifecycle_events:
            event_type = event.get("event_type")
            if event_type == event_topics.API_CALL_FAILURE:
                score -= self.resilience_score_penalties.get(
                    "api_call_failure_penalty", 0.0
                )
            elif event_type == event_topics.API_KEY_ROTATION:
                score -= self.resilience_score_penalties.get(
                    "api_key_rotation_penalty", 0.0
                )
            elif event_type == event_topics.MODEL_FAILOVER:
                score -= self.resilience_score_penalties.get(
                    "model_failover_penalty", 0.0
                )
            elif event_type == event_topics.PROVIDER_FAILOVER:
                score -= self.resilience_score_penalties.get(
                    "provider_failover_penalty", 0.0
                )
            elif event_type == event_topics.CIRCUIT_TRIPPED:
                score -= self.resilience_score_penalties.get(
                    "circuit_tripped_penalty", 0.0
                )
            elif event_type == event_topics.ALL_PROVIDERS_FAILED:
                score -= self.resilience_score_penalties.get(
                    "all_providers_failed_penalty", 0.0
                )
            elif event_type == event_topics.PROMPT_HUMANIZATION_ATTEMPT:
                score -= self.resilience_score_penalties.get(
                    "mitigated_success_penalty", 0.0
                )
        return max(0.0, min(1.0, score))

    def _estimate_prompt_tokens(self, messages: List[ChatMessage]) -> int:
        total_chars = 0
        for message in messages:
            if message.content:
                total_chars += len(message.content)
        return math.ceil(total_chars / 4) if total_chars > 0 else 0

    def _estimate_call_cost(
        self,
        provider_name: str,
        model_name: str,
        input_tokens: int,
        output_tokens_estimate: int,
    ) -> Optional[Decimal]:
        if not self.provider_profiles:
            log.warning("Provider profiles not loaded. Cannot estimate cost.")
            return None
        try:
            provider_data = self.provider_profiles.profiles.get(provider_name)
            if not provider_data:
                log.warning(
                    f"No cost data for provider '{provider_name}'. Cannot estimate cost."
                )
                return None
            cost_profile: Optional[CostProfile] = provider_data.get(model_name)
            if not cost_profile:
                cost_profile = provider_data.get("_default")
                if cost_profile:
                    log.debug(
                        f"Cost profile for '{provider_name}/{model_name}' not found. Using provider default."
                    )
                else:
                    log.warning(
                        f"No cost profile or default found for provider '{provider_name}'. Cannot estimate cost."
                    )
                    return None
            input_cost = (Decimal(input_tokens) * cost_profile.input_cpm) / Decimal(
                "1000000"
            )
            output_cost = (
                Decimal(output_tokens_estimate) * cost_profile.output_cpm
            ) / Decimal("1000000")
            total_cost = (input_cost + output_cost).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )
            return total_cost
        except Exception as e:
            log.error(
                f"Error calculating estimated cost for {provider_name}/{model_name}: {e}",
                exc_info=True,
            )
            return None

    def _get_dynamic_provider_priority(
        self, model_priority_map: Dict[str, List[str]]
    ) -> List[str]:
        static_providers = list(model_priority_map.keys())
        if not self.provider_ranking_engine:
            log.warning(
                "ProviderRankingEngine not configured. Using static provider order from request."
            )
            return [
                p
                for p in static_providers
                if self._is_provider_healthy_for_dynamic_selection(p)
            ]

        ranked_providers: List[str] = (
            self.provider_ranking_engine.get_ranked_providers()
        )
        if not ranked_providers:
            log.info(
                "No provider rankings available yet. Using static provider order from request."
            )
            return [
                p
                for p in static_providers
                if self._is_provider_healthy_for_dynamic_selection(p)
            ]

        static_providers_set = {p.lower() for p in static_providers}
        filtered_ranked_providers = [
            p for p in ranked_providers if p.lower() in static_providers_set
        ]
        filtered_ranked_set = {p.lower() for p in filtered_ranked_providers}
        unranked_providers = [
            p for p in static_providers if p.lower() not in filtered_ranked_set
        ]

        final_priority_unfiltered = filtered_ranked_providers + unranked_providers
        final_priority = [
            p
            for p in final_priority_unfiltered
            if self._is_provider_healthy_for_dynamic_selection(p)
        ]

        log.debug(
            f"Dynamic provider order set: {final_priority}. Original map keys: {static_providers}. Rankings: {ranked_providers}"
        )
        return final_priority

    def _is_provider_healthy_for_dynamic_selection(self, provider_name: str) -> bool:
        if provider_name not in self.providers:
            log.debug(
                f"Provider '{provider_name}' not initialized. Skipping in dynamic selection."
            )
            return False

        breaker = self.circuit_breakers.get_breaker(provider_name)
        if breaker.state == CircuitBreakerState.OPEN:
            log.debug(
                f"Provider '{provider_name}' skipped in dynamic selection: circuit is OPEN."
            )
            return False

        guard = self.guards.get(provider_name)
        if not guard or not guard.has_healthy_resources():
            log.debug(
                f"Provider '{provider_name}' skipped in dynamic selection: no healthy resources."
            )
            return False

        return True

    # ... (execute_request and _attempt_request_sequence remain largely the same, but call the updated _attempt_model_with_keys) ...
    async def execute_request(
        self,
        model_priority_map: Dict[str, List[str]],
        messages: List[ChatMessage],
        request_id: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        max_estimated_cost_usd: Optional[float] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        context_request_id = request_id or str(uuid.uuid4())

        context = RequestContext(
            request_id=context_request_id,
            initial_messages=messages,
            final_messages=messages,
            preferred_provider=preferred_provider,
            max_estimated_cost_usd=max_estimated_cost_usd,
            cost_cap_enforced=False,
            cost_cap_skip_reason=None,
        )

        final_response: Optional[CompletionResponse] = None
        final_exception: Optional[Exception] = None
        final_outcome: str = "FAILURE"
        initial_selection_mode: str = (
            "PREFERENCE_DRIVEN" if preferred_provider else "VALUE_DRIVEN"
        )
        failover_reason: Optional[str] = None

        provider_sequence_to_attempt: Optional[List[str]] = None

        try:
            estimated_input_tokens = self._estimate_prompt_tokens(
                context.initial_messages
            )
            output_tokens_for_estimate = kwargs.get(
                "max_tokens", DEFAULT_MAX_OUTPUT_TOKEN_ESTIMATE
            )

            if preferred_provider:
                provider_name_lower = preferred_provider.lower()

                if provider_name_lower not in self.providers:
                    failover_reason = f"PREFERRED_PROVIDER_UNKNOWN_OR_UNCONFIGURED:{preferred_provider}"
                else:
                    breaker = self.circuit_breakers.get_breaker(provider_name_lower)
                    guard = self.guards[provider_name_lower]
                    try:
                        breaker.check()
                    except CircuitBreakerError:
                        failover_reason = (
                            f"PREFERRED_PROVIDER_CIRCUIT_OPEN:{preferred_provider}"
                        )
                        self._record_lifecycle_event(
                            context,
                            event_topics.CIRCUIT_TRIPPED,
                            "CRITICAL",
                            {
                                "provider": provider_name_lower,
                                "context": "pre-check",
                            },
                        )
                    if not failover_reason and not guard.has_healthy_resources():
                        failover_reason = (
                            f"ALL_PREFERRED_KEYS_UNHEALTHY:{preferred_provider}"
                        )
                    if not failover_reason and max_estimated_cost_usd is not None:
                        preferred_models = model_priority_map.get(
                            provider_name_lower, []
                        )
                        all_too_expensive = True
                        for model in preferred_models:
                            cost = self._estimate_call_cost(
                                provider_name_lower,
                                model,
                                estimated_input_tokens,
                                output_tokens_for_estimate,
                            )
                            if cost is None or cost <= Decimal(
                                str(max_estimated_cost_usd)
                            ):
                                all_too_expensive = False
                                break
                        if all_too_expensive:
                            failover_reason = f"ALL_PREFERRED_MODELS_EXCEED_COST_CAP_INITIAL_CHECK:{preferred_provider}"
                            context.cost_cap_enforced = True
                            context.cost_cap_skip_reason = (
                                "ALL_MODELS_TOO_EXPENSIVE_FOR_PREFERENCE"
                            )

                if not failover_reason:
                    log.info(f"Attempting preferred provider: {preferred_provider}")
                    provider_sequence_to_attempt = [provider_name_lower]
                else:
                    log.warning(
                        f"{failover_reason}. Falling back to dynamic selection."
                    )

            if provider_sequence_to_attempt is None:
                log.info(
                    "Entering dynamic fallback mode or proceeding with value-driven selection."
                )
                dynamic_provider_priority = self._get_dynamic_provider_priority(
                    model_priority_map
                )
                if not dynamic_provider_priority:
                    failover_reason = failover_reason or "NO_VIABLE_DYNAMIC_PROVIDERS"
                    final_exception = AllProvidersFailedError(
                        errors=[
                            "No viable dynamic providers after health/cost filtering."
                        ]
                    )
                else:
                    provider_sequence_to_attempt = dynamic_provider_priority

            if provider_sequence_to_attempt:
                try:
                    final_response = await self._attempt_request_sequence(
                        context,
                        provider_sequence_to_attempt,
                        model_priority_map,
                        context.final_messages,
                        estimated_input_tokens,
                        output_tokens_for_estimate,
                        **kwargs,
                    )
                    final_outcome = "SUCCESS"
                except ContentPolicyError as e:
                    context.mitigation_attempted = True
                    self._record_lifecycle_event(
                        context,
                        event_topics.PROMPT_HUMANIZATION_ATTEMPT,
                        "WARNING",
                        {
                            "provider": e.provider,
                            "model": e.model,
                            "original_error": str(e.original_error),
                        },
                    )
                    if not self.prompt_rewriter:
                        failover_reason = (
                            failover_reason or "DYNAMIC_MODE_CONTENT_POLICY_NO_REWRITER"
                        )
                        final_exception = AllProvidersFailedError(
                            errors=[str(e), "No prompt rewriter configured."]
                        )
                    else:
                        try:
                            rephrased_messages = await self.prompt_rewriter.rephrase_for_policy_compliance(
                                messages
                            )
                            context.final_messages = rephrased_messages
                            self._record_lifecycle_event(
                                context,
                                event_topics.PROMPT_HUMANIZATION_SUCCESS,
                                "INFO",
                                {},
                            )
                            dynamic_provider_priority = (
                                self._get_dynamic_provider_priority(model_priority_map)
                            )
                            final_response = await self._attempt_request_sequence(
                                context,
                                dynamic_provider_priority,
                                model_priority_map,
                                context.final_messages,
                                estimated_input_tokens,
                                output_tokens_for_estimate,
                                **kwargs,
                            )
                            final_outcome = "MITIGATED_SUCCESS"
                        except (
                            RewriteFailedError,
                            ContentPolicyError,
                            AllProvidersFailedError,
                        ) as rfe:
                            failover_reason = (
                                failover_reason or "DYNAMIC_MODE_MITIGATION_FAILED"
                            )
                            final_exception = AllProvidersFailedError(
                                errors=[
                                    str(e),
                                    f"Mitigation attempt failed: {rfe}",
                                ]
                            )
                except AllProvidersFailedError as e:
                    final_exception = e
                    if preferred_provider and not failover_reason:
                        failover_reason = (
                            f"PREFERRED_PROVIDER_EXHAUSTED:{preferred_provider}"
                        )
                    elif not failover_reason:
                        failover_reason = "ALL_DYNAMIC_PROVIDERS_FAILED"

        except Exception as e:
            final_exception = e
            failover_reason = failover_reason or "UNEXPECTED_ENGINE_FAILURE"
            log.error(f"Unexpected error in FailoverEngine: {e}", exc_info=True)

        finally:
            resilience_score = self._calculate_resilience_score(context, final_outcome)
            if self.bias_ledger:
                ledger_entry = self.bias_ledger.log_request_lifecycle(
                    context=context,
                    initial_selection_mode=initial_selection_mode,
                    final_response=final_response,
                    final_error=final_exception,
                    resilience_score=resilience_score,
                    failover_reason=failover_reason,
                    cost_cap_enforced=context.cost_cap_enforced,
                    cost_cap_skip_reason=context.cost_cap_skip_reason,
                )
                if self.event_bus and ledger_entry:
                    payload = json.loads(ledger_entry.model_dump_json())
                    self.event_bus.publish(
                        event_type=event_topics.LEARNING_FEEDBACK_PUBLISHED,
                        payload=payload,
                    )

        if final_response:
            return final_response

        raise final_exception or AllProvidersFailedError(
            errors=["Request failed due to an unknown issue."]
        )

    async def _attempt_request_sequence(
        self,
        context: RequestContext,
        provider_priority: List[str],
        model_priority_map: Dict[str, List[str]],
        messages: List[ChatMessage],
        estimated_input_tokens: int,
        output_tokens_for_estimate: int,
        **kwargs: Any,
    ) -> CompletionResponse:
        overall_errors = []
        last_provider = None

        for provider_name in provider_priority:
            if last_provider:
                self._record_lifecycle_event(
                    context,
                    event_topics.PROVIDER_FAILOVER,
                    "WARNING",
                    {
                        "from_provider": last_provider,
                        "to_provider": provider_name,
                    },
                )
            last_provider = provider_name

            if provider_name not in self.providers:
                log.warning(
                    f"Provider '{provider_name}' not found in initialized providers. Skipping."
                )
                overall_errors.append(
                    f"Provider '{provider_name}' was skipped: not initialized."
                )
                continue

            model_priority = model_priority_map.get(provider_name)
            if not model_priority:
                log.warning(
                    f"No model priority list found for provider '{provider_name}'. Skipping."
                )
                overall_errors.append(
                    f"Provider '{provider_name}' was skipped: no model list provided."
                )
                continue

            guard = self.guards[provider_name]
            breaker = self.circuit_breakers.get_breaker(provider_name)

            try:
                breaker.check()
            except CircuitBreakerError as e:
                self._record_lifecycle_event(
                    context,
                    event_topics.CIRCUIT_TRIPPED,
                    "CRITICAL",
                    {"provider": provider_name, "error": str(e)},
                )
                overall_errors.append(str(e))
                continue

            last_model = None
            viable_models_for_provider = []
            for model_name in model_priority:
                estimated_cost = self._estimate_call_cost(
                    provider_name,
                    model_name,
                    estimated_input_tokens,
                    output_tokens_for_estimate,
                )
                if (
                    context.max_estimated_cost_usd is not None
                    and estimated_cost is not None
                ):
                    if estimated_cost > Decimal(str(context.max_estimated_cost_usd)):
                        self._record_lifecycle_event(
                            context,
                            event_topics.MODEL_SKIPPED_DUE_TO_COST,
                            "INFO",
                            {
                                "provider": provider_name,
                                "model": model_name,
                                "estimated_cost_usd": str(estimated_cost),
                                "max_cost_cap_usd": str(context.max_estimated_cost_usd),
                            },
                        )
                        context.cost_cap_enforced = True
                        context.cost_cap_skip_reason = "MODEL_TOO_EXPENSIVE"
                        continue
                viable_models_for_provider.append(model_name)

            if not viable_models_for_provider:
                log.warning(
                    f"All models for provider '{provider_name}' were skipped due to cost. Skipping provider."
                )
                overall_errors.append(
                    f"Provider '{provider_name}' was skipped: no viable models after cost checks."
                )
                continue

            for model in viable_models_for_provider:
                if last_model:
                    self._record_lifecycle_event(
                        context,
                        event_topics.MODEL_FAILOVER,
                        "WARNING",
                        {
                            "provider": provider_name,
                            "from_model": last_model,
                            "to_model": model,
                        },
                    )
                last_model = model
                try:
                    response = await self._attempt_model_with_keys(
                        context,
                        provider_name,
                        self.providers[provider_name],
                        guard,
                        breaker,
                        model,
                        messages,
                        **kwargs,
                    )
                    was_half_open = breaker.state == CircuitBreakerState.HALF_OPEN
                    breaker.record_success()
                    if was_half_open:
                        self._record_lifecycle_event(
                            context,
                            event_topics.CIRCUIT_RESET,
                            "INFO",
                            {"provider": provider_name},
                        )
                    return response
                except (
                    AllProviderKeysFailedError,
                    ContentPolicyError,
                ) as model_failure:
                    if isinstance(model_failure, ContentPolicyError):
                        raise model_failure
                    overall_errors.append(str(model_failure))
                    continue

        raise AllProvidersFailedError(errors=overall_errors)

    async def _attempt_model_with_keys(
        self,
        context: RequestContext,
        provider_name: str,
        provider: LLMProvider,
        guard: ResourceGuard,
        breaker: CircuitBreaker,
        model: str,
        messages: List[ChatMessage],
        **kwargs: Any,
    ) -> CompletionResponse:
        key_attempts = 0
        last_key_error = "No resources available."
        while True:
            try:
                with guard.get_resource() as resource:
                    key_attempts += 1
                    context.api_call_count += 1
                    if key_attempts > 1:
                        self._record_lifecycle_event(
                            context,
                            event_topics.API_KEY_ROTATION,
                            "INFO",
                            {
                                "provider": provider_name,
                                "model": model,
                                "new_resource_id": resource.safe_value,
                            },
                        )
                    try:
                        request_kwargs = kwargs.copy()
                        request_kwargs["model"] = model
                        response = await provider.agenerate_completion(
                            messages,
                            api_key_override=resource.value,
                            **request_kwargs,
                        )
                        if response.success:
                            return response

                        last_key_error = (
                            response.error_message or "Provider returned non-success."
                        )
                        raw_exception = (
                            response.metadata.get("raw_exception")
                            if response.metadata
                            else None
                        )

                        # ======================================================================
                        # REFACTOR 2: Use ErrorParser for robust classification
                        # ======================================================================
                        error_details = self.error_parser.classify_error(
                            raw_exception or Exception(last_key_error),
                            provider_name,
                        )

                        guard.penalize_resource(resource.value)

                        # If it's a model-specific issue, stop trying keys and fail over to the next model.
                        if error_details.category == ErrorCategory.MODEL_ISSUE:
                            raise AllProviderKeysFailedError(
                                provider_name, key_attempts, last_key_error
                            )

                        if error_details.category == ErrorCategory.TRANSIENT:
                            breaker.record_failure()
                        continue

                    except Exception as e:
                        last_key_error = f"{type(e).__name__}: {e}"
                        error_details = self.error_parser.classify_error(
                            e, provider_name
                        )

                        if error_details.category == ErrorCategory.CONTENT_POLICY:
                            raise ContentPolicyError(
                                provider=provider_name,
                                model=model,
                                original_error=e,
                            )

                        guard.penalize_resource(resource.value)

                        if error_details.category == ErrorCategory.MODEL_ISSUE:
                            raise AllProviderKeysFailedError(
                                provider_name, key_attempts, last_key_error
                            )

                        if error_details.category == ErrorCategory.TRANSIENT:
                            breaker.record_failure()
                        continue
            except NoResourcesAvailableError:
                error_msg = f"All keys for '{provider_name}' (model: {model}) failed or are in cooldown. Last key error: {last_key_error}"
                raise AllProviderKeysFailedError(provider_name, key_attempts, error_msg)

    # ======================================================================
    # REFACTOR 2: Remove the old, fragile string-matching method.
    # We no longer need this as its logic is replaced by the ErrorParser.
    # ======================================================================
    # def _is_model_specific_error(self, error_message: str) -> bool:
    #     ... (this method is now deleted)
