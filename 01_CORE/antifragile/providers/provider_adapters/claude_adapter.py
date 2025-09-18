# antifragile_framework/providers/provider_adapters/claude_adapter.py

import asyncio
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional

import anthropic
from anthropic import AsyncAnthropic
from antifragile_framework.providers.api_abstraction_layer import (
    ChatMessage,
    CompletionResponse,
    LLMProvider,
    TokenUsage,
)

log = logging.getLogger(__name__)


def _extract_anthropic_usage(response: Any) -> Optional[TokenUsage]:
    """Safely extracts token usage from an Anthropic SDK response object."""
    try:
        if response and response.usage:
            return TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
    except (AttributeError, TypeError) as e:
        log.warning(f"Could not extract token usage from Anthropic response: {e}")
    return None


class ClaudeProvider(LLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.default_model = self.config.get(
            "default_model", "claude-3-5-sonnet-20240620"
        )
        self.primary_key = self.config.get("api_key")
        self.max_retries = self.config.get("max_retries", 2)
        try:
            self.client = AsyncAnthropic(
                api_key=self.primary_key, max_retries=self.max_retries
            )
            log.info(
                f"ClaudeProvider initialized for model '{self.default_model}' with max_retries={self.max_retries}."
            )
        except Exception as e:
            log.error(f"Failed to initialize Anthropic client: {e}")
            raise

    def get_provider_name(self) -> str:
        return "anthropic"

    async def _generate_mock_response(
        self, model_to_use: str, start_time: float
    ) -> CompletionResponse:
        """Generates a realistic, mock completion response for performance testing."""
        # Simulate a small, variable I/O delay (5ms to 15ms)
        await asyncio.sleep(random.uniform(0.005, 0.015))

        # Simulate variable token usage
        input_tokens = random.randint(50, 250)
        output_tokens = random.randint(100, 500)
        usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)

        # Create a unique response content
        content = f"Mocked Anthropic response for model {model_to_use}. UUID: {uuid.uuid4().hex[:12]}"

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        metadata = {"provider_name": self.get_provider_name(), "mock": True}

        return CompletionResponse(
            success=True,
            content=content,
            model_used=model_to_use,
            usage=usage,
            latency_ms=latency_ms,
            error_message=None,
            raw_response={"mock_reason": "PERFORMANCE_TEST_MODE"},
            metadata=metadata,
        )

    async def agenerate_completion(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key_override: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        # Start timing framework processing
        framework_start = time.perf_counter()
        model_to_use = kwargs.get("model", self.default_model)

        # Performance test mocking logic (keep existing mock logic)
        if os.getenv("PERFORMANCE_TEST_MODE", "False").lower() == "true":
            return await self._generate_mock_response(model_to_use, framework_start)

        # Framework processing: client setup
        usage = None
        if api_key_override and api_key_override != self.primary_key:
            try:
                client_to_use = AsyncAnthropic(
                    api_key=api_key_override, max_retries=self.max_retries
                )
                log.debug("Created temporary Anthropic client for key override.")
            except Exception as e:
                error_msg = f"Failed to initialize temporary Anthropic client with override key: {e}"
                log.error(error_msg)
                framework_latency = (time.perf_counter() - framework_start) * 1000
                return CompletionResponse(
                    success=False,
                    content=None,
                    model_used=model_to_use,
                    latency_ms=framework_latency,
                    error_message=error_msg,
                )
        else:
            client_to_use = self.client

        # Framework processing: message preparation
        system_prompt, user_assistant_messages = self._prepare_provider_messages(
            messages
        )
        if not user_assistant_messages:
            framework_latency = (time.perf_counter() - framework_start) * 1000
            return CompletionResponse(
                success=False,
                latency_ms=framework_latency,
                error_message="Cannot make a provider call with no user/assistant messages.",
            )

        # End framework preprocessing timing
        api_call_start = time.perf_counter()

        try:
            # 🚨 EXTERNAL API CALL - This time should NOT count as framework overhead
            response = await client_to_use.messages.create(
                model=model_to_use,
                system=system_prompt,
                messages=user_assistant_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs.get("provider_specific_params", {}),
            )

            # End external API timing, start framework post-processing
            api_call_end = time.perf_counter()

            # Framework processing: response validation and formatting
            if (
                not response.content
                or not isinstance(response.content, list)
                or len(response.content) == 0
                or not response.content[0].text
            ):
                framework_end = time.perf_counter()
                framework_latency = (
                    (api_call_start - framework_start) + (framework_end - api_call_end)
                ) * 1000
                return CompletionResponse(
                    success=False,
                    content=None,
                    model_used=model_to_use,
                    latency_ms=framework_latency,
                    error_message="Anthropic response was successful but contained no valid content.",
                    raw_response=response.model_dump(),
                )

            # Framework processing: extract data
            content = response.content[0].text
            usage = _extract_anthropic_usage(response)
            metadata = {"provider_name": self.get_provider_name()}

            # End framework processing
            framework_end = time.perf_counter()

            # Calculate ONLY framework overhead (excluding external API time)
            framework_latency = (
                (api_call_start - framework_start) + (framework_end - api_call_end)
            ) * 1000

            return CompletionResponse(
                success=True,
                content=content,
                model_used=response.model,
                usage=usage,
                latency_ms=framework_latency,
                error_message=None,
                raw_response=response.model_dump(),
                metadata=metadata,
            )

        except anthropic.APIError as e:
            api_call_end = time.perf_counter()
            error_body = getattr(e, "body", {})
            error_details = (
                error_body.get("error", {}) if isinstance(error_body, dict) else {}
            )
            error_message = (
                error_details.get("message", "An unknown API error occurred.")
                if isinstance(error_details, dict)
                else str(e)
            )
            error_type = (
                error_details.get("type", "unknown_error")
                if isinstance(error_details, dict)
                else "unknown_error"
            )
            error_msg = f"Anthropic API Error ({error_type}): {error_message}"
        except Exception as e:
            api_call_end = time.perf_counter()
            error_msg = f"An unexpected error occurred with Anthropic: {str(e)}"
            log.exception("Unexpected Anthropic Error")

        # Calculate framework overhead for error cases
        framework_end = time.perf_counter()
        framework_latency = (
            (api_call_start - framework_start) + (framework_end - api_call_end)
        ) * 1000

        log.error(
            f"Failed Anthropic call for model {model_to_use}. Reason: {error_msg}"
        )
        return CompletionResponse(
            success=False,
            content=None,
            model_used=model_to_use,
            latency_ms=framework_latency,
            error_message=error_msg,
            raw_response=None,
        )
