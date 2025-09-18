# antifragile_framework/providers/api_abstraction_layer.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message author (e.g., 'user', 'assistant', 'system').",
    )
    content: str = Field(..., description="The content of the message.")

    @model_validator(mode="after")
    def check_role_and_content_not_empty(self) -> "ChatMessage":
        if not self.role or not self.role.strip():
            raise ValueError("Message 'role' cannot be empty.")
        if not self.content or not self.content.strip():
            raise ValueError("Message 'content' cannot be empty or just whitespace.")
        return self


class TokenUsage(BaseModel):
    """
    Represents the token usage for a single completion request.
    """

    input_tokens: int = Field(..., description="Number of tokens in the input prompt.")
    output_tokens: int = Field(
        ..., description="Number of tokens in the generated completion."
    )


class CompletionResponse(BaseModel):
    success: bool = Field(...)
    content: Optional[str] = Field(None)
    model_used: Optional[str] = Field(None)
    usage: Optional[TokenUsage] = Field(
        None,
        description="Token usage information for the request, if available.",
    )
    latency_ms: float = Field(...)
    error_message: Optional[str] = Field(None)
    raw_response: Optional[Dict[str, Any]] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)


class LLMProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("Provider configuration must be a dictionary.")
        self.config = config
        self.provider_name = self.get_provider_name()

    def _prepare_provider_messages(
        self, messages: List[ChatMessage]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        system_prompt, other_messages = "", []
        if not messages:
            return system_prompt, other_messages
        message_list = list(messages)
        if message_list and message_list[0].role == "system":
            system_prompt = message_list.pop(0).content
        other_messages = [msg.model_dump() for msg in message_list]
        return system_prompt, other_messages

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

    @abstractmethod
    async def agenerate_completion(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        api_key_override: Optional[str] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        """Generates a text completion, allowing for a per-call API key override."""
        pass
