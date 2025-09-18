# antifragile_framework/resilience/llm_based_rewriter.py

import copy
from typing import List, Optional

from antifragile_framework.core.exceptions import RewriteFailedError
from antifragile_framework.providers.api_abstraction_layer import (
    ChatMessage,
    LLMProvider,
)

from .prompt_rewriter import PromptRewriter

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant specializing in content safety and compliance. "
    "Your task is to rephrase the user's text to be as neutral, professional, "
    "and compliant with standard content policies as possible, while preserving the core "
    "intent and meaning. Do not add any commentary, disclaimers, or introductory phrases. "
    "Only provide the rephrased text directly."
)


class LLMBasedRewriter(PromptRewriter):
    """
    A concrete implementation of PromptRewriter that uses an LLM to rephrase prompts.
    """

    def __init__(
        self,
        llm_client: LLMProvider,
        model: str = "gpt-4o",
        system_prompt: Optional[str] = None,
    ):
        """
        Initializes the LLMBasedRewriter.

        Args:
            llm_client: An instance of a class that conforms to the LLMProvider interface.
            model: The specific model to use for the rephrasing task.
            system_prompt: An optional custom system prompt to guide the rephrasing.
        """
        if not isinstance(llm_client, LLMProvider):
            raise TypeError(
                "llm_client must be an instance of a class that implements LLMProvider."
            )
        self.llm_client = llm_client
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def _create_meta_prompt(
        self, original_messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        """
        Creates a "meta-prompt" to instruct an LLM on how to rephrase the user's content.
        """
        user_content_to_rephrase = ""
        for msg in reversed(original_messages):
            if msg.role == "user":
                user_content_to_rephrase = msg.content
                break

        if not user_content_to_rephrase:
            raise RewriteFailedError(
                "No user content found in the messages to rephrase."
            )

        return [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=user_content_to_rephrase),
        ]

    def _replace_last_user_message(
        self, original_messages: List[ChatMessage], new_content: str
    ) -> List[ChatMessage]:
        """
        Creates a new list of messages with the content of the last user message updated.
        """
        messages_copy = copy.deepcopy(original_messages)

        for i in range(len(messages_copy) - 1, -1, -1):
            if messages_copy[i].role == "user":
                messages_copy[i].content = new_content
                return messages_copy

        return messages_copy

    async def rephrase_for_policy_compliance(
        self, messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        """
        Rephrases the last user message in a list of messages to be more policy-compliant.
        """
        meta_prompt = self._create_meta_prompt(messages)

        try:
            response = await self.llm_client.agenerate_completion(
                messages=meta_prompt,
                model=self.model,
                max_tokens=1000,
                temperature=0.3,
            )

            if not response.success or not response.content:
                error_msg = response.error_message or "LLM returned no content."
                raise RewriteFailedError(
                    f"LLM API call for rephrasing failed: {error_msg}"
                )

            rephrased_content = response.content.strip()
            return self._replace_last_user_message(messages, rephrased_content)

        except Exception as e:
            if isinstance(e, RewriteFailedError):
                raise
            raise RewriteFailedError(
                f"An unexpected error occurred during rephrasing: {e}"
            ) from e
