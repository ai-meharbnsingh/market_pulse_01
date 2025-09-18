# antifragile_framework/resilience/prompt_rewriter.py

from abc import ABC, abstractmethod
from typing import List

from antifragile_framework.providers.api_abstraction_layer import ChatMessage


class PromptRewriter(ABC):
    """
    Abstract Base Class for prompt rewriting components.

    This class defines the contract for any component that can take a list of
    ChatMessages and rephrase them, typically for a specific purpose like
    bypassing content policy filters or improving clarity.
    """

    @abstractmethod
    async def rephrase_for_policy_compliance(
        self, messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        """
        Takes a list of messages, identifies the user-facing content,
        and rephrases it to be more neutral and compliant with content policies.

        Args:
            messages: The original list of ChatMessage objects.

        Returns:
            A new list of ChatMessage objects with the user content rephrased.
            If rephrasing fails or is not possible, it should raise an exception.
        """
        pass
