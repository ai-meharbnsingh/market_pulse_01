# antifragile_framework/providers/provider_registry.py

from typing import Dict, Type

from .api_abstraction_layer import LLMProvider
from .provider_adapters.claude_adapter import ClaudeProvider
from .provider_adapters.gemini_adapter import GeminiProvider
from .provider_adapters.openai_adapter import OpenAIProvider


class ProviderRegistry:
    """A registry to manage and access LLM provider classes."""

    def __init__(self):
        self._providers: Dict[str, Type[LLMProvider]] = {}

    def register_provider(self, name: str, provider_class: Type[LLMProvider]):
        """Registers a provider class with a given name."""
        if not issubclass(provider_class, LLMProvider):
            raise TypeError(
                f"Provider class {provider_class.__name__} must be a subclass of LLMProvider."
            )
        self._providers[name.lower()] = provider_class

    def get_provider_class(self, name: str) -> Type[LLMProvider]:
        """Retrieves a provider class by name."""
        provider_class = self._providers.get(name.lower())
        if not provider_class:
            raise KeyError(f"Provider '{name}' not found in registry.")
        return provider_class

    def list_providers(self) -> Dict[str, Type[LLMProvider]]:
        """Returns a copy of the current providers in the registry."""
        return self._providers.copy()


def get_default_provider_registry() -> ProviderRegistry:
    """
    Creates and returns a ProviderRegistry with the default built-in providers.
    This makes the system extensible, as users can create their own registry
    with custom providers.
    """
    registry = ProviderRegistry()
    registry.register_provider("openai", OpenAIProvider)
    registry.register_provider("anthropic", ClaudeProvider)
    registry.register_provider("google_gemini", GeminiProvider)
    return registry
