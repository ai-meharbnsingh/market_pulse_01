# antifragile_framework/core/exceptions.py


class AntifragileError(Exception):
    """Base exception for all custom errors in the antifragile framework."""

    pass


class NoResourcesAvailableError(AntifragileError):
    """
    Raised by the ResourceGuard when no healthy and available resources
    can be found for a given provider.
    """

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"No healthy and available resources for provider '{provider}'."
        )


class AllProviderKeysFailedError(AntifragileError):
    """
    Raised by the FailoverEngine when all attempted keys for a specific
    provider/model combination have failed during a request cycle.
    """

    def __init__(self, provider: str, attempts: int, last_error: str = "N/A"):
        self.provider = provider
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"All {attempts} attempted key(s) for provider '{provider}' failed in succession. Last error: {last_error}"
        )


class AllProvidersFailedError(AntifragileError):
    """
    Raised by the FailoverEngine when all configured providers in the priority list
    have failed, either through key exhaustion or an open circuit breaker.
    This represents a total system failure for the request.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        error_details = "\n - ".join(errors)
        super().__init__(
            f"All configured providers failed. Failure chain:\n - {error_details}"
        )


class RewriteFailedError(AntifragileError):
    """
    Raised by a PromptRewriter when it fails to rephrase a prompt,
    either due to an API error or inability to parse the response.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Prompt rewriting failed: {reason}")


class ContentPolicyError(AntifragileError):
    """
    Raised when a provider blocks a request due to a content policy violation.
    This is a special error that can trigger mitigation strategies.
    """

    def __init__(self, provider: str, model: str, original_error: Exception):
        self.provider = provider
        self.model = model
        self.original_error = original_error
        super().__init__(
            f"Provider '{provider}' blocked request to model '{model}' due to content policy. Original error: {original_error}"
        )
