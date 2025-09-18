# antifragile_framework/core/provider_ranking_engine.py

import threading
from typing import Dict, List
from telemetry.core_logger import UniversalEventSchema, core_logger


class ProviderRankingEngine:
    """
    A thread-safe class that maintains a real-time performance ranking of providers
    based on an Exponential Moving Average (EMA) of their ResilienceScores.
    """

    def __init__(
        self,
        smoothing_factor: float = 0.2,
        default_score: float = 0.75,
        min_requests_threshold: int = 5,
    ):
        """
        Initializes the ProviderRankingEngine.

        Args:
            smoothing_factor (float): The alpha value for the EMA calculation.
                                      Higher values react faster to recent data. Must be between 0 and 1.
            default_score (float): A default score assigned to new providers or those
                                   below the request threshold to handle the "cold start" problem.
            min_requests_threshold (int): The number of requests a provider must have
                                          before its real EMA score is used for ranking.
        """
        if not (0.0 < smoothing_factor <= 1.0):
            raise ValueError("Smoothing factor must be between 0.0 and 1.0.")

        self._alpha = smoothing_factor
        self._default_score = default_score
        self._min_requests = min_requests_threshold

        self._lock = threading.Lock()
        self._provider_emas: Dict[str, float] = {}
        self._request_counts: Dict[str, int] = {}
        self.logger = core_logger

    def update_provider_score(self, provider_name: str, resilience_score: float):
        """
        Updates a provider's performance score using the latest ResilienceScore.
        This method is thread-safe.

        Args:
            provider_name (str): The name of the provider to update.
            resilience_score (float): The ResilienceScore from the completed request.
        """
        if not (0.0 <= resilience_score <= 1.0):
            self.logger.log(
                UniversalEventSchema(
                    event_type="learning.score.invalid",
                    event_source=self.__class__.__name__,
                    severity="WARNING",
                    payload={
                        "provider": provider_name,
                        "received_score": resilience_score,
                        "reason": "Score out of 0.0-1.0 range.",
                    },
                )
            )
            return

        with self._lock:
            current_count = self._request_counts.get(provider_name, 0)

            if provider_name in self._provider_emas:
                # Standard EMA update
                previous_ema = self._provider_emas[provider_name]
                new_ema = (self._alpha * resilience_score) + (
                    1 - self._alpha
                ) * previous_ema
                self._provider_emas[provider_name] = new_ema
            else:
                # First data point for this provider (cold start)
                new_ema = resilience_score
                self._provider_emas[provider_name] = new_ema

            self._request_counts[provider_name] = current_count + 1

            self.logger.log(
                UniversalEventSchema(
                    event_type="learning.score.update",
                    event_source=self.__class__.__name__,
                    severity="DEBUG",
                    payload={
                        "provider": provider_name,
                        "new_ema_score": new_ema,
                        "request_count": self._request_counts[provider_name],
                    },
                )
            )

    def get_ranked_providers(self) -> List[str]:
        """
        Returns a list of provider names, sorted from highest score to lowest.
        Providers with too few requests are ranked by the default score.
        This method is thread-safe.

        Returns:
            List[str]: A sorted list of provider names.
        """
        with self._lock:
            # Create a list of all known providers to ensure none are missed
            all_providers = list(self._provider_emas.keys())

            def get_score(provider_name: str) -> float:
                # Use the real EMA if we have enough data, otherwise use the default score
                if self._request_counts.get(provider_name, 0) >= self._min_requests:
                    return self._provider_emas.get(provider_name, self._default_score)
                return self._default_score

            sorted_providers = sorted(
                all_providers,
                key=get_score,
                reverse=True,  # Highest score first
            )
            return sorted_providers

    def get_provider_scores(self) -> Dict[str, Dict]:
        """
        Returns a dictionary of all providers and their current scores and request counts for observability.

        Returns:
            Dict[str, Dict]: A dictionary containing detailed stats for each provider.
        """
        with self._lock:
            return {
                provider: {
                    "ema_score": self._provider_emas.get(provider),
                    "request_count": self._request_counts.get(provider, 0),
                }
                for provider in self._provider_emas
            }
