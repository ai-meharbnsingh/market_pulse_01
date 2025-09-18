# antifragile_framework/core/online_learning_subscriber.py

import sys
import logging
from datetime import datetime, timezone

# Standardized path setup relative to the current file
# Assuming current file is in PROJECT_ROOT/01_Framework_Core/antifragile_framework/core/
from pathlib import Path
from typing import Any, Dict

from pydantic import ValidationError

CURRENT_DIR = Path(__file__).parent
FRAMEWORK_ROOT = CURRENT_DIR.parent.parent  # Points to antifragile_framework/
TELEMETRY_PATH = (
    CURRENT_DIR.parent.parent.parent.parent / "telemetry"
)  # Points to 01_Framework_Core/telemetry/
RESILIENCE_PATH = (
    FRAMEWORK_ROOT / "resilience"
)  # Points to antifragile_framework/resilience/


sys.path.insert(0, str(FRAMEWORK_ROOT))
sys.path.insert(0, str(TELEMETRY_PATH))
sys.path.insert(0, str(RESILIENCE_PATH))
sys.path.insert(
    0, str(CURRENT_DIR)
)  # For sibling core modules like provider_ranking_engine

# Import from standardized locations
try:
    from antifragile_framework.core.provider_ranking_engine import (
        ProviderRankingEngine,
    )
    from antifragile_framework.resilience.bias_ledger import BiasLedgerEntry
    from telemetry.core_logger import UniversalEventSchema, core_logger

    # No need to import event_topics here, as the payload should contain enough context
except ImportError as e:
    logging.critical(
        f"CRITICAL ERROR: Failed to import core dependencies for OnlineLearningSubscriber: {e}. "
        f"Online learning will not function correctly.",
        exc_info=True,
    )

    # Define minimal mocks to prevent hard crashes
    class ProviderRankingEngine:
        def __init__(self, *args, **kwargs):
            pass

        def update_provider_score(self, *args, **kwargs):
            logging.warning("Mock ProviderRankingEngine used.")

    class BiasLedgerEntry:
        def __init__(self, **kwargs):
            pass

        def model_validate(cls, data):
            raise ValidationError("Mock validation error")

    class MockCoreLogger:
        def log(self, event):
            logging.info(f"MockCoreLogger: {event.get('event_type')}")

    core_logger = MockCoreLogger()

    class UniversalEventSchema:
        def __init__(self, **kwargs):
            pass

        def model_dump(self):
            return {}


log = logging.getLogger(__name__)


class OnlineLearningSubscriber:
    """
    A subscriber that listens for learning feedback events and updates the
    ProviderRankingEngine with new performance data.
    """

    def __init__(self, ranking_engine: ProviderRankingEngine):
        """
        Initializes the subscriber with a reference to the ranking engine.

        Args:
            ranking_engine (ProviderRankingEngine): The stateful engine that maintains provider scores.
        """
        if not isinstance(ranking_engine, ProviderRankingEngine):
            raise TypeError(
                "ranking_engine must be an instance of ProviderRankingEngine"
            )
        self.ranking_engine = ranking_engine
        self.logger = core_logger
        log.info("OnlineLearningSubscriber initialized.")

    def handle_event(self, event_data: Dict[str, Any]):  # CORRECTED: Removed 'async'
        """
        Handles incoming events from the EventBus.

        This method safely parses the event data into a BiasLedgerEntry,
        extracts the necessary performance metrics, and updates the ranking engine.

        Args:
            event_data (Dict[str, Any]): The payload from the LEARNING_FEEDBACK_PUBLISHED event.
                                         Expected to be a dictionary representation of BiasLedgerEntry.
        """
        try:
            # Safely parse the dictionary back into a Pydantic model for type safety
            # The event_data itself is the BiasLedgerEntry's payload, not wrapped in UniversalEventSchema.
            ledger_entry = BiasLedgerEntry.model_validate(event_data)

            provider = ledger_entry.final_provider
            score = ledger_entry.resilience_score

            # We only learn from requests that actually used a provider and have a score
            if provider is not None and score is not None:
                self.ranking_engine.update_provider_score(
                    provider_name=provider, resilience_score=score
                )
                self.logger.log(
                    UniversalEventSchema(
                        event_type="learning.feedback.processed",
                        event_topic="learning.feedback",
                        event_source=self.__class__.__name__,
                        severity="DEBUG",
                        payload={
                            "provider": provider,
                            "resilience_score": round(score, 4),
                            "request_id": ledger_entry.request_id,
                        },
                    ).model_dump()
                )
            else:
                self.logger.log(
                    UniversalEventSchema(
                        event_type="learning.event.skip",
                        event_topic="learning.feedback",
                        event_source=self.__class__.__name__,
                        severity="DEBUG",
                        payload={
                            "reason": "Ledger entry missing final_provider or resilience_score for learning.",
                            "request_id": ledger_entry.request_id,
                        },
                    ).model_dump()
                )

        except ValidationError as e:
            # This handles cases where the event data is malformed against BiasLedgerEntry schema
            self.logger.log(
                UniversalEventSchema(
                    event_type="learning.event.parse_error",
                    event_topic="learning.feedback",
                    event_source=self.__class__.__name__,
                    severity="ERROR",
                    payload={
                        "error": str(e),
                        "malformed_data_preview": str(event_data)[:500],
                    },
                ).model_dump()
            )
        except Exception as e:
            # Catch any other unexpected errors during event handling
            self.logger.log(
                UniversalEventSchema(
                    event_type="learning.event.handler_error",
                    event_topic="learning.feedback",
                    event_source=self.__class__.__name__,
                    severity="CRITICAL",
                    payload={
                        "error": str(e),
                        "event_data_preview": str(event_data)[:500],
                    },
                ).model_dump()
            )


# Example Usage (for testing this module in isolation)
async def main():
    print("Starting OnlineLearningSubscriber demo...")

    # Mock ProviderRankingEngine
    class MockProviderRankingEngine(ProviderRankingEngine):
        def __init__(self):
            super().__init__()
            self.updates_received = []

        def update_provider_score(self, provider_name: str, resilience_score: float):
            super().update_provider_score(provider_name, resilience_score)
            self.updates_received.append((provider_name, resilience_score))
            print(
                f"Mock Ranking Engine received update: {provider_name} with score {resilience_score:.2f}"
            )

    mock_ranking_engine = MockProviderRankingEngine()
    subscriber = OnlineLearningSubscriber(mock_ranking_engine)

    # Simulate a valid LEARNING_FEEDBACK_PUBLISHED event payload (which is a BiasLedgerEntry dict)
    valid_event_payload = {
        "request_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": 4,
        "initial_prompt_hash": "mockhash",
        "initial_prompt_preview": "test prompt",
        "final_prompt_preview": None,
        "final_response_preview": "test response",
        "outcome": "SUCCESS",
        "total_latency_ms": 150.5,
        "total_api_calls": 1,
        "final_provider": "openai",
        "final_model": "gpt-4o",
        "resilience_events": [],
        "mitigation_attempted": False,
        "mitigation_succeeded": None,
        "resilience_score": 0.95,
        "preferred_provider_requested": None,
        "initial_selection_mode": "VALUE_DRIVEN",
        "failover_reason": None,
        "cost_cap_enforced": False,
        "cost_cap_skip_reason": None,
        "input_tokens": 10,
        "output_tokens": 20,
        "estimated_cost_usd": 0.001,
    }

    # Simulate an event with missing score
    missing_score_payload = valid_event_payload.copy()
    missing_score_payload["request_id"] = str(uuid.uuid4())
    missing_score_payload["resilience_score"] = None
    missing_score_payload["final_provider"] = "google"

    # Simulate an invalid payload (missing required field)
    invalid_payload = {
        "request_id": str(uuid.uuid4()),
        "timestamp_utc": "invalid",
    }

    print("\n--- Processing Valid Event ---")
    subscriber.handle_event(valid_event_payload)
    print(f"Updates received by mock engine: {mock_ranking_engine.updates_received}")

    print("\n--- Processing Event with Missing Score ---")
    subscriber.handle_event(missing_score_payload)
    print(
        f"Updates received by mock engine: {mock_ranking_engine.updates_received}"
    )  # Should be no new update

    print("\n--- Processing Invalid Event ---")
    subscriber.handle_event(invalid_payload)  # This should log an error

    print("\n--- Current Provider Rankings from Mock Engine ---")
    print(mock_ranking_engine.get_provider_scores())

    print("\nOnlineLearningSubscriber demo completed.")


if __name__ == "__main__":
    import asyncio  # Ensure asyncio is imported for main() to run
    import uuid  # Ensure uuid is imported for main() to run

    asyncio.run(main())
