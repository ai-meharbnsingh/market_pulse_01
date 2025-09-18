# antifragile_framework/config/config_loader.py

import json
import logging
import os
from typing import Any, Dict, Optional

import yaml
from antifragile_framework.config.schemas import ProviderProfiles
from pydantic import ValidationError

log = logging.getLogger(__name__)


def _get_config_path(default_filename: str, provided_path: Optional[str] = None) -> str:
    """Helper function to determine the absolute path for a configuration file."""
    if provided_path:
        return provided_path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes /config is a direct child of /antifragile_framework, which is in the project root
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    return os.path.join(
        project_root, "antifragile_framework", "config", default_filename
    )


def load_resilience_config(
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads the resilience framework configuration from a YAML file.

    Args:
        config_path: Optional path to the resilience config YAML file.
                     If None, it defaults to 'antifragile_framework/config/resilience_config.yaml'.

    Returns:
        A dictionary containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    if config_path is None:
        # Check for fast config first if FAST_DEMO_MODE is set
        if os.getenv("FAST_DEMO_MODE") == "true":
            fast_config_path = _get_config_path("resilience_config_fast.yaml")
            if os.path.exists(fast_config_path):
                config_path = fast_config_path

    abs_config_path = _get_config_path("resilience_config.yaml", config_path)

    if not os.path.exists(abs_config_path):
        log.error(f"Resilience config file not found at: {abs_config_path}")
        raise FileNotFoundError(
            f"Resilience config file not found at: {abs_config_path}"
        )

    try:
        with open(abs_config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            log.info(f"Successfully loaded resilience config from {abs_config_path}")
            return config
    except yaml.YAMLError as e:
        log.error(f"Error parsing resilience config YAML file {abs_config_path}: {e}")
        raise
    except Exception as e:
        log.error(
            f"An unexpected error occurred while loading resilience config from {abs_config_path}: {e}"
        )
        raise


def load_provider_profiles(
    config_path: Optional[str] = None,
) -> ProviderProfiles:
    """
    Loads and validates the provider profiles configuration from a JSON file.

    This function enforces a strict schema using Pydantic, ensuring that the
    application does not start with a malformed or invalid cost configuration.

    Args:
        config_path: Optional path to the provider profiles JSON file.
                     If None, it defaults to 'antifragile_framework/config/provider_profiles.json'.

    Returns:
        A validated ProviderProfiles Pydantic object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the JSON is malformed or fails schema validation.
    """
    abs_config_path = _get_config_path("provider_profiles.json", config_path)

    if not os.path.exists(abs_config_path):
        log.error(f"Provider profiles file not found at: {abs_config_path}")
        raise FileNotFoundError(
            f"Provider profiles file not found at: {abs_config_path}"
        )

    try:
        with open(abs_config_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        validated_profiles = ProviderProfiles.model_validate(data)
        log.info(
            f"Successfully loaded and validated provider profiles from {abs_config_path}"
        )
        return validated_profiles

    except json.JSONDecodeError as e:
        log.error(f"Error parsing provider profiles JSON file {abs_config_path}: {e}")
        raise ValueError(
            f"Invalid JSON in provider profiles file: {abs_config_path}"
        ) from e

    except ValidationError as e:
        log.error(
            f"Schema validation failed for provider profiles file {abs_config_path}. Errors: {e}"
        )
        raise ValueError(
            "Provider profiles configuration is invalid. Please check the format."
        ) from e

    except Exception as e:
        log.error(
            f"An unexpected error occurred while loading provider profiles from {abs_config_path}: {e}"
        )
        raise


if __name__ == "__main__":
    # Example usage for testing purposes
    logging.basicConfig(level=logging.INFO)
    try:
        # Test resilience config loader
        resilience_config = load_resilience_config()
        print("\n--- Loaded Resilience Config ---")
        print(yaml.dump(resilience_config, indent=2))
        print("------------------------------")
    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"Resilience config test failed: {e}")

    try:
        # Test provider profiles loader
        profiles_config = load_provider_profiles()
        print("\n--- Loaded Provider Profiles ---")
        print(profiles_config.model_dump_json(indent=2))
        print("--------------------------------")
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Provider profiles test failed: {e}")
