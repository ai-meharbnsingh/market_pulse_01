# antifragile_framework/config/schemas.py

from decimal import Decimal
from typing import Dict

from pydantic import BaseModel, Field


class CostProfile(BaseModel):
    """
    Defines the cost structure for a single AI model.
    Costs are stored as decimals for financial precision.
    """

    input_cpm: Decimal = Field(
        ..., gt=Decimal("0"), description="Cost per million input tokens."
    )
    output_cpm: Decimal = Field(
        ..., gt=Decimal("0"), description="Cost per million output tokens."
    )


class ProviderProfiles(BaseModel):
    """
    Defines the overall schema for the provider_profiles.json configuration file.
    It includes metadata and a nested dictionary of provider and model cost profiles.
    """

    schema_version: str = Field(
        ..., description="The version of the configuration schema."
    )
    last_updated_utc: str = Field(
        ...,
        description="The ISO 8601 timestamp of when the configuration was last updated.",
    )
    profiles: Dict[str, Dict[str, CostProfile]] = Field(
        ...,
        description="A nested dictionary of provider and model cost profiles.",
    )
