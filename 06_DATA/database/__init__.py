"""
MarketPulse Database Package
PostgreSQL + TimescaleDB Database Layer

Expert Recommendation: "One is useless without the other - tackle data and database
as one cohesive unit: The System's Memory and Senses."

Location: #06_DATA/database/__init__.py
"""

from .db_setup import DatabaseSetup, Base
from .models import (
    MarketData,
    Trade,
    Signal,
    Portfolio,
    PortfolioSnapshot,
    AlphaModelPrediction,
    TechnicalIndicator,
    RiskMetric,
    SystemLog,
    ConfigParameter
)

# Version
__version__ = "1.0.0"

# Convenient imports
__all__ = [
    # Setup and base
    'DatabaseSetup',
    'Base',

    # Core trading models
    'MarketData',
    'Trade',
    'Signal',
    'Portfolio',
    'PortfolioSnapshot',

    # ML and analytics models
    'AlphaModelPrediction',
    'TechnicalIndicator',
    'RiskMetric',

    # System models
    'SystemLog',
    'ConfigParameter'
]


# Quick setup function
def setup_database(config_path=None):
    """
    Quick database setup function

    Args:
        config_path: Path to database config file

    Returns:
        DatabaseSetup: Configured database setup instance
    """
    db_setup = DatabaseSetup(config_path)

    if db_setup.setup_complete_database():
        return db_setup
    else:
        raise RuntimeError("Database setup failed")


# Connection helper
def get_database_session(config_path=None):
    """
    Get a database session for ORM operations

    Args:
        config_path: Path to database config file

    Returns:
        Session: SQLAlchemy session
    """
    db_setup = setup_database(config_path)
    return db_setup.get_session()


# Engine helper
def get_database_engine(config_path=None):
    """
    Get database engine for direct SQL operations

    Args:
        config_path: Path to database config file

    Returns:
        Engine: SQLAlchemy engine
    """
    db_setup = setup_database(config_path)
    return db_setup.get_engine()