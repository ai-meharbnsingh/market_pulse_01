# 06_DATA/database_config.py
"""
Centralized Database Configuration
Single source of truth for all database paths

This module provides consistent database paths across the entire application.
Import this module instead of hardcoding database paths.

Location: #06_DATA/database_config.py
"""

from pathlib import Path
import os

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class DatabaseConfig:
    """Centralized database configuration"""

    # Production database (live trading)
    PRODUCTION_DB = PROJECT_ROOT / "marketpulse_production.db"

    # Training database (ML training with 170+ Indian stocks)
    TRAINING_DB = PROJECT_ROOT / "06_DATA" / "marketpulse_training.db"

    # Performance database (analytics and monitoring)
    PERFORMANCE_DB = PROJECT_ROOT / "10_DATA_STORAGE" / "marketpulse_performance.db"

    @classmethod
    def get_production_db_path(cls) -> str:
        """Get production database path"""
        return str(cls.PRODUCTION_DB)

    @classmethod
    def get_training_db_path(cls) -> str:
        """Get training database path"""
        return str(cls.TRAINING_DB)

    @classmethod
    def get_performance_db_path(cls) -> str:
        """Get performance database path"""
        return str(cls.PERFORMANCE_DB)

    @classmethod
    def ensure_db_directories(cls):
        """Ensure all database directories exist"""
        cls.TRAINING_DB.parent.mkdir(exist_ok=True)
        cls.PERFORMANCE_DB.parent.mkdir(exist_ok=True)

    @classmethod
    def get_db_for_purpose(cls, purpose: str) -> str:
        """Get database path based on purpose"""
        purpose_map = {
            'trading': cls.get_production_db_path(),
            'live': cls.get_production_db_path(),
            'production': cls.get_production_db_path(),
            'quotes': cls.get_production_db_path(),
            'orders': cls.get_production_db_path(),

            'training': cls.get_training_db_path(),
            'ml': cls.get_training_db_path(),
            'models': cls.get_training_db_path(),
            'features': cls.get_training_db_path(),
            'backtest': cls.get_training_db_path(),

            'performance': cls.get_performance_db_path(),
            'analytics': cls.get_performance_db_path(),
            'monitoring': cls.get_performance_db_path(),
            'errors': cls.get_performance_db_path(),
            'alerts': cls.get_performance_db_path(),
        }

        return purpose_map.get(purpose.lower(), cls.get_production_db_path())


# Convenience constants for backward compatibility
PRODUCTION_DB_PATH = DatabaseConfig.get_production_db_path()
TRAINING_DB_PATH = DatabaseConfig.get_training_db_path()
PERFORMANCE_DB_PATH = DatabaseConfig.get_performance_db_path()

# Legacy aliases (will be deprecated)
MARKETPULSE_DB = PRODUCTION_DB_PATH
DEFAULT_DB_PATH = PRODUCTION_DB_PATH

# Ensure directories exist when module is imported
DatabaseConfig.ensure_db_directories()
