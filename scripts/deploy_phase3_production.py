# scripts/deploy_phase3_production.py
"""
Phase 3 Production Deployment Helper
Resolves compatibility issues and prepares system for production

Usage: python scripts/deploy_phase3_production.py
Location: #scripts/deploy_phase3_production.py
"""

import sqlite3
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_database_schema(db_path="marketpulse_production.db"):
    """Update database schema for Phase 3 compatibility"""

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check and add missing columns
        schema_updates = [
            # Add executed_price column to trades if it doesn't exist
            """
            SELECT sql FROM sqlite_master WHERE type='table' AND name='trades'
            """,
            # Add day_pnl to portfolios if it doesn't exist
            """
            SELECT sql FROM sqlite_master WHERE type='table' AND name='portfolios'
            """
        ]

        # Check trades table
        cursor.execute("PRAGMA table_info(trades)")
        trades_columns = [row[1] for row in cursor.fetchall()]

        if 'executed_price' not in trades_columns:
            cursor.execute("ALTER TABLE trades ADD COLUMN executed_price REAL DEFAULT 0.0")
            logger.info("Added executed_price column to trades table")

        # Check portfolios table
        cursor.execute("PRAGMA table_info(portfolios)")
        portfolio_columns = [row[1] for row in cursor.fetchall()]

        if 'day_pnl' not in portfolio_columns:
            cursor.execute("ALTER TABLE portfolios ADD COLUMN day_pnl REAL DEFAULT 0.0")
            logger.info("Added day_pnl column to portfolios table")

        # Add performance indexes
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_portfolios_created_at ON portfolios(created_at)"
        ]

        for query in index_queries:
            cursor.execute(query)

        conn.commit()
        conn.close()

        logger.info(f"Database schema updated successfully: {db_path}")
        return True

    except Exception as e:
        logger.error(f"Database schema update failed: {e}")
        return False


def setup_environment_variables():
    """Setup environment variables for production"""

    env_vars = {
        'ALPHA_VANTAGE_API_KEY': 'Get from https://www.alphavantage.co/support/#api-key',
        'FINNHUB_API_KEY': 'Get from https://finnhub.io/register',
        'MARKETPULSE_DB_PATH': 'marketpulse_production.db',
        'MARKETPULSE_TRADING_MODE': 'PAPER'  # Change to LIVE for production
    }

    logger.info("Environment Variables Setup:")
    for var, description in env_vars.items():
        current_value = os.getenv(var)
        status = "SET" if current_value else "NOT SET"
        logger.info(f"  {var}: {status}")
        if not current_value:
            logger.info(f"    → {description}")

    return env_vars


def validate_components():
    """Validate all Phase 3 components"""

    components = {
        '06_DATA/live_market_data_fetcher.py': 'Live Market Data Fetcher',
        '05_EXECUTION/live_trading_engine.py': 'Live Trading Engine',
        '04_RISK/advanced_risk_management.py': 'Advanced Risk Management',
        '08_TESTS/test_phase3_live_trading_integration.py': 'Integration Test Suite',
        '07_DASHBOARD/live_trading_dashboard.py': 'Live Trading Dashboard'
    }

    logger.info("Component Validation:")
    all_present = True

    for file_path, description in components.items():
        if Path(file_path).exists():
            logger.info(f"  ✅ {description}: Present")
        else:
            logger.error(f"  ❌ {description}: Missing ({file_path})")
            all_present = False

    return all_present


def test_basic_functionality():
    """Test basic functionality of components"""

    logger.info("Basic Functionality Tests:")

    # Test database connectivity
    try:
        conn = sqlite3.connect("marketpulse_production.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM market_data")
        count = cursor.fetchone()[0]
        conn.close()
        logger.info(f"  ✅ Database: Connected ({count} market data records)")
    except Exception as e:
        logger.error(f"  ❌ Database: {e}")

    # Test component imports
    try:
        from advanced_risk_management import AdvancedRiskManager
        risk_manager = AdvancedRiskManager()
        logger.info("  ✅ Risk Management: Importable")
    except Exception as e:
        logger.error(f"  ❌ Risk Management: {e}")

    # Test market data (expected to fail due to API limits)
    try:
        from live_market_data_fetcher import LiveMarketDataFetcher
        fetcher = LiveMarketDataFetcher()
        status = fetcher.get_provider_status()
        logger.info(f"  ✅ Market Data Fetcher: {len(status)} providers configured")
    except Exception as e:
        logger.error(f"  ❌ Market Data Fetcher: {e}")


def generate_deployment_report():
    """Generate deployment readiness report"""

    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3 PRODUCTION DEPLOYMENT REPORT")
    logger.info("=" * 50)

    # Check requirements
    requirements_met = []

    # Database
    db_updated = update_database_schema()
    requirements_met.append(("Database Schema", db_updated))

    # Components
    components_present = validate_components()
    requirements_met.append(("Components Present", components_present))

    # Environment
    env_vars = setup_environment_variables()
    api_keys_configured = bool(os.getenv('ALPHA_VANTAGE_API_KEY')) or bool(os.getenv('FINNHUB_API_KEY'))
    requirements_met.append(("API Keys", api_keys_configured))

    # Test functionality
    test_basic_functionality()

    # Summary
    met_count = sum(1 for _, met in requirements_met if met)
    total_count = len(requirements_met)
    readiness_pct = (met_count / total_count) * 100

    logger.info(f"\nDEPLOYMENT READINESS: {met_count}/{total_count} ({readiness_pct:.0f}%)")

    for requirement, met in requirements_met:
        status = "✅ READY" if met else "❌ NEEDS ATTENTION"
        logger.info(f"  {requirement}: {status}")

    if readiness_pct >= 80:
        logger.info("\n🎉 PHASE 3 SYSTEM READY FOR PRODUCTION!")
        logger.info("Recommendation: Proceed with live deployment")
    else:
        logger.info("\n⚠️ PHASE 3 SYSTEM NEEDS ATTENTION")
        logger.info("Recommendation: Address missing requirements before production")

    return readiness_pct


def main():
    """Main deployment preparation function"""

    print("Phase 3 Production Deployment Preparation")
    print("=" * 45)

    # Generate comprehensive report
    readiness = generate_deployment_report()

    print(f"\nDeployment preparation complete.")
    print(f"System readiness: {readiness:.0f}%")

    if readiness >= 80:
        print("\nNext steps:")
        print("1. Set up API keys (Alpha Vantage, Finnhub)")
        print("2. Configure production database (PostgreSQL recommended)")
        print("3. Deploy to cloud infrastructure")
        print("4. Set up monitoring and alerting")
        print("5. Begin paper trading validation")

    return True


if __name__ == "__main__":
    main()