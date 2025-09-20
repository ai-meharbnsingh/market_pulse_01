#!/usr/bin/env python3
"""
Database Schema Enhancement for MarketPulse Live Trading
Addresses compatibility issues identified in Phase 3 integration testing

File location: #06_DATA/database/schema_enhancement.py
"""

import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseSchemaEnhancer:
    """
    Enhances database schema for production deployment
    Addresses price column naming and compatibility issues
    """

    def __init__(self, db_path: str = "marketpulse_production.db"):
        self.db_path = db_path
        self.backup_created = False

    def create_backup(self) -> str:
        """Create backup before schema changes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"marketpulse_backup_{timestamp}.db"

        try:
            # Create backup using SQLite backup API
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)

            logger.info(f"Database backup created: {backup_path}")
            self.backup_created = True
            return backup_path

        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise

    def check_schema_compatibility(self) -> Dict[str, List[str]]:
        """Check current schema for compatibility issues"""
        issues = {
            "price_columns": [],
            "missing_indexes": [],
            "constraint_issues": []
        }

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check trades table for price column consistency
            cursor.execute("PRAGMA table_info(trades)")
            trades_cols = cursor.fetchall()

            price_cols = [col for col in trades_cols if 'price' in col[1].lower()]
            logger.info(f"Price columns in trades: {[col[1] for col in price_cols]}")

            # Check if we need executed_price column
            has_executed_price = any(col[1] == 'executed_price' for col in trades_cols)
            if not has_executed_price:
                issues["price_columns"].append("Missing executed_price column for order tracking")

            # Check market_data table for price consistency
            cursor.execute("PRAGMA table_info(market_data)")
            market_cols = cursor.fetchall()

            # Verify all required price columns exist
            required_cols = ['open_price', 'high_price', 'low_price', 'close_price']
            existing_cols = [col[1] for col in market_cols]

            for req_col in required_cols:
                if req_col not in existing_cols:
                    issues["price_columns"].append(f"Missing {req_col} in market_data")

            # Check for performance indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [idx[0] for idx in cursor.fetchall()]

            required_indexes = [
                "idx_market_data_symbol_timestamp",
                "idx_trades_symbol_status",
                "idx_signals_timestamp"
            ]

            for req_idx in required_indexes:
                if req_idx not in indexes:
                    issues["missing_indexes"].append(req_idx)

        return issues

    def add_executed_price_column(self) -> bool:
        """Add executed_price column for better order tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if column exists
                cursor.execute("PRAGMA table_info(trades)")
                columns = [col[1] for col in cursor.fetchall()]

                if 'executed_price' not in columns:
                    cursor.execute("""
                        ALTER TABLE trades 
                        ADD COLUMN executed_price REAL DEFAULT 0.0
                    """)
                    logger.info("Added executed_price column to trades table")

                    # Update existing records to use entry_price as executed_price
                    cursor.execute("""
                        UPDATE trades 
                        SET executed_price = entry_price 
                        WHERE executed_price = 0.0 AND entry_price > 0
                    """)
                    logger.info("Updated existing records with executed_price values")

                else:
                    logger.info("executed_price column already exists")

                return True

        except Exception as e:
            logger.error(f"Failed to add executed_price column: {e}")
            return False

    def create_performance_indexes(self) -> bool:
        """Create indexes for better query performance"""
        indexes = [
            ("idx_market_data_symbol_timestamp", "market_data", "symbol, timestamp"),
            ("idx_trades_symbol_status", "trades", "symbol, status"),
            ("idx_trades_timestamp", "trades", "entry_timestamp"),
            ("idx_signals_timestamp", "signals", "timestamp"),
            ("idx_technical_indicators_symbol", "technical_indicators", "symbol")
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for idx_name, table, columns in indexes:
                    try:
                        cursor.execute(f"""
                            CREATE INDEX IF NOT EXISTS {idx_name} 
                            ON {table} ({columns})
                        """)
                        logger.info(f"Created/verified index: {idx_name}")
                    except Exception as e:
                        logger.warning(f"Index creation warning for {idx_name}: {e}")

                return True

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False

    def add_production_columns(self) -> bool:
        """Add columns needed for production live trading"""
        production_updates = [
            ("trades", "order_id", "TEXT DEFAULT ''", "Broker order ID for live trading"),
            ("trades", "fill_price", "REAL DEFAULT 0.0", "Actual fill price from broker"),
            ("trades", "slippage", "REAL DEFAULT 0.0", "Price slippage calculation"),
            ("market_data", "bid_price", "REAL DEFAULT 0.0", "Best bid price"),
            ("market_data", "ask_price", "REAL DEFAULT 0.0", "Best ask price"),
            ("market_data", "spread", "REAL DEFAULT 0.0", "Bid-ask spread")
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for table, column, definition, description in production_updates:
                    try:
                        # Check if column exists
                        cursor.execute(f"PRAGMA table_info({table})")
                        existing_cols = [col[1] for col in cursor.fetchall()]

                        if column not in existing_cols:
                            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                            logger.info(f"Added {column} to {table}: {description}")
                        else:
                            logger.info(f"Column {column} already exists in {table}")

                    except Exception as e:
                        logger.warning(f"Column addition warning for {table}.{column}: {e}")

                return True

        except Exception as e:
            logger.error(f"Failed to add production columns: {e}")
            return False

    def validate_schema_integrity(self) -> Dict[str, bool]:
        """Validate database integrity after enhancements"""
        results = {
            "foreign_keys": False,
            "indexes": False,
            "data_consistency": False
        }

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check foreign key constraints
                cursor.execute("PRAGMA foreign_key_check")
                fk_issues = cursor.fetchall()
                results["foreign_keys"] = len(fk_issues) == 0

                if fk_issues:
                    logger.warning(f"Foreign key issues found: {fk_issues}")

                # Verify indexes exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
                indexes = cursor.fetchall()
                results["indexes"] = len(indexes) >= 5  # Should have at least 5 indexes

                # Check data consistency
                cursor.execute("SELECT COUNT(*) FROM market_data WHERE close_price > 0")
                valid_prices = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM market_data")
                total_records = cursor.fetchone()[0]

                consistency_ratio = valid_prices / max(total_records, 1)
                results["data_consistency"] = consistency_ratio > 0.8  # 80% valid data

                logger.info(f"Schema validation results: {results}")

        except Exception as e:
            logger.error(f"Schema validation error: {e}")

        return results

    def run_enhancement(self) -> Dict[str, bool]:
        """Run complete schema enhancement process"""
        results = {
            "backup_created": False,
            "executed_price_added": False,
            "indexes_created": False,
            "production_columns_added": False,
            "validation_passed": False
        }

        logger.info("Starting database schema enhancement...")

        # Step 1: Create backup
        try:
            backup_path = self.create_backup()
            results["backup_created"] = True
            logger.info(f"Backup successful: {backup_path}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return results

        # Step 2: Check current issues
        issues = self.check_schema_compatibility()
        logger.info(f"Schema issues identified: {issues}")

        # Step 3: Add executed_price column
        results["executed_price_added"] = self.add_executed_price_column()

        # Step 4: Create performance indexes
        results["indexes_created"] = self.create_performance_indexes()

        # Step 5: Add production columns
        results["production_columns_added"] = self.add_production_columns()

        # Step 6: Validate integrity
        validation_results = self.validate_schema_integrity()
        results["validation_passed"] = all(validation_results.values())

        # Summary
        success_count = sum(results.values())
        total_steps = len(results)
        success_rate = (success_count / total_steps) * 100

        logger.info(
            f"Schema enhancement completed: {success_count}/{total_steps} steps successful ({success_rate:.1f}%)")

        if success_rate >= 80:
            logger.info("Database schema enhancement SUCCESSFUL - Production ready!")
        else:
            logger.warning("Database schema enhancement had issues - Review required")

        return results


if __name__ == "__main__":
    enhancer = DatabaseSchemaEnhancer()
    results = enhancer.run_enhancement()

    print("\n" + "=" * 50)
    print("DATABASE SCHEMA ENHANCEMENT RESULTS")
    print("=" * 50)

    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{step.replace('_', ' ').title()}: {status}")

    success_rate = (sum(results.values()) / len(results)) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("🎉 PRODUCTION READY - Schema enhanced successfully!")
    else:
        print("⚠️  REVIEW NEEDED - Some enhancements failed")