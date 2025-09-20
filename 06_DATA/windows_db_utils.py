# 06_DATA/windows_db_utils.py
"""
Windows-Specific Database Utilities for SQLite
Handles Windows file locking and connection management issues

Location: #06_DATA/windows_db_utils.py
"""

import sqlite3
import os
import time
import gc
import threading
from pathlib import Path
from typing import Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


class WindowsSQLiteManager:
    """Windows-compatible SQLite database manager"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connections = []
        self.lock = threading.Lock()

    def get_connection(self, timeout: float = 30.0) -> sqlite3.Connection:
        """Get a new database connection with Windows-optimized settings"""
        conn = sqlite3.connect(self.db_path, timeout=timeout)

        # Windows-specific SQLite optimizations
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Balanced safety/performance
        conn.execute("PRAGMA temp_store=MEMORY")  # Store temp data in memory
        conn.execute("PRAGMA cache_size=10000")  # Larger cache
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout

        with self.lock:
            self.connections.append(conn)

        return conn

    def close_connection(self, conn: sqlite3.Connection):
        """Properly close a database connection"""
        try:
            if conn:
                conn.close()
                with self.lock:
                    if conn in self.connections:
                        self.connections.remove(conn)
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    def close_all_connections(self):
        """Close all tracked connections"""
        with self.lock:
            connections_to_close = self.connections.copy()
            self.connections.clear()

        for conn in connections_to_close:
            try:
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing connection during cleanup: {e}")

        # Force garbage collection and wait for Windows file system
        gc.collect()
        time.sleep(0.1)

    def execute_with_retry(self, query: str, params: tuple = (),
                           max_retries: int = 3) -> Optional[List[tuple]]:
        """Execute query with automatic retry for Windows lock issues"""
        for attempt in range(max_retries):
            conn = None
            try:
                conn = self.get_connection()
                cursor = conn.execute(query, params)
                result = cursor.fetchall()
                conn.commit()
                return result

            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying... (attempt {attempt + 1})")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Database error: {e}")
                raise
            finally:
                if conn:
                    self.close_connection(conn)

        return None

    def safe_file_delete(self, file_path: str, max_retries: int = 5) -> bool:
        """Safely delete a database file on Windows"""
        if not os.path.exists(file_path):
            return True

        # Close all connections first
        self.close_all_connections()

        # Additional cleanup for Windows
        gc.collect()
        time.sleep(0.2)

        for attempt in range(max_retries):
            try:
                os.remove(file_path)
                logger.info(f"Successfully deleted {file_path}")
                return True

            except PermissionError:
                if attempt == max_retries - 1:
                    logger.warning(f"Could not delete {file_path} - file still in use")
                    return False
                else:
                    logger.debug(f"File delete attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.5 * (attempt + 1))

            except Exception as e:
                logger.error(f"Error deleting file: {e}")
                return False

        return False


def windows_safe_db_operation(db_path: str, operation: Callable, *args, **kwargs):
    """
    Execute a database operation with Windows-safe connection management

    Args:
        db_path: Path to SQLite database
        operation: Function to execute with database connection
        *args, **kwargs: Arguments to pass to operation

    Returns:
        Result of operation or None if failed
    """
    manager = WindowsSQLiteManager(db_path)
    conn = None

    try:
        conn = manager.get_connection()
        return operation(conn, *args, **kwargs)

    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        return None

    finally:
        if conn:
            manager.close_connection(conn)


def fix_windows_database_locks(db_path: str) -> bool:
    """
    Fix common Windows SQLite locking issues

    Args:
        db_path: Path to SQLite database

    Returns:
        True if issues were resolved
    """
    try:
        # Check if WAL files exist and clean them up
        wal_file = f"{db_path}-wal"
        shm_file = f"{db_path}-shm"

        manager = WindowsSQLiteManager(db_path)
        manager.close_all_connections()

        # Force checkpoint to merge WAL back to main DB
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not checkpoint WAL: {e}")
        finally:
            if conn:
                conn.close()

        # Clean up WAL files if they exist
        for temp_file in [wal_file, shm_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.info(f"Cleaned up {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove {temp_file}: {e}")

        return True

    except Exception as e:
        logger.error(f"Error fixing database locks: {e}")
        return False


# Example usage functions for your Yahoo Finance integration
def windows_compatible_quote_storage(db_path: str, quote_data: dict):
    """Store quote data with Windows-compatible connection management"""

    def store_operation(conn, data):
        conn.execute("""
            INSERT OR REPLACE INTO real_time_quotes 
            (symbol, current_price, bid, ask, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data['symbol'], data['current_price'],
            data.get('bid'), data.get('ask'), data['updated_at']
        ))
        conn.commit()
        return True

    return windows_safe_db_operation(db_path, store_operation, quote_data)


def windows_compatible_health_logging(db_path: str, health_data: dict):
    """Log health data with Windows-compatible connection management"""

    def log_operation(conn, data):
        conn.execute("""
            INSERT INTO connection_health_log 
            (status, failure_rate, response_time, details)
            VALUES (?, ?, ?, ?)
        """, (
            data['status'], data.get('failure_rate', 0),
            data.get('response_time', 0), data.get('details', '{}')
        ))
        conn.commit()
        return True

    return windows_safe_db_operation(db_path, log_operation, health_data)


if __name__ == "__main__":
    # Test the Windows SQLite utilities
    test_db = "test_windows_compat.db"

    print("Testing Windows SQLite compatibility...")

    # Test connection management
    manager = WindowsSQLiteManager(test_db)

    # Create test table
    conn = manager.get_connection()
    conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    manager.close_connection(conn)

    # Test retry mechanism
    result = manager.execute_with_retry(
        "INSERT INTO test (value) VALUES (?)",
        ("test_value",)
    )

    # Test safe cleanup
    success = manager.safe_file_delete(test_db)

    print(f"Windows compatibility test: {'PASSED' if success else 'FAILED'}")