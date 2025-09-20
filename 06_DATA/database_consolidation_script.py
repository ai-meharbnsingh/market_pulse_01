# 06_DATA/database_consolidation_script.py
"""
Database Consolidation Script for MarketPulse
Consolidates 11+ scattered databases into 3 focused databases

Target Architecture:
- marketpulse_production.db     → Live trading operations
- 06_DATA/marketpulse_training.db → ML training (174 stocks)
- 10_DATA_STORAGE/marketpulse_performance.db → Analytics

Location: #06_DATA/database_consolidation_script.py
"""

import sqlite3
import os
import shutil
from datetime import datetime
import logging
from typing import List, Dict
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConsolidator:
    """Consolidates scattered databases into organized structure"""

    def __init__(self, base_path: str = "/home/claude/market_pulse_01"):
        self.base_path = base_path
        self.backup_dir = os.path.join(base_path, "database_backups",
                                       f"consolidation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Target database paths
        self.target_dbs = {
            'production': 'marketpulse_production.db',
            'training': '06_DATA/marketpulse_training.db',
            'performance': '10_DATA_STORAGE/marketpulse_performance.db'
        }

    def analyze_current_databases(self) -> Dict:
        """Analyze all current databases and their contents"""

        logger.info("🔍 ANALYZING CURRENT DATABASE STRUCTURE...")

        # Find all .db files
        db_files = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.db') and not file.startswith('test_'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.base_path)
                    db_files.append(rel_path)

        analysis = {
            'total_databases': len(db_files),
            'database_files': db_files,
            'database_details': {},
            'consolidation_plan': {}
        }

        # Analyze each database
        for db_path in db_files:
            full_path = os.path.join(self.base_path, db_path)

            try:
                conn = sqlite3.connect(full_path)
                cursor = conn.cursor()

                # Get table list
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]

                # Get record counts
                table_counts = {}
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        table_counts[table] = count
                    except:
                        table_counts[table] = 'error'

                # Get file size
                file_size = os.path.getsize(full_path)

                analysis['database_details'][db_path] = {
                    'tables': tables,
                    'table_counts': table_counts,
                    'file_size_bytes': file_size,
                    'file_size_mb': round(file_size / (1024 * 1024), 2)
                }

                conn.close()

            except Exception as e:
                analysis['database_details'][db_path] = {
                    'error': str(e),
                    'file_size_bytes': os.path.getsize(full_path) if os.path.exists(full_path) else 0
                }

        # Create consolidation plan
        analysis['consolidation_plan'] = self._create_consolidation_plan(analysis['database_details'])

        return analysis

    def _create_consolidation_plan(self, db_details: Dict) -> Dict:
        """Create consolidation plan based on database contents"""

        plan = {
            'production': [],  # Live trading data
            'training': [],  # ML training data
            'performance': [],  # Analytics and performance
            'archive': []  # Old/test databases to archive
        }

        for db_path, details in db_details.items():
            if 'error' in details:
                plan['archive'].append(db_path)
                continue

            tables = details.get('tables', [])

            # Categorize based on table contents
            if any(t in ['market_data_enhanced', 'indian_stocks'] for t in tables):
                plan['training'].append(db_path)
            elif any(t in ['trades', 'portfolios', 'signals'] for t in tables):
                plan['production'].append(db_path)
            elif any(t in ['performance_metrics', 'performance_data', 'ml_reliability'] for t in tables):
                plan['performance'].append(db_path)
            elif 'backup' in db_path or 'test_' in db_path:
                plan['archive'].append(db_path)
            else:
                # Default based on path
                if 'production' in db_path:
                    plan['production'].append(db_path)
                elif 'training' in db_path:
                    plan['training'].append(db_path)
                elif 'performance' in db_path:
                    plan['performance'].append(db_path)
                else:
                    plan['archive'].append(db_path)

        return plan

    def create_backup(self, analysis: Dict):
        """Create backup of all databases before consolidation"""

        logger.info(f"📦 CREATING BACKUP IN: {self.backup_dir}")

        os.makedirs(self.backup_dir, exist_ok=True)

        backup_manifest = {
            'timestamp': datetime.now().isoformat(),
            'total_databases': analysis['total_databases'],
            'backed_up_files': []
        }

        for db_path in analysis['database_files']:
            src = os.path.join(self.base_path, db_path)
            if os.path.exists(src):
                # Create safe filename for backup
                safe_name = db_path.replace('/', '_').replace('\\', '_')
                dst = os.path.join(self.backup_dir, safe_name)

                shutil.copy2(src, dst)
                backup_manifest['backed_up_files'].append({
                    'original': db_path,
                    'backup': safe_name,
                    'size_mb': analysis['database_details'][db_path].get('file_size_mb', 0)
                })

                logger.info(f"✅ Backed up: {db_path}")

        # Save backup manifest
        manifest_path = os.path.join(self.backup_dir, 'backup_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(backup_manifest, f, indent=2)

        logger.info(f"📝 Backup manifest saved: {manifest_path}")

    def create_target_databases(self):
        """Create the 3 target databases with proper schemas"""

        logger.info("🗂️ CREATING TARGET DATABASE SCHEMAS...")

        # Create directories if needed
        os.makedirs(os.path.join(self.base_path, '06_DATA'), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, '10_DATA_STORAGE'), exist_ok=True)

        # 1. Production Database Schema
        self._create_production_schema()

        # 2. Training Database Schema
        self._create_training_schema()

        # 3. Performance Database Schema
        self._create_performance_schema()

    def _create_production_schema(self):
        """Create production database schema for live trading"""

        db_path = os.path.join(self.base_path, self.target_dbs['production'])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Portfolios table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"✅ Created production database: {db_path}")

    def _create_training_schema(self):
        """Create training database schema for ML training"""

        db_path = os.path.join(self.base_path, self.target_dbs['training'])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Enhanced market data for training
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_enhanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                symbol_clean TEXT NOT NULL,
                market_cap_category TEXT NOT NULL,
                sector TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                adj_close REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Technical indicators
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                bollinger_upper REAL,
                bollinger_lower REAL,
                volume_sma REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Alpha predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alpha_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                prediction_value REAL NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"✅ Created training database: {db_path}")

    def _create_performance_schema(self):
        """Create performance database schema for analytics"""

        db_path = os.path.join(self.base_path, self.target_dbs['performance'])
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Performance metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component_name TEXT NOT NULL,
                method_name TEXT NOT NULL,
                execution_time_ms REAL NOT NULL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                success BOOLEAN DEFAULT TRUE,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # System logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ML reliability tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_reliability (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT,
                recovery_action TEXT,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"✅ Created performance database: {db_path}")

    def consolidate_databases(self, analysis: Dict):
        """Consolidate databases according to the plan"""

        logger.info("🔄 CONSOLIDATING DATABASES...")

        plan = analysis['consolidation_plan']

        # Migrate data for each category
        for category, db_list in plan.items():
            if category == 'archive':
                continue  # Skip archived databases

            target_db = self.target_dbs.get(category)
            if not target_db:
                continue

            logger.info(f"📊 Consolidating {category} databases: {len(db_list)} sources")

            for source_db in db_list:
                self._migrate_database_data(source_db, target_db, category)

    def _migrate_database_data(self, source_path: str, target_db: str, category: str):
        """Migrate data from source database to target database"""

        source_full = os.path.join(self.base_path, source_path)
        target_full = os.path.join(self.base_path, target_db)

        if not os.path.exists(source_full):
            logger.warning(f"⚠️ Source database not found: {source_path}")
            return

        try:
            source_conn = sqlite3.connect(source_full)
            target_conn = sqlite3.connect(target_full)

            source_cursor = source_conn.cursor()
            target_cursor = target_conn.cursor()

            # Get tables from source
            source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in source_cursor.fetchall()]

            migrated_records = 0

            for table in tables:
                # Check if table exists in target
                target_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if target_cursor.fetchone():
                    # Table exists, migrate data
                    try:
                        source_cursor.execute(f"SELECT * FROM {table}")
                        rows = source_cursor.fetchall()

                        if rows:
                            # Get column info
                            source_cursor.execute(f"PRAGMA table_info({table})")
                            columns = [col[1] for col in source_cursor.fetchall()]

                            placeholders = ','.join(['?' for _ in columns])
                            insert_sql = f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})"

                            target_cursor.executemany(insert_sql, rows)
                            migrated_records += len(rows)

                            logger.info(f"  ✅ Migrated {len(rows)} records from {table}")

                    except Exception as e:
                        logger.warning(f"  ⚠️ Failed to migrate table {table}: {e}")

            target_conn.commit()
            source_conn.close()
            target_conn.close()

            logger.info(f"✅ Migrated {migrated_records} total records from {source_path}")

        except Exception as e:
            logger.error(f"❌ Failed to migrate {source_path}: {e}")

    def cleanup_old_databases(self, analysis: Dict):
        """Archive old databases after consolidation"""

        logger.info("🧹 CLEANING UP OLD DATABASES...")

        archive_dir = os.path.join(self.backup_dir, "archived_originals")
        os.makedirs(archive_dir, exist_ok=True)

        # Archive databases that aren't the new target databases
        for db_path in analysis['database_files']:
            full_path = os.path.join(self.base_path, db_path)

            # Skip if it's one of our target databases
            if db_path in self.target_dbs.values():
                continue

            # Skip if it's already a backup
            if 'backup' in db_path or 'database_backups' in db_path:
                continue

            if os.path.exists(full_path):
                # Move to archive
                safe_name = db_path.replace('/', '_').replace('\\', '_')
                archive_path = os.path.join(archive_dir, safe_name)

                shutil.move(full_path, archive_path)
                logger.info(f"📦 Archived: {db_path} → {archive_path}")

    def generate_report(self, analysis: Dict) -> str:
        """Generate consolidation report"""

        report_path = os.path.join(self.backup_dir, "consolidation_report.json")

        report = {
            'timestamp': datetime.now().isoformat(),
            'consolidation_summary': {
                'databases_before': analysis['total_databases'],
                'databases_after': 3,
                'consolidation_plan': analysis['consolidation_plan']
            },
            'target_databases': self.target_dbs,
            'backup_location': self.backup_dir
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📋 Consolidation report saved: {report_path}")
        return report_path


def main():
    """Main consolidation process"""
    print("🗂️ MARKETPULSE DATABASE CONSOLIDATION")
    print("=" * 50)

    consolidator = DatabaseConsolidator()

    # Step 1: Analyze current state
    print("\n🔍 Step 1: Analyzing current database structure...")
    analysis = consolidator.analyze_current_databases()

    print(f"Found {analysis['total_databases']} databases")
    for category, db_list in analysis['consolidation_plan'].items():
        print(f"  {category}: {len(db_list)} databases")

    # Step 2: Create backup
    print("\n📦 Step 2: Creating backup...")
    consolidator.create_backup(analysis)

    # Step 3: Create target databases
    print("\n🗂️ Step 3: Creating target database schemas...")
    consolidator.create_target_databases()

    # Step 4: Consolidate data
    print("\n🔄 Step 4: Consolidating database data...")
    consolidator.consolidate_databases(analysis)

    # Step 5: Cleanup
    print("\n🧹 Step 5: Cleaning up old databases...")
    consolidator.cleanup_old_databases(analysis)

    # Step 6: Generate report
    print("\n📋 Step 6: Generating consolidation report...")
    report_path = consolidator.generate_report(analysis)

    print("\n✅ DATABASE CONSOLIDATION COMPLETE!")
    print(f"📋 Report: {report_path}")
    print(f"📦 Backup: {consolidator.backup_dir}")
    print("\n🎯 NEW DATABASE STRUCTURE:")
    for purpose, path in consolidator.target_dbs.items():
        print(f"  {purpose}: {path}")


if __name__ == "__main__":
    main()