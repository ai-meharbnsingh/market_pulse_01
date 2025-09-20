# 06_DATA/update_code_for_new_databases.py
"""
Code Update Script for New Database Structure
Update all Python files to use new consolidated database paths

This script automatically updates all .py files to use:
- marketpulse_production.db (live trading)
- 06_DATA/marketpulse_marketpulse_training.db (ML training)
- 10_DATA_STORAGE/marketpulse_marketpulse_performance.db (analytics)

Location: #06_DATA/update_code_for_new_databases.py
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeUpdater:
    """Update code files to use new database structure"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()

        # Database path mappings: old -> new
        self.db_mappings = {
            # Old paths -> New paths
            '"marketpulse_production.db"': '"marketpulse_production.db"',
            "'marketpulse_production.db'": "'marketpulse_production.db'",
            '"marketpulse_production.db"': '"marketpulse_production.db"',  # Keep same
            "'marketpulse_production.db'": "'marketpulse_production.db'",  # Keep same

            # Data directory databases
            '"06_DATA/marketpulse_marketpulse_training.db"',
            "'06_DATA/marketpulse_marketpulse_training.db'",
            '"marketpulse_production.db"': '"marketpulse_production.db"',
            "'marketpulse_production.db'": "'marketpulse_production.db'",

            # Streaming and test databases
            '"marketpulse_production.db"': '"marketpulse_production.db"',
            "'marketpulse_production.db'": "'marketpulse_production.db'",
            '"marketpulse_production.db"': '"marketpulse_production.db"',  # Test databases
            "'marketpulse_production.db'": "'marketpulse_production.db'",

            # Performance databases
            '"10_DATA_STORAGE/marketpulse_marketpulse_performance.db"',
            "'10_DATA_STORAGE/marketpulse_marketpulse_performance.db'",
            '"10_DATA_STORAGE/marketpulse_marketpulse_performance.db"',
            "'10_DATA_STORAGE/marketpulse_marketpulse_performance.db'",
        }

        # Files to skip (backups, etc.)
        self.skip_patterns = [
            '.git',
            '__pycache__',
            '.pytest_cache',
            'database_backups',
            'marketpulse.egg-info',
            '.venv',
            'node_modules',
            '.backup',
            '_backup',
            'backup_',
            'test_',
            '.pyc'
        ]

    def find_python_files(self) -> List[Path]:
        """Find all Python files in project"""

        python_files = []

        for file_path in self.project_root.rglob("*.py"):
            # Skip files in ignored directories
            if any(pattern in str(file_path) for pattern in self.skip_patterns):
                continue

            python_files.append(file_path)

        logger.info(f"Found {len(python_files)} Python files to analyze")
        return python_files

    def analyze_file_database_usage(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Analyze file for database path usage"""

        database_references = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                # Look for database path patterns
                for old_path, new_path in self.db_mappings.items():
                    if old_path in line:
                        database_references.append((line_num, old_path, new_path))

                # Also look for generic .db references
                if '.db' in line and ('sqlite3.connect' in line or 'database' in line.lower()):
                    # This line likely contains a database reference
                    database_references.append((line_num, 'generic_db', line.strip()))

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")

        return database_references

    def update_file_database_paths(self, file_path: Path, dry_run: bool = True) -> Dict:
        """Update database paths in a file"""

        result = {
            'file': str(file_path),
            'changes_made': [],
            'errors': [],
            'success': False
        }

        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            changes_made = 0

            # Apply database path mappings
            for old_path, new_path in self.db_mappings.items():
                if old_path in content:
                    content = content.replace(old_path, new_path)
                    changes_made += 1
                    result['changes_made'].append(f"Changed {old_path} -> {new_path}")

            # Special patterns for common database initialization
            special_patterns = [
                # Default database path pattern
                (r'db_path\s*=\s*["\']marketpulse\.db["\']', 'db_path = "marketpulse_production.db"'),
                (r'database\s*=\s*["\']marketpulse\.db["\']', 'database = "marketpulse_production.db"'),

                # Training data patterns
                (r'marketpulse_training.db'),
                (r'marketpulse_training.db'),

                # Performance patterns
                (r'marketpulse_performance.db'),
                (r'marketpulse_performance.db'),
            ]

            for pattern, replacement in special_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    content = re.sub(pattern, replacement, content)
                    changes_made += len(matches)
                    result['changes_made'].append(f"Pattern update: {pattern}")

            # Write updated content (if not dry run)
            if not dry_run and changes_made > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                result['success'] = True
                logger.info(f"✅ Updated {file_path}: {changes_made} changes")
            elif changes_made > 0:
                result['success'] = True
                logger.info(f"📝 Would update {file_path}: {changes_made} changes")

        except Exception as e:
            result['errors'].append(str(e))
            logger.error(f"❌ Error updating {file_path}: {e}")

        return result

    def create_database_config_module(self):
        """Create centralized database configuration module"""

        config_content = '''# 06_DATA/database_config.py
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
    TRAINING_DB = PROJECT_ROOT / "06_DATA" / "marketpulse_marketpulse_training.db"

    # Performance database (analytics and monitoring)
    PERFORMANCE_DB = PROJECT_ROOT / "10_DATA_STORAGE" / "marketpulse_marketpulse_performance.db"

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
'''

        config_path = self.project_root / "06_DATA" / "database_config.py"
        with open(config_path, 'w') as f:
            f.write(config_content)

        logger.info(f"✅ Created database config module: {config_path}")
        return config_path

    def generate_migration_guide(self, analysis_results: List[Dict]) -> str:
        """Generate migration guide for manual updates"""

        guide_content = """# Database Migration Guide - Manual Updates Required

## Files That Need Manual Review:

"""

        files_needing_attention = []

        for result in analysis_results:
            if result['changes_made'] or result['errors']:
                files_needing_attention.append(result)

        if files_needing_attention:
            guide_content += "### Files with Database References:\n"
            for result in files_needing_attention:
                guide_content += f"\n**{result['file']}:**\n"
                if result['changes_made']:
                    guide_content += "- Changes made:\n"
                    for change in result['changes_made']:
                        guide_content += f"  - {change}\n"
                if result['errors']:
                    guide_content += "- Manual review needed:\n"
                    for error in result['errors']:
                        guide_content += f"  - {error}\n"

        guide_content += """
## Recommended Code Updates:

### 1. Use Database Config Module:
```python
# OLD (hardcoded paths)
db_path = "marketpulse_production.db"
conn = sqlite3.connect(db_path)

# NEW (centralized config)
from database_config import DatabaseConfig
db_path = DatabaseConfig.get_production_db_path()
conn = sqlite3.connect(db_path)
```

### 2. Purpose-based Database Selection:
```python
# For live trading
db_path = DatabaseConfig.get_db_for_purpose('trading')

# For ML training
db_path = DatabaseConfig.get_db_for_purpose('training')

# For performance monitoring
db_path = DatabaseConfig.get_db_for_purpose('performance')
```

### 3. Update Import Statements:
```python
# Add to top of files using databases
import sys
sys.path.append('06_DATA')
from database_config import DatabaseConfig
```

## Testing After Migration:
1. Run your main trading script: `python main.py`
2. Test dashboard: `streamlit run 07_DASHBOARD/dashboard_app.py`
3. Test ML training: `python 03_ML_ENGINE/models/alpha_model.py`
4. Verify paper trading: Test paper trading functionality

## Rollback Plan:
If issues arise, restore from backups in `database_backups/` directory.
"""

        guide_path = self.project_root / "DATABASE_MIGRATION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)

        logger.info(f"📋 Migration guide created: {guide_path}")
        return guide_path

    def update_all_code_files(self, dry_run: bool = True):
        """Update all code files for new database structure"""

        logger.info("🔄 UPDATING CODE FOR NEW DATABASE STRUCTURE")
        logger.info("=" * 60)

        # Step 1: Find all Python files
        python_files = self.find_python_files()

        # Step 2: Analyze and update each file
        analysis_results = []

        for file_path in python_files:
            logger.info(f"Analyzing {file_path}")

            # Check for database usage
            db_refs = self.analyze_file_database_usage(file_path)

            if db_refs:
                logger.info(f"  Found {len(db_refs)} database references")

                # Update the file
                result = self.update_file_database_paths(file_path, dry_run=dry_run)
                analysis_results.append(result)

        # Step 3: Create database config module
        config_path = self.create_database_config_module()

        # Step 4: Generate migration guide
        guide_path = self.generate_migration_guide(analysis_results)

        # Step 5: Summary
        total_files_updated = sum(1 for r in analysis_results if r['success'])
        total_changes = sum(len(r['changes_made']) for r in analysis_results)

        logger.info("✅ CODE UPDATE COMPLETE")
        logger.info("=" * 40)
        logger.info(f"Files analyzed: {len(python_files)}")
        logger.info(f"Files updated: {total_files_updated}")
        logger.info(f"Total changes: {total_changes}")
        logger.info(f"Config module: {config_path}")
        logger.info(f"Migration guide: {guide_path}")

        if dry_run:
            logger.info("\n⚠️ DRY RUN MODE - No files were actually modified")
            logger.info("Run with dry_run=False to apply changes")

        return {
            'files_analyzed': len(python_files),
            'files_updated': total_files_updated,
            'total_changes': total_changes,
            'analysis_results': analysis_results,
            'config_module': config_path,
            'migration_guide': guide_path
        }


def main():
    """Main function to update code for new database structure"""
    print("🔄 UPDATING CODE FOR NEW DATABASE STRUCTURE")
    print("=" * 60)
    print("This script will update all Python files to use the new database paths")
    print("")

    updater = CodeUpdater()

    # First run in dry-run mode to see what would change
    print("🔍 ANALYZING CODE (DRY RUN)...")
    dry_results = updater.update_all_code_files(dry_run=True)

    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Files to update: {dry_results['files_updated']}")
    print(f"Total changes: {dry_results['total_changes']}")

    if dry_results['total_changes'] > 0:
        print("\n⚠️ Changes needed! Run again with apply=True to make actual changes")

        # Ask user if they want to apply changes
        apply = input("\nApply changes now? (y/N): ").lower().strip()

        if apply == 'y':
            print("\n🔧 APPLYING CHANGES...")
            real_results = updater.update_all_code_files(dry_run=False)

            print(f"\n✅ SUCCESS: Updated {real_results['files_updated']} files")
            print("📋 See DATABASE_MIGRATION_GUIDE.md for manual review items")

            return True
        else:
            print("\n📋 Changes not applied. See migration guide for manual updates.")
            return False
    else:
        print("\n✅ No database path updates needed!")
        return True


if __name__ == "__main__":
    success = main()