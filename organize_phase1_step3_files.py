"""
MarketPulse File Organization Script - Phase 1, Step 3
Moves enhanced trading files to proper project structure directories

Location: #root/organize_phase1_step3_files.py
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileOrganizer:
    """Organize Phase 1, Step 3 files into proper project structure"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.moved_files = []
        self.created_dirs = []

    def ensure_directory_exists(self, dir_path: Path) -> bool:
        """Create directory if it doesn't exist"""
        try:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                self.created_dirs.append(str(dir_path))
                logger.info(f"Created directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return False

    def move_file(self, source: str, destination: str, backup: bool = True) -> bool:
        """Move file from source to destination with optional backup"""

        source_path = Path(source)
        dest_path = Path(destination)

        if not source_path.exists():
            logger.warning(f"Source file not found: {source_path}")
            return False

        # Ensure destination directory exists
        self.ensure_directory_exists(dest_path.parent)

        try:
            # Create backup if file exists at destination
            if dest_path.exists() and backup:
                backup_path = dest_path.with_suffix(dest_path.suffix + '.backup')
                shutil.copy2(dest_path, backup_path)
                logger.info(f"Backed up existing file: {backup_path}")

            # Move file
            shutil.move(source_path, dest_path)
            self.moved_files.append((str(source_path), str(dest_path)))
            logger.info(f"Moved: {source_path} → {dest_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to move {source_path} to {dest_path}: {e}")
            return False

    def organize_enhanced_files(self) -> dict:
        """Organize all Phase 1, Step 3 enhanced files"""

        print("🗂️ MarketPulse File Organization - Phase 1, Step 3")
        print("=" * 55)
        print("Moving enhanced trading files to proper project structure...")

        # File organization map
        file_moves = [
            # Enhanced Trading System → 02_ANALYSIS/enhanced/
            {
                'source': 'enhanced_trading_system.py',
                'destination': '02_ANALYSIS/enhanced/enhanced_trading_system.py',
                'description': 'Enhanced trading strategies with advanced indicators'
            },

            # Backtesting Framework → 03_ML_ENGINE/backtesting/
            {
                'source': 'backtesting_framework.py',
                'destination': '03_ML_ENGINE/backtesting/backtesting_framework.py',
                'description': 'Historical strategy validation framework'
            },

            # Enhanced Dashboard → 07_DASHBOARD/enhanced/
            {
                'source': 'enhanced_dashboard.py',
                'destination': '07_DASHBOARD/enhanced/enhanced_dashboard.py',
                'description': 'Advanced technical analysis dashboard'
            },

            # Test Files → 08_TESTS/phase1_step3/
            {
                'source': 'test_enhanced_strategies.py',
                'destination': '08_TESTS/phase1_step3/test_enhanced_strategies.py',
                'description': 'Comprehensive Phase 1 Step 3 tests'
            },

            # Data Fetcher Enhancement → 06_DATA/enhanced/
            {
                'source': '06_DATA/data_fetcher.py',
                'destination': '06_DATA/enhanced/data_fetcher.py',
                'description': 'Enhanced market data fetcher with demo mode'
            }
        ]

        # Additional files to organize (if they exist)
        optional_moves = [
            # Any cleanup scripts → 09_DOCS/scripts/
            {
                'source': 'cleanup_phase1_step2.py',
                'destination': '09_DOCS/scripts/cleanup_phase1_step2.py',
                'description': 'Phase 1 Step 2 cleanup script'
            },

            # Session context files → 09_DOCS/context/
            {
                'source': 'context_summary.md',
                'destination': '09_DOCS/context/context_summary.md',
                'description': 'Session context summary'
            },

            # Changelog → 09_DOCS/
            {
                'source': 'changelog.md',
                'destination': '09_DOCS/changelog.md',
                'description': 'Project changelog'
            }
        ]

        results = {
            'successful_moves': 0,
            'failed_moves': 0,
            'skipped_files': 0,
            'directories_created': 0
        }

        # Process required file moves
        print("\\n📋 Required File Moves:")
        print("-" * 30)

        for move in file_moves:
            source = move['source']
            dest = move['destination']
            desc = move['description']

            print(f"\\n📄 {Path(source).name}")
            print(f"   📁 Moving to: {dest}")
            print(f"   📝 Purpose: {desc}")

            if self.move_file(source, dest):
                results['successful_moves'] += 1
            else:
                results['failed_moves'] += 1

        # Process optional file moves
        print("\\n📋 Optional File Moves:")
        print("-" * 30)

        for move in optional_moves:
            source = move['source']
            dest = move['destination']
            desc = move['description']

            if Path(source).exists():
                print(f"\\n📄 {Path(source).name}")
                print(f"   📁 Moving to: {dest}")
                print(f"   📝 Purpose: {desc}")

                if self.move_file(source, dest):
                    results['successful_moves'] += 1
                else:
                    results['failed_moves'] += 1
            else:
                results['skipped_files'] += 1

        results['directories_created'] = len(self.created_dirs)

        return results

    def create_integration_links(self):
        """Create integration files to connect enhanced components"""

        print("\\n🔗 Creating Integration Links...")
        print("-" * 35)

        # Integration script for enhanced trading system
        integration_script = '''"""
Enhanced Trading System Integration
Connects Phase 1 Step 3 components with main system

Usage:
    from integration.enhanced_integration import EnhancedTradingIntegration

    # Initialize enhanced system
    enhanced = EnhancedTradingIntegration()
    enhanced.start_enhanced_trading()
"""

import sys
from pathlib import Path

# Add enhanced component paths
sys.path.append(str(Path(__file__).parent.parent / "02_ANALYSIS" / "enhanced"))
sys.path.append(str(Path(__file__).parent.parent / "03_ML_ENGINE" / "backtesting"))
sys.path.append(str(Path(__file__).parent.parent / "07_DASHBOARD" / "enhanced"))

try:
    from enhanced_trading_system import EnhancedTradingSystem
    from backtesting_framework import BacktestingEngine
    print("✅ Enhanced components imported successfully")
except ImportError as e:
    print(f"⚠️ Enhanced component import failed: {e}")
    print("Make sure enhanced files are in proper directories")

class EnhancedTradingIntegration:
    """Integration class for enhanced trading components"""

    def __init__(self):
        self.enhanced_system = None
        self.backtester = None

    def initialize_enhanced_system(self):
        """Initialize enhanced trading system"""
        try:
            self.enhanced_system = EnhancedTradingSystem()
            print("✅ Enhanced trading system initialized")
            return True
        except Exception as e:
            print(f"❌ Enhanced system initialization failed: {e}")
            return False

    def initialize_backtester(self):
        """Initialize backtesting framework"""
        try:
            self.backtester = BacktestingEngine()
            print("✅ Backtesting framework initialized")
            return True
        except Exception as e:
            print(f"❌ Backtester initialization failed: {e}")
            return False

    def run_enhanced_analysis(self, symbol: str):
        """Run enhanced analysis on symbol"""
        if not self.enhanced_system:
            if not self.initialize_enhanced_system():
                return None

        return self.enhanced_system.analyze_symbol(symbol)

    def run_backtest(self, strategy, symbols: list, days: int = 30):
        """Run backtest on strategy"""
        if not self.backtester:
            if not self.initialize_backtester():
                return None

        return self.backtester.backtest_strategy(strategy, symbols, days)

    def start_enhanced_trading(self):
        """Start enhanced trading with all components"""
        print("🚀 Starting Enhanced Trading System...")

        # Initialize all components
        enhanced_ready = self.initialize_enhanced_system()
        backtest_ready = self.initialize_backtester()

        if enhanced_ready:
            # Run opportunity scan
            opportunities = self.enhanced_system.scan_opportunities()
            print(f"💡 Found {len(opportunities)} trading opportunities")

            return opportunities
        else:
            print("❌ Enhanced trading system not ready")
            return []

# Test integration
if __name__ == "__main__":
    integration = EnhancedTradingIntegration()
    opportunities = integration.start_enhanced_trading()

    for opp in opportunities[:3]:  # Show top 3
        print(f"📈 {opp['signal']} {opp['symbol']} (confidence: {opp['confidence']:.0%})")
'''

        # Create integration directory and file
        integration_dir = Path('01_CORE/integration')
        self.ensure_directory_exists(integration_dir)

        integration_file = integration_dir / 'enhanced_integration.py'

        try:
            with open(integration_file, 'w', encoding='utf-8') as f:
                f.write(integration_script)

            print(f"✅ Created integration file: {integration_file}")

            # Create __init__.py for package
            init_file = integration_dir / '__init__.py'
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('"""Enhanced Trading System Integration Package"""\\n')

            print(f"✅ Created package init: {init_file}")

        except Exception as e:
            print(f"❌ Failed to create integration files: {e}")

    def update_main_imports(self):
        """Update main.py to include enhanced components"""

        print("\\n🔧 Updating Main System Imports...")
        print("-" * 40)

        main_file = Path('main.py')

        if not main_file.exists():
            print("⚠️ main.py not found, skipping import updates")
            return

        try:
            # Read current main.py
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Enhanced import statement to add
            enhanced_import = """
# Enhanced Trading System Integration (Phase 1, Step 3)
try:
    from 01_CORE.integration.enhanced_integration import EnhancedTradingIntegration
    ENHANCED_TRADING_AVAILABLE = True
    logger.info("✅ Enhanced trading components loaded")
except ImportError as e:
    ENHANCED_TRADING_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced trading components not available: {e}")
"""

            # Add import if not already present
            if "EnhancedTradingIntegration" not in content:
                # Find where to insert (after other imports)
                lines = content.split('\\n')
                insert_pos = 0

                for i, line in enumerate(lines):
                    if line.strip().startswith('logger =') or 'logger.info' in line:
                        insert_pos = i
                        break

                lines.insert(insert_pos, enhanced_import)

                # Write back to file
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write('\\n'.join(lines))

                print("✅ Updated main.py with enhanced trading imports")
            else:
                print("✅ main.py already has enhanced trading integration")

        except Exception as e:
            print(f"❌ Failed to update main.py: {e}")

    def show_final_structure(self):
        """Show the organized project structure"""

        print("\\n📁 ORGANIZED PROJECT STRUCTURE:")
        print("=" * 45)

        structure = {
            '02_ANALYSIS/enhanced/': ['enhanced_trading_system.py'],
            '03_ML_ENGINE/backtesting/': ['backtesting_framework.py'],
            '06_DATA/enhanced/': ['data_fetcher.py'],
            '07_DASHBOARD/enhanced/': ['enhanced_dashboard.py'],
            '08_TESTS/phase1_step3/': ['test_enhanced_strategies.py'],
            '01_CORE/integration/': ['enhanced_integration.py', '__init__.py'],
            '09_DOCS/scripts/': ['cleanup scripts (if any)'],
            '09_DOCS/context/': ['context_summary.md (if exists)']
        }

        for directory, files in structure.items():
            dir_path = Path(directory)
            if dir_path.exists() or directory in [str(d) for d in self.created_dirs]:
                print(f"\\n📂 {directory}")
                for file in files:
                    file_path = dir_path / file
                    if file_path.exists():
                        print(f"   ✅ {file}")
                    else:
                        print(f"   📄 {file} (expected)")


def main():
    """Main file organization function"""

    organizer = FileOrganizer()

    # Organize enhanced files
    results = organizer.organize_enhanced_files()

    # Create integration links
    organizer.create_integration_links()

    # Update main system imports
    organizer.update_main_imports()

    # Show final structure
    organizer.show_final_structure()

    # Summary
    print("\\n" + "=" * 60)
    print("🎉 FILE ORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Files moved successfully: {results['successful_moves']}")
    print(f"❌ Files failed to move: {results['failed_moves']}")
    print(f"⏭️ Files skipped (not found): {results['skipped_files']}")
    print(f"📁 Directories created: {results['directories_created']}")

    if results['failed_moves'] == 0:
        print("\\n🚀 All files organized successfully!")
        print("\\n📋 Next Steps:")
        print("1. Test integration: python 01_CORE/integration/enhanced_integration.py")
        print("2. Run tests: python 08_TESTS/phase1_step3/test_enhanced_strategies.py")
        print("3. Update imports in your existing code to use new paths")
        print("4. Proceed to Phase 2 with clean, organized codebase")
    else:
        print("\\n⚠️ Some files failed to move. Check errors above.")

    print("\\n✨ Your project structure now follows professional standards!")

    return results['failed_moves'] == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)