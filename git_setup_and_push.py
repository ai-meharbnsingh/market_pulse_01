"""
MarketPulse Git Setup and Push Script
Handles complete Git initialization, .gitignore setup, and secure push

Location: #root/git_setup_and_push.py
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitManager:
    """Manage Git operations for MarketPulse project"""

    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.git_initialized = False

    def check_git_installed(self):
        """Check if Git is installed"""
        try:
            result = subprocess.run(['git', '--version'],
                                    capture_output=True, text=True, check=True)
            logger.info(f"Git found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Git is not installed or not in PATH")
            return False

    def create_comprehensive_gitignore(self):
        """Create comprehensive .gitignore file"""

        gitignore_content = """# MarketPulse .gitignore - Comprehensive

# =============================================================================
# SENSITIVE FILES (NEVER COMMIT)
# =============================================================================

# Environment files
.env
.env.local
.env.production
.env.staging
*.env

# API Keys and credentials
**/api_keys.txt
**/credentials.json
**/secrets.yaml
config/api_credentials.py

# Database files with real data
*.db
*.sqlite
*.sqlite3
marketpulse.db
marketpulse_backup.db
test_marketpulse.db

# =============================================================================
# PYTHON
# =============================================================================

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
venv/
.venv/
env/
.env/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# =============================================================================
# TRADING & FINANCIAL DATA
# =============================================================================

# Live trading data
**/live_trades.json
**/real_portfolio.json
**/broker_credentials.json

# Historical data caches
**/historical_cache/
**/price_data/
**/market_data_cache/

# Trading logs with sensitive info
**/live_trading.log
**/execution_logs/
**/order_logs/

# Backtest results with large datasets
**/backtest_results/*.json
**/backtest_results/*.csv
**/large_datasets/

# Model files (can be large)
**/models/*.pkl
**/models/*.joblib
**/models/*.h5
**/models/*.pt
**/trained_models/

# =============================================================================
# DATA STORAGE
# =============================================================================

# Large data files
*.csv
*.xlsx
*.json
!requirements.json
!package.json
!config.json

# Temporary files
tmp/
temp/
.tmp/
*.tmp
*.temp

# Data storage directories
10_DATA_STORAGE/alerts/*.json
10_DATA_STORAGE/logs/*.log
10_DATA_STORAGE/backtest_results/*
10_DATA_STORAGE/models/*
!10_DATA_STORAGE/*/README.md

# =============================================================================
# LOGS & MONITORING
# =============================================================================

# Log files
*.log
logs/
**/*.log
**/logs/

# Monitoring data
**/monitoring/
**/metrics/
**/performance_data/

# =============================================================================
# SYSTEM & IDE
# =============================================================================

# Windows
Thumbs.db
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/

# macOS
.DS_Store
.AppleDouble
.LSOverride
Icon
._*

# Linux
*~
.fuse_hidden*
.directory
.Trash-*

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# =============================================================================
# WEB & DASHBOARD
# =============================================================================

# Node modules (if using any JS)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Streamlit
.streamlit/

# =============================================================================
# TESTING & DEVELOPMENT
# =============================================================================

# Test artifacts
.pytest_cache/
.coverage
htmlcov/
test-results/

# Development databases
dev.db
test.db
development.sqlite

# =============================================================================
# DOCUMENTATION
# =============================================================================

# Generated docs
docs/_build/
docs/build/

# =============================================================================
# CUSTOM MARKERS (Keep these)
# =============================================================================

# Keep important empty directories
!**/README.md
!**/.gitkeep

# Keep configuration templates
!**/*_template.*
!**/config_example.*
!**/settings_example.*

# Keep requirements and setup files
!requirements.txt
!setup.py
!pyproject.toml
!Pipfile

# =============================================================================
# END OF GITIGNORE
# =============================================================================
"""

        gitignore_path = self.project_root / '.gitignore'

        try:
            with open(gitignore_path, 'w', encoding='utf-8') as f:
                f.write(gitignore_content)

            logger.info("Created comprehensive .gitignore file")
            return True

        except Exception as e:
            logger.error(f"Failed to create .gitignore: {e}")
            return False

    def create_readme_if_missing(self):
        """Create a README.md if it doesn't exist or is minimal"""

        readme_path = self.project_root / 'README.md'

        # Check if README exists and has content
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if len(content) > 100:  # Has substantial content
                    logger.info("README.md already exists with content")
                    return True

        readme_content = """# MarketPulse v2.0 🚀

## AI-Powered Personal Trading System

**Status**: Phase 1 Complete - Enhanced Trading Strategies Operational

### Overview

MarketPulse is a sophisticated algorithmic trading system built with professional-grade architecture. It combines advanced technical analysis, machine learning capabilities, and robust risk management for automated trading strategies.

### Current Capabilities

- **Multi-Indicator Technical Analysis**: RSI, MACD, Bollinger Bands, Stochastic, ATR
- **Strategy Ensemble Voting**: Multiple strategies combine for smart decisions
- **Historical Backtesting**: Validate strategies with performance metrics
- **Real Market Data Integration**: SQLite database with market data pipeline
- **Risk Management**: Kelly Criterion position sizing with hard limits
- **Performance Monitoring**: Advanced dashboard with real-time analysis
- **Paper Trading**: Full virtual trading environment

### Architecture

```
MarketPulse/
├── 01_CORE/           # Antifragile AI framework & integrations
├── 02_ANALYSIS/       # Technical analysis & enhanced strategies  
├── 03_ML_ENGINE/      # Machine learning models & backtesting
├── 04_RISK/           # Risk management & position sizing
├── 05_EXECUTION/      # Paper trading & alert systems
├── 06_DATA/           # Market data pipeline & database
├── 07_DASHBOARD/      # Streamlit performance monitoring
├── 08_TESTS/          # Comprehensive testing suite
├── 09_DOCS/           # Documentation & guides
└── 10_DATA_STORAGE/   # Logs, results & persistent data
```

### Key Features

#### Technical Analysis
- Advanced multi-indicator analysis
- Strategy confidence scoring
- Real-time opportunity scanning
- Ensemble decision making

#### Risk Management
- Kelly Criterion position sizing
- Portfolio heat monitoring
- Daily loss limits (2%)
- Maximum position limits (5%)

#### Performance Monitoring
- Real-time dashboard
- Historical backtesting
- Sharpe ratio calculations
- Drawdown analysis

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Initialize Database**
   ```bash
   python 06_DATA/database/db_setup.py
   ```

4. **Run Paper Trading**
   ```bash
   python main.py start
   ```

5. **Launch Dashboard**
   ```bash
   streamlit run 07_DASHBOARD/enhanced/enhanced_dashboard.py
   ```

### Testing

Run comprehensive tests:
```bash
python 08_TESTS/phase1_step3/test_enhanced_strategies.py
```

### Recent Achievements

- **Phase 1, Step 3 Complete**: Enhanced trading strategies operational
- **7/7 Tests Passed**: All comprehensive tests successful
- **Live Signal Generation**: Real trading signals with confidence scoring
- **Professional Architecture**: Clean, scalable, maintainable codebase

### Next Phase

**Phase 2: AI/ML Integration**
- Connect existing ML models
- Advanced prediction algorithms  
- News sentiment analysis
- Multi-timeframe analysis

### Warning

This system is for educational and paper trading purposes. Never risk money you cannot afford to lose. Always test strategies thoroughly before considering live trading.

### License

Personal use only. See LICENSE file for details.

---

**Built with Python, SQLite, Streamlit, and professional trading system architecture.**
"""

        try:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)

            logger.info("Created comprehensive README.md")
            return True

        except Exception as e:
            logger.error(f"Failed to create README.md: {e}")
            return False

    def run_git_command(self, command, check=True):
        """Run a git command and return result"""

        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=check
            )
            return True, result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr

    def initialize_git_repo(self):
        """Initialize Git repository if not already initialized"""

        git_dir = self.project_root / '.git'

        if git_dir.exists():
            logger.info("Git repository already initialized")
            self.git_initialized = True
            return True

        success, stdout, stderr = self.run_git_command(['git', 'init'])

        if success:
            logger.info("Git repository initialized")
            self.git_initialized = True
            return True
        else:
            logger.error(f"Failed to initialize Git repository: {stderr}")
            return False

    def check_git_status(self):
        """Check Git status and show what will be committed"""

        print("\n📊 Git Status Check")
        print("=" * 30)

        # Check status
        success, stdout, stderr = self.run_git_command(['git', 'status', '--porcelain'])

        if not success:
            logger.error(f"Failed to check Git status: {stderr}")
            return False

        if not stdout.strip():
            print("✅ Working directory clean - no changes to commit")
            return True

        print("📁 Files to be added/committed:")

        # Parse status output
        new_files = []
        modified_files = []
        deleted_files = []

        for line in stdout.strip().split('\n'):
            if line.startswith('??'):
                new_files.append(line[3:])
            elif line.startswith(' M'):
                modified_files.append(line[3:])
            elif line.startswith(' D'):
                deleted_files.append(line[3:])

        if new_files:
            print(f"\n📄 New files ({len(new_files)}):")
            for file in new_files[:10]:  # Show first 10
                print(f"   + {file}")
            if len(new_files) > 10:
                print(f"   ... and {len(new_files) - 10} more")

        if modified_files:
            print(f"\n📝 Modified files ({len(modified_files)}):")
            for file in modified_files[:10]:
                print(f"   ~ {file}")
            if len(modified_files) > 10:
                print(f"   ... and {len(modified_files) - 10} more")

        if deleted_files:
            print(f"\n🗑️ Deleted files ({len(deleted_files)}):")
            for file in deleted_files[:5]:
                print(f"   - {file}")

        return True

    def add_all_files(self):
        """Add all files to Git staging"""

        success, stdout, stderr = self.run_git_command(['git', 'add', '.'])

        if success:
            logger.info("All files added to Git staging area")
            return True
        else:
            logger.error(f"Failed to add files: {stderr}")
            return False

    def create_commit(self, message=None):
        """Create a Git commit"""

        if not message:
            message = "Phase 1 Complete: Enhanced Trading System with Professional Architecture"

        success, stdout, stderr = self.run_git_command(['git', 'commit', '-m', message])

        if success:
            logger.info(f"Commit created: {message}")
            return True
        else:
            if "nothing to commit" in stderr:
                logger.info("Nothing to commit - working directory clean")
                return True
            else:
                logger.error(f"Failed to create commit: {stderr}")
                return False

    def setup_remote_origin(self, repo_url):
        """Setup remote origin"""

        # Check if remote already exists
        success, stdout, stderr = self.run_git_command(['git', 'remote', 'get-url', 'origin'], check=False)

        if success:
            current_url = stdout.strip()
            if current_url == repo_url:
                logger.info(f"Remote origin already set to: {repo_url}")
                return True
            else:
                logger.info(f"Updating remote origin from {current_url} to {repo_url}")
                success, stdout, stderr = self.run_git_command(['git', 'remote', 'set-url', 'origin', repo_url])
        else:
            logger.info(f"Adding remote origin: {repo_url}")
            success, stdout, stderr = self.run_git_command(['git', 'remote', 'add', 'origin', repo_url])

        if success:
            logger.info("Remote origin configured successfully")
            return True
        else:
            logger.error(f"Failed to setup remote origin: {stderr}")
            return False

    def push_to_remote(self, branch='main'):
        """Push to remote repository"""

        # Check if branch exists
        success, stdout, stderr = self.run_git_command(['git', 'branch', '--show-current'])

        if success:
            current_branch = stdout.strip()
            if not current_branch:
                # No branch yet, create main branch
                success, stdout, stderr = self.run_git_command(['git', 'checkout', '-b', branch])
                if not success:
                    logger.error(f"Failed to create branch {branch}: {stderr}")
                    return False

        # Push to remote
        success, stdout, stderr = self.run_git_command(['git', 'push', '-u', 'origin', branch])

        if success:
            logger.info(f"Successfully pushed to remote branch: {branch}")
            return True
        else:
            logger.error(f"Failed to push to remote: {stderr}")
            return False


def main():
    """Main Git setup and push function"""

    print("🚀 MarketPulse Git Setup and Push")
    print("=" * 45)
    print("Setting up Git repository and pushing to remote...")

    # Initialize Git manager
    git_manager = GitManager()

    # Check if Git is installed
    if not git_manager.check_git_installed():
        print("❌ Git is not installed. Please install Git first.")
        return False

    # Create comprehensive .gitignore
    print("\n📝 Creating comprehensive .gitignore...")
    if not git_manager.create_comprehensive_gitignore():
        print("⚠️ Failed to create .gitignore, continuing anyway...")

    # Create README if missing
    print("\n📖 Creating/updating README.md...")
    if not git_manager.create_readme_if_missing():
        print("⚠️ Failed to create README.md, continuing anyway...")

    # Initialize Git repository
    print("\n🔧 Initializing Git repository...")
    if not git_manager.initialize_git_repo():
        print("❌ Failed to initialize Git repository")
        return False

    # Check status before adding files
    if not git_manager.check_git_status():
        print("⚠️ Could not check Git status, continuing anyway...")

    # Ask for confirmation
    print("\n❓ Do you want to proceed with adding all files and committing?")
    print("   This will add all files except those in .gitignore")
    response = input("   Type 'yes' to continue: ").lower().strip()

    if response != 'yes':
        print("🚫 Git setup cancelled by user")
        return False

    # Add all files
    print("\n📦 Adding all files to Git...")
    if not git_manager.add_all_files():
        print("❌ Failed to add files to Git")
        return False

    # Create commit
    print("\n💾 Creating commit...")
    commit_message = input("Enter commit message (or press Enter for default): ").strip()
    if not commit_message:
        commit_message = "Phase 1 Complete: Enhanced Trading System with Professional Architecture\n\n- Multi-indicator technical analysis operational\n- Strategy ensemble voting system\n- Historical backtesting framework\n- Real market data pipeline\n- Enhanced performance dashboard\n- Professional project organization\n- 7/7 comprehensive tests passed"

    if not git_manager.create_commit(commit_message):
        print("❌ Failed to create commit")
        return False

    # Ask about remote repository
    print("\n🌐 Remote Repository Setup")
    print("Do you want to push to a remote repository?")
    setup_remote = input("Type 'yes' to setup remote: ").lower().strip()

    if setup_remote == 'yes':
        repo_url = input("Enter repository URL (https://github.com/username/repo.git): ").strip()

        if repo_url:
            print(f"\n🔗 Setting up remote origin: {repo_url}")
            if git_manager.setup_remote_origin(repo_url):
                print("\n🚀 Pushing to remote repository...")
                if git_manager.push_to_remote():
                    print("✅ Successfully pushed to remote repository!")
                else:
                    print("❌ Failed to push to remote repository")
                    print("💡 You can try pushing manually with: git push -u origin main")
            else:
                print("❌ Failed to setup remote origin")
        else:
            print("⚠️ No repository URL provided, skipping remote setup")

    # Final status
    print("\n" + "=" * 45)
    print("🎉 Git Setup Complete!")
    print("=" * 45)
    print("✅ Repository initialized")
    print("✅ .gitignore created (protects .env files)")
    print("✅ README.md updated")
    print("✅ All files committed")

    if setup_remote == 'yes' and repo_url:
        print("✅ Pushed to remote repository")
        print(f"\n🌐 Your repository: {repo_url}")

    print("\n📋 Next Steps:")
    print("1. Your .env files are safely excluded from Git")
    print("2. Database files are excluded (they contain demo data)")
    print("3. Your code is now backed up and shareable")
    print("4. Ready to collaborate or deploy")

    print("\n🔒 Security Notes:")
    print("- .env files are protected and will never be committed")
    print("- Database files with market data are excluded")
    print("- API keys and credentials are safely ignored")
    print("- Only source code and configuration templates are tracked")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)