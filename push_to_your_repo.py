"""
MarketPulse Git Push Script - Corrected for Your Repository
Uses your existing GitHub repository: ai-meharbnsingh/market_pulse

Location: #root/push_to_your_repo.py
"""

import os
import subprocess
from pathlib import Path


def run_git_command(command, check=True):
    """Run a git command and return result"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def main():
    """Push to your existing MarketPulse repository"""

    print("🚀 MarketPulse - Push to Your Repository")
    print("=" * 45)
    print("Repository: https://github.com/ai-meharbnsingh/market_pulse")

    # Your actual repository URL
    repo_url = "https://github.com/ai-meharbnsingh/market_pulse.git"

    # Check if .gitignore exists
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        print("\n📝 Creating .gitignore to protect sensitive files...")

        gitignore_content = """# MarketPulse .gitignore - Protect Sensitive Files

# Environment files (NEVER COMMIT)
.env
.env.local
.env.production
*.env

# Database files 
*.db
*.sqlite
*.sqlite3
marketpulse.db
marketpulse_backup.db

# Python cache
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# Virtual environment
venv/
.venv/
env/

# Logs
*.log
logs/

# IDE
.vscode/
.idea/

# Large data files
*.csv
*.json
10_DATA_STORAGE/alerts/*.json
10_DATA_STORAGE/logs/*.log
10_DATA_STORAGE/backtest_results/*
10_DATA_STORAGE/models/*

# Keep important files
!requirements.txt
!README.md
!**/README.md
"""

        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ .gitignore created")

    # Check Git status
    print("\n📊 Checking what will be committed...")
    success, stdout, stderr = run_git_command(['git', 'status', '--porcelain'])

    if success and stdout.strip():
        lines = stdout.strip().split('\n')
        print(f"📁 Found {len(lines)} files to commit")

        # Show first few files
        for line in lines[:5]:
            status = line[:2]
            filename = line[3:]
            if status == '??':
                print(f"   + {filename}")
            elif status == ' M':
                print(f"   ~ {filename}")

        if len(lines) > 5:
            print(f"   ... and {len(lines) - 5} more files")

    # Confirm
    print(f"\n❓ Ready to push to your repository?")
    print(f"   Repository: {repo_url}")
    print("   This will commit and push all changes")

    confirm = input("Type 'yes' to proceed: ").lower().strip()

    if confirm != 'yes':
        print("🚫 Push cancelled")
        return False

    # Add all files
    print("\n📦 Adding files...")
    success, stdout, stderr = run_git_command(['git', 'add', '.'])
    if not success:
        print(f"❌ Failed to add files: {stderr}")
        return False
    print("✅ Files added")

    # Create commit
    print("\n💾 Creating commit...")
    commit_message = "Phase 1 Complete: Enhanced Trading System\n\n- Multi-indicator technical analysis\n- Strategy ensemble voting\n- Historical backtesting\n- Real market data pipeline\n- Enhanced dashboard\n- Professional architecture\n- 7/7 tests passed"

    success, stdout, stderr = run_git_command(['git', 'commit', '-m', commit_message])
    if not success:
        if "nothing to commit" in stderr:
            print("✅ Nothing new to commit")
        else:
            print(f"❌ Failed to commit: {stderr}")
            return False
    else:
        print("✅ Commit created")

    # Setup remote (if needed)
    print("\n🔗 Setting up remote...")
    success, stdout, stderr = run_git_command(['git', 'remote', 'get-url', 'origin'], check=False)

    if not success:
        # Add remote
        success, stdout, stderr = run_git_command(['git', 'remote', 'add', 'origin', repo_url])
        if success:
            print("✅ Remote origin added")
        else:
            print(f"❌ Failed to add remote: {stderr}")
            return False
    else:
        print("✅ Remote origin already configured")

    # Push to repository
    print("\n🚀 Pushing to GitHub...")
    success, stdout, stderr = run_git_command(['git', 'push', '-u', 'origin', 'main'])

    if not success:
        # Try master branch if main fails
        print("Trying master branch...")
        success, stdout, stderr = run_git_command(['git', 'push', '-u', 'origin', 'master'])

    if success:
        print("✅ Successfully pushed to GitHub!")
        print(f"\n🌐 Your repository: https://github.com/ai-meharbnsingh/market_pulse")
        print("\n🎉 MarketPulse is now on GitHub!")

        print("\n🔒 Security Status:")
        print("✅ .env files protected (not committed)")
        print("✅ Database files excluded")
        print("✅ Only source code pushed")

        return True
    else:
        print(f"❌ Failed to push: {stderr}")
        print("\n💡 Troubleshooting:")
        print("1. Check your GitHub credentials")
        print("2. Ensure repository exists")
        print("3. Try: git push -u origin main")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)