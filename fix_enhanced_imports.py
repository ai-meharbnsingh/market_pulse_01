"""
Fix Import Paths for Enhanced Components
Resolves Python module import issues after file organization

Location: #root/fix_enhanced_imports.py
"""

import sys
from pathlib import Path
import re

def fix_integration_file():
    """Fix import paths in integration file"""

    integration_file = Path('01_CORE/integration/enhanced_integration.py')

    if not integration_file.exists():
        print("❌ Integration file not found")
        return False

    try:
        with open(integration_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix import paths
        fixed_content = content.replace(
            'from enhanced_trading_system import',
            'sys.path.append(str(Path(__file__).parent.parent.parent / "02_ANALYSIS" / "enhanced"))\nfrom enhanced_trading_system import'
        ).replace(
            'from backtesting_framework import',
            'sys.path.append(str(Path(__file__).parent.parent.parent / "03_ML_ENGINE" / "backtesting"))\nfrom backtesting_framework import'
        )

        # Add necessary imports at the top
        if 'from pathlib import Path' not in fixed_content:
            fixed_content = 'from pathlib import Path\n' + fixed_content

        # Write back fixed content
        with open(integration_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print("✅ Fixed integration file imports")
        return True

    except Exception as e:
        print(f"❌ Failed to fix integration file: {e}")
        return False

def fix_test_file():
    """Fix import paths in test file"""

    test_file = Path('08_TESTS/phase1_step3/test_enhanced_strategies.py')

    if not test_file.exists():
        print("❌ Test file not found")
        return False

    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add proper import paths at the top
        import_fix = '''import sys
from pathlib import Path

# Add enhanced component paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "02_ANALYSIS" / "enhanced"))
sys.path.append(str(project_root / "03_ML_ENGINE" / "backtesting"))
sys.path.append(str(project_root / "07_DASHBOARD" / "enhanced"))
sys.path.append(str(project_root / "06_DATA" / "enhanced"))

'''

        # Find where to insert (after first few lines)
        lines = content.split('\n')

        # Check if already fixed
        if 'project_root =' in content:
            print("✅ Test file already has correct imports")
            return True

        # Insert after imports
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('logging.basicConfig'):
                insert_pos = i
                break

        lines.insert(insert_pos, import_fix)

        # Write back
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print("✅ Fixed test file imports")
        return True

    except Exception as e:
        print(f"❌ Failed to fix test file: {e}")
        return False

def fix_dashboard_imports():
    """Fix dashboard import paths"""

    dashboard_file = Path('07_DASHBOARD/enhanced/enhanced_dashboard.py')

    if not dashboard_file.exists():
        print("❌ Dashboard file not found")
        return False

    try:
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Fix the data fetcher import
        if 'sys.path.append(str(Path.cwd() / \'06_DATA\'))' in content:
            fixed_content = content.replace(
                'sys.path.append(str(Path.cwd() / \'06_DATA\'))',
                'sys.path.append(str(Path(__file__).parent.parent.parent / "06_DATA" / "enhanced"))'
            )

            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            print("✅ Fixed dashboard file imports")
        else:
            print("✅ Dashboard file already has correct imports")

        return True

    except Exception as e:
        print(f"❌ Failed to fix dashboard file: {e}")
        return False

def create_init_files():
    """Create __init__.py files for Python packages"""

    directories = [
        '02_ANALYSIS/enhanced',
        '03_ML_ENGINE/backtesting',
        '06_DATA/enhanced',
        '07_DASHBOARD/enhanced',
        '08_TESTS/phase1_step3'
    ]

    for directory in directories:
        dir_path = Path(directory)
        init_file = dir_path / '__init__.py'

        if not init_file.exists():
            try:
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(f'"""Enhanced MarketPulse Components - {directory}"""\\n')
                print(f"✅ Created {init_file}")
            except Exception as e:
                print(f"❌ Failed to create {init_file}: {e}")
        else:
            print(f"✅ {init_file} already exists")

def test_fixed_imports():
    """Test if the fixed imports work"""

    print("\\n🧪 Testing Fixed Imports...")
    print("-" * 30)

    # Test integration
    try:
        sys.path.append(str(Path('01_CORE/integration')))
        import enhanced_integration
        print("✅ Integration imports working")
    except Exception as e:
        print(f"❌ Integration import failed: {e}")

    # Test direct imports
    test_imports = [
        ('02_ANALYSIS/enhanced', 'enhanced_trading_system'),
        ('03_ML_ENGINE/backtesting', 'backtesting_framework'),
        ('07_DASHBOARD/enhanced', 'enhanced_dashboard')
    ]

    for path, module in test_imports:
        try:
            sys.path.append(str(Path(path)))
            __import__(module)
            print(f"✅ {module} import working")
        except Exception as e:
            print(f"❌ {module} import failed: {e}")

def main():
    """Main fix function"""

    print("🔧 MarketPulse Enhanced Components - Import Fix")
    print("=" * 50)
    print("Fixing Python import paths after file organization...")

    fixes = [
        ("Integration File", fix_integration_file),
        ("Test File", fix_test_file),
        ("Dashboard File", fix_dashboard_imports),
        ("Package Init Files", create_init_files)
    ]

    results = {}

    for fix_name, fix_func in fixes:
        print(f"\\n🔧 Fixing {fix_name}...")
        results[fix_name] = fix_func()

    # Test fixes
    test_fixed_imports()

    # Summary
    print(f"\\n{'='*50}")
    print("🎉 IMPORT FIX COMPLETE!")
    print("="*50)

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    print(f"✅ Fixes applied: {successful}/{total}")

    if successful == total:
        print("\\n🚀 All imports fixed successfully!")
        print("\\n📋 Now try:")
        print("1. python 01_CORE/integration/enhanced_integration.py")
        print("2. python 08_TESTS/phase1_step3/test_enhanced_strategies.py")
        print("3. streamlit run 07_DASHBOARD/enhanced/enhanced_dashboard.py")
    else:
        print("\\n⚠️ Some fixes failed. Check errors above.")

    return successful == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)