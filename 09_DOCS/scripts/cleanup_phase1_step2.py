"""
MarketPulse Phase 1, Step 2 - Cleanup Script
Removes all temporary and test files, keeps only essential working files

Location: #root/cleanup_phase1_step2.py
"""

import os
from pathlib import Path
import shutil


def get_files_to_delete():
    """List all files that should be deleted"""

    # Files created during today's session that are no longer needed
    cleanup_files = [
        # Test and fix scripts (temporary)
        'clean_and_test_db.py',
        'fixed_clean_and_test_db.py',
        'updated_native_database_setup.py',
        'verify_database_completion.py',
        'integrate_real_data.py',
        'integrated_trading_system_real.py',
        'test_real_data_integration.py',
        'quick_verify_real_data.py',
        'phase1_step2_setup_guide.py',
        'emergency_fix.py',
        'quick_fix_data_issues.py',
        'data_fetcher_fixed.py',
        'demo_mode_integration.py',
        'fix_timestamp_issue.py',
        'phase1_step2_completion.py',
        'simple_integration_test.py',

        # Backup files (if any)
        'integrated_trading_system.py.backup',
        'main.py.backup',

        # Test files
        'test_tciker.py',

        # Any .pyc files
        '__pycache__',
    ]

    # Text files and guides (temporary)
    text_files = [
        'PHASE1_STEP2_SETUP_GUIDE.txt',
        'SOLUTION_GUIDE.py',
    ]

    # Backup directories (if created)
    backup_dirs = [
        'backups',
    ]

    return cleanup_files, text_files, backup_dirs


def get_files_to_keep():
    """List essential files that should be kept"""

    essential_files = [
        # Core system files
        'main.py',
        'integrated_trading_system.py',
        'requirements.txt',
        '.env',
        '.gitignore',
        'README.md',

        # Database files
        'marketpulse.db',
        'native_database_setup.py',

        # Working data pipeline
        '06_DATA/data_fetcher.py',

        # Documentation
        'master_doc.md',
        'mast_doc.md',
        'expert_validated_roadmap.md',
        'project_structure.txt',

        # Core components
        'paper_trading_engine.py',
        'risk_calculator.py',
        'telegram_alerts.py',
        'dashboard_app.py',

        # Logs (recent)
        'marketpulse.log',
        'test_report.json',

        # Verification scripts
        'verify_bot.py',
        'verify_keys.py',
        'show_structure.py',
    ]

    return essential_files


def preview_cleanup():
    """Show what will be deleted vs kept"""

    cleanup_files, text_files, backup_dirs = get_files_to_delete()
    essential_files = get_files_to_keep()

    print("🗑️ MarketPulse Phase 1, Step 2 - Cleanup Preview")
    print("=" * 55)

    # Check what exists and will be deleted
    files_to_delete = []
    dirs_to_delete = []

    print("\n📋 FILES TO DELETE:")
    print("-" * 25)

    for file_name in cleanup_files + text_files:
        file_path = Path(file_name)
        if file_path.exists():
            if file_path.is_file():
                files_to_delete.append(file_path)
                size = file_path.stat().st_size
                print(f"   🗑️ {file_name} ({size:,} bytes)")
            elif file_path.is_dir():
                dirs_to_delete.append(file_path)
                print(f"   📁 {file_name}/ (directory)")
        else:
            print(f"   ⚪ {file_name} (not found)")

    # Check backup directories
    for dir_name in backup_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            dirs_to_delete.append(dir_path)
            file_count = len(list(dir_path.rglob('*')))
            print(f"   📁 {dir_name}/ ({file_count} items)")

    print(f"\n📊 CLEANUP SUMMARY:")
    print(f"   🗑️ Files to delete: {len(files_to_delete)}")
    print(f"   📁 Directories to delete: {len(dirs_to_delete)}")

    # Calculate space savings
    total_size = 0
    for file_path in files_to_delete:
        if file_path.is_file():
            total_size += file_path.stat().st_size

    print(f"   💾 Space to free: {total_size:,} bytes ({total_size / 1024:.1f} KB)")

    print("\n✅ ESSENTIAL FILES (WILL KEEP):")
    print("-" * 35)

    kept_files = []
    missing_essential = []

    for file_name in essential_files:
        file_path = Path(file_name)
        if file_path.exists():
            kept_files.append(file_path)
            print(f"   ✅ {file_name}")
        else:
            missing_essential.append(file_name)
            print(f"   ⚠️ {file_name} (missing - should exist)")

    if missing_essential:
        print(f"\n⚠️ WARNING: {len(missing_essential)} essential files are missing!")
        print("   You may need to recreate these files.")

    return files_to_delete, dirs_to_delete, kept_files


def perform_cleanup(files_to_delete, dirs_to_delete):
    """Actually delete the files"""

    deleted_files = []
    deleted_dirs = []
    errors = []

    print("\n🗑️ PERFORMING CLEANUP...")
    print("-" * 30)

    # Delete files
    for file_path in files_to_delete:
        try:
            if file_path.exists():
                file_path.unlink()
                deleted_files.append(str(file_path))
                print(f"   ✅ Deleted: {file_path}")
        except Exception as e:
            errors.append(f"Failed to delete {file_path}: {e}")
            print(f"   ❌ Error: {file_path} - {e}")

    # Delete directories
    for dir_path in dirs_to_delete:
        try:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                deleted_dirs.append(str(dir_path))
                print(f"   ✅ Deleted directory: {dir_path}")
        except Exception as e:
            errors.append(f"Failed to delete directory {dir_path}: {e}")
            print(f"   ❌ Error: {dir_path} - {e}")

    # Clean up __pycache__ directories
    try:
        for pycache in Path('.').rglob('__pycache__'):
            if pycache.is_dir():
                shutil.rmtree(pycache)
                deleted_dirs.append(str(pycache))
                print(f"   ✅ Deleted cache: {pycache}")
    except Exception as e:
        errors.append(f"Failed to clean __pycache__: {e}")

    # Clean up .pyc files
    try:
        for pyc_file in Path('.').rglob('*.pyc'):
            if pyc_file.is_file():
                pyc_file.unlink()
                deleted_files.append(str(pyc_file))
                print(f"   ✅ Deleted: {pyc_file}")
    except Exception as e:
        errors.append(f"Failed to clean .pyc files: {e}")

    return deleted_files, deleted_dirs, errors


def show_final_status():
    """Show clean project structure"""

    print("\n📁 CLEAN PROJECT STRUCTURE:")
    print("=" * 35)

    # Show essential directories and key files
    structure = {
        'Root': ['main.py', 'integrated_trading_system.py', 'requirements.txt', 'marketpulse.db'],
        '01_CORE/': ['antifragile/ (framework)'],
        '02_ANALYSIS/': ['technical/indicators.py'],
        '03_ML_ENGINE/': ['models/ (alpha_model.py, lstm_intraday.py)'],
        '04_RISK/': ['risk_calculator.py'],
        '05_EXECUTION/': ['paper_trading/', 'alerts/'],
        '06_DATA/': ['data_fetcher.py'],
        '07_DASHBOARD/': ['dashboard_app.py'],
        '10_DATA_STORAGE/': ['logs/', 'models/']
    }

    for category, files in structure.items():
        print(f"\n📂 {category}")
        for file_item in files:
            if Path(file_item.split()[0]).exists():
                print(f"   ✅ {file_item}")
            else:
                print(f"   📁 {file_item}")


def main():
    """Main cleanup function"""

    print("🧹 MarketPulse Phase 1, Step 2 - PROJECT CLEANUP")
    print("=" * 55)
    print("This script will remove temporary files created during today's session")
    print("and keep only the essential working files for your trading system.")

    # Preview what will be deleted
    files_to_delete, dirs_to_delete, kept_files = preview_cleanup()

    if not files_to_delete and not dirs_to_delete:
        print("\n✨ PROJECT ALREADY CLEAN!")
        print("No temporary files found to delete.")
        show_final_status()
        return True

    # Ask for confirmation
    print(f"\n❓ PROCEED WITH CLEANUP?")
    print(f"This will permanently delete {len(files_to_delete)} files and {len(dirs_to_delete)} directories.")

    response = input("Type 'yes' to proceed, 'no' to cancel: ").lower().strip()

    if response != 'yes':
        print("\n🚫 Cleanup cancelled by user.")
        return False

    # Perform cleanup
    deleted_files, deleted_dirs, errors = perform_cleanup(files_to_delete, dirs_to_delete)

    # Show results
    print(f"\n🎉 CLEANUP COMPLETE!")
    print("=" * 25)
    print(f"   ✅ Files deleted: {len(deleted_files)}")
    print(f"   ✅ Directories deleted: {len(dirs_to_delete)}")

    if errors:
        print(f"   ⚠️ Errors: {len(errors)}")
        for error in errors:
            print(f"      - {error}")

    print(f"\n📁 Your project is now clean and ready for Phase 1, Step 3!")

    # Show final structure
    show_final_status()

    print(f"\n🚀 READY FOR NEXT PHASE:")
    print("   - Enhanced trading strategies")
    print("   - Technical indicators integration")
    print("   - Backtesting framework")
    print("   - Performance analytics")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)