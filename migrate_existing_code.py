"""
Script to copy your existing code to new structure
Run from your new MarketPulse directory
"""
import shutil
import os
from pathlib import Path

# Define source and destination paths
OLD_PROJECT = r"D:\Users\OMEN\Trading_App"
NEW_PROJECT = r"D:\Users\OMEN\MarketPulse"

# Migration mapping
COPY_MAP = {
    # Antifragile Framework - YOUR CROWN JEWEL
    f"{OLD_PROJECT}/01_Framework_Core/antifragile_framework":
        f"{NEW_PROJECT}/01_CORE/antifragile",

    # Technical Analysis
    f"{OLD_PROJECT}/src/ai_trading/professional_technical_analyzer.py":
        f"{NEW_PROJECT}/02_ANALYSIS/technical/indicators.py",

    # ML Models
    f"{OLD_PROJECT}/src/models/alpha_model/alpha_core.py":
        f"{NEW_PROJECT}/03_ML_ENGINE/models/alpha_model.py",
    f"{OLD_PROJECT}/src/models/timeseries_forecaster.py":
        f"{NEW_PROJECT}/03_ML_ENGINE/models/lstm_intraday.py",

    # Risk Management
    f"{OLD_PROJECT}/src/ai_trading/risk_calculator.py":
        f"{NEW_PROJECT}/04_RISK/risk_calculator.py",

    # Data Streaming
    f"{OLD_PROJECT}/src/data/streaming/websocket_service.py":
        f"{NEW_PROJECT}/06_DATA/streaming/websocket_service.py",

    # Dashboard
    f"{OLD_PROJECT}/src/dashboard":
        f"{NEW_PROJECT}/07_DASHBOARD/components",
}


def migrate_code():
    """Copy existing code to new structure"""
    for source, dest in COPY_MAP.items():
        try:
            if os.path.isdir(source):
                shutil.copytree(source, dest, dirs_exist_ok=True)
                print(f"✅ Copied directory: {source} → {dest}")
            elif os.path.isfile(source):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(source, dest)
                print(f"✅ Copied file: {source} → {dest}")
        except Exception as e:
            print(f"❌ Failed to copy {source}: {e}")


if __name__ == "__main__":
    migrate_code()
    print("\n✅ Migration complete! Review and update imports.")