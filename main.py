"""
MarketPulse v2.0 - Main System Orchestrator
Windows-Compatible Version (Unicode Fixed)

Location: #main.py
"""

import os
import sys
import asyncio
import logging
import importlib.util
from datetime import datetime
from pathlib import Path
import argparse

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging with Windows-compatible encoding
log_handlers = [
    logging.FileHandler('marketpulse.log', encoding='utf-8')
]

# Only add console handler if we can handle Unicode
try:
    # Test if console can handle Unicode
    sys.stdout.write('\u2713')
    sys.stdout.flush()
    log_handlers.append(logging.StreamHandler())
except UnicodeEncodeError:
    # Console can't handle Unicode, skip console logging
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger('MarketPulse')

# Enhanced Trading System Integration with proper import handling
ENHANCED_TRADING_AVAILABLE = False
EnhancedTradingIntegration = None

def safe_import_module(module_path, class_name):
    """Safely import modules from numbered directories"""
    try:
        spec = importlib.util.spec_from_file_location("temp_module", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, class_name, None)
    except Exception as e:
        logger.debug(f"Failed to import {class_name} from {module_path}: {e}")
    return None

# Try to load enhanced trading components
enhanced_integration_path = project_root / "01_CORE" / "integration" / "enhanced_integration.py"
if enhanced_integration_path.exists():
    EnhancedTradingIntegration = safe_import_module(enhanced_integration_path, "EnhancedTradingIntegration")
    if EnhancedTradingIntegration:
        ENHANCED_TRADING_AVAILABLE = True
        logger.info("Enhanced trading components loaded")

if not ENHANCED_TRADING_AVAILABLE:
    logger.warning("Enhanced trading components not available")


class MarketPulseSystem:
    """
    Main MarketPulse System Controller - Windows Compatible
    """

    def __init__(self, mode: str = "paper"):
        self.mode = mode
        self.running = False
        self.components = {}

        logger.info(f"Initializing MarketPulse System (Mode: {mode})")

        # Initialize core components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all system components with safe imports"""

        try:
            # Initialize data components
            self._init_data_components()

            # Initialize ML components
            self._init_ml_components()

            # Initialize trading components
            self._init_trading_components()

            # Initialize dashboard components
            self._init_dashboard_components()

            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise

    def _init_data_components(self):
        """Initialize data fetching and storage components"""

        # Try to load Indian stock universe
        indian_universe_path = project_root / "06_DATA" / "indian_stock_universe.py"
        if indian_universe_path.exists():
            IndianStockUniverse = safe_import_module(indian_universe_path, "IndianStockUniverse")
            if IndianStockUniverse:
                self.components['data_source'] = IndianStockUniverse()
                logger.info("Indian Stock Universe loaded")

    def _init_ml_components(self):
        """Initialize ML model components"""

        # Check for trained models
        model_dir = project_root / "03_ML_ENGINE" / "trained_models"
        alpha_model_path = model_dir / "alpha_model_ensemble_fixed.pkl"

        if alpha_model_path.exists():
            try:
                import pickle
                with open(alpha_model_path, 'rb') as f:
                    self.components['ml_models'] = pickle.load(f)
                logger.info("Trained ML models loaded")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}")

    def _init_trading_components(self):
        """Initialize paper trading components"""

        # Try to load paper trading engine
        paper_trading_path = project_root / "05_EXECUTION" / "paper_trading"
        if paper_trading_path.exists():
            # Look for paper trading modules
            for py_file in paper_trading_path.glob("*.py"):
                if "paper" in py_file.name.lower() and "trading" in py_file.name.lower():
                    try:
                        module = safe_import_module(py_file, "PaperTradingEngine")
                        if module:
                            self.components['paper_trading'] = module()
                            logger.info("Paper trading engine loaded")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to load {py_file}: {e}")

    def _init_dashboard_components(self):
        """Initialize dashboard components"""

        dashboard_path = project_root / "07_DASHBOARD" / "dashboard_app.py"
        if dashboard_path.exists():
            logger.info("Dashboard available")
            self.components['dashboard_available'] = True

    async def start_system(self):
        """Start the MarketPulse trading system"""

        logger.info("Starting MarketPulse Trading System...")
        self.running = True

        try:
            # Start data monitoring
            if 'data_source' in self.components:
                logger.info("Starting data monitoring...")
                # Add your data monitoring logic here

            # Start ML predictions
            if 'ml_models' in self.components:
                logger.info("Starting ML prediction engine...")
                # Add your ML prediction logic here

            # Start paper trading
            if 'paper_trading' in self.components:
                logger.info("Starting paper trading...")
                # Add your paper trading logic here

            # Main system loop
            await self._main_loop()

        except KeyboardInterrupt:
            logger.info("System shutdown requested")
            await self.stop_system()
        except Exception as e:
            logger.error(f"System error: {e}")
            await self.stop_system()

    async def _main_loop(self):
        """Main system execution loop"""

        logger.info("Entering main system loop...")

        while self.running:
            try:
                # System health check
                await self._health_check()

                # Process trading signals
                if 'ml_models' in self.components:
                    await self._process_trading_signals()

                # Update portfolio
                if 'paper_trading' in self.components:
                    await self._update_portfolio()

                # Wait before next iteration
                await asyncio.sleep(60)  # 1 minute intervals

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)

    async def _health_check(self):
        """Perform system health check"""

        # Check database connectivity
        try:
            if 'data_source' in self.components:
                stats = self.components['data_source'].get_training_statistics()
                if stats.get('total_stocks', 0) > 0:
                    logger.debug("Database connectivity OK")
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")

    async def _process_trading_signals(self):
        """Process ML trading signals"""

        # Placeholder for ML signal processing
        logger.debug("Processing ML trading signals...")

    async def _update_portfolio(self):
        """Update portfolio status"""

        # Placeholder for portfolio updates
        logger.debug("Updating portfolio status...")

    async def stop_system(self):
        """Stop the MarketPulse system gracefully"""

        logger.info("Stopping MarketPulse system...")
        self.running = False

        # Cleanup components
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                logger.info(f"{component_name} stopped")
            except Exception as e:
                logger.warning(f"Error stopping {component_name}: {e}")

    def show_status(self):
        """Show system status"""

        print("\nMARKETPULSE SYSTEM STATUS")
        print("=" * 40)
        print(f"Mode: {self.mode}")
        print(f"Running: {self.running}")
        print(f"Components: {len(self.components)}")

        for name, component in self.components.items():
            if name == 'ml_models':
                models = component.get('models', {})
                print(f"  ML Models: {len(models)} trained")
            elif name == 'data_source':
                try:
                    stats = component.get_training_statistics()
                    print(f"  Data: {stats.get('total_stocks', 0)} stocks")
                except:
                    print(f"  Data: Available")
            else:
                print(f"  {name}: Ready")

        print()


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(description='MarketPulse Trading System')
    parser.add_argument('action', choices=['start', 'stop', 'status'],
                       help='Action to perform')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')

    args = parser.parse_args()

    # Initialize system
    system = MarketPulseSystem(mode=args.mode)

    if args.action == 'status':
        system.show_status()

    elif args.action == 'start':
        logger.info("Starting MarketPulse...")
        try:
            asyncio.run(system.start_system())
        except KeyboardInterrupt:
            logger.info("MarketPulse stopped by user")

    elif args.action == 'stop':
        logger.info("Stop command received")
        # In a real implementation, this would send a stop signal
        print("Send CTRL+C to running MarketPulse instance")


if __name__ == "__main__":
    main()