"""
MarketPulse v2.0 - Main System Orchestrator
Integrated AI-Powered Trading System
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "01_CORE" / "antifragile"))
sys.path.insert(0, str(project_root / "05_EXECUTION" / "paper_trading"))
sys.path.insert(0, str(project_root / "05_EXECUTION" / "alerts"))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('marketpulse.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('MarketPulse')

class MarketPulseSystem:
    """
    Main MarketPulse System Controller
    """

    def __init__(self, mode: str = "paper"):
        """Initialize MarketPulse system"""
        self.mode = mode
        self.running = False

        logger.info("="*60)
        logger.info("   MarketPulse Trading System v2.0")
        logger.info(f"   Mode: {mode.upper()}")
        logger.info(f"   Time: {datetime.now()}")
        logger.info("="*60)

        # Load configuration
        self._load_configuration()

        # Initialize components
        self._initialize_components()

        logger.info("System initialized successfully!")

    def _load_configuration(self):
        """Load system configuration"""
        # Try to load config file
        config_file = project_root / "01_CORE" / "config" / "settings.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default configuration
            self.config = {
                'trading': {
                    'mode': self.mode,
                    'capital': 100000,
                    'watchlist': ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI']
                },
                'risk': {
                    'max_daily_loss': 0.02,
                    'max_position_size': 0.05,
                    'max_positions': 6,
                    'min_confidence': 0.6
                },
                'execution': {
                    'scan_interval': 60,  # seconds
                    'paper_trading': True,
                    'telegram_alerts': False
                }
            }
            logger.info("Using default configuration")

    def _initialize_components(self):
        """Initialize all trading components"""

        # Initialize Antifragile Framework
        self._init_ai_framework()

        # Initialize Trading Components
        self._init_trading_components()

        # Initialize Analysis Components
        self._init_analysis_components()

    def _init_ai_framework(self):
        """Initialize AI framework"""
        try:
            from core.failover_engine import FailoverEngine
            from providers.provider_registry import ProviderRegistry

            self.provider_registry = ProviderRegistry()
            self.failover_engine = FailoverEngine(self.provider_registry)
            logger.info("✅ Antifragile AI Framework initialized")
        except Exception as e:
            logger.warning(f"AI Framework not available: {e}")
            self.failover_engine = None

    def _init_trading_components(self):
        """Initialize trading components"""
        try:
            from paper_trading_engine import PaperTradingEngine
            self.trading_engine = PaperTradingEngine(
                initial_capital=self.config['trading']['capital']
            )
            logger.info("✅ Paper Trading Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            self.trading_engine = None

        try:
            from telegram_alerts import TelegramAlertSystem
            self.alert_system = TelegramAlertSystem()
            if self.alert_system.enabled:
                logger.info("✅ Telegram Alert System initialized")
            else:
                logger.info("ℹ️ Telegram alerts not configured")
        except Exception as e:
            logger.warning(f"Alert system not available: {e}")
            self.alert_system = None

    def _init_analysis_components(self):
        """Initialize analysis components"""
        try:
            from integrated_trading_system import IntegratedTradingSystem
            self.trading_system = IntegratedTradingSystem(
                initial_capital=self.config['trading']['capital']
            )
            logger.info("✅ Integrated Trading System initialized")
        except Exception as e:
            logger.warning(f"Trading strategies not available: {e}")
            self.trading_system = None

    async def start(self):
        """Start the trading system"""
        self.running = True
        logger.info("\nStarting trading system...")

        # Check components
        if not self.trading_engine:
            logger.error("Trading engine not available. Cannot start.")
            return

        # Main trading loop
        try:
            await self._trading_loop()
        except KeyboardInterrupt:
            logger.info("\nShutdown signal received")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            await self.stop()

    async def _trading_loop(self):
        """Main trading loop"""
        logger.info("Entering main trading loop...")
        scan_interval = self.config['execution']['scan_interval']

        while self.running:
            try:
                # Run market scan
                await self._scan_and_trade()

                # Show portfolio status
                self._show_portfolio_status()

                # Wait for next scan
                logger.info(f"Next scan in {scan_interval} seconds...")
                await asyncio.sleep(scan_interval)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)

    async def _scan_and_trade(self):
        """Scan market and execute trades"""
        logger.info(f"\n--- Market Scan at {datetime.now().strftime('%H:%M:%S')} ---")

        if self.trading_system:
            # Use integrated system
            self.trading_system.execute_signals()
        else:
            # Basic trading logic
            logger.info("Using basic trading logic...")

            for symbol in self.config['trading']['watchlist']:
                # Simple demo trade
                import random
                if random.random() > 0.8:  # 20% chance
                    action = random.choice(['BUY', 'SELL'])
                    quantity = random.randint(1, 5)

                    if self.trading_engine:
                        success, message, order = self.trading_engine.place_order(
                            symbol=symbol,
                            side=action,
                            quantity=quantity,
                            order_type='MARKET'
                        )
                        logger.info(f"{symbol}: {message}")

    def _show_portfolio_status(self):
        """Show current portfolio status"""
        if not self.trading_engine:
            return

        summary = self.trading_engine.get_portfolio_summary()
        positions = self.trading_engine.get_positions()

        logger.info("\n--- Portfolio Status ---")
        logger.info(f"Capital: ₹{summary['current_capital']:,.2f}")
        logger.info(f"Cash: ₹{summary['cash_available']:,.2f}")
        logger.info(f"Positions: {summary['total_positions']}/{self.config['risk']['max_positions']}")
        logger.info(f"Return: {summary['total_return_pct']:.2f}%")

        if positions:
            logger.info("\nOpen Positions:")
            for pos in positions:
                logger.info(f"  {pos['symbol']}: {pos['quantity']} @ ₹{pos['avg_price']:.2f} "
                          f"(P&L: ₹{pos['unrealized_pnl']:,.2f})")

    async def stop(self):
        """Stop the trading system"""
        logger.info("\nStopping MarketPulse...")
        self.running = False

        # Save portfolio snapshot
        if self.trading_engine:
            self.trading_engine.save_portfolio_snapshot()
            logger.info("Portfolio snapshot saved")

        # Send shutdown alert
        if self.alert_system and self.alert_system.enabled:
            self.alert_system.send_system_status(
                "MarketPulse system stopped",
                {'time': datetime.now().isoformat()}
            )

        logger.info("MarketPulse stopped successfully")

    def run_backtest(self, days: int = 30):
        """Run backtest mode"""
        logger.info(f"\n=== Running {days}-day Backtest ===")

        if self.trading_system:
            self.trading_system.run_backtest(days)
        else:
            logger.error("Trading system not available for backtest")

    def show_commands(self):
        """Show available commands"""
        print("\n" + "="*60)
        print("   MarketPulse Commands")
        print("="*60)
        print("  start    - Start live paper trading")
        print("  backtest - Run historical backtest")
        print("  status   - Show system status")
        print("  help     - Show this help")
        print("  exit     - Stop the system")
        print("="*60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='MarketPulse Trading System')
    parser.add_argument('command', nargs='?', default='help',
                       choices=['start', 'backtest', 'status', 'help'],
                       help='Command to execute')
    parser.add_argument('--mode', default='paper',
                       choices=['paper', 'live'],
                       help='Trading mode')
    parser.add_argument('--days', type=int, default=30,
                       help='Days for backtest')

    args = parser.parse_args()

    # Initialize system
    system = MarketPulseSystem(mode=args.mode)

    # Execute command
    if args.command == 'start':
        # Start trading
        asyncio.run(system.start())

    elif args.command == 'backtest':
        # Run backtest
        system.run_backtest(days=args.days)

    elif args.command == 'status':
        # Show status
        system._show_portfolio_status()

    elif args.command == 'help':
        # Show help
        system.show_commands()

    else:
        system.show_commands()


if __name__ == "__main__":
    main()