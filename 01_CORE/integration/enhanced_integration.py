"""
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
    sys.path.append(str(Path(__file__).parent.parent.parent / "02_ANALYSIS" / "enhanced"))
from enhanced_trading_system import EnhancedTradingSystem
    sys.path.append(str(Path(__file__).parent.parent.parent / "03_ML_ENGINE" / "backtesting"))
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
