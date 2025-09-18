"""
MarketPulse Integrated Trading Strategy
Connects ML models, technical analysis, and paper trading
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import asyncio

# Setup paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent / "05_EXECUTION" / "paper_trading"))
sys.path.append(str(current_dir.parent / "05_EXECUTION" / "alerts"))
sys.path.append(str(current_dir.parent / "02_ANALYSIS" / "technical"))
sys.path.append(str(current_dir.parent / "03_ML_ENGINE" / "models"))
sys.path.append(str(current_dir.parent / "04_RISK"))

# Import components
try:
    from paper_trading_engine import PaperTradingEngine
except:
    print("Paper trading engine not found")
    PaperTradingEngine = None

try:
    from indicators import TechnicalAnalyzer
except:
    # Create simple technical analyzer
    class TechnicalAnalyzer:
        def calculate_rsi(self, prices, period=14):
            """Calculate RSI"""
            deltas = np.diff(prices)
            seed = deltas[:period + 1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 100
            rsi = 100 - (100 / (1 + rs))
            return rsi

        def calculate_sma(self, prices, period=20):
            """Calculate Simple Moving Average"""
            return np.mean(prices[-period:]) if len(prices) >= period else prices[-1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingStrategy:
    """
    Base Trading Strategy Class
    Combines technical, fundamental, and ML signals
    """

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.technical_analyzer = TechnicalAnalyzer()
        self.signals = []
        self.active_positions = {}

    def analyze(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Analyze symbol and generate signals

        Returns:
            Dict with signal, confidence, entry, target, stop_loss
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def should_enter(self, symbol: str, data: pd.DataFrame) -> bool:
        """Check if should enter position"""
        return False

    def should_exit(self, symbol: str, data: pd.DataFrame) -> bool:
        """Check if should exit position"""
        return False


class MomentumStrategy(TradingStrategy):
    """
    Momentum Trading Strategy
    Buys when momentum is strong and RSI confirms
    """

    def __init__(self):
        super().__init__(name="MomentumStrategy")
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_volume_ratio = 1.2  # 20% above average

    def analyze(self, symbol: str, prices: List[float], volumes: List[float] = None) -> Dict:
        """
        Analyze using momentum indicators
        """
        if len(prices) < 20:
            return {'signal': 'HOLD', 'confidence': 0.0}

        # Calculate indicators
        current_price = prices[-1]
        sma_20 = self.technical_analyzer.calculate_sma(prices, 20)
        sma_50 = self.technical_analyzer.calculate_sma(prices, 50) if len(prices) >= 50 else sma_20
        rsi = self.technical_analyzer.calculate_rsi(prices)

        # Calculate momentum
        momentum = (current_price - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0

        # Generate signal
        signal = 'HOLD'
        confidence = 0.0

        # BUY conditions
        if (current_price > sma_20 and
                sma_20 > sma_50 and
                rsi < 65 and rsi > 35 and
                momentum > 1):
            signal = 'BUY'
            confidence = min(0.8, momentum / 10)

        # SELL conditions
        elif (current_price < sma_20 or
              rsi > self.rsi_overbought or
              momentum < -2):
            signal = 'SELL'
            confidence = 0.7

        # Calculate targets
        atr = np.std(prices[-14:]) if len(prices) >= 14 else current_price * 0.02

        return {
            'signal': signal,
            'confidence': confidence,
            'entry': current_price,
            'target': current_price + (2 * atr) if signal == 'BUY' else current_price - (2 * atr),
            'stop_loss': current_price - atr if signal == 'BUY' else current_price + atr,
            'indicators': {
                'rsi': rsi,
                'sma_20': sma_20,
                'momentum': momentum
            }
        }


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Strategy
    Buys oversold, sells overbought
    """

    def __init__(self):
        super().__init__(name="MeanReversionStrategy")
        self.bb_period = 20
        self.bb_std = 2

    def analyze(self, symbol: str, prices: List[float], volumes: List[float] = None) -> Dict:
        """
        Analyze using Bollinger Bands for mean reversion
        """
        if len(prices) < 20:
            return {'signal': 'HOLD', 'confidence': 0.0}

        current_price = prices[-1]
        sma = self.technical_analyzer.calculate_sma(prices, self.bb_period)
        std = np.std(prices[-self.bb_period:])

        # Bollinger Bands
        upper_band = sma + (self.bb_std * std)
        lower_band = sma - (self.bb_std * std)

        # Calculate position in band
        band_width = upper_band - lower_band
        position_in_band = (current_price - lower_band) / band_width if band_width > 0 else 0.5

        # RSI for confirmation
        rsi = self.technical_analyzer.calculate_rsi(prices)

        signal = 'HOLD'
        confidence = 0.0

        # BUY when oversold
        if position_in_band < 0.2 and rsi < 35:
            signal = 'BUY'
            confidence = 0.7

        # SELL when overbought
        elif position_in_band > 0.8 and rsi > 65:
            signal = 'SELL'
            confidence = 0.7

        return {
            'signal': signal,
            'confidence': confidence,
            'entry': current_price,
            'target': sma,  # Target is mean
            'stop_loss': lower_band - std if signal == 'BUY' else upper_band + std,
            'indicators': {
                'upper_band': upper_band,
                'lower_band': lower_band,
                'sma': sma,
                'position_in_band': position_in_band,
                'rsi': rsi
            }
        }


class IntegratedTradingSystem:
    """
    Main Trading System that combines all strategies
    """

    def __init__(self, initial_capital: float = 100000):
        """Initialize integrated system"""
        self.strategies = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy()
        }

        # Initialize paper trading
        self.paper_trading = PaperTradingEngine(initial_capital) if PaperTradingEngine else None

        # Watchlist
        self.watchlist = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI']

        # Risk parameters
        self.max_positions = 6
        self.max_position_size = 0.05  # 5% of capital
        self.min_confidence = 0.6  # Minimum confidence to trade

        logger.info(f"Integrated Trading System initialized with ₹{initial_capital:,.0f}")

    def get_market_data(self, symbol: str) -> Dict:
        """
        Get market data for symbol
        In production, this would connect to real data feed
        """
        # Generate sample data for testing
        import random

        base_prices = {
            'RELIANCE': 2500,
            'TCS': 3500,
            'INFY': 1500,
            'HDFC': 1600,
            'ICICI': 950
        }

        base = base_prices.get(symbol, 1000)

        # Generate 50 price points
        prices = []
        for i in range(50):
            change = random.uniform(-0.02, 0.02)
            base = base * (1 + change)
            prices.append(base)

        # Generate volumes
        volumes = [random.randint(1000000, 5000000) for _ in range(50)]

        return {
            'symbol': symbol,
            'prices': prices,
            'volumes': volumes,
            'current_price': prices[-1]
        }

    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Analyze symbol using all strategies
        """
        # Get market data
        data = self.get_market_data(symbol)

        # Run all strategies
        all_signals = {}
        for name, strategy in self.strategies.items():
            signal = strategy.analyze(symbol, data['prices'], data['volumes'])
            all_signals[name] = signal

        # Combine signals (ensemble)
        buy_votes = sum(1 for s in all_signals.values() if s['signal'] == 'BUY')
        sell_votes = sum(1 for s in all_signals.values() if s['signal'] == 'SELL')

        # Average confidence
        avg_confidence = np.mean([s['confidence'] for s in all_signals.values()])

        # Determine final signal
        if buy_votes > sell_votes and buy_votes > len(self.strategies) / 2:
            final_signal = 'BUY'
        elif sell_votes > buy_votes and sell_votes > len(self.strategies) / 2:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'

        return {
            'symbol': symbol,
            'final_signal': final_signal,
            'confidence': avg_confidence,
            'current_price': data['current_price'],
            'strategy_signals': all_signals
        }

    def execute_signals(self):
        """
        Scan watchlist and execute high-confidence signals
        """
        if not self.paper_trading:
            logger.warning("Paper trading not available")
            return

        logger.info("Scanning watchlist for signals...")

        for symbol in self.watchlist:
            analysis = self.analyze_symbol(symbol)

            # Log analysis
            logger.info(f"{symbol}: {analysis['final_signal']} "
                        f"(confidence: {analysis['confidence']:.2f})")

            # Execute if high confidence
            if analysis['confidence'] >= self.min_confidence:
                if analysis['final_signal'] == 'BUY':
                    # Check if we can buy
                    portfolio = self.paper_trading.get_portfolio_summary()
                    if portfolio['total_positions'] < self.max_positions:
                        # Calculate position size
                        position_value = portfolio['current_capital'] * 0.02  # 2% per trade
                        quantity = int(position_value / analysis['current_price'])

                        if quantity > 0:
                            success, message, order = self.paper_trading.place_order(
                                symbol=symbol,
                                side='BUY',
                                quantity=quantity,
                                order_type='MARKET'
                            )
                            logger.info(f"BUY order: {message}")

                elif analysis['final_signal'] == 'SELL':
                    # Check if we have position to sell
                    positions = self.paper_trading.get_positions()
                    for pos in positions:
                        if pos['symbol'] == symbol:
                            success, message, order = self.paper_trading.place_order(
                                symbol=symbol,
                                side='SELL',
                                quantity=pos['quantity'],
                                order_type='MARKET'
                            )
                            logger.info(f"SELL order: {message}")

    def run_backtest(self, days: int = 30):
        """
        Run backtest on historical data
        """
        logger.info(f"Running {days}-day backtest...")

        # This would use historical data in production
        # For now, simulate multiple trading sessions
        for day in range(days):
            logger.info(f"\n--- Day {day + 1} ---")
            self.execute_signals()

            # Update portfolio (simulate price changes)
            if self.paper_trading:
                market_data = {symbol: self.get_market_data(symbol)['current_price']
                               for symbol in self.watchlist}
                self.paper_trading.update_portfolio(market_data)

        # Show results
        if self.paper_trading:
            summary = self.paper_trading.get_portfolio_summary()
            logger.info("\n=== Backtest Results ===")
            logger.info(f"Final Capital: ₹{summary['current_capital']:,.2f}")
            logger.info(f"Total Return: {summary['total_return_pct']:.2f}%")
            logger.info(f"Win Rate: {summary['win_rate']:.2f}%")
            logger.info(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")

    def start_live_scanning(self, interval_seconds: int = 60):
        """
        Start live market scanning
        """
        logger.info("Starting live market scanning...")

        async def scan_loop():
            while True:
                try:
                    self.execute_signals()
                    await asyncio.sleep(interval_seconds)
                except KeyboardInterrupt:
                    logger.info("Stopping scanner...")
                    break
                except Exception as e:
                    logger.error(f"Error in scan loop: {e}")
                    await asyncio.sleep(interval_seconds)

        asyncio.run(scan_loop())


def test_integrated_system():
    """Test the integrated trading system"""
    print("\n=== Testing Integrated Trading System ===\n")

    # Initialize system
    system = IntegratedTradingSystem(initial_capital=100000)

    # Test single symbol analysis
    print("Testing single symbol analysis...")
    analysis = system.analyze_symbol('RELIANCE')
    print(f"\nRELIANCE Analysis:")
    print(f"  Signal: {analysis['final_signal']}")
    print(f"  Confidence: {analysis['confidence']:.2%}")
    print(f"  Current Price: ₹{analysis['current_price']:.2f}")

    # Test signal execution
    print("\nTesting signal execution...")
    system.execute_signals()

    # Show portfolio
    if system.paper_trading:
        portfolio = system.paper_trading.get_portfolio_summary()
        print(f"\nPortfolio Summary:")
        print(f"  Capital: ₹{portfolio['current_capital']:,.2f}")
        print(f"  Positions: {portfolio['total_positions']}")
        print(f"  Cash: ₹{portfolio['cash_available']:,.2f}")

    # Run mini backtest
    print("\nRunning 5-day backtest...")
    system.run_backtest(days=5)

    print("\n✅ Integrated System Test Complete!")


if __name__ == "__main__":
    test_integrated_system()