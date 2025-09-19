"""
Enhanced Integrated Trading System - Phase 2, Step 1
Combines technical analysis with ML predictions

Location: #integrated_trading_system_enhanced.py

Features:
- Integration of Alpha Model and LSTM predictions
- Enhanced signal confidence scoring
- ML-powered strategy selection
- Advanced risk management with ML insights
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
sys.path.append(str(current_dir / "03_ML_ENGINE" / "models"))
sys.path.append(str(current_dir / "05_EXECUTION" / "paper_trading"))
sys.path.append(str(current_dir / "05_EXECUTION" / "alerts"))
sys.path.append(str(current_dir / "02_ANALYSIS" / "technical"))
sys.path.append(str(current_dir / "04_RISK"))

# Import components
try:
    from ml_signal_enhancer import MLSignalEnhancer

    print("✅ ML Signal Enhancer imported")
except ImportError as e:
    print(f"⚠️ ML Signal Enhancer not found: {e}")
    MLSignalEnhancer = None

try:
    from paper_trading_engine import PaperTradingEngine

    print("✅ Paper Trading Engine imported")
except ImportError as e:
    print(f"⚠️ Paper Trading Engine not found: {e}")
    PaperTradingEngine = None

try:
    from telegram_alerts import TelegramNotifier

    print("✅ Telegram Alerts imported")
except ImportError as e:
    print(f"⚠️ Telegram Alerts not found: {e}")
    TelegramNotifier = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """
    Enhanced Technical Analysis with ML Integration
    Provides base technical signals that ML models can enhance
    """

    def __init__(self):
        """Initialize Technical Analyzer"""
        self.indicators = {}
        logger.info("📊 Technical Analyzer initialized")

    def analyze(self, symbol: str, prices: List[float], volumes: List[float] = None) -> Dict:
        """
        Perform technical analysis on price data

        Args:
            symbol: Stock symbol
            prices: List of prices (most recent last)
            volumes: List of volumes (optional)

        Returns:
            Technical analysis results
        """

        if len(prices) < 20:
            logger.warning(f"Insufficient price data for {symbol}: {len(prices)} points")
            return self._create_neutral_analysis(symbol)

        try:
            analysis = {
                'symbol': symbol,
                'current_price': prices[-1],
                'timestamp': datetime.now(),
                'indicators': {},
                'signal': 'HOLD',
                'confidence': 0.5,
                'strength': 'MEDIUM'
            }

            # Calculate technical indicators
            analysis['indicators'] = self._calculate_indicators(prices, volumes)

            # Generate signal from indicators
            signal_result = self._generate_technical_signal(analysis['indicators'])
            analysis.update(signal_result)

            return analysis

        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            return self._create_neutral_analysis(symbol)

    def _calculate_indicators(self, prices: List[float], volumes: List[float] = None) -> Dict:
        """Calculate technical indicators"""

        indicators = {}
        prices_array = np.array(prices)

        try:
            # RSI (14-period)
            indicators['rsi'] = self._calculate_rsi(prices_array, 14)

            # Moving Averages
            indicators['sma_20'] = np.mean(prices_array[-20:])
            indicators['sma_50'] = np.mean(prices_array[-50:]) if len(prices) >= 50 else indicators['sma_20']

            # MACD
            macd_result = self._calculate_macd(prices_array)
            indicators.update(macd_result)

            # Bollinger Bands
            bb_result = self._calculate_bollinger_bands(prices_array, 20, 2)
            indicators.update(bb_result)

            # Stochastic Oscillator
            if len(prices) >= 14:
                stoch_result = self._calculate_stochastic(prices_array, 14)
                indicators.update(stoch_result)

            # Volume indicators (if available)
            if volumes and len(volumes) >= 20:
                indicators['volume_sma'] = np.mean(volumes[-20:])
                indicators['volume_ratio'] = volumes[-1] / indicators['volume_sma']

        except Exception as e:
            logger.warning(f"Error calculating indicators: {e}")

        return indicators

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""

        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> Dict:
        """Calculate MACD"""

        if len(prices) < 26:
            return {'macd': 0, 'macd_signal': 0, 'macd_histogram': 0}

        # Exponential Moving Averages
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)

        macd_line = ema_12 - ema_26

        # Signal line (9-period EMA of MACD)
        if len(prices) >= 35:  # Need enough data for signal line
            macd_values = []
            for i in range(26, len(prices)):
                ema_12_i = self._calculate_ema(prices[:i + 1], 12)
                ema_26_i = self._calculate_ema(prices[:i + 1], 26)
                macd_values.append(ema_12_i - ema_26_i)

            if len(macd_values) >= 9:
                signal_line = self._calculate_ema(np.array(macd_values), 9)
            else:
                signal_line = macd_line
        else:
            signal_line = macd_line

        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        }

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""

        if len(prices) < period:
            return np.mean(prices)

        alpha = 2 / (period + 1)
        ema = prices[0]

        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Dict:
        """Calculate Bollinger Bands"""

        if len(prices) < period:
            sma = np.mean(prices)
            return {
                'bb_upper': sma * 1.02,
                'bb_lower': sma * 0.98,
                'bb_middle': sma,
                'bb_position': 0.5
            }

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        # Current position within bands (0 = lower band, 1 = upper band)
        current_price = prices[-1]
        bb_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5

        return {
            'bb_upper': upper_band,
            'bb_lower': lower_band,
            'bb_middle': sma,
            'bb_position': bb_position
        }

    def _calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""

        if len(prices) < period:
            return {'stoch_k': 50, 'stoch_d': 50}

        # For simplicity, using close prices as high/low approximation
        highs = prices  # In real implementation, use actual highs
        lows = prices  # In real implementation, use actual lows

        recent_high = np.max(highs[-period:])
        recent_low = np.min(lows[-period:])

        if recent_high == recent_low:
            stoch_k = 50
        else:
            stoch_k = ((prices[-1] - recent_low) / (recent_high - recent_low)) * 100

        # %D is 3-period SMA of %K (simplified)
        stoch_d = stoch_k  # In real implementation, calculate 3-period average

        return {
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        }

    def _generate_technical_signal(self, indicators: Dict) -> Dict:
        """Generate trading signal from technical indicators"""

        signals = []
        confidences = []

        # RSI Signal
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:
                signals.append('BUY')
                confidences.append(0.7)
            elif rsi > 70:
                signals.append('SELL')
                confidences.append(0.7)
            else:
                signals.append('HOLD')
                confidences.append(0.4)

        # MACD Signal
        if 'macd' in indicators and 'macd_signal' in indicators:
            if indicators['macd'] > indicators['macd_signal']:
                signals.append('BUY')
                confidences.append(0.6)
            else:
                signals.append('SELL')
                confidences.append(0.6)

        # Bollinger Bands Signal
        if 'bb_position' in indicators:
            bb_pos = indicators['bb_position']
            if bb_pos < 0.2:  # Near lower band
                signals.append('BUY')
                confidences.append(0.65)
            elif bb_pos > 0.8:  # Near upper band
                signals.append('SELL')
                confidences.append(0.65)
            else:
                signals.append('HOLD')
                confidences.append(0.3)

        # Aggregate signals
        if not signals:
            return {'signal': 'HOLD', 'confidence': 0.5, 'strength': 'LOW'}

        # Count signal types
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        hold_count = signals.count('HOLD')

        # Determine final signal
        if buy_count > sell_count and buy_count > hold_count:
            final_signal = 'BUY'
            signal_strength = buy_count / len(signals)
        elif sell_count > buy_count and sell_count > hold_count:
            final_signal = 'SELL'
            signal_strength = sell_count / len(signals)
        else:
            final_signal = 'HOLD'
            signal_strength = 0.5

        # Calculate average confidence
        avg_confidence = np.mean(confidences) * signal_strength

        # Determine strength
        if avg_confidence > 0.7:
            strength = 'HIGH'
        elif avg_confidence > 0.5:
            strength = 'MEDIUM'
        else:
            strength = 'LOW'

        return {
            'signal': final_signal,
            'confidence': avg_confidence,
            'strength': strength
        }

    def _create_neutral_analysis(self, symbol: str) -> Dict:
        """Create neutral analysis when calculation fails"""

        return {
            'symbol': symbol,
            'current_price': 0.0,
            'timestamp': datetime.now(),
            'indicators': {},
            'signal': 'HOLD',
            'confidence': 0.5,
            'strength': 'LOW'
        }


class EnhancedTradingSystem:
    """
    Enhanced Trading System with ML Integration
    Combines technical analysis with ML predictions
    """

    def __init__(self, initial_capital: float = 100000):
        """Initialize enhanced trading system"""

        self.initial_capital = initial_capital

        # Initialize components
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_enhancer = MLSignalEnhancer() if MLSignalEnhancer else None
        self.paper_trading = PaperTradingEngine(initial_capital) if PaperTradingEngine else None
        self.telegram = TelegramNotifier() if TelegramNotifier else None

        # Trading parameters
        self.watchlist = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI', 'AAPL', 'MSFT']
        self.max_positions = 5
        self.max_position_size = 0.15  # 15% of capital per position
        self.min_confidence = 0.65  # Minimum confidence for ML-enhanced trades

        # Performance tracking
        self.signals_generated = []
        self.trades_executed = []

        logger.info(f"🚀 Enhanced Trading System initialized with ₹{initial_capital:,.0f}")
        logger.info(f"ML Enhancement: {'✅ Active' if self.ml_enhancer else '❌ Disabled'}")

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """
        Get market data for symbol
        In production, this would connect to real data feed
        """

        # Generate realistic sample data for testing
        np.random.seed(hash(symbol) % 1000)

        # Base prices for different symbols
        base_prices = {
            'RELIANCE': 2500, 'TCS': 3500, 'INFY': 1500,
            'HDFC': 1600, 'ICICI': 950, 'AAPL': 150, 'MSFT': 300
        }

        base_price = base_prices.get(symbol, 1000)

        # Generate 100 data points with realistic patterns
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = []
        volumes = []

        current_price = base_price
        trend = np.random.choice([-0.001, 0, 0.001])  # Random trend

        for i in range(100):
            # Price movement with trend and noise
            daily_change = np.random.normal(trend, 0.02)
            mean_reversion = (base_price - current_price) * 0.01
            current_price = current_price * (1 + daily_change) + mean_reversion
            prices.append(max(current_price, 1))  # Ensure positive prices

            # Volume (correlated with price movement)
            base_volume = 2000000
            volume_change = abs(daily_change) * 0.5 + np.random.normal(0, 0.3)
            volume = max(base_volume * (1 + volume_change), 100000)
            volumes.append(int(volume))

        # Create OHLC data
        market_data = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': volumes
        })

        return market_data

    def analyze_symbol_enhanced(self, symbol: str) -> Dict:
        """
        Enhanced symbol analysis with ML integration
        """

        logger.info(f"🔍 Analyzing {symbol} with ML enhancement")

        try:
            # Get market data
            market_data = self.get_market_data(symbol)

            # Perform technical analysis
            prices = market_data['close'].tolist()
            volumes = market_data['volume'].tolist()
            technical_analysis = self.technical_analyzer.analyze(symbol, prices, volumes)

            # Enhance with ML if available
            if self.ml_enhancer:
                enhanced_signal = self.ml_enhancer.enhance_signal(
                    symbol, technical_analysis, market_data
                )

                # Combine results
                analysis = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'current_price': technical_analysis['current_price'],
                    'technical_analysis': technical_analysis,
                    'ml_enhanced': enhanced_signal,
                    'final_signal': enhanced_signal['ensemble_signal'],
                    'final_confidence': enhanced_signal['confidence'],
                    'risk_adjusted_confidence': enhanced_signal['risk_adjusted_confidence'],
                    'models_used': enhanced_signal['models_used'],
                    'enhancement_active': True
                }

            else:
                # Fall back to technical analysis only
                analysis = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'current_price': technical_analysis['current_price'],
                    'technical_analysis': technical_analysis,
                    'ml_enhanced': None,
                    'final_signal': technical_analysis['signal'],
                    'final_confidence': technical_analysis['confidence'],
                    'risk_adjusted_confidence': technical_analysis['confidence'] * 0.8,
                    'models_used': ['technical_only'],
                    'enhancement_active': False
                }

            # Store signal
            self.signals_generated.append(analysis)

            return analysis

        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed for {symbol}: {e}")
            return self._create_neutral_analysis(symbol)

    def should_trade(self, analysis: Dict) -> bool:
        """
        Determine if we should execute a trade based on analysis
        """

        signal = analysis['final_signal']
        confidence = analysis['risk_adjusted_confidence']

        # Basic rules
        if signal == 'HOLD':
            return False

        if confidence < self.min_confidence:
            logger.info(f"❌ {analysis['symbol']}: Confidence too low ({confidence:.2%} < {self.min_confidence:.2%})")
            return False

        # Check position limits
        if self.paper_trading:
            portfolio = self.paper_trading.get_portfolio_summary()
            if portfolio['total_positions'] >= self.max_positions:
                logger.info(f"❌ {analysis['symbol']}: Max positions reached ({self.max_positions})")
                return False

        # ML enhancement bonus
        if analysis['enhancement_active'] and len(analysis['models_used']) >= 2:
            logger.info(f"✅ {analysis['symbol']}: ML enhancement active with {len(analysis['models_used'])} models")
            return True

        # High confidence technical signal
        if confidence >= 0.75:
            logger.info(f"✅ {analysis['symbol']}: High confidence signal ({confidence:.2%})")
            return True

        return True  # Trade if we reach here

    def execute_trade(self, analysis: Dict) -> bool:
        """
        Execute trade based on analysis
        """

        if not self.should_trade(analysis):
            return False

        if not self.paper_trading:
            logger.warning("Paper trading engine not available")
            return False

        symbol = analysis['symbol']
        signal = analysis['final_signal']
        confidence = analysis['risk_adjusted_confidence']
        current_price = analysis['current_price']

        # Calculate position size based on confidence
        base_position_size = self.max_position_size
        confidence_multiplier = min(confidence / self.min_confidence, 1.5)  # Max 1.5x
        position_size = base_position_size * confidence_multiplier

        try:
            if signal == 'BUY':
                success = self.paper_trading.buy_stock(
                    symbol=symbol,
                    price=current_price,
                    percentage=position_size * 100,
                    strategy_name="ML_Enhanced"
                )

                if success:
                    logger.info(f"🟢 BUY {symbol} at ₹{current_price:.2f} "
                                f"({position_size:.1%} position, {confidence:.1%} confidence)")

                    # Send alert
                    if self.telegram:
                        message = (f"🟢 BUY SIGNAL\n"
                                   f"Symbol: {symbol}\n"
                                   f"Price: ₹{current_price:.2f}\n"
                                   f"Confidence: {confidence:.1%}\n"
                                   f"Models: {', '.join(analysis['models_used'])}")
                        self.telegram.send_alert(message)

                    self.trades_executed.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'confidence': confidence,
                        'models_used': analysis['models_used']
                    })

                    return True

            elif signal == 'SELL':
                success = self.paper_trading.sell_stock(
                    symbol=symbol,
                    price=current_price,
                    percentage=100,  # Sell all holdings
                    strategy_name="ML_Enhanced"
                )

                if success:
                    logger.info(f"🔴 SELL {symbol} at ₹{current_price:.2f} "
                                f"({confidence:.1%} confidence)")

                    # Send alert
                    if self.telegram:
                        message = (f"🔴 SELL SIGNAL\n"
                                   f"Symbol: {symbol}\n"
                                   f"Price: ₹{current_price:.2f}\n"
                                   f"Confidence: {confidence:.1%}\n"
                                   f"Models: {', '.join(analysis['models_used'])}")
                        self.telegram.send_alert(message)

                    self.trades_executed.append({
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'confidence': confidence,
                        'models_used': analysis['models_used']
                    })

                    return True

        except Exception as e:
            logger.error(f"❌ Trade execution failed for {symbol}: {e}")
            return False

        return False

    def scan_and_trade(self) -> Dict:
        """
        Scan watchlist and execute trades
        """

        logger.info(f"🔍 Scanning {len(self.watchlist)} symbols...")

        scan_results = {
            'timestamp': datetime.now(),
            'symbols_scanned': len(self.watchlist),
            'signals_generated': 0,
            'trades_executed': 0,
            'ml_enhanced_signals': 0,
            'analysis_results': []
        }

        for symbol in self.watchlist:
            try:
                # Analyze symbol
                analysis = self.analyze_symbol_enhanced(symbol)
                scan_results['analysis_results'].append(analysis)

                # Count signals
                if analysis['final_signal'] != 'HOLD':
                    scan_results['signals_generated'] += 1

                    if analysis['enhancement_active']:
                        scan_results['ml_enhanced_signals'] += 1

                # Execute trade if criteria met
                if self.execute_trade(analysis):
                    scan_results['trades_executed'] += 1

            except Exception as e:
                logger.error(f"❌ Error scanning {symbol}: {e}")

        logger.info(f"✅ Scan complete: {scan_results['signals_generated']} signals, "
                    f"{scan_results['trades_executed']} trades, "
                    f"{scan_results['ml_enhanced_signals']} ML-enhanced")

        return scan_results

    def get_performance_summary(self) -> Dict:
        """Get system performance summary"""

        summary = {
            'timestamp': datetime.now(),
            'signals_generated': len(self.signals_generated),
            'trades_executed': len(self.trades_executed),
            'ml_enhancement_rate': 0.0,
            'average_confidence': 0.0,
            'portfolio_performance': {}
        }

        if self.signals_generated:
            ml_enhanced = sum(1 for s in self.signals_generated if s['enhancement_active'])
            summary['ml_enhancement_rate'] = ml_enhanced / len(self.signals_generated)

            confidences = [s['final_confidence'] for s in self.signals_generated]
            summary['average_confidence'] = np.mean(confidences)

        if self.paper_trading:
            summary['portfolio_performance'] = self.paper_trading.get_portfolio_summary()

        if self.ml_enhancer:
            ml_performance = self.ml_enhancer.get_performance_summary()
            summary['ml_model_performance'] = ml_performance

        return summary

    def _create_neutral_analysis(self, symbol: str) -> Dict:
        """Create neutral analysis when processing fails"""

        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': 0.0,
            'technical_analysis': {},
            'ml_enhanced': None,
            'final_signal': 'HOLD',
            'final_confidence': 0.5,
            'risk_adjusted_confidence': 0.4,
            'models_used': ['error'],
            'enhancement_active': False
        }


def test_enhanced_trading_system():
    """Test the Enhanced Trading System"""

    print("\n=== Testing Enhanced Trading System ===\n")

    # Initialize system
    system = EnhancedTradingSystem(initial_capital=500000)

    # Test single symbol analysis
    print("Testing enhanced analysis...")
    test_symbol = 'RELIANCE'
    analysis = system.analyze_symbol_enhanced(test_symbol)

    print(f"\n{test_symbol} Enhanced Analysis:")
    print(f"  Final Signal: {analysis['final_signal']}")
    print(f"  Confidence: {analysis['final_confidence']:.2%}")
    print(f"  Risk Adjusted: {analysis['risk_adjusted_confidence']:.2%}")
    print(f"  Current Price: ₹{analysis['current_price']:.2f}")
    print(f"  ML Enhanced: {analysis['enhancement_active']}")
    print(f"  Models Used: {analysis['models_used']}")

    # Test scanning and trading
    print(f"\nTesting market scan...")
    scan_results = system.scan_and_trade()

    print(f"\nScan Results:")
    print(f"  Symbols Scanned: {scan_results['symbols_scanned']}")
    print(f"  Signals Generated: {scan_results['signals_generated']}")
    print(f"  ML Enhanced Signals: {scan_results['ml_enhanced_signals']}")
    print(f"  Trades Executed: {scan_results['trades_executed']}")

    # Show top signals
    signals = [r for r in scan_results['analysis_results'] if r['final_signal'] != 'HOLD']
    if signals:
        print(f"\nTop Signals:")
        for signal in sorted(signals, key=lambda x: x['final_confidence'], reverse=True)[:3]:
            print(f"  {signal['symbol']}: {signal['final_signal']} "
                  f"({signal['final_confidence']:.1%} confidence)")

    # Performance summary
    performance = system.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  ML Enhancement Rate: {performance['ml_enhancement_rate']:.1%}")
    print(f"  Average Confidence: {performance['average_confidence']:.1%}")

    if 'portfolio_performance' in performance and performance['portfolio_performance']:
        portfolio = performance['portfolio_performance']
        print(f"  Portfolio Value: ₹{portfolio.get('current_capital', 0):,.0f}")
        print(f"  Active Positions: {portfolio.get('total_positions', 0)}")

    print("\n✅ Enhanced Trading System Test Complete!")


if __name__ == "__main__":
    test_enhanced_trading_system()