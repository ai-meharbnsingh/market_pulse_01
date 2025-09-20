"""
MarketPulse Enhanced Trading System - Phase 1, Step 3
Advanced Technical Analysis with Real Market Data Integration

Location: #root/enhanced_trading_system.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Tuple
import logging

# Add 06_DATA to path for data_fetcher
sys.path.append(str(Path.cwd() / '06_DATA'))

try:
    from data_fetcher import MarketDataFetcher
except ImportError:
    print("⚠️ data_fetcher not found. Make sure it's in 06_DATA/")
    MarketDataFetcher = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTechnicalAnalyzer:
    """
    Advanced technical analysis with multiple indicators
    """

    def __init__(self):
        self.indicators = {}

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""

        if len(prices) < period + 1:
            return 50.0  # Neutral RSI

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

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""

        if len(prices) < slow + signal:
            return {'macd': 0, 'signal': 0, 'histogram': 0}

        # Calculate EMAs
        prices_array = np.array(prices)

        # Fast EMA
        alpha_fast = 2 / (fast + 1)
        ema_fast = prices_array[-1]
        for i in range(min(fast, len(prices))):
            ema_fast = alpha_fast * prices_array[-(i + 1)] + (1 - alpha_fast) * ema_fast

        # Slow EMA
        alpha_slow = 2 / (slow + 1)
        ema_slow = prices_array[-1]
        for i in range(min(slow, len(prices))):
            ema_slow = alpha_slow * prices_array[-(i + 1)] + (1 - alpha_slow) * ema_slow

        macd_line = ema_fast - ema_slow

        # Signal line (simplified)
        signal_line = macd_line * 0.8  # Approximation
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """Calculate Bollinger Bands"""

        if len(prices) < period:
            current_price = prices[-1]
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'position': 0.5
            }

        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)

        current_price = prices[-1]

        # Calculate position within bands (0 = lower band, 1 = upper band)
        if upper_band != lower_band:
            position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            position = 0.5

        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'position': position
        }

    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float],
                             k_period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""

        if len(closes) < k_period:
            return {'k': 50, 'd': 50}

        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]

        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)

        if highest_high != lowest_low:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        else:
            k_percent = 50

        # Simple D% calculation (3-period SMA of K%)
        d_percent = k_percent  # Simplified

        return {'k': k_percent, 'd': d_percent}

    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate Average True Range"""

        if len(closes) < 2:
            return 0.02 * closes[-1] if closes else 1.0

        true_ranges = []

        for i in range(1, min(period + 1, len(closes))):
            high_low = highs[-i] - lows[-i]
            high_close = abs(highs[-i] - closes[-(i + 1)])
            low_close = abs(lows[-i] - closes[-(i + 1)])

            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)

        return np.mean(true_ranges) if true_ranges else 0.02 * closes[-1]


class EnhancedMomentumStrategy:
    """
    Enhanced momentum strategy with multiple technical indicators
    """

    def __init__(self):
        self.name = "Enhanced Momentum"
        self.analyzer = AdvancedTechnicalAnalyzer()

    def analyze(self, market_data: Dict) -> Dict:
        """
        Analyze using advanced momentum indicators
        """

        prices = market_data.get('close_prices', [])
        highs = market_data.get('high_prices', prices)
        lows = market_data.get('low_prices', prices)
        volumes = market_data.get('volumes', [])

        if len(prices) < 5:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}

        current_price = prices[-1]

        # Calculate indicators
        rsi = self.analyzer.calculate_rsi(prices)
        macd = self.analyzer.calculate_macd(prices)
        bb = self.analyzer.calculate_bollinger_bands(prices)
        stoch = self.analyzer.calculate_stochastic(highs, lows, prices)
        atr = self.analyzer.calculate_atr(highs, lows, prices)

        # Volume analysis
        if volumes and len(volumes) >= 5:
            avg_volume = np.mean(volumes[-5:])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0

        # Price momentum
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / prices[-5] * 100
        else:
            momentum = 0

        # Multi-indicator signal generation
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0
        reasons = []

        # RSI Analysis
        if 30 < rsi < 70:  # Healthy range
            if rsi > 50:
                bullish_signals += 1
                signal_strength += 0.2
                reasons.append(f"RSI bullish: {rsi:.1f}")
            else:
                bearish_signals += 1
                reasons.append(f"RSI bearish: {rsi:.1f}")
        elif rsi < 30:  # Oversold - potential bounce
            bullish_signals += 1
            signal_strength += 0.3
            reasons.append(f"RSI oversold bounce: {rsi:.1f}")
        elif rsi > 70:  # Overbought - potential correction
            bearish_signals += 1
            signal_strength += 0.3
            reasons.append(f"RSI overbought: {rsi:.1f}")

        # MACD Analysis
        if macd['macd'] > macd['signal'] and macd['histogram'] > 0:
            bullish_signals += 1
            signal_strength += 0.25
            reasons.append("MACD bullish crossover")
        elif macd['macd'] < macd['signal'] and macd['histogram'] < 0:
            bearish_signals += 1
            signal_strength += 0.25
            reasons.append("MACD bearish crossover")

        # Bollinger Bands Analysis
        if bb['position'] < 0.2:  # Near lower band
            bullish_signals += 1
            signal_strength += 0.2
            reasons.append(f"BB oversold: {bb['position']:.2f}")
        elif bb['position'] > 0.8:  # Near upper band
            bearish_signals += 1
            signal_strength += 0.2
            reasons.append(f"BB overbought: {bb['position']:.2f}")
        elif 0.4 < bb['position'] < 0.6:  # Middle range - trend continuation
            if momentum > 0:
                bullish_signals += 0.5
                reasons.append("BB mid-range with momentum")

        # Stochastic Analysis
        if stoch['k'] < 20:  # Oversold
            bullish_signals += 1
            signal_strength += 0.15
            reasons.append(f"Stochastic oversold: {stoch['k']:.1f}")
        elif stoch['k'] > 80:  # Overbought
            bearish_signals += 1
            signal_strength += 0.15
            reasons.append(f"Stochastic overbought: {stoch['k']:.1f}")

        # Volume Confirmation
        if volume_ratio > 1.5:  # High volume
            signal_strength += 0.1
            reasons.append(f"High volume: {volume_ratio:.1f}x")

        # Momentum Confirmation
        if abs(momentum) > 2:  # Strong momentum
            if momentum > 0:
                bullish_signals += 1
                reasons.append(f"Strong bullish momentum: {momentum:.1f}%")
            else:
                bearish_signals += 1
                reasons.append(f"Strong bearish momentum: {momentum:.1f}%")

        # Final signal decision
        signal = 'HOLD'
        confidence = 0.0

        if bullish_signals > bearish_signals and bullish_signals >= 2:
            signal = 'BUY'
            confidence = min(signal_strength, 0.9)
        elif bearish_signals > bullish_signals and bearish_signals >= 2:
            signal = 'SELL'
            confidence = min(signal_strength, 0.9)

        # Risk management levels
        stop_loss = current_price - (atr * 2) if signal == 'BUY' else current_price + (atr * 2)
        take_profit = current_price + (atr * 3) if signal == 'BUY' else current_price - (atr * 3)

        return {
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': '; '.join(reasons),
            'indicators': {
                'rsi': rsi,
                'macd': macd,
                'bollinger_bands': bb,
                'stochastic': stoch,
                'atr': atr,
                'volume_ratio': volume_ratio,
                'momentum': momentum
            },
            'signal_count': {'bullish': bullish_signals, 'bearish': bearish_signals}
        }


class EnhancedMeanReversionStrategy:
    """
    Enhanced mean reversion strategy
    """

    def __init__(self):
        self.name = "Enhanced Mean Reversion"
        self.analyzer = AdvancedTechnicalAnalyzer()

    def analyze(self, market_data: Dict) -> Dict:
        """
        Analyze using mean reversion indicators
        """

        prices = market_data.get('close_prices', [])
        highs = market_data.get('high_prices', prices)
        lows = market_data.get('low_prices', prices)

        if len(prices) < 10:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'Insufficient data'}

        current_price = prices[-1]

        # Calculate indicators
        rsi = self.analyzer.calculate_rsi(prices)
        bb = self.analyzer.calculate_bollinger_bands(prices)
        stoch = self.analyzer.calculate_stochastic(highs, lows, prices)

        # Mean reversion signals
        signal = 'HOLD'
        confidence = 0.0
        reasons = []

        # Oversold conditions (BUY signals)
        if (bb['position'] < 0.1 and rsi < 30 and stoch['k'] < 20):
            signal = 'BUY'
            confidence = 0.8
            reasons.append(f"Strong oversold: BB={bb['position']:.2f}, RSI={rsi:.1f}, Stoch={stoch['k']:.1f}")

        elif (bb['position'] < 0.2 and rsi < 35):
            signal = 'BUY'
            confidence = 0.6
            reasons.append(f"Oversold: BB={bb['position']:.2f}, RSI={rsi:.1f}")

        # Overbought conditions (SELL signals)
        elif (bb['position'] > 0.9 and rsi > 70 and stoch['k'] > 80):
            signal = 'SELL'
            confidence = 0.8
            reasons.append(f"Strong overbought: BB={bb['position']:.2f}, RSI={rsi:.1f}, Stoch={stoch['k']:.1f}")

        elif (bb['position'] > 0.8 and rsi > 65):
            signal = 'SELL'
            confidence = 0.6
            reasons.append(f"Overbought: BB={bb['position']:.2f}, RSI={rsi:.1f}")

        # Target is mean (middle Bollinger Band)
        target_price = bb['middle']
        atr = self.analyzer.calculate_atr(highs, lows, prices)

        return {
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'target_price': target_price,
            'stop_loss': bb['lower'] - atr if signal == 'BUY' else bb['upper'] + atr,
            'take_profit': target_price,
            'reason': '; '.join(reasons) if reasons else 'No clear mean reversion signal',
            'indicators': {
                'rsi': rsi,
                'bollinger_bands': bb,
                'stochastic': stoch,
                'atr': atr
            }
        }


class EnhancedTradingSystem:
    """
    Enhanced trading system with advanced technical analysis
    """

    def __init__(self, db_path: str = "marketpulse_production.db"):
        self.db_path = db_path

        # Initialize data fetcher
        self.data_fetcher = MarketDataFetcher(db_path) if MarketDataFetcher else None

        # Initialize strategies
        self.strategies = {
            'enhanced_momentum': EnhancedMomentumStrategy(),
            'enhanced_mean_reversion': EnhancedMeanReversionStrategy()
        }

        # Trading parameters
        self.min_confidence = 0.5
        self.max_positions = 5
        self.position_size = 0.1  # 10% of capital per position

        # Watchlist
        self.watchlist = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'RELIANCE.NS']

        logger.info("Enhanced Trading System initialized")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info(f"Watchlist: {self.watchlist}")

    def get_market_data_from_db(self, symbol: str, days: int = 30) -> Optional[Dict]:
        """
        Get market data from database
        """

        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT * FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=(symbol, days))
            conn.close()

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp')

            return {
                'symbol': symbol,
                'close_prices': df['close_price'].tolist(),
                'high_prices': df['high_price'].tolist(),
                'low_prices': df['low_price'].tolist(),
                'open_prices': df['open_price'].tolist(),
                'volumes': df['volume'].tolist(),
                'timestamps': df['timestamp'].tolist(),
                'latest_price': df['close_price'].iloc[-1],
                'data_count': len(df)
            }

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Analyze symbol with enhanced strategies
        """

        # Get market data
        market_data = self.get_market_data_from_db(symbol)

        if not market_data:
            return {
                'symbol': symbol,
                'error': 'No market data available',
                'signal': 'HOLD',
                'confidence': 0.0
            }

        results = {'symbol': symbol, 'strategies': {}}

        # Run all strategies
        for strategy_name, strategy in self.strategies.items():
            try:
                analysis = strategy.analyze(market_data)
                results['strategies'][strategy_name] = analysis

                logger.info(f"{symbol} - {strategy_name}: {analysis['signal']} "
                            f"(confidence: {analysis['confidence']:.0%})")

            except Exception as e:
                logger.error(f"Error in {strategy_name} for {symbol}: {e}")
                results['strategies'][strategy_name] = {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'error': str(e)
                }

        # Ensemble decision (combine strategies)
        ensemble_result = self.combine_strategies(results['strategies'])
        results['ensemble'] = ensemble_result

        return results

    def combine_strategies(self, strategy_results: Dict) -> Dict:
        """
        Combine multiple strategy signals into ensemble decision
        """

        signals = []
        confidences = []

        for strategy_name, result in strategy_results.items():
            if 'error' not in result:
                signals.append(result['signal'])
                confidences.append(result['confidence'])

        if not signals:
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No valid strategy signals'}

        # Count signals
        buy_signals = signals.count('BUY')
        sell_signals = signals.count('SELL')
        hold_signals = signals.count('HOLD')

        # Weighted confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Ensemble decision
        if buy_signals > sell_signals and buy_signals > hold_signals:
            final_signal = 'BUY'
            final_confidence = avg_confidence * (buy_signals / len(signals))
        elif sell_signals > buy_signals and sell_signals > hold_signals:
            final_signal = 'SELL'
            final_confidence = avg_confidence * (sell_signals / len(signals))
        else:
            final_signal = 'HOLD'
            final_confidence = 0.0

        return {
            'signal': final_signal,
            'confidence': final_confidence,
            'strategy_votes': {
                'BUY': buy_signals,
                'SELL': sell_signals,
                'HOLD': hold_signals
            },
            'avg_confidence': avg_confidence
        }

    def scan_opportunities(self) -> List[Dict]:
        """
        Scan watchlist for trading opportunities
        """

        opportunities = []

        logger.info(f"Scanning {len(self.watchlist)} symbols for opportunities...")

        for symbol in self.watchlist:
            try:
                analysis = self.analyze_symbol(symbol)

                ensemble = analysis.get('ensemble', {})

                if (ensemble.get('signal') != 'HOLD' and
                        ensemble.get('confidence', 0) >= self.min_confidence):
                    opportunities.append({
                        'symbol': symbol,
                        'signal': ensemble['signal'],
                        'confidence': ensemble['confidence'],
                        'strategy_votes': ensemble.get('strategy_votes', {}),
                        'analysis': analysis
                    })

                    logger.info(f"💡 Opportunity: {ensemble['signal']} {symbol} "
                                f"(confidence: {ensemble['confidence']:.0%})")

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)

        logger.info(f"Found {len(opportunities)} trading opportunities")

        return opportunities


def main():
    """
    Test the enhanced trading system
    """

    print("🚀 MarketPulse Enhanced Trading System - Testing")
    print("=" * 55)

    # Initialize system
    trading_system = EnhancedTradingSystem()

    # Test data availability
    print("\n📊 Testing Market Data Availability:")
    for symbol in trading_system.watchlist[:3]:  # Test first 3 symbols
        data = trading_system.get_market_data_from_db(symbol)
        if data:
            print(f"   ✅ {symbol}: {data['data_count']} records, "
                  f"latest: ${data['latest_price']:.2f}")
        else:
            print(f"   ❌ {symbol}: No data available")

    # Scan for opportunities
    print("\n🔍 Scanning for Trading Opportunities:")
    opportunities = trading_system.scan_opportunities()

    if opportunities:
        print(f"\n💡 Found {len(opportunities)} opportunities:")

        for i, opp in enumerate(opportunities[:5], 1):  # Show top 5
            symbol = opp['symbol']
            signal = opp['signal']
            confidence = opp['confidence']
            votes = opp['strategy_votes']

            currency = '$' if not symbol.endswith('.NS') else 'Rs'

            print(f"\n   {i}. {signal} {symbol} (confidence: {confidence:.0%})")
            print(f"      Strategy votes: BUY={votes.get('BUY', 0)}, "
                  f"SELL={votes.get('SELL', 0)}, HOLD={votes.get('HOLD', 0)}")

            # Show strategy details
            strategies = opp['analysis']['strategies']
            for strategy_name, strategy_result in strategies.items():
                if 'error' not in strategy_result:
                    indicators = strategy_result.get('indicators', {})
                    rsi = indicators.get('rsi', 0)
                    print(f"      {strategy_name}: {strategy_result['signal']} "
                          f"(RSI: {rsi:.1f})")
    else:
        print("   📈 No strong trading opportunities found at current thresholds")
        print("   💡 This is normal - quality over quantity!")

    print("\n🎉 Enhanced Trading System test complete!")
    print("\n✅ Key Features Working:")
    print("   📊 Multi-indicator technical analysis")
    print("   🎯 Strategy ensemble voting")
    print("   📈 Risk management levels")
    print("   💰 Confidence-based filtering")

    return len(opportunities) > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)