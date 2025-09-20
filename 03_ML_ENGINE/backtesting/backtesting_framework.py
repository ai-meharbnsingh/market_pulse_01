"""
MarketPulse Backtesting Framework - Phase 1, Step 3
Historical strategy validation and performance analysis

Location: #root/backtesting_framework.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Comprehensive backtesting engine for strategy validation
    """

    def __init__(self, db_path: str = "marketpulse_production.db", initial_capital: float = 100000):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.reset_portfolio()

    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}  # symbol: {'quantity': int, 'avg_price': float}
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []

    def get_historical_data(self, symbol: str, days: int = 60) -> Optional[pd.DataFrame]:
        """Get historical market data for backtesting"""

        try:
            conn = sqlite3.connect(self.db_path)

            query = """
                SELECT * FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp ASC 
                LIMIT ?
            """

            df = pd.read_sql_query(query, conn, params=(symbol, days))
            conn.close()

            if df.empty:
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df

        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    def execute_trade(self, symbol: str, action: str, price: float,
                      timestamp: datetime, quantity: int = None,
                      strategy: str = "backtest") -> bool:
        """Execute a trade in the backtest"""

        # Calculate position size if not specified (10% of portfolio)
        if quantity is None:
            portfolio_value = self.get_portfolio_value(price, symbol)
            position_value = portfolio_value * 0.1  # 10% position size
            quantity = int(position_value / price)

        if quantity <= 0:
            return False

        trade_value = quantity * price
        commission = trade_value * 0.001  # 0.1% commission

        if action == 'BUY':
            total_cost = trade_value + commission

            if total_cost > self.cash:
                logger.debug(f"Insufficient cash for {symbol} BUY: need {total_cost:.2f}, have {self.cash:.2f}")
                return False

            # Execute buy
            self.cash -= total_cost

            if symbol in self.positions:
                # Average down
                old_qty = self.positions[symbol]['quantity']
                old_price = self.positions[symbol]['avg_price']
                new_qty = old_qty + quantity
                new_avg_price = ((old_qty * old_price) + (quantity * price)) / new_qty

                self.positions[symbol] = {
                    'quantity': new_qty,
                    'avg_price': new_avg_price
                }
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price
                }

            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'strategy': strategy
            })

            logger.debug(f"BUY {quantity} {symbol} @ {price:.2f}")
            return True

        elif action == 'SELL':
            if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                logger.debug(f"Insufficient position for {symbol} SELL")
                return False

            # Execute sell
            trade_proceeds = trade_value - commission
            self.cash += trade_proceeds

            self.positions[symbol]['quantity'] -= quantity

            if self.positions[symbol]['quantity'] == 0:
                del self.positions[symbol]

            self.trades.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'strategy': strategy
            })

            logger.debug(f"SELL {quantity} {symbol} @ {price:.2f}")
            return True

        return False

    def get_portfolio_value(self, current_prices: Dict = None, symbol_for_price: str = None) -> float:
        """Calculate current portfolio value"""

        total_value = self.cash

        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                current_price = current_prices[symbol]
            elif symbol_for_price == symbol and current_prices:
                current_price = current_prices  # Single price passed
            else:
                current_price = position['avg_price']  # Use avg price as fallback

            total_value += position['quantity'] * current_price

        return total_value

    def backtest_strategy(self, strategy, symbols: List[str], days: int = 30) -> Dict:
        """
        Run backtest for a strategy across multiple symbols
        """

        logger.info(f"Starting backtest: {strategy.name} on {symbols} for {days} days")

        self.reset_portfolio()

        # Get historical data for all symbols
        historical_data = {}
        for symbol in symbols:
            data = self.get_historical_data(symbol, days)
            if data is not None and len(data) > 10:
                historical_data[symbol] = data
            else:
                logger.warning(f"Insufficient data for {symbol}, skipping")

        if not historical_data:
            return {'error': 'No historical data available'}

        # Get all unique timestamps and sort
        all_timestamps = set()
        for symbol, data in historical_data.items():
            all_timestamps.update(data['timestamp'].tolist())

        sorted_timestamps = sorted(all_timestamps)

        # Run day-by-day simulation
        for i, timestamp in enumerate(sorted_timestamps):
            daily_prices = {}

            # Analyze each symbol for this timestamp
            for symbol, data in historical_data.items():

                # Get data up to current timestamp
                current_data = data[data['timestamp'] <= timestamp]

                if len(current_data) < 5:  # Need minimum data
                    continue

                current_price = current_data['close_price'].iloc[-1]
                daily_prices[symbol] = current_price

                # Prepare market data for strategy
                market_data = {
                    'close_prices': current_data['close_price'].tolist(),
                    'high_prices': current_data['high_price'].tolist(),
                    'low_prices': current_data['low_price'].tolist(),
                    'open_prices': current_data['open_price'].tolist(),
                    'volumes': current_data['volume'].tolist(),
                }

                # Get strategy signal
                try:
                    analysis = strategy.analyze(market_data)
                    signal = analysis.get('signal', 'HOLD')
                    confidence = analysis.get('confidence', 0.0)

                    # Execute trades based on signal
                    if signal == 'BUY' and confidence >= 0.6:  # High confidence threshold
                        self.execute_trade(symbol, 'BUY', current_price, timestamp,
                                           strategy=strategy.name)

                    elif signal == 'SELL' and symbol in self.positions:
                        # Sell entire position
                        quantity = self.positions[symbol]['quantity']
                        self.execute_trade(symbol, 'SELL', current_price, timestamp,
                                           quantity=quantity, strategy=strategy.name)

                except Exception as e:
                    logger.error(f"Strategy error for {symbol} at {timestamp}: {e}")
                    continue

            # Record portfolio value for this day
            portfolio_value = self.get_portfolio_value(daily_prices)
            self.portfolio_values.append({
                'timestamp': timestamp,
                'value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })

            # Calculate daily return
            if len(self.portfolio_values) > 1:
                prev_value = self.portfolio_values[-2]['value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)

        # Calculate performance metrics
        performance = self.calculate_performance_metrics()

        return performance

    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""

        if not self.portfolio_values:
            return {'error': 'No portfolio data to analyze'}

        final_value = self.portfolio_values[-1]['value']
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Annualized metrics
        days_traded = len(self.portfolio_values)
        annualized_return = ((1 + total_return) ** (365 / days_traded)) - 1 if days_traded > 0 else 0

        # Risk metrics
        returns_array = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        peak_value = self.initial_capital
        max_drawdown = 0

        for portfolio_data in self.portfolio_values:
            value = portfolio_data['value']
            if value > peak_value:
                peak_value = value

            drawdown = (peak_value - value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Trade statistics
        total_trades = len(self.trades)
        buy_trades = len([t for t in self.trades if t['action'] == 'BUY'])
        sell_trades = len([t for t in self.trades if t['action'] == 'SELL'])

        # Win rate calculation (simplified)
        winning_trades = 0
        total_completed_trades = 0

        # Group trades by symbol to calculate P&L
        for symbol in set(t['symbol'] for t in self.trades):
            symbol_trades = [t for t in self.trades if t['symbol'] == symbol]

            position = 0
            avg_buy_price = 0

            for trade in symbol_trades:
                if trade['action'] == 'BUY':
                    if position == 0:
                        avg_buy_price = trade['price']
                    else:
                        avg_buy_price = ((position * avg_buy_price) +
                                         (trade['quantity'] * trade['price'])) / (position + trade['quantity'])
                    position += trade['quantity']

                elif trade['action'] == 'SELL' and position > 0:
                    pnl = (trade['price'] - avg_buy_price) * trade['quantity']

                    if pnl > 0:
                        winning_trades += 1
                    total_completed_trades += 1

                    position -= trade['quantity']

        win_rate = winning_trades / total_completed_trades if total_completed_trades > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'volatility': volatility,
            'volatility_pct': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'days_traded': days_traded,
            'final_cash': self.cash,
            'final_positions': len(self.positions),
            'avg_daily_return': np.mean(returns_array) if len(returns_array) > 0 else 0,
            'avg_daily_return_pct': np.mean(returns_array) * 100 if len(returns_array) > 0 else 0
        }


def compare_strategies(strategies: List, symbols: List[str], days: int = 30) -> Dict:
    """
    Compare multiple strategies against each other
    """

    logger.info(f"Comparing {len(strategies)} strategies on {len(symbols)} symbols")

    results = {}

    for strategy in strategies:
        try:
            backtester = BacktestingEngine()
            performance = backtester.backtest_strategy(strategy, symbols, days)

            if 'error' not in performance:
                results[strategy.name] = performance

                logger.info(f"{strategy.name}: {performance['total_return_pct']:.1f}% return, "
                            f"{performance['win_rate_pct']:.1f}% win rate")
            else:
                logger.error(f"Backtest failed for {strategy.name}: {performance['error']}")

        except Exception as e:
            logger.error(f"Error backtesting {strategy.name}: {e}")
            continue

    # Rank strategies by Sharpe ratio
    if results:
        ranked_strategies = sorted(results.items(),
                                   key=lambda x: x[1]['sharpe_ratio'],
                                   reverse=True)

        comparison = {
            'strategy_count': len(results),
            'symbols_tested': symbols,
            'days_tested': days,
            'results': results,
            'ranking': [{'strategy': name, 'sharpe_ratio': perf['sharpe_ratio']}
                        for name, perf in ranked_strategies]
        }

        return comparison

    return {'error': 'No valid backtest results'}


def main():
    """
    Test the backtesting framework
    """

    print("📊 MarketPulse Backtesting Framework - Testing")
    print("=" * 50)

    # Import strategies for testing
    try:
        from enhanced_trading_system import EnhancedMomentumStrategy, EnhancedMeanReversionStrategy

        strategies = [
            EnhancedMomentumStrategy(),
            EnhancedMeanReversionStrategy()
        ]

        symbols = ['SPY', 'AAPL', 'MSFT']

        print(f"\n🧪 Testing strategies: {[s.name for s in strategies]}")
        print(f"📈 Testing symbols: {symbols}")
        print(f"📅 Testing period: Last 30 days of available data")

        # Run comparison
        comparison = compare_strategies(strategies, symbols, days=30)

        if 'error' not in comparison:
            print(f"\n📊 BACKTEST RESULTS:")
            print("=" * 30)

            for strategy_name, performance in comparison['results'].items():
                print(f"\n🎯 {strategy_name}:")
                print(f"   💰 Total Return: {performance['total_return_pct']:.1f}%")
                print(f"   📈 Annualized Return: {performance['annualized_return_pct']:.1f}%")
                print(f"   ⚡ Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
                print(f"   📉 Max Drawdown: {performance['max_drawdown_pct']:.1f}%")
                print(f"   🎲 Win Rate: {performance['win_rate_pct']:.1f}%")
                print(f"   📊 Total Trades: {performance['total_trades']}")
                print(f"   💵 Final Value: ${performance['final_value']:,.2f}")

            print(f"\n🏆 STRATEGY RANKING (by Sharpe Ratio):")
            for i, rank in enumerate(comparison['ranking'], 1):
                print(f"   {i}. {rank['strategy']} (Sharpe: {rank['sharpe_ratio']:.2f})")

            print(f"\n✅ Backtesting framework working!")
            print(f"💡 Use this to validate strategy improvements")

            return True

        else:
            print(f"\n❌ Backtest comparison failed: {comparison['error']}")
            return False

    except ImportError as e:
        print(f"\n⚠️ Could not import enhanced strategies: {e}")
        print("Run enhanced_trading_system.py first to create the strategies")
        return False

    except Exception as e:
        print(f"\n❌ Backtesting error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)