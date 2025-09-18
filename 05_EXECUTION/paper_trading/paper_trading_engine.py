"""
MarketPulse Paper Trading Engine
Simulates real trading without risking actual money
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]
    timestamp: datetime
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    commission: float = 0.0

    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class Position:
    """Position representation"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_price(self, current_price: float):
        """Update position with current price"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'entry_time': self.entry_time.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'value': self.quantity * self.current_price
        }


class PaperTradingEngine:
    """
    Paper Trading Engine for MarketPulse
    Simulates real trading with virtual money
    """

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize paper trading engine

        Args:
            initial_capital: Starting virtual capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash_available = initial_capital

        # Trading state
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[Dict] = []

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_commission = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital

        # Configuration
        self.commission_rate = 0.0003  # 0.03%
        self.slippage_rate = 0.0005  # 0.05%
        self.max_position_size = 0.05  # 5% of capital per position
        self.max_positions = 6  # Maximum concurrent positions

        # Risk limits
        self.daily_loss_limit = 0.02  # 2% daily loss limit
        self.daily_starting_capital = initial_capital
        self.last_reset_date = datetime.now().date()

        # Order ID counter
        self.order_counter = 0

        # Data storage
        self.data_dir = Path("10_DATA_STORAGE/paper_trading")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Paper Trading Engine initialized with ₹{initial_capital:,.2f}")

    def place_order(
            self,
            symbol: str,
            side: str,
            quantity: int,
            order_type: str = "MARKET",
            price: Optional[float] = None
    ) -> Tuple[bool, str, Optional[Order]]:
        """
        Place a new order

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET, LIMIT, etc.
            price: Price for limit orders

        Returns:
            Tuple of (success, message, order)
        """
        # Validate order
        validation = self._validate_order(symbol, side, quantity, price)
        if not validation[0]:
            return validation

        # Create order
        self.order_counter += 1
        order = Order(
            order_id=f"ORD_{self.order_counter:06d}",
            symbol=symbol,
            side=OrderSide[side],
            order_type=OrderType[order_type],
            quantity=quantity,
            price=price,
            timestamp=datetime.now()
        )

        # Add to pending orders
        self.orders.append(order)

        # Execute immediately if market order
        if order_type == "MARKET":
            success, message = self._execute_order(order)
            if not success:
                order.status = OrderStatus.REJECTED
                return False, message, order

        logger.info(f"Order placed: {order.order_id} - {side} {quantity} {symbol}")
        return True, f"Order {order.order_id} placed successfully", order

    def _validate_order(
            self,
            symbol: str,
            side: str,
            quantity: int,
            price: Optional[float] = None
    ) -> Tuple[bool, str, None]:
        """Validate order before placement"""

        # Check daily loss limit
        if self._check_daily_loss_limit():
            return False, "Daily loss limit reached", None

        # Check position limits
        if side == "BUY":
            if len(self.positions) >= self.max_positions:
                if symbol not in self.positions:
                    return False, f"Maximum {self.max_positions} positions allowed", None

            # Check position size
            if price:
                position_value = price * quantity
            else:
                position_value = self._get_market_price(symbol) * quantity

            if position_value > self.current_capital * self.max_position_size:
                return False, f"Position size exceeds {self.max_position_size * 100}% limit", None

            # Check available cash
            required_cash = position_value * (1 + self.commission_rate)
            if required_cash > self.cash_available:
                return False, f"Insufficient cash. Required: ₹{required_cash:,.2f}", None

        elif side == "SELL":
            # Check if position exists for selling
            if symbol not in self.positions:
                return False, f"No position in {symbol} to sell", None

            if self.positions[symbol].quantity < quantity:
                return False, f"Insufficient quantity. Have: {self.positions[symbol].quantity}", None

        return True, "Valid", None

    def _execute_order(self, order: Order) -> Tuple[bool, str]:
        """Execute a market order"""

        # Get market price with slippage
        market_price = self._get_market_price(order.symbol)

        if order.side == OrderSide.BUY:
            # Apply slippage (buy at slightly higher price)
            execution_price = market_price * (1 + self.slippage_rate)
        else:
            # Apply slippage (sell at slightly lower price)
            execution_price = market_price * (1 - self.slippage_rate)

        # Calculate commission
        trade_value = execution_price * order.quantity
        commission = trade_value * self.commission_rate

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.commission = commission

        # Update positions and cash
        if order.side == OrderSide.BUY:
            self._add_position(order.symbol, order.quantity, execution_price)
            self.cash_available -= (trade_value + commission)
        else:
            pnl = self._remove_position(order.symbol, order.quantity, execution_price)
            self.cash_available += (trade_value - commission)

        # Update metrics
        self.total_commission += commission
        self.total_trades += 1

        # Record trade
        self._record_trade(order)

        logger.info(f"Order executed: {order.order_id} at ₹{execution_price:.2f}")
        return True, "Order executed successfully"

    def _add_position(self, symbol: str, quantity: int, price: float):
        """Add or update position"""
        if symbol in self.positions:
            # Average into existing position
            position = self.positions[symbol]
            total_value = (position.quantity * position.avg_price) + (quantity * price)
            total_quantity = position.quantity + quantity
            position.avg_price = total_value / total_quantity
            position.quantity = total_quantity
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                entry_time=datetime.now()
            )

    def _remove_position(self, symbol: str, quantity: int, price: float) -> float:
        """Remove or reduce position"""
        position = self.positions[symbol]

        # Calculate realized PnL
        realized_pnl = (price - position.avg_price) * quantity
        position.realized_pnl += realized_pnl

        # Update position
        position.quantity -= quantity

        if position.quantity == 0:
            # Close position
            del self.positions[symbol]

            # Update win/loss statistics
            if realized_pnl > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

        return realized_pnl

    def _get_market_price(self, symbol: str) -> float:
        """Get simulated market price"""
        # TODO: Connect to real market data
        # For now, return dummy price
        base_prices = {
            "RELIANCE": 2500.0,
            "TCS": 3500.0,
            "INFY": 1500.0,
            "HDFC": 1600.0,
            "ICICI": 950.0
        }

        import random
        base = base_prices.get(symbol, 1000.0)
        # Add some randomness
        return base * (1 + random.uniform(-0.01, 0.01))

    def _check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is reached"""
        # Reset if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_starting_capital = self.current_capital
            self.last_reset_date = current_date

        # Calculate daily loss
        daily_loss = (self.daily_starting_capital - self.current_capital) / self.daily_starting_capital

        if daily_loss >= self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {daily_loss * 100:.2f}%")
            return True

        return False

    def _record_trade(self, order: Order):
        """Record trade in history"""
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.filled_quantity,
            'price': order.filled_price,
            'commission': order.commission,
            'capital': self.current_capital
        }

        self.trade_history.append(trade_record)

        # Save to file
        self._save_trade_history()

    def update_portfolio(self, market_data: Dict[str, float]):
        """Update portfolio with current market prices"""
        portfolio_value = self.cash_available

        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                position.update_price(current_price)
                portfolio_value += position.quantity * current_price

        self.current_capital = portfolio_value

        # Update drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_value = self.cash_available
        total_unrealized_pnl = 0

        for position in self.positions.values():
            total_value += position.quantity * position.current_price
            total_unrealized_pnl += position.unrealized_pnl

        return {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'cash_available': self.cash_available,
            'positions_value': total_value - self.cash_available,
            'total_positions': len(self.positions),
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': self.current_capital - self.initial_capital,
            'total_return_pct': ((self.current_capital / self.initial_capital) - 1) * 100,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
            'total_commission': self.total_commission
        }

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        return [pos.to_dict() for pos in self.positions.values()]

    def get_orders(self, status: Optional[str] = None) -> List[Dict]:
        """Get orders with optional status filter"""
        if status:
            filtered = [o for o in self.orders if o.status == OrderStatus[status]]
            return [o.to_dict() for o in filtered]
        return [o.to_dict() for o in self.orders]

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel a pending order"""
        for order in self.orders:
            if order.order_id == order_id and order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")
                return True, f"Order {order_id} cancelled"

        return False, f"Order {order_id} not found or already executed"

    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.__init__(self.initial_capital)
        logger.info("Portfolio reset to initial state")

    def _save_trade_history(self):
        """Save trade history to file"""
        file_path = self.data_dir / "trade_history.json"
        with open(file_path, 'w') as f:
            json.dump(self.trade_history, f, indent=2)

    def save_portfolio_snapshot(self):
        """Save current portfolio snapshot"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_portfolio_summary(),
            'positions': self.get_positions(),
            'pending_orders': self.get_orders('PENDING')
        }

        file_path = self.data_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(file_path, 'w') as f:
            json.dump(snapshot, f, indent=2)

        logger.info(f"Portfolio snapshot saved to {file_path}")


# Example usage and testing
def test_paper_trading():
    """Test paper trading functionality"""

    # Initialize engine
    engine = PaperTradingEngine(initial_capital=100000)

    # Place some test orders
    print("\n=== Testing Paper Trading Engine ===\n")

    # Buy order
    success, message, order = engine.place_order(
        symbol="RELIANCE",
        side="BUY",
        quantity=10,
        order_type="MARKET"
    )
    print(f"Buy Order: {message}")

    # Check portfolio
    summary = engine.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Capital: ₹{summary['current_capital']:,.2f}")
    print(f"  Cash: ₹{summary['cash_available']:,.2f}")
    print(f"  Positions: {summary['total_positions']}")

    # Check positions
    positions = engine.get_positions()
    if positions:
        print(f"\nCurrent Positions:")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} @ ₹{pos['avg_price']:.2f}")

    # Sell order
    if positions:
        success, message, order = engine.place_order(
            symbol="RELIANCE",
            side="SELL",
            quantity=5,
            order_type="MARKET"
        )
        print(f"\nSell Order: {message}")

    # Final summary
    summary = engine.get_portfolio_summary()
    print(f"\nFinal Summary:")
    print(f"  Total Return: {summary['total_return_pct']:.2f}%")
    print(f"  Win Rate: {summary['win_rate']:.2f}%")
    print(f"  Commission Paid: ₹{summary['total_commission']:.2f}")

    # Save snapshot
    engine.save_portfolio_snapshot()
    print("\n✅ Paper Trading Engine Test Complete!")


if __name__ == "__main__":
    test_paper_trading()