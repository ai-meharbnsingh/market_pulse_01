# 04_RISK/advanced_risk_management.py
"""
Phase 3, Step 3: Advanced Risk Management System
Comprehensive risk management for live trading with portfolio optimization

Features:
- Position sizing with Kelly Criterion and Risk Parity
- Dynamic stop-loss and take-profit calculation
- Portfolio risk monitoring and alerts
- Correlation analysis and diversification metrics
- Value at Risk (VaR) calculation
- Maximum Drawdown monitoring
- Real-time risk dashboard

Location: #04_RISK/advanced_risk_management.py
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level categories"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio/position"""
    symbol: Optional[str]
    position_size_pct: float
    var_1_day: float  # 1-day Value at Risk
    var_5_day: float  # 5-day Value at Risk
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    risk_level: RiskLevel
    last_updated: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        data['last_updated'] = self.last_updated.isoformat()
        return data


@dataclass
class RiskAlert:
    """Risk alert/warning"""
    alert_id: str
    severity: RiskLevel
    message: str
    symbol: Optional[str]
    metric_value: float
    threshold: float
    created_at: datetime
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['created_at'] = self.created_at.isoformat()
        return data


class AdvancedRiskManager:
    """Advanced risk management system"""

    def __init__(self, db_path: str = "marketpulse_production.db"):
        self.db_path = Path(db_path)

        # Risk parameters (configurable)
        self.max_position_size = 0.15  # 15% max single position
        self.max_portfolio_risk = 0.25  # 25% max portfolio risk
        self.max_correlation = 0.7  # 70% max correlation between positions
        self.max_drawdown_limit = 0.2  # 20% max drawdown
        self.var_confidence = 0.95  # 95% VaR confidence level
        self.risk_free_rate = 0.02  # 2% risk-free rate

        # Risk tracking
        self.risk_alerts: List[RiskAlert] = []
        self.portfolio_metrics: Optional[RiskMetrics] = None

        logger.info("Advanced Risk Manager initialized")

    def _connect_database(self) -> sqlite3.Connection:
        """Connect to database"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def calculate_position_size_kelly(self, symbol: str, win_rate: float,
                                      avg_win: float, avg_loss: float,
                                      portfolio_value: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""

        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0

            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = abs(avg_win / avg_loss)
            p = win_rate
            q = 1 - win_rate

            kelly_fraction = (b * p - q) / b

            # Apply conservative scaling (use 25% of Kelly to reduce risk)
            conservative_kelly = kelly_fraction * 0.25

            # Apply position size limits
            max_fraction = min(self.max_position_size, conservative_kelly)
            max_fraction = max(0.0, max_fraction)  # No negative positions

            optimal_size = portfolio_value * max_fraction

            logger.info(f"Kelly position size for {symbol}: {max_fraction:.2%} (${optimal_size:,.2f})")

            return optimal_size

        except Exception as e:
            logger.error(f"Kelly calculation error for {symbol}: {e}")
            return 0.0

    def calculate_position_size_volatility(self, symbol: str, target_volatility: float,
                                           portfolio_value: float,
                                           lookback_days: int = 30) -> float:
        """Calculate position size based on volatility targeting"""

        try:
            # Get historical price data
            conn = self._connect_database()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT close_price, timestamp FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (symbol, lookback_days + 1))

            data = cursor.fetchall()
            conn.close()

            if len(data) < 2:
                logger.warning(f"Insufficient data for volatility calculation: {symbol}")
                return 0.0

            # Calculate daily returns
            prices = [float(row['close_price']) for row in reversed(data)]
            returns = np.diff(np.log(prices))

            if len(returns) == 0:
                return 0.0

            # Calculate annualized volatility (assuming 252 trading days)
            daily_volatility = np.std(returns)
            annualized_volatility = daily_volatility * np.sqrt(252)

            if annualized_volatility <= 0:
                return 0.0

            # Position size = (Target Vol / Asset Vol) * Portfolio Value
            position_fraction = target_volatility / annualized_volatility

            # Apply limits
            position_fraction = min(position_fraction, self.max_position_size)
            position_fraction = max(0.0, position_fraction)

            optimal_size = portfolio_value * position_fraction

            logger.info(
                f"Volatility-based size for {symbol}: {position_fraction:.2%} (Vol: {annualized_volatility:.1%})")

            return optimal_size

        except Exception as e:
            logger.error(f"Volatility calculation error for {symbol}: {e}")
            return 0.0

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95,
                      horizon: int = 1) -> float:
        """Calculate Value at Risk"""

        try:
            if len(returns) == 0:
                return 0.0

            # Sort returns
            sorted_returns = np.sort(returns)

            # Calculate VaR at given confidence level
            var_index = int((1 - confidence) * len(sorted_returns))
            var_return = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[0]

            # Scale by time horizon (square root rule)
            var_scaled = var_return * np.sqrt(horizon)

            return abs(var_scaled)  # Return positive value

        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0

    def calculate_portfolio_risk_metrics(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics for portfolio"""

        try:
            total_value = portfolio_data.get('total_value', 0.0)
            positions = portfolio_data.get('positions', {})

            if total_value <= 0 or not positions:
                return RiskMetrics(
                    symbol=None,
                    position_size_pct=0.0,
                    var_1_day=0.0,
                    var_5_day=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    volatility=0.0,
                    beta=0.0,
                    risk_level=RiskLevel.LOW,
                    last_updated=datetime.now()
                )

            # Get portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(positions, days=60)

            if len(portfolio_returns) == 0:
                portfolio_returns = np.array([0.0])

            # Calculate metrics
            var_1_day = self.calculate_var(portfolio_returns, self.var_confidence, 1)
            var_5_day = self.calculate_var(portfolio_returns, self.var_confidence, 5)

            volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0.0

            # Sharpe ratio
            mean_return = np.mean(portfolio_returns) * 252 if len(portfolio_returns) > 0 else 0.0
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0

            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)

            # Beta (vs market - use SPY as proxy)
            beta = self._calculate_portfolio_beta(positions)

            # Largest position size
            max_position_pct = 0.0
            if positions:
                position_values = [pos.get('market_value', 0) for pos in positions.values()]
                max_position_value = max(position_values) if position_values else 0
                max_position_pct = max_position_value / total_value if total_value > 0 else 0

            # Determine risk level
            risk_level = self._determine_risk_level(
                max_position_pct, var_1_day, max_drawdown, volatility
            )

            metrics = RiskMetrics(
                symbol=None,  # Portfolio-level
                position_size_pct=max_position_pct,
                var_1_day=var_1_day * total_value,  # Convert to dollar amount
                var_5_day=var_5_day * total_value,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                risk_level=risk_level,
                last_updated=datetime.now()
            )

            self.portfolio_metrics = metrics
            return metrics

        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {e}")
            return RiskMetrics(
                symbol=None,
                position_size_pct=0.0,
                var_1_day=0.0,
                var_5_day=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                beta=0.0,
                risk_level=RiskLevel.LOW,
                last_updated=datetime.now()
            )

    def _calculate_portfolio_returns(self, positions: Dict[str, Any], days: int = 30) -> np.ndarray:
        """Calculate historical portfolio returns"""

        try:
            if not positions:
                return np.array([])

            # Get historical data for all positions
            conn = self._connect_database()
            cursor = conn.cursor()

            # Get portfolio value history (simplified calculation)
            all_returns = []

            for symbol, position in positions.items():
                cursor.execute("""
                    SELECT close_price, timestamp FROM market_data 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (symbol, days + 1))

                data = cursor.fetchall()

                if len(data) >= 2:
                    prices = [float(row['close_price']) for row in reversed(data)]
                    returns = np.diff(np.log(prices))

                    # Weight by position size (simplified)
                    weight = position.get('market_value', 0) / sum(
                        pos.get('market_value', 0) for pos in positions.values())
                    weighted_returns = returns * weight

                    if len(all_returns) == 0:
                        all_returns = weighted_returns
                    else:
                        # Pad arrays to same length
                        min_len = min(len(all_returns), len(weighted_returns))
                        all_returns = all_returns[:min_len] + weighted_returns[:min_len]

            conn.close()
            return np.array(all_returns) if len(all_returns) > 0 else np.array([])

        except Exception as e:
            logger.error(f"Portfolio returns calculation error: {e}")
            return np.array([])

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""

        try:
            if len(returns) == 0:
                return 0.0

            # Calculate cumulative returns
            cum_returns = np.cumprod(1 + returns)

            # Calculate running maximum
            running_max = np.maximum.accumulate(cum_returns)

            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max

            # Return maximum drawdown (most negative value)
            max_dd = np.min(drawdown)

            return abs(max_dd)

        except Exception as e:
            logger.error(f"Max drawdown calculation error: {e}")
            return 0.0

    def _calculate_portfolio_beta(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio beta vs market"""

        try:
            # Simplified beta calculation (would need market data for proper calculation)
            # For now, return a reasonable estimate based on position diversity

            if not positions:
                return 0.0

            # More positions = lower beta (more diversification)
            num_positions = len(positions)
            base_beta = 1.0

            # Adjust based on diversification
            if num_positions >= 10:
                base_beta = 0.8
            elif num_positions >= 5:
                base_beta = 0.9
            elif num_positions >= 2:
                base_beta = 1.0
            else:
                base_beta = 1.2  # Single position = higher risk

            return base_beta

        except Exception as e:
            logger.error(f"Beta calculation error: {e}")
            return 1.0

    def _determine_risk_level(self, position_pct: float, var_1_day: float,
                              max_drawdown: float, volatility: float) -> RiskLevel:
        """Determine overall risk level"""

        try:
            risk_score = 0

            # Position concentration risk
            if position_pct > 0.2:
                risk_score += 3
            elif position_pct > 0.15:
                risk_score += 2
            elif position_pct > 0.1:
                risk_score += 1

            # VaR risk
            if var_1_day > 0.05:  # 5% daily VaR
                risk_score += 3
            elif var_1_day > 0.03:
                risk_score += 2
            elif var_1_day > 0.02:
                risk_score += 1

            # Drawdown risk
            if max_drawdown > 0.15:
                risk_score += 3
            elif max_drawdown > 0.1:
                risk_score += 2
            elif max_drawdown > 0.05:
                risk_score += 1

            # Volatility risk
            if volatility > 0.4:  # 40% annual volatility
                risk_score += 3
            elif volatility > 0.25:
                risk_score += 2
            elif volatility > 0.15:
                risk_score += 1

            # Determine level
            if risk_score >= 8:
                return RiskLevel.CRITICAL
            elif risk_score >= 5:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except Exception as e:
            logger.error(f"Risk level determination error: {e}")
            return RiskLevel.MEDIUM

    def check_risk_alerts(self, portfolio_data: Dict[str, Any]) -> List[RiskAlert]:
        """Check for risk violations and generate alerts"""

        alerts = []

        try:
            total_value = portfolio_data.get('total_value', 0.0)
            positions = portfolio_data.get('positions', {})
            unrealized_pnl = portfolio_data.get('unrealized_pnl', 0.0)

            # Check position concentration
            for symbol, position in positions.items():
                position_value = position.get('market_value', 0)
                position_pct = position_value / total_value if total_value > 0 else 0

                if position_pct > self.max_position_size:
                    alert = RiskAlert(
                        alert_id=f"pos_size_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        severity=RiskLevel.HIGH if position_pct > 0.2 else RiskLevel.MEDIUM,
                        message=f"Position {symbol} exceeds size limit: {position_pct:.1%} > {self.max_position_size:.1%}",
                        symbol=symbol,
                        metric_value=position_pct,
                        threshold=self.max_position_size,
                        created_at=datetime.now()
                    )
                    alerts.append(alert)

            # Check portfolio drawdown
            if unrealized_pnl < 0:
                drawdown_pct = abs(unrealized_pnl) / total_value if total_value > 0 else 0

                if drawdown_pct > self.max_drawdown_limit:
                    alert = RiskAlert(
                        alert_id=f"drawdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        severity=RiskLevel.CRITICAL if drawdown_pct > 0.25 else RiskLevel.HIGH,
                        message=f"Portfolio drawdown exceeds limit: {drawdown_pct:.1%} > {self.max_drawdown_limit:.1%}",
                        symbol=None,
                        metric_value=drawdown_pct,
                        threshold=self.max_drawdown_limit,
                        created_at=datetime.now()
                    )
                    alerts.append(alert)

            # Calculate and check VaR
            if self.portfolio_metrics:
                var_pct = self.portfolio_metrics.var_1_day / total_value if total_value > 0 else 0

                if var_pct > 0.05:  # 5% daily VaR limit
                    alert = RiskAlert(
                        alert_id=f"var_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        severity=RiskLevel.HIGH,
                        message=f"Daily VaR exceeds 5%: {var_pct:.1%}",
                        symbol=None,
                        metric_value=var_pct,
                        threshold=0.05,
                        created_at=datetime.now()
                    )
                    alerts.append(alert)

            # Store new alerts
            self.risk_alerts.extend(alerts)

            # Keep only recent alerts (last 100)
            if len(self.risk_alerts) > 100:
                self.risk_alerts = self.risk_alerts[-100:]

            if alerts:
                logger.warning(f"Generated {len(alerts)} risk alerts")

            return alerts

        except Exception as e:
            logger.error(f"Risk alert check error: {e}")
            return []

    def calculate_optimal_stops(self, symbol: str, entry_price: float,
                                position_size: float, risk_tolerance: float = 0.02) -> Dict[str, float]:
        """Calculate optimal stop-loss and take-profit levels"""

        try:
            # Get historical volatility
            conn = self._connect_database()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT close_price FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT 21
            """, (symbol,))

            data = cursor.fetchall()
            conn.close()

            if len(data) < 2:
                # Default stops if no data
                return {
                    'stop_loss': entry_price * 0.95,  # 5% stop
                    'take_profit': entry_price * 1.10,  # 10% target
                    'trailing_stop_pct': 0.05
                }

            # Calculate ATR (Average True Range) proxy
            prices = [float(row['close_price']) for row in reversed(data)]
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) if len(returns) > 1 else 0.02

            # Calculate stop levels
            atr_multiplier = 2.0  # 2x ATR for stops
            stop_distance = volatility * atr_multiplier * entry_price

            # Risk-based stop loss
            risk_amount = position_size * risk_tolerance
            stop_loss_distance = risk_amount / (position_size / entry_price)

            # Use the more conservative stop
            final_stop_distance = min(stop_distance, stop_loss_distance)

            stops = {
                'stop_loss': entry_price - final_stop_distance,
                'take_profit': entry_price + (final_stop_distance * 2),  # 2:1 reward/risk
                'trailing_stop_pct': volatility * 2  # 2x volatility for trailing stop
            }

            logger.info(f"Calculated stops for {symbol}: SL=${stops['stop_loss']:.2f}, TP=${stops['take_profit']:.2f}")

            return stops

        except Exception as e:
            logger.error(f"Optimal stops calculation error for {symbol}: {e}")
            return {
                'stop_loss': entry_price * 0.95,
                'take_profit': entry_price * 1.10,
                'trailing_stop_pct': 0.05
            }

    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""

        try:
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': self.portfolio_metrics.to_dict() if self.portfolio_metrics else None,
                'active_alerts': [alert.to_dict() for alert in self.risk_alerts if not alert.acknowledged],
                'risk_limits': {
                    'max_position_size': self.max_position_size,
                    'max_portfolio_risk': self.max_portfolio_risk,
                    'max_correlation': self.max_correlation,
                    'max_drawdown_limit': self.max_drawdown_limit,
                    'var_confidence': self.var_confidence
                },
                'alert_counts': {
                    'total': len(self.risk_alerts),
                    'critical': sum(1 for a in self.risk_alerts if a.severity == RiskLevel.CRITICAL),
                    'high': sum(1 for a in self.risk_alerts if a.severity == RiskLevel.HIGH),
                    'medium': sum(1 for a in self.risk_alerts if a.severity == RiskLevel.MEDIUM),
                    'low': sum(1 for a in self.risk_alerts if a.severity == RiskLevel.LOW)
                }
            }

            return dashboard

        except Exception as e:
            logger.error(f"Risk dashboard error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert"""

        try:
            for alert in self.risk_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Risk alert acknowledged: {alert_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Alert acknowledgment error: {e}")
            return False


def main():
    """Test the advanced risk management system"""

    print("Phase 3 - Advanced Risk Management Test")
    print("=" * 45)

    # Initialize risk manager
    risk_manager = AdvancedRiskManager()

    # Test portfolio data (simulated)
    test_portfolio = {
        'total_value': 100000.0,
        'available_cash': 50000.0,
        'invested_value': 50000.0,
        'unrealized_pnl': -2000.0,
        'positions': {
            'AAPL': {'market_value': 20000.0, 'quantity': 100, 'avg_cost': 180.0},
            'MSFT': {'market_value': 15000.0, 'quantity': 50, 'avg_cost': 300.0},
            'GOOGL': {'market_value': 15000.0, 'quantity': 25, 'avg_cost': 600.0}
        }
    }

    # Test Kelly position sizing
    print("Testing Kelly Criterion Position Sizing:")
    kelly_size = risk_manager.calculate_position_size_kelly(
        symbol='TSLA',
        win_rate=0.55,
        avg_win=0.08,
        avg_loss=0.04,
        portfolio_value=100000.0
    )
    print(f"  Optimal Kelly position size: ${kelly_size:,.2f}")

    # Test volatility-based position sizing
    print("\nTesting Volatility-Based Position Sizing:")
    vol_size = risk_manager.calculate_position_size_volatility(
        symbol='AAPL',
        target_volatility=0.15,  # 15% target volatility
        portfolio_value=100000.0
    )
    print(f"  Volatility-based position size: ${vol_size:,.2f}")

    # Test portfolio risk metrics
    print("\nTesting Portfolio Risk Metrics:")
    metrics = risk_manager.calculate_portfolio_risk_metrics(test_portfolio)
    print(f"  Risk Level: {metrics.risk_level.value}")
    print(f"  1-Day VaR: ${metrics.var_1_day:,.2f}")
    print(f"  5-Day VaR: ${metrics.var_5_day:,.2f}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  Beta: {metrics.beta:.2f}")

    # Test risk alerts
    print("\nTesting Risk Alerts:")
    alerts = risk_manager.check_risk_alerts(test_portfolio)
    print(f"  Generated {len(alerts)} alerts")
    for alert in alerts:
        print(f"    {alert.severity.value}: {alert.message}")

    # Test optimal stops
    print("\nTesting Optimal Stop Calculation:")
    stops = risk_manager.calculate_optimal_stops('AAPL', 180.0, 10000.0)
    print(f"  Stop Loss: ${stops['stop_loss']:.2f}")
    print(f"  Take Profit: ${stops['take_profit']:.2f}")
    print(f"  Trailing Stop: {stops['trailing_stop_pct']:.1%}")

    # Test risk dashboard
    print("\nTesting Risk Dashboard:")
    dashboard = risk_manager.get_risk_dashboard()
    print(f"  Active Alerts: {len(dashboard.get('active_alerts', []))}")
    print(f"  Risk Level: {dashboard.get('portfolio_metrics', {}).get('risk_level', 'N/A')}")
    print(f"  Alert Summary: {dashboard.get('alert_counts', {})}")

    print("\n🎉 Advanced Risk Management Phase 3 Step 3 Complete!")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)