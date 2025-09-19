# 03_ML_ENGINE/models/advanced_mock_models.py
"""
Advanced Mock Models for ML Signal Enhancer - Phase 2, Step 2
Realistic market dynamics simulation for comprehensive testing

Location: #03_ML_ENGINE/models/advanced_mock_models.py

This module provides sophisticated mock models that simulate:
- Market regime detection (bull, bear, sideways)
- Time-series correlation and momentum
- Volatility clustering effects
- Sector rotation patterns
- Economic cycle impacts
- Risk-on/risk-off sentiment shifts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


@dataclass
class MarketContext:
    """Market context for realistic predictions"""
    regime: MarketRegime
    volatility_level: float  # 0.0 to 1.0
    trend_strength: float  # -1.0 to 1.0
    volume_profile: str  # 'high', 'normal', 'low'
    sector_rotation: str  # 'tech', 'finance', 'energy', 'defensive'
    risk_sentiment: float  # -1.0 (risk-off) to 1.0 (risk-on)


class AdvancedAlphaModel:
    """
    Advanced Alpha Model with realistic market behavior simulation

    Features:
    - Market regime awareness
    - Factor model simulation (momentum, mean reversion, quality, volatility)
    - Sector-specific behavior
    - Time-series correlation
    - Economic cycle sensitivity
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize advanced alpha model"""

        self.config = config or {
            'lookback_periods': 30,
            'regime_sensitivity': 0.7,
            'factor_weights': {
                'momentum': 0.25,
                'mean_reversion': 0.20,
                'quality': 0.20,
                'volatility': 0.15,
                'sentiment': 0.20
            },
            'sector_betas': {
                'TECH': 1.3,
                'FINANCE': 1.1,
                'ENERGY': 1.4,
                'CONSUMER': 0.8,
                'HEALTHCARE': 0.9
            }
        }

        # State tracking for realistic predictions
        self.prediction_history = {}
        self.market_memory = []

        logger.info("🧠 Advanced Alpha Model initialized")

    def detect_market_regime(self, market_data: pd.DataFrame) -> MarketContext:
        """
        Detect current market regime based on price action

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            MarketContext: Current market conditions
        """
        if len(market_data) < 20:
            return MarketContext(
                regime=MarketRegime.SIDEWAYS,
                volatility_level=0.5,
                trend_strength=0.0,
                volume_profile='normal',
                sector_rotation='tech',
                risk_sentiment=0.0
            )

        # Calculate market metrics
        returns = market_data['close'].pct_change().dropna()
        recent_returns = returns.tail(10)
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Trend analysis
        sma_20 = market_data['close'].rolling(20).mean()
        current_price = market_data['close'].iloc[-1]
        trend_strength = (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1] if not pd.isna(sma_20.iloc[-1]) else 0

        # Volume analysis
        avg_volume = market_data['volume'].rolling(20).mean().iloc[-1] if 'volume' in market_data.columns else 1000000
        recent_volume = market_data['volume'].iloc[-1] if 'volume' in market_data.columns else 1000000
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Regime classification
        if trend_strength > 0.05 and recent_returns.mean() > 0.01:
            regime = MarketRegime.BULL
        elif trend_strength < -0.05 and recent_returns.mean() < -0.01:
            regime = MarketRegime.BEAR
        elif volatility > 0.3:
            regime = MarketRegime.VOLATILE
        else:
            regime = MarketRegime.SIDEWAYS

        # Volume profile
        if volume_ratio > 1.3:
            volume_profile = 'high'
        elif volume_ratio < 0.7:
            volume_profile = 'low'
        else:
            volume_profile = 'normal'

        # Risk sentiment (simplified)
        risk_sentiment = np.clip(trend_strength * 2, -1.0, 1.0)

        return MarketContext(
            regime=regime,
            volatility_level=np.clip(volatility / 0.5, 0.0, 1.0),
            trend_strength=np.clip(trend_strength, -1.0, 1.0),
            volume_profile=volume_profile,
            sector_rotation='tech',  # Simplified for now
            risk_sentiment=risk_sentiment
        )

    def calculate_factor_scores(self, symbol: str, market_data: pd.DataFrame, context: MarketContext) -> Dict[
        str, float]:
        """
        Calculate multi-factor scores for alpha generation

        Args:
            symbol: Trading symbol
            market_data: Historical market data
            context: Current market context

        Returns:
            Dict: Factor scores for prediction
        """
        scores = {}

        if len(market_data) < 10:
            return {factor: 0.0 for factor in self.config.get('factor_weights', {}).keys()}

        returns = market_data['close'].pct_change().dropna()

        # Momentum Factor
        recent_momentum = returns.tail(5).mean()
        momentum_score = np.tanh(recent_momentum * 100)  # Normalize to [-1, 1]

        # Mean Reversion Factor
        sma_20 = market_data['close'].rolling(20).mean()
        current_price = market_data['close'].iloc[-1]
        if not pd.isna(sma_20.iloc[-1]) and sma_20.iloc[-1] > 0:
            price_deviation = (current_price - sma_20.iloc[-1]) / sma_20.iloc[-1]
            mean_reversion_score = -np.tanh(price_deviation * 5)  # Contrarian signal
        else:
            mean_reversion_score = 0.0

        # Quality Factor (simplified using price stability)
        if len(returns) >= 20:
            price_stability = 1 / (1 + returns.tail(20).std())
            quality_score = np.tanh((price_stability - 0.5) * 10)
        else:
            quality_score = 0.0

        # Volatility Factor
        if len(returns) >= 10:
            recent_vol = returns.tail(10).std()
            vol_percentile = (recent_vol - returns.std()) / returns.std() if returns.std() > 0 else 0
            volatility_score = -np.tanh(vol_percentile)  # Prefer lower volatility
        else:
            volatility_score = 0.0

        # Sentiment Factor (based on market context)
        sentiment_score = context.risk_sentiment * 0.8  # Scale down the impact

        scores = {
            'momentum': momentum_score,
            'mean_reversion': mean_reversion_score,
            'quality': quality_score,
            'volatility': volatility_score,
            'sentiment': sentiment_score
        }

        return scores

    def predict_profitability(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate advanced alpha prediction with market regime awareness

        Args:
            symbol: Trading symbol
            market_data: Historical market data

        Returns:
            Dict: Comprehensive prediction with factors and reasoning
        """
        # Detect market context
        context = self.detect_market_regime(market_data)

        # Calculate factor scores
        factor_scores = self.calculate_factor_scores(symbol, market_data, context)

        # Compute weighted alpha score
        alpha_score = 0.0
        factor_weights = self.config.get('factor_weights', {
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'quality': 0.20,
            'volatility': 0.15,
            'sentiment': 0.20
        })

        for factor, score in factor_scores.items():
            weight = factor_weights.get(factor, 0.0)
            alpha_score += weight * score

        # Apply market regime adjustments
        regime_multiplier = {
            MarketRegime.BULL: 1.2,
            MarketRegime.BEAR: 0.7,
            MarketRegime.SIDEWAYS: 0.9,
            MarketRegime.VOLATILE: 0.8
        }

        adjusted_alpha = alpha_score * regime_multiplier[context.regime]

        # Generate signal
        if adjusted_alpha > 0.15:
            signal = 'BUY'
            confidence = min(0.9, 0.6 + abs(adjusted_alpha))
        elif adjusted_alpha < -0.15:
            signal = 'SELL'
            confidence = min(0.9, 0.6 + abs(adjusted_alpha))
        else:
            signal = 'HOLD'
            confidence = 0.5 + abs(adjusted_alpha) * 0.5

        # Store prediction for consistency
        prediction_key = f"{symbol}_{datetime.now().strftime('%Y%m%d')}"
        self.prediction_history[prediction_key] = {
            'alpha_score': adjusted_alpha,
            'context': context,
            'timestamp': datetime.now()
        }

        return {
            'signal': signal,
            'confidence': confidence,
            'alpha_score': adjusted_alpha,
            'factors': factor_scores,
            'market_regime': context.regime.value,
            'volatility_level': context.volatility_level,
            'trend_strength': context.trend_strength,
            'risk_sentiment': context.risk_sentiment,
            'reasoning': f"{signal} signal in {context.regime.value} market (α={adjusted_alpha:.3f})"
        }


class AdvancedLSTMModel:
    """
    Advanced LSTM Model with realistic time-series behavior

    Features:
    - Multi-timeframe analysis
    - Volatility clustering simulation
    - Autocorrelation patterns
    - Regime-dependent predictions
    - Confidence bands
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize advanced LSTM model"""

        self.config = config or {
            'sequence_length': 30,
            'forecast_horizon': 5,
            'confidence_intervals': [0.68, 0.95],  # 1σ and 2σ
            'volatility_memory': 0.9,  # GARCH-like volatility persistence
            'trend_persistence': 0.7
        }

        # State tracking
        self.volatility_state = {}
        self.trend_state = {}
        self.prediction_cache = {}

        logger.info("🧠 Advanced LSTM Model initialized")

    def simulate_lstm_prediction(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate sophisticated LSTM prediction with time-series characteristics

        Args:
            symbol: Trading symbol
            market_data: Historical market data

        Returns:
            Dict: Advanced prediction with confidence bands and reasoning
        """
        if len(market_data) < 10:
            return self._default_prediction()

        # Extract time series features
        returns = market_data['close'].pct_change().dropna()
        current_price = market_data['close'].iloc[-1]

        # Volatility clustering (GARCH-like behavior)
        if symbol not in self.volatility_state:
            self.volatility_state[symbol] = returns.std()

        recent_vol = returns.tail(5).std()
        persistence = self.config.get('volatility_memory', 0.9)
        self.volatility_state[symbol] = (persistence * self.volatility_state[symbol] +
                                         (1 - persistence) * recent_vol)

        # Trend persistence
        if symbol not in self.trend_state:
            self.trend_state[symbol] = 0.0

        recent_trend = returns.tail(5).mean()
        trend_persistence = self.config.get('trend_persistence', 0.7)
        self.trend_state[symbol] = (trend_persistence * self.trend_state[symbol] +
                                    (1 - trend_persistence) * recent_trend)

        # Simulate LSTM prediction logic
        # 1. Momentum component
        momentum_signal = np.tanh(self.trend_state[symbol] * 50)

        # 2. Mean reversion component
        sma_20 = market_data['close'].rolling(20).mean().iloc[-1] if len(market_data) >= 20 else current_price
        price_deviation = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        mean_reversion_signal = -np.tanh(price_deviation * 3)

        # 3. Volatility component
        vol_factor = 1 / (1 + self.volatility_state[symbol] * 10)  # Lower volatility = higher confidence

        # Combine components (simplified LSTM-like logic)
        base_prediction = 0.6 * momentum_signal + 0.4 * mean_reversion_signal

        # Price prediction
        horizon_days = self.config.get('forecast_horizon', 5)
        expected_return = base_prediction * 0.02 * horizon_days  # Scale to reasonable return
        predicted_price = current_price * (1 + expected_return)

        # Generate signal and confidence
        if expected_return > 0.01:
            signal = 'BUY'
            base_confidence = 0.6 + abs(expected_return) * 5
        elif expected_return < -0.01:
            signal = 'SELL'
            base_confidence = 0.6 + abs(expected_return) * 5
        else:
            signal = 'HOLD'
            base_confidence = 0.5

        # Apply volatility adjustment to confidence
        confidence = min(0.9, base_confidence * vol_factor)

        # Confidence intervals
        volatility_daily = self.volatility_state[symbol]
        horizon_vol = volatility_daily * np.sqrt(horizon_days)

        confidence_bands = {
            '68%': (predicted_price * (1 - horizon_vol), predicted_price * (1 + horizon_vol)),
            '95%': (predicted_price * (1 - 2 * horizon_vol), predicted_price * (1 + 2 * horizon_vol))
        }

        return {
            'signal': signal,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'price_change_pct': expected_return,
            'horizon': horizon_days,
            'confidence_bands': confidence_bands,
            'volatility_forecast': volatility_daily,
            'trend_momentum': self.trend_state[symbol],
            'reasoning': f"{signal} prediction: {expected_return:.2%} over {horizon_days}d (σ={volatility_daily:.3f})"
        }

    def predict_profitability(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Main prediction interface matching original API

        Args:
            symbol: Trading symbol
            market_data: Historical market data

        Returns:
            Dict: Prediction results
        """
        return self.simulate_lstm_prediction(symbol, market_data)

    def _default_prediction(self) -> Dict[str, Any]:
        """Default prediction for insufficient data"""
        return {
            'signal': 'HOLD',
            'confidence': 0.5,
            'predicted_price': 100.0,
            'price_change_pct': 0.0,
            'horizon': self.config.get('forecast_horizon', 5),
            'confidence_bands': {'68%': (95, 105), '95%': (90, 110)},
            'volatility_forecast': 0.02,
            'trend_momentum': 0.0,
            'reasoning': "Insufficient data for prediction"
        }


class AdvancedMarketSimulator:
    """
    Market scenario simulator for comprehensive testing

    Creates various market conditions:
    - Bull market momentum
    - Bear market selloffs
    - Sideways consolidation
    - Flash crash scenarios
    - Sector rotation events
    """

    def __init__(self):
        """Initialize market simulator"""
        self.scenarios = {
            'bull_momentum': self._create_bull_scenario,
            'bear_selloff': self._create_bear_scenario,
            'sideways_range': self._create_sideways_scenario,
            'flash_crash': self._create_crash_scenario,
            'volatility_spike': self._create_volatile_scenario
        }

        logger.info("🎯 Advanced Market Simulator initialized")

    def generate_scenario(self, scenario_type: str, symbol: str, days: int = 100) -> pd.DataFrame:
        """
        Generate market data for specific scenario

        Args:
            scenario_type: Type of market scenario
            symbol: Trading symbol
            days: Number of days to simulate

        Returns:
            DataFrame: Simulated market data
        """
        if scenario_type not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_type}")

        return self.scenarios[scenario_type](symbol, days)

    def _create_bull_scenario(self, symbol: str, days: int) -> pd.DataFrame:
        """Create bull market scenario"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # Bull market: upward trend with occasional pullbacks
        base_trend = np.cumsum(np.random.normal(0.001, 0.015, days))  # 0.1% daily drift
        noise = np.random.normal(0, 0.01, days)

        # Add pullbacks every 20-30 days
        for i in range(20, days, 25):
            pullback_length = np.random.randint(3, 8)
            end_idx = min(i + pullback_length, days)
            base_trend[i:end_idx] -= np.linspace(0, 0.05, end_idx - i)

        returns = base_trend + noise
        prices = 100 * np.exp(np.cumsum(returns))

        return self._create_ohlcv_data(dates, prices, 'bull')

    def _create_bear_scenario(self, symbol: str, days: int) -> pd.DataFrame:
        """Create bear market scenario"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # Bear market: downward trend with sharp rallies
        base_trend = np.cumsum(np.random.normal(-0.0008, 0.02, days))  # -0.08% daily drift
        noise = np.random.normal(0, 0.015, days)

        # Add relief rallies
        for i in range(15, days, 20):
            rally_length = np.random.randint(2, 5)
            end_idx = min(i + rally_length, days)
            base_trend[i:end_idx] += np.linspace(0, 0.03, end_idx - i)

        returns = base_trend + noise
        prices = 100 * np.exp(np.cumsum(returns))

        return self._create_ohlcv_data(dates, prices, 'bear')

    def _create_sideways_scenario(self, symbol: str, days: int) -> pd.DataFrame:
        """Create sideways market scenario"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # Sideways: mean-reverting around 100
        prices = [100]
        for i in range(1, days):
            # Mean reversion force
            reversion = -0.01 * (prices[-1] - 100) / 100
            noise = np.random.normal(0, 0.012)
            daily_return = reversion + noise
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)

        return self._create_ohlcv_data(dates, prices, 'sideways')

    def _create_crash_scenario(self, symbol: str, days: int) -> pd.DataFrame:
        """Create flash crash scenario"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # Normal market followed by crash and recovery
        normal_days = days // 2
        crash_day = normal_days

        # Normal period
        normal_returns = np.random.normal(0, 0.01, normal_days)

        # Crash period
        crash_returns = [-0.08, -0.06, -0.04]  # 3-day crash

        # Recovery period
        recovery_days = days - normal_days - len(crash_returns)
        recovery_returns = np.random.normal(0.002, 0.015, recovery_days)  # Gradual recovery

        all_returns = np.concatenate([normal_returns, crash_returns, recovery_returns])
        prices = 100 * np.exp(np.cumsum(all_returns))

        return self._create_ohlcv_data(dates, prices, 'crash')

    def _create_volatile_scenario(self, symbol: str, days: int) -> pd.DataFrame:
        """Create high volatility scenario"""
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')

        # High volatility with volatility clustering
        volatility = [0.02]
        returns = []

        for i in range(days):
            # GARCH-like volatility clustering
            if i > 0:
                new_vol = 0.8 * volatility[-1] + 0.15 * abs(returns[-1]) + 0.05 * 0.02
                volatility.append(new_vol)

            daily_return = np.random.normal(0, volatility[-1])
            returns.append(daily_return)

        prices = 100 * np.exp(np.cumsum(returns))

        return self._create_ohlcv_data(dates, prices, 'volatile')

    def _create_ohlcv_data(self, dates: pd.DatetimeIndex, close_prices: np.ndarray, scenario_type: str) -> pd.DataFrame:
        """Create OHLCV data from close prices"""

        n = len(close_prices)

        # Generate realistic OHLC from close prices
        opens = np.roll(close_prices, 1)
        opens[0] = close_prices[0]

        # Highs and lows based on intraday volatility
        daily_vol = 0.015 if scenario_type == 'sideways' else 0.025
        if scenario_type == 'volatile':
            daily_vol = 0.04

        highs = close_prices * (1 + np.abs(np.random.normal(0, daily_vol, n)))
        lows = close_prices * (1 - np.abs(np.random.normal(0, daily_vol, n)))

        # Ensure OHLC relationships are valid
        for i in range(n):
            high_candidate = max(opens[i], close_prices[i], highs[i])
            low_candidate = min(opens[i], close_prices[i], lows[i])
            highs[i] = high_candidate
            lows[i] = low_candidate

        # Volume based on scenario
        base_volume = 1000000
        if scenario_type == 'crash':
            volume_multiplier = np.where(np.arange(n) == n // 2, 5, 1)  # Spike on crash day
        elif scenario_type == 'volatile':
            volume_multiplier = 1 + np.random.exponential(0.5, n)
        else:
            volume_multiplier = 1 + np.random.gamma(2, 0.2, n)

        volumes = base_volume * volume_multiplier

        return pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes.astype(int)
        })


# Factory function for easy model creation
def create_advanced_models(config: Optional[Dict] = None) -> Tuple[AdvancedAlphaModel, AdvancedLSTMModel]:
    """
    Factory function to create advanced mock models

    Args:
        config: Optional configuration for models

    Returns:
        Tuple: (AdvancedAlphaModel, AdvancedLSTMModel)
    """
    alpha_model = AdvancedAlphaModel(config)
    lstm_model = AdvancedLSTMModel(config)

    logger.info("✅ Advanced mock models created successfully")

    return alpha_model, lstm_model


if __name__ == "__main__":
    # Test the advanced mock models
    print("\n=== Testing Advanced Mock Models ===\n")

    # Create models
    alpha_model, lstm_model = create_advanced_models()

    # Create market simulator
    simulator = AdvancedMarketSimulator()

    # Test different scenarios
    scenarios = ['bull_momentum', 'bear_selloff', 'sideways_range', 'flash_crash']

    for scenario in scenarios:
        print(f"\n--- Testing {scenario.upper()} scenario ---")

        # Generate market data
        market_data = simulator.generate_scenario(scenario, 'TEST', days=50)

        # Test Alpha Model
        alpha_pred = alpha_model.predict_profitability('TEST', market_data)
        print(f"Alpha Model: {alpha_pred['signal']} (confidence: {alpha_pred['confidence']:.2%})")
        print(f"  Reasoning: {alpha_pred['reasoning']}")

        # Test LSTM Model
        lstm_pred = lstm_model.predict_profitability('TEST', market_data)
        print(f"LSTM Model: {lstm_pred['signal']} (confidence: {lstm_pred['confidence']:.2%})")
        print(f"  Reasoning: {lstm_pred['reasoning']}")

    print("\n✅ Advanced Mock Models test complete!")