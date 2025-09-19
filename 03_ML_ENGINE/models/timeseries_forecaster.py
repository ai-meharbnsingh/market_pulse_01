# src/models/timeseries_forecaster.py
"""
Time-Series Forecasting Module - Phase 1 Day 9
Advanced price prediction using LSTM and Prophet models

Provides multi-timeframe forecasting capabilities:
- LSTM for short-term intraday predictions (1H, 4H)
- Prophet for daily/weekly trend forecasting
- Ensemble approach combining both methods
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import joblib
import sqlite3
from pathlib import Path

# Deep Learning with fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True


    class LSTMPricePredictor(nn.Module):
        """
        LSTM Neural Network for short-term price prediction

        Architecture optimized for financial time series with:
        - Multiple LSTM layers for pattern recognition
        - Dropout for regularization
        - Dense layers for final prediction
        """

        def __init__(self, input_features: int = 20, sequence_length: int = 60,
                     hidden_size: int = 50, num_layers: int = 2, dropout: float = 0.2):
            super(LSTMPricePredictor, self).__init__()

            self.input_features = input_features
            self.sequence_length = sequence_length
            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers
            self.lstm = nn.LSTM(
                input_features,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )

            # Dense layers for prediction
            self.dropout = nn.Dropout(dropout)
            self.dense1 = nn.Linear(hidden_size, 25)
            self.dense2 = nn.Linear(25, 1)
            self.relu = nn.ReLU()

        def forward(self, x):
            # LSTM forward pass
            lstm_out, (hn, cn) = self.lstm(x)

            # Take the last output
            last_output = lstm_out[:, -1, :]

            # Dense layers
            out = self.dropout(last_output)
            out = self.relu(self.dense1(out))
            out = self.dropout(out)
            out = self.dense2(out)

            return out

except ImportError:
    TORCH_AVAILABLE = False
    LSTMPricePredictor = None

# Prophet disabled due to NumPy 2.0 compatibility issues
PROPHET_AVAILABLE = False
Prophet = None

# Technical Analysis with fallback
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not TORCH_AVAILABLE:
    logger.warning("PyTorch not available, using statistical fallback")
if not PROPHET_AVAILABLE:
    logger.warning("Prophet not available, skipping long-term forecasting")
if not PANDAS_TA_AVAILABLE:
    logger.warning("pandas_ta not available, using basic technical indicators")


class TimeSeriesForecaster:
    """
    Comprehensive time-series forecasting system combining LSTM and Prophet

    Features:
    - Multi-timeframe predictions (1H, 4H, 1D, 1W)
    - LSTM for short-term technical patterns
    - Prophet for long-term trends and seasonality
    - Ensemble predictions combining both approaches
    - Feature engineering for enhanced predictions
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the forecasting system"""

        # Configuration
        self.sequence_length = 60  # Look back 60 periods
        self.prediction_horizons = {
            '1H': 1,  # 1 hour ahead
            '4H': 4,  # 4 hours ahead
            '1D': 24,  # 24 hours ahead (1 day)
            '1W': 168  # 168 hours ahead (1 week)
        }

        # Model storage
        self.models_dir = Path("models/timeseries")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Scalers for normalization
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()

        # Models
        self.lstm_models = {}
        self.prophet_models = {}
        self.ensemble_weights = {}

        # Database connection
        self.db_path = "marketpulse.db"  # Use relative path
        self._init_database()

        logger.info("Time-Series Forecaster initialized")

    def _init_database(self):
        """Initialize database tables for forecasting data"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Price predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_predictions (
                prediction_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

                -- Current market data
                current_price REAL NOT NULL,
                current_volume REAL,

                -- LSTM Predictions
                lstm_1h_price REAL,
                lstm_4h_price REAL,
                lstm_1d_price REAL,
                lstm_confidence REAL,

                -- Prophet Predictions
                prophet_1d_price REAL,
                prophet_1w_price REAL,
                prophet_trend REAL,
                prophet_confidence REAL,

                -- Ensemble Predictions
                ensemble_1h_price REAL,
                ensemble_4h_price REAL,
                ensemble_1d_price REAL,
                ensemble_1w_price REAL,
                ensemble_confidence REAL,

                -- Technical context
                rsi_14 REAL,
                macd_signal REAL,
                bb_position REAL,
                volume_sma_ratio REAL,

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_performance (
                performance_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL CHECK (model_type IN ('LSTM', 'PROPHET', 'ENSEMBLE')),
                timeframe TEXT NOT NULL,
                symbol TEXT NOT NULL,

                -- Accuracy metrics
                mae REAL,  -- Mean Absolute Error
                rmse REAL, -- Root Mean Square Error
                mape REAL, -- Mean Absolute Percentage Error
                directional_accuracy REAL, -- % correct direction predictions

                -- Sample statistics
                prediction_count INTEGER,
                evaluation_period_days INTEGER,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

        logger.info("Forecasting database tables initialized")

    def predict(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Generate prediction for symbol compatible with ML Signal Enhancer

        Args:
            symbol: Stock symbol
            market_data: DataFrame with OHLCV data

        Returns:
            Dictionary with prediction results
        """

        logger.info(f"Generating LSTM prediction for {symbol}")

        try:
            # Generate multi-timeframe predictions
            predictions = self.predict_prices(symbol, market_data)

            # Extract key prediction for signal generation
            current_price = predictions['current_price']

            # Use 1H prediction as primary signal (most reliable for LSTM)
            predicted_price = predictions['ensemble_predictions'].get('1H')

            if predicted_price is None:
                # Fallback to statistical prediction
                return self._statistical_prediction(symbol, market_data)

            # Calculate price change
            price_change = predicted_price - current_price
            price_change_pct = price_change / current_price if current_price != 0 else 0

            # Determine signal based on prediction
            signal = self._price_change_to_signal(price_change_pct)

            # Get confidence from ensemble
            confidence = predictions['confidence_scores'].get('1H', 0.5)

            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'signal': signal,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'horizon': 1,
                'model': 'lstm_forecaster',
                'method': 'ensemble' if TORCH_AVAILABLE else 'statistical',
                'data_points': len(market_data),
                'multi_timeframe_predictions': predictions
            }

            logger.info(f"LSTM prediction: {signal} (Change: {price_change_pct:.2%}, "
                        f"Confidence: {confidence:.2%})")

            return prediction

        except Exception as e:
            logger.error(f"LSTM prediction failed for {symbol}: {e}")
            return self._create_neutral_prediction(symbol, f"error: {str(e)}")

    def _statistical_prediction(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """Fallback statistical prediction when deep learning is not available"""

        if len(market_data) < 20:
            return self._create_neutral_prediction(symbol, "insufficient_data")

        prices = market_data['close'].values
        current_price = prices[-1]

        # Simple trend + mean reversion model
        recent_prices = prices[-20:]

        # Linear trend component
        x = np.arange(len(recent_prices))
        trend_slope = np.polyfit(x, recent_prices, 1)[0]

        # Mean reversion component
        sma = np.mean(recent_prices)
        mean_reversion = (sma - current_price) * 0.1

        # Short-term momentum
        if len(prices) >= 5:
            momentum = (prices[-1] / prices[-5]) - 1
        else:
            momentum = 0

        # Combine components
        predicted_price = current_price + trend_slope + mean_reversion + (momentum * current_price * 0.1)

        # Calculate change
        price_change = predicted_price - current_price
        price_change_pct = price_change / current_price if current_price != 0 else 0

        # Determine signal
        signal = self._price_change_to_signal(price_change_pct)

        # Confidence based on price stability
        if len(prices) >= 20:
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            confidence = max(0.3, 1.0 - (volatility * 2))
            confidence = min(confidence, 0.8)
        else:
            confidence = 0.5

        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signal': signal,
            'confidence': confidence,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'horizon': 1,
            'model': 'lstm_forecaster',
            'method': 'statistical',
            'data_points': len(market_data)
        }

    def _price_change_to_signal(self, price_change_pct: float) -> str:
        """Convert predicted price change to trading signal"""

        # Thresholds for signal generation
        buy_threshold = 0.01  # 1% expected gain
        sell_threshold = -0.01  # 1% expected loss

        if price_change_pct >= buy_threshold:
            return 'BUY'
        elif price_change_pct <= sell_threshold:
            return 'SELL'
        else:
            return 'HOLD'

    def prepare_lstm_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for LSTM training

        Args:
            df: DataFrame with OHLCV data and technical indicators

        Returns:
            Tuple of (X, y, feature_names) where:
            - X: Feature sequences for training
            - y: Target prices for prediction
            - feature_names: List of feature column names
        """

        # Engineer technical features
        df = self._engineer_technical_features(df)

        # Select features for LSTM
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'sma_20', 'ema_12', 'ema_26', 'atr_14', 'adx_14',
            'stoch_k', 'stoch_d', 'williams_r', 'cci_14', 'roc_10'
        ]

        # Use only available columns
        available_features = [col for col in feature_columns if col in df.columns]

        if len(available_features) < 10:
            logger.warning(f"Limited features available: {len(available_features)}")

        # Prepare feature matrix
        feature_data = df[available_features].values

        # Normalize features
        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)

        # Prepare target (next period close price)
        target_data = df['close'].values
        target_data_scaled = self.price_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []

        for i in range(self.sequence_length, len(feature_data_scaled)):
            # Features: last sequence_length periods
            X.append(feature_data_scaled[i - self.sequence_length:i])
            # Target: next period close price
            y.append(target_data_scaled[i])

        return np.array(X), np.array(y), available_features

    def _engineer_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive technical features for LSTM"""

        df = df.copy()

        if PANDAS_TA_AVAILABLE:
            # Use pandas_ta for comprehensive indicators
            df = self._engineer_features_with_pandas_ta(df)
        else:
            # Use manual calculations as fallback
            df = self._engineer_features_manual(df)

        return df.fillna(method='bfill').fillna(method='ffill')

    def _engineer_features_with_pandas_ta(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using pandas_ta library"""

        try:
            # Basic technical indicators
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['rsi_21'] = ta.rsi(df['close'], length=21)

            # MACD with robust handling
            macd_data = ta.macd(df['close'])
            if isinstance(macd_data, pd.DataFrame) and not macd_data.empty:
                macd_cols = macd_data.columns.tolist()

                # Find MACD columns
                macd_col = next((col for col in macd_cols if 'MACD_12_26_9' in col), None)
                signal_col = next((col for col in macd_cols if 'MACDs_12_26_9' in col), None)
                hist_col = next((col for col in macd_cols if 'MACDh_12_26_9' in col), None)

                if macd_col:
                    df['macd'] = macd_data[macd_col]
                if signal_col:
                    df['macd_signal'] = macd_data[signal_col]
                if hist_col:
                    df['macd_histogram'] = macd_data[hist_col]

            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20)
            if isinstance(bb_data, pd.DataFrame) and not bb_data.empty:
                bb_cols = bb_data.columns.tolist()

                upper_col = next((col for col in bb_cols if 'BBU_20_2.0' in col), None)
                lower_col = next((col for col in bb_cols if 'BBL_20_2.0' in col), None)
                middle_col = next((col for col in bb_cols if 'BBM_20_2.0' in col), None)

                if upper_col and lower_col:
                    df['bb_upper'] = bb_data[upper_col]
                    df['bb_lower'] = bb_data[lower_col]

                    if middle_col:
                        df['bb_middle'] = bb_data[middle_col]
                    else:
                        df['bb_middle'] = (df['bb_upper'] + df['bb_lower']) / 2

                    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            # Moving Averages
            df['sma_5'] = ta.sma(df['close'], length=5)
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)

            # Volatility indicators
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

            # ADX
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if isinstance(adx_data, pd.DataFrame) and not adx_data.empty:
                adx_cols = adx_data.columns.tolist()
                adx_col = next((col for col in adx_cols if 'ADX_14' in col), None)
                if adx_col:
                    df['adx_14'] = adx_data[adx_col]

            # Stochastic
            stoch_data = ta.stoch(df['high'], df['low'], df['close'])
            if isinstance(stoch_data, pd.DataFrame) and not stoch_data.empty:
                stoch_cols = stoch_data.columns.tolist()
                k_col = next((col for col in stoch_cols if 'STOCHk_14_3_3' in col), None)
                d_col = next((col for col in stoch_cols if 'STOCHd_14_3_3' in col), None)

                if k_col:
                    df['stoch_k'] = stoch_data[k_col]
                if d_col:
                    df['stoch_d'] = stoch_data[d_col]

            # Other momentum indicators
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
            df['cci_14'] = ta.cci(df['high'], df['low'], df['close'], length=14)
            df['roc_10'] = ta.roc(df['close'], length=10)

            # Volume indicators
            df['volume_sma_20'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        except Exception as e:
            logger.warning(f"pandas_ta feature engineering failed: {e}, using manual calculation")
            df = self._engineer_features_manual(df)

        return df

    def _engineer_features_manual(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manual feature engineering as fallback"""

        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Simple moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std_dev = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Average True Range (simplified)
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = np.abs(df['high'] - df['close'].shift())
        df['low_close_prev'] = np.abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()

        # Stochastic oscillator
        lowest_low = df['low'].rolling(14).min()
        highest_high = df['high'].rolling(14).max()
        df['stoch_k'] = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Williams %R
        df['williams_r'] = ((highest_high - df['close']) / (highest_high - lowest_low)) * -100

        # Rate of Change
        df['roc_10'] = ((df['close'] / df['close'].shift(10)) - 1) * 100

        # Volume analysis
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price patterns
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']

        # Clean up temporary columns
        df = df.drop(['high_low', 'high_close_prev', 'low_close_prev', 'true_range'], axis=1, errors='ignore')

        return df

    def predict_prices(self, symbol: str, current_data: pd.DataFrame,
                       horizons: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate price predictions using ensemble of LSTM and Prophet

        Args:
            symbol: Stock symbol
            current_data: Recent market data for prediction
            horizons: List of prediction horizons ['1H', '4H', '1D', '1W']

        Returns:
            Dictionary with predictions for each horizon and model
        """

        if horizons is None:
            horizons = ['1H', '4H', '1D', '1W']

        predictions = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_data['close'].iloc[-1]),
            'lstm_predictions': {},
            'prophet_predictions': {},
            'ensemble_predictions': {},
            'confidence_scores': {}
        }

        # LSTM Predictions (for short-term horizons)
        if TORCH_AVAILABLE and symbol in self.lstm_models and len(current_data) >= self.sequence_length:
            predictions['lstm_predictions'] = self._get_lstm_predictions(
                symbol, current_data, horizons
            )
        else:
            # Use statistical method as fallback
            predictions['lstm_predictions'] = self._get_statistical_predictions(
                symbol, current_data, horizons
            )

        # Prophet Predictions (for longer-term horizons)
        if PROPHET_AVAILABLE and symbol in self.prophet_models:
            predictions['prophet_predictions'] = self._get_prophet_predictions(
                symbol, current_data, horizons
            )

        # Ensemble Predictions (combine both models)
        predictions['ensemble_predictions'] = self._create_ensemble_predictions(
            predictions['lstm_predictions'],
            predictions['prophet_predictions'],
            horizons
        )

        # Calculate confidence scores
        predictions['confidence_scores'] = self._calculate_confidence_scores(
            predictions, current_data
        )

        # Log predictions to database
        self._log_predictions(predictions)

        return predictions

    def _get_statistical_predictions(self, symbol: str, data: pd.DataFrame,
                                     horizons: List[str]) -> Dict[str, float]:
        """Get statistical predictions as fallback"""

        predictions = {}

        if len(data) < 20:
            return predictions

        prices = data['close'].values
        current_price = prices[-1]

        # Simple trend analysis
        recent_prices = prices[-20:]
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

        # Volatility estimate
        returns = np.diff(prices[-20:]) / prices[-21:-1]
        volatility = np.std(returns)

        for horizon in horizons:
            hours_ahead = self.prediction_horizons[horizon]

            # Project trend forward with some mean reversion
            trend_component = trend_slope * hours_ahead

            # Add some randomness based on volatility
            volatility_component = np.random.normal(0, volatility * current_price * np.sqrt(hours_ahead / 24))

            # Mean reversion component
            sma_20 = np.mean(recent_prices)
            mean_reversion = (sma_20 - current_price) * 0.1 * (hours_ahead / 24)

            predicted_price = current_price + trend_component + mean_reversion + volatility_component
            predictions[horizon] = float(max(predicted_price, current_price * 0.5))  # Prevent negative prices

        return predictions

    def _get_lstm_predictions(self, symbol: str, data: pd.DataFrame,
                              horizons: List[str]) -> Dict[str, float]:
        """Get LSTM model predictions"""

        if not TORCH_AVAILABLE:
            return self._get_statistical_predictions(symbol, data, horizons)

        try:
            # Load model if not in memory
            if symbol not in self.lstm_models:
                self._load_lstm_model(symbol)

            if symbol not in self.lstm_models:
                return self._get_statistical_predictions(symbol, data, horizons)

            model = self.lstm_models[symbol]

            # Prepare input sequence
            X, _, _ = self.prepare_lstm_data(data.tail(self.sequence_length + 10))

            if len(X) == 0:
                return self._get_statistical_predictions(symbol, data, horizons)

            # Get the last sequence for prediction
            last_sequence = torch.FloatTensor(X[-1:])

            # Predictions
            lstm_predictions = {}

            with torch.no_grad():
                model.eval()

                # Single step prediction
                prediction_scaled = model(last_sequence).item()

                # Inverse transform to get actual price
                prediction_price = self.price_scaler.inverse_transform([[prediction_scaled]])[0][0]

                # For different horizons, apply scaling factors
                current_price = data['close'].iloc[-1]

                for horizon in horizons:
                    if horizon in ['1H', '4H', '1D']:  # LSTM is good for short-term
                        # Apply volatility-adjusted scaling
                        if len(data) >= 14:
                            recent_prices = data['close'].tail(14).values
                            volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
                        else:
                            volatility = 0.02

                        # Scale prediction based on horizon
                        horizon_hours = self.prediction_horizons[horizon]
                        time_factor = np.sqrt(horizon_hours / 1.0)  # Scale with square root of time

                        # Adjust prediction
                        price_change = (prediction_price - current_price) * time_factor
                        final_prediction = current_price + price_change

                        lstm_predictions[horizon] = float(final_prediction)

            return lstm_predictions

        except Exception as e:
            logger.error(f"LSTM prediction failed for {symbol}: {e}")
            return self._get_statistical_predictions(symbol, data, horizons)

    def _get_prophet_predictions(self, symbol: str, data: pd.DataFrame,
                                 horizons: List[str]) -> Dict[str, float]:
        """Get Prophet model predictions"""

        if not PROPHET_AVAILABLE:
            return {}

        try:
            # Load model if not in memory
            if symbol not in self.prophet_models:
                self._load_prophet_model(symbol)

            if symbol not in self.prophet_models:
                return {}

            model = self.prophet_models[symbol]

            # Create future dataframe
            future_periods = max(self.prediction_horizons[h] for h in horizons if h in ['1D', '1W'])
            future = model.make_future_dataframe(periods=future_periods, freq='H')

            # Add volume regressor if model uses it
            if 'volume' in model.extra_regressors and 'volume' in data.columns:
                # Use last known volume for future periods
                last_volume = data['volume'].iloc[-1]
                future['volume'] = data['volume'].reindex(future.index, method='ffill').fillna(last_volume)

            # Make predictions
            forecast = model.predict(future)

            # Extract predictions for specific horizons
            prophet_predictions = {}
            current_time = pd.Timestamp.now()

            for horizon in horizons:
                if horizon in ['1D', '1W']:  # Prophet is good for longer-term
                    hours_ahead = self.prediction_horizons[horizon]
                    target_time = current_time + pd.Timedelta(hours=hours_ahead)

                    # Find closest forecast point
                    time_diffs = abs(forecast['ds'] - target_time)
                    closest_idx = time_diffs.idxmin()

                    predicted_value = forecast.loc[closest_idx, 'yhat']
                    prophet_predictions[horizon] = float(predicted_value)

            return prophet_predictions

        except Exception as e:
            logger.error(f"Prophet prediction failed for {symbol}: {e}")
            return {}

    def _create_ensemble_predictions(self, lstm_preds: Dict[str, float],
                                     prophet_preds: Dict[str, float],
                                     horizons: List[str]) -> Dict[str, float]:
        """Create ensemble predictions combining LSTM and Prophet"""

        ensemble_predictions = {}

        # Define weights based on horizon (LSTM better for short-term, Prophet for long-term)
        horizon_weights = {
            '1H': {'lstm': 0.8, 'prophet': 0.2},
            '4H': {'lstm': 0.7, 'prophet': 0.3},
            '1D': {'lstm': 0.4, 'prophet': 0.6},
            '1W': {'lstm': 0.2, 'prophet': 0.8}
        }

        for horizon in horizons:
            lstm_pred = lstm_preds.get(horizon)
            prophet_pred = prophet_preds.get(horizon)

            if lstm_pred is not None and prophet_pred is not None:
                # Weighted ensemble
                weights = horizon_weights.get(horizon, {'lstm': 0.5, 'prophet': 0.5})
                ensemble_pred = (weights['lstm'] * lstm_pred +
                                 weights['prophet'] * prophet_pred)
                ensemble_predictions[horizon] = float(ensemble_pred)

            elif lstm_pred is not None:
                ensemble_predictions[horizon] = lstm_pred

            elif prophet_pred is not None:
                ensemble_predictions[horizon] = prophet_pred

        return ensemble_predictions

    def _calculate_confidence_scores(self, predictions: Dict[str, Any],
                                     current_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate confidence scores for predictions based on market conditions"""

        confidence_scores = {}

        try:
            current_price = predictions['current_price']

            # Market volatility assessment
            if len(current_data) >= 20:
                recent_returns = current_data['close'].pct_change().tail(20)
                volatility = recent_returns.std()
                volatility_confidence = max(0.1, 1.0 - (volatility * 10))
            else:
                volatility_confidence = 0.5

            # Trend consistency
            if len(current_data) >= 20:
                recent_prices = current_data['close'].tail(20)
                trend_consistency = abs(np.corrcoef(range(20), recent_prices)[0, 1])
                trend_confidence = max(0.1, trend_consistency)
            else:
                trend_confidence = 0.5

            # Volume confirmation
            if 'volume' in current_data.columns and len(current_data) >= 20:
                volume_sma = current_data['volume'].tail(20).mean()
                current_volume = current_data['volume'].iloc[-1]
                volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1
                volume_confidence = min(1.0, max(0.1, volume_ratio / 2))
            else:
                volume_confidence = 0.5

            # Model agreement
            lstm_preds = predictions.get('lstm_predictions', {})
            prophet_preds = predictions.get('prophet_predictions', {})

            for horizon in lstm_preds.keys():
                base_confidence = np.mean([volatility_confidence, trend_confidence, volume_confidence])

                if horizon in prophet_preds:
                    lstm_val = lstm_preds[horizon]
                    prophet_val = prophet_preds[horizon]

                    # Calculate agreement
                    price_diff = abs(lstm_val - prophet_val) / current_price
                    agreement_confidence = max(0.1, 1.0 - (price_diff * 20))

                    combined_confidence = np.mean([base_confidence, agreement_confidence])
                    confidence_scores[horizon] = float(combined_confidence)
                else:
                    # Single model confidence (reduced)
                    confidence_scores[horizon] = base_confidence * 0.8

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            # Fallback confidence scores
            for horizon in predictions.get('ensemble_predictions', {}):
                confidence_scores[horizon] = 0.5

        return confidence_scores

    def _load_lstm_model(self, symbol: str) -> bool:
        """Load LSTM model from disk"""

        if not TORCH_AVAILABLE or LSTMPricePredictor is None:
            return False

        try:
            model_path = self.models_dir / f"lstm_{symbol}.pth"
            scaler_path = self.models_dir / f"scalers_{symbol}.joblib"

            if not model_path.exists() or not scaler_path.exists():
                return False

            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            model_config = checkpoint['model_config']

            model = LSTMPricePredictor(**model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Load scalers
            scaler_data = joblib.load(scaler_path)
            self.price_scaler = scaler_data['price_scaler']
            self.feature_scaler = scaler_data['feature_scaler']

            self.lstm_models[symbol] = model

            logger.info(f"LSTM model loaded for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LSTM model for {symbol}: {e}")
            return False

    def _load_prophet_model(self, symbol: str) -> bool:
        """Load Prophet model from disk"""

        if not PROPHET_AVAILABLE:
            return False

        try:
            model_path = self.models_dir / f"prophet_{symbol}.joblib"

            if not model_path.exists():
                return False

            model = joblib.load(model_path)
            self.prophet_models[symbol] = model

            logger.info(f"Prophet model loaded for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Prophet model for {symbol}: {e}")
            return False

    def _log_predictions(self, predictions: Dict[str, Any]):
        """Log predictions to database for performance tracking"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract data
            symbol = predictions['symbol']
            current_price = predictions['current_price']
            lstm_preds = predictions.get('lstm_predictions', {})
            prophet_preds = predictions.get('prophet_predictions', {})
            ensemble_preds = predictions.get('ensemble_predictions', {})
            confidence_scores = predictions.get('confidence_scores', {})

            # Insert prediction record
            prediction_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            cursor.execute("""
                INSERT INTO price_predictions (
                    prediction_id, symbol, timeframe, current_price,
                    lstm_1h_price, lstm_4h_price, lstm_1d_price,
                    prophet_1d_price, prophet_1w_price,
                    ensemble_1h_price, ensemble_4h_price, 
                    ensemble_1d_price, ensemble_1w_price,
                    ensemble_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id, symbol, 'MULTI', current_price,
                lstm_preds.get('1H'), lstm_preds.get('4H'), lstm_preds.get('1D'),
                prophet_preds.get('1D'), prophet_preds.get('1W'),
                ensemble_preds.get('1H'), ensemble_preds.get('4H'),
                ensemble_preds.get('1D'), ensemble_preds.get('1W'),
                confidence_scores.get('1D', 0.5)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log predictions: {e}")

    def _create_neutral_prediction(self, symbol: str, reason: str) -> Dict:
        """Create neutral prediction when normal processing fails"""

        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signal': 'HOLD',
            'confidence': 0.5,
            'predicted_price': None,
            'current_price': None,
            'price_change': 0.0,
            'price_change_pct': 0.0,
            'horizon': 1,
            'model': 'lstm_forecaster',
            'method': 'fallback',
            'reason': reason,
            'data_points': 0
        }


# Example usage
if __name__ == "__main__":
    """Example usage of Time-Series Forecaster"""

    # Initialize forecaster
    forecaster = TimeSeriesForecaster()

    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 2500,
        'high': np.random.randn(len(dates)).cumsum() + 2520,
        'low': np.random.randn(len(dates)).cumsum() + 2480,
        'close': np.random.randn(len(dates)).cumsum() + 2500,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }).set_index('timestamp')

    print("Time-Series Forecaster Demo")
    print("=" * 50)

    symbol = "RELIANCE"

    # Generate predictions
    print(f"\nGenerating predictions for {symbol}...")

    # Use last 100 periods for prediction
    recent_data = sample_data.tail(100)
    predictions = forecaster.predict_prices(symbol, recent_data)

    print(f"\nPrediction Results:")
    print(f"Current Price: ₹{predictions['current_price']:.2f}")

    if predictions['ensemble_predictions']:
        print("\nEnsemble Predictions:")
        for horizon, price in predictions['ensemble_predictions'].items():
            confidence = predictions['confidence_scores'].get(horizon, 0.5)
            print(f"  {horizon:>3}: ₹{price:.2f} (Confidence: {confidence:.1%})")

    # Test the prediction interface
    print(f"\nTesting ML Signal Enhancer interface...")
    ml_prediction = forecaster.predict(symbol, recent_data)
    print(f"Signal: {ml_prediction['signal']}")
    print(f"Confidence: {ml_prediction['confidence']:.2%}")
    print(f"Price Change: {ml_prediction['price_change_pct']:.2%}")

    print(f"\nTime-Series Forecasting demo complete!")