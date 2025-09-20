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

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Prophet for seasonal forecasting
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='prophet')

# Technical Analysis
import pandas_ta as ta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.db_path = "marketpulse_production.db"
        self._init_database()

        logger.info("✅ Time-Series Forecaster initialized")

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

        logger.info("✅ Forecasting database tables initialized")

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

        # Basic technical indicators
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_21'] = ta.rsi(df['close'], length=21)

        # MACD
        # MACD with robust column detection
        try:
            macd_data = ta.macd(df['close'])
            if isinstance(macd_data, pd.DataFrame) and not macd_data.empty:
                macd_cols = macd_data.columns.tolist()

                # Find MACD columns
                macd_col = next(
                    (col for col in macd_cols if 'MACD' in col and 'MACDs' not in col and 'MACDh' not in col), None)
                signal_col = next((col for col in macd_cols if 'MACDs' in col), None)
                hist_col = next((col for col in macd_cols if 'MACDh' in col), None)

                if macd_col:
                    df['macd'] = macd_data[macd_col]
                if signal_col:
                    df['macd_signal'] = macd_data[signal_col]
                if hist_col:
                    df['macd_histogram'] = macd_data[hist_col]
            else:
                raise ValueError("MACD data empty")
        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}, using manual calculation")
            # Manual MACD calculation
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        # Bollinger Bands with robust column detection
        try:
            bb_data = ta.bbands(df['close'], length=20)
            if isinstance(bb_data, pd.DataFrame) and not bb_data.empty:
                bb_cols = bb_data.columns.tolist()

                # Find Bollinger Band columns (handle different naming conventions)
                upper_col = next((col for col in bb_cols if 'BBU' in col or 'upper' in col.lower()), None)
                lower_col = next((col for col in bb_cols if 'BBL' in col or 'lower' in col.lower()), None)
                middle_col = next((col for col in bb_cols if 'BBM' in col or 'middle' in col.lower()), None)

                if upper_col and lower_col:
                    df['bb_upper'] = bb_data[upper_col]
                    df['bb_lower'] = bb_data[lower_col]

                    if middle_col:
                        df['bb_middle'] = bb_data[middle_col]
                    else:
                        df['bb_middle'] = (df['bb_upper'] + df['bb_lower']) / 2

                    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                else:
                    raise ValueError("Bollinger Bands columns not found")
            else:
                raise ValueError("Bollinger Bands data empty")
        except Exception as e:
            logger.warning(f"Bollinger Bands calculation failed: {e}, using manual calculation")
            # Manual calculation as fallback
            sma20 = ta.sma(df['close'], length=20)
            if sma20 is None:
                sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['bb_upper'] = sma20 + (2 * std20)
            df['bb_lower'] = sma20 - (2 * std20)
            df['bb_middle'] = sma20
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
        # ADX with robust column detection
        try:
            adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
            if isinstance(adx_data, pd.DataFrame) and not adx_data.empty:
                adx_cols = adx_data.columns.tolist()
                adx_col = next((col for col in adx_cols if 'ADX' in col), None)

                if adx_col:
                    df['adx_14'] = adx_data[adx_col]
                else:
                    df['adx_14'] = 50  # Default neutral value
            else:
                df['adx_14'] = 50
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}, using default value")
            df['adx_14'] = 50  # Default neutral ADX value

        # Momentum indicators
        # Stochastic with robust column detection
        try:
            stoch_data = ta.stoch(df['high'], df['low'], df['close'])
            if isinstance(stoch_data, pd.DataFrame) and not stoch_data.empty:
                stoch_cols = stoch_data.columns.tolist()

                # Find Stochastic columns
                k_col = next((col for col in stoch_cols if 'STOCHk' in col), None)
                d_col = next((col for col in stoch_cols if 'STOCHd' in col), None)

                if k_col:
                    df['stoch_k'] = stoch_data[k_col]
                if d_col:
                    df['stoch_d'] = stoch_data[d_col]
            else:
                raise ValueError("Stochastic data empty")
        except Exception as e:
            logger.warning(f"Stochastic calculation failed: {e}, using manual calculation")
            # Manual Stochastic calculation
            lowest_low = df['low'].rolling(14).min()
            highest_high = df['high'].rolling(14).max()
            df['stoch_k'] = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'])
        df['cci_14'] = ta.cci(df['high'], df['low'], df['close'], length=14)
        df['roc_10'] = ta.roc(df['close'], length=10)

        # Volume indicators
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price patterns
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']

        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)

        return df.fillna(method='bfill').fillna(method='ffill')

    def train_lstm_model(self, symbol: str, df: pd.DataFrame, epochs: int = 50) -> bool:
        """
        Train LSTM model for a specific symbol

        Args:
            symbol: Stock symbol
            df: Historical OHLCV data
            epochs: Number of training epochs

        Returns:
            bool: True if training successful
        """

        try:
            logger.info(f"🤖 Training LSTM model for {symbol}...")

            # Prepare data
            X, y, feature_names = self.prepare_lstm_data(df)

            if len(X) < 100:
                logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
                return False

            # Split data (80% train, 20% validation)
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)

            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Initialize model
            input_features = X.shape[2]
            model = LSTMPricePredictor(input_features=input_features)

            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop
            model.train()
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(epochs):
                epoch_loss = 0
                batch_count = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()

                    # Forward pass
                    predictions = model(batch_X).squeeze()
                    loss = criterion(predictions, batch_y)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                # Validation
                model.eval()
                with torch.no_grad():
                    val_predictions = model(X_val_tensor).squeeze()
                    val_loss = criterion(val_predictions, y_val_tensor).item()

                model.train()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= 10:  # Early stopping
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    avg_loss = epoch_loss / batch_count
                    logger.info(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Load best model
            model.load_state_dict(best_model_state)
            model.eval()

            # Save model
            model_data = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_features': input_features,
                    'sequence_length': self.sequence_length,
                    'hidden_size': 50,
                    'num_layers': 2
                },
                'feature_names': feature_names,
                'symbol': symbol,
                'training_samples': len(X_train),
                'val_loss': best_val_loss
            }

            model_path = self.models_dir / f"lstm_{symbol}.pth"
            torch.save(model_data, model_path)

            # Save scalers
            scaler_data = {
                'price_scaler': self.price_scaler,
                'feature_scaler': self.feature_scaler
            }
            scaler_path = self.models_dir / f"scalers_{symbol}.joblib"
            joblib.dump(scaler_data, scaler_path)

            # Store in memory
            self.lstm_models[symbol] = model

            logger.info(f"✅ LSTM model trained for {symbol} - Val Loss: {best_val_loss:.6f}")
            return True

        except Exception as e:
            logger.error(f"❌ LSTM training failed for {symbol}: {e}")
            return False

    def train_prophet_model(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Train Prophet model for long-term forecasting

        Args:
            symbol: Stock symbol
            df: Historical daily OHLCV data

        Returns:
            bool: True if training successful
        """

        try:
            logger.info(f"📈 Training Prophet model for {symbol}...")

            # Prepare data for Prophet
            prophet_df = df.reset_index()
            prophet_df = prophet_df.rename(columns={'timestamp': 'ds', 'close': 'y'})

            # Select recent data (Prophet works best with daily data)
            if len(prophet_df) > 365:
                prophet_df = prophet_df.tail(365)  # Last year of data

            if len(prophet_df) < 30:
                logger.warning(f"Insufficient data for Prophet {symbol}: {len(prophet_df)} days")
                return False

            # Configure Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=len(prophet_df) > 365,
                changepoint_prior_scale=0.05,  # Less flexible for financial data
                seasonality_prior_scale=0.05,
                holidays_prior_scale=0.05,
                seasonality_mode='multiplicative',
                interval_width=0.95
            )

            # Add volume as regressor if available
            if 'volume' in df.columns:
                model.add_regressor('volume')
                prophet_df['volume'] = df['volume'].values

            # Fit model
            model.fit(prophet_df)

            # Save model
            model_path = self.models_dir / f"prophet_{symbol}.joblib"
            joblib.dump(model, model_path)

            # Store in memory
            self.prophet_models[symbol] = model

            logger.info(f"✅ Prophet model trained for {symbol}")
            return True

        except Exception as e:
            logger.error(f"❌ Prophet training failed for {symbol}: {e}")
            return False

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
        if symbol in self.lstm_models and len(current_data) >= self.sequence_length:
            predictions['lstm_predictions'] = self._get_lstm_predictions(
                symbol, current_data, horizons
            )

        # Prophet Predictions (for longer-term horizons)
        if symbol in self.prophet_models:
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

    def _get_lstm_predictions(self, symbol: str, data: pd.DataFrame,
                              horizons: List[str]) -> Dict[str, float]:
        """Get LSTM model predictions"""

        try:
            # Load model if not in memory
            if symbol not in self.lstm_models:
                self._load_lstm_model(symbol)

            model = self.lstm_models[symbol]

            # Prepare input sequence
            X, _, _ = self.prepare_lstm_data(data.tail(self.sequence_length + 10))

            if len(X) == 0:
                return {}

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

                # For different horizons, we can apply scaling factors
                # This is a simplification - in practice, you'd train separate models
                current_price = data['close'].iloc[-1]

                for horizon in horizons:
                    if horizon in ['1H', '4H', '1D']:  # LSTM is good for short-term
                        # Apply volatility-adjusted scaling
                        atr = data.get('atr_14', pd.Series([current_price * 0.02])).iloc[-1]
                        volatility_factor = atr / current_price

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
            return {}

    def _get_prophet_predictions(self, symbol: str, data: pd.DataFrame,
                                 horizons: List[str]) -> Dict[str, float]:
        """Get Prophet model predictions"""

        try:
            # Load model if not in memory
            if symbol not in self.prophet_models:
                self._load_prophet_model(symbol)

            model = self.prophet_models[symbol]

            # Create future dataframe
            future_periods = max(self.prediction_horizons[h] for h in horizons if h in ['1D', '1W'])
            future = model.make_future_dataframe(periods=future_periods, freq='H')

            # Add volume regressor if model uses it
            if 'volume' in model.extra_regressors:
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
            # Market volatility assessment
            if 'atr_14' in current_data.columns:
                current_price = current_data['close'].iloc[-1]
                atr = current_data['atr_14'].iloc[-1]
                volatility = atr / current_price

                # Lower volatility = higher confidence
                volatility_confidence = max(0.1, 1.0 - (volatility * 10))
            else:
                volatility_confidence = 0.5

            # Trend consistency
            if len(current_data) >= 20:
                recent_prices = current_data['close'].tail(20)
                price_trend = np.polyfit(range(20), recent_prices, 1)[0]
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

            # Model agreement (if both models agree, higher confidence)
            lstm_preds = predictions.get('lstm_predictions', {})
            prophet_preds = predictions.get('prophet_predictions', {})

            for horizon in lstm_preds.keys():
                if horizon in prophet_preds:
                    lstm_val = lstm_preds[horizon]
                    prophet_val = prophet_preds[horizon]
                    current_price = predictions['current_price']

                    # Calculate agreement (lower difference = higher confidence)
                    price_diff = abs(lstm_val - prophet_val) / current_price
                    agreement_confidence = max(0.1, 1.0 - (price_diff * 20))

                    # Combined confidence score
                    combined_confidence = np.mean([
                        volatility_confidence,
                        trend_confidence,
                        volume_confidence,
                        agreement_confidence
                    ])

                    confidence_scores[horizon] = float(combined_confidence)
                else:
                    # Single model confidence (lower)
                    confidence_scores[horizon] = np.mean([
                        volatility_confidence,
                        trend_confidence,
                        volume_confidence
                    ]) * 0.8  # Reduce for single model

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            # Fallback confidence scores
            for horizon in predictions.get('ensemble_predictions', {}):
                confidence_scores[horizon] = 0.5

        return confidence_scores

    def _load_lstm_model(self, symbol: str) -> bool:
        """Load LSTM model from disk"""

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

            logger.info(f"✅ LSTM model loaded for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to load LSTM model for {symbol}: {e}")
            return False

    def _load_prophet_model(self, symbol: str) -> bool:
        """Load Prophet model from disk"""

        try:
            model_path = self.models_dir / f"prophet_{symbol}.joblib"

            if not model_path.exists():
                return False

            model = joblib.load(model_path)
            self.prophet_models[symbol] = model

            logger.info(f"✅ Prophet model loaded for {symbol}")
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

    def evaluate_model_performance(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Evaluate historical prediction accuracy

        Args:
            symbol: Stock symbol to evaluate
            days_back: Number of days to look back for evaluation

        Returns:
            Dictionary with performance metrics
        """

        try:
            conn = sqlite3.connect(self.db_path)

            # Get predictions from the past
            cutoff_date = datetime.now() - timedelta(days=days_back)

            predictions_df = pd.read_sql_query("""
                SELECT * FROM price_predictions 
                WHERE symbol = ? AND prediction_timestamp >= ?
                ORDER BY prediction_timestamp
            """, conn, params=(symbol, cutoff_date))

            conn.close()

            if predictions_df.empty:
                return {'error': 'No historical predictions found'}

            # Get actual price data for evaluation
            # This would need to be integrated with your data collection system
            # For now, return structure for future implementation

            performance_metrics = {
                'symbol': symbol,
                'evaluation_period_days': days_back,
                'total_predictions': len(predictions_df),
                'lstm_performance': {
                    '1H': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0},
                    '4H': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0},
                    '1D': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0}
                },
                'prophet_performance': {
                    '1D': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0},
                    '1W': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0}
                },
                'ensemble_performance': {
                    '1H': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0},
                    '4H': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0},
                    '1D': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0},
                    '1W': {'mae': 0, 'rmse': 0, 'mape': 0, 'directional_accuracy': 0}
                }
            }

            logger.info(f"📊 Performance evaluation completed for {symbol}")
            return performance_metrics

        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {'error': str(e)}

    def retrain_models(self, symbol: str, df: pd.DataFrame) -> Dict[str, bool]:
        """Retrain both LSTM and Prophet models with new data"""

        results = {
            'lstm_success': False,
            'prophet_success': False
        }

        # Retrain LSTM
        if len(df) >= 200:  # Need sufficient data for LSTM
            results['lstm_success'] = self.train_lstm_model(symbol, df)

        # Retrain Prophet
        if len(df) >= 30:  # Need at least 30 days for Prophet
            results['prophet_success'] = self.train_prophet_model(symbol, df)

        logger.info(
            f"🔄 Model retraining for {symbol}: LSTM={results['lstm_success']}, Prophet={results['prophet_success']}")

        return results


# Example usage
if __name__ == "__main__":
    """Example usage of Time-Series Forecaster"""

    # Initialize forecaster
    forecaster = TimeSeriesForecaster()

    # Create sample data (normally from your data collection system)
    dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 2500,
        'high': np.random.randn(len(dates)).cumsum() + 2520,
        'low': np.random.randn(len(dates)).cumsum() + 2480,
        'close': np.random.randn(len(dates)).cumsum() + 2500,
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }).set_index('timestamp')

    print("🚀 Time-Series Forecaster Demo")
    print("=" * 50)

    symbol = "RELIANCE"

    # Train models
    print(f"\n📚 Training models for {symbol}...")
    lstm_success = forecaster.train_lstm_model(symbol, sample_data)
    prophet_success = forecaster.train_prophet_model(symbol, sample_data)

    print(f"LSTM Training: {'✅ Success' if lstm_success else '❌ Failed'}")
    print(f"Prophet Training: {'✅ Success' if prophet_success else '❌ Failed'}")

    # Generate predictions
    if lstm_success or prophet_success:
        print(f"\n🔮 Generating predictions for {symbol}...")

        # Use last 100 periods for prediction
        recent_data = sample_data.tail(100)
        predictions = forecaster.predict_prices(symbol, recent_data)

        print(f"\n📊 Prediction Results:")
        print(f"Current Price: ₹{predictions['current_price']:.2f}")

        if predictions['ensemble_predictions']:
            print("\nEnsemble Predictions:")
            for horizon, price in predictions['ensemble_predictions'].items():
                confidence = predictions['confidence_scores'].get(horizon, 0.5)
                print(f"  {horizon:>3}: ₹{price:.2f} (Confidence: {confidence:.1%})")

        if predictions['lstm_predictions']:
            print("\nLSTM Predictions:")
            for horizon, price in predictions['lstm_predictions'].items():
                print(f"  {horizon:>3}: ₹{price:.2f}")

        if predictions['prophet_predictions']:
            print("\nProphet Predictions:")
            for horizon, price in predictions['prophet_predictions'].items():
                print(f"  {horizon:>3}: ₹{price:.2f}")

    print(f"\n🎉 Time-Series Forecasting demo complete!")
    print(f"📝 Next steps:")
    print(f"   1. Integrate with real market data")
    print(f"   2. Set up automated retraining schedule")
    print(f"   3. Implement prediction performance tracking")
    print(f"   4. Add more sophisticated ensemble methods")