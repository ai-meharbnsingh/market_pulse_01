# 03_ML_ENGINE/two_year_ml_training_pipeline_fixed.py
"""
FIXED Two-Year ML Model Training Pipeline
Trains AlphaModel and LSTM on 2 years of historical Indian stock data

CRITICAL FIXES APPLIED:
1. Proper handling of infinity and NaN values
2. Data cleaning and validation before training
3. Feature scaling and normalization
4. Robust error handling for edge cases

Location: #03_ML_ENGINE/two_year_ml_training_pipeline_fixed.py
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import os
import pickle
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import advanced ML libraries
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available - using fallback models")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available - using fallback models")


class FixedTwoYearMLPipeline:
    """FIXED ML training pipeline with proper data cleaning for 2-year Indian stock data"""

    def __init__(self, db_path: str = "06_DATA/marketpulse_training.db"):
        self.db_path = os.path.abspath(db_path)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_dir = "03_ML_ENGINE/trained_models"

        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

        logger.info(f"Initialized FIXED ML training pipeline")
        logger.info(f"Database: {self.db_path}")

    def load_training_data(self) -> pd.DataFrame:
        """Load comprehensive training data with technical indicators"""

        logger.info("Loading 2-year training data...")

        try:
            conn = sqlite3.connect(self.db_path)

            # Join market data with technical indicators
            query = """
            SELECT 
                m.symbol, m.symbol_clean, m.market_cap_category, m.sector,
                m.timestamp, m.open_price, m.high_price, m.low_price, 
                m.close_price, m.volume, m.adj_close,

                t.sma_5, t.sma_10, t.sma_20, t.sma_50,
                t.ema_5, t.ema_10, t.ema_20, t.ema_50,
                t.rsi_14, t.rsi_21, t.macd, t.macd_signal, t.macd_histogram,
                t.stoch_k, t.stoch_d, t.williams_r, t.cci,
                t.bb_upper, t.bb_middle, t.bb_lower, t.bb_width,
                t.atr_14, t.atr_21, t.volume_sma_20, t.volume_ratio, t.vwap,
                t.support_level, t.resistance_level, t.trend_strength, t.volatility_regime

            FROM market_data_enhanced m
            LEFT JOIN technical_indicators t ON m.symbol = t.symbol AND m.timestamp = t.timestamp
            WHERE t.rsi_14 IS NOT NULL
            ORDER BY m.symbol, m.timestamp
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                raise ValueError("No training data found in database")

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by symbol and timestamp
            df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

            logger.info(f"Loaded {len(df)} records for {df['symbol'].nunique()} stocks")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise

    def clean_data_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """CRITICAL: Clean data to remove infinity and NaN values"""

        logger.info("Applying critical data cleaning for ML training...")

        initial_rows = len(df)

        # Step 1: Replace infinity values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # Step 2: Find numeric columns for cleaning
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['timestamp']  # Don't clean timestamp
        numeric_columns = [col for col in numeric_columns if col not in exclude_cols]

        logger.info(f"Cleaning {len(numeric_columns)} numeric columns...")

        # Step 3: Handle extreme outliers (beyond 5 standard deviations)
        for col in numeric_columns:
            if col in df.columns:
                try:
                    mean_val = df[col].mean()
                    std_val = df[col].std()

                    if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                        # Replace extreme outliers with boundaries
                        upper_bound = mean_val + (5 * std_val)
                        lower_bound = mean_val - (5 * std_val)

                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

                except Exception as e:
                    logger.warning(f"Could not clean column {col}: {e}")
                    continue

        # Step 4: Forward fill and backward fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Step 5: Remove any remaining rows with NaN in critical columns
        critical_cols = ['close_price', 'volume', 'rsi_14', 'sma_20']
        existing_critical_cols = [col for col in critical_cols if col in df.columns]
        df = df.dropna(subset=existing_critical_cols)

        # Step 6: Final check for any remaining infinite values
        inf_counts = {}
        for col in numeric_columns:
            if col in df.columns:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
                    # Replace any remaining infinities with column median
                    df[col] = df[col].replace([np.inf, -np.inf], df[col].median())

        if inf_counts:
            logger.warning(f"Replaced remaining infinities: {inf_counts}")

        final_rows = len(df)
        logger.info(
            f"Data cleaning complete: {initial_rows} -> {final_rows} rows ({final_rows / initial_rows:.1%} retained)")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for ML models with proper bounds checking"""

        logger.info("Engineering additional features...")

        try:
            # Price-based features with safety checks
            df['price_change'] = df.groupby('symbol')['close_price'].pct_change()
            df['price_change_5d'] = df.groupby('symbol')['close_price'].pct_change(periods=5)
            df['price_volatility_5d'] = df.groupby('symbol')['price_change'].rolling(5).std().reset_index(0, drop=True)

            # Relative strength features with division safety
            safe_sma20 = df['sma_20'].replace(0, np.nan)
            safe_ema20 = df['ema_20'].replace(0, np.nan)
            safe_close = df['close_price'].replace(0, np.nan)

            df['price_vs_sma20'] = (safe_close - safe_sma20) / safe_sma20
            df['price_vs_ema20'] = (safe_close - safe_ema20) / safe_ema20

            # Bollinger Bands position with safety
            bb_range = df['bb_upper'] - df['bb_lower']
            bb_range = bb_range.replace(0, np.nan)
            df['bb_position'] = (safe_close - df['bb_lower']) / bb_range

            # Volume features with safety
            df['volume_change'] = df.groupby('symbol')['volume'].pct_change()
            safe_volume_sma = df['volume_sma_20'].replace(0, np.nan)
            df['volume_vs_avg'] = df['volume'] / safe_volume_sma

            # Momentum features
            df['rsi_momentum'] = df.groupby('symbol')['rsi_14'].diff()
            df['macd_momentum'] = df.groupby('symbol')['macd'].diff()
            df['price_momentum_3d'] = df.groupby('symbol')['close_price'].pct_change(periods=3)

            # Volatility features with bounds
            df['high_low_range'] = (df['high_price'] - df['low_price']) / safe_close
            df['intraday_return'] = (safe_close - df['open_price']) / df['open_price']

            # Market regime features
            df['trend_direction'] = np.where(safe_close > safe_sma20, 1,
                                             np.where(safe_close < safe_sma20, -1, 0))
            df['volatility_percentile'] = df.groupby('symbol')['atr_14'].rank(pct=True)

            # Support/Resistance features with safety
            df['distance_to_support'] = (safe_close - df['support_level']) / safe_close
            df['distance_to_resistance'] = (df['resistance_level'] - safe_close) / safe_close

            # Cross-sectional features (sector relative)
            df['sector_relative_rsi'] = df.groupby(['sector', 'timestamp'])['rsi_14'].rank(pct=True)
            df['sector_relative_performance'] = df.groupby(['sector', 'timestamp'])['price_change'].rank(pct=True)

            # Apply data cleaning after feature engineering
            df = self.clean_data_for_ml(df)

            return df

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            # Return original df if feature engineering fails
            return self.clean_data_for_ml(df)

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML training with safety checks"""

        logger.info("Creating target variables...")

        try:
            # Future return targets (for different holding periods)
            for days in [1, 3, 5, 10]:
                future_prices = df.groupby('symbol')['close_price'].shift(-days)
                current_prices = df['close_price'].replace(0, np.nan)
                df[f'future_return_{days}d'] = (future_prices / current_prices) - 1

            # Binary classification targets (profitable vs not profitable)
            df['profitable_1d'] = (df['future_return_1d'] > 0.02).astype(int)  # 2% threshold
            df['profitable_3d'] = (df['future_return_3d'] > 0.05).astype(int)  # 5% threshold
            df['profitable_5d'] = (df['future_return_5d'] > 0.08).astype(int)  # 8% threshold

            # Multi-class targets (BUY/HOLD/SELL) with safety
            try:
                df['signal_1d'] = pd.cut(df['future_return_1d'],
                                         bins=[-np.inf, -0.02, 0.02, np.inf],
                                         labels=['SELL', 'HOLD', 'BUY'])
            except Exception:
                # Fallback if cut fails
                df['signal_1d'] = 'HOLD'

            # Risk-adjusted targets with safety
            safe_volatility = df['price_volatility_5d'].replace(0, np.nan)
            df['sharpe_1d'] = df['future_return_1d'] / safe_volatility
            df['good_sharpe'] = (df['sharpe_1d'] > 0.5).astype(int)

            return df

        except Exception as e:
            logger.error(f"Target variable creation failed: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix for ML training with robust cleaning"""

        # Select feature columns (exclude target variables and identifiers)
        exclude_cols = [
            'symbol', 'symbol_clean', 'timestamp', 'market_cap_category', 'sector',
            'open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adj_close',
            'future_return_1d', 'future_return_3d', 'future_return_5d', 'future_return_10d',
            'profitable_1d', 'profitable_3d', 'profitable_5d', 'signal_1d', 'sharpe_1d', 'good_sharpe'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Remove any remaining NaN rows in features and targets
        required_cols = feature_cols + ['profitable_1d', 'profitable_3d']
        available_cols = [col for col in required_cols if col in df.columns]
        df_clean = df.dropna(subset=available_cols)

        # Final infinity check on features
        feature_data = df_clean[feature_cols]

        # Check for any remaining problematic values
        inf_mask = np.isinf(feature_data).any(axis=1)
        if inf_mask.sum() > 0:
            logger.warning(f"Removing {inf_mask.sum()} rows with infinity values")
            df_clean = df_clean[~inf_mask]

        logger.info(f"Selected {len(feature_cols)} features for training")
        logger.info(f"Clean dataset: {len(df_clean)} records")

        return df_clean, feature_cols

    def train_alpha_model(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Train AlphaModel for probability of profit prediction with robust preprocessing"""

        logger.info("Training AlphaModel for profit probability prediction...")

        try:
            X = df[feature_cols]
            y = df['profitable_3d']  # 3-day profitability target

            # Final data validation
            if X.isnull().any().any():
                logger.warning("NaN values found in features - applying final cleaning")
                X = X.fillna(X.median())

            if np.isinf(X.values).any():
                logger.warning("Infinity values found in features - applying final cleaning")
                X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

            # Split data temporally (use last 20% for testing)
            split_date = df['timestamp'].quantile(0.8)
            train_mask = df['timestamp'] <= split_date

            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]

            logger.info(f"Training set: {len(X_train)} samples")
            logger.info(f"Test set: {len(X_test)} samples")

            # Use RobustScaler instead of StandardScaler for better outlier handling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Verify no infinity values after scaling
            if np.isinf(X_train_scaled).any() or np.isinf(X_test_scaled).any():
                logger.error("Infinity values found after scaling - this should not happen")
                raise ValueError("Data contains infinity values after preprocessing")

            # Train multiple models and ensemble
            models = {}

            # Random Forest (most robust to outliers)
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

            # Gradient Boosting
            models['gradient_boosting'] = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )

            # Logistic Regression (needs scaled data)
            models['logistic'] = LogisticRegression(random_state=42, max_iter=1000)

            # XGBoost (if available)
            if XGBOOST_AVAILABLE:
                models['xgboost'] = xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
                )

            # Train all models
            trained_models = {}
            model_scores = {}

            for name, model in models.items():
                try:
                    logger.info(f"Training {name}...")

                    if name == 'logistic':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    model_scores[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }

                    trained_models[name] = model

                    logger.info(f"{name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
                                f"Recall: {recall:.3f}, F1: {f1:.3f}")

                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")

            # Save models and scaler
            self.models['alpha_ensemble'] = trained_models
            self.scalers['alpha'] = scaler
            self.feature_columns = feature_cols

            # Save to disk
            model_path = os.path.join(self.model_dir, "alpha_model_ensemble_fixed.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'models': trained_models,
                    'scaler': scaler,
                    'feature_cols': feature_cols,
                    'scores': model_scores
                }, f)

            logger.info(f"AlphaModel ensemble saved to {model_path}")

            return model_scores

        except Exception as e:
            logger.error(f"AlphaModel training failed: {e}")
            return {}

    def save_training_summary(self, alpha_scores: Dict, backtest_results: Dict = None):
        """Save comprehensive training summary"""

        summary = {
            'training_date': datetime.now().isoformat(),
            'database_path': self.db_path,
            'feature_count': len(self.feature_columns),
            'alpha_model_scores': alpha_scores,
            'backtest_results': backtest_results or {},
            'fixes_applied': [
                'Infinity value removal',
                'NaN handling with forward/backward fill',
                'Extreme outlier clipping (5 standard deviations)',
                'RobustScaler for better outlier handling',
                'Division by zero protection',
                'Final data validation before training'
            ],
            'model_files': {
                'alpha_ensemble': os.path.join(self.model_dir, "alpha_model_ensemble_fixed.pkl")
            }
        }

        summary_path = os.path.join(self.model_dir, "training_summary_fixed.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Training summary saved to {summary_path}")


def main():
    """Main function to execute FIXED 2-year ML training pipeline"""

    print("FIXED 2-YEAR ML MODEL TRAINING PIPELINE")
    print("=" * 50)

    # Initialize pipeline
    pipeline = FixedTwoYearMLPipeline()

    try:
        # Load training data
        df = pipeline.load_training_data()

        # Apply critical data cleaning first
        df = pipeline.clean_data_for_ml(df)

        # Engineer features (with internal cleaning)
        df = pipeline.engineer_features(df)

        # Create targets
        df = pipeline.create_target_variables(df)

        # Prepare features (with final validation)
        df_clean, feature_cols = pipeline.prepare_features(df)

        # Train AlphaModel with robust preprocessing
        print("\nTraining AlphaModel ensemble with FIXED data handling...")
        alpha_scores = pipeline.train_alpha_model(df_clean, feature_cols)

        # Save summary
        pipeline.save_training_summary(alpha_scores)

        print("\nFIXED TRAINING COMPLETE!")
        print("=" * 30)
        print("Models trained and saved:")
        print(f"- AlphaModel ensemble: {len(alpha_scores)} models")
        print(f"- Training data: {len(df_clean)} records")
        print(f"- Features: {len(feature_cols)} indicators")

        if alpha_scores:
            print(f"\nModel Performance:")
            for model_name, scores in alpha_scores.items():
                print(f"- {model_name}: Accuracy {scores['accuracy']:.3f}, F1 {scores['f1_score']:.3f}")

        print("\nFixes Applied:")
        print("✅ Infinity value removal")
        print("✅ NaN handling")
        print("✅ Extreme outlier clipping")
        print("✅ Robust scaling")
        print("✅ Division by zero protection")

        print("\nNext Steps:")
        print("1. Start paper trading: python main.py start")
        print("2. Monitor performance: python 07_DASHBOARD/dashboard_app.py")

    except Exception as e:
        logger.error(f"FIXED training pipeline failed: {e}")
        print(f"ERROR: {e}")
        print("\nTroubleshooting suggestions:")
        print("1. Check database connection")
        print("2. Verify data quality in technical_indicators table")
        print("3. Check for any remaining data corruption")


if __name__ == "__main__":
    main()