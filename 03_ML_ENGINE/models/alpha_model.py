# src/models/alpha_model/alpha_core.py
"""
Alpha Model Core Engine - Phase 1 Day 9 (Standalone Version)
Fixed version that works without antifragile_framework dependency

The heart of MarketPulse predictive intelligence - standalone implementation
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import json
import sqlite3

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaModelCore:
    """
    The Alpha Model - Core Predictive Engine for Trading Signal Profitability

    This model transforms MarketPulse from reactive analysis to predictive intelligence
    by learning which signal characteristics lead to profitable trades.

    The Four-Stage Learning Loop:
    1. Hypothesis Generation: Log all trading signals with rich features
    2. Experimentation: Track trade execution and lifecycle
    3. Result Analysis: Record outcomes with detailed metrics
    4. Learning & Refinement: Train ML models to predict profitability
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize Alpha Model with ensemble ML approach"""

        # Core configuration
        self.model_version = "1.0.0"
        self.feature_version = "1.0.0"

        # File paths
        self.models_dir = Path("models/alpha_model")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Database connection for trade hypotheses
        self.db_path = "marketpulse_production.db"
        self._init_database()

        # ML Models - Ensemble Approach
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False

        # Feature configuration
        self.feature_columns = [
            # Signal Features
            'confluence_score', 'ai_confidence', 'pattern_type_encoded',
            'entry_price', 'predicted_target', 'predicted_stop_loss',
            'risk_reward_ratio', 'position_size_pct',

            # Technical Indicators
            'rsi_14', 'macd_signal', 'bb_position', 'volume_ratio',
            'atr_14', 'adx_14', 'stoch_k', 'williams_r',

            # Market Context
            'market_regime_encoded', 'volatility_percentile', 'sector_momentum',
            'time_of_day', 'day_of_week', 'intraday_position',

            # AI Provider Context
            'ai_provider_encoded', 'ai_response_time', 'ai_cost_usd'
        ]

        # Target outcomes
        self.target_column = 'is_profitable'

        # Simple AI simulation (replacing antifragile_framework)
        self.ai_engine = None  # Will be simulated

        logger.info("Alpha Model Core initialized - Ready for predictive intelligence")

    def _init_database(self):
        """Initialize database tables for trade hypothesis tracking"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trade hypotheses table - logs every signal before execution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_hypotheses (
                signal_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL', 'HOLD')),

                -- Price and Risk Data
                entry_price REAL NOT NULL,
                predicted_target REAL,
                predicted_stop_loss REAL,
                risk_reward_ratio REAL,
                position_size_pct REAL,

                -- Signal Characteristics
                signal_source TEXT NOT NULL,
                confluence_score REAL,
                ai_confidence REAL,
                pattern_type TEXT,

                -- Technical Indicators (Key Features)
                rsi_14 REAL,
                macd_signal REAL,
                bb_position REAL,
                volume_ratio REAL,
                atr_14 REAL,
                adx_14 REAL,
                stoch_k REAL,
                williams_r REAL,

                -- Market Context
                market_regime TEXT,
                volatility_percentile REAL,
                sector_momentum REAL,
                vix_level REAL,

                -- Timing Features
                time_of_day INTEGER,
                day_of_week INTEGER,
                intraday_position REAL,

                -- AI Provider Context
                ai_provider_used TEXT,
                ai_response_time_ms REAL,
                ai_cost_usd REAL,

                -- Status
                status TEXT DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'EXECUTED', 'CANCELLED')),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trade outcomes table - logs results after trade completion
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_outcomes (
                outcome_id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL,

                -- Exit Information
                exit_timestamp DATETIME,
                exit_price REAL,
                exit_reason TEXT CHECK (exit_reason IN ('TARGET_HIT', 'STOP_LOSS', 'MANUAL_CLOSE', 'TIME_EXIT')),

                -- Performance Metrics
                pnl_actual REAL,
                pnl_percentage REAL,
                holding_period_bars INTEGER,

                -- Drawdown Analysis
                max_favorable_excursion REAL,  -- MFE: Maximum potential profit
                max_adverse_excursion REAL,    -- MAE: Maximum potential loss

                -- Classification
                is_profitable INTEGER CHECK (is_profitable IN (0, 1)),
                hit_target INTEGER CHECK (hit_target IN (0, 1)),
                hit_stop INTEGER CHECK (hit_stop IN (0, 1)),

                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (signal_id) REFERENCES trade_hypotheses (signal_id)
            )
        """)

        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                model_id TEXT PRIMARY KEY,
                model_version TEXT NOT NULL,
                training_date DATETIME DEFAULT CURRENT_TIMESTAMP,

                -- Performance Metrics
                accuracy REAL,
                roc_auc REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,

                -- Data Statistics
                training_samples INTEGER,
                test_samples INTEGER,
                feature_count INTEGER,

                -- Model Configuration
                model_params TEXT,  -- JSON string of hyperparameters
                feature_importance TEXT,  -- JSON string of feature importances

                is_active INTEGER DEFAULT 0 CHECK (is_active IN (0, 1))
            )
        """)

        conn.commit()
        conn.close()

        logger.info("Database tables initialized for Alpha Model")

    def log_trading_hypothesis(self, signal_data: Dict[str, Any]) -> str:
        """
        Log a trading signal as a testable hypothesis BEFORE execution

        This is Stage 1 of the Learning Loop: Hypothesis Generation
        Every signal becomes a formal hypothesis with rich feature context.
        """

        # Generate unique signal ID
        signal_id = str(uuid.uuid4())

        # Prepare data for database insertion
        hypothesis_data = {
            'signal_id': signal_id,
            'symbol': signal_data.get('symbol'),
            'timeframe': signal_data.get('timeframe', '1H'),
            'direction': signal_data.get('direction'),
            'entry_price': signal_data.get('entry_price'),
            'predicted_target': signal_data.get('target_price'),
            'predicted_stop_loss': signal_data.get('stop_loss'),
            'risk_reward_ratio': signal_data.get('risk_reward_ratio'),
            'position_size_pct': signal_data.get('position_size_pct'),
            'signal_source': signal_data.get('signal_source', 'AI_Signal_Generator'),
            'confluence_score': signal_data.get('confluence_score'),
            'ai_confidence': signal_data.get('ai_confidence'),
            'pattern_type': signal_data.get('pattern_type'),

            # Technical indicators
            'rsi_14': signal_data.get('rsi_14'),
            'macd_signal': signal_data.get('macd_signal'),
            'bb_position': signal_data.get('bb_position'),
            'volume_ratio': signal_data.get('volume_ratio'),
            'atr_14': signal_data.get('atr_14'),
            'adx_14': signal_data.get('adx_14'),
            'stoch_k': signal_data.get('stoch_k'),
            'williams_r': signal_data.get('williams_r'),

            # Market context
            'market_regime': signal_data.get('market_regime', 'UNKNOWN'),
            'volatility_percentile': signal_data.get('volatility_percentile'),
            'sector_momentum': signal_data.get('sector_momentum'),
            'vix_level': signal_data.get('vix_level'),

            # Timing
            'time_of_day': signal_data.get('time_of_day', datetime.now().hour),
            'day_of_week': signal_data.get('day_of_week', datetime.now().weekday()),
            'intraday_position': signal_data.get('intraday_position'),

            # AI context
            'ai_provider_used': signal_data.get('ai_provider_used'),
            'ai_response_time_ms': signal_data.get('ai_response_time_ms'),
            'ai_cost_usd': signal_data.get('ai_cost_usd')
        }

        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Dynamic SQL generation for flexibility
        columns = list(hypothesis_data.keys())
        placeholders = ['?' for _ in columns]
        values = [hypothesis_data[col] for col in columns]

        sql = f"""
            INSERT INTO trade_hypotheses ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """

        cursor.execute(sql, values)
        conn.commit()
        conn.close()

        logger.info(
            f"Trade hypothesis logged: {signal_id} - {signal_data.get('symbol')} {signal_data.get('direction')}")

        return signal_id

    def log_trade_outcome(self, signal_id: str, outcome_data: Dict[str, Any]) -> str:
        """
        Log the outcome of a completed trade

        This is Stage 3 of the Learning Loop: Result Analysis
        Links trade results back to the original hypothesis for learning.
        """

        outcome_id = str(uuid.uuid4())

        # Calculate profitability
        pnl_percentage = outcome_data.get('pnl_percentage', 0)
        is_profitable = 1 if pnl_percentage > 0 else 0

        outcome_record = {
            'outcome_id': outcome_id,
            'signal_id': signal_id,
            'exit_timestamp': outcome_data.get('exit_timestamp', datetime.now()),
            'exit_price': outcome_data.get('exit_price'),
            'exit_reason': outcome_data.get('exit_reason'),
            'pnl_actual': outcome_data.get('pnl_actual'),
            'pnl_percentage': pnl_percentage,
            'holding_period_bars': outcome_data.get('holding_period_bars'),
            'max_favorable_excursion': outcome_data.get('mfe'),
            'max_adverse_excursion': outcome_data.get('mae'),
            'is_profitable': is_profitable,
            'hit_target': outcome_data.get('hit_target', 0),
            'hit_stop': outcome_data.get('hit_stop', 0)
        }

        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        columns = list(outcome_record.keys())
        placeholders = ['?' for _ in columns]
        values = [outcome_record[col] for col in columns]

        sql = f"""
            INSERT INTO trade_outcomes ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """

        cursor.execute(sql, values)

        # Update hypothesis status
        cursor.execute("""
            UPDATE trade_hypotheses 
            SET status = 'EXECUTED' 
            WHERE signal_id = ?
        """, (signal_id,))

        conn.commit()
        conn.close()

        logger.info(f"Trade outcome logged: {outcome_id} - PnL: {pnl_percentage:.2f}%")

        return outcome_id

    def prepare_training_data(self, min_samples: int = 50) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare training data from completed trades

        Combines hypothesis data with outcomes to create ML training dataset.
        Only uses completed trades where we know the outcome.
        """

        conn = sqlite3.connect(self.db_path)

        # Query to join hypotheses with outcomes
        query = """
            SELECT 
                h.*,
                o.is_profitable,
                o.pnl_percentage,
                o.exit_reason,
                o.holding_period_bars,
                o.max_favorable_excursion,
                o.max_adverse_excursion
            FROM trade_hypotheses h
            INNER JOIN trade_outcomes o ON h.signal_id = o.signal_id
            WHERE h.status = 'EXECUTED'
            ORDER BY h.timestamp
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < min_samples:
            logger.warning(f"Insufficient training data: {len(df)} samples (need {min_samples})")
            return None

        # Feature engineering and encoding
        df = self._engineer_features(df)

        # Select feature columns that exist in the dataframe
        available_features = [col for col in self.feature_columns if col in df.columns]

        if len(available_features) < 10:  # Minimum viable features
            logger.warning(f"Too few features available: {len(available_features)}")
            return None

        # Prepare feature matrix and target vector
        X = df[available_features].copy()
        y = df[self.target_column].copy()

        # Handle missing values
        X = X.fillna(X.median())

        logger.info(f"Training data prepared: {len(X)} samples, {len(available_features)} features")

        return X, y

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer and encode categorical features for ML models"""

        # Encode categorical variables
        pattern_type_map = {
            'BREAKOUT': 0, 'REVERSAL': 1, 'CONTINUATION': 2, 'CONFLUENCE': 3, 'OTHER': 4
        }
        df['pattern_type_encoded'] = df['pattern_type'].map(pattern_type_map).fillna(4)

        market_regime_map = {
            'BULL': 0, 'BEAR': 1, 'SIDEWAYS': 2, 'UNKNOWN': 3
        }
        df['market_regime_encoded'] = df['market_regime'].map(market_regime_map).fillna(3)

        ai_provider_map = {
            'openai': 0, 'anthropic': 1, 'google_gemini': 2, 'LOCAL': 3
        }
        df['ai_provider_encoded'] = df['ai_provider_used'].map(ai_provider_map).fillna(3)

        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek

        # Risk features
        if 'predicted_target' in df.columns and 'predicted_stop_loss' in df.columns and 'entry_price' in df.columns:
            df['risk_reward_ratio'] = np.where(
                (df['entry_price'] - df['predicted_stop_loss']) != 0,
                abs(df['predicted_target'] - df['entry_price']) / abs(df['entry_price'] - df['predicted_stop_loss']),
                0
            )

        # Confidence ratios
        if 'confluence_score' in df.columns and 'ai_confidence' in df.columns:
            df['confidence_ratio'] = df['confluence_score'] * df['ai_confidence']

        return df

    def train_ensemble_models(self, retrain: bool = False) -> bool:
        """
        Train ensemble ML models for profitability prediction

        This is Stage 4 of the Learning Loop: Learning & Refinement
        Creates multiple models and combines them for better predictions.
        """

        # Check if we need to train
        model_files_exist = all(
            (self.models_dir / f"{name}_model.joblib").exists()
            for name in ['xgboost', 'lightgbm', 'neural_net', 'ensemble']
        )

        if model_files_exist and not retrain:
            logger.info("Loading existing trained models...")
            return self.load_trained_models()

        # Prepare training data
        data_result = self.prepare_training_data()
        if data_result is None:
            return False

        X, y = data_result

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        logger.info("Training Alpha Model ensemble...")

        # 1. XGBoost - Gradient Boosting
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )

        # 2. LightGBM - Fast Gradient Boosting
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )

        # 3. Neural Network - Deep Learning
        nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Train individual models
        individual_models = {
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'neural_net': nn_model
        }

        trained_models = {}
        performance_scores = {}

        for name, model in individual_models.items():
            try:
                # Cross-validation scoring
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')

                # Full training
                model.fit(X_scaled, y)

                # Store results
                trained_models[name] = model
                performance_scores[name] = cv_scores.mean()

                logger.info(f"{name} trained - AUC: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        # Create ensemble voting classifier
        if len(trained_models) >= 2:
            ensemble_estimators = [(name, model) for name, model in trained_models.items()]

            ensemble_model = VotingClassifier(
                estimators=ensemble_estimators,
                voting='soft'  # Use probability predictions
            )

            ensemble_model.fit(X_scaled, y)
            trained_models['ensemble'] = ensemble_model

            # Ensemble performance
            ensemble_scores = cross_val_score(ensemble_model, X_scaled, y, cv=tscv, scoring='roc_auc')
            performance_scores['ensemble'] = ensemble_scores.mean()

            logger.info(f"Ensemble trained - AUC: {ensemble_scores.mean():.3f}")

        # Save models
        self.models = trained_models
        self._save_models(X.columns.tolist(), performance_scores)

        self.is_trained = True

        logger.info(f"Alpha Model training complete! {len(trained_models)} models ready")

        return True

    def predict_profitability(self, signal_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict probability of profitability for a new signal

        This is the core prediction function that outputs PoP (Probability of Profit)
        Used to filter signals before execution.
        """

        if not self.is_trained:
            if not self.load_trained_models():
                logger.warning("No trained models available - using random baseline")
                return {'ensemble_pop': 0.5, 'confidence': 'LOW'}

        try:
            # Convert features to DataFrame for consistency
            feature_df = pd.DataFrame([signal_features])

            # Engineer features like in training
            feature_df = self._engineer_features(feature_df)

            # Select available features
            available_features = [col for col in self.feature_columns if col in feature_df.columns]
            X = feature_df[available_features].fillna(0)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get predictions from all models
            predictions = {}

            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    # Get probability of profitable class (class 1)
                    proba = model.predict_proba(X_scaled)[0][1]
                    predictions[f'{name}_pop'] = proba
                else:
                    # Binary prediction fallback
                    pred = model.predict(X_scaled)[0]
                    predictions[f'{name}_pop'] = pred

            # Ensemble prediction (weighted average or use ensemble model)
            if 'ensemble_pop' in predictions:
                ensemble_pop = predictions['ensemble_pop']
            else:
                # Fallback: average of individual predictions
                individual_pops = [v for k, v in predictions.items() if k != 'ensemble_pop']
                ensemble_pop = np.mean(individual_pops) if individual_pops else 0.5

            # Confidence assessment
            if ensemble_pop >= 0.8:
                confidence = 'VERY_HIGH'
            elif ensemble_pop >= 0.65:
                confidence = 'HIGH'
            elif ensemble_pop >= 0.35:
                confidence = 'MEDIUM'
            elif ensemble_pop >= 0.2:
                confidence = 'LOW'
            else:
                confidence = 'VERY_LOW'

            predictions['ensemble_pop'] = ensemble_pop
            predictions['confidence'] = confidence

            logger.info(f"Alpha Prediction: PoP = {ensemble_pop:.3f} ({confidence})")

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {'ensemble_pop': 0.5, 'confidence': 'LOW', 'error': str(e)}

    def _save_models(self, feature_names: List[str], performance_scores: Dict[str, float]):
        """Save trained models and metadata to disk"""

        # Save individual models
        for name, model in self.models.items():
            model_path = self.models_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)

        # Save scaler
        scaler_path = self.models_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        metadata = {
            'model_version': self.model_version,
            'feature_version': self.feature_version,
            'feature_names': feature_names,
            'performance_scores': performance_scores,
            'training_timestamp': datetime.now().isoformat(),
            'sample_count': len(feature_names)
        }

        metadata_path = self.models_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Log to database
        self._log_model_performance(performance_scores)

        logger.info("Models saved successfully")

    def load_trained_models(self) -> bool:
        """Load previously trained models from disk"""

        try:
            model_files = {
                'xgboost': self.models_dir / "xgboost_model.joblib",
                'lightgbm': self.models_dir / "lightgbm_model.joblib",
                'neural_net': self.models_dir / "neural_net_model.joblib",
                'ensemble': self.models_dir / "ensemble_model.joblib"
            }

            # Load available models
            loaded_models = {}
            for name, path in model_files.items():
                if path.exists():
                    loaded_models[name] = joblib.load(path)

            if not loaded_models:
                logger.warning("No saved models found")
                return False

            # Load scaler
            scaler_path = self.models_dir / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            self.models = loaded_models
            self.is_trained = True

            logger.info(f"Loaded {len(loaded_models)} trained models")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def _log_model_performance(self, performance_scores: Dict[str, float]):
        """Log model training performance to database"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Deactivate previous models
        cursor.execute("UPDATE model_performance SET is_active = 0")

        # Insert new model performance
        for model_name, score in performance_scores.items():
            model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            cursor.execute("""
                INSERT INTO model_performance (
                    model_id, model_version, accuracy, roc_auc, 
                    training_samples, feature_count, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, self.model_version, score, score,
                0, len(self.feature_columns), 1
            ))

        conn.commit()
        conn.close()

    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive Alpha Model statistics and performance metrics"""

        conn = sqlite3.connect(self.db_path)

        # Get hypothesis count
        hypothesis_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM trade_hypotheses", conn
        ).iloc[0]['count']

        # Get completed trades count
        completed_trades = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM trade_outcomes", conn
        ).iloc[0]['count']

        # Get profitability stats
        if completed_trades > 0:
            profit_stats = pd.read_sql_query("""
                SELECT 
                    AVG(is_profitable) as win_rate,
                    AVG(pnl_percentage) as avg_pnl,
                    MAX(pnl_percentage) as best_trade,
                    MIN(pnl_percentage) as worst_trade,
                    COUNT(CASE WHEN is_profitable = 1 THEN 1 END) as winners,
                    COUNT(CASE WHEN is_profitable = 0 THEN 1 END) as losers
                FROM trade_outcomes
            """, conn).iloc[0]
        else:
            profit_stats = {
                'win_rate': 0, 'avg_pnl': 0, 'best_trade': 0,
                'worst_trade': 0, 'winners': 0, 'losers': 0
            }

        # Get latest model performance
        latest_model = pd.read_sql_query("""
            SELECT * FROM model_performance 
            WHERE is_active = 1 
            ORDER BY training_date DESC 
            LIMIT 1
        """, conn)

        conn.close()

        return {
            'total_hypotheses': int(hypothesis_count),
            'completed_trades': int(completed_trades),
            'win_rate': float(profit_stats['win_rate']) if profit_stats['win_rate'] else 0,
            'avg_pnl': float(profit_stats['avg_pnl']) if profit_stats['avg_pnl'] else 0,
            'best_trade': float(profit_stats['best_trade']) if profit_stats['best_trade'] else 0,
            'worst_trade': float(profit_stats['worst_trade']) if profit_stats['worst_trade'] else 0,
            'total_winners': int(profit_stats['winners']) if profit_stats['winners'] else 0,
            'total_losers': int(profit_stats['losers']) if profit_stats['losers'] else 0,
            'is_trained': self.is_trained,
            'model_count': len(self.models),
            'latest_model_auc': float(latest_model.iloc[0]['roc_auc']) if not latest_model.empty else 0,
            'model_version': self.model_version
        }


# Example usage and testing
if __name__ == "__main__":
    """Example usage of Alpha Model Core"""

    # Initialize Alpha Model
    alpha_model = AlphaModelCore()

    # Example signal data
    sample_signal = {
        'symbol': 'RELIANCE',
        'direction': 'BUY',
        'entry_price': 2500.0,
        'target_price': 2600.0,
        'stop_loss': 2450.0,
        'confluence_score': 85.0,
        'ai_confidence': 0.78,
        'pattern_type': 'BREAKOUT',
        'rsi_14': 65.5,
        'macd_signal': 1.2,
        'volume_ratio': 1.8,
        'market_regime': 'BULL',
        'ai_provider_used': 'openai'
    }

    # Log as hypothesis (Stage 1: Hypothesis Generation)
    signal_id = alpha_model.log_trading_hypothesis(sample_signal)
    print(f"Signal logged with ID: {signal_id}")

    # Simulate trade outcome (Stage 3: Result Analysis)
    sample_outcome = {
        'exit_price': 2580.0,
        'exit_reason': 'TARGET_HIT',
        'pnl_percentage': 3.2,
        'holding_period_bars': 24,
        'hit_target': 1
    }

    outcome_id = alpha_model.log_trade_outcome(signal_id, sample_outcome)
    print(f"Outcome logged with ID: {outcome_id}")

    # Get model statistics
    stats = alpha_model.get_model_stats()
    print(f"Model Stats: {stats}")

    # Try prediction (will show warning if no training data)
    prediction = alpha_model.predict_profitability(sample_signal)
    print(f"Prediction: {prediction}")

    print("\nAlpha Model Core demonstration complete!")
    print("Next steps:")
    print("   1. Collect real trading data to train the model")
    print("   2. Integrate with existing signal generator")
    print("   3. Set up automated retraining schedule")
    print("   4. Build monitoring dashboard for model performance")