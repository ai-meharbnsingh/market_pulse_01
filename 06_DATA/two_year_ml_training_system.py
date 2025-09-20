# 06_DATA/two_year_ml_training_system.py
"""
Two-Year Historical Data Fetcher & ML Training Pipeline
Enhanced system for comprehensive machine learning with 2 years of Indian stock data

Features:
- 2-year historical data collection
- Advanced feature engineering
- ML model training pipeline
- Backtesting framework
- Performance validation

Location: #06_DATA/two_year_ml_training_system.py
"""

import yfinance as yf
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple, Optional
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TwoYearMLTrainingSystem:
    """Comprehensive 2-year data collection and ML training system"""

    def __init__(self, db_path: str = "06_DATA/marketpulse_training.db"):
        self.db_path = os.path.abspath(db_path)
        self.training_days = 730  # 2 years

        # Comprehensive Indian stock universe
        self.indian_universe = self._create_expanded_indian_universe()

        # Create directories
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        logger.info(f"Initialized 2-year ML training system")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Training period: {self.training_days} days")

    def _create_expanded_indian_universe(self) -> Dict[str, Dict[str, List[str]]]:
        """Create expanded Indian stock universe for comprehensive ML training"""

        return {
            'large_cap': {
                'banking_finance': [
                    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
                    'AXISBANK.NS', 'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS',
                    'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
                    'HDFCLIFE.NS', 'ICICIPRULI.NS', 'SBILIFE.NS'
                ],
                'information_technology': [
                    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
                    'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS'
                ],
                'oil_gas_energy': [
                    'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'HINDPETRO.NS',
                    'GAIL.NS', 'COALINDIA.NS', 'NTPC.NS', 'POWERGRID.NS'
                ],
                'automobiles': [
                    'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS',
                    'EICHERMOT.NS', 'HEROMOTOCO.NS', 'TVSMOTORS.NS'
                ],
                'pharmaceutical': [
                    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS',
                    'BIOCON.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'CADILAHC.NS'
                ],
                'fmcg': [
                    'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
                    'DABUR.NS', 'GODREJCP.NS', 'MARICO.NS', 'COLPAL.NS'
                ],
                'metals_mining': [
                    'TATASTEEL.NS', 'HINDALCO.NS', 'VEDL.NS', 'JSWSTEEL.NS',
                    'SAIL.NS', 'NMDC.NS', 'MOIL.NS'
                ],
                'cement': [
                    'ULTRACEMCO.NS', 'SHREECEM.NS', 'GRASIM.NS', 'ACC.NS',
                    'AMBUJCEM.NS', 'RAMCOCEM.NS'
                ]
            },
            'mid_cap': {
                'telecom': [
                    'BHARTIARTL.NS', 'IDEA.NS'
                ],
                'infrastructure': [
                    'LT.NS', 'INFRATEL.NS', 'IRCON.NS'
                ],
                'textiles': [
                    'PAGEIND.NS', 'AIAENG.NS'
                ],
                'chemicals': [
                    'UPL.NS', 'PIDILITIND.NS', 'AAVAS.NS'
                ],
                'capital_goods': [
                    'BHEL.NS', 'BEML.NS', 'CONCOR.NS'
                ]
            },
            'small_cap': {
                'specialty_finance': [
                    'MANAPPURAM.NS', 'MUTHOOTFIN.NS', 'CHOLAFIN.NS'
                ],
                'consumer_discretionary': [
                    'BATAINDIA.NS', 'RELAXO.NS', 'VBL.NS'
                ],
                'healthcare': [
                    'APOLLOHOSP.NS', 'FORTIS.NS', 'MAXHEALTH.NS'
                ],
                'real_estate': [
                    'DLF.NS', 'GODREJPROP.NS', 'BRIGADE.NS'
                ],
                'utilities': [
                    'TORNTPOWER.NS', 'CESC.NS', 'NHPC.NS'
                ]
            }
        }

    def get_comprehensive_stock_list(self, max_stocks: int = 100) -> List[str]:
        """Get comprehensive list of stocks for training"""

        stocks = []
        stock_count = {'large_cap': 0, 'mid_cap': 0, 'small_cap': 0}

        # Prioritize large cap stocks for stability
        for market_cap in ['large_cap', 'mid_cap', 'small_cap']:
            if len(stocks) >= max_stocks:
                break

            for sector, stock_list in self.indian_universe[market_cap].items():
                for stock in stock_list:
                    if len(stocks) >= max_stocks:
                        break
                    stocks.append(stock)
                    stock_count[market_cap] += 1

        logger.info(f"Selected {len(stocks)} stocks: {stock_count}")
        return stocks

    def create_enhanced_database_schema(self):
        """Create comprehensive database schema for ML training"""

        logger.info("Creating enhanced database schema for ML training...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_enhanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                symbol_clean TEXT NOT NULL,
                market_cap_category TEXT NOT NULL,
                sector TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                adj_close REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Technical indicators table for ML features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Trend indicators
                sma_5 REAL, sma_10 REAL, sma_20 REAL, sma_50 REAL,
                ema_5 REAL, ema_10 REAL, ema_20 REAL, ema_50 REAL,

                -- Momentum indicators
                rsi_14 REAL, rsi_21 REAL,
                macd REAL, macd_signal REAL, macd_histogram REAL,
                stoch_k REAL, stoch_d REAL,
                williams_r REAL, cci REAL,

                -- Volatility indicators
                bb_upper REAL, bb_middle REAL, bb_lower REAL, bb_width REAL,
                atr_14 REAL, atr_21 REAL,

                -- Volume indicators
                volume_sma_20 REAL, volume_ratio REAL,
                vwap REAL, mfi REAL,

                -- Pattern indicators
                support_level REAL, resistance_level REAL,
                trend_strength REAL, volatility_regime REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # ML model predictions and performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,

                -- Predictions
                predicted_direction TEXT, -- BUY/SELL/HOLD
                probability_profit REAL,
                predicted_return REAL,
                predicted_holding_period INTEGER,

                -- Features used (JSON-like text)
                feature_vector TEXT,
                model_confidence REAL,

                -- Actual outcomes (filled later)
                actual_direction TEXT,
                actual_return REAL,
                actual_holding_period INTEGER,
                prediction_accuracy REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Backtesting results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,

                -- Performance metrics
                total_return REAL,
                annualized_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,

                -- Trade statistics
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                avg_winning_trade REAL,
                avg_losing_trade REAL,

                -- Risk metrics
                volatility REAL,
                var_95 REAL,
                calmar_ratio REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create performance indexes (after tables are created)
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data_enhanced(symbol, timestamp)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_technical_symbol_time ON technical_indicators(symbol, timestamp)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol_time ON ml_predictions(symbol, timestamp)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name, start_date)")
        except Exception as e:
            logger.warning(f"Index creation failed (non-critical): {e}")

        conn.commit()
        conn.close()

        logger.info("Enhanced database schema created successfully")

    def fetch_stock_data_parallel(self, stocks: List[str], days: int = 730) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple stocks in parallel"""

        def fetch_single_stock(symbol: str) -> Tuple[str, pd.DataFrame]:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days + 30)  # Extra buffer

                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)

                if not hist.empty and len(hist) >= days // 2:  # At least 50% data
                    # Clean and validate data
                    hist = hist.dropna()
                    if len(hist) >= 100:  # Minimum viable dataset
                        return symbol, hist

                logger.warning(f"Insufficient data for {symbol}: {len(hist) if not hist.empty else 0} records")
                return symbol, pd.DataFrame()

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                return symbol, pd.DataFrame()

        logger.info(f"Starting parallel data fetch for {len(stocks)} stocks...")

        stock_data = {}
        successful_downloads = 0

        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all download tasks
            future_to_symbol = {executor.submit(fetch_single_stock, symbol): symbol for symbol in stocks}

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    symbol, data = future.result()
                    if not data.empty:
                        stock_data[symbol] = data
                        successful_downloads += 1
                        logger.info(f"Success {successful_downloads}/{len(stocks)}: {symbol} - {len(data)} records")
                    else:
                        logger.warning(f"Empty data for {symbol}")

                    # Rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        logger.info(f"Data fetch complete: {successful_downloads}/{len(stocks)} stocks successful")
        return stock_data

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for ML features"""

        try:
            # Trend indicators
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['Close'].rolling(period).mean()
                df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()

            # Momentum indicators
            def calculate_rsi(prices, period=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            df['rsi_14'] = calculate_rsi(df['Close'], 14)
            df['rsi_21'] = calculate_rsi(df['Close'], 21)

            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Stochastic
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

            # Williams %R
            df['williams_r'] = -100 * (high_max - df['Close']) / (high_max - low_min)

            # CCI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            df['cci'] = (typical_price - sma_tp) / (0.015 * mad)

            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['bb_middle'] = df['Close'].rolling(bb_period).mean()
            bb_std_dev = df['Close'].rolling(bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr_14'] = true_range.rolling(14).mean()
            df['atr_21'] = true_range.rolling(21).mean()

            # Volume indicators
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

            # VWAP
            df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

            # Pattern recognition features
            df['support_level'] = df['Low'].rolling(20).min()
            df['resistance_level'] = df['High'].rolling(20).max()
            df['trend_strength'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
            df['volatility_regime'] = df['atr_14'] / df['Close']

            # Fill NaN values
            df = df.fillna(method='bfill').fillna(method='ffill')

            return df

        except Exception as e:
            logger.error(f"Technical indicator calculation failed: {e}")
            return df

    def populate_training_database(self, max_stocks: int = 75):
        """Populate database with 2 years of comprehensive training data"""

        logger.info(f"Starting 2-year data population for up to {max_stocks} stocks...")

        # Create database schema
        self.create_enhanced_database_schema()

        # Get stock list
        stocks = self.get_comprehensive_stock_list(max_stocks)

        # Fetch data in parallel
        stock_data = self.fetch_stock_data_parallel(stocks, self.training_days)

        if not stock_data:
            logger.error("No stock data fetched - aborting database population")
            return 0, len(stocks)

        # Populate database
        conn = sqlite3.connect(self.db_path)
        successful_stocks = 0
        total_records = 0

        for symbol, df in stock_data.items():
            try:
                # Get stock categorization
                market_cap, sector = self._get_stock_category(symbol)

                # Calculate technical indicators
                df_with_indicators = self.calculate_technical_indicators(df)

                # Insert market data
                for date, row in df.iterrows():
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_data_enhanced 
                        (symbol, symbol_clean, market_cap_category, sector, timestamp, 
                         open_price, high_price, low_price, close_price, volume, adj_close)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, symbol.replace('.NS', ''), market_cap, sector,
                        date.strftime('%Y-%m-%d'),
                        float(row['Open']), float(row['High']), float(row['Low']),
                        float(row['Close']), int(row['Volume']), float(row['Close'])
                    ))

                # Insert technical indicators
                for date, row in df_with_indicators.iterrows():
                    if pd.notna(row.get('rsi_14')):  # Only insert if indicators calculated
                        cursor.execute("""
                            INSERT OR REPLACE INTO technical_indicators 
                            (symbol, timestamp, sma_5, sma_10, sma_20, sma_50,
                             ema_5, ema_10, ema_20, ema_50, rsi_14, rsi_21,
                             macd, macd_signal, macd_histogram, stoch_k, stoch_d,
                             williams_r, cci, bb_upper, bb_middle, bb_lower, bb_width,
                             atr_14, atr_21, volume_sma_20, volume_ratio, vwap,
                             support_level, resistance_level, trend_strength, volatility_regime)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol, date.strftime('%Y-%m-%d'),
                            row.get('sma_5'), row.get('sma_10'), row.get('sma_20'), row.get('sma_50'),
                            row.get('ema_5'), row.get('ema_10'), row.get('ema_20'), row.get('ema_50'),
                            row.get('rsi_14'), row.get('rsi_21'), row.get('macd'), row.get('macd_signal'),
                            row.get('macd_histogram'), row.get('stoch_k'), row.get('stoch_d'),
                            row.get('williams_r'), row.get('cci'), row.get('bb_upper'), row.get('bb_middle'),
                            row.get('bb_lower'), row.get('bb_width'), row.get('atr_14'), row.get('atr_21'),
                            row.get('volume_sma_20'), row.get('volume_ratio'), row.get('vwap'),
                            row.get('support_level'), row.get('resistance_level'),
                            row.get('trend_strength'), row.get('volatility_regime')
                        ))

                conn.commit()
                successful_stocks += 1
                records_count = len(df)
                total_records += records_count

                logger.info(f"SUCCESS {successful_stocks}: {symbol} - {records_count} records with indicators")

            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue

        conn.close()

        logger.info(f"""
        2-YEAR DATA POPULATION COMPLETE:
        Successful stocks: {successful_stocks}
        Failed stocks: {len(stocks) - successful_stocks}
        Total records: {total_records}
        Training period: {self.training_days} days
        Database: {self.db_path}
        """)

        return successful_stocks, len(stocks) - successful_stocks

    def _get_stock_category(self, symbol: str) -> Tuple[str, str]:
        """Get market cap category and sector for a stock"""

        for market_cap, sectors in self.indian_universe.items():
            for sector, stocks in sectors.items():
                if symbol in stocks:
                    return market_cap, sector

        return 'unknown', 'unknown'

    def get_training_statistics(self) -> Dict:
        """Get comprehensive statistics about the training data"""

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            stats = {}

            # Market data statistics
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data_enhanced")
            stats['total_stocks'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM market_data_enhanced")
            stats['total_market_records'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM technical_indicators")
            stats['total_indicator_records'] = cursor.fetchone()[0]

            # Date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data_enhanced")
            date_range = cursor.fetchone()
            stats['date_range'] = {'start': date_range[0], 'end': date_range[1]}

            # Market cap distribution
            cursor.execute("""
                SELECT market_cap_category, COUNT(DISTINCT symbol) 
                FROM market_data_enhanced 
                GROUP BY market_cap_category
            """)
            stats['market_cap_distribution'] = dict(cursor.fetchall())

            # Sector distribution
            cursor.execute("""
                SELECT sector, COUNT(DISTINCT symbol) 
                FROM market_data_enhanced 
                GROUP BY sector
            """)
            stats['sector_distribution'] = dict(cursor.fetchall())

            # Data quality metrics
            cursor.execute("""
                SELECT symbol, COUNT(*) as record_count 
                FROM market_data_enhanced 
                GROUP BY symbol 
                ORDER BY record_count DESC 
                LIMIT 10
            """)
            stats['top_stocks_by_records'] = dict(cursor.fetchall())

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"Statistics retrieval failed: {e}")
            return {'error': str(e)}


def main():
    """Main function to execute 2-year ML training data collection"""

    print("2-YEAR ML TRAINING DATA COLLECTION SYSTEM")
    print("=" * 60)

    # Initialize system
    training_system = TwoYearMLTrainingSystem()

    print(f"Database: {training_system.db_path}")
    print(f"Training period: {training_system.training_days} days (2 years)")

    # Populate training database with 75 stocks
    print("\nStarting comprehensive data collection...")
    successful, failed = training_system.populate_training_database(max_stocks=75)

    # Display comprehensive statistics
    print("\nTRAINING DATA STATISTICS:")
    print("-" * 40)

    stats = training_system.get_training_statistics()

    if 'error' not in stats:
        print(f"Total Stocks: {stats['total_stocks']}")
        print(f"Market Data Records: {stats['total_market_records']:,}")
        print(f"Technical Indicator Records: {stats['total_indicator_records']:,}")
        print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"Market Cap Distribution: {stats['market_cap_distribution']}")
        print(f"Sector Distribution: {stats['sector_distribution']}")

        print(f"\nTop Stocks by Record Count:")
        for symbol, count in list(stats['top_stocks_by_records'].items())[:5]:
            print(f"  {symbol}: {count} records")

        print(f"\nSUCCESS: Ready for ML model training!")
        print(f"Next steps:")
        print(f"1. Train AlphaModel: python 03_ML_ENGINE/models/alpha_model.py")
        print(f"2. Train LSTM: python 03_ML_ENGINE/models/lstm_intraday.py")
        print(f"3. Run backtesting: python 03_ML_ENGINE/backtesting/backtesting_framework.py")
        print(f"4. Start paper trading: python main.py start")

    else:
        print(f"ERROR: {stats['error']}")

    print("\n2-YEAR ML TRAINING SYSTEM READY!")


if __name__ == "__main__":
    main()