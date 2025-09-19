"""
Working Data Fetcher - Timestamp Fixed
Demo mode with proper SQLite timestamp handling

Location: #06_DATA/data_fetcher.py
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import random

class MarketDataFetcher:
    """Working data fetcher with timestamp fix"""

    def __init__(self, db_path="marketpulse.db"):
        self.db_path = Path(db_path)
        self.demo_mode = True
        self.demo_symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'RELIANCE.NS']
        print("Data fetcher initialized in DEMO MODE")

    def connect_database(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def generate_demo_data(self, symbol, days=30):
        """Generate realistic demo market data"""

        base_prices = {
            'SPY': 450.0,
            'AAPL': 180.0,
            'MSFT': 350.0, 
            'GOOGL': 140.0,
            'RELIANCE.NS': 2800.0
        }

        base_price = base_prices.get(symbol, 100.0)
        data = []
        current_price = base_price

        for i in range(days):
            # Generate realistic OHLCV data
            daily_change = random.uniform(-0.03, 0.03)

            open_price = current_price
            high_price = open_price * (1 + abs(daily_change) * random.uniform(0.5, 1.5))
            low_price = open_price * (1 - abs(daily_change) * random.uniform(0.5, 1.5))
            close_price = open_price * (1 + daily_change)
            volume = random.randint(1000000, 10000000)

            # Ensure logical price relationships
            if high_price < low_price:
                high_price, low_price = low_price, high_price

            if close_price > high_price:
                close_price = high_price * 0.99
            elif close_price < low_price:
                close_price = low_price * 1.01

            # FIXED: Convert timestamp to string for SQLite
            timestamp = datetime.now() - timedelta(days=days-i)
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

            data.append({
                'symbol': symbol,
                'timestamp': timestamp_str,  # Already a string
                'timeframe': '1d',
                'open_price': round(open_price, 2),
                'high_price': round(high_price, 2),
                'low_price': round(low_price, 2), 
                'close_price': round(close_price, 2),
                'volume': volume,
                'data_source': 'demo'
            })

            current_price = close_price

        return pd.DataFrame(data)

    def fetch_historical_data(self, symbol, period="1mo"):
        """Fetch demo historical data"""

        print(f"Generating demo data for {symbol}")
        return self.generate_demo_data(symbol, days=30)

    def store_historical_data(self, data):
        """Store data in database with proper timestamp handling"""

        if data is None or data.empty:
            return False

        try:
            conn = self.connect_database()
            cursor = conn.cursor()

            stored = 0
            for _, row in data.iterrows():
                # FIXED: timestamp is already a string, no conversion needed
                cursor.execute("""
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, timeframe, open_price, high_price, low_price, 
                     close_price, volume, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'], 
                    row['timestamp'],  # Already string format
                    row['timeframe'],
                    row['open_price'], 
                    row['high_price'], 
                    row['low_price'],
                    row['close_price'], 
                    row['volume'], 
                    row['data_source']
                ))
                stored += 1

            conn.commit()
            conn.close()

            print(f"Stored {stored} records for {data['symbol'].iloc[0]}")
            return True

        except Exception as e:
            print(f"Error storing data: {e}")
            return False

    def get_latest_price(self, symbol):
        """Get latest price for symbol"""

        try:
            # Try database first
            conn = self.connect_database()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT close_price FROM market_data 
                WHERE symbol = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return float(result['close_price'])

            # Generate fresh demo data
            data = self.fetch_historical_data(symbol)
            if data is not None and not data.empty:
                self.store_historical_data(data)
                return float(data['close_price'].iloc[-1])

            return None

        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None

    def populate_reliable_data(self):
        """Populate with demo data"""

        results = {}

        print("Populating database with demo market data...")

        for symbol in self.demo_symbols:
            try:
                data = self.fetch_historical_data(symbol)
                if data is not None:
                    success = self.store_historical_data(data)
                    results[symbol] = success
                else:
                    results[symbol] = False
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                results[symbol] = False

        successful = sum(1 for s in results.values() if s)
        print(f"Population complete: {successful}/{len(self.demo_symbols)} successful")

        return results

    def get_market_data_summary(self):
        """Get database summary"""

        try:
            conn = self.connect_database()
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(DISTINCT symbol) as symbols FROM market_data")
            symbols_count = cursor.fetchone()['symbols']

            cursor.execute("SELECT COUNT(*) as total FROM market_data") 
            total_records = cursor.fetchone()['total']

            cursor.execute("SELECT DISTINCT symbol FROM market_data")
            symbols = [row['symbol'] for row in cursor.fetchall()]

            conn.close()

            return {
                'symbols_count': symbols_count,
                'total_records': total_records,
                'symbols': symbols,
                'earliest_date': 'Demo data',
                'latest_date': 'Demo data'
            }

        except Exception as e:
            print(f"Error getting summary: {e}")
            return {}

def main():
    """Test the fixed data fetcher"""

    print("FIXED Demo Data Fetcher Test")
    print("=" * 35)

    fetcher = MarketDataFetcher()

    # Populate with demo data
    results = fetcher.populate_reliable_data()

    if any(results.values()):
        print("SUCCESS: Demo data populated!")

        # Test prices
        print("\nDemo stock prices:")
        for symbol in fetcher.demo_symbols[:3]:
            price = fetcher.get_latest_price(symbol)
            if price:
                if symbol.endswith('.NS'):
                    print(f"  {symbol}: Rs {price:.2f}")
                else:
                    print(f"  {symbol}: ${price:.2f}")

        # Show summary
        summary = fetcher.get_market_data_summary()
        print(f"\nDatabase: {summary['symbols_count']} symbols, {summary['total_records']} records")

        print("\n✅ TIMESTAMP ISSUE FIXED!")
        print("✅ Demo data integration working!")

        return True
    else:
        print("FAILED: Could not generate demo data")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
