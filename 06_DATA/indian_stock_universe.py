# 06_DATA/indian_stock_universe.py
"""
Indian Stock Universe for BSE/NSE Trading
Comprehensive stock selection for Indian markets with proper categorization

Features:
- Large Cap, Mid Cap, Small Cap categorization
- Sector-wise distribution
- NSE and BSE symbols
- Market cap based selection
- Liquidity considerations

Location: #06_DATA/indian_stock_universe.py
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndianStockUniverse:
    """Comprehensive Indian stock universe for ML training"""

    def __init__(self, db_path: str = "marketpulse_production.db"):
        self.db_path = db_path
        self.indian_universe = self._create_indian_universe()

    def _create_indian_universe(self) -> Dict[str, Dict[str, List[str]]]:
        """Create comprehensive Indian stock universe with proper categorization"""

        universe = {
            # LARGE CAP (Market Cap > ₹20,000 Cr) - Top 100 companies
            'large_cap': {
                'banking_finance': [
                    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
                    'AXISBANK.NS', 'INDUSINDBK.NS', 'BANKBARODA.NS', 'PNB.NS',
                    'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS',
                    'HDFCLIFE.NS', 'ICICIPRULI.NS', 'SBILIFE.NS', 'LICI.NS'
                ],

                'information_technology': [
                    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS',
                    'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'PERSISTENT.NS'
                ],

                'oil_gas_energy': [
                    'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'HINDPETRO.NS',
                    'GAIL.NS', 'COALINDIA.NS', 'NTPC.NS', 'POWERGRID.NS', 'ADANIGREEN.NS'
                ],

                'automobiles': [
                    'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS',
                    'HEROMOTOCO.NS', 'TVSMOTORS.NS', 'EICHERMOT.NS', 'ASHOKLEY.NS'
                ],

                'pharmaceuticals': [
                    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS',
                    'BIOCON.NS', 'LUPIN.NS', 'AUROPHARMA.NS', 'CADILAHC.NS'
                ],

                'fmcg_consumer': [
                    'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS',
                    'DABUR.NS', 'MARICO.NS', 'GODREJCP.NS', 'COLPAL.NS'
                ],

                'metals_mining': [
                    'TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS',
                    'NMDC.NS', 'SAIL.NS', 'JINDALSTEL.NS', 'MOIL.NS'
                ],

                'cement': [
                    'ULTRACEMCO.NS', 'SHREECEM.NS', 'GRASIM.NS', 'ACC.NS',
                    'AMBUJACEMENT.NS', 'JKCEMENT.NS', 'RAMCOCEM.NS'
                ],

                'telecom': [
                    'BHARTIARTL.NS', 'IDEA.NS', 'RCOM.NS'
                ],

                'infrastructure': [
                    'LT.NS', 'ADANIPORTS.NS', 'ADANIENT.NS', 'GMRINFRA.NS'
                ]
            },

            # MID CAP (Market Cap ₹5,000-20,000 Cr) - Rank 101-250
            'mid_cap': {
                'banking_finance': [
                    'BANDHANBNK.NS', 'RBLBANK.NS', 'YESBANK.NS', 'CANBK.NS',
                    'UNIONBANK.NS', 'INDIANB.NS', 'ICICIGI.NS', 'SBICARD.NS'
                ],

                'information_technology': [
                    'LTTS.NS', 'HEXAWARE.NS', 'CYIENT.NS', 'OFSS.NS',
                    'KPITTECH.NS', 'RATEGAIN.NS', 'ROUTE.NS'
                ],

                'pharmaceuticals': [
                    'TORNTPHARM.NS', 'ALKEM.NS', 'ABBOTINDIA.NS', 'PFIZER.NS',
                    'GLAXO.NS', 'IPCALAB.NS', 'LALPATHLAB.NS'
                ],

                'automobiles': [
                    'ESCORTS.NS', 'SONACOMS.NS', 'MOTHERSUMI.NS', 'BALKRISIND.NS',
                    'APOLLOTYRE.NS', 'MRF.NS', 'CEAT.NS'
                ],

                'consumer_goods': [
                    'PIDILITIND.NS', 'BATINDIA.NS', 'MCDOWELL-N.NS', 'JUBLFOOD.NS',
                    'TATACONSUM.NS', 'EMAMILTD.NS', 'VSTIND.NS'
                ],

                'textiles': [
                    'PAGEIND.NS', 'AIAENG.NS', 'WELCORP.NS', 'TRIDENT.NS'
                ],

                'chemicals': [
                    'BALRAMCHIN.NS', 'DEEPAKNTR.NS', 'GNFC.NS', 'CHAMBLFERT.NS',
                    'TATACHEM.NS', 'NAVINFLUO.NS'
                ],

                'real_estate': [
                    'DLF.NS', 'GODREJPROP.NS', 'OBEROIRLTY.NS', 'PRESTIGE.NS',
                    'BRIGADE.NS', 'SOBHA.NS'
                ]
            },

            # SMALL CAP (Market Cap ₹500-5,000 Cr) - Rank 251+
            'small_cap': {
                'specialty_chemicals': [
                    'CLEAN.NS', 'FINPIPE.NS', 'ROSSARI.NS', 'ANUPAMRASAYAN.NS',
                    'CHEMCON.NS', 'TATVA.NS'
                ],

                'pharmaceuticals': [
                    'STAR.NS', 'JBCHEPHARM.NS', 'CAPLIPOINT.NS', 'LAURUS.NS',
                    'SEQUENT.NS', 'SUVEN.NS'
                ],

                'engineering': [
                    'TIINDIA.NS', 'THERMAX.NS', 'CUMMINSIND.NS', 'KIRLOSENG.NS',
                    'ELECON.NS', 'KEI.NS'
                ],

                'textiles': [
                    'GARFIBRES.NS', 'RSWM.NS', 'NIITLTD.NS', 'JCHAC.NS'
                ],

                'metals': [
                    'APARINDS.NS', 'KALYANKJIL.NS', 'RATNAMANI.NS', 'MANAPPURAM.NS'
                ],

                'consumer_discretionary': [
                    'RELAXO.NS', 'VIP.NS', 'BATA.NS', 'CROMPTON.NS',
                    'HAVELLS.NS', 'WHIRLPOOL.NS'
                ],

                'logistics': [
                    'BLUEDART.NS', 'GATI.NS', 'CONCOR.NS', 'ALLCARGO.NS'
                ],

                'agriculture': [
                    'RALLIS.NS', 'COROMANDEL.NS', 'KRIBHCO.NS', 'ZUARI.NS'
                ]
            }
        }

        return universe

    def get_comprehensive_stock_list(self, include_small_cap: bool = True) -> List[str]:
        """Get comprehensive list of Indian stocks for training"""

        all_stocks = []

        # Add Large Cap stocks
        for sector in self.indian_universe['large_cap'].values():
            all_stocks.extend(sector)

        # Add Mid Cap stocks
        for sector in self.indian_universe['mid_cap'].values():
            all_stocks.extend(sector)

        # Add Small Cap stocks (optional)
        if include_small_cap:
            for sector in self.indian_universe['small_cap'].values():
                all_stocks.extend(sector)

        # Remove duplicates and sort
        all_stocks = sorted(list(set(all_stocks)))

        logger.info(f"Created comprehensive Indian stock universe with {len(all_stocks)} stocks")
        return all_stocks

    def get_stocks_by_market_cap(self, market_cap: str) -> List[str]:
        """Get stocks by market cap category"""

        if market_cap.lower() not in ['large_cap', 'mid_cap', 'small_cap']:
            raise ValueError("market_cap must be 'large_cap', 'mid_cap', or 'small_cap'")

        stocks = []
        for sector in self.indian_universe[market_cap.lower()].values():
            stocks.extend(sector)

        return sorted(list(set(stocks)))

    def get_stocks_by_sector(self, sector: str, market_cap: str = None) -> List[str]:
        """Get stocks by sector, optionally filtered by market cap"""

        stocks = []

        if market_cap:
            if sector in self.indian_universe[market_cap]:
                stocks = self.indian_universe[market_cap][sector]
        else:
            # Search across all market caps
            for cap in self.indian_universe.values():
                if sector in cap:
                    stocks.extend(cap[sector])

        return sorted(list(set(stocks)))

    def get_market_cap_distribution(self) -> Dict[str, int]:
        """Get distribution of stocks by market cap"""

        distribution = {}
        for market_cap, sectors in self.indian_universe.items():
            count = 0
            for sector_stocks in sectors.values():
                count += len(sector_stocks)
            distribution[market_cap] = count

        return distribution

    def get_sector_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of stocks by sector within each market cap"""

        distribution = {}
        for market_cap, sectors in self.indian_universe.items():
            distribution[market_cap] = {}
            for sector, stocks in sectors.items():
                distribution[market_cap][sector] = len(stocks)

        return distribution

    def create_training_database_schema(self):
        """Create database schema with market cap and sector categorization"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create enhanced market_data table with categorization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_enhanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                symbol_clean TEXT NOT NULL,  -- Without .NS suffix
                company_name TEXT,
                market_cap_category TEXT NOT NULL,  -- large_cap, mid_cap, small_cap
                sector TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                adj_close REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Create index for efficient querying
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_enhanced_symbol_time 
            ON market_data_enhanced(symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_enhanced_category 
            ON market_data_enhanced(market_cap_category, sector)
        """)

        conn.commit()
        conn.close()

        logger.info("Enhanced database schema created with market cap and sector categorization")

    def populate_sample_data(self, days: int = 30):
        """Populate database with sample data from comprehensive Indian universe"""

        # Get comprehensive stock list (excluding small cap for initial training)
        stocks = self.get_comprehensive_stock_list(include_small_cap=False)

        logger.info(f"Starting data population for {len(stocks)} Indian stocks...")

        # Create database schema
        self.create_training_database_schema()

        conn = sqlite3.connect(self.db_path)

        successful_stocks = 0
        failed_stocks = 0

        for i, symbol in enumerate(stocks[:50]):  # Start with top 50 for testing
            try:
                logger.info(f"Fetching data for {symbol} ({i + 1}/{min(50, len(stocks))})")

                # Determine market cap and sector
                market_cap, sector = self._get_stock_category(symbol)

                # Fetch historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)

                # Use yf.download instead of ticker.download
                hist = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if not hist.empty:
                    # Prepare data for insertion
                    for date, row in hist.iterrows():
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO market_data_enhanced 
                            (symbol, symbol_clean, market_cap_category, sector, timestamp, 
                             open_price, high_price, low_price, close_price, volume, adj_close)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol,
                            symbol.replace('.NS', ''),
                            market_cap,
                            sector,
                            date.strftime('%Y-%m-%d'),
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            int(row['Volume']),
                            float(row['Adj Close'])
                        ))

                    conn.commit()
                    successful_stocks += 1
                    logger.info(f"✅ Successfully added {len(hist)} records for {symbol}")
                else:
                    failed_stocks += 1
                    logger.warning(f"⚠️ No data available for {symbol}")

                # Rate limiting to avoid API limits
                time.sleep(0.1)

            except Exception as e:
                failed_stocks += 1
                logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
                continue

        conn.close()

        logger.info(f"""
        📊 DATA POPULATION COMPLETE:
        ✅ Successful: {successful_stocks} stocks
        ❌ Failed: {failed_stocks} stocks
        📈 Total records: {successful_stocks * days} (approx)
        """)

        return successful_stocks, failed_stocks

    def _get_stock_category(self, symbol: str) -> Tuple[str, str]:
        """Get market cap category and sector for a stock"""

        for market_cap, sectors in self.indian_universe.items():
            for sector, stocks in sectors.items():
                if symbol in stocks:
                    return market_cap, sector

        # Default if not found
        return 'unknown', 'unknown'

    def get_training_statistics(self) -> Dict:
        """Get statistics about the training data"""

        conn = sqlite3.connect(self.db_path)

        stats = {}

        # Total stocks and records
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data_enhanced")
        stats['total_stocks'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM market_data_enhanced")
        stats['total_records'] = cursor.fetchone()[0]

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

        # Date range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM market_data_enhanced")
        date_range = cursor.fetchone()
        stats['date_range'] = {
            'start': date_range[0],
            'end': date_range[1]
        }

        conn.close()

        return stats


def main():
    """Main function to set up comprehensive Indian stock universe"""
    print("🇮🇳 SETTING UP COMPREHENSIVE INDIAN STOCK UNIVERSE")
    print("=" * 60)

    universe = IndianStockUniverse()

    # Show universe statistics
    distribution = universe.get_market_cap_distribution()
    print("\n📊 STOCK UNIVERSE COMPOSITION:")
    print(f"Large Cap: {distribution['large_cap']} stocks")
    print(f"Mid Cap: {distribution['mid_cap']} stocks")
    print(f"Small Cap: {distribution['small_cap']} stocks")
    print(f"Total Universe: {sum(distribution.values())} stocks")

    # Show sector distribution
    sector_dist = universe.get_sector_distribution()
    print("\n🏢 SECTOR DISTRIBUTION:")
    for market_cap, sectors in sector_dist.items():
        print(f"\n{market_cap.upper()}:")
        for sector, count in sectors.items():
            print(f"  {sector}: {count} stocks")

    # Populate database with comprehensive data
    print("\n🔄 POPULATING DATABASE WITH INDIAN MARKET DATA...")
    successful, failed = universe.populate_sample_data(days=60)

    # Show training statistics
    print("\n📈 TRAINING DATA STATISTICS:")
    stats = universe.get_training_statistics()
    print(f"Total Stocks: {stats['total_stocks']}")
    print(f"Total Records: {stats['total_records']}")
    print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")

    print("\n🎯 READY FOR ML TRAINING ON COMPREHENSIVE INDIAN MARKET DATA!")


if __name__ == "__main__":
    main()