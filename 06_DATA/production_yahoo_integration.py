
# Production PostgreSQL Configuration for Yahoo Finance Integration
# Replace SQLite connection with PostgreSQL in production

from postgresql_timescale_setup import PostgreSQLTimescaleSetup

class ProductionYahooFinanceIntegration(YahooFinanceIntegration):
    """Production version with PostgreSQL backend"""

    def __init__(self, postgres_config: dict, cache_ttl: int = 30):
        self.postgres_config = postgres_config
        self.postgres_setup = PostgreSQLTimescaleSetup(postgres_config)

        # Initialize connection pool
        self.postgres_setup.initialize_connection_pool()

        # Call parent with PostgreSQL path
        super().__init__(db_path="postgresql", cache_ttl=cache_ttl)

    def _store_real_time_quote(self, quote_data):
        """Store quote in PostgreSQL instead of SQLite"""
        try:
            with self.postgres_setup.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO market_data.real_time_quotes 
                    (symbol, current_price, bid, ask, bid_size, ask_size,
                     day_change, day_change_percent, day_high, day_low,
                     day_volume, market_cap, pe_ratio, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                    current_price = EXCLUDED.current_price,
                    updated_at = EXCLUDED.updated_at
                """, (
                    quote_data['symbol'], quote_data['current_price'],
                    quote_data['bid'], quote_data['ask'],
                    quote_data['bid_size'], quote_data['ask_size'],
                    quote_data['day_change'], quote_data['day_change_percent'],
                    quote_data['day_high'], quote_data['day_low'],
                    quote_data['day_volume'], quote_data['market_cap'],
                    quote_data['pe_ratio'], quote_data['updated_at']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store quote in PostgreSQL: {e}")
