# 06_DATA/database/schema_update_phase3.sql
"""
Phase 3 Database Schema Updates
Fix compatibility issues for live trading engine

Execute this to update your existing database schema
Location: #06_DATA/database/schema_update_phase3.sql
"""

-- Update trades table to match live trading engine expectations
ALTER TABLE trades ADD COLUMN executed_price REAL DEFAULT 0.0;

-- Update portfolios table to match expected columns
ALTER TABLE portfolios ADD COLUMN day_pnl REAL DEFAULT 0.0;

-- Create index for better performance
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);

-- Example usage in Python:
-- conn = sqlite3.connect('marketpulse.db')
-- with open('06_DATA/database/schema_update_phase3.sql', 'r') as f:
--     conn.executescript(f.read())
-- conn.close()