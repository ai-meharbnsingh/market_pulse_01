# Database Migration Guide - Manual Updates Required

## Files That Need Manual Review:

### Files with Database References:

**D:\Users\OMEN\MarketPulse\update_code_for_new_databases.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed 'marketpulse.db' -> 'marketpulse_production.db'
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"
  - Changed 'marketpulse_production.db' -> 'marketpulse_production.db'
  - Changed "06_DATA/marketpulse.db" -> "06_DATA/marketpulse_training.db"
  - Changed '06_DATA/marketpulse.db' -> '06_DATA/marketpulse_training.db'
  - Changed "data/marketpulse.db" -> "marketpulse_production.db"
  - Changed 'data/marketpulse.db' -> 'marketpulse_production.db'
  - Changed "streaming_data.db" -> "marketpulse_production.db"
  - Changed 'streaming_data.db' -> 'marketpulse_production.db'
  - Changed "test_marketpulse" -> "marketpulse_production.db"
  - Changed 'test_marketpulse -> 'marketpulse_production.db'
  - Changed "10_DATA_STORAGE/performance/performance_metrics.db" -> "10_DATA_STORAGE/marketpulse_performance.db"
  - Changed '10_DATA_STORAGE/performance/performance_metrics.db' -> '10_DATA_STORAGE/marketpulse_performance.db'
  - Changed "10_DATA_STORAGE/ml_reliability/error_tracking.db" -> "10_DATA_STORAGE/marketpulse_performance.db"
  - Changed '10_DATA_STORAGE/ml_reliability/error_tracking.db' -> '10_DATA_STORAGE/marketpulse_performance.db'
  - Pattern update: training.*\.db
  - Pattern update: ml.*data.*\.db
  - Pattern update: performance.*\.db
  - Pattern update: error.*tracking.*\.db

**D:\Users\OMEN\MarketPulse\04_RISK\advanced_risk_management.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\05_EXECUTION\live_trading_engine.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\complete_option_a_sqlite_production.py:**
- Changes made:
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\complete_option_a_with_existing_postgres.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\database_cleanup_consolidation.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed 'marketpulse.db' -> 'marketpulse_production.db'
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"
  - Changed 'marketpulse_production.db' -> 'marketpulse_production.db'
  - Changed "06_DATA/marketpulse.db" -> "06_DATA/marketpulse_training.db"
  - Changed "data/marketpulse.db" -> "marketpulse_production.db"
  - Changed "streaming_data.db" -> "marketpulse_production.db"
  - Changed "10_DATA_STORAGE/performance/performance_metrics.db" -> "10_DATA_STORAGE/marketpulse_performance.db"
  - Changed "10_DATA_STORAGE/ml_reliability/error_tracking.db" -> "10_DATA_STORAGE/marketpulse_performance.db"
  - Pattern update: training.*\.db
  - Pattern update: performance.*\.db

**D:\Users\OMEN\MarketPulse\06_DATA\database_config.py:**
- Changes made:
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"
  - Pattern update: training.*\.db
  - Pattern update: performance.*\.db

**D:\Users\OMEN\MarketPulse\06_DATA\live_market_data_fetcher.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\production_database_migration.py:**
- Changes made:
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\real_time_streaming_system.py:**
- Changes made:
  - Changed "streaming_data.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\update_code_for_new_databases.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed 'marketpulse.db' -> 'marketpulse_production.db'
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"
  - Changed 'marketpulse_production.db' -> 'marketpulse_production.db'
  - Changed "06_DATA/marketpulse.db" -> "06_DATA/marketpulse_training.db"
  - Changed '06_DATA/marketpulse.db' -> '06_DATA/marketpulse_training.db'
  - Changed "data/marketpulse.db" -> "marketpulse_production.db"
  - Changed 'data/marketpulse.db' -> 'marketpulse_production.db'
  - Changed "streaming_data.db" -> "marketpulse_production.db"
  - Changed 'streaming_data.db' -> 'marketpulse_production.db'
  - Changed "test_marketpulse" -> "marketpulse_production.db"
  - Changed 'test_marketpulse -> 'marketpulse_production.db'
  - Changed "10_DATA_STORAGE/performance/performance_metrics.db" -> "10_DATA_STORAGE/marketpulse_performance.db"
  - Changed '10_DATA_STORAGE/performance/performance_metrics.db' -> '10_DATA_STORAGE/marketpulse_performance.db'
  - Changed "10_DATA_STORAGE/ml_reliability/error_tracking.db" -> "10_DATA_STORAGE/marketpulse_performance.db"
  - Changed '10_DATA_STORAGE/ml_reliability/error_tracking.db' -> '10_DATA_STORAGE/marketpulse_performance.db'
  - Pattern update: training.*\.db
  - Pattern update: ml.*data.*\.db
  - Pattern update: performance.*\.db
  - Pattern update: error.*tracking.*\.db

**D:\Users\OMEN\MarketPulse\06_DATA\yahoo_finance_integration.py:**
- Changes made:
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\07_DASHBOARD\live_trading_dashboard.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\scripts\deploy_phase3_production.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed 'marketpulse.db' -> 'marketpulse_production.db'
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"
  - Changed 'marketpulse_production.db' -> 'marketpulse_production.db'

**D:\Users\OMEN\MarketPulse\09_DOCS\scripts\cleanup_phase1_step2.py:**
- Changes made:
  - Changed 'marketpulse.db' -> 'marketpulse_production.db'
  - Changed 'marketpulse_production.db' -> 'marketpulse_production.db'

**D:\Users\OMEN\MarketPulse\07_DASHBOARD\enhanced\enhanced_dashboard.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\database\schema_enhancement.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\06_DATA\enhanced\data_fetcher.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\backtesting\backtesting_framework.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\integration\unified_model_integration.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\models\alpha_model.py:**
- Changes made:
  - Changed "data/marketpulse.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\models\lstm_intraday.py:**
- Changes made:
  - Changed "data/marketpulse.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\models\real_alpha_model_integration.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\models\real_lstm_model_integration.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\models\timeseries_forecaster.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\monitoring\persistent_monitoring_dashboard.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\optimization\production_performance_optimizer.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\performance\performance_logger.py:**
- Changes made:
  - Pattern update: performance.*\.db

**D:\Users\OMEN\MarketPulse\03_ML_ENGINE\reliability\error_handler.py:**
- Changes made:
  - Pattern update: error.*tracking.*\.db

**D:\Users\OMEN\MarketPulse\02_ANALYSIS\enhanced\enhanced_trading_system.py:**
- Changes made:
  - Changed "marketpulse.db" -> "marketpulse_production.db"
  - Changed "marketpulse_production.db" -> "marketpulse_production.db"

## Recommended Code Updates:

### 1. Use Database Config Module:
```python
# OLD (hardcoded paths)
db_path = "marketpulse.db"
conn = sqlite3.connect(db_path)

# NEW (centralized config)
from database_config import DatabaseConfig
db_path = DatabaseConfig.get_production_db_path()
conn = sqlite3.connect(db_path)
```

### 2. Purpose-based Database Selection:
```python
# For live trading
db_path = DatabaseConfig.get_db_for_purpose('trading')

# For ML training
db_path = DatabaseConfig.get_db_for_purpose('training')

# For performance monitoring
db_path = DatabaseConfig.get_db_for_purpose('performance')
```

### 3. Update Import Statements:
```python
# Add to top of files using databases
import sys
sys.path.append('06_DATA')
from database_config import DatabaseConfig
```

## Testing After Migration:
1. Run your main trading script: `python main.py`
2. Test dashboard: `streamlit run 07_DASHBOARD/dashboard_app.py`
3. Test ML training: `python 03_ML_ENGINE/models/alpha_model.py`
4. Verify paper trading: Test paper trading functionality

## Rollback Plan:
If issues arise, restore from backups in `database_backups/` directory.
