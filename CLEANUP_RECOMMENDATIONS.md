# Cleanup Recommendations for New_Start Project

## Files to Remove (Duplicates/Redundant)

### 1. Duplicate Files
```bash
# Remove duplicate train_models.py from root
rm ./train_models.py

# Remove older paper trading module (keep v2)
rm ./paper_trading_module.py
```

### 2. Redundant Test Files
```bash
# Move test files to tests directory
mkdir -p tests
mv test_*.py tests/
```

### 3. Old/Unused Files
```bash
# Remove debug files that are no longer needed
rm debug_supertrend.py
rm debug_technical_backtest.py
rm fix_supertrend.py

# Remove old launchers (keep Turkish versions)
rm start_paper_trading.py
rm start_demo_with_telegram.py
```

## Files to Keep and Organize

### Primary Files to Use:

1. **Paper Trading:**
   - Keep: `paper_trading_module_v2.py`
   - Keep: `baslat_integrated_trading.py`

2. **Telegram Bot:**
   - Keep: `telegram_bot_integrated.py`
   - Archive others: `telegram_bot_simple.py`, `telegram_bot_advanced.py`

3. **Best Backtests:**
   - Keep: `realistic_high_return_backtest.py`
   - Keep: `ultra_aggressive_backtest.py`
   - Archive: `simple_mtf_backtest.py`

## Suggested Directory Reorganization Script

```bash
#!/bin/bash
# cleanup_and_organize.sh

# Create new directory structure
mkdir -p core/api core/data core/trading
mkdir -p strategies/backtest strategies/live strategies/optimization
mkdir -p telegram/bots telegram/launchers
mkdir -p tests
mkdir -p archive

# Move API related files
mv core/algolab_api.py core/api/
mv core/algolab_socket.py core/api/

# Move data files
mv core/data_fetcher.py core/data/
mv core/mtf_data_fetcher.py core/data/
mv core/smart_yahoo_fetcher.py core/data/

# Move trading files
mv core/limit_order_manager.py core/trading/
mv core/order_book_simulator.py core/trading/
mv trading/risk_manager.py core/trading/

# Move backtest strategies
mv realistic_high_return_backtest.py strategies/backtest/
mv ultra_aggressive_backtest.py strategies/backtest/
mv enhanced_mtf_backtest.py strategies/backtest/

# Move live strategies
mv paper_trading_module_v2.py strategies/live/
mv strategies/signal_generator.py strategies/live/

# Move optimization
mv walk_forward_optimizer.py strategies/optimization/
mv dynamic_portfolio_optimizer.py strategies/optimization/

# Move telegram files
mv telegram_bot_integrated.py telegram/bots/
mv baslat_integrated_trading.py telegram/launchers/
mv baslat_demo_trading.py telegram/launchers/

# Archive old versions
mv telegram_bot_simple.py archive/
mv telegram_bot_advanced.py archive/
mv paper_trading_module.py archive/
mv simple_mtf_backtest.py archive/

# Move test files
mv test_*.py tests/

# Remove duplicates
rm -f ./train_models.py

echo "Cleanup complete!"
```

## Quick Cleanup Commands

For immediate cleanup without reorganization:

```bash
# Remove obvious duplicates and debug files
rm ./train_models.py
rm debug_*.py
rm fix_supertrend.py
rm paper_trading_module.py
rm start_paper_trading.py
rm start_demo_with_telegram.py

# Create archive for old versions
mkdir -p archive
mv telegram_bot_simple.py telegram_bot_advanced.py archive/
mv simple_mtf_backtest.py simple_technical_test.py archive/
```

## Performance Comparison of Strategies

| Strategy | Monthly Return Target | Risk Level | Best Use Case |
|----------|---------------------|------------|---------------|
| Realistic High Return | 5-6% | Medium-High | Production trading |
| Ultra Aggressive | 8-10% | Very High | High risk tolerance |
| Enhanced MTF | 3-4% | Medium | Conservative approach |

## Recommended Production Setup

1. Use `paper_trading_module_v2.py` for trading engine
2. Use `telegram_bot_integrated.py` for control
3. Launch with `baslat_integrated_trading.py`
4. Best strategy: `realistic_high_return_backtest.py` parameters