#!/usr/bin/env python3
"""
Quick backtest runner with preset configurations.

Usage:
    python run_backtest.py --preset quick     # 3 months, fast
    python run_backtest.py --preset standard  # 6 months, balanced
    python run_backtest.py --preset full      # 12 months, comprehensive
    python run_backtest.py --preset custom --months 18 --symbols BTC/USDT
"""

import argparse
import sys
import os
from datetime import datetime

# Preset configurations
PRESETS = {
    'quick': {
        'months': 3,
        'symbols': ['BTC/USDT'],
        'description': 'Quick 3-month test on BTC only (~3 min)'
    },
    'standard': {
        'months': 6,
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'description': 'Standard 6-month test on BTC & ETH (~8 min)'
    },
    'full': {
        'months': 12,
        'symbols': ['BTC/USDT', 'ETH/USDT'],
        'description': 'Full 12-month comprehensive backtest (~15 min)'
    },
    'extended': {
        'months': 18,
        'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        'description': 'Extended 18-month test on 3 symbols (~25 min)'
    }
}


def run_backtest(months, symbols, clear_cache=False):
    """
    Run backtest with specified parameters.
    """
    print("\n" + "="*70)
    print(f"  BACKTEST CONFIGURATION")
    print("="*70)
    print(f"  Data period: {months} months")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Clear cache: {clear_cache}")
    print("="*70 + "\n")

    # Clear cache if requested
    if clear_cache:
        cache_dir = 'data/cache'
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print("✓ Cache cleared\n")

    # Temporarily modify the backtest script's configuration
    # We'll do this by setting environment variables that the script can read
    os.environ['BACKTEST_MONTHS'] = str(months)
    os.environ['BACKTEST_SYMBOLS'] = ','.join(symbols)

    # Import and run the backtest
    print("Starting backtest...\n")
    start_time = datetime.now()

    try:
        # Import the backtest module
        import importlib.util
        spec = importlib.util.spec_from_file_location("backtest", "backtest_improved.py")
        backtest = importlib.util.module_from_spec(spec)
        
        # Modify its configuration
        backtest.BACKTEST_MONTHS = months
        
        # Override symbols in config
        import config
        original_symbols = config.SYMBOLS
        config.SYMBOLS = symbols
        
        # Execute the backtest
        spec.loader.exec_module(backtest)
        
        # Restore original symbols
        config.SYMBOLS = original_symbols
        
        duration = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*70)
        print(f"  BACKTEST COMPLETE")
        print("="*70)
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"  Results saved in: logs/")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run backtest with preset configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  quick      - 3 months, BTC only (~3 min)
  standard   - 6 months, BTC & ETH (~8 min)
  full       - 12 months, BTC & ETH (~15 min)
  extended   - 18 months, BTC, ETH & SOL (~25 min)

Examples:
  python run_backtest.py --preset quick
  python run_backtest.py --preset full --clear-cache
  python run_backtest.py --preset custom --months 9 --symbols BTC/USDT ETH/USDT
        """
    )
    
    parser.add_argument(
        '--preset',
        choices=['quick', 'standard', 'full', 'extended', 'custom'],
        default='standard',
        help='Preset configuration (default: standard)'
    )
    
    parser.add_argument(
        '--months',
        type=int,
        help='Number of months to backtest (for custom preset)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Trading symbols (for custom preset), e.g., BTC/USDT ETH/USDT'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cached data before running'
    )
    
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List all available presets'
    )
    
    args = parser.parse_args()
    
    # List presets and exit
    if args.list_presets:
        print("\nAvailable Presets:\n")
        for name, config in PRESETS.items():
            print(f"  {name:10} - {config['description']}")
            print(f"  {'':10}   Months: {config['months']}, Symbols: {', '.join(config['symbols'])}\n")
        return
    
    # Get configuration
    if args.preset == 'custom':
        if not args.months or not args.symbols:
            print("❌ Error: --months and --symbols required for custom preset")
            parser.print_help()
            sys.exit(1)
        months = args.months
        symbols = args.symbols
    else:
        preset = PRESETS[args.preset]
        months = preset['months']
        symbols = preset['symbols']
        print(f"\nUsing preset: {args.preset}")
        print(f"Description: {preset['description']}\n")
    
    # Run backtest
    success = run_backtest(months, symbols, args.clear_cache)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()