#!/usr/bin/env python3
"""
Simple script to run the Advanced Pattern Scanner.

This script demonstrates how to use the pattern detection system
to scan for stock patterns in real market data.
"""

import sys
import os
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_basic_scan():
    """Run a basic pattern scan on popular stocks."""
    print("🚀 Advanced Pattern Scanner - Quick Scan")
    print("=" * 60)
    
    try:
        # Import the pattern detection modules
        from advanced_pattern_scanner.core.models import PatternConfig, MarketData
        from advanced_pattern_scanner.patterns.head_shoulders import HeadShouldersDetector
        from advanced_pattern_scanner.patterns.double_bottom import DoubleBottomDetector
        from advanced_pattern_scanner.patterns.cup_handle import CupHandleDetector
        
        # Import data fetching
        import yfinance as yf
        import pandas as pd
        import numpy as np
        
        # Configuration
        config = PatternConfig(
            min_confidence=0.6,
            volume_confirmation_required=True,
            min_volume_ratio=1.2
        )
        
        # Initialize detectors
        hs_detector = HeadShouldersDetector(config)
        db_detector = DoubleBottomDetector(config)
        ch_detector = CupHandleDetector(config)
        
        # Test stocks
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        
        print(f"📊 Scanning {len(symbols)} stocks for patterns...")
        print(f"🔍 Looking for: Head & Shoulders, Double Bottom, Cup & Handle")
        print("-" * 60)
        
        total_patterns = 0
        
        for symbol in symbols:
            print(f"\n📈 Analyzing {symbol}...")
            
            try:
                # Fetch data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                
                if data.empty:
                    print(f"   ❌ No data available for {symbol}")
                    continue
                
                # Convert to MarketData format
                ohlcv_data = np.column_stack([
                    data['Open'].values,
                    data['High'].values,
                    data['Low'].values,
                    data['Close'].values,
                    data['Volume'].values
                ])
                
                market_data = MarketData(
                    symbol=symbol,
                    timeframe="1d",
                    data=ohlcv_data,
                    timestamps=[dt.to_pydatetime() for dt in data.index]
                )
                
                # Detect patterns
                patterns_found = 0
                
                # Head & Shoulders
                hs_patterns = hs_detector.detect_pattern(market_data)
                if hs_patterns:
                    patterns_found += len(hs_patterns)
                    for pattern in hs_patterns:
                        print(f"   ✅ Head & Shoulders - Confidence: {pattern.confidence:.1%}")
                        if hasattr(pattern, 'target_price'):
                            print(f"      🎯 Target: ${pattern.target_price:.2f}")
                
                # Double Bottom
                db_patterns = db_detector.detect_pattern(market_data)
                if db_patterns:
                    patterns_found += len(db_patterns)
                    for pattern in db_patterns:
                        print(f"   ✅ Double Bottom - Confidence: {pattern.confidence:.1%}")
                        if hasattr(pattern, 'target_price'):
                            print(f"      🎯 Target: ${pattern.target_price:.2f}")
                
                # Cup & Handle
                ch_patterns = ch_detector.detect_pattern(market_data)
                if ch_patterns:
                    patterns_found += len(ch_patterns)
                    for pattern in ch_patterns:
                        print(f"   ✅ Cup & Handle - Confidence: {pattern.confidence:.1%}")
                        if hasattr(pattern, 'target_price'):
                            print(f"      🎯 Target: ${pattern.target_price:.2f}")
                
                if patterns_found == 0:
                    print(f"   📊 No significant patterns detected")
                
                total_patterns += patterns_found
                
            except Exception as e:
                print(f"   ❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        print("\n" + "=" * 60)
        print(f"🎯 SCAN COMPLETE")
        print(f"📊 Total patterns found: {total_patterns}")
        print(f"📈 Stocks scanned: {len(symbols)}")
        print("=" * 60)
        
        if total_patterns > 0:
            print("\n💡 Tip: Use the Streamlit web interface for detailed charts:")
            print("   python3 -m streamlit run advanced_pattern_scanner/ui/streamlit_app.py")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Make sure you're in the correct directory and all dependencies are installed.")
        print("   Try: pip install yfinance pandas numpy")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Unexpected error occurred")

def run_batch_scan():
    """Run a batch scan on multiple stocks."""
    print("🚀 Advanced Pattern Scanner - Batch Mode")
    print("=" * 60)
    
    try:
        # Read symbols from file if it exists
        symbols_file = "stock_symbols.txt"
        if os.path.exists(symbols_file):
            with open(symbols_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            print(f"📋 Loaded {len(symbols)} symbols from {symbols_file}")
        else:
            # Use default list
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META", "AMZN", "NFLX"]
            print(f"📋 Using default symbol list: {len(symbols)} stocks")
        
        run_basic_scan()
        
    except Exception as e:
        print(f"❌ Error in batch scan: {e}")
        logger.exception("Batch scan error")

if __name__ == "__main__":
    print("🔍 Advanced Stock Pattern Scanner")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        run_batch_scan()
    else:
        run_basic_scan()
