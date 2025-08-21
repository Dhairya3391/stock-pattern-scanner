"""
Batch Pattern Scanner with Apple Silicon Optimization.

High-performance batch processing for multiple stock scanning with
optimizations for Apple Silicon (M1/M2/M3) processors.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd

from core.models import PatternConfig
from core.pattern_engine import PatternEngine
from core.pattern_scorer import PatternScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchPatternScanner:
    """
    High-performance batch scanner optimized for Apple Silicon.
    
    Features:
    - Parallel processing using all available CPU cores
    - Memory-efficient data handling
    - Progress tracking and reporting
    - Results export in multiple formats
    - Apple Silicon specific optimizations
    """
    
    def __init__(self, config: PatternConfig, max_workers: Optional[int] = None):
        """
        Initialize batch scanner.
        
        Args:
            config: Pattern detection configuration
            max_workers: Maximum number of worker processes (None for auto-detect)
        """
        self.config = config
        
        # Optimize for Apple Silicon
        if max_workers is None:
            # Use all available cores but leave one for system
            max_workers = max(1, mp.cpu_count() - 1)
        
        self.max_workers = max_workers
        self.pattern_engine = PatternEngine(config)
        self.pattern_scorer = PatternScorer(config)
        
        # Performance tracking
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_symbols": 0,
            "successful_scans": 0,
            "failed_scans": 0,
            "total_patterns": 0,
            "processing_times": [],
            "memory_usage": []
        }
        
        logger.info(f"BatchPatternScanner initialized with {max_workers} workers")
    
    def load_symbol_list(self, source: str) -> List[str]:
        """
        Load stock symbols from various sources.
        
        Args:
            source: Source of symbols ('file', 'nifty50', 'sp500', or comma-separated list)
            
        Returns:
            List of stock symbols
        """
        symbols = []
        
        if source == 'nifty50':
            # Load NIFTY 50 symbols
            symbols = self._load_nifty50_symbols()
        elif source == 'sp500':
            # Load S&P 500 symbols (simplified list)
            symbols = self._load_sp500_symbols()
        elif source.endswith('.txt') or source.endswith('.csv'):
            # Load from file
            symbols = self._load_symbols_from_file(source)
        else:
            # Treat as comma-separated list
            symbols = [s.strip().upper() for s in source.split(',') if s.strip()]
        
        logger.info(f"Loaded {len(symbols)} symbols from {source}")
        return symbols
    
    def _load_nifty50_symbols(self) -> List[str]:
        """Load NIFTY 50 symbols."""
        # Check if nifty50.txt exists
        nifty_file = Path("nifty50.txt")
        if nifty_file.exists():
            return self._load_symbols_from_file(str(nifty_file))
        
        # Fallback to hardcoded list (top 20 NIFTY stocks)
        return [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
            "BAJFINANCE.NS", "LICI.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS",
            "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "ONGC.NS", "NESTLEIND.NS"
        ]
    
    def _load_sp500_symbols(self) -> List[str]:
        """Load S&P 500 symbols."""
        # Check if stock_symbols.txt exists
        symbols_file = Path("stock_symbols.txt")
        if symbols_file.exists():
            return self._load_symbols_from_file(str(symbols_file))
        
        # Fallback to major tech stocks
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "ADBE", "CRM", "ORCL", "INTC", "AMD", "PYPL", "UBER", "ZOOM",
            "SHOP", "SQ", "TWTR", "SNAP", "ROKU", "PINS", "DOCU", "ZM"
        ]
    
    def _load_symbols_from_file(self, filepath: str) -> List[str]:
        """Load symbols from text file."""
        try:
            with open(filepath, 'r') as f:
                symbols = [line.strip().upper() for line in f if line.strip()]
            return symbols
        except Exception as e:
            logger.error(f"Error loading symbols from {filepath}: {e}")
            return []
    
    def scan_batch_parallel(self, symbols: List[str], period: str = "1y",
                          pattern_types: Optional[List[str]] = None,
                          chunk_size: int = 10) -> Dict[str, Dict]:
        """
        Scan multiple symbols in parallel with Apple Silicon optimization.
        
        Args:
            symbols: List of stock symbols to scan
            period: Time period for analysis
            pattern_types: Pattern types to detect
            chunk_size: Number of symbols per processing chunk
            
        Returns:
            Dictionary with scan results
        """
        self.stats["start_time"] = datetime.now()
        self.stats["total_symbols"] = len(symbols)
        
        logger.info(f"Starting batch scan of {len(symbols)} symbols with {self.max_workers} workers")
        
        # Split symbols into chunks for efficient processing
        symbol_chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
        
        results = {}
        
        # Use ProcessPoolExecutor for CPU-intensive pattern detection
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    self._process_symbol_chunk, 
                    chunk, period, pattern_types
                ): chunk for chunk in symbol_chunks
            }
            
            # Collect results as they complete
            completed_chunks = 0
            for future in future_to_chunk:
                try:
                    chunk_results = future.result(timeout=300)  # 5 minute timeout per chunk
                    results.update(chunk_results)
                    
                    completed_chunks += 1
                    progress = (completed_chunks / len(symbol_chunks)) * 100
                    logger.info(f"Progress: {progress:.1f}% ({completed_chunks}/{len(symbol_chunks)} chunks)")
                    
                except Exception as e:
                    chunk = future_to_chunk[future]
                    logger.error(f"Error processing chunk {chunk}: {e}")
                    self.stats["failed_scans"] += len(chunk)
        
        self.stats["end_time"] = datetime.now()
        self.stats["successful_scans"] = len(results)
        self.stats["total_patterns"] = sum(
            len(patterns) for symbol_results in results.values() 
            for patterns in symbol_results.values()
        )
        
        logger.info(f"Batch scan completed: {len(results)} symbols processed, "
                   f"{self.stats['total_patterns']} patterns found")
        
        return results
    
    def _process_symbol_chunk(self, symbols: List[str], period: str,
                            pattern_types: Optional[List[str]]) -> Dict[str, Dict]:
        """
        Process a chunk of symbols (runs in separate process).
        
        Args:
            symbols: Chunk of symbols to process
            period: Time period
            pattern_types: Pattern types to detect
            
        Returns:
            Results for the chunk
        """
        # Create new engine instance for this process
        engine = PatternEngine(self.config)
        chunk_results = {}
        
        for symbol in symbols:
            try:
                symbol_results = engine.detect_patterns_single_stock(
                    symbol=symbol,
                    period=period,
                    pattern_types=pattern_types
                )
                
                if symbol_results:
                    chunk_results[symbol] = symbol_results
                    
            except Exception as e:
                logger.warning(f"Failed to process {symbol}: {e}")
        
        return chunk_results
    
    async def scan_batch_async(self, symbols: List[str], period: str = "1y",
                             pattern_types: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Asynchronous batch scanning for I/O intensive operations.
        
        Args:
            symbols: List of stock symbols
            period: Time period
            pattern_types: Pattern types to detect
            
        Returns:
            Scan results
        """
        self.stats["start_time"] = datetime.now()
        self.stats["total_symbols"] = len(symbols)
        
        logger.info(f"Starting async batch scan of {len(symbols)} symbols")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def scan_symbol(symbol: str) -> Tuple[str, Optional[Dict]]:
            async with semaphore:
                try:
                    # Run pattern detection in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        self.pattern_engine.detect_patterns_single_stock,
                        symbol, period, pattern_types
                    )
                    return symbol, result
                except Exception as e:
                    logger.warning(f"Async scan failed for {symbol}: {e}")
                    return symbol, None
        
        # Execute all scans concurrently
        tasks = [scan_symbol(symbol) for symbol in symbols]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for item in results_list:
            if isinstance(item, Exception):
                logger.error(f"Async scan exception: {item}")
                continue
            
            symbol, symbol_results = item
            if symbol_results:
                results[symbol] = symbol_results
        
        self.stats["end_time"] = datetime.now()
        self.stats["successful_scans"] = len(results)
        self.stats["total_patterns"] = sum(
            len(patterns) for symbol_results in results.values() 
            for patterns in symbol_results.values()
        )
        
        logger.info(f"Async batch scan completed: {len(results)} symbols processed")
        return results
    
    def export_results(self, results: Dict[str, Dict], output_format: str = "json",
                      output_file: Optional[str] = None) -> str:
        """
        Export scan results to file.
        
        Args:
            results: Scan results to export
            output_format: Export format ('json', 'csv', 'excel')
            output_file: Output filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"pattern_scan_results_{timestamp}.{output_format}"
        
        if output_format == "json":
            self._export_json(results, output_file)
        elif output_format == "csv":
            self._export_csv(results, output_file)
        elif output_format == "excel":
            self._export_excel(results, output_file)
        else:
            raise ValueError(f"Unsupported export format: {output_format}")
        
        logger.info(f"Results exported to {output_file}")
        return output_file
    
    def _export_json(self, results: Dict[str, Dict], filename: str):
        """Export results to JSON format."""
        # Convert Pattern objects to dictionaries
        json_results = {}
        
        for symbol, symbol_results in results.items():
            json_results[symbol] = {}
            
            for pattern_type, patterns in symbol_results.items():
                json_results[symbol][pattern_type] = []
                
                for pattern in patterns:
                    pattern_dict = {
                        "type": pattern.type,
                        "symbol": pattern.symbol,
                        "confidence": pattern.confidence,
                        "combined_score": pattern.combined_score,
                        "entry_price": pattern.entry_price,
                        "target_price": pattern.target_price,
                        "stop_loss": pattern.stop_loss,
                        "risk_reward_ratio": pattern.risk_reward_ratio,
                        "volume_confirmation": pattern.volume_confirmation,
                        "status": pattern.status,
                        "formation_start": pattern.formation_start.isoformat(),
                        "formation_end": pattern.formation_end.isoformat(),
                        "breakout_date": pattern.breakout_date.isoformat() if pattern.breakout_date else None,
                        "pattern_height": pattern.pattern_height,
                        "duration_days": pattern.duration_days,
                        "is_bullish": pattern.is_bullish,
                        "is_bearish": pattern.is_bearish,
                        "potential_profit_percent": pattern.potential_profit_percent
                    }
                    json_results[symbol][pattern_type].append(pattern_dict)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def _export_csv(self, results: Dict[str, Dict], filename: str):
        """Export results to CSV format."""
        rows = []
        
        for symbol, symbol_results in results.items():
            for pattern_type, patterns in symbol_results.items():
                for pattern in patterns:
                    row = {
                        "Symbol": pattern.symbol,
                        "Pattern_Type": pattern.type,
                        "Confidence": pattern.confidence,
                        "Combined_Score": pattern.combined_score,
                        "Entry_Price": pattern.entry_price,
                        "Target_Price": pattern.target_price,
                        "Stop_Loss": pattern.stop_loss,
                        "Risk_Reward_Ratio": pattern.risk_reward_ratio,
                        "Volume_Confirmation": pattern.volume_confirmation,
                        "Status": pattern.status,
                        "Formation_Start": pattern.formation_start.date(),
                        "Formation_End": pattern.formation_end.date(),
                        "Breakout_Date": pattern.breakout_date.date() if pattern.breakout_date else None,
                        "Pattern_Height": pattern.pattern_height,
                        "Duration_Days": pattern.duration_days,
                        "Direction": "Bullish" if pattern.is_bullish else "Bearish",
                        "Potential_Profit_Percent": pattern.potential_profit_percent
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    
    def _export_excel(self, results: Dict[str, Dict], filename: str):
        """Export results to Excel format with multiple sheets."""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = self.generate_summary_report(results)
            summary_df = pd.DataFrame([summary_data])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet
            rows = []
            for symbol, symbol_results in results.items():
                for pattern_type, patterns in symbol_results.items():
                    for pattern in patterns:
                        row = {
                            "Symbol": pattern.symbol,
                            "Pattern_Type": pattern.type,
                            "Confidence": pattern.confidence,
                            "Combined_Score": pattern.combined_score,
                            "Entry_Price": pattern.entry_price,
                            "Target_Price": pattern.target_price,
                            "Stop_Loss": pattern.stop_loss,
                            "Risk_Reward_Ratio": pattern.risk_reward_ratio,
                            "Volume_Confirmation": pattern.volume_confirmation,
                            "Status": pattern.status,
                            "Formation_Start": pattern.formation_start.date(),
                            "Formation_End": pattern.formation_end.date(),
                            "Breakout_Date": pattern.breakout_date.date() if pattern.breakout_date else None,
                            "Duration_Days": pattern.duration_days,
                            "Direction": "Bullish" if pattern.is_bullish else "Bearish",
                            "Potential_Profit_Percent": pattern.potential_profit_percent
                        }
                        rows.append(row)
            
            if rows:
                details_df = pd.DataFrame(rows)
                details_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Pattern type breakdown
            pattern_counts = {}
            for symbol_results in results.values():
                for pattern_type, patterns in symbol_results.items():
                    pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + len(patterns)
            
            if pattern_counts:
                breakdown_df = pd.DataFrame(list(pattern_counts.items()), 
                                          columns=['Pattern_Type', 'Count'])
                breakdown_df.to_excel(writer, sheet_name='Pattern_Breakdown', index=False)
    
    def generate_summary_report(self, results: Dict[str, Dict]) -> Dict:
        """Generate summary statistics for the scan results."""
        total_symbols_scanned = self.stats["total_symbols"]
        symbols_with_patterns = len(results)
        total_patterns = sum(len(patterns) for symbol_results in results.values() 
                           for patterns in symbol_results.values())
        
        # Calculate pattern type distribution
        pattern_counts = {}
        confidence_scores = []
        risk_rewards = []
        
        for symbol_results in results.values():
            for pattern_type, patterns in symbol_results.items():
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + len(patterns)
                for pattern in patterns:
                    confidence_scores.append(pattern.confidence)
                    if pattern.risk_reward_ratio > 0:
                        risk_rewards.append(pattern.risk_reward_ratio)
        
        # Calculate timing statistics
        total_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        summary = {
            "scan_timestamp": self.stats["start_time"].isoformat(),
            "total_symbols_scanned": total_symbols_scanned,
            "symbols_with_patterns": symbols_with_patterns,
            "success_rate": (symbols_with_patterns / total_symbols_scanned) * 100 if total_symbols_scanned > 0 else 0,
            "total_patterns_found": total_patterns,
            "avg_patterns_per_symbol": total_patterns / symbols_with_patterns if symbols_with_patterns > 0 else 0,
            "pattern_type_distribution": pattern_counts,
            "avg_confidence_score": np.mean(confidence_scores) if confidence_scores else 0,
            "avg_risk_reward_ratio": np.mean(risk_rewards) if risk_rewards else 0,
            "total_processing_time_seconds": total_time,
            "avg_time_per_symbol": total_time / total_symbols_scanned if total_symbols_scanned > 0 else 0,
            "max_workers_used": self.max_workers
        }
        
        return summary
    
    def print_summary_report(self, results: Dict[str, Dict]):
        """Print a formatted summary report to console."""
        summary = self.generate_summary_report(results)
        
        print("\n" + "="*60)
        print("           BATCH PATTERN SCAN SUMMARY")
        print("="*60)
        print(f"Scan Time: {summary['scan_timestamp']}")
        print(f"Total Symbols Scanned: {summary['total_symbols_scanned']}")
        print(f"Symbols with Patterns: {summary['symbols_with_patterns']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Patterns Found: {summary['total_patterns_found']}")
        print(f"Avg Patterns per Symbol: {summary['avg_patterns_per_symbol']:.1f}")
        print(f"Avg Confidence Score: {summary['avg_confidence_score']:.3f}")
        print(f"Avg Risk/Reward Ratio: {summary['avg_risk_reward_ratio']:.2f}")
        print(f"Processing Time: {summary['total_processing_time_seconds']:.1f}s")
        print(f"Time per Symbol: {summary['avg_time_per_symbol']:.2f}s")
        print(f"Workers Used: {summary['max_workers_used']}")
        
        print("\nPattern Type Distribution:")
        for pattern_type, count in summary['pattern_type_distribution'].items():
            print(f"  {pattern_type}: {count}")
        
        print("="*60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Batch Pattern Scanner")
    parser.add_argument("--symbols", required=True, 
                       help="Symbol source: 'nifty50', 'sp500', file path, or comma-separated list")
    parser.add_argument("--period", default="1y", 
                       help="Time period for analysis (default: 1y)")
    parser.add_argument("--patterns", nargs="+", 
                       default=["Head and Shoulders", "Double Bottom", "Cup and Handle"],
                       help="Pattern types to detect")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--format", choices=["json", "csv", "excel"], default="json",
                       help="Output format (default: json)")
    parser.add_argument("--workers", type=int, help="Number of worker processes")
    parser.add_argument("--async", action="store_true", help="Use async processing")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                       help="Minimum confidence threshold")
    
    args = parser.parse_args()
    
    # Create configuration
    config = PatternConfig(
        min_confidence=args.min_confidence,
        enable_parallel_processing=True
    )
    
    # Initialize scanner
    scanner = BatchPatternScanner(config, max_workers=args.workers)
    
    # Load symbols
    symbols = scanner.load_symbol_list(args.symbols)
    if not symbols:
        print("Error: No symbols loaded")
        return
    
    # Run scan
    if args.async:
        results = asyncio.run(scanner.scan_batch_async(symbols, args.period, args.patterns))
    else:
        results = scanner.scan_batch_parallel(symbols, args.period, args.patterns)
    
    # Export results
    if results:
        output_file = scanner.export_results(results, args.format, args.output)
        scanner.print_summary_report(results)
        print(f"\nResults exported to: {output_file}")
    else:
        print("No patterns detected")


if __name__ == "__main__":
    main()