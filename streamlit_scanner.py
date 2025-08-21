#!/usr/bin/env python3
"""
Streamlit Web Application for Advanced Pattern Scanner.

This is a standalone version that works around Python import issues.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import sys
import os
import yfinance as yf

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import our pattern detection components
try:
    from advanced_pattern_scanner.core.models import PatternConfig, MarketData
    from advanced_pattern_scanner.patterns.head_shoulders import HeadShouldersDetector
    from advanced_pattern_scanner.patterns.double_bottom import DoubleBottomDetector
    from advanced_pattern_scanner.patterns.cup_handle import CupHandleDetector
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please make sure you're running this from the correct directory.")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Pattern Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .pattern-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def get_confidence_color(confidence):
    """Return CSS class based on confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def fetch_stock_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Fetch stock data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data if not data.empty else None
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def convert_to_market_data(symbol: str, data: pd.DataFrame) -> MarketData:
    """Convert pandas DataFrame to MarketData object."""
    ohlcv_data = np.column_stack([
        data['Open'].values,
        data['High'].values,
        data['Low'].values,
        data['Close'].values,
        data['Volume'].values
    ])
    
    return MarketData(
        symbol=symbol,
        timeframe="1d",
        data=ohlcv_data,
        timestamps=[dt.to_pydatetime() for dt in data.index]
    )

def detect_patterns(market_data: MarketData, config: PatternConfig):
    """Detect all patterns for the given market data."""
    patterns = {}
    
    try:
        # Initialize detectors
        hs_detector = HeadShouldersDetector(config)
        db_detector = DoubleBottomDetector(config)
        ch_detector = CupHandleDetector(config)
        
        # Detect patterns
        patterns["Head and Shoulders"] = hs_detector.detect_pattern(market_data)
        patterns["Double Bottom"] = db_detector.detect_pattern(market_data)
        patterns["Cup and Handle"] = ch_detector.detect_pattern(market_data)
        
    except Exception as e:
        st.error(f"Error detecting patterns: {e}")
        
    return patterns

def create_price_chart(data: pd.DataFrame, symbol: str, patterns: Dict):
    """Create an interactive price chart with patterns."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{symbol} Price Chart with Patterns", "Volume"]
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color="rgba(158,202,225,0.8)"
        ),
        row=2, col=1
    )
    
    # Add pattern annotations
    colors = ["red", "green", "blue", "orange", "purple"]
    color_idx = 0
    
    for pattern_type, pattern_list in patterns.items():
        if pattern_list:
            for pattern in pattern_list:
                if hasattr(pattern, 'key_points') and pattern.key_points:
                    # Add pattern points
                    x_points = []
                    y_points = []
                    
                    for idx, price in pattern.key_points:
                        if idx < len(data):
                            x_points.append(data.index[idx])
                            y_points.append(price)
                    
                    if x_points:
                        fig.add_trace(
                            go.Scatter(
                                x=x_points,
                                y=y_points,
                                mode="markers+lines",
                                name=f"{pattern_type} ({pattern.confidence:.1%})",
                                marker=dict(size=8, color=colors[color_idx % len(colors)]),
                                line=dict(width=2, color=colors[color_idx % len(colors)])
                            ),
                            row=1, col=1
                        )
                        color_idx += 1
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} - Pattern Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Advanced Stock Pattern Scanner</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Enter a stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
    ).upper()
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Time Period",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
        help="Select the time period for analysis"
    )
    
    # Pattern detection settings
    st.sidebar.subheader("🎯 Detection Settings")
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Minimum confidence score for pattern detection"
    )
    
    volume_confirmation = st.sidebar.checkbox(
        "Require Volume Confirmation",
        value=True,
        help="Only show patterns with volume confirmation"
    )
    
    # Pattern type selection
    st.sidebar.subheader("📊 Pattern Types")
    show_hs = st.sidebar.checkbox("Head and Shoulders", value=True)
    show_db = st.sidebar.checkbox("Double Bottom", value=True)
    show_ch = st.sidebar.checkbox("Cup and Handle", value=True)
    
    # Analyze button
    if st.sidebar.button("🔍 Analyze Patterns", type="primary"):
        
        if not symbol:
            st.error("Please enter a stock symbol")
            return
        
        # Show loading message
        with st.spinner(f"Analyzing {symbol} for patterns..."):
            
            # Fetch data
            data = fetch_stock_data(symbol, period)
            
            if data is None:
                st.error(f"Could not fetch data for {symbol}")
                return
            
            # Convert to MarketData
            market_data = convert_to_market_data(symbol, data)
            
            # Configuration
            config = PatternConfig(
                min_confidence=min_confidence,
                volume_confirmation_required=volume_confirmation,
                min_volume_ratio=1.2
            )
            
            # Detect patterns
            patterns = detect_patterns(market_data, config)
            
            # Filter patterns based on user selection
            filtered_patterns = {}
            if show_hs:
                filtered_patterns["Head and Shoulders"] = patterns.get("Head and Shoulders", [])
            if show_db:
                filtered_patterns["Double Bottom"] = patterns.get("Double Bottom", [])
            if show_ch:
                filtered_patterns["Cup and Handle"] = patterns.get("Cup and Handle", [])
            
            # Display results
            st.header(f"📊 Analysis Results for {symbol}")
            
            # Show summary
            total_patterns = sum(len(p) for p in filtered_patterns.values())
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patterns", total_patterns)
            with col2:
                st.metric("Time Period", period)
            with col3:
                st.metric("Data Points", len(data))
            with col4:
                current_price = data['Close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
            
            # Display chart
            if total_patterns > 0:
                st.subheader("📈 Price Chart with Detected Patterns")
                chart = create_price_chart(data, symbol, filtered_patterns)
                st.plotly_chart(chart, use_container_width=True)
                
                # Display pattern details
                st.subheader("🎯 Pattern Details")
                
                for pattern_type, pattern_list in filtered_patterns.items():
                    if pattern_list:
                        st.write(f"### {pattern_type} ({len(pattern_list)} found)")
                        
                        for i, pattern in enumerate(pattern_list, 1):
                            confidence_class = get_confidence_color(pattern.confidence)
                            
                            with st.expander(f"Pattern {i} - Confidence: {pattern.confidence:.1%}"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Pattern Details:**")
                                    st.write(f"- Type: {pattern.type}")
                                    st.write(f"- Confidence: {pattern.confidence:.1%}")
                                    if hasattr(pattern, 'target_price'):
                                        st.write(f"- Target Price: ${pattern.target_price:.2f}")
                                    if hasattr(pattern, 'stop_loss'):
                                        st.write(f"- Stop Loss: ${pattern.stop_loss:.2f}")
                                
                                with col2:
                                    st.write("**Trading Info:**")
                                    if hasattr(pattern, 'entry_price'):
                                        st.write(f"- Entry Price: ${pattern.entry_price:.2f}")
                                    if hasattr(pattern, 'risk_reward_ratio'):
                                        st.write(f"- Risk/Reward: {pattern.risk_reward_ratio:.2f}")
                                    if hasattr(pattern, 'formation_start'):
                                        st.write(f"- Formation Start: {pattern.formation_start.strftime('%Y-%m-%d')}")
            else:
                st.info(f"No patterns detected for {symbol} with current settings. Try adjusting the confidence threshold or time period.")
                
                # Still show the price chart
                st.subheader("📈 Price Chart")
                chart = create_price_chart(data, symbol, {})
                st.plotly_chart(chart, use_container_width=True)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This tool detects common stock chart patterns:
    
    - **Head & Shoulders**: Reversal pattern
    - **Double Bottom**: Bullish reversal
    - **Cup & Handle**: Bullish continuation
    
    Higher confidence scores indicate more reliable patterns.
    """)
    
    # Instructions
    if not st.session_state.get('analyzed', False):
        st.info("👈 Enter a stock symbol in the sidebar and click 'Analyze Patterns' to get started!")
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Enter a stock symbol** (e.g., AAPL, GOOGL, MSFT)
        2. **Select time period** for analysis
        3. **Adjust confidence threshold** if needed
        4. **Click 'Analyze Patterns'** to see results
        
        The tool will display interactive charts with detected patterns and provide trading insights.
        """)

if __name__ == "__main__":
    main()
