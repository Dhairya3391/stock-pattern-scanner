"""
Streamlit Web Application for Advanced Pattern Scanner.

Interactive web interface with Plotly charts showing detected patterns
with confidence scores, target prices, and risk assessment.
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

# Import our pattern detection components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import PatternConfig
from core.pattern_engine import PatternEngine
from core.pattern_scorer import PatternScorer
from core.data_manager import DataManager

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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .bullish-pattern {
        border-left-color: #2ca02c;
    }
    .bearish-pattern {
        border-left-color: #d62728;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitPatternApp:
    """Main Streamlit application class for pattern detection interface."""
    
    def __init__(self):
        """Initialize the Streamlit application."""
        self.initialize_session_state()
        self.config = self.load_configuration()
        self.pattern_engine = PatternEngine(self.config)
        self.pattern_scorer = PatternScorer(self.config)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = {}
        if 'selected_symbols' not in st.session_state:
            st.session_state.selected_symbols = ['AAPL']
        if 'last_scan_time' not in st.session_state:
            st.session_state.last_scan_time = None
    
    def load_configuration(self) -> PatternConfig:
        """Load pattern detection configuration from sidebar."""
        st.sidebar.header("⚙️ Configuration")
        
        # Detection thresholds
        st.sidebar.subheader("Detection Thresholds")
        min_confidence = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.05)
        min_combined_score = st.sidebar.slider("Minimum Combined Score", 0.0, 1.0, 0.65, 0.05)
        
        # Volume settings
        st.sidebar.subheader("Volume Analysis")
        volume_confirmation = st.sidebar.checkbox("Require Volume Confirmation", True)
        min_volume_ratio = st.sidebar.slider("Minimum Volume Ratio", 1.0, 3.0, 1.2, 0.1)
        
        # Pattern duration
        st.sidebar.subheader("Pattern Duration")
        min_duration = st.sidebar.number_input("Minimum Duration (days)", 10, 100, 20)
        max_duration = st.sidebar.number_input("Maximum Duration (days)", 50, 500, 200)
        
        # Risk management
        st.sidebar.subheader("Risk Management")
        max_risk_percent = st.sidebar.slider("Maximum Risk per Trade (%)", 0.5, 5.0, 2.0, 0.1)
        min_risk_reward = st.sidebar.slider("Minimum Risk/Reward Ratio", 1.0, 5.0, 1.5, 0.1)
        
        return PatternConfig(
            min_confidence=min_confidence,
            min_combined_score=min_combined_score,
            volume_confirmation_required=volume_confirmation,
            min_volume_ratio=min_volume_ratio,
            min_pattern_duration=int(min_duration),
            max_pattern_duration=int(max_duration),
            max_risk_percent=max_risk_percent,
            min_risk_reward_ratio=min_risk_reward
        )
    
    def render_header(self):
        """Render the main application header."""
        st.markdown('<h1 class="main-header">📈 Advanced Stock Pattern Scanner</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Detect technical patterns using advanced algorithms and ML validation**
        
        This application combines traditional technical analysis with machine learning 
        to identify high-probability trading patterns with confidence scores and risk assessment.
        """)
    
    def render_input_section(self):
        """Render the stock symbol input section."""
        st.header("🎯 Stock Selection")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Stock symbol input
            symbol_input = st.text_input(
                "Enter stock symbols (comma-separated)",
                value=",".join(st.session_state.selected_symbols),
                help="Enter stock symbols like AAPL, MSFT, GOOGL"
            )
            
            # Parse symbols
            if symbol_input:
                symbols = [s.strip().upper() for s in symbol_input.split(",") if s.strip()]
                st.session_state.selected_symbols = symbols
        
        with col2:
            # Time period selection
            period = st.selectbox(
                "Time Period",
                ["6mo", "1y", "2y", "5y"],
                index=1,
                help="Historical data period for analysis"
            )
        
        with col3:
            # Pattern type selection
            pattern_types = st.multiselect(
                "Pattern Types",
                ["Head and Shoulders", "Double Bottom", "Cup and Handle"],
                default=["Head and Shoulders", "Double Bottom", "Cup and Handle"],
                help="Select pattern types to detect"
            )
        
        # Scan button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔍 Scan for Patterns", type="primary", use_container_width=True):
                self.run_pattern_detection(st.session_state.selected_symbols, period, pattern_types)
        
        return period, pattern_types
    
    def run_pattern_detection(self, symbols: List[str], period: str, pattern_types: List[str]):
        """Run pattern detection for selected symbols."""
        if not symbols:
            st.error("Please enter at least one stock symbol")
            return
        
        if not pattern_types:
            st.error("Please select at least one pattern type")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing pattern detection...")
            progress_bar.progress(10)
            
            # Run detection
            status_text.text(f"🔄 Scanning {len(symbols)} symbols for patterns...")
            progress_bar.progress(30)
            
            results = self.pattern_engine.detect_patterns_batch(
                symbols=symbols,
                period=period,
                pattern_types=pattern_types
            )
            
            progress_bar.progress(80)
            status_text.text("🔄 Processing results...")
            
            # Store results
            st.session_state.detection_results = results
            st.session_state.last_scan_time = datetime.now()
            
            progress_bar.progress(100)
            status_text.text("✅ Pattern detection completed!")
            
            # Clear progress indicators after a short delay
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            if results:
                st.success(f"Found patterns in {len(results)} symbols")
            else:
                st.warning("No patterns detected with current settings")
        
        except Exception as e:
            st.error(f"Error during pattern detection: {str(e)}")
            logger.error(f"Pattern detection error: {e}")
    
    def render_results_section(self):
        """Render the pattern detection results."""
        if not st.session_state.detection_results:
            st.info("👆 Enter stock symbols and click 'Scan for Patterns' to get started")
            return
        
        st.header("📊 Detection Results")
        
        # Results summary
        self.render_results_summary()
        
        # Pattern details
        self.render_pattern_details()
        
        # Charts
        self.render_pattern_charts()
    
    def render_results_summary(self):
        """Render summary statistics for detection results."""
        results = st.session_state.detection_results
        
        # Calculate summary statistics
        total_symbols = len(results)
        total_patterns = sum(len(patterns) for symbol_results in results.values() 
                           for patterns in symbol_results.values())
        
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
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Symbols Scanned", total_symbols)
        
        with col2:
            st.metric("Patterns Found", total_patterns)
        
        with col3:
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col4:
            avg_risk_reward = np.mean(risk_rewards) if risk_rewards else 0
            st.metric("Avg Risk/Reward", f"{avg_risk_reward:.2f}")
        
        # Pattern type breakdown
        if pattern_counts:
            st.subheader("Pattern Distribution")
            
            # Create pie chart
            fig = px.pie(
                values=list(pattern_counts.values()),
                names=list(pattern_counts.keys()),
                title="Detected Patterns by Type"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_pattern_details(self):
        """Render detailed pattern information."""
        results = st.session_state.detection_results
        
        st.subheader("📋 Pattern Details")
        
        # Create a list of all patterns for display
        all_patterns = []
        for symbol, symbol_results in results.items():
            for pattern_type, patterns in symbol_results.items():
                for pattern in patterns:
                    all_patterns.append({
                        'Symbol': symbol,
                        'Pattern': pattern_type,
                        'Confidence': f"{pattern.confidence:.3f}",
                        'Combined Score': f"{pattern.combined_score:.3f}",
                        'Entry Price': f"${pattern.entry_price:.2f}",
                        'Target Price': f"${pattern.target_price:.2f}",
                        'Stop Loss': f"${pattern.stop_loss:.2f}",
                        'Risk/Reward': f"{pattern.risk_reward_ratio:.2f}",
                        'Volume Confirmed': "✅" if pattern.volume_confirmation else "❌",
                        'Status': pattern.status.title(),
                        'Direction': "🟢 Bullish" if pattern.is_bullish else "🔴 Bearish"
                    })
        
        if all_patterns:
            # Convert to DataFrame for display
            df = pd.DataFrame(all_patterns)
            
            # Add filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                symbol_filter = st.multiselect(
                    "Filter by Symbol",
                    options=df['Symbol'].unique(),
                    default=df['Symbol'].unique()
                )
            
            with col2:
                pattern_filter = st.multiselect(
                    "Filter by Pattern",
                    options=df['Pattern'].unique(),
                    default=df['Pattern'].unique()
                )
            
            with col3:
                min_confidence_filter = st.slider(
                    "Minimum Confidence",
                    0.0, 1.0, 0.0, 0.05,
                    key="detail_confidence_filter"
                )
            
            # Apply filters
            filtered_df = df[
                (df['Symbol'].isin(symbol_filter)) &
                (df['Pattern'].isin(pattern_filter)) &
                (df['Confidence'].astype(float) >= min_confidence_filter)
            ]
            
            # Display filtered results
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Export functionality
            if st.button("📥 Export Results to CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"pattern_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No patterns to display")
    
    def render_pattern_charts(self):
        """Render interactive charts for detected patterns."""
        results = st.session_state.detection_results
        
        if not results:
            return
        
        st.subheader("📈 Pattern Charts")
        
        # Symbol selection for charting
        symbols_with_patterns = list(results.keys())
        selected_symbol = st.selectbox(
            "Select symbol to chart",
            symbols_with_patterns,
            help="Choose a symbol to view its pattern chart"
        )
        
        if selected_symbol and selected_symbol in results:
            self.create_pattern_chart(selected_symbol, results[selected_symbol])
    
    def create_pattern_chart(self, symbol: str, symbol_results: Dict):
        """Create an interactive chart for a specific symbol's patterns."""
        try:
            # Fetch market data for charting
            df = self.pattern_engine.data_manager.fetch_stock_data(symbol, "1y")
            if df is None or df.empty:
                st.error(f"Could not fetch data for {symbol}")
                return
            
            # Create subplot with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} Price Chart with Patterns', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=symbol
                ),
                row=1, col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='rgba(158,202,225,0.8)'
                ),
                row=2, col=1
            )
            
            # Add pattern annotations
            colors = {
                'Head and Shoulders': 'red',
                'Double Bottom': 'green',
                'Cup and Handle': 'blue'
            }
            
            for pattern_type, patterns in symbol_results.items():
                color = colors.get(pattern_type, 'purple')
                
                for i, pattern in enumerate(patterns):
                    # Add pattern key points
                    if pattern.key_points:
                        x_coords = []
                        y_coords = []
                        
                        for point_idx, price in pattern.key_points:
                            if point_idx < len(df):
                                x_coords.append(df.index[point_idx])
                                y_coords.append(price)
                        
                        if x_coords and y_coords:
                            # Add pattern line
                            fig.add_trace(
                                go.Scatter(
                                    x=x_coords,
                                    y=y_coords,
                                    mode='lines+markers',
                                    name=f'{pattern_type} #{i+1}',
                                    line=dict(color=color, width=2),
                                    marker=dict(size=8, color=color)
                                ),
                                row=1, col=1
                            )
                            
                            # Add entry, target, and stop-loss lines
                            if pattern.breakout_date and pattern.breakout_date <= df.index.max():
                                breakout_x = pattern.breakout_date
                                
                                # Entry price line
                                fig.add_hline(
                                    y=pattern.entry_price,
                                    line_dash="dash",
                                    line_color="blue",
                                    annotation_text=f"Entry: ${pattern.entry_price:.2f}",
                                    row=1
                                )
                                
                                # Target price line
                                fig.add_hline(
                                    y=pattern.target_price,
                                    line_dash="dash",
                                    line_color="green",
                                    annotation_text=f"Target: ${pattern.target_price:.2f}",
                                    row=1
                                )
                                
                                # Stop loss line
                                fig.add_hline(
                                    y=pattern.stop_loss,
                                    line_dash="dash",
                                    line_color="red",
                                    annotation_text=f"Stop: ${pattern.stop_loss:.2f}",
                                    row=1
                                )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Pattern Analysis',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Pattern information cards
            self.render_pattern_cards(symbol_results)
            
        except Exception as e:
            st.error(f"Error creating chart for {symbol}: {str(e)}")
            logger.error(f"Chart creation error: {e}")
    
    def render_pattern_cards(self, symbol_results: Dict):
        """Render pattern information cards."""
        for pattern_type, patterns in symbol_results.items():
            for i, pattern in enumerate(patterns):
                # Determine card style based on pattern direction
                card_class = "bullish-pattern" if pattern.is_bullish else "bearish-pattern"
                
                with st.container():
                    st.markdown(f"""
                    <div class="pattern-card {card_class}">
                        <h4>{pattern_type} #{i+1} - {pattern.status.title()}</h4>
                        <p><strong>Direction:</strong> {'🟢 Bullish' if pattern.is_bullish else '🔴 Bearish'}</p>
                        <p><strong>Confidence:</strong> {pattern.confidence:.3f} | 
                           <strong>Combined Score:</strong> {pattern.combined_score:.3f}</p>
                        <p><strong>Entry:</strong> ${pattern.entry_price:.2f} | 
                           <strong>Target:</strong> ${pattern.target_price:.2f} | 
                           <strong>Stop:</strong> ${pattern.stop_loss:.2f}</p>
                        <p><strong>Risk/Reward:</strong> {pattern.risk_reward_ratio:.2f} | 
                           <strong>Potential Profit:</strong> {pattern.potential_profit_percent:.1f}%</p>
                        <p><strong>Volume Confirmed:</strong> {'✅ Yes' if pattern.volume_confirmation else '❌ No'}</p>
                        <p><strong>Formation Period:</strong> {pattern.formation_start.strftime('%Y-%m-%d')} to {pattern.formation_end.strftime('%Y-%m-%d')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_footer(self):
        """Render application footer with statistics and information."""
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Engine Statistics")
            stats = self.pattern_engine.get_engine_stats()
            
            if stats.get('total_scans', 0) > 0:
                st.write(f"**Total Scans:** {stats.get('total_scans', 0)}")
                st.write(f"**Success Rate:** {stats.get('success_rate', 0):.1%}")
                st.write(f"**Patterns Detected:** {stats.get('patterns_detected', 0)}")
                st.write(f"**Validation Rate:** {stats.get('validation_rate', 0):.1%}")
            else:
                st.write("No scans performed yet")
        
        with col2:
            st.subheader("⚡ Performance")
            if st.session_state.last_scan_time:
                st.write(f"**Last Scan:** {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if stats.get('avg_detection_time'):
                st.write(f"**Avg Detection Time:** {stats['avg_detection_time']:.2f}s")
            
            cache_stats = stats.get('cache_stats', {})
            if cache_stats.get('size', 0) > 0:
                st.write(f"**Cache Size:** {cache_stats['size']} items")
        
        with col3:
            st.subheader("ℹ️ About")
            st.write("""
            **Advanced Pattern Scanner** combines traditional technical analysis 
            with machine learning for accurate pattern detection.
            
            - **Patterns:** Head & Shoulders, Double Bottom, Cup & Handle
            - **Validation:** Hybrid ML + Traditional algorithms
            - **Optimization:** Apple Silicon optimized
            """)
        
        # Cache management
        if st.button("🗑️ Clear Cache"):
            self.pattern_engine.clear_cache()
            st.success("Cache cleared successfully!")
    
    def run(self):
        """Run the main Streamlit application."""
        try:
            self.render_header()
            
            # Main content
            period, pattern_types = self.render_input_section()
            
            st.markdown("---")
            
            self.render_results_section()
            
            st.markdown("---")
            
            self.render_footer()
            
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            logger.error(f"Streamlit app error: {e}")


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitPatternApp()
    app.run()


if __name__ == "__main__":
    main()