# 📈 Stock Pattern Scanner - Consolidated Edition

Lightweight stock pattern detection tool with traditional algorithms and scikit-learn ML models.

## 🚀 Features

### Pattern Detection
- **Head & Shoulders**: Bearish reversal pattern
- **Double Bottom**: Bullish reversal pattern  
- **Double Top**: Bearish reversal pattern
- **Cup & Handle**: Bullish continuation pattern

### Detection Methods
- **Traditional**: Rule-based algorithms with geometric pattern recognition
- **Lightweight ML**: Random Forest, SVM, and Isolation Forest models
- **Hybrid**: Combines both approaches for maximum accuracy

### Key Advantages
- **Fast & Lightweight**: No heavy dependencies like TensorFlow
- **Single File**: Everything in one consolidated script
- **Real-time Data**: Yahoo Finance integration
- **Interactive Charts**: Plotly visualizations
- **Easy Setup**: Minimal requirements

## 🛠️ Quick Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
streamlit run stock_scanner.py
```

That's it! The app will open in your browser.

## 📁 Project Structure (Simplified)

```
Trend_Analysis/
├── stock_scanner.py     # Complete application (all-in-one)
├── requirements.txt     # Minimal dependencies
└── README.md           # This file
```

## 🎯 Usage

### Quick Start
1. **Select Date Range**: Choose analysis period (default: 1 year)
2. **Pick Stocks**: Enter symbols or use predefined lists
3. **Choose Patterns**: Select which patterns to detect
4. **Configure ML**: Enable lightweight ML detection
5. **Start Scanning**: Click 🚀 and view results

### Example Workflow
```
Date Range: Last 6 months
Stocks: AAPL, MSFT, GOOGL, TSLA
Patterns: All selected
ML: Enabled (60% confidence)
```

## 🤖 Lightweight ML Models

### Random Forest
- Fast ensemble method
- Good for pattern classification
- No GPU required

### SVM (Support Vector Machine)
- Excellent pattern boundary detection
- Probability estimates for confidence
- Robust to overfitting

### Isolation Forest
- Anomaly detection
- Flags unusual patterns
- Helps filter false positives

## 📊 Supported Patterns

| Pattern | Type | Signal | Target Calculation |
|---------|------|--------|-------------------|
| Head & Shoulders 📉 | Bearish | Break below neckline | Pattern height down |
| Double Bottom 📈 | Bullish | Break above neckline | Pattern height up |
| Double Top 📉 | Bearish | Break below neckline | Pattern height down |
| Cup & Handle ☕ | Bullish | Break above rim | Cup depth up |

## ⚙️ Configuration

All settings are built into the UI:
- **Pattern Parameters**: Automatically optimized
- **ML Confidence**: Adjustable threshold (30%-90%)
- **Stock Lists**: Individual entry or predefined lists
- **Date Ranges**: Flexible selection

## 🔧 Technical Details

### Lightweight Dependencies
- **Streamlit**: Web interface
- **scikit-learn**: ML models (no TensorFlow!)
- **yfinance**: Stock data
- **Plotly**: Interactive charts
- **pandas/numpy**: Data processing
- **scipy**: Signal processing

### Performance Features
- **Cached Models**: ML models stored in session
- **Efficient Algorithms**: Optimized pattern detection
- **Batch Processing**: Multiple stocks at once
- **Memory Efficient**: No heavy neural networks

## 📈 Results Interpretation

### Confidence Levels
- **🟢 High (70%+)**: Strong pattern signals
- **🟡 Medium (50-70%)**: Moderate confidence
- **🔴 Low (<50%)**: Filtered out by default

### Detection Methods
- **Traditional**: Rule-based geometric detection
- **ML**: Machine learning classification
- **Hybrid**: Best of both approaches

## 🚨 Important Notes

- **Educational Purpose**: Not financial advice
- **Risk Warning**: Always do your own research
- **Data Accuracy**: Depends on Yahoo Finance availability
- **Pattern Reliability**: Past patterns don't guarantee future results

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Data Issues**: Check internet connection
3. **Slow Performance**: Reduce number of stocks or date range
4. **Memory Issues**: Disable ML detection for large scans

### Performance Tips
- Use shorter date ranges for faster scanning
- Start with fewer stocks to test
- Enable ML only when needed
- Check stock symbols are valid

## 🎉 What's New in Consolidated Edition

✅ **Single File**: Everything in `stock_scanner.py`  
✅ **No TensorFlow**: Removed heavy dependencies  
✅ **Faster Startup**: Lightweight ML models  
✅ **Simplified Setup**: Minimal requirements  
✅ **Better Performance**: Optimized algorithms  
✅ **Cleaner Code**: Consolidated and organized  

## 📞 Support

- Check the code comments for detailed explanations
- Adjust confidence thresholds if results seem off
- Try different date ranges for better patterns
- Remember: This is for learning and research only!

Happy pattern hunting! 📈🔍