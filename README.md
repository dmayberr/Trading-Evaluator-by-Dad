# ⚡ Trading Strategy Evaluator

A polished, local Streamlit application for backtesting and optimizing equity and options trading strategies with automated data pulling, flexible parameters, and detailed performance analytics.

---

## Features

### Strategies

**Equity / Directional (7)**
| Strategy | Description |
|---|---|
| SMA Crossover | Fast/slow simple moving average crossover |
| EMA Crossover | Fast/slow exponential moving average crossover |
| RSI Reversal | Buy oversold, sell overbought using RSI |
| MACD Crossover | MACD line vs signal line crossover |
| Bollinger Bands | Mean reversion at upper/lower bands |
| Stochastic Oscillator | %K/%D crossover in oversold/overbought zones |
| Triple EMA | Three-EMA trend alignment system |

**Options (4)**
| Strategy | Description |
|---|---|
| Covered Call | Own stock + sell OTM calls on a rolling basis |
| Bull Put Spread | Sell put spread on RSI oversold signals |
| Iron Condor | Sell call + put spreads during low volatility (BB squeeze) |
| Long Straddle | Buy ATM call + put during BB squeeze, profit from breakout |

### Analysis & Output
- **12 performance metrics**: P&L, return %, win rate, Sharpe, Sortino, Calmar, profit factor, expectancy, max drawdown, avg hold time, consecutive wins/losses, win/loss ratio
- **5 chart types**: Equity curve with drawdown, candlestick with buy/sell signals, P&L distribution, trade scatter with cumulative P&L, monthly returns heatmap
- **Trade log** with filtering, CSV and Excel export
- **Optimization mode** with parameter sweep, heatmap visualization, and automatic best-result backtest

### Risk Management
- Stop loss (fixed %)
- Take profit (fixed %)
- Trailing stop
- Position sizing (% of equity)
- Commission modeling

---

## Installation

### Prerequisites
- **Python 3.9+** (3.10 or 3.11 recommended)
- **pip** (comes with Python)

### Step 1: Download the Project

Copy the entire `trading-evaluator` folder to your preferred location, for example:

```
C:\Users\YourName\Documents\trading-evaluator
```

### Step 2: Open a Terminal

- **Windows**: Open Command Prompt or PowerShell, then `cd` to the project folder:
  ```
  cd C:\Users\YourName\Documents\trading-evaluator
  ```
- **Mac/Linux**: Open Terminal:
  ```
  cd ~/Documents/trading-evaluator
  ```

### Step 3: (Recommended) Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:
- **Windows (Command Prompt)**: `venv\Scripts\activate`
- **Windows (PowerShell)**: `venv\Scripts\Activate.ps1`
- **Mac/Linux**: `source venv/bin/activate`

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: Streamlit, Pandas, NumPy, Plotly, yfinance, openpyxl, and scipy.

---

## Usage

### Launch the App

From the project folder, run:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### Quick Start: Run Your First Backtest

1. **Ticker**: Enter any ticker symbol (e.g., `SPY`, `AAPL`, `VRT`, `CLS`)
2. **Date Range**: Select start and end dates
3. **Interval**: Choose timeframe (`1d` for daily, `1h` for hourly, etc.)
4. **Strategy**: Pick a strategy from the dropdown
5. **Parameters**: Adjust the sliders for the strategy's parameters
6. **Risk Management**: Optionally enable stop loss, take profit, or trailing stop
7. **Click RUN**

### Backtest Mode

After running, you'll see:

- **Top Row**: 12 key performance metrics in styled cards
- **Equity Curve Tab**: Portfolio value over time with drawdown overlay
- **Price & Signals Tab**: Candlestick chart with buy/sell markers and indicator overlays
- **P&L Analysis Tab**: Histogram of trade P&L, scatter plot of trades over time, win/loss count
- **Monthly Returns Tab**: Heatmap of returns by month/year
- **Trade Log Tab**: Full table of every trade with export buttons (CSV and Excel)

### Optimization Mode

1. Switch the **Mode** toggle from "Backtest" to "Optimize"
2. Choose the **Optimize By** metric (e.g., Total P&L, Sharpe Ratio)
3. For each strategy parameter, expand it and set:
   - **Min**: Lower bound of the range to test
   - **Max**: Upper bound of the range to test
   - **Steps**: How many values to test between min and max
4. **Click OPTIMIZE**

The app will:
- Run every combination of parameters
- Show the **best result** with its parameter values and metrics
- Provide a **heatmap** to visualize how two parameters affect performance
- Display the **full results table** (sortable, exportable)
- Automatically backtest the **top parameter set** and show its equity curve

### Exporting Results

- **Trade Log**: Download as CSV or Excel from the Trade Log tab
- **Optimization Results**: Download as CSV from the Optimization tab
- **Excel Report**: Includes both the trade log and a metrics summary sheet

---

## Project Structure

```
trading-evaluator/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── src/
    ├── __init__.py
    ├── data_fetcher.py    # Automated data pulling (yfinance)
    ├── strategies.py      # Strategy definitions and registry
    ├── backtester.py      # Backtest engine and optimizer
    └── charts.py          # All Plotly visualizations
```

---

## Adding Custom Strategies

You can add your own strategies by editing `src/strategies.py`:

1. **Define the strategy function**:
```python
def strategy_my_custom(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Your custom strategy logic."""
    period = params.get("my_param", 14)
    df = df.copy()
    # ... your logic here ...
    df["signal"] = SignalType.NONE.value
    # Set entries:
    df.loc[your_entry_condition, "signal"] = SignalType.LONG_ENTRY.value
    # Set exits:
    df.loc[your_exit_condition, "signal"] = SignalType.LONG_EXIT.value
    return df
```

2. **Register it** in `EQUITY_STRATEGIES` or `OPTIONS_STRATEGIES`:
```python
"My Custom Strategy": {
    "func": strategy_my_custom,
    "params": {
        "my_param": {"default": 14, "min": 5, "max": 50, "step": 1, "label": "My Parameter"},
    },
    "description": "Description of what it does."
}
```

3. Restart the app — your strategy appears in the dropdown automatically.

---

## Data Source Notes

- **yfinance** pulls data from Yahoo Finance — free, no API key required
- **Daily data**: Available for most tickers going back 20+ years
- **Intraday data** (1h, 5m, etc.): Limited to the last 60 days for free data
- **Options chains**: Real-time options data available for current expirations
- Data is cached for 5 minutes to avoid redundant API calls

### Schwab API (Future Enhancement)

The architecture supports adding Schwab/TD Ameritrade API integration in `data_fetcher.py` for:
- Extended intraday history
- Real-time streaming data
- Historical options data for more accurate options backtesting

---

## Tips for Effective Use

1. **Start with daily data** — it's the most reliable and has the deepest history
2. **Use Optimization mode** to find the best parameter ranges, then fine-tune in Backtest mode
3. **Always check the trade log** — metrics can hide important patterns in individual trades
4. **Compare strategies** by running them on the same ticker/timeframe and noting the metrics
5. **Use risk management** — even great strategies can blow up without stops
6. **Options strategies use simplified premium estimates** — for production use, consider integrating real historical options data

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| No data returned | Check ticker symbol spelling; try a well-known ticker like `SPY` |
| Intraday data missing | Yahoo Finance limits free intraday to ~60 days |
| App won't start | Make sure you're in the project directory when running `streamlit run app.py` |
| Slow optimization | Reduce the number of steps per parameter, or reduce the date range |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

## License

For personal use. Built with Streamlit, Plotly, yfinance, and Pandas.
