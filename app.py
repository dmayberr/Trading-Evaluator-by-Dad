"""
Trading Strategy Evaluator
Main Streamlit Application
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

from src.data_fetcher import fetch_equity_data, fetch_options_chain, get_current_price
from src.strategies import ALL_STRATEGIES, EQUITY_STRATEGIES, OPTIONS_STRATEGIES
from src.backtester import (
    run_backtest, compute_metrics, run_optimization,
    trades_to_dataframe, run_portfolio_backtest
)
from src.charts import (
    chart_equity_curve, chart_pnl_distribution, chart_trade_scatter,
    chart_monthly_returns, chart_price_with_signals,
    chart_optimization_heatmap, chart_win_loss_bar,
    chart_portfolio_equity, chart_portfolio_allocation,
    chart_portfolio_comparison,
    chart_drawdown_duration, chart_rolling_metrics,
    chart_mae_mfe, chart_trade_duration_vs_pnl,
    COLORS
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Strategy Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: #111827;
    --bg-card-hover: #1a2332;
    --border: #1e293b;
    --text-primary: #e2e8f0;
    --text-muted: #64748b;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --accent-amber: #f59e0b;
    --accent-cyan: #06b6d4;
    --accent-purple: #8b5cf6;
}

.stApp {
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

section[data-testid="stSidebar"] {
    background-color: #080c14;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-cyan) !important;
}

.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-hover) 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover {
    border-color: var(--accent-cyan);
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.1);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
}
.metric-value.positive { color: var(--accent-green); }
.metric-value.negative { color: var(--accent-red); }
.metric-value.neutral { color: var(--accent-cyan); }

.app-header {
    text-align: center;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}
.app-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--accent-cyan);
    margin: 0;
    letter-spacing: -0.5px;
}
.app-header p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-muted);
    margin: 4px 0 0 0;
}

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: var(--accent-amber);
    padding: 12px 0 8px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: var(--bg-card);
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background-color: var(--bg-card-hover) !important;
    color: var(--accent-cyan) !important;
}

.stDataFrame { border-radius: 8px; overflow: hidden; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 100%);
    color: var(--bg-primary);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 13px;
    border: none;
    border-radius: 8px;
    padding: 8px 24px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
}

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
}
.badge-equity { background: rgba(59,130,246,0.15); color: var(--accent-blue); border: 1px solid rgba(59,130,246,0.3); }
.badge-options { background: rgba(139,92,246,0.15); color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.3); }
.badge-portfolio { background: rgba(245,158,11,0.15); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.3); }

div[data-testid="stExpander"] {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
}

.ticker-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def render_metric(label: str, value: str, style: str = "neutral"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {style}">{value}</div>
    </div>
    """

def get_metric_style(value: float, invert: bool = False) -> str:
    if invert:
        return "positive" if value < 0 else "negative" if value > 0 else "neutral"
    return "positive" if value > 0 else "negative" if value < 0 else "neutral"

def render_metrics_row(metrics, cols_per_row=6):
    """Render the standard 2-row metrics dashboard."""
    row1_items = [
        ("Total P&L", f"${metrics['total_pnl']:,.0f}", get_metric_style(metrics['total_pnl'])),
        ("Return", f"{metrics['total_return_pct']:.1f}%", get_metric_style(metrics['total_return_pct'])),
        ("Win Rate", f"{metrics['win_rate']:.1f}%", "positive" if metrics['win_rate'] > 50 else "negative"),
        ("Sharpe", f"{metrics['sharpe_ratio']:.2f}", get_metric_style(metrics['sharpe_ratio'])),
        ("Max DD", f"{metrics['max_drawdown_pct']:.1f}%", "negative" if metrics['max_drawdown_pct'] < -10 else "neutral"),
        ("Trades", f"{metrics['total_trades']}", "neutral"),
    ]
    row2_items = [
        ("Profit Factor", f"{metrics['profit_factor']:.2f}", "positive" if metrics['profit_factor'] > 1 else "negative"),
        ("Expectancy", f"${metrics['expectancy']:,.0f}", get_metric_style(metrics['expectancy'])),
        ("Avg Win", f"${metrics['avg_win']:,.0f}", "positive"),
        ("Avg Loss", f"${metrics['avg_loss']:,.0f}", "negative"),
        ("Sortino", f"{metrics['sortino_ratio']:.2f}", get_metric_style(metrics['sortino_ratio'])),
        ("Avg Hold", f"{metrics['avg_hold_days']}d", "neutral"),
    ]

    cols = st.columns(cols_per_row)
    for i, (label, val, style) in enumerate(row1_items):
        with cols[i]:
            st.markdown(render_metric(label, val, style), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    cols2 = st.columns(cols_per_row)
    for i, (label, val, style) in enumerate(row2_items):
        with cols2[i]:
            st.markdown(render_metric(label, val, style), unsafe_allow_html=True)


def generate_recipe_text(
    ticker: str,
    strategy_name: str,
    strategy_config: dict,
    params: dict,
    metrics: dict,
    capital: float,
    sizing_mode: str,
    reinvest_pct: float,
    position_size: float,
    stop_loss=None,
    take_profit=None,
    trailing_stop=None,
    commission: float = 0.0,
    start_date: str = "",
    end_date: str = "",
    notes: str = "",
) -> str:
    """Generate a plain-text strategy recipe card."""
    lines = []
    lines.append("=" * 60)
    lines.append("  STRATEGY RECIPE CARD")
    lines.append("=" * 60)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"  Ticker:       {ticker}")
    lines.append(f"  Strategy:     {strategy_name}")
    lines.append(f"  Description:  {strategy_config.get('description', '')}")
    lines.append(f"  Backtest:     {start_date} → {end_date}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  PARAMETERS")
    lines.append("-" * 60)
    for pname, pval in params.items():
        label = strategy_config["params"][pname]["label"] if pname in strategy_config.get("params", {}) else pname
        lines.append(f"    {label:30s}  {pval}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  RISK MANAGEMENT")
    lines.append("-" * 60)
    lines.append(f"    Capital:              ${capital:,.0f}")
    lines.append(f"    Position Size:        {position_size}%")
    lines.append(f"    Sizing Mode:          {sizing_mode}")
    if sizing_mode == "Fractional (Partial Reinvest)":
        lines.append(f"    Reinvest %:           {reinvest_pct}%")
    lines.append(f"    Stop Loss:            {f'{stop_loss}%' if stop_loss else 'Off'}")
    lines.append(f"    Take Profit:          {f'{take_profit}%' if take_profit else 'Off'}")
    lines.append(f"    Trailing Stop:        {f'{trailing_stop}%' if trailing_stop else 'Off'}")
    lines.append(f"    Commission/Share:     ${commission:.2f}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  BACKTEST RESULTS")
    lines.append("-" * 60)
    lines.append(f"    Total P&L:            ${metrics['total_pnl']:,.2f}")
    lines.append(f"    Return:               {metrics['total_return_pct']:.1f}%")
    lines.append(f"    Win Rate:             {metrics['win_rate']:.1f}%")
    lines.append(f"    Profit Factor:        {metrics['profit_factor']:.2f}")
    lines.append(f"    Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
    lines.append(f"    Sortino Ratio:        {metrics['sortino_ratio']:.2f}")
    lines.append(f"    Max Drawdown:         {metrics['max_drawdown_pct']:.1f}%")
    lines.append(f"    Expectancy:           ${metrics['expectancy']:,.2f}")
    lines.append(f"    Avg Win:              ${metrics['avg_win']:,.2f}")
    lines.append(f"    Avg Loss:             ${metrics['avg_loss']:,.2f}")
    lines.append(f"    Win/Loss Ratio:       {metrics['win_loss_ratio']:.2f}")
    lines.append(f"    Total Trades:         {metrics['total_trades']}")
    lines.append(f"    Avg Hold:             {metrics['avg_hold_days']} days")
    lines.append(f"    Max Consec Wins:      {metrics['max_consecutive_wins']}")
    lines.append(f"    Max Consec Losses:    {metrics['max_consecutive_losses']}")
    lines.append("")
    lines.append("-" * 60)
    lines.append("  EXECUTION CHECKLIST")
    lines.append("-" * 60)

    # Generate plain-English entry/exit rules
    entry_rules, exit_rules = _get_strategy_rules(strategy_name, params)
    lines.append("  ENTRY:")
    for rule in entry_rules:
        lines.append(f"    ☐ {rule}")
    lines.append("  EXIT:")
    for rule in exit_rules:
        lines.append(f"    ☐ {rule}")
    if stop_loss:
        lines.append(f"    ☐ Hard stop loss at {stop_loss}% below entry")
    if take_profit:
        lines.append(f"    ☐ Take profit at {take_profit}% above entry")
    if trailing_stop:
        lines.append(f"    ☐ Trailing stop at {trailing_stop}% from highest point")

    if notes:
        lines.append("")
        lines.append("-" * 60)
        lines.append("  NOTES")
        lines.append("-" * 60)
        for line in notes.strip().split("\n"):
            lines.append(f"    {line}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def _get_strategy_rules(strategy_name: str, params: dict):
    """Return plain-English entry and exit rules for a strategy."""
    rules = {
        "SMA Crossover": (
            [f"SMA({params.get('fast_period', 10)}) crosses ABOVE SMA({params.get('slow_period', 30)})"],
            [f"SMA({params.get('fast_period', 10)}) crosses BELOW SMA({params.get('slow_period', 30)})"],
        ),
        "EMA Crossover": (
            [f"EMA({params.get('fast_period', 9)}) crosses ABOVE EMA({params.get('slow_period', 21)})"],
            [f"EMA({params.get('fast_period', 9)}) crosses BELOW EMA({params.get('slow_period', 21)})"],
        ),
        "RSI Reversal": (
            [f"RSI({params.get('rsi_period', 14)}) crosses ABOVE {params.get('oversold', 30)} (leaving oversold)"],
            [f"RSI({params.get('rsi_period', 14)}) crosses BELOW {params.get('overbought', 70)} (leaving overbought)"],
        ),
        "MACD Crossover": (
            [f"MACD({params.get('fast_period', 12)},{params.get('slow_period', 26)}) crosses ABOVE Signal({params.get('signal_period', 9)})"],
            [f"MACD crosses BELOW Signal line"],
        ),
        "Bollinger Bands": (
            [f"Price closes BELOW lower Bollinger Band (period={params.get('bb_period', 20)}, std={params.get('bb_std', 2.0)})"],
            [f"Price closes ABOVE upper Bollinger Band"],
        ),
        "Stochastic Oscillator": (
            [f"%K({params.get('k_period', 14)}) crosses ABOVE %D({params.get('d_period', 3)}) near oversold zone ({params.get('oversold', 20)})"],
            [f"%K crosses BELOW %D near overbought zone ({params.get('overbought', 80)})"],
        ),
        "Triple EMA": (
            [f"EMA({params.get('fast_period', 8)}) > EMA({params.get('mid_period', 21)}) > EMA({params.get('slow_period', 55)}) — all aligned bullish"],
            [f"EMAs align bearish (fast < mid < slow)"],
        ),
        "Covered Call": (
            [f"Buy stock, sell OTM call at {params.get('otm_pct', 5)}% above current price",
             f"Roll every {params.get('hold_days', 30)} days"],
            [f"Close position at roll date, re-enter with new call"],
        ),
        "Bull Put Spread": (
            [f"RSI({params.get('rsi_period', 14)}) drops below {params.get('rsi_entry', 35)} — sell put spread",
             f"Spread width: {params.get('spread_width_pct', 3)}% of stock price"],
            [f"Hold for {params.get('hold_days', 21)} days or until expiration"],
        ),
        "Iron Condor": (
            [f"Bollinger Band width below {params.get('squeeze_threshold', 0.04)} (low volatility)",
             f"Sell call + put spreads with {params.get('wing_width_pct', 5)}% wings"],
            [f"Hold for {params.get('hold_days', 30)} days"],
        ),
        "Long Straddle": (
            [f"Bollinger Band width below {params.get('squeeze_threshold', 0.03)} (squeeze detected)",
             f"Buy ATM call + ATM put"],
            [f"Hold for {params.get('hold_days', 14)} days"],
        ),
        "SuperTrend": (
            [f"Price crosses ABOVE SuperTrend line (ATR period={params.get('atr_period', 10)}, multiplier={params.get('atr_multiplier', 3.0)})",
             f"Trend flips from bearish to bullish"],
            [f"Price crosses BELOW SuperTrend line (trend flips bearish)"],
        ),
    }
    return rules.get(strategy_name, (["See strategy documentation"], ["See strategy documentation"]))


# ============================================================
# APP HEADER
# ============================================================
st.markdown("""
<div class="app-header">
    <h1>⚡ TRADING STRATEGY EVALUATOR</h1>
    <p>backtest · optimize · portfolio · compound · analyze</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    mode = st.radio("Mode", ["Backtest", "Portfolio", "Optimize"], horizontal=True)

    st.markdown("---")

    # --- Shared: Capital & Sizing ---
    st.markdown("### 💰 Capital & Sizing")
    capital = st.number_input("Starting Capital ($)", value=100000, min_value=1000, step=5000)

    sizing_mode = st.selectbox("Position Sizing Mode", [
        "Compound (Full Reinvest)",
        "Fixed (No Compounding)",
        "Fractional (Partial Reinvest)",
    ], help="Controls how profits affect future position sizes")

    sizing_mode_key = {
        "Compound (Full Reinvest)": "compound",
        "Fixed (No Compounding)": "fixed",
        "Fractional (Partial Reinvest)": "fractional",
    }[sizing_mode]

    reinvest_pct = 50.0
    if sizing_mode_key == "fractional":
        reinvest_pct = st.slider("Reinvest % of Profits", 10, 90, 50, 5,
                                  help="What percentage of profits to reinvest into next trade")

    commission = st.number_input("Commission per Share ($)", value=0.00, min_value=0.00, step=0.01, format="%.2f")

    # ==============================
    # BACKTEST & OPTIMIZE SIDEBAR
    # ==============================
    if mode in ["Backtest", "Optimize"]:
        st.markdown("---")
        st.markdown("### 📊 Data Settings")

        ticker = st.text_input("Ticker Symbol", value="SPY").upper()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
        with col_d2:
            end_date = st.date_input("End Date", value=datetime.now())

        interval = st.selectbox("Interval", ["1d", "1h", "5m", "15m", "30m", "1wk", "1mo"], index=0)

        st.markdown("---")
        st.markdown("### 🎯 Strategy")

        strategy_type = st.radio("Category", ["Equity / Directional", "Options"], horizontal=True)
        if strategy_type == "Equity / Directional":
            strategy_names = list(EQUITY_STRATEGIES.keys())
        else:
            strategy_names = list(OPTIONS_STRATEGIES.keys())

        strategy_name = st.selectbox("Strategy", strategy_names)
        strategy_config = ALL_STRATEGIES[strategy_name]
        st.caption(strategy_config["description"])

        st.markdown("---")
        st.markdown("### 🔧 Parameters")

        params = {}
        for pname, pconfig in strategy_config["params"].items():
            params[pname] = st.slider(
                pconfig["label"], min_value=pconfig["min"], max_value=pconfig["max"],
                value=pconfig["default"], step=pconfig["step"], key=f"param_{pname}"
            )

        st.markdown("---")
        st.markdown("### 🛡️ Risk Management")

        position_size = st.slider("Position Size %", 10, 100, 100, 5)

        use_sl = st.checkbox("Stop Loss")
        stop_loss = st.slider("Stop Loss %", 0.5, 20.0, 5.0, 0.5) if use_sl else None

        use_tp = st.checkbox("Take Profit")
        take_profit = st.slider("Take Profit %", 1.0, 50.0, 10.0, 0.5) if use_tp else None

        use_ts = st.checkbox("Trailing Stop")
        trailing_stop = st.slider("Trailing Stop %", 0.5, 15.0, 3.0, 0.5) if use_ts else None

        st.markdown("---")
        st.markdown("### 📐 Analysis Options")

        use_benchmark = st.checkbox("Compare to Benchmark", value=True,
                                     help="Overlay buy-and-hold of the same ticker on equity curve")
        benchmark_ticker = ticker  # same ticker by default
        if use_benchmark:
            benchmark_ticker = st.text_input("Benchmark Ticker", value=ticker,
                                              help="Default: same ticker. Change to SPY, QQQ, etc.").upper()

        rolling_window = st.slider("Rolling Metrics Window (trades)", 5, 50, 15, 1,
                                    help="Number of trades for rolling win rate calculation")

        if mode == "Optimize":
            st.markdown("---")
            st.markdown("### 🔬 Optimization")
            opt_metric = st.selectbox("Optimize By", [
                "total_pnl", "sharpe_ratio", "win_rate", "profit_factor",
                "total_return_pct", "sortino_ratio", "expectancy"
            ], format_func=lambda x: x.replace("_", " ").title())

            st.markdown("**Parameter Ranges**")
            param_ranges = {}
            for pname, pconfig in strategy_config["params"].items():
                with st.expander(f"🔹 {pconfig['label']}"):
                    range_min = st.number_input(f"Min", value=pconfig["min"], key=f"opt_min_{pname}")
                    range_max = st.number_input(f"Max", value=pconfig["max"], key=f"opt_max_{pname}")
                    n_steps = st.slider(f"Steps", 3, 15, 5, key=f"opt_steps_{pname}")
                    if isinstance(pconfig["default"], float):
                        param_ranges[pname] = list(np.linspace(range_min, range_max, n_steps).round(3))
                    else:
                        param_ranges[pname] = [int(v) for v in np.linspace(range_min, range_max, n_steps)]

    # ==============================
    # PORTFOLIO SIDEBAR
    # ==============================
    elif mode == "Portfolio":
        st.markdown("---")
        st.markdown("### 📊 Data Settings")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
        with col_d2:
            end_date = st.date_input("End Date", value=datetime.now())

        interval = st.selectbox("Interval", ["1d", "1h", "5m", "15m", "30m", "1wk", "1mo"], index=0)

        st.markdown("---")
        st.markdown("### 📂 Portfolio Positions")
        st.caption("Add up to 10 tickers with individual strategies")

        num_positions = st.slider("Number of Positions", 1, 10, 3)

        # Initialize session state for portfolio
        if "portfolio_configs" not in st.session_state:
            st.session_state.portfolio_configs = []

        portfolio_ui_configs = []
        all_strategy_names = list(ALL_STRATEGIES.keys())

        for p in range(num_positions):
            with st.expander(f"📌 Position {p+1}", expanded=(p < 3)):
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    p_ticker = st.text_input("Ticker", value=["SPY", "QQQ", "AAPL", "VRT", "CLS", "MSFT", "AMZN", "GOOG", "META", "NVDA"][p % 10], key=f"pticker_{p}").upper()
                with pcol2:
                    p_alloc = st.number_input("Allocation %", value=round(100/num_positions, 1), min_value=1.0, max_value=100.0, step=1.0, key=f"palloc_{p}")

                p_strategy = st.selectbox("Strategy", all_strategy_names, key=f"pstrat_{p}")
                p_config = ALL_STRATEGIES[p_strategy]

                p_params = {}
                for pname, pconf in p_config["params"].items():
                    p_params[pname] = st.slider(
                        pconf["label"], min_value=pconf["min"], max_value=pconf["max"],
                        value=pconf["default"], step=pconf["step"], key=f"pp_{p}_{pname}"
                    )

                p_sl = st.checkbox("Stop Loss", key=f"psl_{p}")
                p_sl_val = st.slider("SL %", 0.5, 20.0, 5.0, 0.5, key=f"pslv_{p}") if p_sl else None

                p_tp = st.checkbox("Take Profit", key=f"ptp_{p}")
                p_tp_val = st.slider("TP %", 1.0, 50.0, 10.0, 0.5, key=f"ptpv_{p}") if p_tp else None

                p_ts = st.checkbox("Trailing Stop", key=f"pts_{p}")
                p_ts_val = st.slider("TS %", 0.5, 15.0, 3.0, 0.5, key=f"ptsv_{p}") if p_ts else None

                portfolio_ui_configs.append({
                    "ticker": p_ticker,
                    "allocation_pct": p_alloc,
                    "strategy_name": p_strategy,
                    "params": p_params,
                    "stop_loss_pct": p_sl_val,
                    "take_profit_pct": p_tp_val,
                    "trailing_stop_pct": p_ts_val,
                    "commission": commission,
                })

        total_alloc = sum(c["allocation_pct"] for c in portfolio_ui_configs)
        if abs(total_alloc - 100) > 0.5:
            st.warning(f"Total allocation: {total_alloc:.1f}% (should be ~100%)")

    st.markdown("---")
    btn_labels = {"Backtest": "🚀 RUN", "Optimize": "🔬 OPTIMIZE", "Portfolio": "📂 RUN PORTFOLIO"}
    run_button = st.button(btn_labels[mode], use_container_width=True)


# ============================================================
# MAIN CONTENT
# ============================================================
if run_button:

    # ========================================
    # BACKTEST MODE
    # ========================================
    if mode == "Backtest":
        with st.spinner(f"Fetching {ticker} data..."):
            df = fetch_equity_data(ticker, str(start_date), str(end_date), interval)
        if df.empty:
            st.error(f"No data found for {ticker}.")
            st.stop()

        # Fetch benchmark data if enabled
        benchmark_df = None
        if use_benchmark:
            with st.spinner(f"Fetching benchmark ({benchmark_ticker})..."):
                if benchmark_ticker == ticker:
                    benchmark_df = df
                else:
                    benchmark_df = fetch_equity_data(benchmark_ticker, str(start_date), str(end_date), interval)

        st.success(f"Loaded {len(df)} bars for **{ticker}** | Last: **${df['close'].iloc[-1]:,.2f}** | Sizing: **{sizing_mode}**")

        with st.spinner("Running backtest..."):
            signal_df, trades, equity_df = run_backtest(
                df, strategy_name, params, capital,
                position_size, stop_loss, take_profit, trailing_stop,
                commission, sizing_mode_key, reinvest_pct, ticker
            )
            metrics = compute_metrics(trades, equity_df, capital)

        st.markdown('<div class="section-header">📈 Performance Summary</div>', unsafe_allow_html=True)
        render_metrics_row(metrics)
        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs([
            "📊 Equity Curve",
            "🕯️ Price & Signals",
            "📉 P&L Analysis",
            "🎯 MAE / MFE",
            "📈 Rolling Metrics",
            "⏱️ Drawdowns",
            "📅 Monthly Returns",
            "📋 Trade Log",
        ])

        with tabs[0]:
            st.plotly_chart(
                chart_equity_curve(equity_df, capital, trades=trades, benchmark_df=benchmark_df),
                use_container_width=True
            )
            if use_benchmark and benchmark_df is not None and not benchmark_df.empty:
                bench_ret = ((benchmark_df["close"].iloc[-1] / benchmark_df["close"].iloc[0]) - 1) * 100
                st.caption(f"Benchmark ({benchmark_ticker}) buy & hold return: **{bench_ret:+.1f}%** vs Strategy: **{metrics['total_return_pct']:+.1f}%** → Alpha: **{metrics['total_return_pct'] - bench_ret:+.1f}%**")

        with tabs[1]:
            st.plotly_chart(chart_price_with_signals(signal_df, trades, strategy_name), use_container_width=True)

        with tabs[2]:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(chart_pnl_distribution(trades), use_container_width=True)
            with col_b:
                st.plotly_chart(chart_trade_scatter(trades), use_container_width=True)
            col_c, col_d = st.columns(2)
            with col_c:
                st.plotly_chart(chart_win_loss_bar(trades), use_container_width=True)
            with col_d:
                st.plotly_chart(chart_trade_duration_vs_pnl(trades), use_container_width=True)

        with tabs[3]:
            st.plotly_chart(chart_mae_mfe(trades), use_container_width=True)
            st.caption("**Left (MAE vs P&L):** Points near the diagonal lost everything they ever gave back — tighter stops may help. "
                       "**Right (MFE vs P&L):** Points far below the diagonal had big unrealized gains but gave them back — consider trailing targets.")
            # Summary stats
            if trades:
                mae_vals = [t.mae_pct for t in trades]
                mfe_vals = [t.mfe_pct for t in trades]
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(render_metric("Avg MAE", f"{np.mean(mae_vals):.1f}%", "negative"), unsafe_allow_html=True)
                with col_m2:
                    st.markdown(render_metric("Worst MAE", f"{min(mae_vals):.1f}%", "negative"), unsafe_allow_html=True)
                with col_m3:
                    st.markdown(render_metric("Avg MFE", f"{np.mean(mfe_vals):.1f}%", "positive"), unsafe_allow_html=True)
                with col_m4:
                    avg_capture = np.mean([t.pnl_pct / t.mfe_pct * 100 if t.mfe_pct > 0 else 0 for t in trades])
                    st.markdown(render_metric("Avg Capture", f"{avg_capture:.0f}%", "neutral"), unsafe_allow_html=True)

        with tabs[4]:
            st.plotly_chart(chart_rolling_metrics(equity_df, trades, rolling_window), use_container_width=True)

        with tabs[5]:
            st.plotly_chart(chart_drawdown_duration(equity_df), use_container_width=True)

        with tabs[6]:
            st.plotly_chart(chart_monthly_returns(equity_df), use_container_width=True)

        with tabs[7]:
            trade_df = trades_to_dataframe(trades)
            if not trade_df.empty:
                st.dataframe(trade_df, use_container_width=True, height=400)

                # Notes field for recipe
                recipe_notes = st.text_area("📝 Strategy Notes (included in recipe export)",
                                             placeholder="Why did you choose these parameters? What to watch for...",
                                             key="bt_notes", height=80)

                dl_col1, dl_col2, dl_col3 = st.columns(3)
                with dl_col1:
                    csv_buf = io.StringIO()
                    trade_df.to_csv(csv_buf, index=False)
                    st.download_button("📥 Trade Log (CSV)", csv_buf.getvalue(),
                                       f"{ticker}_{strategy_name.replace(' ', '_')}_trades.csv", "text/csv")
                with dl_col2:
                    xlsx_buf = io.BytesIO()
                    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                        trade_df.to_excel(writer, sheet_name="Trades", index=False)
                        pd.DataFrame([metrics]).to_excel(writer, sheet_name="Metrics", index=False)
                    st.download_button("📥 Full Report (Excel)", xlsx_buf.getvalue(),
                                       f"{ticker}_{strategy_name.replace(' ', '_')}_report.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with dl_col3:
                    recipe = generate_recipe_text(
                        ticker, strategy_name, strategy_config, params, metrics,
                        capital, sizing_mode, reinvest_pct, position_size,
                        stop_loss, take_profit, trailing_stop, commission,
                        str(start_date), str(end_date), recipe_notes,
                    )
                    st.download_button("📥 Recipe Card (.txt)", recipe,
                                       f"{ticker}_{strategy_name.replace(' ', '_')}_recipe.txt",
                                       "text/plain")
            else:
                st.info("No trades generated with these parameters.")

    # ========================================
    # PORTFOLIO MODE
    # ========================================
    elif mode == "Portfolio":
        # Fetch data for all tickers
        portfolio_data = []
        with st.spinner("Fetching data for all positions..."):
            for config in portfolio_ui_configs:
                tk = config["ticker"]
                df = fetch_equity_data(tk, str(start_date), str(end_date), interval)
                if df.empty:
                    st.warning(f"No data for {tk} — skipping.")
                    continue
                config["df"] = df
                portfolio_data.append(config)

        if not portfolio_data:
            st.error("No valid data for any position.")
            st.stop()

        tickers_loaded = [c["ticker"] for c in portfolio_data]
        st.success(f"Loaded data for **{', '.join(tickers_loaded)}** | Sizing: **{sizing_mode}**")

        with st.spinner("Running portfolio backtest..."):
            portfolio_results = run_portfolio_backtest(
                portfolio_data, capital, sizing_mode_key, reinvest_pct
            )

        per_ticker = portfolio_results["per_ticker"]
        combined_equity = portfolio_results["combined_equity"]
        combined_metrics = portfolio_results["combined_metrics"]

        # --- Combined Metrics ---
        st.markdown('<div class="section-header">📂 Portfolio Summary</div>', unsafe_allow_html=True)
        render_metrics_row(combined_metrics)
        st.markdown("<br>", unsafe_allow_html=True)

        # --- Portfolio Charts ---
        ptabs = st.tabs(["📊 Portfolio Equity", "📈 Comparison", "🥧 Allocation", "📋 All Trades"] +
                         [f"📌 {t}" for t in per_ticker.keys()])

        with ptabs[0]:
            st.plotly_chart(chart_portfolio_equity(per_ticker, combined_equity, capital), use_container_width=True)

        with ptabs[1]:
            st.plotly_chart(chart_portfolio_comparison(per_ticker), use_container_width=True)

            # Per-ticker metrics table
            summary_rows = []
            for tk, data in per_ticker.items():
                m = data["metrics"]
                summary_rows.append({
                    "Ticker": tk,
                    "Strategy": [c["strategy_name"] for c in portfolio_data if c["ticker"] == tk][0],
                    "Allocation %": data["allocation_pct"],
                    "Capital": f"${data['capital']:,.0f}",
                    "P&L": f"${m['total_pnl']:,.0f}",
                    "Return %": f"{m['total_return_pct']:.1f}%",
                    "Win Rate": f"{m['win_rate']:.1f}%",
                    "Sharpe": f"{m['sharpe_ratio']:.2f}",
                    "Max DD": f"{m['max_drawdown_pct']:.1f}%",
                    "Trades": m['total_trades'],
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        with ptabs[2]:
            st.plotly_chart(chart_portfolio_allocation(per_ticker), use_container_width=True)

        with ptabs[3]:
            all_trade_df = trades_to_dataframe(portfolio_results["combined_trades"], include_ticker=True)
            if not all_trade_df.empty:
                st.dataframe(all_trade_df, use_container_width=True, height=400)
                csv_buf = io.StringIO()
                all_trade_df.to_csv(csv_buf, index=False)
                st.download_button("📥 Download All Trades (CSV)", csv_buf.getvalue(),
                                   "portfolio_trades.csv", "text/csv")
                xlsx_buf = io.BytesIO()
                with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                    all_trade_df.to_excel(writer, sheet_name="All Trades", index=False)
                    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
                    pd.DataFrame([combined_metrics]).to_excel(writer, sheet_name="Portfolio Metrics", index=False)
                st.download_button("📥 Download Portfolio Report (Excel)", xlsx_buf.getvalue(),
                                   "portfolio_report.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("No trades generated across the portfolio.")

        # Per-ticker detail tabs
        for tab_idx, (tk, data) in enumerate(per_ticker.items()):
            with ptabs[4 + tab_idx]:
                st.markdown(f'<div class="section-header">📌 {tk} Detail</div>', unsafe_allow_html=True)
                render_metrics_row(data["metrics"])
                st.markdown("<br>", unsafe_allow_html=True)

                tk_tabs = st.tabs([f"Equity", f"P&L", f"Trades"])
                with tk_tabs[0]:
                    st.plotly_chart(chart_equity_curve(data["equity_df"], data["capital"]), use_container_width=True)
                with tk_tabs[1]:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.plotly_chart(chart_pnl_distribution(data["trades"]), use_container_width=True)
                    with col_b:
                        st.plotly_chart(chart_trade_scatter(data["trades"]), use_container_width=True)
                with tk_tabs[2]:
                    tk_trade_df = trades_to_dataframe(data["trades"])
                    if not tk_trade_df.empty:
                        st.dataframe(tk_trade_df, use_container_width=True, height=300)

    # ========================================
    # OPTIMIZATION MODE
    # ========================================
    elif mode == "Optimize":
        with st.spinner(f"Fetching {ticker} data..."):
            df = fetch_equity_data(ticker, str(start_date), str(end_date), interval)
        if df.empty:
            st.error(f"No data found for {ticker}.")
            st.stop()

        total_combos = 1
        for v in param_ranges.values():
            total_combos *= len(v)

        st.info(f"Running **{total_combos}** parameter combinations on **{ticker}** | Sizing: **{sizing_mode}**")

        with st.spinner("Optimizing..."):
            opt_results = run_optimization(
                df, strategy_name, param_ranges, capital,
                position_size, stop_loss, take_profit, trailing_stop,
                commission, opt_metric, sizing_mode_key, reinvest_pct
            )

        if opt_results.empty:
            st.error("Optimization produced no results.")
            st.stop()

        st.markdown('<div class="section-header">🔬 Optimization Results</div>', unsafe_allow_html=True)

        best = opt_results.iloc[0]
        st.markdown(f"**Best Result** (optimized by *{opt_metric.replace('_', ' ').title()}*)")

        param_cols = list(strategy_config["params"].keys())

        bcols = st.columns(len(param_cols))
        for i, pname in enumerate(param_cols):
            with bcols[i]:
                label = strategy_config["params"][pname]["label"]
                st.markdown(render_metric(label, f"{best[pname]}", "neutral"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        mcols = st.columns(6)
        key_metrics = ["total_pnl", "total_return_pct", "win_rate", "sharpe_ratio", "max_drawdown_pct", "profit_factor"]
        for i, m in enumerate(key_metrics):
            if m in best:
                with mcols[i]:
                    val = best[m]
                    if m == "total_pnl":
                        display = f"${val:,.0f}"
                    elif m in ["total_return_pct", "win_rate", "max_drawdown_pct"]:
                        display = f"{val:.1f}%"
                    else:
                        display = f"{val:.2f}"
                    style = get_metric_style(val) if m != "max_drawdown_pct" else ("negative" if val < -10 else "neutral")
                    st.markdown(render_metric(m.replace("_", " ").title(), display, style), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        opt_tabs = st.tabs(["📊 Heatmap", "📋 Full Results", "🏆 Top 10"])

        with opt_tabs[0]:
            if len(param_cols) >= 2:
                hm_col1, hm_col2 = st.columns(2)
                with hm_col1:
                    hm_x = st.selectbox("X-Axis Parameter", param_cols, index=0)
                with hm_col2:
                    remaining = [p for p in param_cols if p != hm_x]
                    hm_y = st.selectbox("Y-Axis Parameter", remaining, index=0)
                hm_metric = st.selectbox("Heatmap Metric", key_metrics,
                                         format_func=lambda x: x.replace("_", " ").title())
                st.plotly_chart(chart_optimization_heatmap(opt_results, hm_x, hm_y, hm_metric), use_container_width=True)
            else:
                st.info("Need at least 2 parameters for heatmap visualization.")

        with opt_tabs[1]:
            st.dataframe(opt_results, use_container_width=True, height=400)
            csv_buf = io.StringIO()
            opt_results.to_csv(csv_buf, index=False)
            st.download_button("📥 Download Optimization Results (CSV)", csv_buf.getvalue(),
                               f"{ticker}_{strategy_name}_optimization.csv", "text/csv")

        with opt_tabs[2]:
            st.dataframe(opt_results.head(10), use_container_width=True)

        # Best combo backtest
        st.markdown('<div class="section-header">🏆 Best Strategy Backtest</div>', unsafe_allow_html=True)
        best_params = {p: best[p] for p in param_cols}
        best_params = {k: float(v) if isinstance(strategy_config["params"][k]["default"], float) else int(v)
                      for k, v in best_params.items()}

        _, best_trades, best_equity = run_backtest(
            df, strategy_name, best_params, capital,
            position_size, stop_loss, take_profit, trailing_stop,
            commission, sizing_mode_key, reinvest_pct
        )
        best_metrics = compute_metrics(best_trades, best_equity, capital)
        st.plotly_chart(chart_equity_curve(best_equity, capital, trades=best_trades), use_container_width=True)

        # Recipe Card Export
        st.markdown('<div class="section-header">📋 Strategy Recipe Card</div>', unsafe_allow_html=True)

        recipe_notes = st.text_area(
            "📝 Strategy Notes (included in recipe)",
            placeholder="Why this optimization looks promising, market conditions, what to watch...",
            key="opt_notes", height=80,
        )

        recipe = generate_recipe_text(
            ticker, strategy_name, strategy_config, best_params, best_metrics,
            capital, sizing_mode, reinvest_pct, position_size,
            stop_loss, take_profit, trailing_stop, commission,
            str(start_date), str(end_date), recipe_notes,
        )

        # Preview
        with st.expander("👁️ Preview Recipe Card", expanded=True):
            st.code(recipe, language=None)

        rc_col1, rc_col2 = st.columns(2)
        with rc_col1:
            st.download_button(
                "📥 Download Recipe Card (.txt)", recipe,
                f"{ticker}_{strategy_name.replace(' ', '_')}_recipe.txt",
                "text/plain", use_container_width=True,
            )
        with rc_col2:
            # Also export as Excel with recipe + trades
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                # Recipe as a sheet
                recipe_rows = []
                for line in recipe.split("\n"):
                    recipe_rows.append({"Recipe": line})
                pd.DataFrame(recipe_rows).to_excel(writer, sheet_name="Recipe", index=False)
                # Best trades
                best_trade_df = trades_to_dataframe(best_trades)
                if not best_trade_df.empty:
                    best_trade_df.to_excel(writer, sheet_name="Trades", index=False)
                pd.DataFrame([best_metrics]).to_excel(writer, sheet_name="Metrics", index=False)
                opt_results.head(20).to_excel(writer, sheet_name="Top 20 Combos", index=False)
            st.download_button(
                "📥 Download Full Package (Excel)", xlsx_buf.getvalue(),
                f"{ticker}_{strategy_name.replace(' ', '_')}_recipe_package.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <p style="font-family: 'JetBrains Mono', monospace; font-size: 48px; margin-bottom: 8px;">⚡</p>
        <p style="font-family: 'Space Grotesk', sans-serif; font-size: 20px; color: #64748b; margin-bottom: 24px;">
            Configure your strategy in the sidebar, then hit <strong style="color: #06b6d4;">RUN</strong>
        </p>
        <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <p style="font-size: 28px; margin:0;">12</p>
                <p style="font-size: 12px; color: #64748b; font-family: 'JetBrains Mono', monospace;">STRATEGIES</p>
            </div>
            <div style="text-align: center;">
                <p style="font-size: 28px; margin:0;">3</p>
                <p style="font-size: 12px; color: #64748b; font-family: 'JetBrains Mono', monospace;">SIZING MODES</p>
            </div>
            <div style="text-align: center;">
                <p style="font-size: 28px; margin:0;">10</p>
                <p style="font-size: 12px; color: #64748b; font-family: 'JetBrains Mono', monospace;">PORTFOLIO SLOTS</p>
            </div>
            <div style="text-align: center;">
                <p style="font-size: 28px; margin:0;">∞</p>
                <p style="font-size: 12px; color: #64748b; font-family: 'JetBrains Mono', monospace;">PARAMETER COMBOS</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
