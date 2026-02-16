"""
Trading Strategy Evaluator — Streamlit Application
Streamlined UI: collapsible sidebar + parameters in main area
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

from settings_manager import (
    SettingsManager,
    render_settings_sidebar,
    initialize_settings_in_session_state,
    save_current_ui_settings
)
from src.data_fetcher import fetch_equity_data, fetch_options_chain, get_current_price
from src.strategies import ALL_STRATEGIES, EQUITY_STRATEGIES, OPTIONS_STRATEGIES
from src.backtester import (
    run_backtest, compute_metrics, run_optimization,
    trades_to_dataframe, run_portfolio_backtest,
    run_monte_carlo, split_walk_forward,
)
from src.charts import (
    chart_equity_curve, chart_pnl_distribution, chart_trade_scatter,
    chart_monthly_returns, chart_price_with_signals,
    chart_optimization_heatmap, chart_win_loss_bar,
    chart_portfolio_equity, chart_portfolio_allocation,
    chart_portfolio_comparison,
    chart_drawdown_duration, chart_rolling_metrics,
    chart_mae_mfe, chart_trade_duration_vs_pnl,
    chart_monte_carlo_fan, chart_monte_carlo_distribution,
    chart_correlation_heatmap,
    COLORS
)

# ============================================================
# PAGE CONFIG & CSS
# ============================================================
st.set_page_config(page_title="Strategy Evaluator", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
:root {
    --bg-primary: #0a0e17; --bg-card: #111827; --bg-card-hover: #1a2332;
    --border: #1e293b; --text-primary: #e2e8f0; --text-muted: #64748b;
    --accent-green: #10b981; --accent-red: #ef4444; --accent-blue: #3b82f6;
    --accent-amber: #f59e0b; --accent-cyan: #06b6d4; --accent-purple: #8b5cf6;
}
.stApp { background-color: var(--bg-primary); color: var(--text-primary); }
section[data-testid="stSidebar"] { background-color: #080c14; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 { color: var(--accent-cyan) !important; }
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-hover) 100%);
    border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover { border-color: var(--accent-cyan); box-shadow: 0 0 20px rgba(6,182,212,0.1); }
.metric-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px; }
.metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; }
.metric-value.positive { color: var(--accent-green); }
.metric-value.negative { color: var(--accent-red); }
.metric-value.neutral { color: var(--accent-cyan); }
.app-header { text-align: center; padding: 6px 0 12px 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.app-header h1 { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700;
    color: var(--accent-cyan); margin: 0; letter-spacing: -0.5px; }
.app-header p { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--text-muted); margin: 2px 0 0 0; }
.section-header { font-family: 'Space Grotesk', sans-serif; font-size: 17px; font-weight: 600;
    color: var(--accent-amber); padding: 10px 0 6px 0; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: var(--bg-card); border-radius: 8px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 12px; }
.stTabs [aria-selected="true"] { background-color: var(--bg-card-hover) !important; color: var(--accent-cyan) !important; }
.stButton > button { background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--accent-blue) 100%);
    color: var(--bg-primary); font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 13px;
    border: none; border-radius: 8px; padding: 8px 24px; text-transform: uppercase; letter-spacing: 0.5px; }
div[data-testid="stExpander"] { background-color: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SETTINGS MANAGER INITIALIZATION
# ============================================================
if 'settings_manager' not in st.session_state:
    st.session_state.settings_manager = SettingsManager()
settings_mgr = st.session_state.settings_manager
initialize_settings_in_session_state(settings_mgr)

# ============================================================
# HELPERS
# ============================================================
def render_metric(label, value, style="neutral"):
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {style}">{value}</div></div>'

def get_metric_style(v, invert=False):
    if invert: return "positive" if v < 0 else "negative" if v > 0 else "neutral"
    return "positive" if v > 0 else "negative" if v < 0 else "neutral"

def render_metrics_row(metrics):
    row1 = [("P&L", f"${metrics['total_pnl']:,.0f}", get_metric_style(metrics['total_pnl'])),
            ("Return", f"{metrics['total_return_pct']:.1f}%", get_metric_style(metrics['total_return_pct'])),
            ("Win Rate", f"{metrics['win_rate']:.1f}%", "positive" if metrics['win_rate'] > 50 else "negative"),
            ("Sharpe", f"{metrics['sharpe_ratio']:.2f}", get_metric_style(metrics['sharpe_ratio'])),
            ("Max DD", f"{metrics['max_drawdown_pct']:.1f}%", "negative" if metrics['max_drawdown_pct'] < -10 else "neutral"),
            ("Trades", f"{metrics['total_trades']}", "neutral")]
    row2 = [("PF", f"{metrics['profit_factor']:.2f}", "positive" if metrics['profit_factor'] > 1 else "negative"),
            ("Expect", f"${metrics['expectancy']:,.0f}", get_metric_style(metrics['expectancy'])),
            ("Avg Win", f"${metrics['avg_win']:,.0f}", "positive"),
            ("Avg Loss", f"${metrics['avg_loss']:,.0f}", "negative"),
            ("Sortino", f"{metrics['sortino_ratio']:.2f}", get_metric_style(metrics['sortino_ratio'])),
            ("Hold", f"{metrics['avg_hold_days']}d", "neutral")]
    for items in [row1, row2]:
        cols = st.columns(6)
        for i, (l, v, s) in enumerate(items):
            with cols[i]:
                st.markdown(render_metric(l, v, s), unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)


def generate_report_card(metrics, trades, strategy_name):
    """Generate plain-English strategy report card."""
    n = metrics["total_trades"]
    if n == 0:
        return "No trades generated — adjust parameters or date range."
    lines = []
    # Performance grade
    sharpe = metrics["sharpe_ratio"]
    if sharpe >= 1.5: grade = "A — Excellent risk-adjusted returns"
    elif sharpe >= 1.0: grade = "B — Strong risk-adjusted returns"
    elif sharpe >= 0.5: grade = "C — Moderate risk-adjusted returns"
    elif sharpe >= 0: grade = "D — Weak positive returns"
    else: grade = "F — Negative risk-adjusted returns"
    lines.append(f"**Overall Grade: {grade}** (Sharpe: {sharpe:.2f})")
    lines.append("")
    # Win/loss character
    wr = metrics["win_rate"]
    pf = metrics["profit_factor"]
    if wr > 60 and pf > 1.5:
        lines.append(f"This strategy wins frequently ({wr:.0f}%) with a healthy profit factor of {pf:.1f}. It's a reliable earner.")
    elif wr < 40 and pf > 1.5:
        lines.append(f"Low win rate ({wr:.0f}%) but high profit factor ({pf:.1f}) — classic trend-following behavior. Wins are much larger than losses.")
    elif wr > 50 and pf < 1.0:
        lines.append(f"Wins often ({wr:.0f}%) but average losses exceed wins (PF: {pf:.1f}). Consider tighter stop losses.")
    else:
        lines.append(f"Win rate: {wr:.0f}%, Profit factor: {pf:.1f}. {'Needs optimization.' if pf < 1 else 'Reasonable balance.'}")
    # Drawdown commentary
    dd = metrics["max_drawdown_pct"]
    if dd < -30:
        lines.append(f"\n⚠️ **Deep drawdown warning**: {dd:.1f}% max drawdown. Ask yourself: can you hold through a {abs(dd):.0f}% decline without panic selling?")
    elif dd < -15:
        lines.append(f"\nDrawdown of {dd:.1f}% is moderate. Most investors can tolerate this, but size positions accordingly.")
    else:
        lines.append(f"\nDrawdown of {dd:.1f}% is manageable — low psychological pain.")
    # Consistency
    cw = metrics["max_consecutive_wins"]
    cl = metrics["max_consecutive_losses"]
    if cl >= 5:
        lines.append(f"\n⚠️ Hit {cl} consecutive losses at one point. Losing streaks test discipline — have a plan for this scenario.")
    # Hold time
    avg_hold = metrics["avg_hold_days"]
    if avg_hold < 3:
        lines.append(f"\nAvg hold of {avg_hold} days — this is a short-term trading strategy. Watch for commission drag.")
    elif avg_hold > 30:
        lines.append(f"\nAvg hold of {avg_hold} days — position trading. Requires patience but lower transaction costs.")
    return "\n".join(lines)


def generate_recipe_text(ticker, strategy_name, strategy_config, params, metrics,
                          capital, sizing_mode, reinvest_pct, position_size,
                          stop_loss=None, take_profit=None, trailing_stop=None,
                          commission=0.0, start_date="", end_date="", notes="",
                          slippage=0.0, commission_model="per_share"):
    lines = []
    lines.append("=" * 56)
    lines.append("  STRATEGY RECIPE CARD")
    lines.append("=" * 56)
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n  Ticker:       {ticker}")
    lines.append(f"  Strategy:     {strategy_name}")
    lines.append(f"  Description:  {strategy_config.get('description', '')}")
    lines.append(f"  Backtest:     {start_date} → {end_date}")
    lines.append("\n" + "-" * 56 + "\n  PARAMETERS\n" + "-" * 56)
    for pn, pv in params.items():
        label = strategy_config["params"].get(pn, {}).get("label", pn)
        lines.append(f"    {label:28s}  {pv}")
    lines.append("\n" + "-" * 56 + "\n  RISK & COSTS\n" + "-" * 56)
    lines.append(f"    Capital:          ${capital:,.0f}")
    lines.append(f"    Position Size:    {position_size}%")
    lines.append(f"    Sizing Mode:      {sizing_mode}")
    lines.append(f"    Slippage:         {slippage}%")
    lines.append(f"    Commission:       {commission} ({commission_model})")
    lines.append(f"    Stop Loss:        {f'{stop_loss}%' if stop_loss else 'Off'}")
    lines.append(f"    Take Profit:      {f'{take_profit}%' if take_profit else 'Off'}")
    lines.append(f"    Trailing Stop:    {f'{trailing_stop}%' if trailing_stop else 'Off'}")
    lines.append("\n" + "-" * 56 + "\n  RESULTS\n" + "-" * 56)
    for k, fmt in [("total_pnl", "${:,.2f}"), ("total_return_pct", "{:.1f}%"), ("win_rate", "{:.1f}%"),
                    ("profit_factor", "{:.2f}"), ("sharpe_ratio", "{:.2f}"), ("sortino_ratio", "{:.2f}"),
                    ("max_drawdown_pct", "{:.1f}%"), ("expectancy", "${:,.2f}"), ("total_trades", "{}")]:
        lines.append(f"    {k.replace('_', ' ').title():24s}  {fmt.format(metrics[k])}")
    lines.append("\n" + "-" * 56 + "\n  EXECUTION CHECKLIST\n" + "-" * 56)
    entry_rules, exit_rules = _get_strategy_rules(strategy_name, params)
    lines.append("  ENTRY:")
    for r in entry_rules: lines.append(f"    ☐ {r}")
    lines.append("  EXIT:")
    for r in exit_rules: lines.append(f"    ☐ {r}")
    if stop_loss: lines.append(f"    ☐ Hard stop at {stop_loss}% below entry")
    if trailing_stop: lines.append(f"    ☐ Trailing stop at {trailing_stop}% from high")
    if notes:
        lines.append("\n" + "-" * 56 + "\n  NOTES\n" + "-" * 56)
        for l in notes.strip().split("\n"): lines.append(f"    {l}")
    lines.append("\n" + "=" * 56)
    return "\n".join(lines)


def _get_strategy_rules(name, params):
    rules = {
        "SMA Crossover": ([f"SMA({params.get('fast_period',10)}) crosses ABOVE SMA({params.get('slow_period',30)})"],
                           [f"SMA({params.get('fast_period',10)}) crosses BELOW SMA({params.get('slow_period',30)})"]),
        "EMA Crossover": ([f"EMA({params.get('fast_period',9)}) crosses ABOVE EMA({params.get('slow_period',21)})"],
                           [f"EMA({params.get('fast_period',9)}) crosses BELOW EMA({params.get('slow_period',21)})"]),
        "RSI Reversal": ([f"RSI({params.get('rsi_period',14)}) crosses ABOVE {params.get('oversold',30)}"],
                          [f"RSI crosses BELOW {params.get('overbought',70)}"]),
        "MACD Crossover": ([f"MACD({params.get('fast_period',12)},{params.get('slow_period',26)}) > Signal({params.get('signal_period',9)})"],
                            ["MACD crosses BELOW Signal"]),
        "Bollinger Bands": ([f"Price below lower BB (period={params.get('bb_period',20)})"], ["Price above upper BB"]),
        "Stochastic Oscillator": ([f"%K({params.get('k_period',14)}) crosses ABOVE %D near {params.get('oversold',20)}"],
                                   [f"%K crosses BELOW %D near {params.get('overbought',80)}"]),
        "Triple EMA": ([f"EMA({params.get('fast_period',8)}) > EMA({params.get('mid_period',21)}) > EMA({params.get('slow_period',55)})"],
                        ["EMAs align bearish"]),
        "SuperTrend": ([f"Price crosses ABOVE SuperTrend (ATR={params.get('atr_period',10)}, mult={params.get('atr_multiplier',3.0)})"],
                        ["Price crosses BELOW SuperTrend"]),
        "Covered Call": ([f"Buy stock + sell {params.get('otm_pct',5)}% OTM call"], [f"Roll every {params.get('hold_days',30)} days"]),
        "Bull Put Spread": ([f"RSI < {params.get('rsi_entry',35)} → sell put spread"], [f"Hold {params.get('hold_days',21)} days"]),
        "Iron Condor": ([f"BB width < {params.get('squeeze_threshold',0.04)} → sell condor"], [f"Hold {params.get('hold_days',30)} days"]),
        "Long Straddle": ([f"BB width < {params.get('squeeze_threshold',0.03)} → buy straddle"], [f"Hold {params.get('hold_days',14)} days"]),
    }
    return rules.get(name, (["See strategy docs"], ["See strategy docs"]))


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="app-header"><h1>⚡ TRADING STRATEGY EVALUATOR</h1><p>backtest · optimize · portfolio · monte carlo · analyze</p></div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR (Streamlined — essentials only)
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    mode = st.radio("Mode", ["Backtest", "Portfolio", "Optimize"], horizontal=True)

    with st.expander("💰 Capital & Sizing", expanded=False):
        capital = st.number_input("Starting Capital ($)", value=float(settings_mgr.get("default_initial_capital", 100000)), min_value=1000.0, step=5000.0)
        sizing_mode = st.selectbox("Position Sizing", ["Compound (Full Reinvest)", "Fixed (No Compounding)", "Fractional (Partial Reinvest)"])
        sizing_mode_key = {"Compound (Full Reinvest)": "compound", "Fixed (No Compounding)": "fixed", "Fractional (Partial Reinvest)": "fractional"}[sizing_mode]
        reinvest_pct = st.slider("Reinvest % of Profits", 10, 90, 50, 5) if sizing_mode_key == "fractional" else 50.0

    with st.expander("💸 Trading Costs", expanded=False):
        slippage_pct = st.number_input("Slippage %", value=0.0, min_value=0.0, max_value=2.0, step=0.01, format="%.2f",
                                        help="Simulates price impact — entry gets worse by this %, exit gets worse by this %")
        commission_model = st.selectbox("Commission Model", ["Per Share", "Per Trade", "Percentage"])
        commission_model_key = {"Per Share": "per_share", "Per Trade": "per_trade", "Percentage": "percentage"}[commission_model]
        comm_help = {"Per Share": "$ per share each side", "Per Trade": "$ flat per trade each side", "Percentage": "% of trade value each side"}
        commission = st.number_input(f"Commission ({comm_help[commission_model]})", value=float(settings_mgr.get("default_commission", 0.00)), min_value=0.0, step=0.01, format="%.2f")

    if mode in ["Backtest", "Optimize"]:
        with st.expander("📊 Data & Strategy", expanded=True):
            ticker = st.text_input("Ticker", value=settings_mgr.get("default_ticker", "SPY")).upper()
            c1, c2 = st.columns(2)
            default_start = datetime.strptime(settings_mgr.get("default_start_date", (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")), "%Y-%m-%d").date()
            default_end = datetime.strptime(settings_mgr.get("default_end_date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d").date()
            with c1: start_date = st.date_input("Start", value=default_start)
            with c2: end_date = st.date_input("End", value=default_end)
            interval = st.selectbox("Interval", ["1d", "1h", "5m", "15m", "30m", "1wk"], index=0)
            auto_adjust = st.checkbox("Adjusted Data (splits/dividends)", value=True,
                                       help="On = adjusted for splits & dividends (recommended). Off = raw prices.")
            strategy_type = st.radio("Category", ["Equity", "Options"], horizontal=True)
            strategy_names = list(EQUITY_STRATEGIES.keys()) if strategy_type == "Equity" else list(OPTIONS_STRATEGIES.keys())
            strategy_name = st.selectbox("Strategy", strategy_names)
            strategy_config = ALL_STRATEGIES[strategy_name]
            st.caption(strategy_config["description"])

    elif mode == "Portfolio":
        with st.expander("📊 Data Settings", expanded=True):
            c1, c2 = st.columns(2)
            default_start = datetime.strptime(settings_mgr.get("default_start_date", (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")), "%Y-%m-%d").date()
            default_end = datetime.strptime(settings_mgr.get("default_end_date", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d").date()
            with c1: start_date = st.date_input("Start", value=default_start)
            with c2: end_date = st.date_input("End", value=default_end)
            interval = st.selectbox("Interval", ["1d", "1h", "5m", "15m", "30m", "1wk"], index=0)
            auto_adjust = st.checkbox("Adjusted Data", value=True)
            num_positions = st.slider("Positions", 1, 10, 3)
            port_benchmark = st.checkbox("Compare to Benchmark", value=True, key="port_bench")
            if port_benchmark:
                port_benchmark_ticker = st.text_input("Benchmark Ticker", value=settings_mgr.get("benchmark_ticker", "SPY"), key="port_bench_tk").upper()

    if mode in ["Backtest"]:
        with st.expander("🔬 Analysis Options", expanded=False):
            use_benchmark = st.checkbox("Compare to Benchmark", value=True)
            benchmark_src = st.selectbox("Benchmark Source", ["Ticker", "Upload CSV"], key="bsrc")
            if benchmark_src == "Ticker":
                benchmark_ticker_input = st.text_input("Benchmark Ticker", value=settings_mgr.get("benchmark_ticker", "SPY"), key="bticker").upper()
            rolling_window = st.slider("Rolling Metrics Window", 5, 50, 15, 1)
            run_mc = st.checkbox("🎲 Monte Carlo Simulation", value=False,
                                  help="Shuffle trade sequence 1,000 times to test robustness and Risk of Ruin")

    btn_labels = {"Backtest": "🚀 RUN", "Optimize": "🔬 OPTIMIZE", "Portfolio": "📂 RUN PORTFOLIO"}
    run_button = st.button(btn_labels[mode], use_container_width=True)

    # Save Current as Default
    st.markdown("---")
    if st.button("💾 Save Current as Default Settings", use_container_width=True, type="primary"):
        # Save common settings that are available in all modes
        settings_mgr.set("default_initial_capital", capital)
        settings_mgr.set("default_commission", commission)
        settings_mgr.set("default_start_date", str(start_date))
        settings_mgr.set("default_end_date", str(end_date))

        # Save ticker if in Backtest or Optimize mode
        if mode in ["Backtest", "Optimize"]:
            settings_mgr.set("default_ticker", ticker)

        # Save benchmark ticker if applicable
        if mode == "Backtest" and use_benchmark and benchmark_src == "Ticker":
            settings_mgr.set("benchmark_ticker", benchmark_ticker_input)
        elif mode == "Portfolio" and port_benchmark:
            settings_mgr.set("benchmark_ticker", port_benchmark_ticker)

        if settings_mgr.save_settings():
            st.success("✅ Settings saved as defaults!")
        else:
            st.error("Failed to save settings")

    # Settings Management UI
    render_settings_sidebar(settings_mgr)


# Set defaults for analysis options if not in Backtest mode
if mode != "Backtest":
    use_benchmark = False
    benchmark_src = "Ticker"
    rolling_window = 15
    run_mc = False

# Track run state in session so results persist across widget interactions
if run_button:
    st.session_state["has_run"] = True
    st.session_state["run_mode"] = mode

# Reset if mode changed
if st.session_state.get("run_mode") != mode:
    st.session_state["has_run"] = False

# ============================================================
# MAIN CONTENT
# ============================================================
if st.session_state.get("has_run", False):
    # ========================================
    # BACKTEST MODE
    # ========================================
    if mode == "Backtest":
        with st.spinner(f"Fetching {ticker}..."):
            df = fetch_equity_data(ticker, str(start_date), str(end_date), interval)
        if df.empty:
            st.error(f"No data for {ticker}."); st.stop()

        # --- Parameters & Risk (main area, compact) ---
        st.markdown('<div class="section-header">🔧 Parameters & Risk</div>', unsafe_allow_html=True)
        param_cols = st.columns(len(strategy_config["params"]))
        params = {}
        for i, (pname, pc) in enumerate(strategy_config["params"].items()):
            with param_cols[i]:
                params[pname] = st.slider(pc["label"], pc["min"], pc["max"], pc["default"], pc["step"], key=f"p_{pname}")

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1: position_size = st.slider("Position %", 10, 100, 100, 5)
        with rc2: stop_loss = st.number_input("Stop Loss %", value=0.0, min_value=0.0, max_value=20.0, step=0.5, format="%.1f") or None
        with rc3: take_profit = st.number_input("Take Profit %", value=0.0, min_value=0.0, max_value=50.0, step=0.5, format="%.1f") or None
        with rc4: trailing_stop = st.number_input("Trailing Stop %", value=0.0, min_value=0.0, max_value=15.0, step=0.5, format="%.1f") or None

        # Analysis options (sourced from sidebar)
        benchmark_df = None
        benchmark_ticker = ticker
        if use_benchmark:
            if benchmark_src == "Ticker":
                benchmark_ticker = benchmark_ticker_input
                if benchmark_ticker != ticker:
                    with st.spinner(f"Fetching benchmark {benchmark_ticker}..."):
                        benchmark_df = fetch_equity_data(benchmark_ticker, str(start_date), str(end_date), interval)
                else:
                    benchmark_df = df
            else:
                uploaded = st.file_uploader("Upload benchmark CSV (must have 'date' and 'close' columns)", type=["csv"])
                if uploaded:
                    try:
                        bench_raw = pd.read_csv(uploaded, parse_dates=["date"], index_col="date")
                        benchmark_df = bench_raw[["close"]] if "close" in bench_raw.columns else None
                    except:
                        st.warning("Could not parse CSV. Need 'date' and 'close' columns.")

        # Run backtest
        st.markdown('<div class="section-header">📈 Results</div>', unsafe_allow_html=True)
        st.success(f"**{ticker}** | {len(df)} bars | {strategy_name} | Slippage: {slippage_pct}% | Commission: {commission} ({commission_model})")

        signal_df, trades, equity_df = run_backtest(
            df, strategy_name, params, capital, position_size,
            stop_loss, take_profit, trailing_stop, commission,
            sizing_mode_key, reinvest_pct, ticker, slippage_pct, commission_model_key
        )
        metrics = compute_metrics(trades, equity_df, capital)
        render_metrics_row(metrics)

        # Walk-forward OOS
        oos_split_date = None
        if len(df) > 100:
            is_df, oos_df, oos_split_date = split_walk_forward(df, 70.0)

        # Tabs
        tab_names = ["📊 Equity", "🕯️ Signals", "📉 P&L", "🎯 MAE/MFE", "📈 Rolling", "⏱️ Drawdown", "📅 Monthly", "📋 Trades", "📝 Report"]
        if run_mc: tab_names.append("🎲 Monte Carlo")
        tabs = st.tabs(tab_names)

        with tabs[0]:
            fig = chart_equity_curve(equity_df, capital, trades=trades, benchmark_df=benchmark_df)
            # Add OOS shading
            if oos_split_date is not None:
                fig.add_vrect(x0=oos_split_date, x1=equity_df.index[-1],
                              fillcolor="rgba(139,92,246,0.06)", line_width=0,
                              annotation_text="Out-of-Sample →", annotation_position="top left",
                              annotation_font=dict(color="#8b5cf6", size=10), row=1, col=1)
            st.plotly_chart(fig, use_container_width=True)
            if use_benchmark and benchmark_df is not None and not benchmark_df.empty:
                bench_ret = ((benchmark_df["close"].iloc[-1] / benchmark_df["close"].iloc[0]) - 1) * 100
                st.caption(f"Benchmark: **{bench_ret:+.1f}%** | Strategy: **{metrics['total_return_pct']:+.1f}%** | Alpha: **{metrics['total_return_pct'] - bench_ret:+.1f}%**")

        with tabs[1]:
            st.plotly_chart(chart_price_with_signals(signal_df, trades, strategy_name), use_container_width=True)

        with tabs[2]:
            ca, cb = st.columns(2)
            with ca: st.plotly_chart(chart_pnl_distribution(trades), use_container_width=True)
            with cb: st.plotly_chart(chart_trade_scatter(trades), use_container_width=True)
            cc, cd = st.columns(2)
            with cc: st.plotly_chart(chart_win_loss_bar(trades), use_container_width=True)
            with cd: st.plotly_chart(chart_trade_duration_vs_pnl(trades), use_container_width=True)

        with tabs[3]:
            st.plotly_chart(chart_mae_mfe(trades), use_container_width=True)
            st.caption("**Left**: MAE vs P&L — stops analysis. **Right**: MFE vs P&L — target analysis. Below diagonal = left money on table.")
            if trades:
                mc1, mc2, mc3, mc4 = st.columns(4)
                mae_v = [t.mae_pct for t in trades]; mfe_v = [t.mfe_pct for t in trades]
                with mc1: st.markdown(render_metric("Avg MAE", f"{np.mean(mae_v):.1f}%", "negative"), unsafe_allow_html=True)
                with mc2: st.markdown(render_metric("Worst MAE", f"{min(mae_v):.1f}%", "negative"), unsafe_allow_html=True)
                with mc3: st.markdown(render_metric("Avg MFE", f"{np.mean(mfe_v):.1f}%", "positive"), unsafe_allow_html=True)
                with mc4:
                    cap_rate = np.mean([t.pnl_pct / t.mfe_pct * 100 if t.mfe_pct > 0 else 0 for t in trades])
                    st.markdown(render_metric("Capture", f"{cap_rate:.0f}%", "neutral"), unsafe_allow_html=True)

        with tabs[4]:
            st.plotly_chart(chart_rolling_metrics(equity_df, trades, rolling_window), use_container_width=True)

        with tabs[5]:
            st.plotly_chart(chart_drawdown_duration(equity_df), use_container_width=True)

        with tabs[6]:
            st.plotly_chart(chart_monthly_returns(equity_df), use_container_width=True)

        with tabs[7]:
            trade_df = trades_to_dataframe(trades)
            if not trade_df.empty:
                st.dataframe(trade_df, use_container_width=True, height=350)
                recipe_notes = st.text_area("📝 Notes", placeholder="Strategy notes...", key="bt_notes", height=60)
                d1, d2, d3 = st.columns(3)
                with d1:
                    buf = io.StringIO(); trade_df.to_csv(buf, index=False)
                    st.download_button("📥 CSV", buf.getvalue(), f"{ticker}_trades.csv", "text/csv")
                with d2:
                    xbuf = io.BytesIO()
                    with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                        trade_df.to_excel(w, sheet_name="Trades", index=False)
                        pd.DataFrame([metrics]).to_excel(w, sheet_name="Metrics", index=False)
                    st.download_button("📥 Excel", xbuf.getvalue(), f"{ticker}_report.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with d3:
                    recipe = generate_recipe_text(ticker, strategy_name, strategy_config, params, metrics,
                                                   capital, sizing_mode, reinvest_pct, position_size,
                                                   stop_loss, take_profit, trailing_stop, commission,
                                                   str(start_date), str(end_date), recipe_notes, slippage_pct, commission_model)
                    st.download_button("📥 Recipe", recipe, f"{ticker}_recipe.txt", "text/plain")

        with tabs[8]:
            st.markdown('<div class="section-header">📝 Strategy Report Card</div>', unsafe_allow_html=True)
            report = generate_report_card(metrics, trades, strategy_name)
            st.markdown(report)

        if run_mc and len(tabs) > 9:
            with tabs[9]:
                with st.spinner("Running Monte Carlo simulation..."):
                    mc_sims = st.slider("Simulations", 100, 2000, 1000, 100, key="mc_n") if False else 1000
                    mc_results = run_monte_carlo(trades, capital, n_simulations=1000)
                st.plotly_chart(chart_monte_carlo_fan(mc_results, capital), use_container_width=True)
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1: st.markdown(render_metric("Risk of Ruin", f"{mc_results['risk_of_ruin']:.1f}%",
                                                     "negative" if mc_results['risk_of_ruin'] > 10 else "positive"), unsafe_allow_html=True)
                with mc2: st.markdown(render_metric("Median Equity", f"${mc_results['median_equity']:,.0f}",
                                                     get_metric_style(mc_results['median_equity'] - capital)), unsafe_allow_html=True)
                with mc3: st.markdown(render_metric("5th %ile", f"${mc_results['p5_equity']:,.0f}",
                                                     get_metric_style(mc_results['p5_equity'] - capital)), unsafe_allow_html=True)
                with mc4: st.markdown(render_metric("95th %ile", f"${mc_results['p95_equity']:,.0f}", "positive"), unsafe_allow_html=True)
                st.plotly_chart(chart_monte_carlo_distribution(mc_results, capital), use_container_width=True)
                st.caption(f"Based on {mc_results['n_simulations']} random reshuffles of {mc_results['n_trades']} trades. "
                           f"Risk of ruin = probability of losing 90%+ of capital.")

    # ========================================
    # PORTFOLIO MODE
    # ========================================
    elif mode == "Portfolio":
        all_strat_names = list(ALL_STRATEGIES.keys())
        # Build portfolio config in main area
        st.markdown('<div class="section-header">📂 Portfolio Positions</div>', unsafe_allow_html=True)
        portfolio_ui = []
        for p in range(num_positions):
            with st.expander(f"Position {p+1}", expanded=(p < 3)):
                pc1, pc2, pc3 = st.columns([2, 2, 1])
                with pc1:
                    ptk = st.text_input("Ticker", value=["SPY","QQQ","AAPL","VRT","CLS","MSFT","AMZN","GOOG","META","NVDA"][p%10], key=f"pt_{p}").upper()
                with pc2:
                    pstrat = st.selectbox("Strategy", all_strat_names, key=f"ps_{p}")
                with pc3:
                    palloc = st.number_input("Alloc %", value=round(100/num_positions, 1), min_value=1.0, max_value=100.0, step=1.0, key=f"pa_{p}")
                pcfg = ALL_STRATEGIES[pstrat]
                pcols = st.columns(len(pcfg["params"]))
                pp = {}
                for i, (pn, pc2) in enumerate(pcfg["params"].items()):
                    with pcols[i]: pp[pn] = st.slider(pc2["label"], pc2["min"], pc2["max"], pc2["default"], pc2["step"], key=f"pp_{p}_{pn}")
                r1, r2, r3 = st.columns(3)
                with r1: psl = st.number_input("SL %", value=0.0, min_value=0.0, max_value=20.0, step=0.5, key=f"psl_{p}", format="%.1f") or None
                with r2: ptp = st.number_input("TP %", value=0.0, min_value=0.0, max_value=50.0, step=0.5, key=f"ptp_{p}", format="%.1f") or None
                with r3: pts = st.number_input("TS %", value=0.0, min_value=0.0, max_value=15.0, step=0.5, key=f"pts_{p}", format="%.1f") or None
                portfolio_ui.append({"ticker": ptk, "allocation_pct": palloc, "strategy_name": pstrat,
                                     "params": pp, "stop_loss_pct": psl, "take_profit_pct": ptp,
                                     "trailing_stop_pct": pts, "commission": commission,
                                     "slippage_pct": slippage_pct, "commission_model": commission_model_key})

        total_alloc = sum(c["allocation_pct"] for c in portfolio_ui)
        if abs(total_alloc - 100) > 0.5:
            st.warning(f"Total allocation: {total_alloc:.1f}% (should be ~100%)")

        # Build cache key from all portfolio inputs
        port_cache_key = str([(c["ticker"], c["strategy_name"], str(c["params"]),
                               c["allocation_pct"], c["stop_loss_pct"], c["take_profit_pct"],
                               c["trailing_stop_pct"], c["commission"], c["slippage_pct"],
                               c["commission_model"])
                              for c in portfolio_ui]) + str((capital, sizing_mode_key, reinvest_pct,
                              str(start_date), str(end_date), interval))

        if st.session_state.get("port_cache_key") != port_cache_key or "port_results" not in st.session_state:
            # Fetch & run
            portfolio_data = []
            with st.spinner("Fetching data..."):
                for cfg in portfolio_ui:
                    d = fetch_equity_data(cfg["ticker"], str(start_date), str(end_date), interval)
                    if d.empty: st.warning(f"No data for {cfg['ticker']}"); continue
                    cfg["df"] = d; portfolio_data.append(cfg)
            if not portfolio_data: st.error("No valid data."); st.stop()

            with st.spinner("Running portfolio backtest..."):
                pr = run_portfolio_backtest(portfolio_data, capital, sizing_mode_key, reinvest_pct)
            st.session_state["port_results"] = pr
            st.session_state["port_cache_key"] = port_cache_key
        else:
            pr = st.session_state["port_results"]

        st.markdown('<div class="section-header">📈 Portfolio Results</div>', unsafe_allow_html=True)
        render_metrics_row(pr["combined_metrics"])

        ptabs = st.tabs(["📊 Equity", "📈 Compare", "🔗 Correlation", "🥧 Allocation", "📋 Trades"] +
                         [f"📌 {t}" for t in pr["per_ticker"]])

        with ptabs[0]:
            # Fetch portfolio benchmark if enabled
            port_bench_df = None
            if mode == "Portfolio" and st.session_state.get("port_bench", True):
                btk = st.session_state.get("port_bench_tk", "SPY") or "SPY"
                port_bench_df = fetch_equity_data(btk.upper(), str(start_date), str(end_date), interval)

            st.plotly_chart(chart_portfolio_equity(pr["per_ticker"], pr["combined_equity"], capital,
                                                    benchmark_df=port_bench_df), use_container_width=True)
            if port_bench_df is not None and not port_bench_df.empty:
                bench_ret = ((port_bench_df["close"].iloc[-1] / port_bench_df["close"].iloc[0]) - 1) * 100
                port_ret = pr["combined_metrics"]["total_return_pct"]
                st.caption(f"Benchmark ({btk.upper()}): **{bench_ret:+.1f}%** | Portfolio: **{port_ret:+.1f}%** | Alpha: **{port_ret - bench_ret:+.1f}%**")
        with ptabs[1]:
            st.plotly_chart(chart_portfolio_comparison(pr["per_ticker"]), use_container_width=True)
            rows = []
            for tk, d in pr["per_ticker"].items():
                m = d["metrics"]
                # Get strategy name from the trades (works with cached results)
                strat_name = d["trades"][0].strategy if d["trades"] else "—"
                rows.append({"Ticker": tk, "Strategy": strat_name,
                             "Alloc%": d["allocation_pct"], "P&L": f"${m['total_pnl']:,.0f}",
                             "Return": f"{m['total_return_pct']:.1f}%", "Win%": f"{m['win_rate']:.1f}%",
                             "Sharpe": f"{m['sharpe_ratio']:.2f}", "MaxDD": f"{m['max_drawdown_pct']:.1f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        with ptabs[2]:
            st.plotly_chart(chart_correlation_heatmap(pr["correlation"]), use_container_width=True)
            if not pr["correlation"].empty:
                max_corr = pr["correlation"].where(np.triu(np.ones(pr["correlation"].shape), k=1).astype(bool)).max().max()
                if max_corr > 0.7:
                    st.warning(f"⚠️ High correlation detected ({max_corr:.2f}). These positions may move together — limited diversification benefit.")
                else:
                    st.success(f"✅ Max pairwise correlation: {max_corr:.2f}. Good diversification.")
        with ptabs[3]:
            st.plotly_chart(chart_portfolio_allocation(pr["per_ticker"]), use_container_width=True)
        with ptabs[4]:
            atdf = trades_to_dataframe(pr["combined_trades"], include_ticker=True)
            if not atdf.empty:
                st.dataframe(atdf, use_container_width=True, height=350)
                buf = io.StringIO(); atdf.to_csv(buf, index=False)
                st.download_button("📥 All Trades CSV", buf.getvalue(), "portfolio_trades.csv", "text/csv")

        for ti, (tk, d) in enumerate(pr["per_ticker"].items()):
            with ptabs[5 + ti]:
                render_metrics_row(d["metrics"])
                st.plotly_chart(chart_equity_curve(d["equity_df"], d["capital"], trades=d["trades"]), use_container_width=True)

    # ========================================
    # OPTIMIZE MODE
    # ========================================
    elif mode == "Optimize":
        with st.spinner(f"Fetching {ticker}..."):
            df = fetch_equity_data(ticker, str(start_date), str(end_date), interval)
        if df.empty: st.error(f"No data for {ticker}."); st.stop()

        # Parameters & optimization config in main area
        st.markdown('<div class="section-header">🔧 Parameters & Optimization</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1: position_size = st.slider("Position %", 10, 100, 100, 5, key="opt_ps")
        with rc2: stop_loss = st.number_input("SL %", value=0.0, min_value=0.0, max_value=20.0, step=0.5, format="%.1f", key="opt_sl") or None
        with rc3: take_profit = st.number_input("TP %", value=0.0, min_value=0.0, max_value=50.0, step=0.5, format="%.1f", key="opt_tp") or None
        with rc4: trailing_stop = st.number_input("TS %", value=0.0, min_value=0.0, max_value=15.0, step=0.5, format="%.1f", key="opt_ts") or None

        param_ranges = {}
        for pname, pc in strategy_config["params"].items():
            with st.expander(f"🔹 {pc['label']}"):
                c1, c2, c3 = st.columns(3)
                with c1: rmin = st.number_input("Min", value=pc["min"], key=f"om_{pname}")
                with c2: rmax = st.number_input("Max", value=pc["max"], key=f"ox_{pname}")
                with c3: nsteps = st.slider("Steps", 3, 15, 5, key=f"os_{pname}")
                if isinstance(pc["default"], float):
                    param_ranges[pname] = list(np.linspace(rmin, rmax, nsteps).round(3))
                else:
                    param_ranges[pname] = [int(v) for v in np.linspace(rmin, rmax, nsteps)]

        total_combos = 1
        for v in param_ranges.values(): total_combos *= len(v)

        # Build a cache key from the inputs that matter for the optimization run
        opt_cache_key = (ticker, strategy_name, str(start_date), str(end_date), interval,
                         str(param_ranges), capital, position_size, stop_loss, take_profit,
                         trailing_stop, commission, sizing_mode_key, reinvest_pct,
                         slippage_pct, commission_model_key)

        # Only re-run optimization if inputs changed
        if st.session_state.get("opt_cache_key") != opt_cache_key or "opt_results" not in st.session_state:
            st.info(f"Running **{total_combos}** combinations on **{ticker}**")
            with st.spinner("Optimizing..."):
                opt_results = run_optimization(df, strategy_name, param_ranges, capital, position_size,
                                                stop_loss, take_profit, trailing_stop, commission,
                                                "total_pnl", sizing_mode_key, reinvest_pct, slippage_pct, commission_model_key)
            st.session_state["opt_results"] = opt_results
            st.session_state["opt_cache_key"] = opt_cache_key
            st.session_state["opt_df"] = df
        else:
            opt_results = st.session_state["opt_results"]
            df = st.session_state["opt_df"]

        if opt_results.empty: st.error("No results."); st.stop()

        # Sort by selected metric (this is cheap and can change without re-running)
        opt_metric = st.selectbox("Sort Results By", ["total_pnl", "sharpe_ratio", "win_rate", "profit_factor",
                                                    "total_return_pct", "sortino_ratio", "expectancy"],
                                   format_func=lambda x: x.replace("_", " ").title(), key="opt_metric")

        sorted_results = opt_results.sort_values(opt_metric, ascending=False).reset_index(drop=True)
        best = sorted_results.iloc[0]
        param_cols = list(strategy_config["params"].keys())

        st.markdown(f'<div class="section-header">🏆 Best by {opt_metric.replace("_"," ").title()}</div>', unsafe_allow_html=True)

        bcols = st.columns(len(param_cols) + 4)
        for i, pn in enumerate(param_cols):
            with bcols[i]: st.markdown(render_metric(strategy_config["params"][pn]["label"], f"{best[pn]}", "neutral"), unsafe_allow_html=True)
        for i, (k, fmt) in enumerate([("total_pnl","${:,.0f}"),("sharpe_ratio","{:.2f}"),("win_rate","{:.1f}%"),("max_drawdown_pct","{:.1f}%")]):
            with bcols[len(param_cols)+i]:
                v = best[k]; st.markdown(render_metric(k.replace("_"," ").title(), fmt.format(v),
                    get_metric_style(v) if k!="max_drawdown_pct" else ("negative" if v<-10 else "neutral")), unsafe_allow_html=True)

        ot = st.tabs(["📊 Heatmap", "📋 Results", "🏆 Best Backtest", "📝 Recipe"])

        with ot[0]:
            if len(param_cols) >= 2:
                h1, h2 = st.columns(2)
                with h1: hx = st.selectbox("X-Axis", param_cols)
                with h2: hy = st.selectbox("Y-Axis", [p for p in param_cols if p != hx])
                hm = st.selectbox("Metric", ["total_pnl","sharpe_ratio","win_rate","profit_factor","max_drawdown_pct"],
                                   format_func=lambda x: x.replace("_"," ").title())
                st.plotly_chart(chart_optimization_heatmap(sorted_results, hx, hy, hm), use_container_width=True)
        with ot[1]:
            st.dataframe(sorted_results, use_container_width=True, height=350)
            buf = io.StringIO(); sorted_results.to_csv(buf, index=False)
            st.download_button("📥 CSV", buf.getvalue(), f"{ticker}_optimization.csv", "text/csv")

        with ot[2]:
            best_params = {p: float(best[p]) if isinstance(strategy_config["params"][p]["default"], float) else int(best[p]) for p in param_cols}
            _, bt, be = run_backtest(df, strategy_name, best_params, capital, position_size,
                                     stop_loss, take_profit, trailing_stop, commission,
                                     sizing_mode_key, reinvest_pct, ticker, slippage_pct, commission_model_key)
            bm = compute_metrics(bt, be, capital)
            render_metrics_row(bm)
            st.plotly_chart(chart_equity_curve(be, capital, trades=bt), use_container_width=True)

        with ot[3]:
            notes = st.text_area("📝 Notes", placeholder="Why this setup looks promising...", key="opt_notes", height=60)
            recipe = generate_recipe_text(ticker, strategy_name, strategy_config, best_params, bm,
                                           capital, sizing_mode, reinvest_pct, position_size,
                                           stop_loss, take_profit, trailing_stop, commission,
                                           str(start_date), str(end_date), notes, slippage_pct, commission_model)
            with st.expander("👁️ Preview", expanded=True): st.code(recipe, language=None)
            r1, r2 = st.columns(2)
            with r1: st.download_button("📥 Recipe (.txt)", recipe, f"{ticker}_recipe.txt", "text/plain", use_container_width=True)
            with r2:
                xbuf = io.BytesIO()
                with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                    pd.DataFrame([{"Recipe": l} for l in recipe.split("\n")]).to_excel(w, sheet_name="Recipe", index=False)
                    trades_to_dataframe(bt).to_excel(w, sheet_name="Trades", index=False)
                    sorted_results.head(20).to_excel(w, sheet_name="Top 20", index=False)
                st.download_button("📥 Full Package (.xlsx)", xbuf.getvalue(), f"{ticker}_package.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 50px 20px;">
        <p style="font-size: 44px; margin-bottom: 6px;">⚡</p>
        <p style="font-family: 'Space Grotesk'; font-size: 18px; color: #64748b; margin-bottom: 20px;">
            Configure in sidebar → hit <strong style="color: #06b6d4;">RUN</strong></p>
        <div style="display:flex; justify-content:center; gap:32px; flex-wrap:wrap;">
            <div><p style="font-size:24px; margin:0;">12</p><p style="font-size:11px; color:#64748b; font-family:'JetBrains Mono';">STRATEGIES</p></div>
            <div><p style="font-size:24px; margin:0;">3</p><p style="font-size:11px; color:#64748b; font-family:'JetBrains Mono';">SIZING MODES</p></div>
            <div><p style="font-size:24px; margin:0;">10</p><p style="font-size:11px; color:#64748b; font-family:'JetBrains Mono';">PORTFOLIO</p></div>
            <div><p style="font-size:24px; margin:0;">MC</p><p style="font-size:11px; color:#64748b; font-family:'JetBrains Mono';">MONTE CARLO</p></div>
        </div>
    </div>""", unsafe_allow_html=True)
