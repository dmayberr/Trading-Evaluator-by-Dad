"""
Demo Trading Strategy App with Settings Management
This demonstrates the settings manager integration
"""

import streamlit as st
from datetime import datetime, timedelta
from settings_manager import (
    SettingsManager, 
    render_settings_sidebar, 
    initialize_settings_in_session_state,
    save_current_ui_settings
)

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Tester - Demo",
    page_icon="📈",
    layout="wide"
)

# Initialize settings manager (do this ONCE at the top)
if 'settings_manager' not in st.session_state:
    st.session_state.settings_manager = SettingsManager()

settings_mgr = st.session_state.settings_manager

# Load saved settings into session state
initialize_settings_in_session_state(settings_mgr)

# Main title
st.title("📈 Trading Strategy Tester")
st.markdown("### Demo: Settings Management System")

# Sidebar
st.sidebar.title("Configuration")

# Settings Management UI (collapsible section)
render_settings_sidebar(settings_mgr)

st.sidebar.markdown("---")
st.sidebar.markdown("### Backtest Parameters")

# Input fields using saved settings as defaults
ticker = st.sidebar.text_input(
    "Ticker Symbol", 
    value=settings_mgr.get("default_ticker", "SPY"),
    help="Stock ticker symbol to backtest"
)

# Date inputs with saved defaults
default_start = datetime.strptime(
    settings_mgr.get("default_start_date"), 
    "%Y-%m-%d"
).date()
start_date = st.sidebar.date_input(
    "Start Date", 
    value=default_start,
    help="Backtest start date"
)

default_end = datetime.strptime(
    settings_mgr.get("default_end_date"),
    "%Y-%m-%d"
).date()
end_date = st.sidebar.date_input(
    "End Date", 
    value=default_end,
    help="Backtest end date"
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", 
    value=float(settings_mgr.get("default_initial_capital", 10000.0)),
    min_value=100.0,
    step=1000.0,
    help="Starting capital for backtest"
)

position_size = st.sidebar.number_input(
    "Position Size (% of Capital)",
    value=float(settings_mgr.get("default_position_size", 1.0)),
    min_value=0.01,
    max_value=1.0,
    step=0.05,
    help="Percentage of capital to use per trade"
)

commission = st.sidebar.number_input(
    "Commission per Trade ($)",
    value=float(settings_mgr.get("default_commission", 0.0)),
    min_value=0.0,
    step=0.01,
    help="Transaction costs per trade"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Advanced Settings")

mc_simulations = st.sidebar.number_input(
    "Monte Carlo Simulations",
    value=int(settings_mgr.get("monte_carlo_simulations", 1000)),
    min_value=100,
    max_value=10000,
    step=100,
    help="Number of Monte Carlo simulation runs"
)

mc_confidence_level = st.sidebar.slider(
    "Confidence Level (%)",
    value=int(settings_mgr.get("monte_carlo_confidence_level", 95)),
    min_value=90,
    max_value=99,
    help="Confidence level for Monte Carlo analysis"
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate",
    value=float(settings_mgr.get("risk_free_rate", 0.045)),
    min_value=0.0,
    max_value=0.2,
    step=0.001,
    format="%.3f",
    help="Annual risk-free rate for Sharpe ratio calculation"
)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark Ticker",
    value=settings_mgr.get("benchmark_ticker", "SPY"),
    help="Benchmark for comparison"
)

# Save Current Settings button
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Current as Default Settings", use_container_width=True, type="primary"):
    save_current_ui_settings(
        settings_mgr,
        ticker=ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=initial_capital,
        position_size=position_size,
        commission=commission,
        mc_sims=mc_simulations,
        mc_confidence=mc_confidence_level,
        risk_free_rate=risk_free_rate,
        benchmark=benchmark_ticker
    )
    if settings_mgr.save_settings():
        st.sidebar.success("✅ Settings saved as defaults!")
    else:
        st.sidebar.error("❌ Failed to save settings")

# Main content area
st.markdown("---")

# Display current settings
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Current Configuration")
    st.markdown(f"""
    - **Ticker:** {ticker}
    - **Date Range:** {start_date} to {end_date}
    - **Initial Capital:** ${initial_capital:,.2f}
    - **Position Size:** {position_size:.0%}
    - **Commission:** ${commission:.2f}
    """)

with col2:
    st.markdown("### ⚙️ Advanced Settings")
    st.markdown(f"""
    - **MC Simulations:** {mc_simulations:,}
    - **Confidence Level:** {mc_confidence_level}%
    - **Risk-Free Rate:** {risk_free_rate:.3%}
    - **Benchmark:** {benchmark_ticker}
    """)

st.markdown("---")

# Instructions
st.markdown("### 📝 How to Use Settings Management")

tab1, tab2, tab3 = st.tabs(["Quick Start", "Features", "Tips"])

with tab1:
    st.markdown("""
    #### Getting Started
    
    1. **Adjust settings** in the sidebar to your preference
    2. **Click "Save Current as Default Settings"** to save them
    3. **Restart the app** - your settings will be remembered!
    
    The settings are saved to `trading_app_settings.json` in your app directory.
    """)

with tab2:
    st.markdown("""
    #### Available Features
    
    **⚙️ Settings Management (in sidebar):**
    - **💾 Save Current:** Saves current settings to file
    - **📂 Reload:** Reloads settings from file (useful if you edited manually)
    - **🔄 Reset to Defaults:** Restores factory defaults
    - **⬇️ Export Settings:** Creates a backup file with timestamp
    - **⬆️ Import Settings:** Loads settings from a file
    
    **💾 Save Current as Default:**
    - Saves all current UI values as your new defaults
    - These become the starting values when you restart the app
    """)

with tab3:
    st.markdown("""
    #### Pro Tips
    
    1. **Multiple Configurations:** Export different settings files for different strategies
       - `conservative_settings.json` - Low risk parameters
       - `aggressive_settings.json` - High risk parameters
       - `quick_test_settings.json` - Fast testing parameters
    
    2. **Backup Before Changes:** Always export before experimenting with new settings
    
    3. **Version Control:** Track your settings file in git to see how your preferences evolve
    
    4. **Share Configurations:** Export and share settings files with team members
    
    5. **Manual Editing:** The JSON file can be edited directly in a text editor
    """)

st.markdown("---")

# Show saved settings file content
with st.expander("🔍 View Current Settings File", expanded=False):
    st.markdown("**Current saved settings:**")
    st.json(settings_mgr.get_all())
    
    if st.button("📋 Copy Settings to Clipboard"):
        st.code(str(settings_mgr.get_all()), language="python")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Settings are automatically saved to <code>trading_app_settings.json</code><br>
    This file persists across app restarts and can be backed up or version controlled
</div>
""", unsafe_allow_html=True)
