# Settings Manager Integration Guide

## Overview
This guide shows how to integrate the SettingsManager into your existing Streamlit trading strategy application.

## Step 1: Import the Settings Manager

Add this to the top of your main Streamlit app file (after other imports):

```python
from settings_manager import (
    SettingsManager, 
    render_settings_sidebar, 
    initialize_settings_in_session_state,
    save_current_ui_settings
)
```

## Step 2: Initialize Settings Manager at App Start

Add this near the beginning of your main app code (before any UI elements):

```python
# Initialize settings manager
if 'settings_manager' not in st.session_state:
    st.session_state.settings_manager = SettingsManager()

settings_mgr = st.session_state.settings_manager

# Load saved settings into session state
initialize_settings_in_session_state(settings_mgr)
```

## Step 3: Update Your Sidebar Inputs to Use Saved Settings

Replace your current sidebar input defaults with saved settings. Here's the pattern:

### BEFORE (example):
```python
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*3))
initial_capital = st.sidebar.number_input("Initial Capital", value=10000.0)
```

### AFTER:
```python
ticker = st.sidebar.text_input(
    "Ticker Symbol", 
    value=settings_mgr.get("default_ticker", "SPY")
)

# For dates, convert string to datetime
default_start = datetime.strptime(
    settings_mgr.get("default_start_date"), 
    "%Y-%m-%d"
).date()
start_date = st.sidebar.date_input("Start Date", value=default_start)

default_end = datetime.strptime(
    settings_mgr.get("default_end_date"),
    "%Y-%m-%d"
).date()
end_date = st.sidebar.date_input("End Date", value=default_end)

initial_capital = st.sidebar.number_input(
    "Initial Capital", 
    value=float(settings_mgr.get("default_initial_capital", 10000.0)),
    min_value=100.0,
    step=1000.0
)

position_size = st.sidebar.number_input(
    "Position Size (% of Capital)",
    value=float(settings_mgr.get("default_position_size", 1.0)),
    min_value=0.01,
    max_value=1.0,
    step=0.05
)

commission = st.sidebar.number_input(
    "Commission per Trade",
    value=float(settings_mgr.get("default_commission", 0.0)),
    min_value=0.0,
    step=0.01
)
```

## Step 4: Add Settings Management UI to Sidebar

Add this anywhere in your sidebar (recommend near the top or bottom):

```python
# Render settings management controls
render_settings_sidebar(settings_mgr)
```

## Step 5: Add Auto-Save on Changes (Optional but Recommended)

Add a "Save as Default" button in your sidebar or main area:

```python
# In your sidebar, after all inputs
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Current as Default Settings", use_container_width=True):
    save_current_ui_settings(
        settings_mgr,
        ticker=ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=initial_capital,
        position_size=position_size,
        commission=commission
    )
    if settings_mgr.save_settings():
        st.sidebar.success("✅ Settings saved as defaults!")
    else:
        st.sidebar.error("❌ Failed to save settings")
```

## Step 6: Update Monte Carlo and Advanced Settings (if applicable)

If you have Monte Carlo simulation settings, update them similarly:

```python
mc_simulations = st.number_input(
    "Number of Simulations",
    value=int(settings_mgr.get("monte_carlo_simulations", 1000)),
    min_value=100,
    max_value=10000,
    step=100
)

mc_confidence_level = st.slider(
    "Confidence Level",
    value=int(settings_mgr.get("monte_carlo_confidence_level", 95)),
    min_value=90,
    max_value=99
)

risk_free_rate = st.number_input(
    "Risk-Free Rate",
    value=float(settings_mgr.get("risk_free_rate", 0.045)),
    min_value=0.0,
    max_value=0.2,
    step=0.001,
    format="%.3f"
)
```

## Complete Example: Sidebar Section

Here's a complete example of how your sidebar section might look:

```python
import streamlit as st
from datetime import datetime, timedelta
from settings_manager import (
    SettingsManager, 
    render_settings_sidebar, 
    initialize_settings_in_session_state,
    save_current_ui_settings
)

# Initialize settings manager (do this once at the top of your app)
if 'settings_manager' not in st.session_state:
    st.session_state.settings_manager = SettingsManager()

settings_mgr = st.session_state.settings_manager
initialize_settings_in_session_state(settings_mgr)

# Sidebar
st.sidebar.title("Trading Strategy Tester")

# Settings Management UI
render_settings_sidebar(settings_mgr)

st.sidebar.markdown("---")
st.sidebar.markdown("### Backtest Parameters")

# Load defaults from saved settings
ticker = st.sidebar.text_input(
    "Ticker Symbol", 
    value=settings_mgr.get("default_ticker", "SPY")
)

default_start = datetime.strptime(
    settings_mgr.get("default_start_date"), 
    "%Y-%m-%d"
).date()
start_date = st.sidebar.date_input("Start Date", value=default_start)

default_end = datetime.strptime(
    settings_mgr.get("default_end_date"),
    "%Y-%m-%d"
).date()
end_date = st.sidebar.date_input("End Date", value=default_end)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", 
    value=float(settings_mgr.get("default_initial_capital", 10000.0)),
    min_value=100.0,
    step=1000.0
)

position_size = st.sidebar.number_input(
    "Position Size (% of Capital)",
    value=float(settings_mgr.get("default_position_size", 1.0)),
    min_value=0.01,
    max_value=1.0,
    step=0.05
)

commission = st.sidebar.number_input(
    "Commission per Trade ($)",
    value=float(settings_mgr.get("default_commission", 0.0)),
    min_value=0.0,
    step=0.01
)

# Save as Default button
st.sidebar.markdown("---")
if st.sidebar.button("💾 Save Current as Default", use_container_width=True):
    save_current_ui_settings(
        settings_mgr,
        ticker=ticker,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=initial_capital,
        position_size=position_size,
        commission=commission
    )
    if settings_mgr.save_settings():
        st.sidebar.success("✅ Saved as defaults!")
```

## Benefits You'll Get

1. **Persistence**: Settings survive app restarts
2. **Quick Testing**: Save different configurations (Conservative, Aggressive, etc.)
3. **Backup/Restore**: Export settings before making changes
4. **Sharing**: Export settings to share with others or use on different machines
5. **Version Control**: Settings file can be tracked in git

## Files Created

- `settings_manager.py` - The settings management module
- `trading_app_settings.json` - Auto-created on first save (your settings file)

## Notes

- The settings file is created in the same directory as your app
- You can have multiple settings files (e.g., `conservative_settings.json`, `aggressive_settings.json`)
- Settings are validated on load to ensure compatibility
- If settings file is corrupted, app falls back to defaults
