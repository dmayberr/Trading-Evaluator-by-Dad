# Settings Manager Quick Reference

## Quick Integration (3 Steps)

### 1. Import at Top of App
```python
from settings_manager import (
    SettingsManager, 
    render_settings_sidebar, 
    initialize_settings_in_session_state,
    save_current_ui_settings
)
```

### 2. Initialize Early in App
```python
if 'settings_manager' not in st.session_state:
    st.session_state.settings_manager = SettingsManager()

settings_mgr = st.session_state.settings_manager
initialize_settings_in_session_state(settings_mgr)
```

### 3. Add UI Controls
```python
render_settings_sidebar(settings_mgr)
```

## Common Patterns

### Get a Setting with Default
```python
ticker = settings_mgr.get("default_ticker", "SPY")
```

### Set a Setting
```python
settings_mgr.set("default_ticker", "AAPL")
```

### Save Settings to File
```python
if settings_mgr.save_settings():
    st.success("Saved!")
```

### Convert Saved String Dates to datetime
```python
from datetime import datetime

default_start = datetime.strptime(
    settings_mgr.get("default_start_date"), 
    "%Y-%m-%d"
).date()
```

### Use Setting in Streamlit Input
```python
# Text input
ticker = st.text_input(
    "Ticker", 
    value=settings_mgr.get("default_ticker", "SPY")
)

# Number input
capital = st.number_input(
    "Capital", 
    value=float(settings_mgr.get("default_initial_capital", 10000.0))
)

# Date input
default_date = datetime.strptime(
    settings_mgr.get("default_start_date"), 
    "%Y-%m-%d"
).date()
start_date = st.date_input("Start", value=default_date)

# Slider
confidence = st.slider(
    "Confidence", 
    value=int(settings_mgr.get("monte_carlo_confidence_level", 95))
)
```

### Save All Current UI Values
```python
save_current_ui_settings(
    settings_mgr,
    ticker=ticker,
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d"),
    initial_capital=initial_capital,
    position_size=position_size,
    commission=commission
)
settings_mgr.save_settings()
```

## Default Settings Available

```python
{
    "default_ticker": "SPY",
    "default_start_date": "3 years ago",
    "default_end_date": "today",
    "default_initial_capital": 10000.0,
    "default_position_size": 1.0,
    "default_commission": 0.0,
    "monte_carlo_simulations": 1000,
    "monte_carlo_confidence_level": 95,
    "risk_free_rate": 0.045,
    "benchmark_ticker": "SPY",
    "chart_theme": "plotly",
    "show_debug_info": False,
    "auto_run_backtest": False,
    "default_strategy": None,
    "favorite_strategies": []
}
```

## Advanced Usage

### Add Custom Settings
```python
# Add your own custom setting
settings_mgr.set("my_custom_setting", "custom_value")
settings_mgr.save_settings()

# Later, retrieve it
custom_value = settings_mgr.get("my_custom_setting", "default_value")
```

### Export/Import Settings
```python
# Export
settings_mgr.export_settings("my_settings_backup.json")

# Import
settings_mgr.import_settings("my_settings_backup.json")
```

### Create Multiple Profiles
```python
# Save current as "aggressive"
settings_mgr.config_file = "aggressive_settings.json"
settings_mgr.save_settings()

# Save current as "conservative"
settings_mgr.config_file = "conservative_settings.json"
settings_mgr.save_settings()

# Later, load a specific profile
settings_mgr.config_file = "aggressive_settings.json"
settings_mgr.settings = settings_mgr._load_settings()
```

### Update Multiple Settings at Once
```python
new_settings = {
    "default_ticker": "TSLA",
    "default_initial_capital": 50000.0,
    "monte_carlo_simulations": 5000
}
settings_mgr.update(new_settings)
settings_mgr.save_settings()
```

## Troubleshooting

### Settings Not Persisting?
1. Check that `save_settings()` returns `True`
2. Verify `trading_app_settings.json` exists in app directory
3. Check file permissions

### Reset Everything
```python
# Delete the settings file and restart app
import os
if os.path.exists("trading_app_settings.json"):
    os.remove("trading_app_settings.json")
st.rerun()
```

### Settings Corrupted?
```python
# App automatically falls back to defaults if JSON is invalid
# Or manually reset:
settings_mgr.reset_to_defaults()
settings_mgr.save_settings()
```

## File Locations

- **Settings file:** `trading_app_settings.json` (same dir as app)
- **Exported backups:** `settings_backup_YYYYMMDD_HHMMSS.json`
- **Custom profiles:** Any `.json` file you specify

## UI Components

### Basic Save/Load Buttons
```python
col1, col2 = st.columns(2)
with col1:
    if st.button("Save"):
        settings_mgr.save_settings()
with col2:
    if st.button("Load"):
        settings_mgr.settings = settings_mgr._load_settings()
        st.rerun()
```

### Full Management Panel (Recommended)
```python
render_settings_sidebar(settings_mgr)
```

This includes:
- Save Current
- Reload
- Reset to Defaults
- Export Settings
- Import Settings
