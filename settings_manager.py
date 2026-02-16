"""
Settings Manager for Trading Strategy Application
Handles saving, loading, and managing user preferences and default settings
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import streamlit as st


class SettingsManager:
    """Manages application settings with file-based persistence"""
    
    DEFAULT_SETTINGS = {
        "default_ticker": "SPY",
        "default_start_date": (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d"),
        "default_end_date": datetime.now().strftime("%Y-%m-%d"),
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
    
    def __init__(self, config_file: str = "trading_app_settings.json"):
        """
        Initialize the settings manager
        
        Args:
            config_file: Path to the JSON configuration file
        """
        self.config_file = config_file
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from file, or return defaults if file doesn't exist
        
        Returns:
            Dictionary of settings
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                settings = self.DEFAULT_SETTINGS.copy()
                settings.update(loaded_settings)
                return settings
            except (json.JSONDecodeError, IOError) as e:
                st.warning(f"Could not load settings: {e}. Using defaults.")
                return self.DEFAULT_SETTINGS.copy()
        else:
            return self.DEFAULT_SETTINGS.copy()
    
    def save_settings(self) -> bool:
        """
        Save current settings to file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except IOError as e:
            st.error(f"Could not save settings: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values"""
        self.settings = self.DEFAULT_SETTINGS.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value
        
        Args:
            key: Setting key
            default: Default value if key doesn't exist
            
        Returns:
            Setting value
        """
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a setting value
        
        Args:
            key: Setting key
            value: Setting value
        """
        self.settings[key] = value
    
    def update(self, settings_dict: Dict[str, Any]) -> None:
        """
        Update multiple settings at once
        
        Args:
            settings_dict: Dictionary of settings to update
        """
        self.settings.update(settings_dict)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings"""
        return self.settings.copy()
    
    def export_settings(self, filepath: str) -> bool:
        """
        Export settings to a custom file path
        
        Args:
            filepath: Path to export to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except IOError as e:
            st.error(f"Could not export settings: {e}")
            return False
    
    def import_settings(self, filepath: str) -> bool:
        """
        Import settings from a custom file path
        
        Args:
            filepath: Path to import from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                imported_settings = json.load(f)
            # Validate that imported settings are compatible
            self.settings.update(imported_settings)
            return True
        except (json.JSONDecodeError, IOError) as e:
            st.error(f"Could not import settings: {e}")
            return False


def render_settings_sidebar(settings_manager: SettingsManager) -> None:
    """
    Render settings management UI in Streamlit sidebar
    
    Args:
        settings_manager: SettingsManager instance
    """
    with st.sidebar.expander("⚙️ Settings Management", expanded=False):
        st.markdown("### Save/Load Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Current", use_container_width=True):
                if settings_manager.save_settings():
                    st.success("Settings saved!")
                else:
                    st.error("Failed to save settings")
        
        with col2:
            if st.button("📂 Reload", use_container_width=True):
                settings_manager.settings = settings_manager._load_settings()
                st.success("Settings reloaded!")
                st.rerun()
        
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            settings_manager.reset_to_defaults()
            st.success("Settings reset to defaults!")
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Import/Export")
        
        # Export settings
        if st.button("⬇️ Export Settings", use_container_width=True):
            export_path = f"settings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if settings_manager.export_settings(export_path):
                st.success(f"Settings exported to: {export_path}")
        
        # Import settings
        uploaded_file = st.file_uploader("⬆️ Import Settings", type=['json'], key="settings_import")
        if uploaded_file is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if settings_manager.import_settings(tmp_path):
                st.success("Settings imported successfully!")
                st.rerun()
            
            os.unlink(tmp_path)


def initialize_settings_in_session_state(settings_manager: SettingsManager) -> None:
    """
    Initialize Streamlit session state with saved settings
    
    Args:
        settings_manager: SettingsManager instance
    """
    settings = settings_manager.get_all()
    
    # Initialize session state with saved settings if not already set
    if 'settings_initialized' not in st.session_state:
        for key, value in settings.items():
            session_key = f"setting_{key}"
            if session_key not in st.session_state:
                st.session_state[session_key] = value
        st.session_state['settings_initialized'] = True


def save_current_ui_settings(settings_manager: SettingsManager, 
                             ticker: str,
                             start_date: str,
                             end_date: str,
                             initial_capital: float,
                             position_size: float,
                             commission: float,
                             mc_sims: int = None,
                             mc_confidence: int = None,
                             risk_free_rate: float = None,
                             benchmark: str = None) -> None:
    """
    Save current UI settings to the settings manager
    
    Args:
        settings_manager: SettingsManager instance
        ticker: Current ticker symbol
        start_date: Current start date
        end_date: Current end date
        initial_capital: Current initial capital
        position_size: Current position size
        commission: Current commission rate
        mc_sims: Monte Carlo simulations count (optional)
        mc_confidence: Monte Carlo confidence level (optional)
        risk_free_rate: Risk-free rate (optional)
        benchmark: Benchmark ticker (optional)
    """
    settings_manager.set("default_ticker", ticker)
    settings_manager.set("default_start_date", start_date)
    settings_manager.set("default_end_date", end_date)
    settings_manager.set("default_initial_capital", initial_capital)
    settings_manager.set("default_position_size", position_size)
    settings_manager.set("default_commission", commission)
    
    if mc_sims is not None:
        settings_manager.set("monte_carlo_simulations", mc_sims)
    if mc_confidence is not None:
        settings_manager.set("monte_carlo_confidence_level", mc_confidence)
    if risk_free_rate is not None:
        settings_manager.set("risk_free_rate", risk_free_rate)
    if benchmark is not None:
        settings_manager.set("benchmark_ticker", benchmark)
