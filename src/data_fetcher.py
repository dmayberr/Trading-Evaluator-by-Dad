"""
Data Fetcher Module
Automated data pulling for equity prices and options chains.
Primary: yfinance (reliable, no API key needed)
Fallback: Schwab API integration ready
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import streamlit as st


@st.cache_data(ttl=300, show_spinner=False)
def fetch_equity_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.tz_localize(None)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_chain(ticker: str) -> Tuple[list, dict]:
    """Fetch available expirations and full options chain."""
    try:
        stock = yf.Ticker(ticker)
        expirations = list(stock.options)
        chains = {}
        for exp in expirations:
            opt = stock.option_chain(exp)
            calls = opt.calls.copy()
            puts = opt.puts.copy()
            calls["type"] = "call"
            puts["type"] = "put"
            calls["expiration"] = exp
            puts["expiration"] = exp
            chains[exp] = pd.concat([calls, puts], ignore_index=True)
        return expirations, chains
    except Exception as e:
        st.error(f"Error fetching options for {ticker}: {e}")
        return [], {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_options_for_expiration(ticker: str, expiration: str) -> pd.DataFrame:
    """Fetch options chain for a specific expiration date."""
    try:
        stock = yf.Ticker(ticker)
        opt = stock.option_chain(expiration)
        calls = opt.calls.copy()
        puts = opt.puts.copy()
        calls["type"] = "call"
        puts["type"] = "put"
        calls["expiration"] = expiration
        puts["expiration"] = expiration
        return pd.concat([calls, puts], ignore_index=True)
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return pd.DataFrame()


def get_current_price(ticker: str) -> Optional[float]:
    """Get the current/last price for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        return info.get("lastPrice", None) or info.get("previousClose", None)
    except:
        return None


def compute_indicators(df: pd.DataFrame, indicators: dict) -> pd.DataFrame:
    """
    Add technical indicators to a DataFrame.
    indicators: dict of indicator configs, e.g.
    {
        "sma": [20, 50, 200],
        "ema": [12, 26],
        "rsi": 14,
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "bbands": {"period": 20, "std": 2},
        "atr": 14
    }
    """
    df = df.copy()

    # Simple Moving Averages
    for period in indicators.get("sma", []):
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

    # Exponential Moving Averages
    for period in indicators.get("ema", []):
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # RSI
    rsi_period = indicators.get("rsi", None)
    if rsi_period:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    macd_cfg = indicators.get("macd", None)
    if macd_cfg:
        fast = macd_cfg.get("fast", 12)
        slow = macd_cfg.get("slow", 26)
        signal = macd_cfg.get("signal", 9)
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb_cfg = indicators.get("bbands", None)
    if bb_cfg:
        period = bb_cfg.get("period", 20)
        std_dev = bb_cfg.get("std", 2)
        df["bb_mid"] = df["close"].rolling(window=period).mean()
        bb_std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_mid"] + std_dev * bb_std
        df["bb_lower"] = df["bb_mid"] - std_dev * bb_std

    # ATR
    atr_period = indicators.get("atr", None)
    if atr_period:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_period).mean()

    # Stochastic Oscillator
    stoch_cfg = indicators.get("stochastic", None)
    if stoch_cfg:
        k_period = stoch_cfg.get("k_period", 14)
        d_period = stoch_cfg.get("d_period", 3)
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()

    # VWAP (intraday approximation)
    if "volume" in df.columns:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df
