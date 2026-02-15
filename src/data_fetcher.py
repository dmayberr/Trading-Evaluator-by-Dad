"""
Data Fetcher Module
Automated data pulling for equity prices and options chains.
Primary: yfinance (reliable, no API key needed)
Includes retry logic and fallback methods for cloud deployment.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import streamlit as st
import time


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a yfinance DataFrame to standard column names."""
    if df.empty:
        return pd.DataFrame()

    # Handle MultiIndex columns from yf.download
    if isinstance(df.columns, pd.MultiIndex):
        # yf.download returns MultiIndex (Price, Ticker) — flatten
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Normalize timezone
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Standardize column names (handle both cases)
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[col] = lower
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[required]
    df.index.name = "date"

    # Drop any rows with NaN prices
    df = df.dropna(subset=["close"])

    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_equity_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a ticker.
    Tries Ticker.history first, then yf.download as fallback.
    Retries up to max_retries times with backoff.
    """
    # Method 1: yf.Ticker.history (preferred)
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            df = _normalize_df(df)
            if not df.empty:
                return df
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(1.5 * (attempt + 1))

    # Method 2: yf.download (better on cloud servers)
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True,
                timeout=15,
            )
            df = _normalize_df(df)
            if not df.empty:
                return df
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(1.5 * (attempt + 1))

    st.error(f"Unable to fetch data for {ticker} after {max_retries * 2} attempts. "
             f"Yahoo Finance may be temporarily rate-limiting. Try again in a minute.")
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

    for period in indicators.get("sma", []):
        df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

    for period in indicators.get("ema", []):
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    rsi_period = indicators.get("rsi", None)
    if rsi_period:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

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

    bb_cfg = indicators.get("bbands", None)
    if bb_cfg:
        period = bb_cfg.get("period", 20)
        std_dev = bb_cfg.get("std", 2)
        df["bb_mid"] = df["close"].rolling(window=period).mean()
        bb_std = df["close"].rolling(window=period).std()
        df["bb_upper"] = df["bb_mid"] + std_dev * bb_std
        df["bb_lower"] = df["bb_mid"] - std_dev * bb_std

    atr_period = indicators.get("atr", None)
    if atr_period:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_period).mean()

    stoch_cfg = indicators.get("stochastic", None)
    if stoch_cfg:
        k_period = stoch_cfg.get("k_period", 14)
        d_period = stoch_cfg.get("d_period", 3)
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()

    if "volume" in df.columns:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df
