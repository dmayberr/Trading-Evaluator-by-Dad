"""
Strategy Engine Module
Defines equity/directional and options strategies with flexible parameters.
Each strategy returns a DataFrame of signals and a trade log.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum


class SignalType(Enum):
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"
    NONE = "none"


@dataclass
class Trade:
    entry_date: Any
    exit_date: Any = None
    direction: str = "long"
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    strategy: str = ""
    # Options fields
    option_type: str = ""
    strike: float = 0.0
    premium: float = 0.0
    expiration: str = ""
    legs: list = field(default_factory=list)
    # MAE / MFE fields (% from entry)
    mae_pct: float = 0.0   # Maximum Adverse Excursion (worst drawdown during trade)
    mfe_pct: float = 0.0   # Maximum Favorable Excursion (best unrealized gain)
    # Equity at entry/exit for equity curve markers
    equity_at_entry: float = 0.0
    equity_at_exit: float = 0.0


# ============================================================
# EQUITY / DIRECTIONAL STRATEGIES
# ============================================================

def strategy_sma_crossover(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """SMA Crossover: fast SMA crosses above/below slow SMA."""
    fast = params.get("fast_period", 10)
    slow = params.get("slow_period", 30)
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(window=fast).mean()
    df["sma_slow"] = df["close"].rolling(window=slow).mean()
    df["signal"] = SignalType.NONE.value
    prev_fast = df["sma_fast"].shift(1)
    prev_slow = df["sma_slow"].shift(1)
    df.loc[(prev_fast <= prev_slow) & (df["sma_fast"] > df["sma_slow"]), "signal"] = SignalType.LONG_ENTRY.value
    df.loc[(prev_fast >= prev_slow) & (df["sma_fast"] < df["sma_slow"]), "signal"] = SignalType.LONG_EXIT.value
    return df


def strategy_ema_crossover(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """EMA Crossover strategy."""
    fast = params.get("fast_period", 9)
    slow = params.get("slow_period", 21)
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["signal"] = SignalType.NONE.value
    prev_fast = df["ema_fast"].shift(1)
    prev_slow = df["ema_slow"].shift(1)
    df.loc[(prev_fast <= prev_slow) & (df["ema_fast"] > df["ema_slow"]), "signal"] = SignalType.LONG_ENTRY.value
    df.loc[(prev_fast >= prev_slow) & (df["ema_fast"] < df["ema_slow"]), "signal"] = SignalType.LONG_EXIT.value
    return df


def strategy_rsi(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """RSI Overbought/Oversold strategy."""
    period = params.get("rsi_period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["signal"] = SignalType.NONE.value
    prev_rsi = df["rsi"].shift(1)
    df.loc[(prev_rsi <= oversold) & (df["rsi"] > oversold), "signal"] = SignalType.LONG_ENTRY.value
    df.loc[(prev_rsi >= overbought) & (df["rsi"] < overbought), "signal"] = SignalType.LONG_EXIT.value
    return df


def strategy_macd(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """MACD Crossover strategy."""
    fast = params.get("fast_period", 12)
    slow = params.get("slow_period", 26)
    signal_period = params.get("signal_period", 9)
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["signal"] = SignalType.NONE.value
    prev_macd = df["macd"].shift(1)
    prev_signal = df["macd_signal"].shift(1)
    df.loc[(prev_macd <= prev_signal) & (df["macd"] > df["macd_signal"]), "signal"] = SignalType.LONG_ENTRY.value
    df.loc[(prev_macd >= prev_signal) & (df["macd"] < df["macd_signal"]), "signal"] = SignalType.LONG_EXIT.value
    return df


def strategy_bollinger_bands(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Bollinger Bands mean reversion strategy."""
    period = params.get("bb_period", 20)
    std_dev = params.get("bb_std", 2.0)
    df = df.copy()
    df["bb_mid"] = df["close"].rolling(window=period).mean()
    bb_std = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_mid"] + std_dev * bb_std
    df["bb_lower"] = df["bb_mid"] - std_dev * bb_std
    df["signal"] = SignalType.NONE.value
    df.loc[df["close"] < df["bb_lower"], "signal"] = SignalType.LONG_ENTRY.value
    df.loc[df["close"] > df["bb_upper"], "signal"] = SignalType.LONG_EXIT.value
    return df


def strategy_stochastic(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Stochastic Oscillator strategy."""
    k_period = params.get("k_period", 14)
    d_period = params.get("d_period", 3)
    oversold = params.get("oversold", 20)
    overbought = params.get("overbought", 80)
    df = df.copy()
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
    df["signal"] = SignalType.NONE.value
    prev_k = df["stoch_k"].shift(1)
    prev_d = df["stoch_d"].shift(1)
    df.loc[(prev_k <= prev_d) & (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] < oversold + 10), "signal"] = SignalType.LONG_ENTRY.value
    df.loc[(prev_k >= prev_d) & (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"] > overbought - 10), "signal"] = SignalType.LONG_EXIT.value
    return df


def strategy_triple_ema(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Triple EMA trend strategy (TEMA alignment)."""
    fast = params.get("fast_period", 8)
    mid = params.get("mid_period", 21)
    slow = params.get("slow_period", 55)
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_mid"] = df["close"].ewm(span=mid, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["signal"] = SignalType.NONE.value
    bullish = (df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"])
    bearish = (df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"])
    prev_bullish = bullish.shift(1).fillna(False).infer_objects(copy=False)
    prev_bearish = bearish.shift(1).fillna(False).infer_objects(copy=False)
    df.loc[bullish & ~prev_bullish, "signal"] = SignalType.LONG_ENTRY.value
    df.loc[bearish & ~prev_bearish, "signal"] = SignalType.LONG_EXIT.value
    return df


def strategy_supertrend(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    SuperTrend: ATR-based dynamic trend-following bands.
    Buy when price crosses above SuperTrend, sell when it crosses below.
    """
    atr_period = params.get("atr_period", 10)
    multiplier = params.get("atr_multiplier", 3.0)
    df = df.copy()

    # Calculate ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    # Calculate basic upper and lower bands
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    # Initialize SuperTrend arrays
    n = len(df)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)

    final_upper[0] = basic_upper.iloc[0] if not pd.isna(basic_upper.iloc[0]) else 0
    final_lower[0] = basic_lower.iloc[0] if not pd.isna(basic_lower.iloc[0]) else 0
    supertrend[0] = final_upper[0]
    direction[0] = -1

    for i in range(1, n):
        bu = basic_upper.iloc[i]
        bl = basic_lower.iloc[i]

        if pd.isna(bu) or pd.isna(bl):
            final_upper[i] = final_upper[i - 1]
            final_lower[i] = final_lower[i - 1]
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
            continue

        # Final upper band: use lower of current basic_upper and previous final_upper
        # if previous close was above previous final_upper
        if df["close"].iloc[i - 1] <= final_upper[i - 1]:
            final_upper[i] = bu
        else:
            final_upper[i] = min(bu, final_upper[i - 1])

        # Final lower band: use higher of current basic_lower and previous final_lower
        # if previous close was above previous final_lower
        if df["close"].iloc[i - 1] >= final_lower[i - 1]:
            final_lower[i] = max(bl, final_lower[i - 1])
        else:
            final_lower[i] = bl

        # Determine direction
        if direction[i - 1] == -1:  # was bearish
            if df["close"].iloc[i] > final_upper[i]:
                direction[i] = 1  # flip bullish
                supertrend[i] = final_lower[i]
            else:
                direction[i] = -1
                supertrend[i] = final_upper[i]
        else:  # was bullish
            if df["close"].iloc[i] < final_lower[i]:
                direction[i] = -1  # flip bearish
                supertrend[i] = final_upper[i]
            else:
                direction[i] = 1
                supertrend[i] = final_lower[i]

    df["supertrend"] = supertrend
    df["st_direction"] = direction

    # Generate signals on direction flips
    df["signal"] = SignalType.NONE.value
    prev_dir = df["st_direction"].shift(1)
    df.loc[(prev_dir == -1) & (df["st_direction"] == 1), "signal"] = SignalType.LONG_ENTRY.value
    df.loc[(prev_dir == 1) & (df["st_direction"] == -1), "signal"] = SignalType.LONG_EXIT.value

    return df


# ============================================================
# OPTIONS STRATEGIES (Simulated Backtesting)
# ============================================================

def strategy_covered_call(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Covered Call: Buy stock, sell OTM call.
    Simulates selling a call at X% OTM every N days.
    """
    otm_pct = params.get("otm_pct", 5.0) / 100
    hold_days = params.get("hold_days", 30)
    df = df.copy()
    df["signal"] = SignalType.NONE.value
    df["option_strike"] = 0.0
    df["option_premium"] = 0.0
    for i in range(0, len(df), hold_days):
        if i < len(df):
            price = df.iloc[i]["close"]
            strike = price * (1 + otm_pct)
            # Estimate premium as ~2% of stock price (simplified)
            premium = price * 0.02
            df.iloc[i, df.columns.get_loc("signal")] = SignalType.LONG_ENTRY.value
            df.iloc[i, df.columns.get_loc("option_strike")] = strike
            df.iloc[i, df.columns.get_loc("option_premium")] = premium
            exit_idx = min(i + hold_days, len(df) - 1)
            df.iloc[exit_idx, df.columns.get_loc("signal")] = SignalType.LONG_EXIT.value
    return df


def strategy_bull_put_spread(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Bull Put Spread: Sell higher strike put, buy lower strike put.
    Enter on bullish signals, hold for N days.
    """
    rsi_period = params.get("rsi_period", 14)
    rsi_entry = params.get("rsi_entry", 35)
    spread_width_pct = params.get("spread_width_pct", 3.0) / 100
    hold_days = params.get("hold_days", 21)
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["signal"] = SignalType.NONE.value
    df["spread_credit"] = 0.0
    i = 0
    in_trade = False
    exit_at = 0
    while i < len(df):
        if not in_trade and df.iloc[i]["rsi"] < rsi_entry if not pd.isna(df.iloc[i].get("rsi", np.nan)) else False:
            df.iloc[i, df.columns.get_loc("signal")] = SignalType.LONG_ENTRY.value
            credit = df.iloc[i]["close"] * spread_width_pct * 0.3
            df.iloc[i, df.columns.get_loc("spread_credit")] = credit
            in_trade = True
            exit_at = min(i + hold_days, len(df) - 1)
        elif in_trade and i >= exit_at:
            df.iloc[i, df.columns.get_loc("signal")] = SignalType.LONG_EXIT.value
            in_trade = False
        i += 1
    return df


def strategy_iron_condor(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Iron Condor: Sell OTM call spread + sell OTM put spread.
    Enter during low volatility, collect premium.
    """
    bb_period = params.get("bb_period", 20)
    bb_squeeze_threshold = params.get("squeeze_threshold", 0.04)
    hold_days = params.get("hold_days", 30)
    wing_width_pct = params.get("wing_width_pct", 5.0) / 100
    df = df.copy()
    df["bb_mid"] = df["close"].rolling(window=bb_period).mean()
    bb_std = df["close"].rolling(window=bb_period).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["signal"] = SignalType.NONE.value
    df["ic_credit"] = 0.0
    i = 0
    in_trade = False
    exit_at = 0
    while i < len(df):
        if not in_trade:
            bw = df.iloc[i].get("bb_width", np.nan)
            if not pd.isna(bw) and bw < bb_squeeze_threshold:
                df.iloc[i, df.columns.get_loc("signal")] = SignalType.LONG_ENTRY.value
                credit = df.iloc[i]["close"] * wing_width_pct * 0.25
                df.iloc[i, df.columns.get_loc("ic_credit")] = credit
                in_trade = True
                exit_at = min(i + hold_days, len(df) - 1)
        elif in_trade and i >= exit_at:
            df.iloc[i, df.columns.get_loc("signal")] = SignalType.LONG_EXIT.value
            in_trade = False
        i += 1
    return df


def strategy_long_straddle(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Long Straddle: Buy ATM call + ATM put before expected big move.
    Enter when BB squeeze is very tight, expecting expansion.
    """
    bb_period = params.get("bb_period", 20)
    squeeze_threshold = params.get("squeeze_threshold", 0.03)
    hold_days = params.get("hold_days", 14)
    df = df.copy()
    df["bb_mid"] = df["close"].rolling(window=bb_period).mean()
    bb_std = df["close"].rolling(window=bb_period).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["signal"] = SignalType.NONE.value
    df["straddle_cost"] = 0.0
    i = 0
    in_trade = False
    exit_at = 0
    while i < len(df):
        if not in_trade:
            bw = df.iloc[i].get("bb_width", np.nan)
            if not pd.isna(bw) and bw < squeeze_threshold:
                df.iloc[i, df.columns.get_loc("signal")] = SignalType.LONG_ENTRY.value
                cost = df.iloc[i]["close"] * 0.04  # ~4% premium for straddle
                df.iloc[i, df.columns.get_loc("straddle_cost")] = cost
                in_trade = True
                exit_at = min(i + hold_days, len(df) - 1)
        elif in_trade and i >= exit_at:
            df.iloc[i, df.columns.get_loc("signal")] = SignalType.LONG_EXIT.value
            in_trade = False
        i += 1
    return df


# ============================================================
# STRATEGY REGISTRY
# ============================================================

EQUITY_STRATEGIES = {
    "SMA Crossover": {
        "func": strategy_sma_crossover,
        "params": {
            "fast_period": {"default": 10, "min": 2, "max": 100, "step": 1, "label": "Fast SMA Period"},
            "slow_period": {"default": 30, "min": 5, "max": 300, "step": 1, "label": "Slow SMA Period"},
        },
        "description": "Buy when fast SMA crosses above slow SMA, sell on cross below."
    },
    "EMA Crossover": {
        "func": strategy_ema_crossover,
        "params": {
            "fast_period": {"default": 9, "min": 2, "max": 100, "step": 1, "label": "Fast EMA Period"},
            "slow_period": {"default": 21, "min": 5, "max": 200, "step": 1, "label": "Slow EMA Period"},
        },
        "description": "Buy when fast EMA crosses above slow EMA, sell on cross below."
    },
    "RSI Reversal": {
        "func": strategy_rsi,
        "params": {
            "rsi_period": {"default": 14, "min": 2, "max": 50, "step": 1, "label": "RSI Period"},
            "oversold": {"default": 30, "min": 10, "max": 45, "step": 1, "label": "Oversold Level"},
            "overbought": {"default": 70, "min": 55, "max": 90, "step": 1, "label": "Overbought Level"},
        },
        "description": "Buy when RSI crosses above oversold, sell when crosses below overbought."
    },
    "MACD Crossover": {
        "func": strategy_macd,
        "params": {
            "fast_period": {"default": 12, "min": 2, "max": 50, "step": 1, "label": "MACD Fast"},
            "slow_period": {"default": 26, "min": 5, "max": 100, "step": 1, "label": "MACD Slow"},
            "signal_period": {"default": 9, "min": 2, "max": 30, "step": 1, "label": "Signal Period"},
        },
        "description": "Buy on MACD line crossing above signal line, sell on cross below."
    },
    "Bollinger Bands": {
        "func": strategy_bollinger_bands,
        "params": {
            "bb_period": {"default": 20, "min": 5, "max": 100, "step": 1, "label": "BB Period"},
            "bb_std": {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1, "label": "Std Deviations"},
        },
        "description": "Mean reversion: buy at lower band, sell at upper band."
    },
    "Stochastic Oscillator": {
        "func": strategy_stochastic,
        "params": {
            "k_period": {"default": 14, "min": 5, "max": 50, "step": 1, "label": "%K Period"},
            "d_period": {"default": 3, "min": 2, "max": 10, "step": 1, "label": "%D Period"},
            "oversold": {"default": 20, "min": 5, "max": 40, "step": 1, "label": "Oversold Level"},
            "overbought": {"default": 80, "min": 60, "max": 95, "step": 1, "label": "Overbought Level"},
        },
        "description": "Buy on bullish %K/%D crossover in oversold zone, sell in overbought."
    },
    "Triple EMA": {
        "func": strategy_triple_ema,
        "params": {
            "fast_period": {"default": 8, "min": 2, "max": 50, "step": 1, "label": "Fast EMA"},
            "mid_period": {"default": 21, "min": 5, "max": 100, "step": 1, "label": "Mid EMA"},
            "slow_period": {"default": 55, "min": 20, "max": 300, "step": 1, "label": "Slow EMA"},
        },
        "description": "Enter when all three EMAs align bullish, exit on bearish alignment."
    },
    "SuperTrend": {
        "func": strategy_supertrend,
        "params": {
            "atr_period": {"default": 10, "min": 5, "max": 30, "step": 1, "label": "ATR Period"},
            "atr_multiplier": {"default": 3.0, "min": 1.0, "max": 6.0, "step": 0.1, "label": "ATR Multiplier"},
        },
        "description": "ATR-based trend bands. Buy when price flips above SuperTrend, sell on flip below."
    },
}

OPTIONS_STRATEGIES = {
    "Covered Call": {
        "func": strategy_covered_call,
        "params": {
            "otm_pct": {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.5, "label": "OTM % for Strike"},
            "hold_days": {"default": 30, "min": 7, "max": 90, "step": 1, "label": "Days Between Rolls"},
        },
        "description": "Own stock + sell OTM call. Collect premium, cap upside."
    },
    "Bull Put Spread": {
        "func": strategy_bull_put_spread,
        "params": {
            "rsi_period": {"default": 14, "min": 5, "max": 30, "step": 1, "label": "RSI Period"},
            "rsi_entry": {"default": 35, "min": 15, "max": 50, "step": 1, "label": "RSI Entry Below"},
            "spread_width_pct": {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "label": "Spread Width %"},
            "hold_days": {"default": 21, "min": 7, "max": 60, "step": 1, "label": "Hold Days"},
        },
        "description": "Sell put spread on RSI oversold. Bullish credit strategy."
    },
    "Iron Condor": {
        "func": strategy_iron_condor,
        "params": {
            "bb_period": {"default": 20, "min": 10, "max": 50, "step": 1, "label": "BB Period"},
            "squeeze_threshold": {"default": 0.04, "min": 0.01, "max": 0.10, "step": 0.005, "label": "Squeeze Threshold"},
            "hold_days": {"default": 30, "min": 7, "max": 60, "step": 1, "label": "Hold Days"},
            "wing_width_pct": {"default": 5.0, "min": 2.0, "max": 15.0, "step": 0.5, "label": "Wing Width %"},
        },
        "description": "Sell call + put spread during low volatility. Neutral income strategy."
    },
    "Long Straddle": {
        "func": strategy_long_straddle,
        "params": {
            "bb_period": {"default": 20, "min": 10, "max": 50, "step": 1, "label": "BB Period"},
            "squeeze_threshold": {"default": 0.03, "min": 0.01, "max": 0.08, "step": 0.005, "label": "Squeeze Threshold"},
            "hold_days": {"default": 14, "min": 5, "max": 45, "step": 1, "label": "Hold Days"},
        },
        "description": "Buy ATM call + put during BB squeeze. Profit from big moves either way."
    },
}

ALL_STRATEGIES = {**EQUITY_STRATEGIES, **OPTIONS_STRATEGIES}
