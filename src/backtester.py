"""
Backtester Module
Executes strategies, manages positions with risk controls, computes performance metrics.
Supports multiple position sizing modes and multi-ticker portfolio backtesting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.strategies import Trade, SignalType, ALL_STRATEGIES


# ============================================================
# POSITION SIZING MODES
# ============================================================

def calc_position_size(
    mode: str,
    equity: float,
    capital: float,
    price: float,
    position_size_pct: float,
    reinvest_pct: float = 50.0,
) -> float:
    """
    Calculate number of shares based on sizing mode.

    Modes:
        'compound'   - Size based on current equity (full reinvest)
        'fixed'      - Always size based on initial capital (no compounding)
        'fractional' - Reinvest X% of profits, keep base from initial capital
    """
    if mode == "fixed":
        base = capital * (position_size_pct / 100)
    elif mode == "fractional":
        profits = max(equity - capital, 0)
        reinvested = profits * (reinvest_pct / 100)
        base = (capital + reinvested) * (position_size_pct / 100)
        # If in drawdown, use current equity
        if equity < capital:
            base = equity * (position_size_pct / 100)
    else:  # compound (default)
        base = equity * (position_size_pct / 100)

    shares = base / price if price > 0 else 0
    return shares


# ============================================================
# SINGLE-TICKER BACKTEST
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    capital: float = 100000.0,
    position_size_pct: float = 100.0,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    trailing_stop_pct: Optional[float] = None,
    commission: float = 0.0,
    sizing_mode: str = "compound",
    reinvest_pct: float = 50.0,
    ticker: str = "",
) -> Tuple[pd.DataFrame, List[Trade], pd.DataFrame]:
    """
    Run a full backtest with configurable position sizing.
    Returns: (signal_df, trades_list, equity_curve_df)
    """
    strategy_config = ALL_STRATEGIES.get(strategy_name)
    if not strategy_config:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    strategy_func = strategy_config["func"]
    signal_df = strategy_func(df.copy(), params)

    trades = []
    equity = capital
    equity_curve = []
    in_position = False
    current_trade = None
    highest_since_entry = 0.0
    lowest_since_entry = float("inf")

    for i, (date, row) in enumerate(signal_df.iterrows()):
        if in_position:
            current_price = row["close"]
            highest_since_entry = max(highest_since_entry, current_price)
            lowest_since_entry = min(lowest_since_entry, current_price)
            exit_reason = ""

            # Check stop loss
            if stop_loss_pct is not None:
                sl_price = current_trade.entry_price * (1 - stop_loss_pct / 100)
                if current_price <= sl_price:
                    exit_reason = "Stop Loss"

            # Check take profit
            if not exit_reason and take_profit_pct is not None:
                tp_price = current_trade.entry_price * (1 + take_profit_pct / 100)
                if current_price >= tp_price:
                    exit_reason = "Take Profit"

            # Check trailing stop
            if not exit_reason and trailing_stop_pct is not None:
                ts_price = highest_since_entry * (1 - trailing_stop_pct / 100)
                if current_price <= ts_price:
                    exit_reason = "Trailing Stop"

            # Check signal exit
            if not exit_reason and row["signal"] == SignalType.LONG_EXIT.value:
                exit_reason = "Signal"

            # Force exit on last bar
            if not exit_reason and i == len(signal_df) - 1:
                exit_reason = "End of Data"

            if exit_reason:
                current_trade.exit_date = date
                current_trade.exit_price = current_price
                pnl_gross = (current_price - current_trade.entry_price) * current_trade.shares
                total_commission = commission * 2 * current_trade.shares
                current_trade.pnl = pnl_gross - total_commission
                current_trade.pnl_pct = ((current_price / current_trade.entry_price) - 1) * 100
                current_trade.exit_reason = exit_reason
                # MAE / MFE as % from entry
                current_trade.mae_pct = ((lowest_since_entry / current_trade.entry_price) - 1) * 100
                current_trade.mfe_pct = ((highest_since_entry / current_trade.entry_price) - 1) * 100
                equity += current_trade.pnl
                current_trade.equity_at_exit = equity
                trades.append(current_trade)
                in_position = False
                current_trade = None

        elif row["signal"] == SignalType.LONG_ENTRY.value and not in_position:
            shares = calc_position_size(
                sizing_mode, equity, capital, row["close"],
                position_size_pct, reinvest_pct
            )
            current_trade = Trade(
                entry_date=date,
                direction="long",
                entry_price=row["close"],
                shares=shares,
                strategy=strategy_name,
            )
            # Store ticker on trade for portfolio tracking
            current_trade.option_type = ticker  # reuse field for ticker label
            current_trade.equity_at_entry = equity
            highest_since_entry = row["close"]
            lowest_since_entry = row["close"]
            in_position = True

        equity_curve.append({
            "date": date,
            "equity": equity if not in_position else equity + (
                (row["close"] - current_trade.entry_price) * current_trade.shares
                if current_trade else 0
            ),
            "price": row["close"],
        })

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return signal_df, trades, equity_df


# ============================================================
# PORTFOLIO BACKTEST (MULTI-TICKER)
# ============================================================

def run_portfolio_backtest(
    portfolio_configs: List[Dict[str, Any]],
    capital: float = 100000.0,
    sizing_mode: str = "compound",
    reinvest_pct: float = 50.0,
) -> Dict[str, Any]:
    """
    Run backtests across multiple tickers with capital allocation.

    portfolio_configs: list of dicts, each with:
        {
            "ticker": str,
            "df": pd.DataFrame,
            "strategy_name": str,
            "params": dict,
            "allocation_pct": float,  # % of capital allocated
            "stop_loss_pct": float or None,
            "take_profit_pct": float or None,
            "trailing_stop_pct": float or None,
            "commission": float,
        }

    Returns dict with per-ticker results and combined portfolio metrics.
    """
    results = {}
    all_trades = []
    equity_curves = {}

    for config in portfolio_configs:
        ticker = config["ticker"]
        alloc = config.get("allocation_pct", 100.0 / len(portfolio_configs))
        ticker_capital = capital * (alloc / 100)

        signal_df, trades, equity_df = run_backtest(
            df=config["df"],
            strategy_name=config["strategy_name"],
            params=config["params"],
            capital=ticker_capital,
            position_size_pct=100.0,  # 100% of allocated capital
            stop_loss_pct=config.get("stop_loss_pct"),
            take_profit_pct=config.get("take_profit_pct"),
            trailing_stop_pct=config.get("trailing_stop_pct"),
            commission=config.get("commission", 0.0),
            sizing_mode=sizing_mode,
            reinvest_pct=reinvest_pct,
            ticker=ticker,
        )

        metrics = compute_metrics(trades, equity_df, ticker_capital)

        results[ticker] = {
            "signal_df": signal_df,
            "trades": trades,
            "equity_df": equity_df,
            "metrics": metrics,
            "allocation_pct": alloc,
            "capital": ticker_capital,
        }

        # Tag trades with ticker
        for t in trades:
            t.option_type = ticker
        all_trades.extend(trades)

        equity_curves[ticker] = equity_df["equity"]

    # Build combined portfolio equity curve
    combined_equity = _combine_equity_curves(equity_curves, capital)
    combined_metrics = compute_metrics(all_trades, combined_equity, capital)

    return {
        "per_ticker": results,
        "combined_trades": all_trades,
        "combined_equity": combined_equity,
        "combined_metrics": combined_metrics,
    }


def _combine_equity_curves(curves: Dict[str, pd.Series], capital: float) -> pd.DataFrame:
    """Merge per-ticker equity curves into a single portfolio equity curve."""
    if not curves:
        return pd.DataFrame(columns=["equity"])

    combined = pd.DataFrame(curves)
    combined = combined.ffill().bfill()
    combined["equity"] = combined.sum(axis=1)
    return combined[["equity"]]


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades: List[Trade], equity_df: pd.DataFrame, capital: float) -> Dict[str, Any]:
    """Compute comprehensive performance metrics from trade results."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "avg_hold_days": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "win_loss_ratio": 0.0,
        }

    pnls = [t.pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    win_rate = len(winners) / len(trades) * 100 if trades else 0

    # Hold durations
    hold_days = []
    for t in trades:
        if t.entry_date is not None and t.exit_date is not None:
            try:
                delta = (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
                hold_days.append(max(delta, 1))
            except:
                hold_days.append(0)

    # Max drawdown
    if not equity_df.empty:
        eq = equity_df["equity"] if "equity" in equity_df.columns else equity_df.iloc[:, 0]
        peak = eq.expanding().max()
        drawdown = (eq - peak) / peak * 100
        max_dd = drawdown.min()
    else:
        max_dd = 0.0

    # Daily returns for Sharpe/Sortino
    if not equity_df.empty and len(equity_df) > 1:
        eq = equity_df["equity"] if "equity" in equity_df.columns else equity_df.iloc[:, 0]
        daily_returns = eq.pct_change().dropna()
        avg_daily_return = daily_returns.mean()
        std_daily_return = daily_returns.std()
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001

        sharpe = (avg_daily_return / std_daily_return * np.sqrt(252)) if std_daily_return > 0 else 0
        sortino = (avg_daily_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Profit factor
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    for p in pnls:
        if p > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)

    # Expectancy
    avg_win = np.mean(winners) if winners else 0
    avg_loss = abs(np.mean(losers)) if losers else 0.001
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

    # Calmar ratio
    total_return_pct = (total_pnl / capital) * 100
    calmar = total_return_pct / abs(max_dd) if max_dd != 0 else 0

    return {
        "total_trades": len(trades),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "avg_pnl": round(np.mean(pnls), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(-abs(np.mean(losers)) if losers else 0, 2),
        "max_win": round(max(pnls), 2) if pnls else 0,
        "max_loss": round(min(pnls), 2) if pnls else 0,
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "avg_hold_days": round(np.mean(hold_days), 1) if hold_days else 0,
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        "win_loss_ratio": round(avg_win / avg_loss, 2) if avg_loss > 0 else 0,
    }


# ============================================================
# OPTIMIZATION
# ============================================================

def run_optimization(
    df: pd.DataFrame,
    strategy_name: str,
    param_ranges: Dict[str, list],
    capital: float = 100000.0,
    position_size_pct: float = 100.0,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    trailing_stop_pct: Optional[float] = None,
    commission: float = 0.0,
    optimize_by: str = "total_pnl",
    sizing_mode: str = "compound",
    reinvest_pct: float = 50.0,
) -> pd.DataFrame:
    """
    Parameter sweep optimization.
    """
    from itertools import product

    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    all_combos = list(product(*param_values))

    results = []
    for combo in all_combos:
        params = dict(zip(param_names, combo))
        try:
            _, trades, equity_df = run_backtest(
                df, strategy_name, params, capital,
                position_size_pct, stop_loss_pct, take_profit_pct,
                trailing_stop_pct, commission, sizing_mode, reinvest_pct
            )
            metrics = compute_metrics(trades, equity_df, capital)
            row = {**params, **metrics}
            results.append(row)
        except Exception:
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty and optimize_by in results_df.columns:
        results_df = results_df.sort_values(optimize_by, ascending=False).reset_index(drop=True)
    return results_df


# ============================================================
# TRADE LOG EXPORT
# ============================================================

def trades_to_dataframe(trades: List[Trade], include_ticker: bool = False) -> pd.DataFrame:
    """Convert list of Trade objects to a clean DataFrame."""
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        row = {}
        if include_ticker:
            row["Ticker"] = t.option_type  # ticker stored here
        row.update({
            "Entry Date": t.entry_date,
            "Exit Date": t.exit_date,
            "Direction": t.direction,
            "Entry Price": round(t.entry_price, 2),
            "Exit Price": round(t.exit_price, 2),
            "Shares": round(t.shares, 2),
            "P&L ($)": round(t.pnl, 2),
            "P&L (%)": round(t.pnl_pct, 2),
            "MAE (%)": round(t.mae_pct, 2),
            "MFE (%)": round(t.mfe_pct, 2),
            "Exit Reason": t.exit_reason,
        })
        records.append(row)
    return pd.DataFrame(records)
