"""
Backtester Module
Executes strategies, manages positions with risk controls, computes performance metrics.
Supports: slippage, variable commissions, position sizing modes, Monte Carlo,
walk-forward out-of-sample validation, and multi-ticker portfolio backtesting.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.strategies import Trade, SignalType, ALL_STRATEGIES


# ============================================================
# COMMISSION MODELS
# ============================================================

def calc_commission(
    model: str,
    shares: float,
    price: float,
    flat_rate: float = 0.0,
) -> float:
    """
    Calculate round-trip commission based on model.
    Models: 'per_share', 'per_trade', 'percentage'
    flat_rate meaning varies:
      per_share:  $ per share (applied each side)
      per_trade:  $ per trade (applied each side)
      percentage: % of trade value (applied each side)
    """
    if model == "per_trade":
        return flat_rate * 2  # entry + exit
    elif model == "percentage":
        trade_value = shares * price
        return trade_value * (flat_rate / 100) * 2
    else:  # per_share (default)
        return flat_rate * shares * 2


# ============================================================
# POSITION SIZING
# ============================================================

def calc_position_size(
    mode: str, equity: float, capital: float, price: float,
    position_size_pct: float, reinvest_pct: float = 50.0,
) -> float:
    if mode == "fixed":
        base = capital * (position_size_pct / 100)
    elif mode == "fractional":
        profits = max(equity - capital, 0)
        reinvested = profits * (reinvest_pct / 100)
        base = (capital + reinvested) * (position_size_pct / 100)
        if equity < capital:
            base = equity * (position_size_pct / 100)
    else:
        base = equity * (position_size_pct / 100)
    return base / price if price > 0 else 0


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
    slippage_pct: float = 0.0,
    commission_model: str = "per_share",
) -> Tuple[pd.DataFrame, List[Trade], pd.DataFrame]:
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

            if stop_loss_pct is not None:
                sl_price = current_trade.entry_price * (1 - stop_loss_pct / 100)
                if current_price <= sl_price:
                    exit_reason = "Stop Loss"

            if not exit_reason and take_profit_pct is not None:
                tp_price = current_trade.entry_price * (1 + take_profit_pct / 100)
                if current_price >= tp_price:
                    exit_reason = "Take Profit"

            if not exit_reason and trailing_stop_pct is not None:
                ts_price = highest_since_entry * (1 - trailing_stop_pct / 100)
                if current_price <= ts_price:
                    exit_reason = "Trailing Stop"

            if not exit_reason and row["signal"] == SignalType.LONG_EXIT.value:
                exit_reason = "Signal"

            if not exit_reason and i == len(signal_df) - 1:
                exit_reason = "End of Data"

            if exit_reason:
                # Apply slippage to exit (worse price for seller)
                exit_price = current_price * (1 - slippage_pct / 100)
                current_trade.exit_date = date
                current_trade.exit_price = exit_price
                pnl_gross = (exit_price - current_trade.entry_price) * current_trade.shares
                total_comm = calc_commission(commission_model, current_trade.shares, current_trade.entry_price, commission)
                current_trade.pnl = pnl_gross - total_comm
                current_trade.pnl_pct = ((exit_price / current_trade.entry_price) - 1) * 100
                current_trade.exit_reason = exit_reason
                current_trade.mae_pct = ((lowest_since_entry / current_trade.entry_price) - 1) * 100
                current_trade.mfe_pct = ((highest_since_entry / current_trade.entry_price) - 1) * 100
                equity += current_trade.pnl
                current_trade.equity_at_exit = equity
                trades.append(current_trade)
                in_position = False
                current_trade = None

        elif row["signal"] == SignalType.LONG_ENTRY.value and not in_position:
            # Apply slippage to entry (worse price for buyer)
            entry_price = row["close"] * (1 + slippage_pct / 100)
            shares = calc_position_size(
                sizing_mode, equity, capital, entry_price,
                position_size_pct, reinvest_pct
            )
            current_trade = Trade(
                entry_date=date, direction="long",
                entry_price=entry_price, shares=shares,
                strategy=strategy_name,
            )
            current_trade.option_type = ticker
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
# MONTE CARLO SIMULATION
# ============================================================

def run_monte_carlo(
    trades: List[Trade],
    capital: float,
    n_simulations: int = 1000,
    max_trades_per_sim: int = 500,
) -> Dict[str, Any]:
    """
    Shuffle trade P&L sequence to simulate alternate outcomes.
    Returns distribution of final equity, max drawdowns, and risk of ruin.
    """
    if not trades:
        return {"final_equities": [], "max_drawdowns": [], "risk_of_ruin": 0,
                "median_equity": capital, "p5_equity": capital, "p95_equity": capital,
                "equity_paths": np.array([])}

    pnls = np.array([t.pnl for t in trades])
    n_trades = min(len(pnls), max_trades_per_sim)

    # Cap simulations based on dataset to avoid memory issues
    n_simulations = min(n_simulations, 2000)

    final_equities = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    ruin_count = 0
    ruin_threshold = capital * 0.1  # 90% loss = ruin

    # Store a subset of paths for fan chart (max 200)
    n_paths_to_store = min(200, n_simulations)
    equity_paths = np.zeros((n_paths_to_store, n_trades + 1))

    for sim in range(n_simulations):
        shuffled = np.random.permutation(pnls)[:n_trades]
        equity_curve = np.zeros(n_trades + 1)
        equity_curve[0] = capital

        for j in range(n_trades):
            equity_curve[j + 1] = equity_curve[j] + shuffled[j]

        final_equities[sim] = equity_curve[-1]

        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / np.where(peak > 0, peak, 1) * 100
        max_drawdowns[sim] = dd.min()

        if equity_curve.min() <= ruin_threshold:
            ruin_count += 1

        if sim < n_paths_to_store:
            equity_paths[sim] = equity_curve

    return {
        "final_equities": final_equities,
        "max_drawdowns": max_drawdowns,
        "risk_of_ruin": (ruin_count / n_simulations) * 100,
        "median_equity": float(np.median(final_equities)),
        "p5_equity": float(np.percentile(final_equities, 5)),
        "p95_equity": float(np.percentile(final_equities, 95)),
        "p25_equity": float(np.percentile(final_equities, 25)),
        "p75_equity": float(np.percentile(final_equities, 75)),
        "mean_max_dd": float(np.mean(max_drawdowns)),
        "equity_paths": equity_paths,
        "n_simulations": n_simulations,
        "n_trades": n_trades,
    }


# ============================================================
# WALK-FORWARD (OUT-OF-SAMPLE) SPLIT
# ============================================================

def split_walk_forward(
    df: pd.DataFrame,
    in_sample_pct: float = 70.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Split data into in-sample and out-of-sample portions."""
    split_idx = int(len(df) * (in_sample_pct / 100))
    split_date = df.index[split_idx]
    return df.iloc[:split_idx], df.iloc[split_idx:], split_date


# ============================================================
# PORTFOLIO BACKTEST
# ============================================================

def run_portfolio_backtest(
    portfolio_configs: List[Dict[str, Any]],
    capital: float = 100000.0,
    sizing_mode: str = "compound",
    reinvest_pct: float = 50.0,
) -> Dict[str, Any]:
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
            position_size_pct=100.0,
            stop_loss_pct=config.get("stop_loss_pct"),
            take_profit_pct=config.get("take_profit_pct"),
            trailing_stop_pct=config.get("trailing_stop_pct"),
            commission=config.get("commission", 0.0),
            sizing_mode=sizing_mode,
            reinvest_pct=reinvest_pct,
            ticker=ticker,
            slippage_pct=config.get("slippage_pct", 0.0),
            commission_model=config.get("commission_model", "per_share"),
        )

        metrics = compute_metrics(trades, equity_df, ticker_capital)
        results[ticker] = {
            "signal_df": signal_df, "trades": trades, "equity_df": equity_df,
            "metrics": metrics, "allocation_pct": alloc, "capital": ticker_capital,
        }
        for t in trades:
            t.option_type = ticker
        all_trades.extend(trades)
        equity_curves[ticker] = equity_df["equity"]

    combined_equity = _combine_equity_curves(equity_curves, capital)
    combined_metrics = compute_metrics(all_trades, combined_equity, capital)

    # Portfolio correlation
    correlation = _compute_portfolio_correlation(equity_curves)

    return {
        "per_ticker": results, "combined_trades": all_trades,
        "combined_equity": combined_equity, "combined_metrics": combined_metrics,
        "correlation": correlation,
    }


def _combine_equity_curves(curves: Dict[str, pd.Series], capital: float) -> pd.DataFrame:
    if not curves:
        return pd.DataFrame(columns=["equity"])
    combined = pd.DataFrame(curves)
    combined = combined.ffill().bfill()
    combined["equity"] = combined.sum(axis=1)
    return combined[["equity"]]


def _compute_portfolio_correlation(curves: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute correlation matrix of daily returns across tickers."""
    if len(curves) < 2:
        return pd.DataFrame()
    returns = pd.DataFrame()
    for ticker, eq in curves.items():
        returns[ticker] = eq.pct_change().dropna()
    return returns.corr()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(trades: List[Trade], equity_df: pd.DataFrame, capital: float) -> Dict[str, Any]:
    if not trades:
        return {k: 0 for k in [
            "total_trades", "win_rate", "total_pnl", "total_return_pct",
            "avg_pnl", "avg_win", "avg_loss", "max_win", "max_loss",
            "profit_factor", "expectancy", "max_drawdown_pct", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "avg_hold_days",
            "max_consecutive_wins", "max_consecutive_losses", "win_loss_ratio",
        ]}

    pnls = [t.pnl for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    win_rate = len(winners) / len(trades) * 100

    hold_days = []
    for t in trades:
        if t.entry_date is not None and t.exit_date is not None:
            try:
                delta = (pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days
                hold_days.append(max(delta, 1))
            except:
                hold_days.append(0)

    if not equity_df.empty:
        eq = equity_df["equity"] if "equity" in equity_df.columns else equity_df.iloc[:, 0]
        peak = eq.expanding().max()
        drawdown = (eq - peak) / peak * 100
        max_dd = drawdown.min()
    else:
        max_dd = 0.0

    if not equity_df.empty and len(equity_df) > 1:
        eq = equity_df["equity"] if "equity" in equity_df.columns else equity_df.iloc[:, 0]
        daily_returns = eq.pct_change().dropna()
        avg_dr = daily_returns.mean()
        std_dr = daily_returns.std()
        down_ret = daily_returns[daily_returns < 0]
        down_std = down_ret.std() if len(down_ret) > 0 else 0.001
        sharpe = (avg_dr / std_dr * np.sqrt(252)) if std_dr > 0 else 0
        sortino = (avg_dr / down_std * np.sqrt(252)) if down_std > 0 else 0
    else:
        sharpe = sortino = 0.0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    max_cw = max_cl = cw = cl = 0
    for p in pnls:
        if p > 0:
            cw += 1; cl = 0; max_cw = max(max_cw, cw)
        else:
            cl += 1; cw = 0; max_cl = max(max_cl, cl)

    avg_win = np.mean(winners) if winners else 0
    avg_loss_val = abs(np.mean(losers)) if losers else 0.001
    expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss_val)
    total_return_pct = (total_pnl / capital) * 100
    calmar = total_return_pct / abs(max_dd) if max_dd != 0 else 0

    return {
        "total_trades": len(trades), "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2), "total_return_pct": round(total_return_pct, 2),
        "avg_pnl": round(np.mean(pnls), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(-abs(np.mean(losers)) if losers else 0, 2),
        "max_win": round(max(pnls), 2), "max_loss": round(min(pnls), 2),
        "profit_factor": round(profit_factor, 2), "expectancy": round(expectancy, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2), "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "avg_hold_days": round(np.mean(hold_days), 1) if hold_days else 0,
        "max_consecutive_wins": max_cw, "max_consecutive_losses": max_cl,
        "win_loss_ratio": round(avg_win / avg_loss_val, 2) if avg_loss_val > 0 else 0,
    }


# ============================================================
# OPTIMIZATION
# ============================================================

def run_optimization(
    df: pd.DataFrame, strategy_name: str, param_ranges: Dict[str, list],
    capital: float = 100000.0, position_size_pct: float = 100.0,
    stop_loss_pct=None, take_profit_pct=None, trailing_stop_pct=None,
    commission: float = 0.0, optimize_by: str = "total_pnl",
    sizing_mode: str = "compound", reinvest_pct: float = 50.0,
    slippage_pct: float = 0.0, commission_model: str = "per_share",
) -> pd.DataFrame:
    from itertools import product
    param_names = list(param_ranges.keys())
    all_combos = list(product(*param_ranges.values()))

    results = []
    for combo in all_combos:
        params = dict(zip(param_names, combo))
        try:
            _, trades, equity_df = run_backtest(
                df, strategy_name, params, capital, position_size_pct,
                stop_loss_pct, take_profit_pct, trailing_stop_pct,
                commission, sizing_mode, reinvest_pct, "", slippage_pct, commission_model
            )
            metrics = compute_metrics(trades, equity_df, capital)
            results.append({**params, **metrics})
        except:
            continue

    results_df = pd.DataFrame(results)
    if not results_df.empty and optimize_by in results_df.columns:
        results_df = results_df.sort_values(optimize_by, ascending=False).reset_index(drop=True)
    return results_df


# ============================================================
# TRADE LOG EXPORT
# ============================================================

def trades_to_dataframe(trades: List[Trade], include_ticker: bool = False) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    records = []
    for t in trades:
        row = {}
        if include_ticker:
            row["Ticker"] = t.option_type
        row.update({
            "Entry Date": t.entry_date, "Exit Date": t.exit_date,
            "Direction": t.direction,
            "Entry Price": round(t.entry_price, 2), "Exit Price": round(t.exit_price, 2),
            "Shares": round(t.shares, 2),
            "P&L ($)": round(t.pnl, 2), "P&L (%)": round(t.pnl_pct, 2),
            "MAE (%)": round(t.mae_pct, 2), "MFE (%)": round(t.mfe_pct, 2),
            "Exit Reason": t.exit_reason,
        })
        records.append(row)
    return pd.DataFrame(records)
