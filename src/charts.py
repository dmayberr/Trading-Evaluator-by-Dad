"""
Charts Module
All Plotly-based visualizations for the trading evaluator.
Includes: equity curve with trade markers, trade shading on price,
drawdown duration, rolling metrics, benchmark overlay, MAE/MFE analysis,
plus all original charts.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from src.strategies import Trade


# Color palette
COLORS = {
    "bg": "#0a0e17",
    "card": "#111827",
    "grid": "#1e293b",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "green": "#10b981",
    "red": "#ef4444",
    "blue": "#3b82f6",
    "amber": "#f59e0b",
    "purple": "#8b5cf6",
    "cyan": "#06b6d4",
    "white": "#f8fafc",
}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="JetBrains Mono, monospace", color=COLORS["text"], size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)


# ============================================================
# ENHANCEMENT 1: Equity Curve with Entry/Exit Markers
# ============================================================

def chart_equity_curve(
    equity_df: pd.DataFrame,
    capital: float,
    trades: Optional[List[Trade]] = None,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """Equity curve with drawdown, trade entry/exit markers, and optional benchmark."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=["Equity Curve", "Drawdown"]
    )

    # Equity line
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["equity"],
        name="Portfolio", line=dict(color=COLORS["cyan"], width=2),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.08)"
    ), row=1, col=1)

    # --- ENHANCEMENT 5: Benchmark overlay ---
    if benchmark_df is not None and not benchmark_df.empty:
        # Normalize benchmark to start at same capital
        bench_start = benchmark_df["close"].iloc[0]
        bench_equity = (benchmark_df["close"] / bench_start) * capital
        # Align to equity_df index
        bench_aligned = bench_equity.reindex(equity_df.index, method="ffill")
        fig.add_trace(go.Scatter(
            x=bench_aligned.index, y=bench_aligned.values,
            name="Benchmark (Buy & Hold)", mode="lines",
            line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
            opacity=0.7,
        ), row=1, col=1)

    # Baseline
    fig.add_hline(y=capital, line_dash="dash", line_color=COLORS["muted"],
                  annotation_text="Starting Capital", row=1, col=1)

    # --- ENHANCEMENT 1: Trade markers on equity curve ---
    if trades:
        entry_dates = []
        entry_equities = []
        entry_texts = []
        exit_dates = []
        exit_equities = []
        exit_texts = []

        for t in trades:
            if t.entry_date is not None and t.equity_at_entry > 0:
                entry_dates.append(t.entry_date)
                entry_equities.append(t.equity_at_entry)
                tk_label = t.option_type if t.option_type else ""
                entry_texts.append(f"BUY {tk_label} @ ${t.entry_price:,.2f}")
            if t.exit_date is not None and t.equity_at_exit > 0:
                exit_dates.append(t.exit_date)
                exit_equities.append(t.equity_at_exit)
                tk_label = t.option_type if t.option_type else ""
                color_label = "🟢" if t.pnl > 0 else "🔴"
                exit_texts.append(
                    f"{color_label} {tk_label} P&L: ${t.pnl:,.0f} ({t.pnl_pct:+.1f}%)\n{t.exit_reason}"
                )

        if entry_dates:
            fig.add_trace(go.Scatter(
                x=entry_dates, y=entry_equities,
                mode="markers", name="Entry",
                marker=dict(symbol="triangle-up", size=10, color=COLORS["green"],
                            line=dict(width=1, color=COLORS["white"])),
                text=entry_texts,
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=1)

        if exit_dates:
            fig.add_trace(go.Scatter(
                x=exit_dates, y=exit_equities,
                mode="markers", name="Exit",
                marker=dict(symbol="triangle-down", size=10, color=COLORS["red"],
                            line=dict(width=1, color=COLORS["white"])),
                text=exit_texts,
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=1)

    # Drawdown
    peak = equity_df["equity"].expanding().max()
    dd = (equity_df["equity"] - peak) / peak * 100
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=dd,
        name="Drawdown %", line=dict(color=COLORS["red"], width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)"
    ), row=2, col=1)

    # --- ENHANCEMENT 3: Highlight longest drawdown period ---
    in_dd = dd < 0
    dd_start = None
    longest_start = None
    longest_end = None
    longest_dur = 0
    for idx_i in range(len(dd)):
        if in_dd.iloc[idx_i]:
            if dd_start is None:
                dd_start = dd.index[idx_i]
        else:
            if dd_start is not None:
                dur = (dd.index[idx_i] - dd_start).days
                if dur > longest_dur:
                    longest_dur = dur
                    longest_start = dd_start
                    longest_end = dd.index[idx_i]
                dd_start = None
    # Check if still in drawdown at end
    if dd_start is not None:
        dur = (dd.index[-1] - dd_start).days
        if dur > longest_dur:
            longest_dur = dur
            longest_start = dd_start
            longest_end = dd.index[-1]

    if longest_start is not None and longest_end is not None and longest_dur > 5:
        fig.add_vrect(
            x0=longest_start, x1=longest_end,
            fillcolor="rgba(239,68,68,0.08)", line_width=0,
            annotation_text=f"Longest DD: {longest_dur}d",
            annotation_position="top left",
            annotation_font=dict(color=COLORS["red"], size=10),
            row=2, col=1,
        )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=550,
        showlegend=True,
        title_text=None,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], row=1, col=1)
    fig.update_xaxes(gridcolor=COLORS["grid"], row=2, col=1)
    fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=1, title_text="Equity ($)")
    fig.update_yaxes(gridcolor=COLORS["grid"], row=2, col=1, title_text="DD %")
    fig.update_annotations(font=dict(color=COLORS["text"], size=13))
    return fig


# ============================================================
# ENHANCEMENT 2: Price Chart with Trade Shading
# ============================================================

def chart_price_with_signals(
    signal_df: pd.DataFrame,
    trades: List[Trade],
    strategy_name: str
) -> go.Figure:
    """Price chart with buy/sell markers, indicator overlays, and trade region shading."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
    )

    # Trade region shading (BEFORE candlestick so it's behind)
    for t in trades:
        if t.entry_date is not None and t.exit_date is not None:
            fill_color = "rgba(16,185,129,0.10)" if t.pnl > 0 else "rgba(239,68,68,0.10)"
            border_color = "rgba(16,185,129,0.35)" if t.pnl > 0 else "rgba(239,68,68,0.35)"
            fig.add_vrect(
                x0=t.entry_date, x1=t.exit_date,
                fillcolor=fill_color,
                line=dict(width=1, color=border_color),
                row=1, col=1,
            )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=signal_df.index,
        open=signal_df["open"], high=signal_df["high"],
        low=signal_df["low"], close=signal_df["close"],
        name="Price",
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
    ), row=1, col=1)

    # Overlay indicators if present
    overlay_cols = [c for c in signal_df.columns if c.startswith(("sma_", "ema_", "bb_"))]
    indicator_colors = [COLORS["amber"], COLORS["purple"], COLORS["cyan"], COLORS["blue"], COLORS["green"]]
    for idx, col in enumerate(overlay_cols[:5]):
        fig.add_trace(go.Scatter(
            x=signal_df.index, y=signal_df[col],
            name=col.upper(), mode="lines",
            line=dict(color=indicator_colors[idx % len(indicator_colors)], width=1.2),
            opacity=0.8,
        ), row=1, col=1)

    # SuperTrend overlay with directional coloring
    if "supertrend" in signal_df.columns and "st_direction" in signal_df.columns:
        st_vals = signal_df["supertrend"]
        st_dir = signal_df["st_direction"]
        # Split into bullish and bearish segments for coloring
        bull_st = st_vals.where(st_dir == 1)
        bear_st = st_vals.where(st_dir == -1)
        fig.add_trace(go.Scatter(
            x=signal_df.index, y=bull_st,
            name="SuperTrend (Bull)", mode="lines",
            line=dict(color=COLORS["green"], width=2),
            connectgaps=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=signal_df.index, y=bear_st,
            name="SuperTrend (Bear)", mode="lines",
            line=dict(color=COLORS["red"], width=2),
            connectgaps=False,
        ), row=1, col=1)

    # Buy/Sell markers with richer hover info
    if trades:
        entry_dates = [t.entry_date for t in trades]
        entry_prices = [t.entry_price for t in trades]
        entry_texts = [f"BUY @ ${t.entry_price:,.2f}" for t in trades]

        exit_dates = [t.exit_date for t in trades]
        exit_prices = [t.exit_price for t in trades]
        exit_texts = [
            f"SELL @ ${t.exit_price:,.2f}<br>P&L: ${t.pnl:,.0f} ({t.pnl_pct:+.1f}%)<br>{t.exit_reason}"
            for t in trades
        ]

        fig.add_trace(go.Scatter(
            x=entry_dates, y=entry_prices,
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=12, color=COLORS["green"],
                        line=dict(width=1, color=COLORS["white"])),
            text=entry_texts,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=exit_dates, y=exit_prices,
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=12, color=COLORS["red"],
                        line=dict(width=1, color=COLORS["white"])),
            text=exit_texts,
            hovertemplate="%{text}<extra></extra>",
        ), row=1, col=1)

    # Volume
    vol_colors = [COLORS["green"] if c >= o else COLORS["red"]
                  for c, o in zip(signal_df["close"], signal_df["open"])]
    fig.add_trace(go.Bar(
        x=signal_df.index, y=signal_df["volume"],
        name="Volume", marker=dict(color=vol_colors, opacity=0.4),
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        height=550,
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=1, title_text="Price")
    fig.update_yaxes(gridcolor=COLORS["grid"], row=2, col=1, title_text="Volume")
    return fig


# ============================================================
# ENHANCEMENT 3: Drawdown Duration Chart
# ============================================================

def chart_drawdown_duration(equity_df: pd.DataFrame) -> go.Figure:
    """Underwater chart with drawdown periods colored by duration."""
    if equity_df.empty:
        return go.Figure()

    eq = equity_df["equity"]
    peak = eq.expanding().max()
    dd_pct = (eq - peak) / peak * 100

    # Find drawdown periods
    periods = []
    start = None
    for i in range(len(dd_pct)):
        if dd_pct.iloc[i] < 0:
            if start is None:
                start = i
        else:
            if start is not None:
                dur = (dd_pct.index[i] - dd_pct.index[start]).days
                depth = dd_pct.iloc[start:i].min()
                periods.append({"start": dd_pct.index[start], "end": dd_pct.index[i],
                                "duration": dur, "depth": depth})
                start = None
    if start is not None:
        dur = (dd_pct.index[-1] - dd_pct.index[start]).days
        depth = dd_pct.iloc[start:].min()
        periods.append({"start": dd_pct.index[start], "end": dd_pct.index[-1],
                        "duration": dur, "depth": depth})

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.08, row_heights=[0.6, 0.4],
        subplot_titles=["Underwater Curve", "Drawdown Duration (days)"]
    )

    # Underwater curve
    fig.add_trace(go.Scatter(
        x=dd_pct.index, y=dd_pct.values,
        name="Drawdown %", line=dict(color=COLORS["red"], width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
        hovertemplate="Date: %{x}<br>DD: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)

    # Duration bars
    if periods:
        starts = [p["start"] for p in periods]
        durations = [p["duration"] for p in periods]
        depths = [p["depth"] for p in periods]
        colors = []
        for d in durations:
            if d < 14:
                colors.append(COLORS["amber"])
            elif d < 60:
                colors.append("#f97316")  # orange
            else:
                colors.append(COLORS["red"])

        fig.add_trace(go.Bar(
            x=starts, y=durations,
            name="Duration (days)",
            marker=dict(color=colors, opacity=0.8,
                        line=dict(width=0.5, color=COLORS["bg"])),
            text=[f"{d}d | {dp:.1f}%" for d, dp in zip(durations, depths)],
            hovertemplate="Start: %{x}<br>Duration: %{text}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=450, showlegend=True,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=1, title_text="DD %")
    fig.update_yaxes(gridcolor=COLORS["grid"], row=2, col=1, title_text="Days")
    fig.update_annotations(font=dict(color=COLORS["text"], size=13))
    return fig


# ============================================================
# ENHANCEMENT 4: Rolling Metrics (Sharpe & Win Rate)
# ============================================================

def chart_rolling_metrics(
    equity_df: pd.DataFrame,
    trades: List[Trade],
    rolling_window: int = 20,
) -> go.Figure:
    """Rolling Sharpe ratio and rolling win rate over trade sequence."""
    if not trades or len(trades) < rolling_window:
        fig = go.Figure()
        fig.update_layout(**LAYOUT_DEFAULTS, height=400,
                          title_text=f"Need at least {rolling_window} trades for rolling metrics")
        return fig

    # Rolling win rate over trade sequence
    pnls = pd.Series([t.pnl for t in trades])
    wins = (pnls > 0).astype(int)
    rolling_wr = wins.rolling(window=rolling_window).mean() * 100
    trade_dates = [t.exit_date for t in trades]

    # Rolling Sharpe from equity curve (daily)
    if not equity_df.empty and len(equity_df) > 30:
        daily_ret = equity_df["equity"].pct_change().dropna()
        window_days = min(rolling_window * 5, len(daily_ret) - 1)  # approximate
        if window_days > 10:
            roll_mean = daily_ret.rolling(window=window_days).mean()
            roll_std = daily_ret.rolling(window=window_days).std()
            rolling_sharpe = (roll_mean / roll_std * np.sqrt(252)).dropna()
        else:
            rolling_sharpe = pd.Series(dtype=float)
    else:
        rolling_sharpe = pd.Series(dtype=float)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=[
            f"Rolling Win Rate ({rolling_window}-trade window)",
            f"Rolling Sharpe Ratio"
        ]
    )

    # Rolling win rate
    fig.add_trace(go.Scatter(
        x=trade_dates, y=rolling_wr.values,
        name="Win Rate %", mode="lines",
        line=dict(color=COLORS["cyan"], width=2),
    ), row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color=COLORS["muted"],
                  annotation_text="50%", row=1, col=1)

    # Rolling Sharpe
    if not rolling_sharpe.empty:
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            name="Sharpe", mode="lines",
            line=dict(color=COLORS["amber"], width=2),
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], row=2, col=1)
        fig.add_hline(y=1, line_dash="dot", line_color=COLORS["green"],
                      annotation_text="Sharpe=1", row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=500, showlegend=True,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], row=1, col=1, title_text="Win Rate %")
    fig.update_yaxes(gridcolor=COLORS["grid"], row=2, col=1, title_text="Sharpe Ratio")
    fig.update_annotations(font=dict(color=COLORS["text"], size=13))
    return fig


# ============================================================
# ENHANCEMENT 6: MAE / MFE Scatter Analysis
# ============================================================

def chart_mae_mfe(trades: List[Trade]) -> go.Figure:
    """MAE vs MFE scatter with P&L coloring. Reveals stop/target tuning."""
    if not trades:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=False,
        subplot_titles=["MAE vs P&L (Stop Analysis)", "MFE vs P&L (Target Analysis)"],
        horizontal_spacing=0.1,
    )

    mae_vals = [t.mae_pct for t in trades]
    mfe_vals = [t.mfe_pct for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    colors = [COLORS["green"] if p > 0 else COLORS["red"] for p in pnl_pcts]
    sizes = [max(6, min(18, abs(p) / max(1, max(abs(pp) for pp in pnl_pcts)) * 18)) for p in pnl_pcts]
    hover_texts = [
        f"P&L: {p:+.1f}%<br>MAE: {m:.1f}%<br>MFE: {mf:.1f}%<br>{t.exit_reason}"
        for t, p, m, mf in zip(trades, pnl_pcts, mae_vals, mfe_vals)
    ]

    # MAE vs P&L
    fig.add_trace(go.Scatter(
        x=mae_vals, y=pnl_pcts,
        mode="markers", name="Trades",
        marker=dict(color=colors, size=sizes, opacity=0.75,
                    line=dict(width=1, color=COLORS["bg"])),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # Diagonal reference line (if MAE = P&L, stop was hit exactly)
    mae_range = [min(mae_vals) if mae_vals else -10, 0]
    fig.add_trace(go.Scatter(
        x=mae_range, y=mae_range,
        mode="lines", name="MAE = P&L",
        line=dict(color=COLORS["muted"], dash="dot", width=1),
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], row=1, col=1)

    # MFE vs P&L
    fig.add_trace(go.Scatter(
        x=mfe_vals, y=pnl_pcts,
        mode="markers", name="Trades",
        marker=dict(color=colors, size=sizes, opacity=0.75,
                    line=dict(width=1, color=COLORS["bg"])),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    # Diagonal reference (if MFE = P&L, you captured the full move)
    mfe_range = [0, max(mfe_vals) if mfe_vals else 10]
    fig.add_trace(go.Scatter(
        x=mfe_range, y=mfe_range,
        mode="lines", name="MFE = P&L (full capture)",
        line=dict(color=COLORS["green"], dash="dot", width=1),
        showlegend=False,
    ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], row=1, col=2)

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=420, showlegend=False,
    )
    fig.update_xaxes(title_text="MAE (%)", gridcolor=COLORS["grid"], row=1, col=1)
    fig.update_xaxes(title_text="MFE (%)", gridcolor=COLORS["grid"], row=1, col=2)
    fig.update_yaxes(title_text="P&L (%)", gridcolor=COLORS["grid"], row=1, col=1)
    fig.update_yaxes(title_text="P&L (%)", gridcolor=COLORS["grid"], row=1, col=2)
    fig.update_annotations(font=dict(color=COLORS["text"], size=13))
    return fig


def chart_trade_duration_vs_pnl(trades: List[Trade]) -> go.Figure:
    """Trade duration vs P&L scatter — reveals time-in-trade patterns."""
    if not trades:
        return go.Figure()

    durations = []
    pnl_pcts = []
    colors = []
    hover_texts = []
    for t in trades:
        if t.entry_date and t.exit_date:
            try:
                dur = max((pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days, 1)
            except:
                continue
            durations.append(dur)
            pnl_pcts.append(t.pnl_pct)
            colors.append(COLORS["green"] if t.pnl > 0 else COLORS["red"])
            tk = t.option_type if t.option_type else ""
            hover_texts.append(f"{tk} {dur}d | P&L: {t.pnl_pct:+.1f}% | {t.exit_reason}")

    if not durations:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=durations, y=pnl_pcts,
        mode="markers",
        marker=dict(color=colors, size=9, opacity=0.75,
                    line=dict(width=1, color=COLORS["bg"])),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        name="Trades",
    ))

    # Trend line
    if len(durations) > 3:
        z = np.polyfit(durations, pnl_pcts, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(durations), max(durations), 50)
        fig.add_trace(go.Scatter(
            x=x_range, y=p(x_range),
            mode="lines", name="Trend",
            line=dict(color=COLORS["amber"], width=2, dash="dash"),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=380,
        title_text="Trade Duration vs P&L",
        xaxis_title="Hold Duration (days)",
        yaxis_title="P&L (%)",
    )
    return fig


# ============================================================
# ORIGINAL CHARTS (preserved)
# ============================================================

def chart_pnl_distribution(trades: List[Trade]) -> go.Figure:
    """P&L distribution histogram."""
    pnls = [t.pnl for t in trades]
    if not pnls:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnls, nbinsx=30,
        marker=dict(color=COLORS["blue"],
                    line=dict(color=COLORS["cyan"], width=0.5)),
        opacity=0.8, name="P&L"
    ))

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["amber"])
    avg_pnl = np.mean(pnls)
    fig.add_vline(x=avg_pnl, line_dash="dot", line_color=COLORS["cyan"],
                  annotation_text=f"Avg: ${avg_pnl:,.0f}")

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=350,
        title_text="P&L Distribution",
        xaxis_title="P&L ($)", yaxis_title="Frequency",
        showlegend=False,
    )
    return fig


def chart_trade_scatter(trades: List[Trade]) -> go.Figure:
    """Scatter plot of individual trade P&L over time."""
    if not trades:
        return go.Figure()

    dates = [t.exit_date for t in trades]
    pnls = [t.pnl for t in trades]
    colors = [COLORS["green"] if p > 0 else COLORS["red"] for p in pnls]
    max_abs = max(abs(max(pnls)), abs(min(pnls)), 1)
    sizes = [max(6, min(20, abs(p) / max_abs * 20)) for p in pnls]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=pnls, mode="markers",
        marker=dict(color=colors, size=sizes, opacity=0.7,
                    line=dict(width=1, color=COLORS["bg"])),
        text=[f"P&L: ${p:,.2f}" for p in pnls],
        hovertemplate="%{text}<br>Date: %{x}<extra></extra>",
        name="Trades"
    ))

    fig.add_hline(y=0, line_color=COLORS["muted"], line_dash="dash")

    cum_pnl = np.cumsum(pnls)
    fig.add_trace(go.Scatter(
        x=dates, y=cum_pnl, mode="lines", name="Cumulative P&L",
        line=dict(color=COLORS["amber"], width=2, dash="dot"),
        yaxis="y2"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=350,
        title_text="Trade Results Over Time",
        xaxis_title="Date", yaxis_title="P&L ($)",
        yaxis2=dict(title="Cumulative P&L ($)", overlaying="y", side="right",
                    gridcolor=COLORS["grid"], showgrid=False),
    )
    return fig


def chart_monthly_returns(equity_df: pd.DataFrame) -> go.Figure:
    """Monthly returns heatmap."""
    if equity_df.empty:
        return go.Figure()

    monthly = equity_df["equity"].resample("ME").last()
    monthly_returns = monthly.pct_change() * 100
    monthly_returns = monthly_returns.dropna()

    if monthly_returns.empty:
        return go.Figure()

    df_monthly = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return": monthly_returns.values
    })

    pivot = df_monthly.pivot_table(values="return", index="year", columns="month", aggfunc="first")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[int(m) - 1] for m in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale=[[0, COLORS["red"]], [0.5, COLORS["bg"]], [1, COLORS["green"]]],
        zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11, color=COLORS["white"]),
        hovertemplate="Month: %{x}<br>Year: %{y}<br>Return: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Return %", ticksuffix="%"),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=300,
        title_text="Monthly Returns Heatmap",
        xaxis_title="", yaxis_title="",
    )
    return fig


def chart_optimization_heatmap(opt_df: pd.DataFrame, param_x: str, param_y: str, metric: str) -> go.Figure:
    """Heatmap of optimization results across two parameters."""
    if opt_df.empty or param_x not in opt_df.columns or param_y not in opt_df.columns:
        return go.Figure()

    pivot = opt_df.pivot_table(values=metric, index=param_y, columns=param_x, aggfunc="first")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(x) for x in pivot.columns.tolist()],
        y=[str(y) for y in pivot.index.tolist()],
        colorscale=[[0, COLORS["red"]], [0.5, COLORS["bg"]], [1, COLORS["green"]]],
        zmid=0 if metric in ["total_pnl", "total_return_pct", "sharpe_ratio"] else None,
        text=[[f"{v:.1f}" if not np.isnan(v) else "" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLORS["white"]),
        colorbar=dict(title=metric.replace("_", " ").title()),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=400,
        title_text=f"Optimization: {metric.replace('_', ' ').title()}",
        xaxis_title=param_x.replace("_", " ").title(),
        yaxis_title=param_y.replace("_", " ").title(),
    )
    return fig


def chart_win_loss_bar(trades: List[Trade]) -> go.Figure:
    """Win/Loss breakdown bar chart."""
    if not trades:
        return go.Figure()

    winners = sum(1 for t in trades if t.pnl > 0)
    losers = sum(1 for t in trades if t.pnl <= 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Winners", "Losers"], y=[winners, losers],
        marker=dict(color=[COLORS["green"], COLORS["red"]],
                    line=dict(width=1, color=COLORS["bg"])),
        text=[winners, losers], textposition="outside",
        textfont=dict(color=COLORS["white"], size=14),
    ))

    fig.update_layout(**LAYOUT_DEFAULTS, height=250, showlegend=False, yaxis_title="Count")
    return fig


# ============================================================
# PORTFOLIO CHARTS
# ============================================================

def chart_portfolio_equity(
    per_ticker: dict,
    combined_equity: pd.DataFrame,
    capital: float,
) -> go.Figure:
    """Portfolio equity overlay: each ticker + combined total."""
    fig = go.Figure()

    ticker_colors = [
        COLORS["cyan"], COLORS["amber"], COLORS["purple"],
        COLORS["green"], COLORS["red"], COLORS["blue"],
        "#f472b6", "#a78bfa", "#34d399", "#fbbf24",
    ]

    for idx, (ticker, data) in enumerate(per_ticker.items()):
        eq = data["equity_df"]["equity"]
        color = ticker_colors[idx % len(ticker_colors)]
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=ticker, mode="lines",
            line=dict(color=color, width=1.5), opacity=0.7,
        ))

    if not combined_equity.empty:
        fig.add_trace(go.Scatter(
            x=combined_equity.index,
            y=combined_equity["equity"].values,
            name="PORTFOLIO", mode="lines",
            line=dict(color=COLORS["white"], width=3),
        ))

    fig.add_hline(y=capital, line_dash="dash", line_color=COLORS["muted"],
                  annotation_text="Starting Capital")

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=500,
        title_text="Portfolio Equity Curves",
        yaxis_title="Equity ($)", showlegend=True,
    )
    return fig


def chart_portfolio_allocation(per_ticker: dict) -> go.Figure:
    """Pie chart of portfolio capital allocation."""
    labels = list(per_ticker.keys())
    values = [per_ticker[t]["allocation_pct"] for t in labels]
    ticker_colors = [
        COLORS["cyan"], COLORS["amber"], COLORS["purple"],
        COLORS["green"], COLORS["red"], COLORS["blue"],
        "#f472b6", "#a78bfa", "#34d399", "#fbbf24",
    ]

    fig = go.Figure(data=go.Pie(
        labels=labels, values=values,
        marker=dict(colors=ticker_colors[:len(labels)],
                    line=dict(color=COLORS["bg"], width=2)),
        textfont=dict(color=COLORS["white"], size=13), hole=0.45,
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=350,
        title_text="Capital Allocation", showlegend=True,
    )
    return fig


def chart_portfolio_comparison(per_ticker: dict) -> go.Figure:
    """Grouped bar chart comparing key metrics across tickers."""
    tickers = list(per_ticker.keys())
    metrics_to_show = ["total_return_pct", "win_rate", "sharpe_ratio", "max_drawdown_pct"]
    metric_labels = ["Return %", "Win Rate %", "Sharpe", "Max DD %"]
    ticker_colors = [
        COLORS["cyan"], COLORS["amber"], COLORS["purple"],
        COLORS["green"], COLORS["red"], COLORS["blue"],
    ]

    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        m = per_ticker[ticker]["metrics"]
        fig.add_trace(go.Bar(
            name=ticker, x=metric_labels,
            y=[m.get(k, 0) for k in metrics_to_show],
            marker_color=ticker_colors[i % len(ticker_colors)],
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=400,
        title_text="Strategy Comparison by Ticker",
        barmode="group", yaxis_title="Value",
    )
    return fig


# ============================================================
# MONTE CARLO CHARTS
# ============================================================

def chart_monte_carlo_fan(mc_results: dict, capital: float) -> go.Figure:
    """Fan chart showing Monte Carlo simulation equity paths with confidence bands."""
    paths = mc_results.get("equity_paths", np.array([]))
    if paths.size == 0:
        fig = go.Figure()
        fig.update_layout(**LAYOUT_DEFAULTS, height=400, title_text="No simulation data")
        return fig

    n_paths, n_steps = paths.shape
    x = list(range(n_steps))

    fig = go.Figure()

    # Plot subset of individual paths (faded)
    max_display = min(100, n_paths)
    for i in range(max_display):
        fig.add_trace(go.Scatter(
            x=x, y=paths[i], mode="lines",
            line=dict(color=COLORS["cyan"], width=0.3),
            opacity=0.15, showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", name="95th %ile",
                              line=dict(color=COLORS["green"], width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", name="75th %ile",
                              line=dict(color=COLORS["green"], width=1),
                              fill="tonexty", fillcolor="rgba(16,185,129,0.08)"))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", name="Median",
                              line=dict(color=COLORS["white"], width=2.5)))
    fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", name="25th %ile",
                              line=dict(color=COLORS["red"], width=1),
                              fill="tonexty", fillcolor="rgba(239,68,68,0.08)"))
    fig.add_trace(go.Scatter(x=x, y=p5, mode="lines", name="5th %ile",
                              line=dict(color=COLORS["red"], width=1, dash="dot")))

    fig.add_hline(y=capital, line_dash="dash", line_color=COLORS["muted"])

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=450,
        title_text=f"Monte Carlo Simulation ({mc_results.get('n_simulations', 0)} runs)",
        xaxis_title="Trade Sequence", yaxis_title="Equity ($)",
    )
    return fig


def chart_monte_carlo_distribution(mc_results: dict, capital: float) -> go.Figure:
    """Histogram of final equity outcomes from Monte Carlo."""
    finals = mc_results.get("final_equities", [])
    if len(finals) == 0:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=finals, nbinsx=50,
        marker=dict(color=COLORS["blue"], line=dict(color=COLORS["cyan"], width=0.5)),
        opacity=0.8, name="Final Equity",
    ))

    fig.add_vline(x=capital, line_dash="dash", line_color=COLORS["muted"],
                  annotation_text="Starting Capital")
    fig.add_vline(x=mc_results.get("median_equity", capital), line_dash="dot",
                  line_color=COLORS["amber"], annotation_text="Median")

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=350,
        title_text="Distribution of Final Equity",
        xaxis_title="Final Equity ($)", yaxis_title="Frequency",
    )
    return fig


# ============================================================
# PORTFOLIO CORRELATION CHART
# ============================================================

def chart_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for portfolio positions."""
    if corr_df.empty:
        fig = go.Figure()
        fig.update_layout(**LAYOUT_DEFAULTS, height=300, title_text="Need 2+ positions for correlation")
        return fig

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale=[[0, COLORS["blue"]], [0.5, COLORS["bg"]], [1, COLORS["red"]]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_df.values],
        texttemplate="%{text}",
        textfont=dict(size=14, color=COLORS["white"]),
        colorbar=dict(title="Correlation"),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS, height=350,
        title_text="Portfolio Correlation (Daily Returns)",
    )
    return fig
