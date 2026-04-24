"""
Plotly chart builder for the trading dashboard.
Builds a multi-subplot candlestick chart with technical indicators.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_candlestick_chart(run_data: dict, active_positions: list | None = None) -> str:
    """
    Build an interactive candlestick chart with indicator overlays.
    Returns Plotly figure as JSON string for client-side rendering.
    """
    ohlcv = run_data.get("ohlcv_data", [])
    indicators = run_data.get("chart_indicators", {})

    if not ohlcv:
        fig = go.Figure()
        fig.add_annotation(text="No market data available yet.", showarrow=False, font=dict(size=18, color="#888"))
        fig.update_layout(template="plotly_dark", height=400)
        return fig.to_json()

    timestamps = [c["timestamp"] for c in ohlcv]
    opens = [c["open"] for c in ohlcv]
    highs = [c["high"] for c in ohlcv]
    lows = [c["low"] for c in ohlcv]
    closes = [c["close"] for c in ohlcv]
    volumes = indicators.get("volume", [])

    # Use indicator timestamps if available (they align with processed data)
    ind_ts = indicators.get("timestamps", timestamps)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.15, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("BTC/USD (H1)", "RSI (14)", "MACD", "Volume"),
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=timestamps, open=opens, high=highs, low=lows, close=closes,
        name="BTC/USD", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Bollinger Bands overlay
    bb_high = indicators.get("bb_high", [])
    bb_low = indicators.get("bb_low", [])
    if bb_high and bb_low and len(bb_high) == len(ind_ts):
        fig.add_trace(go.Scatter(
            x=ind_ts, y=bb_high, mode="lines", name="BB Upper",
            line=dict(color="rgba(173,216,230,0.5)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ind_ts, y=bb_low, mode="lines", name="BB Lower",
            line=dict(color="rgba(173,216,230,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(173,216,230,0.08)",
        ), row=1, col=1)

    # EMA 20 overlay
    ema20 = indicators.get("ema20", [])
    if ema20 and len(ema20) == len(ind_ts):
        fig.add_trace(go.Scatter(
            x=ind_ts, y=ema20, mode="lines", name="EMA 20",
            line=dict(color="#ff9800", width=1.5),
        ), row=1, col=1)

    # Prediction arrow on latest candle
    prediction = run_data.get("prediction")
    if prediction is not None and closes:
        arrow_color = "#26a69a" if prediction == 1 else "#ef5350"
        arrow_text = "BUY" if prediction == 1 else "SELL"
        fig.add_annotation(
            x=timestamps[-1], y=closes[-1],
            text=arrow_text,
            showarrow=True,
            arrowhead=2,
            arrowcolor=arrow_color,
            font=dict(color=arrow_color, size=12, family="monospace"),
            ay=-40 if prediction == 1 else 40,
            row=1, col=1,
        )

    # Active Position Levels (Entry, SL, TP)
    if active_positions:
        for pos in active_positions:
            # Check if this is the active asset (e.g. BTC)
            if "BTC" in pos["symbol"]:
                entry = pos["avg_entry_price"]
                sl = pos.get("sl_price") or run_data.get("stop_loss")
                tp = pos.get("tp_price") or run_data.get("take_profit")
                pnl = pos["unrealized_pl"]
                pnl_pct = pos["unrealized_plpc"]
                
                # Entry Line
                if entry:
                    fig.add_shape(
                        type="line", line=dict(color="#ffeb3b", width=1.5, dash="solid"),
                        x0=0, x1=1, xref="paper", y0=entry, y1=entry, yref="y1",
                        row=1, col=1
                    )
                    pnl_color = "#3fb950" if pnl >= 0 else "#f85149"
                    pnl_sign = "+" if pnl >= 0 else ""
                    fig.add_annotation(
                        x=0, xref="x domain", y=entry, yref="y1",
                        text=f"ENTRY: ${entry:.2f} | PnL: {pnl_sign}${pnl:.2f} ({pnl_sign}{pnl_pct:.2f}%)",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color=pnl_color, size=11, family="monospace"),
                        bgcolor="rgba(0,0,0,0.8)",
                        row=1, col=1
                    )
                
                # Stop Loss Line
                if sl:
                    fig.add_shape(
                        type="line", line=dict(color="#f85149", width=1.5, dash="dash"),
                        x0=0, x1=1, xref="paper", y0=sl, y1=sl, yref="y1",
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=0, xref="x domain", y=sl, yref="y1",
                        text=f"SL: ${sl:.2f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color="#f85149", size=10, family="monospace"),
                        bgcolor="rgba(0,0,0,0.8)",
                        row=1, col=1
                    )
                
                # Take Profit Line
                if tp:
                    fig.add_shape(
                        type="line", line=dict(color="#3fb950", width=1.5, dash="dash"),
                        x0=0, x1=1, xref="paper", y0=tp, y1=tp, yref="y1",
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=0, xref="x domain", y=tp, yref="y1",
                        text=f"TP: ${tp:.2f}",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(color="#3fb950", size=10, family="monospace"),
                        bgcolor="rgba(0,0,0,0.8)",
                        row=1, col=1
                    )

    # Row 2: RSI
    rsi = indicators.get("rsi", [])
    if rsi and len(rsi) == len(ind_ts):
        fig.add_trace(go.Scatter(
            x=ind_ts, y=rsi, mode="lines", name="RSI",
            line=dict(color="#ab47bc", width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,0,0,0.4)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,0,0.4)", row=2, col=1)

    # Row 3: MACD
    macd_line = indicators.get("macd", [])
    macd_signal = indicators.get("macd_signal", [])
    if macd_line and macd_signal and len(macd_line) == len(ind_ts):
        fig.add_trace(go.Scatter(
            x=ind_ts, y=macd_line, mode="lines", name="MACD",
            line=dict(color="#2196f3", width=1.5),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=ind_ts, y=macd_signal, mode="lines", name="Signal",
            line=dict(color="#ff5722", width=1.5),
        ), row=3, col=1)
        # Histogram
        hist = [m - s for m, s in zip(macd_line, macd_signal)]
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
        fig.add_trace(go.Bar(
            x=ind_ts, y=hist, name="MACD Hist",
            marker_color=colors, opacity=0.5,
        ), row=3, col=1)

    # Row 4: Volume
    if volumes and len(volumes) == len(ind_ts):
        vol_colors = ["#26a69a" if closes[i] >= opens[i] else "#ef5350"
                      for i in range(min(len(volumes), len(closes)))]
        fig.add_trace(go.Bar(
            x=ind_ts[:len(volumes)], y=volumes, name="Volume",
            marker_color=vol_colors[:len(volumes)], opacity=0.7,
        ), row=4, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", size=10),
        hovermode="x unified",
        dragmode="pan", # TradingView vibe: pan by default
    )

    # Style all axes for TradingView look
    for i in range(1, 5):
        fig.update_yaxes(
            gridcolor="#21262d", 
            zeroline=False, 
            showline=True, 
            linewidth=1, 
            linecolor="#30363d",
            row=i, col=1,
            side="right" # TradingView has price on the right
        )
        fig.update_xaxes(
            gridcolor="#21262d", 
            zeroline=False, 
            showline=True, 
            linewidth=1, 
            linecolor="#30363d",
            spikethickness=1,
            spikedash="dot",
            spikecolor="#8b949e",
            spikemode="across",
            row=i, col=1
        )

    # Specific for the main chart: Range slider and initial view
    fig.update_xaxes(
        row=4, col=1,
        rangeslider=dict(visible=True, thickness=0.05, bgcolor="#161b22")
    )
    
    # Hide the main candlestick range slider since we put it on the volume row
    fig.update_layout(xaxis_rangeslider_visible=False)

    return fig.to_json()


def build_equity_chart(equity_data: list) -> str:
    """
    Build an interactive equity curve chart.
    Returns Plotly figure as JSON string.
    """
    if not equity_data:
        fig = go.Figure()
        fig.add_annotation(text="No closed trades yet to build equity curve.", showarrow=False, font=dict(size=16, color="#888"))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
        return fig.to_json()

    timestamps = [d["timestamp"] for d in equity_data]
    equities = [d["equity"] for d in equity_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=equities, mode="lines+markers", name="Cumulative PnL",
        line=dict(color="#69f0ae", width=2),
        marker=dict(size=6, color="#69f0ae"),
        fill="tozeroy", fillcolor="rgba(105,240,174,0.1)"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=20, t=40, b=30),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        yaxis_title="Cumulative Realized PnL ($)",
        xaxis_title="Trade Exit Time",
        title="Cumulative Realized PnL (from closed trades)"
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")

    return fig.to_json()
