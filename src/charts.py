"""
Plotly chart builder for the trading dashboard.
Builds a multi-subplot candlestick chart with technical indicators.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_candlestick_chart(run_data: dict) -> str:
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
        height=750,
        margin=dict(l=50, r=20, t=40, b=30),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
    )

    # Style all y-axes
    for i in range(1, 5):
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=i, col=1)
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", row=i, col=1)

    return fig.to_json()
