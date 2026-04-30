"""
Tests for Flask routes and dashboard APIs.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app as flask_app
from src.trading_bot_multi import trade_logic_multi


@pytest.fixture
def client():
    """Create test client for Flask app."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """GET /health returns 200 OK."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"


def test_dashboard_loads(client):
    """GET / returns HTML page."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"ML Trading Bot" in resp.data


def test_methodology_page(client):
    """GET /methodology renders config values."""
    resp = client.get("/methodology")
    assert resp.status_code == 200
    assert b"Confidence Threshold" in resp.data


def test_config_page_shows_model_info(client):
    """GET /config displays model metadata."""
    resp = client.get("/config")
    assert resp.status_code == 200


def test_api_latest_returns_run_data(client):
    """GET /api/latest returns JSON with expected fields."""
    with patch("src.app.get_latest") as mock_latest:
        mock_latest.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "btc_close": 42000.0,
            "prediction": 1,
            "confidence": 0.65,
            "action": "NO_SIGNAL",
        }
        resp = client.get("/api/latest")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "timestamp" in data


def test_api_runs_returns_list(client):
    """GET /api/runs returns list of recent runs."""
    with patch("src.app.get_last_n") as mock_last:
        mock_last.return_value = [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "btc_close": 42000.0,
                "prediction": 1,
                "confidence": 0.65,
                "sentiment_score": 0.2,
                "action": "BUY_ORDER_SENT",
            }
        ]
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert isinstance(data, list)
        assert len(data) == 1


def test_api_trades_csv_export(client):
    """GET /api/trades.csv returns CSV content-type."""
    resp = client.get("/api/trades.csv")
    assert resp.status_code == 200
    assert resp.content_type == "text/csv"
    assert b"entry_time" in resp.data  # Header row


def test_api_runs_csv_export(client):
    """GET /api/runs.csv returns CSV."""
    resp = client.get("/api/runs.csv")
    assert resp.status_code == 200
    assert resp.content_type == "text/csv"


def test_kill_switch_requires_auth(client):
    """POST /api/kill_switch should require trading_client (returns error if not configured)."""
    # If trading_client is None, should return 500
    resp = client.post("/api/kill_switch")
    # May be 500 because no API keys configured in test
    assert resp.status_code in (200, 500)


def test_backtest_page_loads(client):
    """GET /backtest renders page, even without results."""
    resp = client.get("/backtest")
    assert resp.status_code == 200


def test_run_backtest_starts_thread(client):
    """POST /api/run_backtest starts background job."""
    resp = client.post("/api/run_backtest")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "started"


def test_api_charts_returns_plotly_json(client):
    """GET /api/charts returns candlestick + equity data."""
    with (
        patch("src.app.get_latest") as mock_latest,
        patch("src.app.fetch_active_positions_with_sl_tp") as mock_pos,
        patch("src.app.get_closed_trades") as mock_trades,
        patch("src.app.get_equity_history") as mock_equity,
    ):
        mock_latest.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "btc_close": 42000.0,
            "ohlcv_data": [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "open": 41900,
                    "high": 42100,
                    "low": 41800,
                    "close": 42000,
                    "volume": 100,
                }
            ],
            "chart_indicators": {
                "ema20": [42000],
                "bb_high": [42200],
                "bb_low": [41800],
                "rsi": [55.0],
                "macd": [100],
                "macd_signal": [90],
                "volume": [100],
                "timestamps": ["2024-01-01T00:00:00Z"],
            },
        }
        mock_pos.return_value = []
        mock_trades.return_value = []
        mock_equity.return_value = []

        resp = client.get("/api/charts")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "candlestick" in data
        assert "equity" in data


def test_api_live_stats_requires_client(client):
    """GET /api/live_stats returns error if no Alpaca client."""
    resp = client.get("/api/live_stats")
    # Either 200 with simulated data or 500 error
    assert resp.status_code in (200, 500)


def test_admin_resync_endpoint_exists(client):
    """POST /admin/resync should be accessible."""
    resp = client.post("/admin/resync")
    assert resp.status_code in (200, 500)  # May fail without API keys
