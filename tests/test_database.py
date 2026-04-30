"""
Database operations tests — trade ledger, statistics, sync logic.
"""

import os
import sys
import pytest
import tempfile
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import (
    init_db,
    store_run,
    store_trade,
    close_trade,
    get_statistics,
    get_todays_statistics,
    get_closed_trades,
    update_drift_metrics,
)
import sqlite3


@pytest.fixture
def db_conn():
    """Use a temporary test database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    old_db = os.environ.get("TRADING_BOT_DB")
    os.environ["TRADING_BOT_DB"] = tmp.name

    init_db()
    yield tmp.name

    os.unlink(tmp.name)
    if old_db is not None:
        os.environ["TRADING_BOT_DB"] = old_db


def test_store_and_retrieve_run(db_conn):
    run_id = store_run(
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "btc_close": 42000.0,
            "prediction": 1,
            "prediction_label": "LONG",
            "confidence": 0.65,
            "sentiment_score": 0.2,
            "action": "BUY_ORDER_SENT",
            "order_id": "order_123",
            "position_qty": 0.1,
            "leverage": 0.25,
            "stop_loss": 41000.0,
            "take_profit": 43000.0,
            "error": None,
        }
    )
    assert run_id is not None

    # Verify via raw query
    conn = sqlite3.connect(db_conn)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()

    assert row[1] == "2024-01-01T00:00:00Z"
    assert row[3] == 1
    assert row[6] == "BUY_ORDER_SENT"


def test_store_and_close_trade(db_conn):
    run_id = store_run(
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "btc_close": 42000.0,
            "prediction": 1,
            "prediction_label": "LONG",
            "confidence": 0.65,
            "sentiment_score": 0.2,
            "action": "BUY_ORDER_SENT",
            "order_id": "order_123",
            "position_qty": 0.1,
            "leverage": 0.25,
            "stop_loss": 41000.0,
            "take_profit": 43000.0,
            "error": None,
        }
    )

    trade_id = store_trade(
        run_id,
        {
            "order_id": "order_123",
            "entry_price": 42000.0,
            "entry_time": "2024-01-01T00:00:00Z",
            "qty": 0.1,
        },
    )
    assert trade_id > 0

    # Close the trade
    exit_time = "2024-01-01T02:00:00Z"
    success = close_trade("order_123", 43000.0, exit_time, "Take Profit")
    assert success

    # Verify P&L
    conn = sqlite3.connect(db_conn)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT pnl_dollars, pnl_percent FROM trades WHERE id = ?", (trade_id,)
    )
    row = cursor.fetchone()
    conn.close()

    pnl_dollars, pnl_percent = row
    expected_pnl = (43000.0 - 42000.0) * 0.1  # $100
    assert abs(pnl_dollars - expected_pnl) < 0.01
    assert abs(pnl_percent - 2.38) < 0.1  # ~2.38% gain


def test_get_statistics_aggregates_correctly(db_conn):
    # Create 3 trades: 2 wins, 1 loss
    for i in range(3):
        run_id = store_run(
            {
                "timestamp": f"2024-01-0{i + 1}T00:00:00Z",
                "btc_close": 42000.0,
                "prediction": 1,
                "prediction_label": "LONG",
                "confidence": 0.65,
                "sentiment_score": 0.2,
                "action": "BUY_ORDER_SENT",
                "order_id": f"order_{i}",
                "position_qty": 0.1,
                "leverage": 0.25,
                "stop_loss": 41000.0,
                "take_profit": 43000.0,
                "error": None,
            }
        )
        store_trade(
            run_id,
            {
                "order_id": f"order_{i}",
                "entry_price": 42000.0,
                "entry_time": f"2024-01-0{i + 1}T00:00:00Z",
                "qty": 0.1,
            },
        )
        # Close: +100 for first two, -50 for last
        exit_price = 43000.0 if i < 2 else 41500.0
        close_trade(f"order_{i}", exit_price, f"2024-01-0{i + 1}T02:00:00Z", "Test")

    stats = get_statistics()
    assert stats["total_trades"] == 3
    assert stats["won_trades"] == 2
    assert stats["lost_trades"] == 1
    assert stats["win_rate"] == pytest.approx(66.67, 0.1)
    assert stats["total_pnl"] > 0


def test_get_todays_statistics_filters_by_date(db_conn):
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    # Today's trade
    run_id_today = store_run(
        {
            "timestamp": today.isoformat(),
            "btc_close": 42000.0,
            "prediction": 1,
            "prediction_label": "LONG",
            "confidence": 0.65,
            "sentiment_score": 0.2,
            "action": "BUY_ORDER_SENT",
            "order_id": "order_today",
            "position_qty": 0.1,
            "leverage": 0.25,
            "stop_loss": 41000.0,
            "take_profit": 43000.0,
            "error": None,
        }
    )
    store_trade(
        run_id_today,
        {
            "order_id": "order_today",
            "entry_price": 42000.0,
            "entry_time": today.isoformat(),
            "qty": 0.1,
        },
    )
    close_trade("order_today", 43000.0, today.isoformat(), "Take Profit")

    # Yesterday's trade
    run_id_yesterday = store_run(
        {
            "timestamp": yesterday.isoformat(),
            "btc_close": 42000.0,
            "prediction": 1,
            "prediction_label": "LONG",
            "confidence": 0.65,
            "sentiment_score": 0.2,
            "action": "BUY_ORDER_SENT",
            "order_id": "order_yesterday",
            "position_qty": 0.1,
            "leverage": 0.25,
            "stop_loss": 41000.0,
            "take_profit": 43000.0,
            "error": None,
        }
    )
    store_trade(
        run_id_yesterday,
        {
            "order_id": "order_yesterday",
            "entry_price": 42000.0,
            "entry_time": yesterday.isoformat(),
            "qty": 0.1,
        },
    )
    close_trade("order_yesterday", 43000.0, yesterday.isoformat(), "Take Profit")

    today_stats = get_todays_statistics()
    assert today_stats["total"] == 1
    assert today_stats["won"] == 1


def test_drift_metrics_update(db_conn):
    run_id = store_run(
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "btc_close": 42000.0,
            "prediction": 1,
            "prediction_label": "LONG",
            "confidence": 0.65,
            "sentiment_score": 0.2,
            "action": "NO_SIGNAL",
            "error": None,
        }
    )

    update_drift_metrics(
        run_id,
        {
            "drift_warning": True,
            "avg_confidence": 0.48,
            "recent_runs": 50,
        },
    )

    # Verify update
    conn = sqlite3.connect(db_conn)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT error, drift_warning FROM runs WHERE id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()

    assert row["drift_warning"] == 1  # SQLite stores BOOLEAN as INTEGER 0/1
    assert "drift_warning" in row["error"]
    assert "avg_confidence" in row["error"]
