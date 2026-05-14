"""
SQLite persistence for runs, trades, and statistics.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import date, datetime, timezone
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)

DB_PATH = "data/trading_bot.db"
_db_lock = threading.Lock()


class RunDict(TypedDict, total=False):
    timestamp: str
    btc_close: float
    prediction: int
    prediction_label: str
    confidence: float
    sentiment_score: float
    action: str
    order_id: Optional[str]
    position_qty: float
    leverage: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    error: Optional[str]
    equity: float
    buying_power: float
    pnl_today: float
    pnl_today_pct: float
    drift_warning: bool


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT UNIQUE NOT NULL,
                btc_close REAL NOT NULL,
                prediction INTEGER NOT NULL,
                prediction_label TEXT,
                confidence REAL NOT NULL,
                sentiment_score REAL,
                action TEXT,
                order_id TEXT,
                position_qty REAL,
                leverage REAL,
                stop_loss REAL,
                take_profit REAL,
                equity REAL,
                buying_power REAL,
                pnl_today REAL,
                pnl_today_pct REAL,
                drift_warning INTEGER DEFAULT 0,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                order_id TEXT UNIQUE,
                entry_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_price REAL,
                exit_time TEXT,
                exit_reason TEXT,
                pnl_dollars REAL,
                pnl_percent REAL,
                duration_hours REAL,
                qty REAL NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            )
        """)
        conn.commit()


def store_run(run: dict) -> Optional[int]:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            try:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO runs (
                        timestamp, btc_close, prediction, prediction_label,
                        confidence, sentiment_score, action, order_id,
                        position_qty, leverage, stop_loss, take_profit, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run.get("timestamp"), run.get("btc_close"), run.get("prediction"),
                    run.get("prediction_label"), run.get("confidence"), run.get("sentiment_score"),
                    run.get("action"), run.get("order_id"), run.get("position_qty"),
                    run.get("leverage"), run.get("stop_loss"), run.get("take_profit"),
                    run.get("error"),
                ))
                conn.commit()
                return c.lastrowid
            except sqlite3.IntegrityError:
                return None


def store_trade(run_id: int, trade: dict) -> int:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO trades (run_id, order_id, entry_price, entry_time, qty)
                VALUES (?, ?, ?, ?, ?)
            """, (run_id, trade.get("order_id"), trade.get("entry_price"), trade.get("entry_time"), trade.get("qty")))
            conn.commit()
            return c.lastrowid


def close_trade(order_id: str, exit_price: float, exit_time: str, exit_reason: str) -> bool:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT entry_price, entry_time, qty FROM trades WHERE order_id = ?", (order_id,))
            row = c.fetchone()
            if not row:
                return False
            entry_price, entry_time_str, qty = row
            pnl_dollars = (exit_price - entry_price) * qty
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            try:
                entry_dt = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                duration_hours = (exit_dt - entry_dt).total_seconds() / 3600
            except Exception:
                duration_hours = 0
            c.execute("""
                UPDATE trades SET exit_price=?, exit_time=?, exit_reason=?,
                    pnl_dollars=?, pnl_percent=?, duration_hours=?
                WHERE order_id=?
            """, (exit_price, exit_time, exit_reason, pnl_dollars, pnl_percent, duration_hours, order_id))
            conn.commit()
            return True


def mark_all_open_as_exited(exit_price: float, exit_reason: str = "KILL_SWITCH") -> int:
    exit_time = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT order_id, entry_price, entry_time, qty FROM trades WHERE exit_price IS NULL")
            rows = c.fetchall()
            for order_id, entry_price, entry_time_str, qty in rows:
                if not entry_price or not qty:
                    continue
                pnl_dollars = (exit_price - entry_price) * qty
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0
                try:
                    duration_hours = (
                        datetime.fromisoformat(exit_time.replace("Z", "+00:00")) -
                        datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    ).total_seconds() / 3600
                except Exception:
                    duration_hours = 0
                c.execute("""
                    UPDATE trades SET exit_price=?, exit_time=?, exit_reason=?,
                        pnl_dollars=?, pnl_percent=?, duration_hours=?
                    WHERE order_id=? AND exit_price IS NULL
                """, (exit_price, exit_time, exit_reason, pnl_dollars, pnl_percent, duration_hours, order_id))
            conn.commit()
            return len(rows)


def get_recent_runs(limit: int = 20) -> list[dict]:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in c.fetchall()]


def get_open_trades() -> list[dict]:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM trades WHERE exit_price IS NULL ORDER BY entry_time DESC")
            return [dict(r) for r in c.fetchall()]


def get_closed_trades(limit: int = 50, offset: int = 0) -> list[dict]:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT * FROM trades WHERE exit_price IS NOT NULL
                ORDER BY exit_time DESC LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(r) for r in c.fetchall()]


def get_equity_history() -> list[dict]:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT exit_time as timestamp, pnl_dollars FROM trades
                WHERE exit_price IS NOT NULL ORDER BY exit_time ASC
            """)
            history = []
            cumulative = 0.0
            for row in c.fetchall():
                cumulative += row["pnl_dollars"]
                history.append({"timestamp": row["timestamp"], "equity": cumulative})
            return history


def get_statistics() -> dict:
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL")
            closed = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL AND pnl_dollars > 0")
            won = c.fetchone()[0]
            c.execute("SELECT COALESCE(SUM(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL")
            total_pnl = c.fetchone()[0]
            c.execute("SELECT COALESCE(AVG(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL AND pnl_dollars > 0")
            avg_win = c.fetchone()[0]
            c.execute("SELECT COALESCE(AVG(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL AND pnl_dollars < 0")
            avg_loss = c.fetchone()[0]
            win_rate = (won / closed * 100) if closed > 0 else 0
            return {
                "closed_trades": closed,
                "won_trades": won,
                "lost_trades": closed - won,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": abs(avg_loss),
            }


def get_todays_statistics() -> dict:
    today = date.today().isoformat()
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = ?", (today,))
            total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = ? AND exit_price IS NOT NULL AND pnl_dollars > 0", (today,))
            won = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = ? AND exit_price IS NOT NULL AND pnl_dollars < 0", (today,))
            lost = c.fetchone()[0]
            c.execute("SELECT COALESCE(SUM(pnl_dollars), 0) FROM trades WHERE DATE(entry_time) = ?", (today,))
            pnl = c.fetchone()[0]
            return {
                "total": total, "won": won, "lost": lost, "pnl": pnl,
                "win_rate": (won / (won + lost) * 100) if (won + lost) > 0 else 0,
            }


def get_daily_pnl(days: int = 14) -> list[dict]:
    """Return per-day PnL for the last N days."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("""
                SELECT DATE(exit_time) as day,
                       COALESCE(SUM(pnl_dollars), 0) as pnl,
                       COUNT(*) as trades
                FROM trades
                WHERE exit_price IS NOT NULL
                  AND exit_time IS NOT NULL
                  AND DATE(exit_time) >= DATE('now', ? || ' days')
                GROUP BY DATE(exit_time)
                ORDER BY day ASC
            """, (f"-{days}",))
            return [dict(r) for r in c.fetchall()]


def sync_closed_trades_only(trading_client) -> int:
    """Lightweight sync: close any open DB trades whose SELL has filled on Alpaca."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderStatus

    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT MIN(entry_time) FROM trades WHERE exit_price IS NULL")
            row = c.fetchone()
            oldest_open = row[0] if row else None

    if not oldest_open:
        return 0

    try:
        after_dt = datetime.fromisoformat(oldest_open.replace("Z", "+00:00"))
    except Exception:
        after_dt = None

    try:
        orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100, after=after_dt))
    except Exception as e:
        logger.error("Sync: error fetching orders: %s", e)
        return 0

    sells = [
        o for o in orders
        if o.filled_at and o.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED)
        and o.side and o.side.name == "SELL"
    ]
    sells.sort(key=lambda o: o.filled_at)

    if not sells:
        return 0

    def _exit_reason(o) -> str:
        t = o.order_type.name if o.order_type else ""
        if t == "LIMIT":
            return "Take Profit"
        if t in ("STOP", "STOP_LIMIT"):
            return "Stop Loss"
        return "Closed"

    closed = 0
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            for o in sells:
                exit_price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
                if exit_price <= 0:
                    continue
                exit_time = o.filled_at.isoformat()
                c.execute("""
                    SELECT id, entry_price, entry_time, qty FROM trades
                    WHERE exit_price IS NULL AND entry_time <= ?
                    ORDER BY entry_time ASC LIMIT 1
                """, (exit_time,))
                row = c.fetchone()
                if not row:
                    continue
                trade_id, entry_price, entry_time_str, qty = row
                pnl_dollars = (exit_price - entry_price) * qty
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100 if entry_price else 0
                try:
                    duration_hours = (
                        datetime.fromisoformat(exit_time.replace("Z", "+00:00")) -
                        datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    ).total_seconds() / 3600
                except Exception:
                    duration_hours = 0
                c.execute("""
                    UPDATE trades SET exit_price=?, exit_time=?, exit_reason=?,
                        pnl_dollars=?, pnl_percent=?, duration_hours=?
                    WHERE id=? AND exit_price IS NULL
                """, (exit_price, exit_time, _exit_reason(o), pnl_dollars, pnl_percent, duration_hours, trade_id))
                if c.rowcount > 0:
                    closed += 1
            conn.commit()

    if closed:
        logger.info("Synced %d closed trade(s)", closed)
    return closed
