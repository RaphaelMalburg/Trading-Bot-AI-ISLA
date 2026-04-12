"""
SQLite database for persistent storage of runs, trades, and statistics.
"""
import sqlite3
import os
from datetime import datetime
import threading

DB_PATH = "data/trading_bot.db"

# Thread-safe lock for database access
_db_lock = threading.Lock()


def init_db():
    """Initialize database schema if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # Runs table: one entry per hourly bot execution
        cursor.execute("""
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
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Trades table: filled orders with outcomes
        cursor.execute("""
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

        # Statistics table: cached daily stats for performance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                trades_total INTEGER DEFAULT 0,
                trades_won INTEGER DEFAULT 0,
                trades_lost INTEGER DEFAULT 0,
                trades_pending INTEGER DEFAULT 0,
                win_rate REAL DEFAULT 0,
                total_pnl_dollars REAL DEFAULT 0,
                total_pnl_percent REAL DEFAULT 0,
                equity_start REAL,
                equity_end REAL,
                best_trade REAL,
                worst_trade REAL,
                avg_duration_hours REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


def store_run(run_data: dict) -> int:
    """Store a bot run. Returns the run_id."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO runs (
                        timestamp, btc_close, prediction, prediction_label,
                        confidence, sentiment_score, action, order_id,
                        position_qty, leverage, stop_loss, take_profit, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_data.get("timestamp"),
                    run_data.get("btc_close"),
                    run_data.get("prediction"),
                    run_data.get("prediction_label"),
                    run_data.get("confidence"),
                    run_data.get("sentiment_score"),
                    run_data.get("action"),
                    run_data.get("order_id"),
                    run_data.get("position_qty"),
                    run_data.get("leverage"),
                    run_data.get("stop_loss"),
                    run_data.get("take_profit"),
                    run_data.get("error"),
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Run already exists for this timestamp (e.g., retry)
                return None


def get_recent_runs(limit: int = 20) -> list[dict]:
    """Get recent bot runs."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]


def store_trade(run_id: int, trade_data: dict) -> int:
    """Store a filled trade (opened order). Returns trade_id."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    run_id, order_id, entry_price, entry_time, qty
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                run_id,
                trade_data.get("order_id"),
                trade_data.get("entry_price"),
                trade_data.get("entry_time"),
                trade_data.get("qty"),
            ))
            conn.commit()
            return cursor.lastrowid


def close_trade(order_id: str, exit_price: float, exit_time: str, exit_reason: str) -> bool:
    """Close/update a trade with exit details. Returns True if successful."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Get entry details
            cursor.execute("SELECT entry_price, entry_time, qty FROM trades WHERE order_id = ?", (order_id,))
            row = cursor.fetchone()
            if not row:
                return False

            entry_price, entry_time_str, qty = row

            # Calculate P&L
            pnl_dollars = (exit_price - entry_price) * qty
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100

            # Calculate duration
            try:
                entry_dt = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                duration_hours = (exit_dt - entry_dt).total_seconds() / 3600
            except:
                duration_hours = 0

            # Update trade
            cursor.execute("""
                UPDATE trades SET
                    exit_price = ?, exit_time = ?, exit_reason = ?,
                    pnl_dollars = ?, pnl_percent = ?, duration_hours = ?
                WHERE order_id = ?
            """, (
                exit_price, exit_time, exit_reason,
                pnl_dollars, pnl_percent, duration_hours,
                order_id
            ))
            conn.commit()
            return True


def get_open_trades() -> list[dict]:
    """Get currently open trades (no exit_price)."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE exit_price IS NULL ORDER BY entry_time DESC")
            return [dict(row) for row in cursor.fetchall()]


def get_closed_trades(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get closed trades with P&L."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades WHERE exit_price IS NOT NULL
                ORDER BY exit_time DESC LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]

def get_equity_history() -> list[dict]:
    """Calculate the historical equity curve from closed trades."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Because the initial capital isn't hard stored as an initial event,
            # we'll approximate an equity curve from the sum of PnL over time.
            cursor.execute("""
                SELECT exit_time as timestamp, pnl_dollars 
                FROM trades 
                WHERE exit_price IS NOT NULL
                ORDER BY exit_time ASC
            """)
            trades = cursor.fetchall()
            
            history = []
            cumulative = 0
            for row in trades:
                cumulative += row['pnl_dollars']
                history.append({
                    "timestamp": row['timestamp'],
                    "equity": cumulative
                })
            return history


def get_statistics() -> dict:
    """Get overall statistics."""
    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # Total trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            total_trades = cursor.fetchone()[0]

            # Closed trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL")
            closed_trades = cursor.fetchone()[0]

            # Open trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NULL")
            open_trades = cursor.fetchone()[0]

            # Won trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL AND pnl_dollars > 0")
            won_trades = cursor.fetchone()[0]

            # Lost trades
            cursor.execute("SELECT COUNT(*) FROM trades WHERE exit_price IS NOT NULL AND pnl_dollars <= 0")
            lost_trades = cursor.fetchone()[0]

            # Total P&L
            cursor.execute("SELECT COALESCE(SUM(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL")
            total_pnl = cursor.fetchone()[0]

            # Average win/loss
            cursor.execute("SELECT COALESCE(AVG(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL AND pnl_dollars > 0")
            avg_win = cursor.fetchone()[0]

            cursor.execute("SELECT COALESCE(AVG(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL AND pnl_dollars < 0")
            avg_loss = cursor.fetchone()[0]

            cursor.execute("SELECT COALESCE(MAX(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL")
            best_trade = cursor.fetchone()[0]

            cursor.execute("SELECT COALESCE(MIN(pnl_dollars), 0) FROM trades WHERE exit_price IS NOT NULL")
            worst_trade = cursor.fetchone()[0]
            
            cursor.execute("SELECT COALESCE(AVG(duration_hours), 0) FROM trades WHERE exit_price IS NOT NULL")
            avg_duration = cursor.fetchone()[0]

            win_rate = (won_trades / closed_trades * 100) if closed_trades > 0 else 0

            return {
                "total_trades": total_trades,
                "closed_trades": closed_trades,
                "open_trades": open_trades,
                "won_trades": won_trades,
                "lost_trades": lost_trades,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": abs(avg_loss),
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "avg_duration_hours": avg_duration
            }


def sync_trades_from_alpaca(trading_client):
    """
    Fetch all bracket orders from Alpaca and sync their status into the trades table.
    Matches the parent BUY order to entry and child SELL orders to exit.
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderStatus

    req = GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True, limit=100)
    try:
        orders = trading_client.get_orders(req)
    except Exception as e:
        print(f"Error fetching orders for sync: {e}")
        return

    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            for o in orders:
                # We only care about the parent orders (e.g. bracket BUYs)
                # or single manual orders. If it's a bracket order, it has legs.
                if o.side.name != "BUY":
                    continue
                
                # We only store filled entries
                if o.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED) or not o.filled_at:
                    continue

                order_id = str(o.id)
                entry_price = float(o.filled_avg_price) if o.filled_avg_price else 0.0
                entry_time = o.filled_at.isoformat()
                qty = float(o.filled_qty) if o.filled_qty else float(o.qty)

                # Check if it already exists
                cursor.execute("SELECT id, exit_price FROM trades WHERE order_id = ?", (order_id,))
                row = cursor.fetchone()
                
                if not row:
                    # Insert new trade
                    cursor.execute("""
                        INSERT INTO trades (
                            run_id, order_id, entry_price, entry_time, qty
                        ) VALUES (NULL, ?, ?, ?, ?)
                    """, (order_id, entry_price, entry_time, qty))
                    conn.commit()
                
                # Check exit conditions (legs)
                exit_price = None
                exit_time = None
                exit_reason = None
                
                if o.legs:
                    for leg in o.legs:
                        if leg.status == OrderStatus.FILLED and leg.filled_at:
                            exit_price = float(leg.filled_avg_price) if leg.filled_avg_price else 0.0
                            exit_time = leg.filled_at.isoformat()
                            # Determine reason based on order type or limit vs stop
                            if leg.order_type.name == "LIMIT":
                                exit_reason = "Take Profit"
                            elif leg.order_type.name == "STOP":
                                exit_reason = "Stop Loss"
                            else:
                                exit_reason = "Closed"
                            break

                if exit_price is not None:
                    # Calculate P&L
                    pnl_dollars = (exit_price - entry_price) * qty
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

                    # Calculate duration
                    try:
                        entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        exit_dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                        duration_hours = (exit_dt - entry_dt).total_seconds() / 3600
                    except:
                        duration_hours = 0

                    cursor.execute("""
                        UPDATE trades SET
                            exit_price = ?, exit_time = ?, exit_reason = ?,
                            pnl_dollars = ?, pnl_percent = ?, duration_hours = ?
                        WHERE order_id = ? AND exit_price IS NULL
                    """, (
                        exit_price, exit_time, exit_reason,
                        pnl_dollars, pnl_percent, duration_hours,
                        order_id
                    ))
                    conn.commit()




def get_todays_statistics() -> dict:
    """Get statistics for today only."""
    from datetime import date

    with _db_lock:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            today = date.today().isoformat()

            # Today's trades
            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE DATE(entry_time) = ?
            """, (today,))
            total = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE DATE(entry_time) = ? AND exit_price IS NOT NULL AND pnl_dollars > 0
            """, (today,))
            won = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE DATE(entry_time) = ? AND exit_price IS NOT NULL AND pnl_dollars < 0
            """, (today,))
            lost = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COALESCE(SUM(pnl_dollars), 0) FROM trades
                WHERE DATE(entry_time) = ?
            """, (today,))
            pnl = cursor.fetchone()[0]

            return {
                "total": total,
                "won": won,
                "lost": lost,
                "pnl": pnl,
                "win_rate": (won / (won + lost) * 100) if (won + lost) > 0 else 0,
            }
