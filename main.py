"""
Entry point: starts the Flask dashboard + background trading loop.
"""

import logging
import os
import sys
import threading

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Dirs must exist before FileHandler is created
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/trading_bot.log", encoding="utf-8"),
    ],
)

if __name__ == "__main__":
    from src.database import init_db
    init_db()

    from src.app import app, bot_loop, exit_watchdog, _restore_active_trade, _seed_history_from_db

    _restore_active_trade()
    _seed_history_from_db()
    threading.Thread(target=bot_loop, daemon=True).start()
    threading.Thread(target=exit_watchdog, daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    logging.getLogger(__name__).info("Starting dashboard on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
