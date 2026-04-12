import sqlite3
import pandas as pd

try:
    with sqlite3.connect("data/trading_bot.db") as conn:
        df = pd.read_sql_query("SELECT timestamp, action, error FROM runs ORDER BY timestamp DESC LIMIT 5", conn)
        print(df)
except Exception as e:
    print(e)
