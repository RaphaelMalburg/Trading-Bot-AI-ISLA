#!/usr/bin/env python3
"""
Verify that bracket orders with SL/TP are being placed correctly.
Run this after the bot executes a trade.
"""

import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

print("\n" + "="*60)
print("BRACKET ORDER FIX VERIFICATION")
print("="*60 + "\n")

# Get open positions
positions = trading_client.get_all_positions()
if not positions:
    print("No open positions. Waiting for next trade...\n")
    exit(0)

print(f"Found {len(positions)} open position(s)\n")

# Get all orders
orders = trading_client.get_orders()
print(f"Total orders in account: {len(orders)}\n")

# Look for recent orders with SL/TP
print("Checking recent orders for bracket structure...\n")

if orders:
    # Sort by creation time, newest first
    sorted_orders = sorted(orders, key=lambda x: x.created_at, reverse=True)

    for i, order in enumerate(sorted_orders[:10], 1):
        print(f"{i}. Order ID: {order.id[:8]}...")
        print(f"   Symbol: {order.symbol} | Status: {order.status}")
        print(f"   Type: {order.order_type} | Side: {order.side}")
        print(f"   Qty: {order.qty}")

        has_sl = order.stop_loss is not None
        has_tp = order.take_profit is not None

        print(f"   Stop Loss: {'YES ✓' if has_sl else 'NO ✗'} {order.stop_loss if has_sl else ''}")
        print(f"   Take Profit: {'YES ✓' if has_tp else 'NO ✗'} {order.take_profit if has_tp else ''}")
        print()

print("="*60)
print("INTERPRETATION:")
print("="*60)
print("✓ If recent orders show 'Stop Loss: YES' and 'Take Profit: YES'")
print("  → THE FIX IS WORKING!")
print()
print("✗ If they show 'NO' for both")
print("  → The fix may not be deployed or there's another issue")
print("="*60 + "\n")
