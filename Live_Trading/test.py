# Test script: test_connection.py
from Hybrid_Binance_Client import HybridBinanceClient

client = HybridBinanceClient()

# test_connection_fixed.py
import os
from binance.client import Client
from datetime import datetime

# Test time synchronization
testnet_key = os.getenv('BINANCE_TESTNET_API_KEY')
testnet_secret = os.getenv('BINANCE_TESTNET_API_SECRET')

if not testnet_key or not testnet_secret:
    print("❌ Testnet credentials not found in environment")
    exit(1)

client = Client(testnet_key, testnet_secret, testnet=True)

# Test 1: Server time
try:
    server_time = client.get_server_time()
    print(f"✅ Server time: {server_time['serverTime']}")
    print(f"   Local time: {int(datetime.now().timestamp() * 1000)}")
except Exception as e:
    print(f"❌ Time sync failed: {e}")

# Test 2: Account balance
try:
    balance = client.futures_account_balance()
    usdt_balance = next((float(b['balance']) for b in balance if b['asset'] == 'USDT'), 0)
    print(f"✅ Testnet balance: ${usdt_balance:.2f}")
except Exception as e:
    print(f"❌ Balance check failed: {e}")

# Test 3: Set leverage
try:
    client.futures_change_leverage(symbol='ETHUSDT', leverage=2)
    print("✅ Leverage set successfully")
except Exception as e:
    print(f"❌ Leverage set failed: {e}")