# Test script: test_connection.py
from Hybrid_Binance_Client import HybridBinanceClient

client = HybridBinanceClient()

# Test mainnet (real prices)
price = client.get_current_price()
print(f"✅ Mainnet working - Current ETH price: ${price:.2f}")

# Test testnet (paper trading)
balance = client.get_testnet_balance()
print(f"✅ Testnet working - Balance: ${balance:.2f}")

if balance < 100:
    print("⚠️  WARNING: Low balance. Fund your testnet account!")
else:
    print("✅ Ready to trade!")