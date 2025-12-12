Here is a **clean, concise, professional README** you can paste directly into your GitHub repo.
It’s shorter, polished, and still explains everything clearly.

---

# 🚀 Hybrid Crypto Trading Bot (Mainnet Data + Testnet Execution)

A machine-learning powered trading bot that uses **real Binance mainnet market data** while executing trades safely on the **Binance Futures Testnet**.
Includes full feature engineering, ML prediction pipeline, liquidation tracking, analytics, and Telegram alerts.

---

## ⚡ Quick Start

```bash
# 1. Set Binance Testnet API credentials
export BINANCE_TESTNET_API_KEY="your_key"
export BINANCE_TESTNET_API_SECRET="your_secret"

# 2. Run the bot
python Hybrid_Trading_Bot.py

# 3. View performance summary
python analyze_results.py
```

---

## 📁 Project Structure

| File                       | Description                                      |
| -------------------------- | ------------------------------------------------ |
| `Hybrid_Trading_Bot.py`    | Main trading engine (signals → orders → exits)   |
| `Hybrid_Binance_Client.py` | Fetch real mainnet data + execute testnet trades |
| `Telegram_alert.py`        | Sends entry/exit/error alerts                    |
| `feature_builder.py`       | Builds 40+ technical features                    |
| `trained_model.pkl`        | Your ML model for predictions                    |
| `analytic.py`              | Performance metrics (Sharpe, PF, DD)             |
| `hybrid_trades.json`       | Trade history                                    |
| `hybrid_state.json`        | Daily PnL, drawdown tracking                     |

---

## 🧠 How the Hybrid System Works

```
Mainnet (Real Prices) → ML Model → Signal → Testnet (Execution & PnL)
```

* Uses **live market volatility, volume, liquidity**
* Trades with **zero financial risk**
* Calculates **real PnL**, **real liquidation prices**, and **entry/exit accuracy**

This allows **forward-testing your strategy in real market conditions**.

---

## ⚙️ Key Configurations (edit in `Config` class)

```python
SYMBOL = "ETHUSDT"        # Trading pair
INTERVAL = "15m"          # Candle size
POSITION_SIZE_PCT = 0.06  # 6% capital per trade
LEVERAGE = 2              # Futures leverage
MIN_CONFIDENCE = 0.60     # ML confidence threshold
HOLD_BARS = 4             # Hold 1 hour (4×15m)
```

---

## 🔔 Telegram Alerts

You receive:

### Entry Alert

* Signal direction
* Entry price
* Confidence
* Leverage
* Position size
* Liquidation price

### Exit Alert

* Entry vs exit
* PnL (USD and %)
* Reason (TIME / EMERGENCY)
* Duration
* Updated performance metrics

### Liquidation Warning

Triggered when price moves too close to liquidation.

---

## 📊 Performance Metrics

Calculated in `analytic.py`:

* **Win Rate**
* **Profit Factor**
* **Sharpe Ratio**
* **Expectancy**
* **Max Drawdown**
* **Daily PnL**

Run:

```bash
python analyze_results.py
```

---

## 🌀 Bot Flow (Simplified)

```
1. Wait for candle close
2. Fetch 500 real mainnet candles
3. Build ML features
4. Predict LONG / SHORT + confidence
5. If confidence ≥ threshold → open testnet position
6. Hold for 1 hour or emergency exit
7. Close trade and log PnL
8. Repeat
```

## System Architecture
---
                   ┌────────────────────────────┐
                   │      User / Developer       │
                   │  - Start bot                │
                   │  - Configure settings       │
                   └───────────────┬────────────┘
                                   │
                                   ▼
                    ┌────────────────────────┐
                    │ Hybrid Trading Engine   │
                    │ (Hybrid_Trading_Bot.py) │
                    └───────────────┬────────┘
                                    │
         ┌──────────────────────────┼───────────────────────────┐
         │                          │                           │
         ▼                          ▼                           ▼
┌───────────────────┐    ┌─────────────────────┐    ┌────────────────────────┐
│ Binance Mainnet    │    │ Machine Learning    │    │ Binance Testnet        │
│ (Real Market Data) │    │ (trained_model.pkl) │    │ (Order Execution)      │
│ - Prices           │    │ - Feature builder   │    │ - Open/close trades    │
│ - Volume           │    │ - Predict LONG/SHORT│    │ - Real PnL calculation │
│ - Volatility       │    │ - Confidence score  │    │ - Liquidation tracking │
└───────────┬────────┘    └─────────────┬──────┘    └──────────────┬─────────┘
            │                           │                           │
            │                           │                           │
            ▼                           ▼                           ▼
      ┌────────────────────────────────────────────────────────────────────┐
      │                  Trade Execution Decision Logic                    │
      │   - Threshold checks (confidence, DD, daily loss)                 │
      │   - Entry position sizing (with leverage)                         │
      │   - Exit logic (time-based or emergency)                          │
      └───────────────┬──────────────────────────────────────────────────┘
                      │
                      ▼
         ┌───────────────────────────────┐
         │   State & Trade Management    │
         │  - hybrid_state.json          │
         │  - hybrid_trades.json         │
         │  - PnL tracking               │
         └──────────────┬────────────────┘
                        │
                        ▼
           ┌──────────────────────────────┐
           │    Performance Analytics     │
           │       (analytic.py)          │
           │ - Sharpe Ratio               │
           │ - Profit Factor              │
           │ - Win Rate                   │
           │ - Max Drawdown               │
           └──────────────┬───────────────┘
                          │
                          ▼
            ┌────────────────────────────────┐
            │   Telegram Alerts (Bot API)    │
            │ - Entry/Exit notifications     │
            │ - Liquidation warnings         │
            │ - Error reporting              │
            └────────────────────────────────┘



## 🆘 Emergency Tools

```python
# Close all open testnet positions
from Hybrid_Binance_Client import HybridBinanceClient
HybridBinanceClient().close_testnet_position()
```

---

## 🛡️ Safety

* Daily loss limit
* Max drawdown stop
* Liquidation proximity alerts
* Only trades on **testnet**
* `.env` keeps credentials private

---

## 📌 Requirements

```
python >= 3.10
binance-connector
pandas, numpy
scikit-learn
requests
python-dotenv
```

---

## ⭐ Summary

This project provides a **full forward-testing framework** combining:

* Real market data
* Machine learning
* Paper trading
* Automated execution
* Analytics
* Telegram notifications

Perfect for safely testing and improving algorithmic trading systems.
