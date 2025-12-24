Here you go — **pure Markdown**, ready to **paste directly into `README.md`**.
(No extra commentary, no formatting tricks.)

---

```md
# 🚀 Hybrid Crypto Trading Bot  
**(Binance Mainnet Data + Testnet Execution)**

This repository contains **two generations** of a machine-learning–driven crypto trading system designed for **safe forward testing** on Binance Futures.

The project evolved from an initial **hybrid trading bot (V1)** into a **production-grade engine-based system (V2)** with clean architecture, deterministic exits, and restart-safe state management.

---

## ⚠️ Version Overview (Important)

### 🟢 V1 – Hybrid Trading Bot (Legacy / Stable)

**Location:** `Live_Trading/`

- Monolithic execution script
- Uses **real Binance mainnet market data**
- Executes trades on **Binance Futures Testnet**
- ML model predicts direction + confidence
- Fixed HOLD_BARS–based exits
- Telegram alerts
- Performance analytics (Sharpe, PF, DD)
- JSON-based trade & state tracking

This version represents the **initial live trading implementation** and is kept for historical reference and comparison.

---

### 🔵 V2 – Engine-Based Trading System (Current / Recommended)

**Location:** `Live_Trading_V2/`

- Modular, production-style architecture
- Strict separation of concerns:
  - Engine
  - Risk Manager
  - Strategy (ML)
  - Execution
  - State Persistence
- **ML is entry-only** (no ML exits)
- **Deterministic time-based exits**
- Volatility regime filtering (ATR vs ATR MA)
- Pyramiding support in expanding volatility
- Candle-synchronized scheduler
- Restart-safe, crash-safe state persistence
- Structured logging
- Designed for long-running VM execution

👉 **V2 is the recommended reference implementation.**

---

## 📁 Repository Structure

```

.
├── Live_Trading/              # V1 – Hybrid trading bot (legacy)
│   ├── Hybrid_Trading_Bot.py
│   ├── Hybrid_Binance_Client.py
│   ├── trained_model.pkl
│   └── ...
│
├── Live_Trading_V2/           # V2 – Engine-based system (current)
│   ├── core/                  # Engine, state manager, logger
│   ├── exchange/              # Binance data + testnet execution
│   ├── features/              # Feature engineering
│   ├── risk/                  # Risk manager
│   ├── strategy/              # ML strategy (entry-only)
│   ├── analytics/             # (future)
│   ├── alerts/                # (future)
│   ├── logs/
│   ├── run.py                 # Scheduler entry point
│   └── global_trading_model.bundle
│
├── Training_Pipeline/         # Model training & research
│   ├── training.ipynb
│   ├── v-2_model.ipynb
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## 🧠 High-Level System Concepts

### V1 – Hybrid Bot Flow

```

Mainnet Prices → Feature Builder → ML Model
→ Signal → Testnet Execution → JSON Logs → Analytics

```

### V2 – Engine-Based Architecture

```

Market Data
↓
Feature Builder
↓
ML Strategy (ENTRY ONLY)
↓
Risk Manager (permission + sizing)
↓
Engine (trade lifecycle owner)
↓
Execution (Testnet)
↓
State Persistence + Logs

````

**Key philosophy in V2:**

> ML suggests.  
> Risk decides.  
> Engine executes.  
> Exits are deterministic.

---

## ⚙️ Running the Systems

### V1 – Hybrid Trading Bot

```bash
export BINANCE_TESTNET_API_KEY=...
export BINANCE_TESTNET_API_SECRET=...

python Live_Trading/Hybrid_Trading_Bot.py
````

Optional analytics:

```bash
python Live_Trading/analyze_results.py
```

---

### V2 – Engine-Based Trading System

```bash
export BINANCE_TESTNET_API_KEY=...
export BINANCE_TESTNET_API_SECRET=...

cd Live_Trading_V2
python run.py
```

The V2 system:

* Runs continuously
* Executes **once per closed candle**
* Is safe to restart (state is persisted)
* Designed for VM / server deployment

---

## 🔒 Risk & Safety (Both Versions)

* Trades only on **Binance Futures Testnet**
* Daily loss & drawdown controls
* Confidence-based entry filtering
* Volatility regime filtering (V2)
* No real capital at risk
* Credentials stored via environment variables

---

## 📌 What Is *Not* Included Yet (V2, by Design)

The following are intentionally deferred until strategy behavior is validated:

* Trading fees & funding accounting
* Telegram alerts
* Performance analytics scripts
* Multi-symbol portfolio execution

These will be added incrementally once forward-testing results are stable.

---

## 🎯 Project Goal

This repository is **not** a “plug-and-play trading bot”.

It is a **research → forward-testing → system-design project** focused on:

* learning how real trading systems are built
* validating ML signals under live conditions
* enforcing professional risk and execution discipline
* evolving architecture over time

---

## ⭐ Summary

This project demonstrates the **evolution of a trading system**:

* V1 shows rapid prototyping and experimentation
* V2 shows architectural maturity and production thinking

Both are kept intentionally to show **learning, iteration, and system design growth**.

```

---

If you want next:
- a **shorter GitHub landing README**
- a **V2-only README**
- or a **system diagram in Markdown**

just tell me.
```
