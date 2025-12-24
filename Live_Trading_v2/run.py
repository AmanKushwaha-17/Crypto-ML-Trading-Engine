import time
from datetime import datetime

from core.engine import Engine
from core.state_manager import setup_state, save_state
from exchange.binance_data import BinanceMarketData
from features.binance_crypto import BinanceCryptoFeatures
from strategy.ml_strategy import MLStrategy
from risk.eth_risk import EthRiskManager
from exchange.binance_testnet_execution import BinanceTestnetExecution
import os 
import warnings
warnings.filterwarnings("ignore")


API_KEY = os.getenv("BINANCE_TESTNET_API_KEY")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET")


SYMBOL = "ETHUSDT"
TIMEFRAME = "15m"
CHECK_INTERVAL_SECONDS = 30      # poll every 30s
INITIAL_CAPITAL = 500.0

MARKET_META = {
    "step_size": 0.01,
    "min_size": 0.01,
}

# =====================
# SETUP
# =====================
state, portfolio = setup_state(initial_capital=INITIAL_CAPITAL)

engine = Engine(
    data=BinanceMarketData(),
    features=BinanceCryptoFeatures(),
    strategy=MLStrategy("global_trading_model.bundle"),
    risk=EthRiskManager(
        leverage=5,
        risk_per_trade=0.06,
        portfolio_dd_limit=-0.30,
        min_confidence=0.60,
    ),
    execution=BinanceTestnetExecution(API_KEY, API_SECRET),
    state=state,
    portfolio=portfolio,
)


market = BinanceMarketData()

print("🚀 Bot started (15m candle scheduler)")

# =====================
# MAIN LOOP
# =====================
while True:
    try:
        candles = market.get_candles(SYMBOL, TIMEFRAME, 3)
        if candles is None or len(candles) < 3:
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        last_closed_time = candles.iloc[-2]["open_time"]

        # ---- first run ----
        if state.get("last_candle_time") is None:
            state["last_candle_time"] = last_closed_time
            save_state(state, portfolio)
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        # ---- new candle detected ----
        if last_closed_time > state["last_candle_time"]:
            print(
                f"🕒 New candle detected: "
                f"{datetime.utcfromtimestamp(last_closed_time / 1000)}"
            )

            engine.run_once(
                symbol=SYMBOL,
                timeframe=TIMEFRAME,
                market_meta=MARKET_META,
                hold_minutes=59,
            )

            state["last_candle_time"] = last_closed_time
            save_state(state, portfolio)

        time.sleep(CHECK_INTERVAL_SECONDS)

    except Exception as e:
        print(f"⚠️ Runtime error: {e}")
        time.sleep(10)
