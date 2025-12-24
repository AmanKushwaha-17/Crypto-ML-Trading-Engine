from datetime import datetime, timedelta

from core.state_manager import save_state
from core.logger import setup_logger


engine_logger = setup_logger("engine", "engine.log")


class Engine:
    def __init__(self, data, features, strategy, risk, execution, state, portfolio):
        self.data = data
        self.features = features
        self.strategy = strategy
        self.risk = risk
        self.execution = execution
        self.state = state
        self.portfolio = portfolio

        # ---- restore trade counter safely (restart-safe) ----
        existing_ids = [
            t["trade_id"] for t in self.state["open_trades"]
            if "trade_id" in t
        ]
        self._trade_counter = max(existing_ids) if existing_ids else 0

        engine_logger.info(
            f"ENGINE_INIT | equity={self.state['equity']:.2f} "
            f"open_trades={len(self.state['open_trades'])}"
        )

    def run_once(self, symbol, timeframe, market_meta, hold_minutes):

        now = datetime.utcnow()

        # =====================
        # TIME-BASED EXITS
        # =====================
        for trade in self.state["open_trades"][:]:
            if trade["status"] == "OPEN" and now >= trade["exit_time"]:

                engine_logger.info(
                    f"EXIT_TRIGGER | trade_id={trade['trade_id']} symbol={symbol}"
                )

                resp = self.execution.close_quantity(
                    symbol=symbol,
                    quantity=trade["size"],
                )

                if resp is None:
                    engine_logger.warning(
                        f"EXIT_FAIL | trade_id={trade['trade_id']} execution_failed"
                    )
                    continue

                fills = resp.get("fills", [])
                if fills:
                    exit_price = sum(
                        float(f["price"]) * float(f["qty"]) for f in fills
                    ) / sum(float(f["qty"]) for f in fills)
                else:
                    exit_price = float(resp["avgPrice"])

                # ---- execution truth ----
                realized_pnl = (
                    trade["direction"]
                    * (exit_price - trade["entry_price"])
                    * trade["size"]
                )

                # ---- execution vs signal slippage ----
                slippage = trade["direction"] * (
                    trade["entry_price"] - trade["signal_price"]
                )

                trade["exit_price"] = exit_price
                trade["exit_time_actual"] = now
                trade["realized_pnl"] = realized_pnl
                trade["slippage"] = slippage
                trade["status"] = "CLOSED"

                # ---- accounting ----
                self.state["equity"] += realized_pnl
                self.state["daily_pnl"] += realized_pnl
                self.portfolio["equity"] += realized_pnl
                self.portfolio["daily_pnl"] += realized_pnl

                engine_logger.info(
                    f"EXIT | trade_id={trade['trade_id']} symbol={symbol} "
                    f"pnl={realized_pnl:.2f} "
                    f"slippage={slippage:.4f} "
                    f"equity={self.state['equity']:.2f}"
                )

                # ---- persist immediately ----
                save_state(self.state, self.portfolio)

        # ---- remove CLOSED trades ----
        self.state["open_trades"] = [
            t for t in self.state["open_trades"]
            if t["status"] == "OPEN"
        ]

        engine_logger.info(
            f"CYCLE_START | symbol={symbol} open_trades={len(self.state['open_trades'])}"
        )

        # =====================
        # MARKET DATA
        # =====================
        candles = self.data.get_candles(symbol, timeframe, 250)
        if candles is None or len(candles) < 50:
            engine_logger.warning(
                f"NO_DATA | symbol={symbol} candles_invalid"
            )
            return

        features = self.features.build(candles)
        if features is None or len(features) < 50:
            engine_logger.warning(
                f"NO_FEATURES | symbol={symbol}"
            )
            return

        # =====================
        # SIGNAL
        # =====================
        signal = self.strategy.generate_signal(features, symbol=symbol)
        if signal is None:
            engine_logger.info(
                f"NO_SIGNAL | symbol={symbol}"
            )
            return

        # =====================
        # MARKET CONTEXT
        # =====================
        signal_price = float(candles.iloc[-2]["close"])
        atr = float(features.iloc[-2]["atr_14"])
        atr_ma = float(features["atr_14"].rolling(20).mean().iloc[-2])

        market_ctx = {
            "price": signal_price,   # reference (may slip)
            "atr": atr,
            "atr_ma": atr_ma,
            "step_size": market_meta["step_size"],
            "min_size": market_meta["min_size"],
        }

        # =====================
        # RISK CHECK
        # =====================
        allow, size, reason = self.risk.allow_trade(
            signal=signal,
            market_ctx=market_ctx,
            state=self.state,
            portfolio=self.portfolio,
        )

        if not allow:
            engine_logger.info(
                f"NO_TRADE | symbol={symbol} reason={reason} "
                f"confidence={signal.get('confidence', 0):.3f} "
                f"equity={self.state['equity']:.2f}"
            )
            return

        # =====================
        # ENTRY
        # =====================
        side = "BUY" if signal["direction"] == 1 else "SELL"

        self.execution.place_order(
            symbol=symbol,
            side=side,
            quantity=size,
        )

        self._trade_counter += 1

        self.state["open_trades"].append({
            "trade_id": self._trade_counter,

            # ---- strategy truth ----
            "signal_price": signal_price,

            # ---- execution truth (placeholder until entry fills wired) ----
            "entry_price": signal_price,

            "direction": signal["direction"],
            "size": size,

            "entry_time": now,
            "exit_time": now + timedelta(minutes=hold_minutes),

            "entry_atr": atr,
            "status": "OPEN",

            "exit_price": None,
            "exit_time_actual": None,
            "realized_pnl": None,
            "slippage": None,
        })

        engine_logger.info(
            f"ENTRY | trade_id={self._trade_counter} symbol={symbol} "
            f"side={side} size={size} "
            f"signal_price={signal_price:.2f} "
            f"atr={atr:.4f} "
            f"equity={self.state['equity']:.2f}"
        )

        # ---- persist immediately ----
        save_state(self.state, self.portfolio)
