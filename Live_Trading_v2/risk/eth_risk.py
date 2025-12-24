import math
from datetime import date
from risk.base import RiskManager


class EthRiskManager(RiskManager):
    def __init__(
        self,
        leverage: int = 5,
        risk_per_trade: float = 0.06,
        portfolio_dd_limit: float = -0.30,
        min_confidence: float = 0.60,
    ):
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.portfolio_dd_limit = portfolio_dd_limit
        self.min_confidence = min_confidence

    # --------------------------------------------------
    # CORE ENTRY POINT
    # --------------------------------------------------
    def allow_trade(self, signal, market_ctx, state, portfolio):
        """
        Returns:
            (allow: bool, size: float, reason: str)
        """

        # ---------- Signal gate ----------
        if signal is None:
            return False, 0.0, "NO_SIGNAL"

        # ---------- Confidence gate ----------
        if signal["confidence"] < self.min_confidence:
            return False, 0.0, "LOW_CONFIDENCE"

        # ---------- Daily reset ----------
        today = date.today()

        if today != state["current_day"]:
            state["daily_pnl"] = 0.0
            state["day_start_equity"] = state["equity"]
            state["current_day"] = today
            state["trading_enabled"] = True

        if today != portfolio["current_day"]:
            portfolio["daily_pnl"] = 0.0
            portfolio["day_start_equity"] = portfolio["equity"]
            portfolio["current_day"] = today
            portfolio["trading_enabled"] = True

        # ---------- Portfolio drawdown ----------
        if portfolio["daily_pnl"] <= self.portfolio_dd_limit * portfolio["day_start_equity"]:
            portfolio["trading_enabled"] = False
            return False, 0.0, "PORTFOLIO_DD_LIMIT"

        if not portfolio["trading_enabled"]:
            return False, 0.0, "TRADING_DISABLED"

        # ---------- ATR regime filter ----------
        atr = market_ctx.get("atr")
        atr_ma = market_ctx.get("atr_ma")

        if atr is None or atr_ma is None:
            return False, 0.0, "ATR_MISSING"

        if atr <= atr_ma:
            return False, 0.0, "LOW_VOLATILITY_REGIME"

        # ---------- Pyramiding rule ----------
        open_trades = state["open_trades"]

        if open_trades:
            last_trade = open_trades[-1]

            # Allow add-on only if ATR is expanding
            if atr <= last_trade["entry_atr"]:
                return False, 0.0, "ATR_NOT_EXPANDING"

        # ---------- Position sizing ----------
        size = self._compute_position_size(
            equity=state["equity"],
            price=market_ctx["price"],
            atr=atr,
            confidence=signal["confidence"],
            step_size=market_ctx["step_size"],
            min_size=market_ctx["min_size"],
        )

        if size <= 0:
            return False, 0.0, "SIZE_TOO_SMALL"

        return True, size, "ALLOW"

    # --------------------------------------------------
    # POSITION SIZING
    # --------------------------------------------------
    def _compute_position_size(
        self,
        equity,
        price,
        atr,
        confidence,
        step_size,
        min_size,
    ):
        if atr is None or atr <= 0 or price is None:
            return 0.0

        # ---- Confidence scaling ----
        if confidence >= 0.70:
            risk_mult = 2.0
        elif confidence >= 0.65:
            risk_mult = 1.5
        else:
            risk_mult = 1.0

        risk_amount = equity * self.risk_per_trade * risk_mult
        raw_size = risk_amount / atr

        # ---- Step-size rounding ----
        size = math.floor(raw_size / step_size) * step_size

        if size < min_size:
            return 0.0

        return size
