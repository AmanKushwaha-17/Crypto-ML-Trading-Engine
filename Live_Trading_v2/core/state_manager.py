import json
import os
import tempfile
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

STATE_FILE = "logs/state.json"

# ======================
# SERIALIZATION HELPERS
# ======================

def _serialize(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


def _deserialize(obj):
    if isinstance(obj, str):
        try:
            return datetime.fromisoformat(obj)
        except ValueError:
            try:
                return date.fromisoformat(obj)
            except ValueError:
                return obj
    return obj


def _restore(value):
    if isinstance(value, dict):
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v) for v in value]
    return _deserialize(value)

# ======================
# SAVE STATE
# ======================

def save_state(state: dict, portfolio: dict):
    """
    Atomically save state + portfolio.
    Ensures JSON-safe, circular-free serialization.
    """
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                # ❗ drop anything non-serializable
                return str(obj)

        payload = {
            "state": sanitize(state),
            "portfolio": sanitize(portfolio),
            "saved_at": datetime.utcnow().isoformat(),
        }

        tmp_file = STATE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(payload, f, indent=2)

        os.replace(tmp_file, STATE_FILE)

        logger.info(
            f"STATE_SAVED | equity={state['equity']:.2f} "
            f"open_trades={len(state['open_trades'])}"
        )

    except Exception as e:
        logger.exception(f"Failed to save state: {e}")


# ======================
# LOAD STATE
# ======================

def load_state():
    """
    Load previous state if it exists.
    Returns (state, portfolio) or (None, None).
    """
    if not os.path.exists(STATE_FILE):
        logger.info("No previous state file found")
        return None, None

    try:
        with open(STATE_FILE, "r") as f:
            payload = json.load(f)

        state = _restore(payload["state"])
        # ---- normalize last_candle_time ----
        if "last_candle_time" in state and state["last_candle_time"] is not None:
            state["last_candle_time"] = int(state["last_candle_time"])

        portfolio = _restore(payload["portfolio"])

        logger.info(
            f"Loaded previous state (saved at {payload.get('saved_at', 'unknown')})"
        )

        return state, portfolio

    except Exception as e:
        logger.exception(f"Failed to load state: {e}")
        return None, None

# ======================
# INITIALIZATION
# ======================

def initialize_state(initial_capital: float):
    """
    Initialize fresh state for first run.
    """
    state = {
        "equity": initial_capital,
        "day_start_equity": initial_capital,
        "daily_pnl": 0.0,
        "current_day": date.today(),
        "trading_enabled": True,
        "open_trades": [],
    }

    portfolio = {
        "equity": initial_capital,
        "day_start_equity": initial_capital,
        "daily_pnl": 0.0,
        "current_day": date.today(),
        "trading_enabled": True,
    }

    logger.info("Initialized fresh trading state")
    return state, portfolio

# ======================
# PUBLIC ENTRY POINT
# ======================

def setup_state(initial_capital: float):
    """
    Load previous state or initialize fresh one.

    Rules:
    - Equity is NEVER reset
    - Daily PnL resets on new day
    - Open trades persist
    """
    state, portfolio = load_state()

    if state is None:
        logger.info("🆕 No previous state found, starting fresh")
        return initialize_state(initial_capital)

    today = date.today()

    if portfolio["current_day"] == today:
        logger.info(f"✅ Resuming trading session for {today}")
        logger.info(f"Portfolio equity: ${portfolio['equity']:.2f}")
        return state, portfolio

    # ---- New trading day: reset daily metrics only ----
    logger.info(
        f"📅 New trading day detected (last run: {portfolio['current_day']})"
    )

    state["daily_pnl"] = 0.0
    state["day_start_equity"] = state["equity"]
    state["current_day"] = today
    state["trading_enabled"] = True

    portfolio["daily_pnl"] = 0.0
    portfolio["day_start_equity"] = portfolio["equity"]
    portfolio["current_day"] = today
    portfolio["trading_enabled"] = True

    return state, portfolio
