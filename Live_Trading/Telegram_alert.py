import os
import sys
import logging
import time
from datetime import datetime
import requests
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Telegram Bot configuration"""

    # Telegram Bot (for alerts)
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    # Log path (added to fix the error)
    LOG_PATH = "trading_bot.log"


# ============================================================
# Ensure UTF-8 console on Windows and reconfigure stdout/stderr
# ============================================================
if os.name == "nt":
    # set console code page to UTF-8 (works on Windows terminals that support it)
    try:
        os.system("chcp 65001 > nul")
    except Exception:
        pass

# reconfigure Python IO streams to UTF-8 if available (Py3.7+)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # ignore if not supported in this environment
    pass


# ============================================================
# LOGGING SETUP (FileHandler uses UTF-8)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Send alerts via Telegram"""

    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.enabled = bool(
            self.token and self.chat_id and self.token != "your_telegram_bot_token"
        )
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram with retries and timeout"""
        if not self.enabled:
            logger.warning("Telegram not configured")
            return False

        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": parse_mode}
        timeout = 20  # seconds (increase if your network is slow)
        max_retries = 2

        for attempt in range(1, max_retries + 2):
            try:
                resp = requests.post(self.api_url, data=payload, timeout=timeout)
                resp.raise_for_status()
                # Log success (short ASCII-safe message)
                logger.info("Telegram alert sent")
                return True
            except requests.RequestException as e:
                # Log a concise warning (no emoji) to avoid encoding errors on some consoles
                logger.warning(
                    f"Telegram send failed (attempt {attempt}): {e.__class__.__name__}: {str(e)}"
                )
                if attempt <= max_retries:
                    # exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    logger.error("Telegram unreachable after retries")
                    return False
            except Exception as e:
                logger.exception("Unexpected error in Telegram send")
                return False

        return False

    def send_trade_alert(self, trade_type: str, signal: Dict):
        """Send formatted trade alert (emoji preserved for Telegram)"""
        emoji = "🟢" if trade_type == "ENTRY" else "🔴"
        direction = "LONG" if signal["direction"] == 1 else "SHORT"
        # Position size may not exist in the signal dict — safe fallback
        position_size = signal.get("position_size", 0.0)

        message = f"""
{emoji} <b>{trade_type}: {direction}</b>

💰 Entry: ${signal['entry_price']:.2f}
🎯 Confidence: {signal['confidence']:.1%}
⚡ Leverage: {signal['leverage']}x
💵 Position Size: ${position_size:.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        # Send to Telegram; send_message logs short status messages
        self.send_message(message)

    def send_error_alert(self, error: str):
        """Send error alert (keeps emoji to make message noticeable on Telegram)"""
        message = f"🚨 <b>BOT ERROR</b>\n\n{error}"
        self.send_message(message)
