"""
HYBRID FORWARD TESTING BOT - FIXED VERSION
- Real market data from Binance mainnet
- Paper trading on Binance testnet
- Full position tracking, PnL, liquidation monitoring
"""

import os
import time
import json
import pickle
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance.enums import *
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


# Import your modules
from Hybrid_Binance_Client import HybridBinanceClient
from Telegram_alert import TelegramBot
from feature_builder import build_crypto_features
from analytic import PerformanceAnalytics



# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Bot configuration"""
    
    # Trading Parameters
    SYMBOL = 'ETHUSDT'
    INTERVAL = '15m'
    POSITION_SIZE_PCT = 0.06  # 6% of capital (will be multiplied by leverage)
    LEVERAGE = 2
    MIN_CONFIDENCE = 0.60
    HOLD_BARS = 4  # 1 hour = 4 × 15min
    
    # Risk Management
    MAX_DRAWDOWN_PCT = 0.30  # 30% account drawdown - bot stops if hit
    DAILY_LOSS_LIMIT_PCT = 0.1  # 10% daily loss limit
    LIQUIDATION_ALERT_PCT = 15  # Alert if within 15% of liquidation
    MIN_BALANCE = 100.0  # Minimum testnet balance to start
    
    # Paths
    MODEL_PATH = 'trained_model.pkl'
    LOG_PATH = 'hybrid_bot.log'
    TRADES_DB = 'hybrid_trades.json'
    STATE_DB = 'hybrid_state.json'
    
    # Testnet Mode
    TESTNET = True  # Set to False for real trading


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logging.FileHandler(Config.LOG_PATH, encoding="utf-8")


# ============================================================
# STRATEGY
# ============================================================

class TradingStrategy:
    """ML-based trading strategy - FIXED VERSION"""
    
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.feature_cols = [
            'volatility_10', 'volatility_24', 'volatility_168', 'vol_ratio',
            'volume_ratio', 'volume_trend', 'buy_pressure', 'buy_strength',
            'roc_4', 'roc_12', 'roc_24',
            'price_to_sma24', 'sma_cross', 'hl_range', 'hl_range_ma', 'trend_regime',
            'day_of_week', 'hour', 'month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'rsi_14', 'rsi_7',
            'macd', 'macd_signal', 'macd_hist',
            'bb_width', 'bb_position',
            'stoch_k', 'stoch_d',
            'atr_14', 'ema_fast', 'ema_slow',
            'obv_ema', 'atr_pct', 'mfi'
        ] + [f'return_lag_{i}' for i in [1, 2, 3, 4, 6, 8, 12, 24]]
    
    def load_model(self, path: str):
        """Load trained model"""
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded: {path}")
            return model
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal from market data"""
        try:
            # Build features
            df = build_crypto_features(df)
            
            # FIXED: Use second-to-last row to avoid look-ahead bias
            # The last candle might be incomplete
            if len(df) < 2:
                logger.warning("Not enough data for prediction")
                return None
            
            latest = df.iloc[-2]  # Use complete bar
            current_price = df['close'].iloc[-1]  # Current price for entry
            
            # Check for missing features
            missing = [col for col in self.feature_cols if col not in df.columns]
            if missing:
                logger.warning(f"Missing features: {missing[:5]}...")
                return None
            
            # Extract features
            X = latest[self.feature_cols].values.reshape(1, -1)
            
            # CRITICAL FIX: Convert to float64 BEFORE checking for NaN
            # pandas DataFrames can have dtype='object' even with numeric values
            try:
                X = X.astype(np.float64)
            except (ValueError, TypeError) as e:
                logger.error(f"Cannot convert features to numeric: {e}")
                logger.error(f"Problematic feature types: {latest[self.feature_cols].dtypes.to_dict()}")
                return None
            
            # NOW check for NaN values (works because X is numeric)
            if np.isnan(X).any():
                nan_indices = np.where(np.isnan(X[0]))[0]
                nan_features = [self.feature_cols[i] for i in nan_indices]
                logger.warning(f"NaN values in features: {nan_features[:5]}")
                return None
            
            # Check for infinite values
            if np.isinf(X).any():
                inf_indices = np.where(np.isinf(X[0]))[0]
                inf_features = [self.feature_cols[i] for i in inf_indices]
                logger.warning(f"Infinite values in features: {inf_features[:5]}")
                return None
            
            # Predict
            prediction = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            confidence = proba.max()
            
            # Check confidence
            if confidence < Config.MIN_CONFIDENCE:
                logger.info(f"Low confidence: {confidence:.1%} < {Config.MIN_CONFIDENCE:.1%}")
                return None
            
            # Log successful signal
            direction = "LONG" if prediction == 1 else "SHORT"
            logger.info(f"✅ Signal generated: {direction} @ {confidence:.1%}")
            
            return {
                'timestamp': latest['open_time'],
                'prediction': int(prediction),
                'direction': 1 if prediction == 1 else -1,
                'confidence': confidence,
                'leverage': Config.LEVERAGE,
                'entry_price': current_price
            }
        
        except Exception as e:
            logger.error(f"Signal generation failed: {e}", exc_info=True)
            return None


# ============================================================
# POSITION MANAGER
# ============================================================

class PositionManager:
    """Track positions and calculate PnL"""
    
    def __init__(self):
        self.active_position = None
    
    def open_position(self, signal: Dict, balance: float, actual_entry: float):
        """Record new position"""
        # FIXED: Apply leverage to position size
        position_value = balance * Config.POSITION_SIZE_PCT * Config.LEVERAGE
        quantity = position_value / actual_entry
        
        self.active_position = {
            'direction': signal['direction'],
            'side': 'LONG' if signal['direction'] == 1 else 'SHORT',
            'entry_price': actual_entry,
            'entry_time': datetime.now(),
            'exit_time': datetime.now() + timedelta(hours=1),
            'confidence': signal['confidence'],
            'leverage': signal['leverage'],
            'position_value': position_value,
            'quantity': quantity,
            'initial_balance': balance
        }
        
        logger.info(f"Position opened: {self.active_position['side']} @ ${actual_entry:.2f}")
        logger.info(f"Position value: ${position_value:.2f} (with {Config.LEVERAGE}x leverage)")
    
    def get_testnet_pnl(self, client) -> tuple[float, float]:
        """Get real PnL from testnet (includes fees, funding, slippage)"""
        pos = client.get_testnet_position()
        
        if not pos or not self.active_position:
            return 0.0, 0.0
        
        # Real PnL from testnet (already includes all costs)
        pnl_usd = pos['unrealized_pnl']
        
        # Calculate percentage based on initial position value
        pnl_pct = (pnl_usd / self.active_position['position_value']) * 100
        
        return pnl_usd, pnl_pct
    
    def should_exit(self, client) -> tuple[bool, str]:
        """
        Check exit conditions
        Returns: (should_exit, reason)
        
        Only exits on:
        1. Time (1 hour hold)
        2. Emergency at -90% (near liquidation)
        """
        if not self.active_position:
            return False, ""
        
        # Time-based exit (1 hour)
        if datetime.now() >= self.active_position['exit_time']:
            return True, "TIME"
        
        # Emergency exit near liquidation
        pnl_usd, pnl_pct = self.get_testnet_pnl(client)
        if pnl_pct < -90:
            return True, "EMERGENCY_LIQUIDATION"
        
        return False, ""
    
    def close_position(self, client, reason: str) -> Dict:
        """Close position using real testnet PnL"""
        if not self.active_position:
            return {}
        
        pos = self.active_position
        
        # Get real PnL from testnet
        pnl_usd, pnl_pct = self.get_testnet_pnl(client)
        
        # Get actual exit price from testnet
        testnet_pos = client.get_testnet_position()
        exit_price = testnet_pos['mark_price'] if testnet_pos else 0.0
        
        result = {
            **pos,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'exit_reason': reason,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd,  # Real PnL with fees included
            'duration_minutes': (datetime.now() - pos['entry_time']).total_seconds() / 60,
            'closed': True
        }
        
        logger.info(f"Position closed: {reason} | Real PnL ${pnl_usd:.2f} ({pnl_pct:+.2f}%)")
        self.active_position = None
        return result


# ============================================================
# MAIN BOT
# ============================================================

class HybridTradingBot:
    """Main bot orchestrator"""
    
    def __init__(self):
        logger.info("Initializing Hybrid Trading Bot...")
        
        # Initialize client
        self.client = HybridBinanceClient()
        
        # FIXED: Validate testnet balance before proceeding
        self.starting_balance = self.client.get_testnet_balance()
        
        if self.starting_balance < Config.MIN_BALANCE:
            error_msg = (
                f"Insufficient testnet balance: ${self.starting_balance:.2f}\n"
                f"Minimum required: ${Config.MIN_BALANCE:.2f}\n\n"
                f"Please fund your testnet account at:\n"
                f"https://testnet.binancefuture.com"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize components
        self.strategy = TradingStrategy(Config.MODEL_PATH)
        self.position_mgr = PositionManager()
        self.telegram = TelegramBot()
        self.analytics = PerformanceAnalytics()
        
        self.trades_history = self.load_trades()
        self.bot_state = self.load_state()

        # Load historical trades into analytics
        for trade in self.trades_history:
            self.analytics.add_trade(trade)
        
        # FIXED: Initialize peak balance (will never be zero now)
        self.peak_balance = self.starting_balance
        
        logger.info("="*60)
        logger.info("HYBRID TRADING BOT INITIALIZED")
        logger.info(f"Mode: TESTNET (Paper Trading)")
        logger.info(f"Market Data: MAINNET (Real Prices)")
        logger.info(f"Testnet Balance: ${self.starting_balance:.2f}")
        logger.info(f"Max Drawdown: {Config.MAX_DRAWDOWN_PCT*100}%")
        logger.info(f"Position Size: {Config.POSITION_SIZE_PCT*100}% × {Config.LEVERAGE}x = {Config.POSITION_SIZE_PCT*Config.LEVERAGE*100}%")
        logger.info("="*60)
        
        self.telegram.send_message(
            f"🤖 <b>Hybrid Bot Started</b>\n\n"
            f"💰 Balance: ${self.starting_balance:.2f}\n"
            f"📊 Symbol: {Config.SYMBOL}\n"
            f"⚡ Leverage: {Config.LEVERAGE}x\n"
            f"🎯 Position Size: {Config.POSITION_SIZE_PCT*100}% × {Config.LEVERAGE}x = {Config.POSITION_SIZE_PCT*Config.LEVERAGE*100}%\n"
            f"⏱ Hold Time: 1 hour\n"
            f"🛡️ Max Drawdown: {Config.MAX_DRAWDOWN_PCT*100}%\n\n"
            f"<i>Paper trading with real market data</i>"
        )
    
    def load_trades(self) -> List[Dict]:
        """Load trade history"""
        try:
            with open(Config.TRADES_DB, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_trades(self):
        """Save trade history"""
        with open(Config.TRADES_DB, 'w') as f:
            json.dump(self.trades_history, f, indent=2, default=str)
    
    def load_state(self) -> Dict:
        """Load bot state"""
        try:
            with open(Config.STATE_DB, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'daily_pnl': 0.0,
                'last_reset_date': datetime.now().date().isoformat(),
                'is_active': True,
                'total_trades': 0,
                'winning_trades': 0
            }
    
    def save_state(self):
        """Save bot state"""
        with open(Config.STATE_DB, 'w') as f:
            json.dump(self.bot_state, f, indent=2)
    
    def check_risk_limits(self, balance: float) -> bool:
        """Verify risk limits"""
        # FIXED: Handle zero peak balance
        if self.peak_balance <= 0:
            logger.error("Peak balance is zero or negative - cannot calculate drawdown")
            self.telegram.send_error_alert("🚨 Invalid peak balance - bot stopped")
            return False
        
        # Account drawdown check (30% limit)
        drawdown = (self.peak_balance - balance) / self.peak_balance
        
        if drawdown > Config.MAX_DRAWDOWN_PCT:
            logger.warning(f"Max drawdown hit: {drawdown:.2%}")
            self.telegram.send_error_alert(
                f"🛑 <b>BOT STOPPED - MAX DRAWDOWN</b>\n\n"
                f"Drawdown: {drawdown:.2%}\n"
                f"Peak: ${self.peak_balance:.2f}\n"
                f"Current: ${balance:.2f}\n\n"
                f"<i>Bot paused to protect capital</i>"
            )
            return False
        
        # Update peak
        if balance > self.peak_balance:
            self.peak_balance = balance
            logger.info(f"New peak balance: ${self.peak_balance:.2f}")
        
        # Daily loss limit
        if self.starting_balance > 0:
            daily_pnl_pct = self.bot_state['daily_pnl'] / self.starting_balance
            if daily_pnl_pct < -Config.DAILY_LOSS_LIMIT_PCT:
                logger.warning(f"Daily loss limit hit: {daily_pnl_pct:.2%}")
                self.telegram.send_error_alert(
                    f"⚠️ Daily loss {daily_pnl_pct:.2%} exceeded!\n"
                    f"Bot paused until tomorrow."
                )
                return False
        
        # Reset daily PnL
        today = datetime.now().date().isoformat()
        if today != self.bot_state['last_reset_date']:
            self.bot_state['daily_pnl'] = 0.0
            self.bot_state['last_reset_date'] = today
            self.save_state()
            logger.info("Daily PnL reset")
        
        return True
    
    def monitor_liquidation_risk(self):
        """Check liquidation proximity"""
        pos = self.client.get_testnet_position()
        if not pos:
            return
        
        current_price = self.client.get_current_price()
        risk = self.client.check_liquidation_risk(
            pos['entry_price'],
            current_price,
            pos['leverage'],
            pos['side']
        )
        
        if risk['at_risk']:
            logger.warning(f"Liquidation risk: {risk['distance_pct']:.1f}% away")
            self.telegram.send_message(
                f"⚠️ <b>LIQUIDATION WARNING</b>\n\n"
                f"Position: {pos['side']}\n"
                f"Entry: ${pos['entry_price']:.2f}\n"
                f"Current: ${current_price:.2f}\n"
                f"Liq Price: ${risk['liquidation_price']:.2f}\n"
                f"Distance: {risk['distance_pct']:.1f}%\n\n"
                f"<i>Consider closing early</i>"
            )
    
    def run_iteration(self):
        """Single bot iteration"""
        try:
            logger.info("="*60)
            logger.info(f"Iteration at {datetime.now()}")
            
            # Get testnet balance
            balance = self.client.get_testnet_balance()
            logger.info(f"Testnet balance: ${balance:.2f}")
            
            # FIXED: Handle invalid balance
            if balance <= 0:
                logger.error("Invalid testnet balance - skipping iteration")
                return
            
            # Risk checks (30% account drawdown limit)
            if not self.check_risk_limits(balance):
                return
            
            # Monitor existing position
            if self.position_mgr.active_position:
                self.monitor_liquidation_risk()
                
                # Check exit conditions (only time-based + emergency)
                should_exit, reason = self.position_mgr.should_exit(self.client)
                
                if should_exit:
                    logger.info(f"Closing position: {reason}")
                    
                    # Close on testnet
                    if self.client.close_testnet_position():
                        # Record trade with real PnL
                        trade = self.position_mgr.close_position(self.client, reason)
                        
                        if trade:  # Only save if trade dict is not empty
                            self.trades_history.append(trade)
                            self.save_trades()
                            
                            # Add to analytics
                            self.analytics.add_trade(trade)
                            
                            # Update state
                            self.bot_state['daily_pnl'] += trade['pnl_usd']
                            self.bot_state['total_trades'] += 1
                            if trade['pnl_usd'] > 0:
                                self.bot_state['winning_trades'] += 1
                            self.save_state()
                            
                            # Get updated metrics
                            metrics = self.analytics.calculate_metrics()
                            balance = self.client.get_testnet_balance()
                            
                            # Enhanced Telegram alert with key metrics
                            self.telegram.send_message(
                                f"{'🟢' if trade['pnl_usd'] > 0 else '🔴'} <b>EXIT: {trade['side']}</b>\n\n"
                                f"Entry: ${trade['entry_price']:.2f}\n"
                                f"Exit: ${trade['exit_price']:.2f}\n"
                                f"PnL: ${trade['pnl_usd']:.2f} ({trade['pnl_pct']:+.2f}%)\n"
                                f"Reason: {reason}\n"
                                f"Duration: {trade['duration_minutes']:.0f}m\n\n"
                                f"📊 <b>Performance Metrics:</b>\n"
                                f"Balance: ${balance:.2f}\n"
                                f"Total Trades: {metrics['total_trades']}\n"
                                f"Win Rate: {metrics['win_rate']:.1f}%\n"
                                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                                f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                                f"Expectancy: ${metrics['expectancy']:.2f}\n"
                                f"Today PnL: ${self.bot_state['daily_pnl']:+.2f}"
                            )
                            
                            # Print summary every 10 trades
                            if metrics['total_trades'] % 10 == 0:
                                self.analytics.print_summary()
                else:
                    # Log current PnL (using real testnet data)
                    pnl_usd, pnl_pct = self.position_mgr.get_testnet_pnl(self.client)
                    logger.info(f"Position open | Real PnL: ${pnl_usd:.2f} ({pnl_pct:+.2f}%)")
                
                return
            
            # Fetch real market data
            logger.info("Fetching mainnet market data...")
            df = self.client.get_historical_data(Config.INTERVAL, limit=500)
            
            if df.empty:
                logger.error("No market data")
                return
            
            # Generate signal
            logger.info("Generating signal...")
            signal = self.strategy.generate_signal(df)
            
            if signal is None:
                logger.info("No signal")
                return
            
            # Open position
            logger.info(f"Signal: {signal['direction']} @ {signal['confidence']:.1%}")
            
            # Get actual entry price
            entry_price = self.client.get_current_price()
            
            # FIXED: Calculate quantity with leverage applied
            position_value = balance * Config.POSITION_SIZE_PCT * Config.LEVERAGE
            quantity = position_value / entry_price
            quantity = round(quantity, 3)
            
            logger.info(f"Position value: ${position_value:.2f} ({Config.LEVERAGE}x leverage)")
            logger.info(f"Quantity: {quantity} {Config.SYMBOL}")
            
            # Place order
            side = SIDE_BUY if signal['direction'] == 1 else SIDE_SELL
            order = self.client.place_testnet_order(side, quantity)
            
            if order:
                self.position_mgr.open_position(signal, balance, entry_price)
                
                # Calculate liquidation
                liq_price = self.client.calculate_liquidation_price(
                    entry_price,
                    signal['leverage'],
                    'LONG' if signal['direction'] == 1 else 'SHORT'
                )
                
                self.telegram.send_message(
                    f"🟢 <b>ENTRY: {'LONG' if signal['direction'] == 1 else 'SHORT'}</b>\n\n"
                    f"💰 Entry: ${entry_price:.2f}\n"
                    f"🎯 Confidence: {signal['confidence']:.1%}\n"
                    f"⚡ Leverage: {signal['leverage']}x\n"
                    f"💵 Size: ${position_value:.2f}\n"
                    f"📦 Qty: {quantity}\n"
                    f"⚠️ Liq: ${liq_price:.2f}\n\n"
                    f"<i>Hold until {self.position_mgr.active_position['exit_time'].strftime('%H:%M')}</i>"
                )
            else:
                logger.error("Order failed")
        
        except Exception as e:
            logger.error(f"Iteration error: {e}", exc_info=True)
            self.telegram.send_error_alert(f"Error: {str(e)}")
    
    def run(self):
        """Main loop"""
        logger.info("Starting main loop...")
        
        try:
            while True:
                self.run_iteration()
                
                # Sleep until next candle
                now = datetime.now()
                next_run = now.replace(
                    minute=(now.minute // 15 + 1) * 15 % 60,
                    second=5,
                    microsecond=0
                )
                if next_run < now:
                    next_run += timedelta(hours=1)
                
                sleep_sec = (next_run - now).total_seconds()
                logger.info(f"Next run: {next_run} ({sleep_sec:.0f}s)")
                time.sleep(sleep_sec)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.telegram.send_message("⏸️ <b>Bot Stopped</b>")
        
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.telegram.send_error_alert(f"🚨 FATAL: {str(e)}")


def print_performance_stats():
    """Print current performance statistics"""
    try:
        with open(Config.TRADES_DB, 'r') as f:
            trades = json.load(f)
        
        analytics = PerformanceAnalytics()
        for trade in trades:
            analytics.add_trade(trade)
        
        analytics.print_summary()
        
        # Export to file
        metrics = analytics.calculate_metrics()
        with open('performance_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print("📄 Metrics exported to: performance_metrics.json")
        
    except FileNotFoundError:
        print("❌ No trades found. Run the bot first!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔═══════════════════════════════════════════╗
    ║   HYBRID FORWARD TESTING BOT - FIXED     ║
    ║   Real Market Data + Paper Trading        ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # Check for stats command
    if len(sys.argv) > 1 and sys.argv[1] == 'stats':
        print_performance_stats()
        exit(0)
    
    # Check testnet credentials
    if not os.getenv('BINANCE_TESTNET_API_KEY'):
        print("⚠️  Set testnet credentials:")
        print("export BINANCE_TESTNET_API_KEY='...'")
        print("export BINANCE_TESTNET_API_SECRET='...'")
        print("\nGet testnet keys: https://testnet.binancefuture.com")
        print("\n💡 To view stats: python Hybrid_Trading_Bot.py stats")
        exit(1)
    
    try:
        bot = HybridTradingBot()
        bot.run()
    except ValueError as e:
        print(f"\n❌ Initialization Error: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)