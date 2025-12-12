"""
HYBRID BINANCE CLIENT
- Market data from MAINNET (real prices)
- Order execution on TESTNET (paper trading)
"""

import os
import logging
from typing import Dict, List, Optional
import pandas as pd
from binance.client import Client
from binance.enums import *

from dotenv import load_dotenv  

load_dotenv() 

logger = logging.getLogger(__name__)


class HybridBinanceClient:
    """
    Dual-client setup:
    - mainnet_client: reads real market data
    - testnet_client: executes paper trades
    """
    
    def __init__(self):
        # MAINNET: Real market data (no API keys needed for public data)
        self.mainnet_client = Client("", "")
        
        # TESTNET: Paper trading execution
        testnet_key = os.getenv('BINANCE_TESTNET_API_KEY')
        testnet_secret = os.getenv('BINANCE_TESTNET_API_SECRET')
        
        self.testnet_client = Client(
            testnet_key, 
            testnet_secret,
            testnet=True
        )
        
        self.symbol = 'ETHUSDT'
        self.leverage = 2
        
        # Set leverage on testnet
        try:
            self.testnet_client.futures_change_leverage(
                symbol=self.symbol,
                leverage=self.leverage
            )
            logger.info(f"Testnet leverage set to {self.leverage}x")
        except Exception as e:
            logger.error(f"Failed to set testnet leverage: {e}")
    
    # ========================================
    # MARKET DATA (from MAINNET - real prices)
    # ========================================
    
    def get_current_price(self) -> float:
        """Get real-time price from mainnet"""
        try:
            ticker = self.mainnet_client.futures_symbol_ticker(symbol=self.symbol)
            price = float(ticker['price'])
            logger.debug(f"Mainnet price: ${price:.2f}")
            return price
        except Exception as e:
            logger.error(f"Failed to get mainnet price: {e}")
            return 0.0
    
    def get_historical_data(self, interval: str = '15m', limit: int = 500) -> pd.DataFrame:
        """Fetch real historical candles from mainnet"""
        try:
            klines = self.mainnet_client.futures_klines(
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['funding_rate'] = 0.0  # Can fetch separately if needed
            
            logger.info(f"Fetched {len(df)} real candles from mainnet")
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch mainnet data: {e}")
            return pd.DataFrame()
    
    def get_funding_rate(self) -> float:
        """Get current funding rate from mainnet"""
        try:
            funding = self.mainnet_client.futures_funding_rate(symbol=self.symbol, limit=1)
            if funding:
                return float(funding[0]['fundingRate'])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get funding rate: {e}")
            return 0.0
    
    def get_mark_price(self) -> float:
        """Get mark price from mainnet (used for liquidation calc)"""
        try:
            mark = self.mainnet_client.futures_mark_price(symbol=self.symbol)
            return float(mark['markPrice'])
        except Exception as e:
            logger.error(f"Failed to get mark price: {e}")
            return self.get_current_price()
    
    # ========================================
    # ORDER EXECUTION (on TESTNET - paper money)
    # ========================================
    
    def get_testnet_balance(self) -> float:
        """Get testnet USDT balance"""
        try:
            balance = self.testnet_client.futures_account_balance()
            for asset in balance:
                if asset['asset'] == 'USDT':
                    return float(asset['balance'])
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get testnet balance: {e}")
            return 0.0
    
    def place_testnet_order(self, side: str, quantity: float) -> Optional[Dict]:
        """Place market order on testnet"""
        try:
            # Round quantity to appropriate precision
            quantity = round(quantity, 3)
            
            order = self.testnet_client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(f"Testnet order: {side} {quantity} {self.symbol}")
            return order
        except Exception as e:
            logger.error(f"Testnet order failed: {e}")
            return None
    
    def close_testnet_position(self) -> bool:
        """Close any open testnet position"""
        try:
            positions = self.testnet_client.futures_position_information(symbol=self.symbol)
            pos_amt = 0.0
            
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    pos_amt = float(pos['positionAmt'])
                    break
            
            if abs(pos_amt) < 0.001:
                logger.info("No testnet position to close")
                return True
            
            # Close position (reverse side)
            side = SIDE_SELL if pos_amt > 0 else SIDE_BUY
            quantity = abs(pos_amt)
            
            order = self.place_testnet_order(side, quantity)
            return order is not None
        
        except Exception as e:
            logger.error(f"Failed to close testnet position: {e}")
            return False
    
    def get_testnet_position(self) -> Optional[Dict]:
        """Get current testnet position info"""
        try:
            positions = self.testnet_client.futures_position_information(symbol=self.symbol)
            
            for pos in positions:
                if pos['symbol'] == self.symbol:
                    pos_amt = float(pos['positionAmt'])
                    if abs(pos_amt) > 0.001:
                        return {
                            'side': 'LONG' if pos_amt > 0 else 'SHORT',
                            'quantity': abs(pos_amt),
                            'entry_price': float(pos['entryPrice']),
                            'mark_price': float(pos['markPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'leverage': int(pos['leverage']),
                            'liquidation_price': float(pos['liquidationPrice'])
                        }
            return None
        except Exception as e:
            logger.error(f"Failed to get testnet position: {e}")
            return None
    
    # ========================================
    # LIQUIDATION CALCULATIONS
    # ========================================
    
    def calculate_liquidation_price(
        self, 
        entry_price: float, 
        leverage: int, 
        side: str
    ) -> float:
        """
        Calculate approximate liquidation price
        Formula (simplified):
        - LONG: liq_price = entry * (1 - 1/leverage)
        - SHORT: liq_price = entry * (1 + 1/leverage)
        """
        maintenance_margin_rate = 0.004  # 0.4% for ETHUSDT
        
        if side == 'LONG':
            liq_price = entry_price * (1 - (1/leverage) + maintenance_margin_rate)
        else:  # SHORT
            liq_price = entry_price * (1 + (1/leverage) - maintenance_margin_rate)
        
        return liq_price
    
    def check_liquidation_risk(
        self,
        entry_price: float,
        current_price: float,
        leverage: int,
        side: str
    ) -> Dict:
        """
        Check how close position is to liquidation
        Returns distance to liquidation in % and USD
        """
        liq_price = self.calculate_liquidation_price(entry_price, leverage, side)
        
        if side == 'LONG':
            distance_pct = ((current_price - liq_price) / current_price) * 100
        else:  # SHORT
            distance_pct = ((liq_price - current_price) / current_price) * 100
        
        distance_usd = abs(current_price - liq_price)
        
        return {
            'liquidation_price': liq_price,
            'distance_pct': distance_pct,
            'distance_usd': distance_usd,
            'at_risk': distance_pct < 10  # Alert if within 10%
        }