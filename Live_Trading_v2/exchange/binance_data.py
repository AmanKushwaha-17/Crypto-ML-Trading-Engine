from exchange.base_data import MarketData
from binance.client import Client
import pandas as pd 

class BinanceMarketData(MarketData):
    def __init__(self):
        # Public endpoint

        self.client = Client("","")

    def get_candles(self, symbol, timeframe, limit):
        klines = self.client.futures_klines(
            symbol=symbol,
            interval=timeframe,
            limit = limit
        )

        df=pd.DataFrame(klines,columns=[
        "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"])
        
        return df