from abc import ABC, abstractmethod


class MarketData(ABC):
    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str, limit: int):
        """Return OHLCV data"""
        pass
