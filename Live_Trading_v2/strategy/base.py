from abc import ABC, abstractmethod


class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data):
        """Return signal or None"""
        pass
