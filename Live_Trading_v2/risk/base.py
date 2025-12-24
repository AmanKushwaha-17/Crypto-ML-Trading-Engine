from abc import ABC, abstractmethod


class RiskManager(ABC):
    @abstractmethod
    def allow_trade(self, signal, account_state):
        """Return (bool, position_size)"""
        pass
