from abc import ABC, abstractmethod


class ExecutionVenue(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float):
        pass


    @abstractmethod
    def close_quantity(self, symbol: str, quantity: float):
        pass


    @abstractmethod
    def get_position(self, symbol: str):
        pass
