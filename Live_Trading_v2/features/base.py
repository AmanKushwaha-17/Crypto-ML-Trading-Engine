from abc import ABC, abstractmethod


class FeatureBuilder(ABC):
    @abstractmethod
    def build(self, candles):
        pass
