from features.base import FeatureBuilder
from features.crypto import build_crypto_features


class BinanceCryptoFeatures(FeatureBuilder):
    def build(self, candles):
        return build_crypto_features(candles)
