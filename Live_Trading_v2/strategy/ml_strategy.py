import joblib
import numpy as np
from strategy.base import Strategy

class MLStrategy(Strategy):
    def __init__(self,bundle_path:str):
        bundle = joblib.load(bundle_path)

        self.model = bundle['model']
        self.feature_cols =bundle['feature_cols']
        self.horizon = bundle.get('horizon',None)

    def generate_signal(self, features,symbol: str):

        if features is None or features.empty:
            return None
        
        latest = features.iloc[-2].copy()
        if "asset_id" in self.feature_cols:

            if symbol == "ETHUSDT":
                latest["asset_id"] = 1

            elif symbol == "BTCUSDT":
                latest["asset_id"] = 0

            else:
                return None  # unknown asset, fail safe

        # Select features in correct order
        X = latest[self.feature_cols]

        # Force numeric conversion (critical)
        X = X.astype(float).values.reshape(1, -1)

        # Now safe
        if not np.isfinite(X).all():
            return None

        
        pred = self.model.predict(X)[0]
        prob = float(self.model.predict_proba(X)[0].max())

        return {
            "direction":1  if pred == 1 else -1,
            "confidence": float(prob),
            "horizon":self.horizon
        }
        