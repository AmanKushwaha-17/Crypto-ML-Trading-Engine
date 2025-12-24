from strategy.base import Strategy


class DummyStrategy(Strategy):
    def generate_signal(self, market_data):
        print("[DummyStrategy] generate_signal called")
        return None
