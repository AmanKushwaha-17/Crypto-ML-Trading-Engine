from binance.client import Client
from exchange.base_execution import ExecutionVenue


class BinanceTestnetExecution(ExecutionVenue):
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        self.client.FUTURES_URL = "https://testnet.binancefuture.com"

    def place_order(self, symbol: str, side: str, quantity: float):
        print(f"[EXECUTION] Placing {side} {quantity} {symbol} (TESTNET)")

        return self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
        )

    def close_quantity(self, symbol: str, quantity: float):
        position_info = self.client.futures_position_information(symbol=symbol)
        pos = float(position_info[0]["positionAmt"])

        if pos == 0:
            return None

        side = "SELL" if pos > 0 else "BUY"

        return self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=abs(quantity),
            reduceOnly=True,
        )


    def get_position(self, symbol: str):
        return self.client.futures_position_information(symbol=symbol)
