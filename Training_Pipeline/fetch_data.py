import requests
import pandas as pd
from datetime import datetime, timezone
import time
import sys

SYMBOL = "ETHUSDT"              # ETH perpetual futures (USD-M)
OUT_CSV = "eth_15min_full.csv"  # output file name for ETH data
INTERVAL = "15m"                # Binance interval string


start_dt = datetime(2019, 9, 8, tzinfo=timezone.utc)
end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
start_ms = int(start_dt.timestamp() * 1000)

# 15-minute candle size in ms
CANDLE_MS = 15 * 60 * 1000

# ============================
# Fetch OHLCV (futures klines)
# ============================
print(f"Fetching 15m OHLCV for {SYMBOL} from fapi...")
all_data = []
url = "https://fapi.binance.com/fapi/v1/klines"

while start_ms < end_ms:
    try:
        r = requests.get(
            url,
            params={
                "symbol": SYMBOL,
                "interval": INTERVAL,
                "startTime": start_ms,
                "limit": 1000,
            },
            timeout=10,
        )

        if r.status_code != 200:
            print("HTTP error while fetching klines:", r.status_code, r.text)
            break

        d = r.json()

        # If dict, probably an error like {"code": -1121, "msg": "Invalid symbol."}
        if isinstance(d, dict):
            print("Klines API returned dict (probably error):", d)
            break

        if not d:
            print("No more kline data returned.")
            break

        all_data.extend(d)

        # move to the next 15m candle after the last one fetched
        start_ms = d[-1][0] + CANDLE_MS

        print(f"Fetched {len(all_data)} candles")
        time.sleep(0.5)

    except Exception as e:
        print(f"Error while fetching klines: {e}, retrying...")
        time.sleep(2)

# If no candles, STOP here
if not all_data:
    print(f"\n❌ No klines returned for symbol {SYMBOL} on fapi.")
    print("   Check that this is the correct futures symbol on Binance.")
    sys.exit(1)

# Build OHLCV DataFrame
df = pd.DataFrame(
    all_data,
    columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "num_trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ],
)
df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

# ====================
# Fetch funding rates
# ====================
print("\nFetching funding rate...")
funding_url = "https://fapi.binance.com/fapi/v1/fundingRate"
start_ms = int(start_dt.timestamp() * 1000)
funding_data = []

while start_ms < end_ms:
    try:
        r = requests.get(
            funding_url,
            params={
                "symbol": SYMBOL,
                "startTime": start_ms,
                "limit": 1000,
            },
            timeout=10,
        )

        if r.status_code != 200:
            print("HTTP error while fetching funding:", r.status_code, r.text)
            break

        d = r.json()

        if isinstance(d, dict):
            print("Funding API returned dict (probably error):", d)
            break

        if not d:
            print("No more funding data returned.")
            break

        funding_data.extend(d)
        start_ms = d[-1]["fundingTime"] + 1

        print(f"Fetched {len(funding_data)} funding points")
        time.sleep(0.5)

    except Exception as e:
        print(f"Error while fetching funding: {e}")
        time.sleep(2)
        break

df_funding = pd.DataFrame(funding_data)

if df_funding.empty:
    print("⚠️ No funding data downloaded. Skipping funding merge.")
    df["funding_rate"] = pd.NA
else:
    if "fundingTime" not in df_funding.columns or "fundingRate" not in df_funding.columns:
        print("⚠️ Unexpected funding response. Columns:", df_funding.columns)
        print(df_funding.head())
        sys.exit("Funding API did not return expected 'fundingTime'/'fundingRate' fields.")

    df_funding["fundingTime"] = pd.to_datetime(
        df_funding["fundingTime"], unit="ms", utc=True
    )
    df_funding = df_funding.rename(
        columns={"fundingTime": "time", "fundingRate": "funding_rate"}
    )

    df = df.merge(
        df_funding[["time", "funding_rate"]],
        left_on="open_time",
        right_on="time",
        how="left",
    ).drop("time", axis=1, errors="ignore")

    df["funding_rate"] = df["funding_rate"].ffill()

# optional: numeric conversion
numeric_cols = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "funding_rate",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved {OUT_CSV}: {len(df)} rows")
