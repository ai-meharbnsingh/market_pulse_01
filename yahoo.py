import yfinance as yf
import pandas as pd

def get_latest_prices(symbols):
    results = []
    for sym in symbols:
        try:
            stock = yf.Ticker(sym)
            hist = stock.history(period="1d")
            if not hist.empty:
                latest_close = hist["Close"].iloc[-1]
                results.append({"Symbol": sym, "Close": latest_close})
            else:
                results.append({"Symbol": sym, "Close": None})
        except Exception as e:
            results.append({"Symbol": sym, "Close": None, "Error": str(e)})

    return pd.DataFrame(results)

if __name__ == "__main__":
    symbols = ["AAPL", "MSFT", "GOOG", "RELIANCE.NS", "TCS.NS"]
    df = get_latest_prices(symbols)
    print(df)
