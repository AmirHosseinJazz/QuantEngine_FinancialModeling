import pandas as pd
import ta
from preprocess import get_backtrader_data


if __name__=="__main__":
    data =get_backtrader_data("BTCUSDT", "1685904000000", "1715904000000", "1d")
    print(data.head())
