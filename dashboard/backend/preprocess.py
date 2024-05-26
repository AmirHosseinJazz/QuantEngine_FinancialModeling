import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd


def get_univariate_kline(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    if interval == "1d":
        interval = "1D"
    elif interval == "1h":
        interval = "1H"
    elif interval == "1m":
        interval = "1M"
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()
    query = f"""
            SELECT "Opentime","{candle}" FROM "{symbol}"."kline_{interval}" WHERE "Opentime" >= {startDate} AND "Opentime" <= {endDate} ORDER BY "Opentime" ASC
            """
    cursor.execute(query)
    data = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    return pd.DataFrame(data, columns=cols)


def get_latest_kline(
    symbol="BTCUSDT",
    candle="Close",
    interval="1d",
    window=20,
):
    load_dotenv()
    host = os.getenv("POSTGRES_HOST")
    database = os.getenv("DATABASE")
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    if interval == "1d":
        interval = "1D"
    elif interval == "1h":
        interval = "1H"
    elif interval == "1m":
        interval = "1M"
    conn = psycopg2.connect(host=host, database=database, user=user, password=password)
    cursor = conn.cursor()
    query = f"""
            SELECT "Opentime","{candle}" FROM "{symbol}"."kline_{interval}" ORDER BY "Opentime" DESC limit {str(int(window)+1)}
            """
    cursor.execute(query)
    data = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    return pd.DataFrame(data, columns=cols)


def preprocess_univariate_candle(
    df,
    candle="Close",
):
    df["prev"] = df[candle].shift(1)
    df["log_return"] = df[candle].pct_change()
    df = df.dropna()
    df.set_index("Opentime", inplace=True)
    return df
    # df['volatility'] = df['log_return'].rolling(window=20).std()


def split_time_series(
    df, train_size=0.8, window_size=20, horizon=1, target="log_return"
):
    train_size = int(len(df) * train_size)
    train = df[:train_size]
    test = df[train_size:]
    X_train, y_train = [], []
    X_test, y_test = [], []
    for i in range(len(train) - window_size - horizon):
        X_train.append(train.iloc[i : i + window_size])
        y_train.append(train["log_return"].iloc[i + window_size + horizon - 1])
    for i in range(len(test) - window_size - horizon):
        X_test.append(test.iloc[i : i + window_size])
        y_test.append(test["log_return"].iloc[i + window_size + horizon - 1])
    return X_train, y_train, X_test, y_test
