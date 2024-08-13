import psycopg2
from dotenv import load_dotenv
import os
import pandas as pd


def get_backtrader_data(
    symbol="BTCUSDT",
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
            SELECT "Opentime","Open","High","Low","Close","Volume" FROM "{symbol}"."kline_{interval}" WHERE "Opentime" >= {startDate} AND "Opentime" <= {endDate} ORDER BY "Opentime" ASC
            """
    cursor.execute(query)
    data = cursor.fetchall()
    cols = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()
    DF = pd.DataFrame(data, columns=cols)
    DF["Date"] = DF["Opentime"].apply(lambda x: pd.to_datetime(x, unit="ms"))
    DF["Low"] = DF["Low"] / 1000
    DF["High"] = DF["High"] / 1000
    DF["Open"] = DF["Open"] / 1000
    DF["Close"] = DF["Close"] / 1000

    DF.set_index("Date", inplace=True)
    return DF


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


def split_time_series(
    df,
    train_size=0.8,
    window_size=20,
    horizon=1,
    target="log_return",
    mode="univariate",
):
    train_size = int(len(df) * train_size)
    train = df[:train_size]
    test = df[train_size:]
    X_train, y_train = [], []
    X_test, y_test = [], []
    train = train.reset_index()
    train_time = train["Opentime"].values
    test = test.reset_index()
    test_time = test["Opentime"].values
    train = train.drop(columns=["Opentime"])
    test = test.drop(columns=["Opentime"])
    if mode == "univariate":
        for i in range(len(train) - window_size - horizon):
            X_train.append(train[target].iloc[i : i + window_size])
            y_train.append(
                train[target].iloc[i + window_size : i + window_size + horizon]
            )
        for i in range(len(test) - window_size - horizon):
            X_test.append(test[target].iloc[i : i + window_size])
            y_test.append(
                test[target].iloc[i + window_size : i + window_size + horizon]
            )
        return X_train, y_train, X_test, y_test, train_time, test_time
