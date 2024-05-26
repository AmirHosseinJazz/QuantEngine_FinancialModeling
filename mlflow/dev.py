import preprocess
import pandas as pd
import benchmark_univariate


if __name__ == "__main__":
    # df=(preprocess.preprocess_univariate_candle())
    # X_train, y_train, X_test, y_test = preprocess.split_time_series(df)
    # benchmark_univariate.naive_forecast()
    # benchmark_univariate.moving_average()
    # for _window in range(5,480,5):
    #     benchmark_univariate.LinearReg(
    #         symbol="BTCUSDT",
    #         candle="Close",
    #         startDate="1600000000000",
    #         endDate="1715904000000",
    #         interval="1m",
    #         train_size=0.8,
    #         window_size=_window,
    #         horizon=1,
    #     )
    # for _window in range(100, 480, 50):
    #     benchmark_univariate.RF(
    #         symbol="BTCUSDT",
    #         candle="Close",
    #         startDate="1600000000000",
    #         endDate="1715904000000",
    #         interval="1m",
    #         train_size=0.8,
    #         window_size=_window,
    #         horizon=1,
    #     )
    # for _window in range(60, 480, 60):
    #     benchmark_univariate.DT(
    #         symbol="BTCUSDT",
    #         candle="Close",
    #         startDate="1600000000000",
    #         endDate="1715904000000",
    #         interval="1m",
    #         train_size=0.8,
    #         window_size=_window,
    #         horizon=1,
    # )
    # for _window in range(120, 480, 120):
    #     print("SVR model is running for window size: ", str(_window), "...")
    #     benchmark_univariate.SVReg(
    #         symbol="BTCUSDT",
    #         candle="Close",
    #         startDate="1600000000000",
    #         endDate="1715904000000",
    #         interval="1m",
    #         train_size=0.8,
    #         window_size=_window,
    #         horizon=1,
    #     )
    # for _window in range(120, 480, 120):
    #     print(
    #         "Gradient Boosting model is running for window size: ", str(_window), "..."
    #     )
    #     benchmark_univariate.GB(
    #         symbol="BTCUSDT",
    #         candle="Close",
    #         startDate="1600000000000",
    #         endDate="1715904000000",
    #         interval="1m",
    #         train_size=0.8,
    #         window_size=_window,
    #         horizon=1,
    #     )
    # for _window in range(480, 720, 120):
    #     benchmark_univariate.XGB(
    #         symbol="BTCUSDT",
    #         candle="Close",
    #         startDate="1600000000000",
    #         endDate="1715904000000",
    #         interval="1m",
    #         train_size=0.8,
    #         window_size=_window,
    #         horizon=1,
    #     )
    # for _window in range(420, 720, 60):
    #     benchmark_univariate.VTRegressor(
    #         symbol="BTCUSDT",
    #         candle="Close",
    #         startDate="1600000000000",
    #         endDate="1715904000000",
    #         interval="1m",
    #         train_size=0.8,
    #         window_size=_window,
    #         horizon=1,
    #     )

    # for _window in range(120, 480, 120):
    benchmark_univariate.simple_LSTM(
        symbol="BTCUSDT",
        candle="Close",
        startDate="1713904000000",
        endDate="1715904000000",
        interval="1m",
        train_size=0.8,
        window_size=120,
        horizon=1,
    )
