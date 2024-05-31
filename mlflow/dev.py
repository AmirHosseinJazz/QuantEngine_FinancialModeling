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
    for _window in range(360, 720, 10):
        benchmark_univariate.XGB(
            symbol="BTCUSDT",
            candle="Close",
            startDate="1600000000000",
            endDate="1715904000000",
            interval="1m",
            train_size=0.8,
            window_size=_window,
            horizon=1,
        )
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
    # benchmark_univariate.simple_LSTM(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1713904000000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    #     window_size=120,
    #     horizon=1,
    # )l
    # benchmark_univariate.RF_create_model(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1600000000000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    #     window_size=250,
    #     horizon=1,
    #     model_params={"n_estimators": 53,
    #                   "max_depth": 2,
    #                   "min_samples_leaf": 4,
    #                   "min_samples_split": 4,
    #                   "random_state": 42},
    # )

    # benchmark_univariate.DT_create_model(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1600000000000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    #     window_size=250,
    #     horizon=1,
    #     model_params={
    #         "max_depth": 6,
    #         "min_samples_leaf": 5,
    #         "min_samples_split": 8,
    #         "random_state": 42,
    #     },
    # )
    # benchmark_univariate.SVR_create_model(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1600000000000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    #     window_size=360,
    #     horizon=1,
    #     model_params={
    #         "C": 2.92528097640191e-05,
    #         "gamma": "scale",
    #         "epsilon": 0.0009806253386798005,
    #     },
    # )
    # benchmark_univariate.GB_create_model(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1600000000000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    #     window_size=360,
    #     horizon=1,
    #     model_params={
    #         "n_estimators": 69,
    #         "max_depth": 2,
    #         "min_samples_leaf": 10,
    #         "min_samples_split": 6,
    #         "random_state": 42,
    #     },
    # )

    # benchmark_univariate.XGB_create_model(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1600000000000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    #     window_size=480,
    #     horizon=1,
    #     model_params={
    #         "n_estimators": 11,
    #         "max_depth": 10,
    #         "reg_alpha": 0.010159975484883675,
    #         "reg_lambda": 0.050043123020554116,
    #         "learning_rate": 0.04028030989911358,
    #         "grow_policy": "depthwise",
    #         "max_leaves": 21,
    #         "random_state": 42,
    #     },
    # )

    # benchmark_univariate.VTReg_create_model(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1600000000000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    #     window_size=480,
    #     horizon=1,
    #     model_params={
    #         "xgb_n_estimators": 11,
    #         "xgb_max_depth": 10,
    #         "xgb_reg_alpha": 0.010159975484883675,
    #         "xgb_reg_lambda": 0.050043123020554116,
    #         "xgb_learning_rate": 0.04028030989911358,
    #         "xgb_grow_policy": "depthwise",
    #         "xgb_max_leaves": 21,
    #         "rf_n_estimators": 53,
    #         "rf_max_depth": 2,
    #         "rf_min_samples_leaf": 4,
    #         "rf_min_samples_split": 4,
    #         "gb_n_estimators": 69,
    #         "gb_max_depth": 2,
    #         "gb_min_samples_leaf": 10,
    #         "gb_min_samples_split": 6,
    #         "n_estimators": 69,
    #     },
    # )

    pass
