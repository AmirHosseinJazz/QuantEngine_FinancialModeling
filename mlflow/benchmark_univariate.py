import preprocess
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from mlflow.models import infer_signature
import optuna
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim

# def naive_forecast(
#     symbol="BTCUSDT",
#     candle="Close",
#     startDate="1514764800000",
#     endDate="1715904000000",
#     interval="1d",
#     train_size=0.8,
#     window_size=20,
#     horizon=1,
# ):
#     params = {
#         "symbol": symbol,
#         "candle": candle,
#         "startDate": startDate,
#         "endDate": endDate,
#         "interval": interval,
#         "train_size": train_size,
#         "window_size": window_size,
#         "horizon": horizon,
#     }
#     data = preprocess.preprocess_univariate_candle(
#         symbol, candle, startDate, endDate, interval
#     )
#     data = data[["log_return"]]
#     X_train, y_train, X_test, y_test = preprocess.split_time_series(
#         data, train_size, window_size, horizon
#     )
#     predictions = []
#     for i in range(len(X_test)):
#         predictions.append(np.array(X_test[i])[-1][0])
#     mse = mean_squared_error(y_test, predictions)

#     mlflow.set_tracking_uri("http://localhost:8080")
#     mlflow.set_experiment("univariate")

#     with mlflow.start_run():
#         mlflow.log_params(params)
#         mlflow.log_metric("mse", mse)
#         mlflow.set_tag("model", "naive_forecast")


# def moving_average(
#     symbol="BTCUSDT",
#     candle="Close",
#     startDate="1514764800000",
#     endDate="1715904000000",
#     interval="1d",
#     train_size=0.8,
#     window_size=20,
#     horizon=1,
# ):
#     params = {
#         "symbol": symbol,
#         "candle": candle,
#         "startDate": startDate,
#         "endDate": endDate,
#         "interval": interval,
#         "train_size": train_size,
#         "window_size": window_size,
#         "horizon": horizon,
#     }
#     data = preprocess.preprocess_univariate_candle(
#         symbol, candle, startDate, endDate, interval
#     )
#     data = data[["log_return"]]
#     X_train, y_train, X_test, y_test = preprocess.split_time_series(
#         data, train_size, window_size, horizon
#     )
#     predictions = []
#     for i in range(len(X_test)):
#         predictions.append(np.array(X_test[i]).mean())
#     mse = mean_squared_error(y_test, predictions)

#     mlflow.set_tracking_uri("http://localhost:8080")
#     mlflow.set_experiment("univariate")

#     with mlflow.start_run():
#         mlflow.log_params(params)
#         mlflow.log_metric("mse", mse)
#         mlflow.set_tag("model", "moving_average")


def LinearReg(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    # mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("lr_univariate_minutely")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("MSE", mse)
        # mlflow.log_metric("MAPE", mape)

        mlflow.set_tag("model", "LinearReg")

        signature = infer_signature(X_train, model.predict(X_train))
        model_info = mlflow.sklearn.log_model(model, "model", signature=signature)
        print(model_info.model_uri)


### Random Forest
def RF_optimizer(trial, X_train, y_train, X_test, y_test, params):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("RF_univariate_minutely")
    # Log parameters and metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            }
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("mse", mse)

    return mse


def RF(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
    mode="inference",
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: RF_optimizer(trial, X_train, y_train, X_test, y_test, params),
        n_trials=20,
    )


### Decision Tree Regressor
def DT_optimizer(trial, X_train, y_train, X_test, y_test, params):
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    criterion = trial.suggest_categorical(
        "criterion", ["squared_error", "friedman_mse"]
    )
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("DT_univariate_minutely")
    # Log parameters and metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "criterion": criterion,
            }
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("mse", mse)

    return mse


def DT(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: DT_optimizer(trial, X_train, y_train, X_test, y_test, params),
        n_trials=20,
    )


### SV Regressor
def SVR_optimizer(trial, X_train, y_train, X_test, y_test, params):
    c = trial.suggest_float("c", 1e-10, 1e10, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-10, 1e10, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    model = SVR(
        C=c,
        epsilon=epsilon,
        gamma=gamma,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("SVR_univariate_minutely")
    # Log parameters and metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "C": c,
                "epsilon": epsilon,
                "gamma": gamma,
            }
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("mse", mse)

    return mse


def SVReg(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_sie": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: SVR_optimizer(trial, X_train, y_train, X_test, y_test, params),
        n_trials=20,
    )


#### Gradient Boosting
def GB_optimizer(trial, X_train, y_train, X_test, y_test, params):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("GB_univariate_minutely")
    # Log parameters and metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
            }
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("mse", mse)

    return mse


def GB(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: GB_optimizer(trial, X_train, y_train, X_test, y_test, params),
        n_trials=20,
    )


#####
def XGB_optimizer(trial, X_train, y_train, X_test, y_test, params):
    n_estimators = trial.suggest_int("n_estimators", 10, 100)
    max_depth = trial.suggest_int("max_depth", 2, 32)
    max_leaves = trial.suggest_int("max_leaves", 2, 32)
    grow_policy = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
    reg_alpha = trial.suggest_float("reg_alpha", 0.01, 0.1)
    reg_lambda = trial.suggest_float("reg_lambda", 0.01, 0.1)
    eval_metric = trial.suggest_categorical(
        "eval_metric", [mean_squared_error, mean_absolute_error]
    )

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_leaves=max_leaves,
        grow_policy=grow_policy,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        eval_metric=eval_metric,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("XGB_univariate_minutely")
    # Log parameters and metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_leaves": max_leaves,
                "grow_policy": grow_policy,
                "learning_rate": learning_rate,
                "reg_alpha": reg_alpha,
                "reg_lambda": reg_lambda,
            }
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("mse", mse)

    return mse


def XGB(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: XGB_optimizer(trial, X_train, y_train, X_test, y_test, params),
        n_trials=30,
    )


#### Voting Regressor


def VotingReg_optimizer(trial, X_train, y_train, X_test, y_test, params):

    rf_n_estimators = trial.suggest_int("n_estimators", 10, 100)
    rf_max_depth = trial.suggest_int("max_depth", 2, 32)
    rf_min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    rf_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    rf = RandomForestRegressor(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
    )

    xgb_n_estimators = trial.suggest_int("n_estimators", 10, 100)
    xgb_max_depth = trial.suggest_int("max_depth", 2, 32)
    xgb_max_leaves = trial.suggest_int("max_leaves", 2, 32)
    xgb_grow_policy = trial.suggest_categorical(
        "grow_policy", ["depthwise", "lossguide"]
    )
    xgb_learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1)
    xgb_reg_alpha = trial.suggest_float("reg_alpha", 0.01, 0.1)
    xgb_reg_lambda = trial.suggest_float("reg_lambda", 0.01, 0.1)
    xgb_eval_metric = trial.suggest_categorical(
        "eval_metric", [mean_squared_error, mean_absolute_error]
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        max_leaves=xgb_max_leaves,
        grow_policy=xgb_grow_policy,
        learning_rate=xgb_learning_rate,
        reg_alpha=xgb_reg_alpha,
        reg_lambda=xgb_reg_lambda,
        eval_metric=xgb_eval_metric,
        random_state=42,
    )

    gb_n_estimators = trial.suggest_int("n_estimators", 10, 100)
    gb_max_depth = trial.suggest_int("max_depth", 2, 32)
    gb_min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    gb_min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    gb = GradientBoostingRegressor(
        n_estimators=gb_n_estimators,
        max_depth=gb_max_depth,
        min_samples_split=gb_min_samples_split,
        min_samples_leaf=gb_min_samples_leaf,
        random_state=42,
    )

    model = VotingRegressor(
        estimators=[
            ("lr", LinearRegression()),
            ("rf", rf),
            ("gb", gb),
            ("xgb", xgb_model),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("VTReg_univariate_minutely")
    # Log parameters and metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "rf_n_estimators": rf_n_estimators,
                "rf_max_depth": rf_max_depth,
                "rf_min_samples_split": rf_min_samples_split,
                "rf_min_samples_leaf": rf_min_samples_leaf,
                "xgb_n_estimators": xgb_n_estimators,
                "xgb_max_depth": xgb_max_depth,
                "xgb_max_leaves": xgb_max_leaves,
                "xgb_grow_policy": xgb_grow_policy,
                "xgb_learning_rate": xgb_learning_rate,
                "xgb_reg_alpha": xgb_reg_alpha,
                "xgb_reg_lambda": xgb_reg_lambda,
                "xgb_eval_metric": xgb_eval_metric,
                "gb_n_estimators": gb_n_estimators,
                "gb_max_depth": gb_max_depth,
                "gb_min_samples_split": gb_min_samples_split,
                "gb_min_samples_leaf": gb_min_samples_leaf,
            }
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("mse", mse)

    return mse


def VTRegressor(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: VotingReg_optimizer(
            trial, X_train, y_train, X_test, y_test, params
        ),
        n_trials=10,
    )


def LSTM_Optimizer(trial, X_train, y_train, X_test, y_test, params):
    n_layers = trial.suggest_int("n_layers", 1, 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]  # Get the output from the last time step
            output = self.fc(lstm_out)
            return output

    model = SimpleLSTM(X_train.shape[2], 64, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    X_train = torch.from_numpy(X_train).float().to(device)
    print(X_train.shape)
    y_train = torch.from_numpy(y_train).float().to(device)
    print(y_train.shape)
    X_test = torch.from_numpy(X_test).float().to(device)
    print(X_test.shape)
    y_test = torch.from_numpy(y_test).float().to(device)
    print(y_test.shape)
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)

    predictions = predictions.cpu().numpy()
    mse = mean_squared_error(y_test.cpu().numpy(), predictions)
    # mse = mean_squared_error(y_test, predictions)
    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("LSTM_univariate_minutely")
    # Log parameters and metrics with MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "n_layers": n_layers,
            }
        )
        for key, value in params.items():
            mlflow.log_param(key, value)
        mlflow.log_metric("mse", mse)

    return mse


def simple_LSTM(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
    window_size=20,
    horizon=1,
):
    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            window_size,
            horizon,
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), window_size, 1)
    y_train = np.array(y_train).reshape(len(y_train), 1)
    X_test = np.array(X_test).reshape(len(X_test), window_size, 1)
    y_test = np.array(y_test).reshape(len(y_test), 1)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: LSTM_Optimizer(trial, X_train, y_train, X_test, y_test, params),
        n_trials=1,
    )
