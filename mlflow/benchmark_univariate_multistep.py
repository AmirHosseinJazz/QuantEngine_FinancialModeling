import preprocess
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import optuna
import xgboost as xgb
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
import torch.nn as nn
import torch.optim as optim


class CombinedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EnsembleLSTM(nn.Module):
    def __init__(self, params):
        super(EnsembleLSTM, self).__init__()
        self.input_size = params["input_size"]
        self.hidden_size = params["hidden_size"]
        self.num_layers = params["num_layers"]
        self.output_size = params["output_size"]
        self.device = "cpu"
        self.lstm = nn.LSTM(
            self.input_size, self.hidden_size, self.num_layers, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def CustomLSTM_Exp(
    symbol="BTCUSDT",
    candle="Close",
    startDate="1514764800000",
    endDate="1715904000000",
    interval="1d",
    train_size=0.8,
):
    mlflow.set_tracking_uri("http://localhost:8080")
    client = MlflowClient()
    exper = "benchmark_uni_xgb_minute_multistep"
    model_xgb1 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step1")
    run_1 = client.get_run(model_xgb1.metadata.run_id)

    model_xgb2 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step2")
    run_2 = client.get_run(model_xgb2.metadata.run_id)

    model_xgb3 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step3")
    run_3 = client.get_run(model_xgb3.metadata.run_id)

    model_xgb4 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step4")
    run_4 = client.get_run(model_xgb4.metadata.run_id)

    model_xgb5 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step5")
    run_5 = client.get_run(model_xgb5.metadata.run_id)

    assert (
        run_1.data.params["window_size"]
        == run_2.data.params["window_size"]
        == run_3.data.params["window_size"]
        == run_4.data.params["window_size"]
        == run_5.data.params["window_size"]
    ), "All window sizes are not the same."

    window_size = run_1.data.params["window_size"]
    max_horizon = max(
        run_1.data.params["horizon"],
        run_2.data.params["horizon"],
        run_3.data.params["horizon"],
        run_4.data.params["horizon"],
        run_5.data.params["horizon"],
    )

    params = {
        "symbol": symbol,
        "candle": candle,
        "startDate": startDate,
        "endDate": endDate,
        "interval": interval,
        "train_size": train_size,
        "window_size": window_size,
        "horizon": max_horizon,
    }
    data = preprocess.preprocess_univariate_candle(
        symbol, candle, startDate, endDate, interval
    )

    data = data[["log_return"]]

    # print("window_size", window_size)
    # print("max_horizon", max_horizon)

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            train_size,
            int(window_size),
            int(max_horizon),
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), int(window_size))
    y_train = np.array(y_train).reshape(len(y_train), int(max_horizon))
    X_test = np.array(X_test).reshape(len(X_test), int(window_size))
    y_test = np.array(y_test).reshape(len(y_test), int(max_horizon))

    xgb_models = [model_xgb1, model_xgb2, model_xgb3, model_xgb4, model_xgb5]

    xgb_predictions_train = np.zeros((X_train.shape[0], int(max_horizon)))
    xgb_predictions_test = np.zeros((X_test.shape[0], int(max_horizon)))
    for i, model in enumerate(xgb_models):
        xgb_predictions_train[:, i] = model.predict(X_train)
        xgb_predictions_test[:, i] = model.predict(X_test)

    print("Shape of X_train: ", X_train.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_test: ", y_test.shape)

    print("Shape of xgb_predictions_train: ", xgb_predictions_train.shape)
    print("Shape of xgb_predictions_test: ", xgb_predictions_test.shape)

    X_train_combined = np.hstack([X_train, xgb_predictions_train])
    X_test_combined = np.hstack([X_test, xgb_predictions_test])

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: CustomLSTM_XGB_Optimizier(
            trial, X_train_combined, y_train, X_test_combined, y_test, params
        ),
        n_trials=50,
    )


def CustomLSTM_XGB_Optimizier(trial, X_train, y_train, X_test, y_test, params):
    # Suggest hyperparameters
    params["input_size"] = X_train.shape[1]
    params["hidden_size"] = trial.suggest_int("hidden_size", 64, 256)
    params["num_layers"] = trial.suggest_int("num_layers", 5, 20)
    params["output_size"] = y_train.shape[1]
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-1)
    params["batch_size"] = trial.suggest_int("batch_size", 32, 128)
    params["num_epochs"] = trial.suggest_int("num_epochs", 200, 300)

    train_dataset = CombinedDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = CombinedDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=params["batch_size"], shuffle=False
    )

    model = EnsembleLSTM(params)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    mlflow.set_experiment("LSTM Hyperparameter Optimization")
    with mlflow.start_run(nested=True):
        for key, value in params.items():
            mlflow.log_param(key, value)
        for epoch in range(1):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            # Validate the model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.unsqueeze(1)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(test_loader.dataset)
            print(
                f"Epoch {epoch+1}/{params['num_epochs']}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}"
            )
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("val_loss", val_loss)
    return val_loss


def CUSTOMLSTM_XGB_CreateFromRun(run_id):
    mlflow.set_tracking_uri("http://localhost:8080")
    client = MlflowClient()
    run_LSTM = client.get_run(run_id)
    params = run_LSTM.data.params
    exper = "benchmark_uni_xgb_minute_multistep"
    model_xgb1 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step1")
    run_1 = client.get_run(model_xgb1.metadata.run_id)

    model_xgb2 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step2")
    run_2 = client.get_run(model_xgb2.metadata.run_id)

    model_xgb3 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step3")
    run_3 = client.get_run(model_xgb3.metadata.run_id)

    model_xgb4 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step4")
    run_4 = client.get_run(model_xgb4.metadata.run_id)

    model_xgb5 = mlflow.pyfunc.load_model(model_uri=f"models:/{exper}@step5")
    run_5 = client.get_run(model_xgb5.metadata.run_id)

    assert (
        run_1.data.params["window_size"]
        == run_2.data.params["window_size"]
        == run_3.data.params["window_size"]
        == run_4.data.params["window_size"]
        == run_5.data.params["window_size"]
    ), "All window sizes are not the same."

    window_size = run_1.data.params["window_size"]
    max_horizon = max(
        run_1.data.params["horizon"],
        run_2.data.params["horizon"],
        run_3.data.params["horizon"],
        run_4.data.params["horizon"],
        run_5.data.params["horizon"],
    )

    data = preprocess.preprocess_univariate_candle(
        params["symbol"],
        params["candle"],
        params["startDate"],
        params["endDate"],
        params["interval"],
    )

    data = data[["log_return"]]

    X_train, y_train, X_test, y_test, train_time, test_time = (
        preprocess.split_time_series(
            data,
            float(params["train_size"]),
            int(window_size),
            int(max_horizon),
            mode="univariate",
            target="log_return",
        )
    )
    X_train = np.array(X_train).reshape(len(X_train), int(window_size))
    y_train = np.array(y_train).reshape(len(y_train), int(max_horizon))
    X_test = np.array(X_test).reshape(len(X_test), int(window_size))
    y_test = np.array(y_test).reshape(len(y_test), int(max_horizon))

    xgb_models = [model_xgb1, model_xgb2, model_xgb3, model_xgb4, model_xgb5]

    xgb_predictions_train = np.zeros((X_train.shape[0], int(max_horizon)))
    xgb_predictions_test = np.zeros((X_test.shape[0], int(max_horizon)))
    for i, model in enumerate(xgb_models):
        xgb_predictions_train[:, i] = model.predict(X_train)
        xgb_predictions_test[:, i] = model.predict(X_test)

    print("Shape of X_train: ", X_train.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_test: ", y_test.shape)

    print("Shape of xgb_predictions_train: ", xgb_predictions_train.shape)
    print("Shape of xgb_predictions_test: ", xgb_predictions_test.shape)

    X_train_combined = np.hstack([X_train, xgb_predictions_train])
    X_test_combined = np.hstack([X_test, xgb_predictions_test])

    train_dataset = CombinedDataset(
        torch.tensor(X_train_combined, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = CombinedDataset(
        torch.tensor(X_test_combined, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=int(params["batch_size"]), shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(params["batch_size"]), shuffle=False
    )

    for k in params.keys():
        if k in [
            "input_size",
            "hidden_size",
            "num_layers",
            "output_size",
            "batch_size",
            "num_epochs",
        ]:
            params[k] = int(params[k])

    model = EnsembleLSTM(params)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]))
    mlflow.set_experiment("Published_Experiments")
    with mlflow.start_run(nested=True):
        for key, value in params.items():
            mlflow.log_param(key, value)
        for epoch in range(params["num_epochs"]):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)

            # Validate the model
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.unsqueeze(1)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(test_loader.dataset)
            print(
                f"Epoch {epoch+1}/{params['num_epochs']}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}"
            )
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss)
            mlflow.log_metric("val_loss", val_loss)
    # print("*******")
    # print(model(X_batch))
    mlflow.set_tag("model", "EnsembleLSTM")
    signature = infer_signature(
        X_batch.unsqueeze(1).cpu().numpy(), model(X_batch).detach().numpy()
    )
    model_info = mlflow.pytorch.log_model(
        model,
        "EnsembleLSTM",
        signature=signature,
        input_example=X_batch.unsqueeze(1).cpu().numpy(),
    )


if __name__ == "__main__":
    # CustomLSTM_Exp(
    #     symbol="BTCUSDT",
    #     candle="Close",
    #     startDate="1514764800000",
    #     endDate="1715904000000",
    #     interval="1m",
    #     train_size=0.8,
    # )

    # CUSTOMLSTM_XGB_CreateFromRun("3768f13b92d8441aa01bd88676cf70fc")
    pass
