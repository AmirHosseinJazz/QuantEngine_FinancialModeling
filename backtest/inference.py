import pandas as pd
import numpy as np
import mlflow
import preprocess
import mlflow.pyfunc
import time



def inference(model: mlflow.pyfunc.PyFuncModel, params: dict) -> None:
    endDate = int(time.time() * 1000)
    if params["interval"] == "1m":
        startDate = endDate - ((int(params["window_size"]) + 1) * 60 * 1000)
    else:
        raise ValueError("Interval not supported")
    df = preprocess.get_univariate_kline(
        symbol=params["symbol"],
        interval=params["interval"],
        startDate=startDate,
        endDate=endDate,
    )
    if len(df) < int(params["window_size"]) + 1:
        raise ValueError("Not enough data")
    df["log_return"] = df["Close"].pct_change()
    df = df.dropna()
    df.set_index("Opentime", inplace=True)
    print(df)
    input_data = df[["log_return"]].values
    input_data = input_data.T
    input_data = input_data.reshape(1, -1)
    print(model.predict(input_data))
    return model.predict(input_data)


def get_model(model_name: str, model_version: int) -> (mlflow.pyfunc.PyFuncModel, dict):
    mlflow.set_tracking_uri("http://mlflow:8080")
    client = mlflow.tracking.MlflowClient()
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    model_version_details = client.get_model_version(
        name=model_name, version=model_version
    )
    source_run_id = model_version_details.run_id
    run_details = mlflow.get_run(run_id=source_run_id)
    return model, run_details.data.params


def get_all_models() -> list:
    mlflow.set_tracking_uri("http://mlflow:8080")
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    return [model.name for model in models]

def get_model_versions(model_name:str ) -> list:
    mlflow.set_tracking_uri("http://mlflow:8080")
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models(filter_string=f"name='{model_name}'")
    return [model.latest_versions[0].version for model in models]
    
def get_model_performance(model_name:str, model_version:int) -> dict:
    mlflow.set_tracking_uri("http://mlflow:8080")
    client = mlflow.tracking.MlflowClient()
    model_version_details = client.get_model_version(
        name=model_name, version=model_version
    )
    source_run_id = model_version_details.run_id
    run_details = mlflow.get_run(run_id=source_run_id)
    return  run_details.data.metrics, run_details.data.params