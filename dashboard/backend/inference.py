import mlflow
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
from mlflow import MlflowClient
import mlflow.pyfunc

from . import preprocess

# import preprocess
from datetime import datetime
import pytz


def univariate_LR_inference(model_name="benchmark_uni_rf_minute", model_version=1):
    client = MlflowClient()
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    run_id = model.metadata.run_id
    run = client.get_run(run_id)

    params = run.data.params

    df = preprocess.get_latest_kline(
        symbol=params["symbol"],
        candle=params["candle"],
        interval=params["interval"],
        window=params["window_size"],
    )
    df = df.sort_values(by="Opentime", ascending=True)
    df_preprocess = preprocess.preprocess_univariate_candle(df)
    df_preprocess = df_preprocess[["log_return"]]

    prediciton_time = df_preprocess.index[-1]
    if params["interval"] == "1m":
        prediciton_time += 60000
    if params["interval"] == "1h":
        prediciton_time += 3600000
    if params["interval"] == "1d":
        prediciton_time += 86400000

    # prediciton_time = datetime.fromtimestamp(prediciton_time / 1000).astimezone(
    #     pytz.timezone("Europe/Paris")
    # )

    X = np.array(df_preprocess).reshape(1, -1)
    prediction = model.predict(X)
    print(df_preprocess)

    transition = pd.DataFrame(
        index=[df_preprocess.index[-1], prediciton_time],
        columns=["log_return"],
        data=[df_preprocess["log_return"].iloc[-1], prediction[0]],
    )
    df_preprocess.index = (
        pd.to_datetime(df_preprocess.index, unit="ms")
        .tz_localize("UTC")
        .tz_convert("CET")
    )
    transition.index = (
        pd.to_datetime(transition.index, unit="ms").tz_localize("UTC").tz_convert("CET")
    )
    return prediction, df_preprocess, transition


def all_models(
    keyword="",
):
    client = MlflowClient()
    requested_models = []
    for rm in client.search_registered_models():
        if keyword in rm.name:
            requested_models.append(rm.name)
    return requested_models


# if __name__ == "__main__":
#     univariate_LR_inference()
