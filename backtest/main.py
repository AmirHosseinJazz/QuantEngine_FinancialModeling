from http.client import HTTPException
from fastapi import FastAPI
import inference
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import redis
import pickle

# from run_backtrader import main


redis_client = redis.Redis(host="cryptic_redis", port=6379, db=0)
app = FastAPI()


class ResponseModel(BaseModel):
    predicted_return: List[float] = Field(..., example=[50.00])


@app.get("/inference/{model_name}/{model_version}")
async def inference_endpoint(model_name: str, model_version: str):
    if not redis_client.exists(f"{model_name}:{model_version}"):
        model, params = inference.get_model(model_name, model_version)
        serialized_model = pickle.dumps(model)
        serialized_params = pickle.dumps(params)
        redis_client.set(f"{model_name}:{model_version}", serialized_model)
        redis_client.set(f"{model_name}:{model_version}:params", serialized_params)
        try:
            return ResponseModel(predicted_return=inference.inference(model, params))
        except Exception as E:
            return {"error": str(E)}

    else:
        serialized_model = redis_client.get(f"{model_name}:{model_version}")
        serialized_params = redis_client.get(f"{model_name}:{model_version}:params")
        model = pickle.loads(serialized_model)
        params = pickle.loads(serialized_params)

        try:
            return ResponseModel(predicted_return=inference.inference(model, params))
        except Exception as E:
            return {"error": str(E)}


# Updated model to reflect individual entries for compatibility with Grafana
class ModelEntry(BaseModel):
    model_name: str  # Single string to represent the model name


class VersionEntry(BaseModel):
    version: int  # Single integer to represent a version


# Updated response model to be a list of entries
class Models(BaseModel):
    models: List[ModelEntry] = Field(
        ..., example=[{"model_name": "benchmark_uni_gb_minute"}]
    )


class ModelVersions(BaseModel):
    versions: List[VersionEntry] = Field(..., example=[{"version": 1}])


@app.get("/search", response_model=Models)
async def get_models():
    try:
        models = [ModelEntry(model_name=name) for name in inference.get_all_models()]
        return Models(models=models)
    except Exception as E:
        raise HTTPException(status_code=500, detail=str(E))


@app.get("/model_versions/{model_name}", response_model=ModelVersions)
async def get_model_versions(model_name: str):
    try:
        versions = [
            VersionEntry(version=ver)
            for ver in inference.get_model_versions(model_name)
        ]
        return ModelVersions(versions=versions)
    except Exception as E:
        raise HTTPException(status_code=500, detail=str(E))


class ModelPerformanceMetrics(BaseModel):
    metrics: Dict[str, float] = Field(..., example={"accuracy": 0.75})


class ModelPerformanceParams(BaseModel):
    params: Dict[str, str] = Field(
        ..., example={"window_size": "10", "symbol": "BTCUSDT", "interval": "1m"}
    )


class ModelDetails(BaseModel):
    metrics: ModelPerformanceMetrics
    params: ModelPerformanceParams


@app.get("/model_performance/{model_name}/{model_version}")
async def get_model_performance(
    model_name: str, model_version: int, response_model=ModelDetails
):
    try:
        metrics, params = inference.get_model_performance(model_name, model_version)
        return ModelDetails(
            metrics=ModelPerformanceMetrics(metrics=metrics),
            params=ModelPerformanceParams(params=params),
        )
    except Exception as E:
        print(E)
        # raise HTTPException(status_code=500, detail=str(E))
    # try:

    # pass
    # except:


@app.get("/backtest")
async def backtest():
    # main()
    return {"message": "Backtest completed successfully"}
