import argparse
import json
import time
from http import HTTPStatus
from typing import Dict

import pandas as pd
import uvicorn
from fastapi import FastAPI
from prometheus_client import Counter, Histogram
from starlette.requests import Request

from madewithml import evaluate, predict
from madewithml.config import MLFLOW_TRACKING_URI, mlflow

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:  # pragma: no cover - optional local dependency
    Instrumentator = None

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["run_id", "threshold"],
)
prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of prediction requests below confidence threshold",
    ["run_id"],
)
model_inference_duration_seconds = Histogram(
    "model_inference_duration_seconds",
    "Prediction latency in seconds",
    ["run_id", "handler"],
)
prediction_label_total = Counter(
    "prediction_label_total",
    "Distribution of predicted labels",
    ["run_id", "label"],
)

app = FastAPI(
    title="Made With ML",
    description="Classify machine learning projects.",
    version="0.1",
)


async def get_prediction_input(request: Request, title: str = "", description: str = "") -> Dict:
    """Get prediction inputs from a JSON body or query parameters."""
    data = {}
    body = await request.body()
    if body:
        data = json.loads(body)
    return {
        "title": data.get("title", title) or "",
        "description": data.get("description", description) or "",
    }


def make_json_safe(results):
    """Convert numpy values in prediction results to JSON-safe Python types."""
    safe_results = []
    for result in results:
        safe_results.append(
            {
                "prediction": result["prediction"],
                "probabilities": {label: float(prob) for label, prob in result["probabilities"].items()},
            }
        )
    return safe_results


def create_app(run_id: str, threshold: float = 0.9) -> FastAPI:
    """Create a local FastAPI app."""
    local_app = FastAPI(
        title="Made With ML",
        description="Classify machine learning projects.",
        version="0.1",
    )
    if Instrumentator:
        Instrumentator().instrument(local_app).expose(local_app)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    predictor = predict.TorchPredictor.from_model_dir(predict.get_model_dir(run_id=run_id))

    @local_app.get("/")
    def _index() -> Dict:
        return {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }

    @local_app.get("/run_id/")
    def _run_id() -> Dict:
        return {"run_id": run_id}

    @local_app.post("/evaluate/")
    async def _evaluate(request: Request) -> Dict:
        data = await request.json()
        results = evaluate.evaluate(run_id=run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    @local_app.post("/predict/")
    async def _predict(request: Request, title: str = "", description: str = ""):
        data = await get_prediction_input(request=request, title=title, description=description)
        sample_df = pd.DataFrame([{"title": data["title"], "description": data["description"], "tag": "other"}])

        start_time = time.perf_counter()
        results = predict.predict_proba(df=sample_df, predictor=predictor)
        duration = time.perf_counter() - start_time

        errors = 0
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < threshold:
                errors += 1
                results[i]["prediction"] = "other"

            prediction_label_total.labels(run_id=run_id, label=results[i]["prediction"]).inc()

        prediction_requests_total.labels(run_id=run_id, threshold=str(threshold)).inc()
        prediction_errors_total.labels(run_id=run_id).inc(errors)
        model_inference_duration_seconds.labels(run_id=run_id, handler="/predict").observe(duration)

        return {"results": make_json_safe(results)}

    return local_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for `other` class.")
    parser.add_argument("--server", choices=["uvicorn"], default="uvicorn", help="server backend to use.")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="port to bind.")
    args = parser.parse_args()
    uvicorn.run(create_app(run_id=args.run_id, threshold=args.threshold), host=args.host, port=args.port)
