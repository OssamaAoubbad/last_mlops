import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from typing_extensions import Annotated

from madewithml.config import logger, mlflow
from madewithml.data import CustomPreprocessor
from madewithml.models import FinetunedLLM
from madewithml.utils import collate_fn

app = typer.Typer()


def decode(indices: Iterable[Any], index_to_class: Dict) -> List:
    """Decode indices to labels."""
    return [index_to_class[index] for index in indices]


def format_prob(prob: Iterable, index_to_class: Dict) -> Dict:
    """Format probabilities as label -> probability."""
    return {index_to_class[i]: item for i, item in enumerate(prob)}


class TorchPredictor:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
        self.model.eval()

    def __call__(self, batch):
        return self.model.predict(collate_fn(batch))

    def predict_proba(self, batch):
        return self.model.predict_proba(collate_fn(batch))

    def get_preprocessor(self):
        return self.preprocessor

    @classmethod
    def from_model_dir(cls, model_dir: Path):
        class_to_index = json.loads(Path(model_dir, "class_to_index.json").read_text())
        preprocessor = CustomPreprocessor(class_to_index=class_to_index)
        model = FinetunedLLM.load(Path(model_dir, "args.json"), Path(model_dir, "model.pt"))
        return cls(preprocessor=preprocessor, model=model)


def get_model_dir(run_id: str) -> Path:
    """Get the logged MLflow model artifact directory for a run."""
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    parsed_uri = urlparse(artifact_uri)
    if parsed_uri.scheme == "file":
        artifact_dir = Path(url2pathname(parsed_uri.netloc + parsed_uri.path))
    else:
        artifact_dir = Path(artifact_uri)
    model_dir = artifact_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifacts not found at {model_dir}")
    return model_dir


def get_best_checkpoint(run_id: str) -> Path:
    """Backward-compatible name used by serve/evaluate."""
    return get_model_dir(run_id)


def predict_proba(df: pd.DataFrame, predictor: TorchPredictor) -> List:
    """Predict tags with probabilities for a dataframe."""
    preprocessor = predictor.get_preprocessor()
    encoded = preprocessor.transform(df)
    y_prob = predictor.predict_proba(encoded)
    results = []
    for prob in y_prob:
        tag = preprocessor.index_to_class[int(prob.argmax())]
        results.append({"prediction": tag, "probabilities": format_prob(prob, preprocessor.index_to_class)})
    return results


@app.command()
def get_best_run_id(experiment_name: str = "", metric: str = "", mode: str = "") -> str:
    """Get the best run_id from an MLflow experiment."""
    sorted_runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=[f"metrics.{metric} {mode}"],
    )
    run_id = sorted_runs.iloc[0].run_id
    print(run_id)
    return run_id


@app.command()
def predict(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    title: Annotated[str, typer.Option(help="project title")] = None,
    description: Annotated[str, typer.Option(help="project description")] = None,
) -> Dict:
    """Predict the tag for a project."""
    predictor = TorchPredictor.from_model_dir(get_model_dir(run_id=run_id))
    sample_df = pd.DataFrame([{"title": title or "", "description": description or "", "tag": "other"}])
    results = predict_proba(df=sample_df, predictor=predictor)
    logger.info(json.dumps(results, cls=NumpyEncoder, indent=2))
    return results


if __name__ == "__main__":
    app()
