import datetime
import json
from collections import OrderedDict
from typing import Dict

import numpy as np
import pandas as pd
import typer
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function
from typing_extensions import Annotated

from madewithml import predict, utils
from madewithml.config import logger
from madewithml.predict import TorchPredictor

app = typer.Typer()


def get_overall_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "precision": metrics[0],
        "recall": metrics[1],
        "f1": metrics[2],
        "num_samples": np.float64(len(y_true)),
    }


def get_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_to_index: Dict) -> Dict:
    labels = list(class_to_index.values())
    metrics = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_metrics = {}
    for _class, index in class_to_index.items():
        i = labels.index(index)
        per_class_metrics[_class] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i]),
        }
    return OrderedDict(sorted(per_class_metrics.items(), key=lambda tag: tag[1]["f1"], reverse=True))


@slicing_function()
def nlp_llm(x):
    """NLP projects that use LLMs."""
    nlp_project = "natural-language-processing" in x.tag
    llm_terms = ["transformer", "llm", "bert"]
    llm_project = any(s.lower() in x.text.lower() for s in llm_terms)
    return nlp_project and llm_project


@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8


def get_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, df: pd.DataFrame) -> Dict:
    slice_metrics = {}
    df = df.copy()
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    slices = PandasSFApplier([nlp_llm, short_text]).apply(df)
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            metrics = precision_recall_fscore_support(y_true[mask], y_pred[mask], average="micro", zero_division=0)
            slice_metrics[slice_name] = {
                "precision": metrics[0],
                "recall": metrics[1],
                "f1": metrics[2],
                "num_samples": len(y_true[mask]),
            }
    return slice_metrics


@app.command()
def evaluate(
    run_id: Annotated[str, typer.Option(help="id of the specific run to load from")] = None,
    dataset_loc: Annotated[str, typer.Option(help="dataset (with labels) to evaluate on")] = None,
    results_fp: Annotated[str, typer.Option(help="location to save evaluation results to")] = None,
) -> Dict:
    """Evaluate on the holdout dataset."""
    logger.info("Starting evaluation")
    logger.info("Loading dataset from %s", dataset_loc)
    df = pd.read_csv(dataset_loc)
    logger.info("Loaded %d rows", len(df))

    logger.info("Loading model for run_id=%s", run_id)
    predictor = TorchPredictor.from_model_dir(predict.get_model_dir(run_id=run_id))
    preprocessor = predictor.get_preprocessor()
    df = df[df["tag"].isin(preprocessor.class_to_index)].reset_index(drop=True)
    if df.empty:
        raise ValueError("No evaluation rows match the classes learned during training.")

    encoded = preprocessor.transform(df)
    y_true = encoded["targets"]
    y_pred = predictor(encoded)

    metrics = {
        "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
        "run_id": run_id,
        "overall": get_overall_metrics(y_true=y_true, y_pred=y_pred),
        "per_class": get_per_class_metrics(y_true=y_true, y_pred=y_pred, class_to_index=preprocessor.class_to_index),
        "slices": get_slice_metrics(y_true=y_true, y_pred=y_pred, df=df),
    }
    logger.info(json.dumps(metrics, indent=2))
    logger.info("Evaluation finished")
    if results_fp:
        logger.info("Saving results to %s", results_fp)
        utils.save_dict(d=metrics, path=results_fp)
    return metrics


if __name__ == "__main__":
    app()
