import datetime
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel

from madewithml import data, utils
from madewithml.config import MLFLOW_TRACKING_URI, logger, mlflow
from madewithml.models import FinetunedLLM

app = typer.Typer()


class ProjectDataset(Dataset):
    """Torch dataset for tokenized project records."""

    def __init__(self, encoded: Dict[str, np.ndarray]):
        self.ids = encoded["ids"]
        self.masks = encoded["masks"]
        self.targets = encoded["targets"]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        return {
            "ids": self.ids[index],
            "masks": self.masks[index],
            "targets": self.targets[index],
        }


def collate_batch(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    arrays = {
        "ids": np.stack([item["ids"] for item in batch]),
        "masks": np.stack([item["masks"] for item in batch]),
        "targets": np.stack([item["targets"] for item in batch]),
    }
    return utils.collate_fn(arrays)


def train_step(
    dataloader: DataLoader,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    loss = 0.0
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        z = model(batch)
        targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
        batch_loss = loss_fn(z, targets)
        batch_loss.backward()
        optimizer.step()
        loss += (batch_loss.detach().item() - loss) / (i + 1)
    return loss


def eval_step(
    dataloader: DataLoader,
    model: nn.Module,
    num_classes: int,
    loss_fn: torch.nn.modules.loss._WeightedLoss,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    with torch.inference_mode():
        for i, batch in enumerate(dataloader):
            z = model(batch)
            targets = F.one_hot(batch["targets"], num_classes=num_classes).float()
            batch_loss = loss_fn(z, targets).item()
            loss += (batch_loss - loss) / (i + 1)
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.array(y_trues), np.array(y_preds)


def save_model_artifacts(model: FinetunedLLM, preprocessor: data.CustomPreprocessor, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    utils.save_dict(preprocessor.class_to_index, str(output_dir / "class_to_index.json"))


@app.command()
def train_model(
    experiment_name: str = "mlops-project",
    dataset_loc: str = "datasets/dataset.csv",
    train_loop_config: str = '{"dropout_p":0.3,"lr":1e-5,"lr_factor":0.8,"lr_patience":3}',
    num_workers: int = 1,
    cpu_per_worker: int = 1,
    gpu_per_worker: int = 0,
    num_samples: int = 100,
    num_epochs: int = 10,
    batch_size: int = 8,
    results_fp: str = "results.json",
) -> Dict:
    """Train the model with plain PyTorch and log artifacts to MLflow."""
    del num_workers, cpu_per_worker, gpu_per_worker
    utils.set_seeds()
    torch.set_num_threads(2)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    config = json.loads(train_loop_config)
    config.update({"num_samples": num_samples, "num_epochs": num_epochs, "batch_size": batch_size})

    df = data.load_data(dataset_loc=dataset_loc, num_samples=num_samples)
    train_df, val_df = data.stratify_split(df, stratify="tag", test_size=0.2)

    preprocessor = data.CustomPreprocessor().fit(train_df)
    train_encoded = preprocessor.transform(train_df)
    val_encoded = preprocessor.transform(val_df)
    num_classes = len(preprocessor.class_to_index)

    train_loader = DataLoader(
        ProjectDataset(train_encoded),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ProjectDataset(val_encoded),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(
        llm=llm,
        dropout_p=config["dropout_p"],
        embedding_dim=llm.config.hidden_size,
        num_classes=num_classes,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
    )

    metrics: List[Dict] = []
    with mlflow.start_run() as run:
        mlflow.log_params(config)
        mlflow.log_param("num_classes", num_classes)
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            train_loss = train_step(train_loader, model, num_classes, loss_fn, optimizer)
            val_loss, _, _ = eval_step(val_loader, model, num_classes, loss_fn)
            scheduler.step(val_loss)
            epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            metrics.append(epoch_metrics)
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            logger.info(
                f"Finished epoch {epoch + 1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        with tempfile.TemporaryDirectory() as dp:
            model_dir = Path(dp) / "model"
            save_model_artifacts(model, preprocessor, model_dir)
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

        results = {
            "timestamp": datetime.datetime.now().strftime("%B %d, %Y %I:%M:%S %p"),
            "run_id": run.info.run_id,
            "params": config,
            "metrics": metrics,
        }

    logger.info(json.dumps(results, indent=2))
    if results_fp:
        utils.save_dict(results, results_fp)
    return results


if __name__ == "__main__":
    app()
