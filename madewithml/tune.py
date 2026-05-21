import typer

from madewithml.config import logger

app = typer.Typer()


@app.command()
def tune() -> None:
    """Hyperparameter tuning is disabled in the simplified pipeline."""
    logger.info("Hyperparameter tuning is disabled because this project now uses a simple PyTorch training loop.")


if __name__ == "__main__":
    app()
