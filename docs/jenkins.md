# Jenkins CI/CD Pipeline

This project includes a declarative Jenkins pipeline in `Jenkinsfile`.

## Jenkins agent prerequisites

- Python 3.10 available as `python3.10`, `py -3.10`, or `python`. On Linux agents without Python 3.10, the pipeline installs `uv` with `curl` and creates a Python 3.10 virtual environment.
- `curl` installed on Linux agents when Python 3.10 is not already available
- Docker installed if `BUILD_DOCKER` or `DEPLOY_LOCAL` is enabled. The pipeline detects Docker and skips Docker stages when it is not available.
- Network access to PyPI and Hugging Face for dependency and model downloads

## Pipeline parameters

- `RUN_TRAIN`: runs a short training smoke test. Disabled by default because model downloads and CPU training are slow.
- `RUN_EVALUATE`: evaluates a trained model. Disabled by default because it needs an MLflow run ID and model checkpoint.
- `BUILD_DOCKER`: builds the Docker image. Disabled by default to keep the first Jenkins build fast.
- `DEPLOY_LOCAL`: runs the API container on the Jenkins agent.
- `TRAIN_EPOCHS`: number of epochs to run when `RUN_TRAIN` is enabled. Default: `10`.
- `TRAIN_SAMPLES`: number of samples to use when `RUN_TRAIN` is enabled. Default: `100`.
- `TRAIN_BATCH_SIZE`: batch size to use when `RUN_TRAIN` is enabled. Default: `8`.
- `EVALUATE_RUN_ID`: MLflow run ID to evaluate. If empty and `RUN_TRAIN=true`, Jenkins uses the `run_id` saved in `results-ci.json`.
- `EVALUATE_DATASET`: labeled dataset used for evaluation. Default: `datasets/holdout.csv`.
- `EVALUATE_RESULTS_FP`: path for the evaluation results JSON artifact. Default: `evaluation-ci.json`.
- `RUN_ID`: MLflow run ID to serve when `DEPLOY_LOCAL` is enabled.
- `DOCKER_IMAGE`: Docker image name. Default: `hugging-face-classifier`.
- `GITHUB_USERNAME`: username propagated to Ray runtime environment. Default: `jenkins`.

## Recommended first run

Use the defaults first. This installs the smaller CI dependency set from `requirements-ci.txt`, runs import/compile checks, and runs smoke tests.

Enable `RUN_TRAIN` with the default training parameters to run the same 10 epoch training smoke used locally:

```text
RUN_TRAIN=true
TRAIN_EPOCHS=10
TRAIN_SAMPLES=100
TRAIN_BATCH_SIZE=8
```

To train and evaluate in the same Jenkins build, enable both:

```text
RUN_TRAIN=true
RUN_EVALUATE=true
```

To evaluate an existing MLflow run without training first, enable `RUN_EVALUATE` and set:

```text
EVALUATE_RUN_ID=06cb00ce853f41b1af18c78646056727
EVALUATE_DATASET=datasets/holdout.csv
```

Enable `BUILD_DOCKER` only after Docker is available on the Jenkins agent. If `BUILD_DOCKER=true` or `DEPLOY_LOCAL=true` and Docker is missing, the pipeline fails with a clear setup message instead of skipping the application build.

To deploy the model trained locally, enable `DEPLOY_LOCAL` and set:

```text
RUN_ID=06cb00ce853f41b1af18c78646056727
```

The local deployment exposes the API on:

```text
http://<jenkins-agent-host>:8000/docs
```

## Run Jenkins with Docker Compose

This repository includes a Docker Compose setup for running the CI/CD controller locally.

Start Jenkins:

```bash
docker compose up -d --build
```

Open Jenkins:

```text
http://localhost:8080
```

The Compose Jenkins image includes `git`, `curl`, the Docker CLI, and the Jenkins plugins needed by the pipeline (`workflow-aggregator`, `git`, `timestamper`, and `docker-workflow`). It also mounts `/var/run/docker.sock`, so Jenkins can run Docker stages such as `BUILD_DOCKER` and `DEPLOY_LOCAL` when Docker is available on the host.

Get the first admin password:

```bash
docker compose exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

Stop Jenkins:

```bash
docker compose down
```

Remove Jenkins data as well:

```bash
docker compose down -v
```
