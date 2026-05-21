# Hugging Face Classifier

Projet MLOps de classification de projets avec un modele Hugging Face/Scibert, PyTorch, MLflow, FastAPI, Jenkins et Docker.

## Objectif

Ce projet entraine un modele de classification de texte sur des projets techniques, puis evalue le modele sur un jeu de donnees holdout. La pipeline CI/CD est geree par Jenkins avec des stages pour les controles qualite, les tests, le train, l'evaluate, le build Docker et le deploy local.

## Structure

- `madewithml/train.py`: entrainement PyTorch et logging MLflow.
- `madewithml/evaluate.py`: evaluation du modele sur un dataset labelise.
- `madewithml/serve.py`: API FastAPI pour servir le modele.
- `datasets/dataset.csv`: dataset principal pour l'entrainement.
- `datasets/holdout.csv`: dataset utilise pour l'evaluation.
- `Jenkinsfile`: pipeline CI/CD declarative Jenkins.
- `Dockerfile`: image applicative Python.
- `docker-compose.yml`: lance une instance Jenkins locale pour executer la CI/CD.
- `cicd/jenkins/`: image Jenkins avec plugins et outils necessaires.

## Pipeline CI/CD

La pipeline Jenkins contient les stages suivants:

- `Checkout`: recupere le code depuis GitHub.
- `Create Python Environment`: cree un environnement Python 3.10.
- `Install Dependencies`: installe les dependances depuis `requirements-ci.txt`.
- `Quality Checks`: compile le package et verifie les imports principaux.
- `Tests`: lance les tests avec `pytest`.
- `Detect Docker`: detecte si Docker est disponible sur l'agent Jenkins.
- `Validate Docker Requirement`: bloque le build si Docker est demande mais indisponible.
- `Short Train Smoke`: entraine le modele si `RUN_TRAIN=true`.
- `Evaluate`: evalue le modele si `RUN_EVALUATE=true`.
- `Docker Build`: construit l'image Docker si `BUILD_DOCKER=true`.
- `Docker Smoke`: teste rapidement l'image Docker.
- `Deploy Local`: lance l'API en container si `DEPLOY_LOCAL=true`.

## Parametres Jenkins

- `RUN_TRAIN`: active l'entrainement.
- `RUN_EVALUATE`: active l'evaluation.
- `BUILD_DOCKER`: active le build de l'image Docker.
- `DEPLOY_LOCAL`: lance l'API localement sur l'agent Jenkins.
- `TRAIN_EPOCHS`: nombre d'epoques. Defaut: `10`.
- `TRAIN_SAMPLES`: nombre d'exemples utilises pour le train. Defaut: `100`.
- `TRAIN_BATCH_SIZE`: batch size. Defaut: `8`.
- `EVALUATE_RUN_ID`: run MLflow a evaluer. Peut rester vide si `RUN_TRAIN=true`.
- `EVALUATE_DATASET`: dataset d'evaluation. Defaut: `datasets/holdout.csv`.
- `EVALUATE_RESULTS_FP`: fichier JSON de sortie. Defaut: `evaluation-ci.json`.
- `RUN_ID`: run MLflow a servir avec l'API.
- `DOCKER_IMAGE`: nom de l'image Docker. Defaut: `hugging-face-classifier`.
- `GITHUB_USERNAME`: username propage au runtime applicatif.

Pour entrainer et evaluer dans le meme build:

```text
RUN_TRAIN=true
RUN_EVALUATE=true
EVALUATE_RUN_ID=
```

Dans ce cas, Jenkins lit automatiquement le `run_id` depuis `results-ci.json`.

## Dernier Resultat Jenkins

Dernier build valide observe:

- Statut: `SUCCESS`
- Tests: `3 passed`
- Train: `10` epoques
- Samples train: `100`
- Batch size: `8`
- Run MLflow: `814de7951ca64cca82272964ea8c630c`

Resultat final du train:

```text
epoch 9
train_loss = 0.1714
val_loss   = 0.3132
```

Resultat global de l'evaluation sur `datasets/holdout.csv`:

```text
num_samples = 191
precision   = 0.8175
recall      = 0.8272
f1          = 0.8135
```

Resultat par classe:

```text
computer-vision              f1=0.8844 samples=71
natural-language-processing  f1=0.8642 samples=78
other                        f1=0.7451 samples=26
mlops                        f1=0.3636 samples=16
```

## Lancer Jenkins avec Docker Compose

Construire et lancer Jenkins:

```bash
docker compose up -d --build
```

Ouvrir Jenkins:

```text
http://localhost:8080
```

Si le port `8080` est deja utilise, modifier `docker-compose.yml`:

```yaml
ports:
  - "8081:8080"
  - "50000:50000"
```

Puis ouvrir:

```text
http://localhost:8081
```

Recuperer le mot de passe initial du Jenkins Docker:

```bash
docker compose exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

Arreter Jenkins:

```bash
docker compose down
```

Supprimer aussi les donnees Jenkins:

```bash
docker compose down -v
```

## Configurer le Job Jenkins

Creer un job `Pipeline` nomme `mlops_project`.

Configuration:

```text
Definition: Pipeline script from SCM
SCM: Git
Repository URL: https://github.com/halaBelmo/hugging-face-classifier.git
Branch: */oum_mlops
Script Path: Jenkinsfile
```

Apres le premier build, Jenkins affiche `Build with Parameters`.

## Commandes Locales Utiles

Installer les dependances:

```bash
python -m pip install -r requirements-ci.txt
```

Lancer les tests:

```bash
python -m pytest
```

Entrainer:

```bash
python -m madewithml.train --num-epochs=10 --num-samples=100 --batch-size=8 --results-fp=results-ci.json
```

Evaluer un run:

```bash
python -m madewithml.evaluate --run-id=<RUN_ID> --dataset-loc=datasets/holdout.csv --results-fp=evaluation-ci.json
```

Construire l'image applicative:

```bash
docker build -t hugging-face-classifier:latest .
```
