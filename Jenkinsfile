pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    triggers {
        pollSCM('H/2 * * * *')
    }

    parameters {
        booleanParam(name: 'RUN_TRAIN', defaultValue: true, description: 'Run a short training job during CI.')
        booleanParam(name: 'RUN_EVALUATE', defaultValue: true, description: 'Evaluate a trained model during CI.')
        booleanParam(name: 'BUILD_DOCKER', defaultValue: true, description: 'Build the Docker image when Docker is available.')
        booleanParam(name: 'PUSH_DOCKER', defaultValue: false, description: 'Push the Docker image to a registry.')
        booleanParam(name: 'DEPLOY_LOCAL', defaultValue: false, description: 'Run the API container on this Jenkins agent.')
        booleanParam(name: 'DEPLOY_COMPOSE', defaultValue: true, description: 'Deploy the app, Prometheus, and Grafana with Docker Compose.')
        string(name: 'TRAIN_EPOCHS', defaultValue: '10', description: 'Number of epochs to run when RUN_TRAIN is enabled.')
        string(name: 'TRAIN_SAMPLES', defaultValue: '100', description: 'Number of samples to train on when RUN_TRAIN is enabled.')
        string(name: 'TRAIN_BATCH_SIZE', defaultValue: '8', description: 'Batch size to use when RUN_TRAIN is enabled.')
        string(name: 'EVALUATE_RUN_ID', defaultValue: '', description: 'MLflow run_id to evaluate. When empty, uses results-ci.json from RUN_TRAIN.')
        string(name: 'EVALUATE_DATASET', defaultValue: 'datasets/holdout.csv', description: 'Dataset with labels to evaluate on.')
        string(name: 'EVALUATE_RESULTS_FP', defaultValue: 'evaluation-ci.json', description: 'Evaluation results JSON output path.')
        string(name: 'RUN_ID', defaultValue: '', description: 'MLflow run_id to serve when DEPLOY_LOCAL is enabled.')
        string(name: 'DOCKER_IMAGE', defaultValue: 'hugging-face-classifier', description: 'Docker image name.')
        string(name: 'DOCKER_REGISTRY', defaultValue: '', description: 'Optional registry namespace, for example docker.io/myuser or ghcr.io/myorg.')
        string(name: 'DOCKER_CREDENTIALS_ID', defaultValue: 'docker-registry', description: 'Jenkins username/password credentials ID for Docker push.')
        string(name: 'GITHUB_USERNAME', defaultValue: 'jenkins', description: 'Username propagated to the training and serving runtime.')
        string(name: 'APP_PORT', defaultValue: '8001', description: 'Host port used when DEPLOY_LOCAL=true.')
    }

    environment {
        GITHUB_USERNAME = "${params.GITHUB_USERNAME}"
        MADEWITHML_EFS_DIR = '/mlops-storage'
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
        PYTHONUNBUFFERED = '1'
        DOCKER_BUILDKIT = '0'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Create Python Environment') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            set -eux
                            rm -rf .venv
                            if command -v python3.10 >/dev/null 2>&1; then
                                python3.10 -m venv .venv
                            else
                                export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.uv-python"
                                if command -v curl >/dev/null 2>&1; then
                                    curl -LsSf https://astral.sh/uv/install.sh | sh
                                elif command -v python3 >/dev/null 2>&1; then
                                    python3 - <<'PY'
import urllib.request

url = "https://astral.sh/uv/install.sh"
with urllib.request.urlopen(url, timeout=60) as response:
    script = response.read()
with open("install-uv.sh", "wb") as fp:
    fp.write(script)
PY
                                    sh install-uv.sh
                                else
                                    echo "python3.10 is missing, and neither curl nor python3 is available to install uv."
                                    exit 1
                                fi
                                export PATH="$HOME/.local/bin:$PATH"
                                uv venv --python 3.10 --seed .venv
                            fi
                            . .venv/bin/activate
                            python --version
                            python -m pip install --upgrade "pip<27" "setuptools==68.2.2" wheel
                        '''
                    } else {
                        bat '''
                            if exist .venv rmdir /s /q .venv
                            py -3.10 -m venv .venv || python -m venv .venv
                            call .venv\\Scripts\\activate.bat
                            python --version
                            python -m pip install --upgrade "pip<27" "setuptools==68.2.2" wheel
                        '''
                    }
                }
            }
        }

        stage('Install Dependencies') {
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            set -eux
                            . .venv/bin/activate
                            python -m pip install -r requirements-ci.txt
                            python -m pip install --force-reinstall "setuptools==68.2.2" "click==8.1.7"
                        '''
                    } else {
                        bat '''
                            call .venv\\Scripts\\activate.bat
                            python -m pip install -r requirements-ci.txt
                            python -m pip install --force-reinstall "setuptools==68.2.2" "click==8.1.7"
                        '''
                    }
                }
            }
        }

        stage('Quality Checks') {
            steps {
                script {
                    def smoke = '''
                        python -m compileall madewithml
                        python -c "import mlflow, pandas, sklearn, torch, transformers; print('mlflow', mlflow.__version__); print('pandas', pandas.__version__); print('sklearn', sklearn.__version__); print('torch', torch.__version__); print('transformers', transformers.__version__)"
                    '''
                    if (isUnix()) {
                        sh """
                            set -eux
                            . .venv/bin/activate
                            ${smoke}
                        """
                    } else {
                        bat """
                            call .venv\\Scripts\\activate.bat
                            ${smoke}
                        """
                    }
                }
            }
        }

        stage('Tests') {
            when {
                expression { fileExists('tests') }
            }
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            set -eux
                            . .venv/bin/activate
                            python -m pytest
                        '''
                    } else {
                        bat '''
                            call .venv\\Scripts\\activate.bat
                            python -m pytest
                        '''
                    }
                }
            }
        }

        stage('Detect Docker') {
            steps {
                script {
                    if (isUnix()) {
                        def status = sh(returnStatus: true, script: 'command -v docker >/dev/null 2>&1 && docker version >/dev/null 2>&1')
                        env.HAS_DOCKER = status == 0 ? 'true' : 'false'
                    } else {
                        def status = bat(returnStatus: true, script: 'docker version >NUL 2>NUL')
                        env.HAS_DOCKER = status == 0 ? 'true' : 'false'
                    }
                    echo "Docker available: ${env.HAS_DOCKER}"
                }
            }
        }

        stage('Validate Docker Requirement') {
            when {
                expression { return params.BUILD_DOCKER || params.PUSH_DOCKER || params.DEPLOY_LOCAL || params.DEPLOY_COMPOSE }
            }
            steps {
                script {
                    if (env.HAS_DOCKER != 'true') {
                        error('Docker is required because a Docker build, push, or deploy stage is enabled, but Docker is not available on this Jenkins agent.')
                    }
                    if ((params.DEPLOY_LOCAL || params.DEPLOY_COMPOSE) && !params.RUN_ID?.trim() && !params.RUN_TRAIN) {
                        error('RUN_ID is required when deployment is enabled without RUN_TRAIN. Enable RUN_TRAIN or provide RUN_ID.')
                    }
                    if (params.PUSH_DOCKER && !params.BUILD_DOCKER) {
                        error('PUSH_DOCKER requires BUILD_DOCKER=true so Jenkins has an image to push.')
                    }
                }
            }
        }

        stage('Short Train Smoke') {
            when {
                expression { return params.RUN_TRAIN }
            }
            steps {
                script {
                    def trainCommand = "python -m madewithml.train --num-epochs=${params.TRAIN_EPOCHS} --num-samples=${params.TRAIN_SAMPLES} --batch-size=${params.TRAIN_BATCH_SIZE} --results-fp=results-ci.json"
                    if (isUnix()) {
                        sh """
                            set -eux
                            . .venv/bin/activate
                            export HF_HOME="\$WORKSPACE/.hf_cache"
                            export TRANSFORMERS_CACHE="\$WORKSPACE/.hf_cache/transformers"
                            ${trainCommand}
                        """
                    } else {
                        bat """
                            call .venv\\Scripts\\activate.bat
                            set HF_HOME=%WORKSPACE%\\.hf_cache
                            set TRANSFORMERS_CACHE=%WORKSPACE%\\.hf_cache\\transformers
                            ${trainCommand}
                        """
                    }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: 'results-ci.json,efs/**/result.json', allowEmptyArchive: true
                }
            }
        }

        stage('Evaluate') {
            when {
                expression { return params.RUN_EVALUATE }
            }
            steps {
                script {
                    def evaluateRunId = params.EVALUATE_RUN_ID?.trim()
                    if (!evaluateRunId && params.RUN_TRAIN && fileExists('results-ci.json')) {
                        if (isUnix()) {
                            evaluateRunId = sh(
                                returnStdout: true,
                                script: '''
                                    . .venv/bin/activate
                                    python -c "import json; print(json.load(open('results-ci.json'))['run_id'])"
                                '''
                            ).trim()
                        } else {
                            evaluateRunId = bat(
                                returnStdout: true,
                                script: '''
                                    @call .venv\\Scripts\\activate.bat
                                    @python -c "import json; print(json.load(open('results-ci.json'))['run_id'])"
                                '''
                            ).trim()
                        }
                    }
                    if (!evaluateRunId) {
                        error('RUN_EVALUATE requires EVALUATE_RUN_ID, or RUN_TRAIN must produce results-ci.json in the same build.')
                    }

                    def evaluateCommand = "python -m madewithml.evaluate --run-id=${evaluateRunId} --dataset-loc=${params.EVALUATE_DATASET} --results-fp=${params.EVALUATE_RESULTS_FP}"
                    if (isUnix()) {
                        sh """
                            set -eux
                            . .venv/bin/activate
                            export HF_HOME="\$WORKSPACE/.hf_cache"
                            export TRANSFORMERS_CACHE="\$WORKSPACE/.hf_cache/transformers"
                            ${evaluateCommand}
                        """
                    } else {
                        bat """
                            call .venv\\Scripts\\activate.bat
                            set HF_HOME=%WORKSPACE%\\.hf_cache
                            set TRANSFORMERS_CACHE=%WORKSPACE%\\.hf_cache\\transformers
                            ${evaluateCommand}
                        """
                    }
                    if (fileExists('results-ci.json')) {
                        if (isUnix()) {
                            env.TRAINED_RUN_ID = sh(
                                returnStdout: true,
                                script: '''
                                    . .venv/bin/activate
                                    python -c "import json; print(json.load(open('results-ci.json'))['run_id'])"
                                '''
                            ).trim()
                        } else {
                            env.TRAINED_RUN_ID = bat(
                                returnStdout: true,
                                script: '''
                                    @call .venv\\Scripts\\activate.bat
                                    @python -c "import json; print(json.load(open('results-ci.json'))['run_id'])"
                                '''
                            ).trim()
                        }
                        echo "Trained run_id: ${env.TRAINED_RUN_ID}"
                    }
                }
            }
            post {
                always {
                    archiveArtifacts artifacts: "${params.EVALUATE_RESULTS_FP}", allowEmptyArchive: true
                }
            }
        }

        stage('Docker Build') {
            when {
                expression { return params.BUILD_DOCKER }
            }
            steps {
                script {
                    def registry = params.DOCKER_REGISTRY?.trim()
                    def imageRepo = registry ? "${registry}/${params.DOCKER_IMAGE}" : params.DOCKER_IMAGE
                    def imageTag = "${imageRepo}:${env.BUILD_NUMBER}"
                    def latestTag = "${imageRepo}:latest"
                    if (isUnix()) {
                        sh "docker build -t ${imageTag} -t ${latestTag} ."
                    } else {
                        bat "docker build -t ${imageTag} -t ${latestTag} ."
                    }
                    env.BUILT_IMAGE = imageTag
                    env.LATEST_IMAGE = latestTag
                    env.IMAGE_REPO = imageRepo
                }
            }
        }

        stage('Docker Smoke') {
            when {
                expression { return params.BUILD_DOCKER }
            }
            steps {
                script {
                    def imageTag = env.BUILT_IMAGE ?: "${params.DOCKER_IMAGE}:${env.BUILD_NUMBER}"
                    def smokeCommand = "docker run --rm ${imageTag} python -c \"import madewithml.config; print('image smoke ok')\""
                    if (isUnix()) {
                        sh smokeCommand
                    } else {
                        bat smokeCommand
                    }
                }
            }
        }

        stage('Docker Push') {
            when {
                expression { return params.PUSH_DOCKER }
            }
            steps {
                script {
                    def registry = params.DOCKER_REGISTRY?.trim()
                    if (!registry) {
                        error('DOCKER_REGISTRY is required when PUSH_DOCKER=true, for example docker.io/myuser or ghcr.io/myorg.')
                    }
                    withCredentials([usernamePassword(credentialsId: params.DOCKER_CREDENTIALS_ID, usernameVariable: 'DOCKER_USERNAME', passwordVariable: 'DOCKER_PASSWORD')]) {
                        if (isUnix()) {
                            sh """
                                set -eux
                                echo "\$DOCKER_PASSWORD" | docker login ${registry} -u "\$DOCKER_USERNAME" --password-stdin
                                docker push ${env.BUILT_IMAGE}
                                docker push ${env.LATEST_IMAGE}
                            """
                        } else {
                            bat """
                                echo %DOCKER_PASSWORD% | docker login ${registry} -u %DOCKER_USERNAME% --password-stdin
                                docker push ${env.BUILT_IMAGE}
                                docker push ${env.LATEST_IMAGE}
                            """
                        }
                    }
                }
            }
        }

        stage('Resolve Deploy Run ID') {
            when {
                expression { return params.DEPLOY_LOCAL || params.DEPLOY_COMPOSE }
            }
            steps {
                script {
                    def deployRunId = params.RUN_ID?.trim()
                    if (!deployRunId) {
                        deployRunId = env.TRAINED_RUN_ID?.trim()
                    }
                    if (!deployRunId && fileExists('results-ci.json')) {
                        if (isUnix()) {
                            deployRunId = sh(
                                returnStdout: true,
                                script: '''
                                    . .venv/bin/activate
                                    python -c "import json; print(json.load(open('results-ci.json'))['run_id'])"
                                '''
                            ).trim()
                        } else {
                            deployRunId = bat(
                                returnStdout: true,
                                script: '''
                                    @call .venv\\Scripts\\activate.bat
                                    @python -c "import json; print(json.load(open('results-ci.json'))['run_id'])"
                                '''
                            ).trim()
                        }
                    }
                    if (!deployRunId) {
                        error('No deploy run_id found. Provide RUN_ID or enable RUN_TRAIN so results-ci.json is produced.')
                    }
                    env.DEPLOY_RUN_ID = deployRunId
                    echo "Deploying run_id: ${env.DEPLOY_RUN_ID}"
                }
            }
        }

        stage('Deploy Local') {
            when {
                expression { return params.DEPLOY_LOCAL }
            }
            steps {
                script {
                    def imageTag = env.BUILT_IMAGE ?: "${params.DOCKER_IMAGE}:latest"
                    def appPort = params.APP_PORT?.trim() ?: '8001'
                    if (isUnix()) {
                        sh """
                            docker rm -f hugging-face-classifier-api || true
                            docker run -d --name hugging-face-classifier-api \
                                -p ${appPort}:8000 \
                                -e GITHUB_USERNAME="${env.GITHUB_USERNAME}" \
                                -e MADEWITHML_EFS_DIR=/mlops-storage \
                                -e HF_HOME=/app/.hf_cache \
                                -e TRANSFORMERS_CACHE=/app/.hf_cache/transformers \
                                -v hugging_face_classifier_storage:/mlops-storage \
                                ${imageTag} \
                                python -m madewithml.serve --run_id ${env.DEPLOY_RUN_ID} --host 0.0.0.0 --port 8000
                        """
                    } else {
                        bat """
                            docker rm -f hugging-face-classifier-api 2>NUL
                            docker run -d --name hugging-face-classifier-api -p ${appPort}:8000 -e GITHUB_USERNAME=%GITHUB_USERNAME% -e MADEWITHML_EFS_DIR=/mlops-storage -e HF_HOME=/app/.hf_cache -e TRANSFORMERS_CACHE=/app/.hf_cache/transformers -v hugging_face_classifier_storage:/mlops-storage ${imageTag} python -m madewithml.serve --run_id ${env.DEPLOY_RUN_ID} --host 0.0.0.0 --port 8000
                        """
                    }
                }
            }
        }

        stage('Deploy Compose') {
            when {
                expression { return params.DEPLOY_COMPOSE }
            }
            steps {
                script {
                    def dockerImageName = params.DOCKER_IMAGE?.trim() ?: 'hugging-face-classifier'
                    def imageForCompose = env.LATEST_IMAGE?.trim() ?: "${dockerImageName}:latest"
                    if (isUnix()) {
                        sh """
                            set -eux
                            if docker compose version >/dev/null 2>&1; then
                                COMPOSE="docker compose"
                            else
                                COMPOSE="docker-compose"
                            fi
                            export RUN_ID="${env.DEPLOY_RUN_ID}"
                            export GITHUB_USERNAME="${env.GITHUB_USERNAME}"
                            export DOCKER_IMAGE="${imageForCompose}"
                            docker rm -f mlops-prometheus mlops-grafana hugging-face-serve >/dev/null 2>&1 || true
                            # Ensure compose is available in PATH
                            if ! command -v docker-compose >/dev/null 2>&1 && ! command -v "docker" >/dev/null 2>&1; then
                                echo "ERROR: docker-compose / docker compose not found in PATH" >&2
                                exit 127
                            fi
                            if [ -f /.dockerenv ]; then
                                echo "Jenkins is running inside Docker; skipping Prometheus/Grafana bind mounts."
                                \$COMPOSE --profile serve up -d --build hugging-face-serve
                            else
                                \$COMPOSE --profile serve up -d --build hugging-face-serve prometheus grafana
                            fi
                            \$COMPOSE ps
                        """
                    } else {
                        bat """
                            set RUN_ID=${env.DEPLOY_RUN_ID}
                            set GITHUB_USERNAME=%GITHUB_USERNAME%
                            set DOCKER_IMAGE=${imageForCompose}
                            docker rm -f mlops-prometheus mlops-grafana hugging-face-serve 2>NUL
                            docker compose version >NUL 2>NUL
                            if %ERRORLEVEL% EQU 0 (
                                docker compose --profile serve up -d --build hugging-face-serve
                                docker compose ps
                            ) else (
                                docker-compose --profile serve up -d --build hugging-face-serve
                                docker-compose ps
                            )
                        """
                    }
                }
            }
        }

        stage('Verify Deployment and Monitoring') {
            when {
                expression { return params.DEPLOY_COMPOSE }
            }
            steps {
                script {
                    if (isUnix()) {
                        sh '''
                            set -eux
                            for i in $(seq 1 30); do
                                if curl -fsS http://hugging-face-serve:8000/ >/tmp/app-health.json; then
                                    break
                                fi
                                sleep 5
                            done
                            cat /tmp/app-health.json
                            curl -fsS http://hugging-face-serve:8000/metrics | head
                            if [ ! -f /.dockerenv ]; then
                                curl -fsS http://prometheus:9090/-/ready
                                curl -fsS http://grafana:3000/api/health
                            fi
                        '''
                    } else {
                        bat '''
                            powershell -Command "$deadline=(Get-Date).AddMinutes(3); do { try { Invoke-WebRequest -UseBasicParsing http://localhost:8000/ | Out-File app-health.txt; exit 0 } catch { Start-Sleep -Seconds 5 } } while ((Get-Date) -lt $deadline); exit 1"
                            type app-health.txt
                            powershell -Command "Invoke-WebRequest -UseBasicParsing http://localhost:8000/metrics | Select-Object -ExpandProperty Content | Select-Object -First 1"
                            powershell -Command "Invoke-WebRequest -UseBasicParsing http://localhost:9090/-/ready"
                            powershell -Command "Invoke-WebRequest -UseBasicParsing http://localhost:3001/api/health"
                        '''
                    }
                }
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'logs/*.log,results*.json', allowEmptyArchive: true
        }
    }
}
