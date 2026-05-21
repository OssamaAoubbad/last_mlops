@echo off
cd /d "%~dp0"
set GITHUB_USERNAME=PC
set HF_HOME=%~dp0.hf_cache
set TRANSFORMERS_CACHE=%~dp0.hf_cache\transformers
".venv\Scripts\python.exe" -m madewithml.serve --run_id 06cb00ce853f41b1af18c78646056727 > "logs\serve.out.log" 2> "logs\serve.err.log"
