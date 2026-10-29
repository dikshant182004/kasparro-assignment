# Kasparro Auditor — Make targets
# Run `make help` to see all targets.
.PHONY: help setup init run test eval lint format clean replay capture

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv and install dev deps
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@echo "✓ Setup complete. Activate with: source $(VENV)/bin/activate"

init: ## Initialize SQLite schema
	$(BIN)/python -m scripts.init_db

run: ## Start FastAPI server on 127.0.0.1:8000
	$(BIN)/uvicorn kasparro_auditor.main:app --reload --host 127.0.0.1 --port 8000

test: ## Run pytest suite
	$(BIN)/pytest -v

eval: ## Run evaluation harness and write comparison report
	$(BIN)/python -m eval.run_eval --out eval/results/comparison_report.md

replay: ## Re-run full pipeline in replay mode (no API key needed)
	EXECUTION_MODE=replay $(BIN)/python -m scripts.run_analysis

capture: ## Capture live LLM responses into the replay cache (needs GROQ_API_KEY)
	$(BIN)/python -m scripts.replay_capture

lint: ## Run ruff
	$(BIN)/ruff check src tests eval scripts
	$(BIN)/ruff format --check src tests eval scripts

format: ## Apply ruff formatting
	$(BIN)/ruff check --fix src tests eval scripts
	$(BIN)/ruff format src tests eval scripts

clean: ## Remove build artifacts and caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__ *.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
