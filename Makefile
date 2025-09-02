.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running ruff"
	@uv run ruff check --fix
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy 
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@uv run deptry .