# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: notebook docs

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Export all environment variables in .env file
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

install-poetry:
	@echo "Installing poetry..."
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

activate:
	@echo "Activating virtual environment..."
	@poetry shell
	@source `poetry env info --path`/bin/activate

install:
	@echo "Installing..."
	@git config commit.gpgsign true
	@git config user.signingkey $(GPG_KEY)
	@git config user.name $(GIT_NAME)
	@git config user.email $(GIT_EMAIL)
	@poetry install
	@poetry run pre-commit install

remove-env:
	@poetry env remove python3
	@echo "Removed virtual environment."

view-docs:
	@echo "Viewing API documentation..."
	@poetry run pdoc src/dr_hatespeech

docs:
	@poetry run pdoc src/dr_hatespeech -o docs
	@echo "Saved documentation."

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache
	@echo "Cleaned repository."
