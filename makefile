# This ensures that we can call `make <target>` even if `<target>` exists as a file or
# directory.
.PHONY: notebook docs

# Exports all variables defined in the makefile available to scripts
.EXPORT_ALL_VARIABLES:

# Includes environment variables from the .env file
include .env

install-poetry:
	@echo "Installing poetry..."
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
	@$(eval include ${HOME}/.poetry/env)

uninstall-poetry:
	@echo "Uninstalling poetry..."
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 - --uninstall

install:
	@echo "Installing..."
	@if [ "$(shell which poetry)" = "" ]; then \
		$(MAKE) install-poetry; \
	fi
	@if [ "$(shell which gpg)" = "" ]; then \
		echo "GPG not installed, so an error will occur. Install GPG on MacOS with "\
			 "`brew install gnupg` or on Ubuntu with `apt install gnupg` and run "\
			 "`make install` again."; \
	fi
	@$(MAKE) setup-poetry
	@$(MAKE) setup-git

setup-poetry:
	@poetry env use python3
	@poetry run python3 -m src.scripts.fix_dot_env_file
	@poetry install
	@poetry run pre-commit install

setup-git:
	@git init
	@git config --local user.name ${GIT_NAME}
	@git config --local user.email ${GIT_EMAIL}
	@if [ ${GPG_KEY_ID} = "" ]; then \
		echo "No GPG key ID specified. Skipping GPG signing."; \
		git config --local commit.gpgsign false; \
	else \
		echo "Signing with GPG key ID ${GPG_KEY_ID}..."; \
		echo 'If you get the "failed to sign the data" error when committing, try running `export GPG_TTY=$$(tty)`.'; \
		git config --local commit.gpgsign true; \
		git config --local user.signingkey ${GPG_KEY_ID}; \
	fi

docs:
	@poetry run pdoc --docformat google -o docs src/hatespeech
	@echo "Saved documentation."

view-docs:
	@echo "Viewing API documentation..."
	@open docs/hatespeech.html

clean:
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache
	@echo "Cleaned repository."

label-offensive:
	@label-studio init dr-offensive --label-config config/offensive-label-config.xml
	@label-studio start dr-offensive --label-config config/offensive-label-config.xml

label-hatespeech:
	@label-studio init dr-hatespeech --label-config config/hatespeech-label-config.xml
	@label-studio start dr-hatespeech --label-config config/hatespeech-label-config.xml

run:
	@poetry run python3 -m src.hatespeech.main

test:
	@pytest && readme-cov

tree:
	@tree -a \
		-I .git \
		-I .mypy_cache . \
		-I .env \
		-I .venv \
		-I poetry.lock \
		-I .ipynb_checkpoints \
		-I dist \
		-I .gitkeep \
		-I docs \
		-I .pytest_cache \
		-I outputs \
		-I .DS_Store \
		-I .cache \
		-I *.parquet \
		-I *.csv \
		-I *.txt \
		-I checkpoint-* \
		-I .coverage* \
		-I *_eda.ipynb \
		-I aelaectra \
		-I aelaectra2 \
		-I attack \
		-I xlmr-base1 \
		-I xlmr-base2 \
		-I xlmr-base3 \
		-I xlmr-large
