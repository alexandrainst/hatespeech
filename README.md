# Hatespeech

Hatespeech detection based on DR Facebook data.

Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/hatespeech/hatespeech.html)
[![License](https://img.shields.io/github/license/alexandrainst/hatespeech)](https://github.com/alexandrainst/hatespeech/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/hatespeech)](https://github.com/alexandrainst/hatespeech/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-81%25-yellowgreen.svg)](https://github.com/alexandrainst/hatespeech/tree/main/tests)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```
poetry add <package-name>
```

### Auto-generate API documentation

To auto-generate API document for your project, run:

```
make docs
```

To view the documentation, run:

```
make view-docs
```

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```
.
├── .flake8
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── config
│   ├── __init__.py
│   ├── config.yaml
│   ├── data
│   │   ├── offensive.yaml
│   │   └── test_offensive.yaml
│   ├── hatespeech-label-config.xml
│   ├── offensive-label-config.xml
│   ├── tfidf_model
│   │   └── tfidf_model.yaml
│   └── transformer_model
│       ├── aelaectra.yaml
│       ├── aelaectra2.yaml
│       ├── xlmr-base.yaml
│       ├── xlmr-base2.yaml
│       ├── xlmr-base3.yaml
│       └── xlmr-large.yaml
├── data
│   ├── final
│   ├── processed
│   └── raw
│       └── scores.xlsx
├── makefile
├── models
│   └── tfidf_model.bin
├── notebooks
│   ├── analyse-weak-labels.ipynb
│   ├── compare_models.ipynb
│   ├── evaluate_agreement.ipynb
│   └── evaluate_models.ipynb
├── poetry.toml
├── pyproject.toml
├── src
│   ├── hatespeech
│   │   ├── __init__.py
│   │   ├── attack.py
│   │   ├── clean_data.py
│   │   ├── create_train_split.py
│   │   ├── labelling_functions.py
│   │   ├── load_data.py
│   │   ├── main.py
│   │   ├── prepare_data_for_annotation.py
│   │   ├── snorkel_utils.py
│   │   ├── train_tfidf.py
│   │   ├── train_transformer.py
│   │   └── weak_supervision.py
│   └── scripts
│       └── fix_dot_env_file.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_data.py
    ├── test_labelling_functions.py
    └── test_weak_supervision.py
```
