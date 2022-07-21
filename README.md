# DR Hatespeech

Hatespeech detection for DR Facebook data.

Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexadalab.github.io/dr-hatespeech/index.html)
[![License](https://img.shields.io/github/license/alexadalab/dr-hatespeech)](https://github.com/alexadalab/dr-hatespeech/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexadalab/dr-hatespeech)](https://github.com/alexadalab/dr-hatespeech/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-68%25-yellow.svg)](https://github.com/alexadalab/dr-hatespeech/tree/dev/tests)


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
│   │   └── tfidf_model1.yaml
│   └── transformer_model
│       ├── transformer_model1.yaml
│       └── transformer_model2.yaml
├── data
│   ├── final
│   ├── processed
│   └── raw
├── makefile
├── models
│   ├── tfidf_model.bin
│   └── transformer_model1
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_args.bin
├── notebooks
│   └── evaluate_agreement.ipynb
├── poetry.toml
├── pyproject.toml
├── src
│   ├── dr_hatespeech
│   │   ├── __init__.py
│   │   ├── clean_data.py
│   │   ├── create_train_split.py
│   │   ├── labeling_functions.py
│   │   ├── load_data.py
│   │   ├── main.py
│   │   ├── prepare_data_for_annotation.py
│   │   ├── train_tfidf.py
│   │   ├── train_transformer.py
│   │   ├── training_args_with_mps_support.py
│   │   └── weak_supervision.py
│   └── scripts
│       └── fix_dot_env_file.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── test_data.py
    ├── test_labeling_functions.py
    └── test_weak_supervision.py
```
