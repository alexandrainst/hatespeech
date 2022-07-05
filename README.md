# DR Hatespeech

Hatespeech detection for DR Facebook data.

Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)
- Anders Jess Pedersen (anders.j.pedersen@alexandra.dk)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```bash
poetry add <package-name>
```

### Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```

To view the documentation, run:

```bash
make view-docs
```

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [hydra](https://hydra.cc/): Manage configuration files
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```bash
.
├── README.md
├── config
│   ├── main.yaml
│   └── model
│       └── model1.yaml
├── data
│   ├── final
│   ├── processed
│   │   ├── test_data_cleaned.parquet
│   │   └── test_data_weakly_supervised.parquet
│   └── raw
│       └── test_data.csv
├── docs
│   └── dr_hatespeech
│       ├── data.html
│       ├── index.html
│       ├── labeling_functions.html
│       ├── tfidf.html
│       └── weak_supervision.html
├── makefile
├── models
│   └── tfidf_model.bin
├── notebooks
├── poetry.lock
├── pyproject.toml
├── src
│   └── dr_hatespeech
│       ├── __init__.py
│       ├── data.py
│       ├── labeling_functions.py
│       ├── tfidf.py
│       └── weak_supervision.py
└── tests
    ├── __init__.py
    ├── test_data.py
    ├── test_labeling_functions.py
    └── test_weak_supervision.py
```
